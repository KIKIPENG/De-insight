"""圖片知識庫 API — 上傳、列表、搜尋、刪除、選取、檔案存取。"""

import asyncio
import logging
import sys
import time
from pathlib import Path
from typing import List

from fastapi import APIRouter, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel

log = logging.getLogger(__name__)

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

router = APIRouter()
_processing_state: dict[str, dict] = {}


async def _get_project_id(project_id: str | None = None) -> str:
    """取得當前專案 ID。優先使用外部指定 project_id，否則 fallback 第一個專案。"""
    from projects.manager import ProjectManager
    pm = ProjectManager()
    if project_id:
        p = await pm.get_project(project_id)
        if p:
            return project_id
    projects = await pm.list_projects()
    if not projects:
        raise HTTPException(status_code=400, detail="No project found")
    return projects[0]["id"]


def _images_dir(project_id: str) -> Path:
    from paths import project_root
    d = project_root(project_id) / "images"
    d.mkdir(parents=True, exist_ok=True)
    return d


@router.get("/images")
async def list_images(project_id: str | None = None):
    """列出當前專案所有圖片。"""
    pid = await _get_project_id(project_id)
    from rag.image_store import list_images as _list
    images = await _list(pid)
    return {"images": images, "project_id": pid}


@router.post("/images/upload")
async def upload_image(
    request: Request,
):
    """上傳圖片：階段一存檔（秒回），階段二背景建索引。"""
    form = await request.form()
    caption = str(form.get("caption", "") or "")
    tags = str(form.get("tags", "") or "")
    project_id = str(form.get("project_id", "") or "")
    pid = await _get_project_id(project_id or None)

    def _is_upload(v: object) -> bool:
        return hasattr(v, "filename") and hasattr(v, "read")

    upload_files: list[UploadFile] = []
    upload_files.extend([f for f in form.getlist("files") if _is_upload(f)])
    upload_files.extend([f for f in form.getlist("files[]") if _is_upload(f)])
    legacy_file = form.get("file")
    if _is_upload(legacy_file):
        upload_files.append(legacy_file)  # type: ignore[arg-type]

    # Fallback: accept any multipart field carrying an upload object, regardless of key name.
    if not upload_files:
        for _, v in form.multi_items():
            if _is_upload(v):
                upload_files.append(v)  # type: ignore[arg-type]
    if not upload_files:
        raise HTTPException(status_code=400, detail="No file uploaded")

    from rag.image_store import save_image_file

    # 階段一：快速存檔（毫秒級）
    saved: list[dict] = []
    errors: list[dict] = []
    for f in upload_files:
        raw_name = f.filename or ""
        safe_name = Path(raw_name).name
        if not safe_name:
            errors.append({"filename": raw_name, "error": "Invalid filename"})
            continue

        content = await f.read()
        if not content:
            errors.append({"filename": safe_name, "error": "Empty file"})
            continue

        try:
            result = await save_image_file(pid, safe_name, content)
            saved.append(result)
        except Exception as e:
            errors.append({"filename": safe_name, "error": str(e)})

    if not saved:
        first_err = errors[0]["error"] if errors else "unknown"
        raise HTTPException(
            status_code=500,
            detail={"message": f"All uploads failed: {first_err}", "errors": errors},
        )

    # 階段二：背景逐張建索引
    _processing_state[pid] = {
        "active": True,
        "total": len(saved),
        "indexed": 0,
        "failed": 0,
        "last_error": "",
        "updated_at": time.time(),
    }
    asyncio.create_task(_process_batch(pid, saved, caption, tags))

    return {
        "project_id": pid,
        "saved": len(saved),
        "processing": True,
        "pending": [s["filename"] for s in saved],
        "errors": errors,
    }


async def _process_batch(
    pid: str, saved_files: list[dict], caption: str, tags: str
) -> None:
    """背景逐張建立圖片索引，每張之間加 2 秒延遲避免速率限制。"""
    from rag.image_store import index_image

    state = _processing_state.get(pid)
    if state is None:
        state = {
            "active": True,
            "total": len(saved_files),
            "indexed": 0,
            "failed": 0,
            "last_error": "",
            "updated_at": time.time(),
        }
        _processing_state[pid] = state

    for i, item in enumerate(saved_files):
        if i > 0:
            await asyncio.sleep(5)  # 避免 Jina API 429（free tier 速率限制嚴格）
        try:
            # Guard against per-image indexing hanging indefinitely.
            await asyncio.wait_for(
                index_image(pid, item["filename"], caption=caption, tags=tags),
                timeout=45.0,
            )
            log.info("Indexed image: %s", item["filename"])
            state["indexed"] = int(state.get("indexed", 0)) + 1
            state["updated_at"] = time.time()
        except asyncio.TimeoutError:
            err_msg = "index timeout (45s)"
            log.error("Index failed for %s: %s", item["filename"], err_msg)
            state["failed"] = int(state.get("failed", 0)) + 1
            state["last_error"] = f"{item['filename']}: {err_msg}"
            state["updated_at"] = time.time()
        except Exception as e:
            log.error("Index failed for %s: %s", item["filename"], e)
            state["failed"] = int(state.get("failed", 0)) + 1
            state["last_error"] = f"{item['filename']}: {e}"
            state["updated_at"] = time.time()

    # 所有圖片索引完成後，觸發偏好萃取
    try:
        from rag.image_store import trigger_preference_update
        from paths import project_root
        db_path = project_root(pid) / "memories.db"

        # _quick_llm_call 不在 backend context 裡，用 litellm 直接呼叫
        async def _llm_call(prompt, max_tokens=300):
            import os
            from config.service import get_config_service
            env = get_config_service().snapshot(include_process=True)
            model = env.get("RAG_LLM_MODEL", "") or env.get("LLM_MODEL", "")
            if not model:
                return ""
            import litellm
            for k in ("GOOGLE_API_KEY", "GEMINI_API_KEY", "OPENAI_API_KEY",
                       "OPENROUTER_API_KEY", "OPENAI_API_BASE"):
                v = env.get(k, "")
                if v:
                    os.environ[k] = v
            if not os.environ.get("GEMINI_API_KEY") and os.environ.get("GOOGLE_API_KEY"):
                os.environ["GEMINI_API_KEY"] = os.environ["GOOGLE_API_KEY"]
            # prefix 處理：裸模型名 + Google key → gemini/ prefix
            if not model.startswith(("gemini/", "openai/", "anthropic/", "ollama/")):
                if os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"):
                    model = f"gemini/{model}"
            resp = await litellm.acompletion(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=max_tokens,
            )
            return resp.choices[0].message.content or ""

        result = await trigger_preference_update(
            pid, llm_call=_llm_call, db_path=db_path,
        )
        if result:
            log.info("Visual preference updated after batch upload: %s", result["summary"][:60])
            # 交叉偵測
            try:
                from memory.thought_tracker import check_cross_modal
                cross = await check_cross_modal(
                    result["summary"], _llm_call, db_path=db_path,
                )
                if cross and cross.get("type") == "cross_modal":
                    from memory.store import add_memory as _add_mem
                    await _add_mem(
                        type="contradiction",
                        content=cross["summary"],
                        source=f"視覺：{cross.get('visual', '')}\n文字：{cross.get('textual', '')}",
                        topic="跨模態矛盾",
                        category="美學偏好",
                        db_path=db_path,
                    )
                    log.info("Cross-modal contradiction detected: %s", cross["summary"][:60])
            except Exception as e:
                log.warning("Cross-modal check failed: %s", e)
    except Exception as e:
        log.warning("Preference update after batch failed: %s", e)
    finally:
        state["active"] = False
        state["updated_at"] = time.time()


@router.get("/images/processing-status")
async def processing_status(project_id: str | None = None):
    """回傳圖片處理進度：已索引數、待處理數。"""
    pid = await _get_project_id(project_id)
    img_dir = _images_dir(pid)
    files_on_disk = {
        f.name for f in img_dir.iterdir()
        if f.is_file() and not f.name.startswith(".")
    }
    from rag.image_store import list_images as _list
    indexed_imgs = await _list(pid)
    indexed_names = {img["filename"] for img in indexed_imgs}
    orphans = files_on_disk - indexed_names
    st = _processing_state.get(pid, {})
    return {
        "total_files": len(files_on_disk),
        "indexed": len(indexed_names),
        "pending": len(orphans),
        "orphan_files": sorted(orphans),
        "processing": bool(st.get("active", False)),
        "batch_total": int(st.get("total", 0)),
        "batch_indexed": int(st.get("indexed", 0)),
        "batch_failed": int(st.get("failed", 0)),
        "last_error": str(st.get("last_error", "")),
    }


@router.post("/images/process-orphans")
async def process_orphans(project_id: str | None = None):
    """重新處理所有有檔案但無索引的圖片。"""
    pid = await _get_project_id(project_id)
    img_dir = _images_dir(pid)
    files_on_disk = {
        f.name for f in img_dir.iterdir()
        if f.is_file() and not f.name.startswith(".")
    }
    from rag.image_store import list_images as _list
    indexed_imgs = await _list(pid)
    indexed_names = {img["filename"] for img in indexed_imgs}
    orphans = files_on_disk - indexed_names
    if not orphans:
        return {"processing": False, "count": 0, "message": "no orphans"}
    saved = [{"filename": fn} for fn in orphans]
    _processing_state[pid] = {
        "active": True,
        "total": len(saved),
        "indexed": 0,
        "failed": 0,
        "last_error": "",
        "updated_at": time.time(),
    }
    asyncio.create_task(_process_batch(pid, saved, caption="", tags=""))
    return {"processing": True, "count": len(orphans)}


@router.get("/images/search")
async def search_images(q: str = "", limit: int = 5, project_id: str | None = None):
    """文字語意搜尋圖片。"""
    if not q.strip():
        raise HTTPException(status_code=400, detail="Query is empty")
    pid = await _get_project_id(project_id)
    from rag.image_store import search_images as _search
    results = await _search(pid, q, limit=limit)
    return {"results": results}


@router.delete("/images/{image_id}")
async def delete_image(image_id: str, project_id: str | None = None):
    """刪除圖片。"""
    pid = await _get_project_id(project_id)
    from rag.image_store import delete_image as _delete
    ok = await _delete(pid, image_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Image not found")
    return {"status": "deleted", "id": image_id}


@router.get("/images/file/{filename}")
async def get_image_file(filename: str, project_id: str | None = None):
    """取得圖片檔案（A7: 路徑安全防護）。"""
    pid = await _get_project_id(project_id)
    img_dir = _images_dir(pid)

    # A7: filename sanitize + path traversal protection
    safe = Path(filename).name
    if not safe or safe != filename:
        raise HTTPException(status_code=400, detail="Invalid filename")

    resolved = (img_dir / safe).resolve()
    if not str(resolved).startswith(str(img_dir.resolve())):
        raise HTTPException(status_code=400, detail="Path traversal denied")

    if not resolved.exists():
        raise HTTPException(status_code=404, detail="File not found")

    return FileResponse(resolved)


@router.patch("/images/{image_id}")
async def update_image_endpoint(
    image_id: str,
    caption: str = Form(""),
    tags: str = Form(""),
    project_id: str = Form(""),
):
    """更新圖片 caption/tags。"""
    pid = await _get_project_id(project_id or None)
    from rag.image_store import update_image
    ok = await update_image(pid, image_id, caption=caption, tags=tags)
    if not ok:
        raise HTTPException(status_code=404, detail="Image not found")
    return {"status": "updated", "id": image_id}


@router.put("/images/{image_id}/caption")
async def update_caption(
    image_id: str,
    request: Request,
    project_id: str | None = None,
):
    """更新圖片的三段式 caption 並重算向量。

    body: {"content": {...}, "style_tags": [...], "description": "..."}
    """
    pid = await _get_project_id(project_id)
    body = await request.json()

    # 驗證結構
    if not isinstance(body, dict) or "description" not in body:
        raise HTTPException(status_code=400, detail="Invalid caption structure")

    from rag.image_store import update_image
    ok = await update_image(pid, image_id, caption=body, tags="", recalc_vector=True)
    if not ok:
        raise HTTPException(status_code=404, detail="Image not found")
    return {"status": "ok", "id": image_id, "project_id": pid}


class SelectRequest(BaseModel):
    image_ids: List[str] = []
    project_id: str | None = None


@router.post("/images/select")
async def select_images(req: SelectRequest):
    """設定選取的圖片（寫入 selected.json，TUI 端讀取）。"""
    pid = await _get_project_id(req.project_id)
    from rag.image_store import set_selected
    selected = await set_selected(pid, req.image_ids)
    return {"selected": selected, "count": len(selected), "project_id": pid}


@router.post("/images/backfill-captions")
async def backfill_captions_endpoint(project_id: str | None = None):
    """批次為沒有 caption 的圖片自動生成描述。"""
    pid = await _get_project_id(project_id)
    from rag.image_store import backfill_captions
    result = await backfill_captions(pid)
    return {"project_id": pid, **result}


@router.get("/images/selected")
async def get_selected_images(project_id: str | None = None):
    """取得目前選取的圖片。"""
    pid = await _get_project_id(project_id)
    from rag.image_store import get_selected
    selected = await get_selected(pid)
    return {"selected": selected, "count": len(selected)}
