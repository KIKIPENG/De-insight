"""圖片知識庫 API — 上傳、列表、搜尋、刪除、選取、檔案存取。"""

import sys
from pathlib import Path
from typing import List

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

router = APIRouter()


async def _get_project_id() -> str:
    """取得當前專案 ID（從 app.db 讀取第一個專案）。"""
    from projects.manager import ProjectManager
    pm = ProjectManager()
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
async def list_images():
    """列出當前專案所有圖片。"""
    pid = await _get_project_id()
    from rag.image_store import list_images as _list
    images = await _list(pid)
    return {"images": images, "project_id": pid}


@router.post("/images/upload")
async def upload_image(
    file: UploadFile = File(...),
    caption: str = Form(""),
    tags: str = Form(""),
):
    """上傳圖片並建立向量索引。"""
    pid = await _get_project_id()
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename")

    # sanitize filename
    safe_name = Path(file.filename).name
    if not safe_name:
        raise HTTPException(status_code=400, detail="Invalid filename")

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Empty file")

    from rag.image_store import add_image
    result = await add_image(
        project_id=pid,
        filename=safe_name,
        image_bytes=content,
        caption=caption,
        tags=tags,
    )
    return result


@router.get("/images/search")
async def search_images(q: str = "", limit: int = 5):
    """文字語意搜尋圖片。"""
    if not q.strip():
        raise HTTPException(status_code=400, detail="Query is empty")
    pid = await _get_project_id()
    from rag.image_store import search_images as _search
    results = await _search(pid, q, limit=limit)
    return {"results": results}


@router.delete("/images/{image_id}")
async def delete_image(image_id: str):
    """刪除圖片。"""
    pid = await _get_project_id()
    from rag.image_store import delete_image as _delete
    ok = await _delete(pid, image_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Image not found")
    return {"status": "deleted", "id": image_id}


@router.get("/images/file/{filename}")
async def get_image_file(filename: str):
    """取得圖片檔案（A7: 路徑安全防護）。"""
    pid = await _get_project_id()
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
async def update_image_endpoint(image_id: str, caption: str = Form(""), tags: str = Form("")):
    """更新圖片 caption/tags。"""
    pid = await _get_project_id()
    from rag.image_store import update_image
    ok = await update_image(pid, image_id, caption=caption, tags=tags)
    if not ok:
        raise HTTPException(status_code=404, detail="Image not found")
    return {"status": "updated", "id": image_id}


class SelectRequest(BaseModel):
    image_ids: List[str] = []


@router.post("/images/select")
async def select_images(req: SelectRequest):
    """設定選取的圖片（寫入 selected.json，TUI 端讀取）。"""
    pid = await _get_project_id()
    from rag.image_store import set_selected
    selected = await set_selected(pid, req.image_ids)
    return {"selected": selected, "count": len(selected)}


@router.get("/images/selected")
async def get_selected_images():
    """取得目前選取的圖片。"""
    pid = await _get_project_id()
    from rag.image_store import get_selected
    selected = await get_selected(pid)
    return {"selected": selected, "count": len(selected)}
