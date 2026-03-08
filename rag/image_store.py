"""圖片知識庫 — LanceDB 圖片語意索引（dim=1024）。

每個專案有獨立的 images table，存放圖片 metadata + embedding。
支援文字語意搜圖（text-to-image retrieval）。
"""

from __future__ import annotations

import base64
import logging
import math
import os
import time
from pathlib import Path

import lancedb
import pyarrow as pa

from paths import DATA_ROOT, project_root
from embeddings.service import get_embedding_service as _get_svc


async def _embed_image(source):
    return await _get_svc().embed_image(source)


async def _embed_text(text):
    return await _get_svc().embed_text(text)


def _truncate(vec: list[float], dim: int | None = None) -> list[float]:
    if dim is None:
        dim = IMAGE_DIM
    out = list(vec[:dim]) if vec else []
    if len(out) < dim:
        out.extend([0.0] * (dim - len(out)))
    return out


def _truncate_and_normalize(vec: list[float], dim: int | None = None) -> list[float]:
    out = _truncate(vec, dim)
    norm = math.sqrt(sum(x * x for x in out))
    if norm > 0:
        out = [x / norm for x in out]
    return out

log = logging.getLogger(__name__)


TABLE_NAME = "images"
IMAGE_DIM = 1024

_db_cache: dict[str, "lancedb.DBConnection"] = {}


def _make_schema(dim: int = IMAGE_DIM) -> pa.Schema:
    return pa.schema([
        pa.field("id", pa.utf8()),
        pa.field("filename", pa.utf8()),
        pa.field("caption", pa.utf8()),
        pa.field("tags", pa.utf8()),
        pa.field("created_at", pa.utf8()),
        pa.field("vector", pa.list_(pa.float32(), dim)),
    ])


def _get_db(lancedb_dir: Path) -> "lancedb.DBConnection":
    lancedb_dir.mkdir(parents=True, exist_ok=True)
    key = str(lancedb_dir)
    if key not in _db_cache:
        _db_cache[key] = lancedb.connect(str(lancedb_dir))
    return _db_cache[key]


def _detect_table_dim(table) -> int | None:
    """偵測既有 LanceDB table 的向量維度。"""
    try:
        schema = table.schema
        for field in schema:
            if field.name == "vector":
                list_type = field.type
                if hasattr(list_type, "list_size"):
                    return list_type.list_size
    except Exception:
        pass
    return None


def _get_or_create_table(db):
    if TABLE_NAME in db.table_names():
        tbl = db.open_table(TABLE_NAME)
        existing_dim = _detect_table_dim(tbl)
        if existing_dim and existing_dim != IMAGE_DIM:
            log.warning(
                "Image table dim=%d != expected dim=%d, rebuilding table",
                existing_dim, IMAGE_DIM,
            )
            db.drop_table(TABLE_NAME)
            return db.create_table(TABLE_NAME, schema=_make_schema())
        return tbl
    return db.create_table(TABLE_NAME, schema=_make_schema())


def _images_dir(project_id: str) -> Path:
    """專案圖片存放目錄。"""
    d = project_root(project_id) / "images"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _lancedb_dir(project_id: str) -> Path:
    return project_root(project_id) / "lancedb"


def _dedup_filename(img_dir: Path, filename: str) -> str:
    """若檔名已存在，自動加上遞增尾碼避免覆蓋。"""
    base = Path(filename).stem
    ext = Path(filename).suffix
    candidate = filename
    idx = 1
    while (img_dir / candidate).exists():
        candidate = f"{base}_{idx}{ext}"
        idx += 1
    return candidate


AESTHETIC_ANALYSIS_PROMPT = """
你是一位策展人，正在為一位創作者建立視覺檔案。

第一步：辨識這張圖裡的物件或場域。
從以下選擇最接近的類型（可複選）：
印刷品、字體排印、書籍、海報、包裝
繪畫、素描、版畫、攝影
裝置、雕塑、物件、模型
建築、空間、室內、展場
介面、螢幕、數位影像
身體、服裝、布料
自然、地景、材料本身

第二步：用藝術和設計的語彙描述它怎麼運作。
選有感覺的維度說，不用全說：

- 排版與版面：字距、行距、網格、留白的處理
- 材質與表面：紙張、塗層、印刷方式、觸感的視覺暗示
- 色彩系統：幾個顏色、怎麼配、冷暖和張力
- 構圖邏輯：重量分佈、視線動線、密度
- 尺度感：這個東西在空間裡是什麼感覺，壓迫還是謙遜
- 時代與脈絡：讓你想到哪個年代、哪個地方、哪個運動
- 製作的態度：精緻還是粗糙、控制還是放任、手工還是工業

格式：
[物件類型]
[藝術設計描述，3-5 句]

繁體中文，不超過 100 字。
"""


async def _auto_caption(image_bytes: bytes, filename: str = "") -> str:
    """用 vision LLM 自動生成圖片美學描述。失敗時回傳空字串。"""
    try:
        import httpx
        from settings import load_env
        env = load_env()

        model = env.get("RAG_LLM_MODEL", "") or env.get("LLM_MODEL", "")
        if not model:
            return ""
        # Strip openai/ prefix — API endpoint expects bare model name
        if model.startswith("openai/"):
            model = model.removeprefix("openai/")

        api_key = (
            env.get("RAG_API_KEY", "")
            or env.get("OPENAI_API_KEY", "")
            or env.get("OPENROUTER_API_KEY", "")
        )
        api_base = (
            env.get("RAG_API_BASE", "")
            or env.get("OPENAI_API_BASE", "")
            or "https://api.openai.com/v1"
        )

        # 圖片轉 base64
        b64 = base64.b64encode(image_bytes).decode()
        suffix = Path(filename).suffix.lower().lstrip(".")
        mime = {
            "jpg": "image/jpeg", "jpeg": "image/jpeg",
            "png": "image/png", "gif": "image/gif",
            "webp": "image/webp",
        }.get(suffix, "image/jpeg")

        from rag.rate_guard import get_rate_guard

        async def _call_caption():
            async with httpx.AsyncClient(timeout=60.0) as client:
                resp = await client.post(
                    f"{api_base}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": model,
                        "messages": [{
                            "role": "user",
                            "content": [
                                {"type": "text", "text": AESTHETIC_ANALYSIS_PROMPT},
                                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
                            ],
                        }],
                        "temperature": 0.3,
                        "max_tokens": 300,
                    },
                )
                if resp.status_code != 200:
                    log.warning("Auto-caption API error %d: %s", resp.status_code, resp.text[:200])
                    return ""
                data = resp.json()
                return (data["choices"][0]["message"]["content"] or "").strip()

        guard = get_rate_guard()
        caption = await guard.call_with_retry(
            "image/auto_caption", _call_caption, max_retries=2,
        )
        log.info("Auto-caption generated: %s", caption[:80])
        return caption
    except Exception as e:
        log.warning("Auto-caption failed for %s: %s", filename, e)
        return ""


CHAT_DESCRIBE_PROMPT = """你是一位策展人的視覺助手。使用者正在對話中引用這張圖片，請提供詳細的視覺描述。

請描述：
1. 這張圖是什麼（物件類型、媒材）
2. 視覺特徵：色彩、構圖、排版、材質、字體等
3. 設計語言：風格、時代感、情緒、態度
4. 值得注意的細節

繁體中文，200-300 字。重點放在使用者的問題可能關心的面向。"""


async def describe_image_for_chat(
    image_path: str | Path,
    user_question: str = "",
) -> str:
    """聊天時即時看圖描述。用 RAG LLM（需支援 vision）。失敗時回傳空字串。"""
    try:
        import httpx
        from settings import load_env
        env = load_env()

        model = env.get("RAG_LLM_MODEL", "") or env.get("LLM_MODEL", "")
        if not model:
            return ""
        if model.startswith("openai/"):
            model = model.removeprefix("openai/")

        api_key = (
            env.get("RAG_API_KEY", "")
            or env.get("OPENAI_API_KEY", "")
            or env.get("OPENROUTER_API_KEY", "")
        )
        api_base = (
            env.get("RAG_API_BASE", "")
            or env.get("OPENAI_API_BASE", "")
            or "https://api.openai.com/v1"
        )

        img_path = Path(image_path)
        image_bytes = img_path.read_bytes()
        b64 = base64.b64encode(image_bytes).decode()
        suffix = img_path.suffix.lower().lstrip(".")
        mime = {
            "jpg": "image/jpeg", "jpeg": "image/jpeg",
            "png": "image/png", "gif": "image/gif",
            "webp": "image/webp",
        }.get(suffix, "image/jpeg")

        prompt = CHAT_DESCRIBE_PROMPT
        if user_question:
            prompt += f"\n\n使用者的問題：{user_question}"

        from rag.rate_guard import get_rate_guard

        async def _call_describe():
            async with httpx.AsyncClient(timeout=60.0) as client:
                resp = await client.post(
                    f"{api_base}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": model,
                        "messages": [{
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
                            ],
                        }],
                        "temperature": 0.3,
                        "max_tokens": 500,
                    },
                )
                if resp.status_code != 200:
                    log.warning("Chat describe API error %d: %s", resp.status_code, resp.text[:200])
                    return ""
                data = resp.json()
                return (data["choices"][0]["message"]["content"] or "").strip()

        guard = get_rate_guard()
        desc = await guard.call_with_retry(
            "image/chat_describe", _call_describe, max_retries=2,
        )
        log.info("Chat image description generated: %s", desc[:80])
        return desc
    except Exception as e:
        log.warning("Chat image description failed for %s: %s", image_path, e)
        return ""


async def save_image_file(
    project_id: str,
    filename: str,
    image_bytes: bytes,
) -> dict:
    """只存檔，不做 API 呼叫（毫秒級完成）。"""
    img_dir = _images_dir(project_id)
    filename = _dedup_filename(img_dir, filename)
    img_path = img_dir / filename
    img_path.write_bytes(image_bytes)
    return {"filename": filename, "path": str(img_path)}


async def index_image(
    project_id: str,
    filename: str,
    caption: str = "",
    tags: str = "",
) -> dict:
    """為已存檔的圖片建立 auto_caption + embedding + LanceDB 索引。"""
    import uuid
    img_path = _images_dir(project_id) / filename
    image_bytes = img_path.read_bytes()

    if not caption.strip():
        caption = await _auto_caption(image_bytes, filename)

    img_vec = _truncate(await _embed_image(image_bytes))
    final_vec = _truncate_and_normalize(img_vec)
    if caption.strip():
        try:
            txt_vec = _truncate(await _embed_text(caption))
            mixed = [(a * 0.5) + (b * 0.5) for a, b in zip(img_vec, txt_vec)]
            final_vec = _truncate_and_normalize(mixed)
        except Exception as e:
            log.warning("Caption embedding failed for %s, use image vector only: %s", filename, e)

    image_id = str(uuid.uuid4())[:8]
    created_at = time.strftime("%Y-%m-%d %H:%M:%S")

    db = _get_db(_lancedb_dir(project_id))
    table = _get_or_create_table(db)
    row = {
        "id": image_id,
        "filename": filename,
        "caption": caption,
        "tags": tags,
        "created_at": created_at,
        "vector": final_vec,
    }
    table.add([row])

    log.info("Indexed image %s: %s", filename, caption[:40])
    return {
        "id": image_id,
        "filename": filename,
        "caption": caption,
        "tags": tags,
        "created_at": created_at,
        "path": str(img_path),
    }


async def add_image(
    project_id: str,
    filename: str,
    image_bytes: bytes,
    caption: str = "",
    tags: str = "",
) -> dict:
    """儲存圖片檔案並建立向量索引（向後相容）。"""
    saved = await save_image_file(project_id, filename, image_bytes)
    return await index_image(project_id, saved["filename"], caption=caption, tags=tags)


async def search_images(
    project_id: str,
    query: str,
    limit: int = 5,
) -> list[dict]:
    """用文字語意搜尋圖片。回傳最相關的圖片 metadata。"""
    db = _get_db(_lancedb_dir(project_id))
    if TABLE_NAME not in db.table_names():
        return []

    table = db.open_table(TABLE_NAME)
    if table.count_rows() == 0:
        return []

    query_vec = await _embed_text(query)
    results = (
        table.search(query_vec, vector_column_name="vector")
        .metric("cosine")
        .limit(limit)
        .to_list()
    )

    img_dir = _images_dir(project_id)
    return [
        {
            "id": r.get("id", ""),
            "filename": r.get("filename", ""),
            "caption": r.get("caption", ""),
            "tags": r.get("tags", ""),
            "created_at": r.get("created_at", ""),
            "path": str(img_dir / r.get("filename", "")),
            "score": 1.0 - r.get("_distance", 0),
        }
        for r in results
    ]


async def list_images(project_id: str) -> list[dict]:
    """列出專案所有圖片 metadata。"""
    db = _get_db(_lancedb_dir(project_id))
    if TABLE_NAME not in db.table_names():
        return []

    table = db.open_table(TABLE_NAME)
    if table.count_rows() == 0:
        return []

    rows = table.to_pandas().drop(columns=["vector"], errors="ignore")
    img_dir = _images_dir(project_id)
    result = []
    for _, r in rows.iterrows():
        result.append({
            "id": r.get("id", ""),
            "filename": r.get("filename", ""),
            "caption": r.get("caption", ""),
            "tags": r.get("tags", ""),
            "created_at": r.get("created_at", ""),
            "path": str(img_dir / r.get("filename", "")),
        })
    return result


async def delete_image(project_id: str, image_id: str) -> bool:
    """刪除圖片（索引 + 檔案）。"""
    db = _get_db(_lancedb_dir(project_id))
    if TABLE_NAME not in db.table_names():
        return False

    table = db.open_table(TABLE_NAME)
    # 先取 filename 以便刪檔
    try:
        df = table.to_pandas()
        match = df[df["id"] == image_id]
        if not match.empty:
            filename = match.iloc[0].get("filename", "")
            if filename:
                img_path = _images_dir(project_id) / filename
                img_path.unlink(missing_ok=True)
    except Exception:
        pass

    safe_id = image_id.replace("'", "''")
    table.delete(f"id = '{safe_id}'")
    return True


async def update_image(project_id: str, image_id: str, caption: str = "", tags: str = "") -> bool:
    """更新圖片的 caption/tags。

    v0.7: 不再因 caption 變更重算向量（避免阻塞與重複重型推論）。
    """
    db = _get_db(_lancedb_dir(project_id))
    if TABLE_NAME not in db.table_names():
        return False

    table = db.open_table(TABLE_NAME)
    df = table.to_pandas()
    match = df[df["id"] == image_id]
    if match.empty:
        return False

    row = match.iloc[0].to_dict()
    safe_id = image_id.replace("'", "''")
    table.delete(f"id = '{safe_id}'")
    row["caption"] = caption
    row["tags"] = tags

    table.add([row])
    return True


async def backfill_captions(project_id: str, notify=None) -> dict:
    """批次為沒有 caption 的圖片自動生成描述並更新向量。

    回傳 {"updated": int, "failed": int, "total": int}
    """
    db = _get_db(_lancedb_dir(project_id))
    if TABLE_NAME not in db.table_names():
        return {"updated": 0, "failed": 0, "total": 0}

    table = db.open_table(TABLE_NAME)
    if table.count_rows() == 0:
        return {"updated": 0, "failed": 0, "total": 0}

    df = table.to_pandas()
    no_caption = df[df["caption"].fillna("").str.strip() == ""]
    total = len(no_caption)
    if total == 0:
        return {"updated": 0, "failed": 0, "total": 0}

    img_dir = _images_dir(project_id)
    updated = 0
    failed = 0

    for _, row in no_caption.iterrows():
        filename = row.get("filename", "")
        image_id = row.get("id", "")
        img_path = img_dir / filename

        if notify:
            notify(f"生成描述中：{filename}（{updated + failed + 1}/{total}）")

        if not img_path.exists():
            log.warning("Image file not found for backfill: %s", img_path)
            failed += 1
            continue

        caption = await _auto_caption(img_path.read_bytes(), filename)
        if not caption:
            failed += 1
            continue

        ok = await update_image(project_id, image_id, caption=caption, tags=row.get("tags", ""))
        if ok:
            updated += 1
            log.info("Backfilled caption for %s: %s", filename, caption[:60])
        else:
            failed += 1

    return {"updated": updated, "failed": failed, "total": total}


def get_selected_path(project_id: str) -> Path:
    """selected.json 路徑（統一放 DATA_ROOT）。"""
    return DATA_ROOT / "selected.json"


async def set_selected(project_id: str, image_ids: list[str]) -> list[dict]:
    """設定選取的圖片 ID，寫入 selected.json，回傳完整 metadata。"""
    import json as _json
    path = get_selected_path(project_id)
    path.write_text(_json.dumps(image_ids))

    if not image_ids:
        return []

    db = _get_db(_lancedb_dir(project_id))
    if TABLE_NAME not in db.table_names():
        return []

    table = db.open_table(TABLE_NAME)
    df = table.to_pandas().drop(columns=["vector"], errors="ignore")
    selected = df[df["id"].isin(image_ids)]
    img_dir = _images_dir(project_id)
    return [
        {
            "id": r.get("id", ""),
            "filename": r.get("filename", ""),
            "caption": r.get("caption", ""),
            "tags": r.get("tags", ""),
            "path": str(img_dir / r.get("filename", "")),
        }
        for _, r in selected.iterrows()
    ]


async def get_selected(project_id: str) -> list[dict]:
    """取得目前選取的圖片 metadata。"""
    import json as _json
    path = get_selected_path(project_id)
    if not path.exists():
        return []
    try:
        ids = _json.loads(path.read_text())
    except Exception:
        return []
    if not ids:
        return []
    return await set_selected(project_id, ids)
