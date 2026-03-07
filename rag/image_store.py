"""圖片知識庫 — LanceDB 圖片語意索引（dim=512）。

每個專案有獨立的 images table，存放圖片 metadata + embedding。
支援文字語意搜圖（text-to-image retrieval）。
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path

import lancedb
import pyarrow as pa

from paths import DATA_ROOT, project_root

log = logging.getLogger(__name__)

TABLE_NAME = "images"
IMAGE_DIM = 512

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


def _get_or_create_table(db):
    if TABLE_NAME in db.table_names():
        return db.open_table(TABLE_NAME)
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


async def add_image(
    project_id: str,
    filename: str,
    image_bytes: bytes,
    caption: str = "",
    tags: str = "",
) -> dict:
    """儲存圖片檔案並建立向量索引。回傳 image metadata dict。"""
    import uuid
    from embeddings.local import embed_image, embed_text

    img_dir = _images_dir(project_id)
    filename = _dedup_filename(img_dir, filename)
    img_path = img_dir / filename
    img_path.write_bytes(image_bytes)

    # 混合向量：圖片 embedding + caption text embedding（如有）
    img_vec = await embed_image(image_bytes)
    if caption.strip():
        txt_vec = await embed_text(caption)
        # 50/50 混合後 L2 normalize
        import torch
        mixed = torch.tensor(img_vec) * 0.5 + torch.tensor(txt_vec) * 0.5
        mixed = torch.nn.functional.normalize(mixed, p=2, dim=0)
        final_vec = mixed.tolist()
    else:
        final_vec = img_vec

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

    return {
        "id": image_id,
        "filename": filename,
        "caption": caption,
        "tags": tags,
        "created_at": created_at,
        "path": str(img_path),
    }


async def search_images(
    project_id: str,
    query: str,
    limit: int = 5,
) -> list[dict]:
    """用文字語意搜尋圖片。回傳最相關的圖片 metadata。"""
    from embeddings.local import embed_text

    db = _get_db(_lancedb_dir(project_id))
    if TABLE_NAME not in db.table_names():
        return []

    table = db.open_table(TABLE_NAME)
    if table.count_rows() == 0:
        return []

    query_vec = await embed_text(query)
    results = table.search(query_vec, vector_column_name="vector").limit(limit).to_list()

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
    """更新圖片的 caption/tags（delete + re-add 保留 vector）。"""
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
