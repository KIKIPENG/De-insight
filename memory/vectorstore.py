"""LanceDB 記憶向量資料庫 — 語意檢索記憶與知識。"""

import asyncio
import json
import logging
import os
from pathlib import Path

import lancedb
import pyarrow as pa

from paths import DATA_ROOT

_DEFAULT_DB_DIR = DATA_ROOT / "projects" / "default" / "lancedb"

TABLE_NAME = "memories"

_db_cache: dict[str, "lancedb.DBConnection"] = {}

_embed_fn = None
_embed_dim = None

log = logging.getLogger(__name__)


def _make_schema(dim: int = 1024) -> pa.Schema:
    return pa.schema([
        pa.field("id", pa.int64()),
        pa.field("type", pa.utf8()),
        pa.field("content", pa.utf8()),
        pa.field("topic", pa.utf8()),
        pa.field("source", pa.utf8()),
        pa.field("created_at", pa.utf8()),
        pa.field("project_id", pa.utf8()),
        pa.field("vector", pa.list_(pa.float32(), dim)),
    ])


def _get_db(db_dir: Path | None = None):
    dir_ = db_dir or _DEFAULT_DB_DIR
    dir_.mkdir(parents=True, exist_ok=True)
    key = str(dir_)
    if key not in _db_cache:
        _db_cache[key] = lancedb.connect(str(dir_))
    return _db_cache[key]


def _get_or_create_table(db, dim: int = 1024):
    if TABLE_NAME in db.table_names():
        tbl = db.open_table(TABLE_NAME)
        existing_dim = _detect_vector_dim(tbl)
        if existing_dim and existing_dim != dim:
            log.warning(
                "memories table dim=%d != expected dim=%d, rebuilding table",
                existing_dim, dim,
            )
            db.drop_table(TABLE_NAME)
            return db.create_table(TABLE_NAME, schema=_make_schema(dim))
        return tbl
    return db.create_table(TABLE_NAME, schema=_make_schema(dim))


def _detect_vector_dim(table) -> int | None:
    """偵測既有 table 的 vector 維度。"""
    try:
        schema = table.schema
        for field in schema:
            if field.name == "vector":
                list_type = field.type
                if hasattr(list_type, "list_size"):
                    return list_type.list_size
        return None
    except Exception:
        return None


async def _get_embedding_fn():
    """取得 embedding 函數 — 透過 EmbeddingService。"""
    global _embed_fn, _embed_dim
    if _embed_fn is not None:
        return _embed_fn, _embed_dim

    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from embeddings.service import get_embedding_service

    svc = get_embedding_service()
    _embed_fn = svc.embed_texts
    _embed_dim = svc.dimension()
    return _embed_fn, _embed_dim


def reset_embed_fn():
    """設定變更後重置 embedding 函數。"""
    global _embed_fn, _embed_dim
    _embed_fn = None
    _embed_dim = None


async def index_memory(memory: dict, lancedb_dir: Path | None = None) -> None:
    """將一條記憶加入向量資料庫。"""
    embed_fn, dim = await _get_embedding_fn()
    content = memory["content"]
    vectors = await embed_fn([content])

    db = _get_db(lancedb_dir)
    table = _get_or_create_table(db, dim=dim)
    table.add([{
        "id": memory.get("id", 0),
        "type": memory.get("type", ""),
        "content": content,
        "topic": memory.get("topic", ""),
        "source": memory.get("source", ""),
        "created_at": memory.get("created_at", ""),
        "project_id": memory.get("project_id", "default") or "default",
        "vector": vectors[0],
    }])


async def index_all_memories(lancedb_dir: Path | None = None, db_path: Path | None = None) -> int:
    """將記憶同步到 LanceDB。回傳索引數量。"""
    from memory.store import get_memories

    all_mems = await get_memories(limit=9999, db_path=db_path)
    if not all_mems:
        return 0

    embed_fn, dim = await _get_embedding_fn()
    texts = [m["content"] for m in all_mems]

    # 分批 embed（每批 50 條）
    all_vectors = []
    batch_size = 50
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        vecs = await embed_fn(batch)
        all_vectors.extend(vecs)

    db = _get_db(lancedb_dir)

    # 全量重建
    if TABLE_NAME in db.table_names():
        db.drop_table(TABLE_NAME)

    rows = []
    for mem, vec in zip(all_mems, all_vectors):
        rows.append({
            "id": mem.get("id", 0),
            "type": mem.get("type", ""),
            "content": mem["content"],
            "topic": mem.get("topic", ""),
            "source": mem.get("source", ""),
            "created_at": mem.get("created_at", ""),
            "project_id": mem.get("project_id", "default") or "default",
            "vector": vec,
        })

    db.create_table(TABLE_NAME, data=rows, schema=_make_schema(dim))
    return len(rows)


async def search_similar(query: str, limit: int = 5, topic: str = "", lancedb_dir: Path | None = None) -> list[dict]:
    """語意搜尋記憶。回傳最相關的記憶列表。"""
    db = _get_db(lancedb_dir)
    if TABLE_NAME not in db.table_names():
        return []

    table = db.open_table(TABLE_NAME)
    if table.count_rows() == 0:
        return []

    embed_fn, _ = await _get_embedding_fn()
    vectors = await embed_fn([query])
    query_vec = vectors[0]

    search = table.search(query_vec, vector_column_name="vector").limit(limit)
    if topic:
        safe_topic = topic.replace("'", "''")
        search = search.where(f"topic = '{safe_topic}'")

    results = search.to_list()
    return [
        {
            "id": r.get("id", 0),
            "type": r.get("type", ""),
            "content": r.get("content", ""),
            "topic": r.get("topic", ""),
            "source": r.get("source", ""),
            "created_at": r.get("created_at", ""),
            "score": 1.0 - r.get("_distance", 0),  # cosine similarity
        }
        for r in results
    ]


async def delete_from_index(memory_id: int, lancedb_dir: Path | None = None) -> None:
    """從向量資料庫刪除一條記憶。"""
    db = _get_db(lancedb_dir)
    if TABLE_NAME not in db.table_names():
        return
    table = db.open_table(TABLE_NAME)
    table.delete(f"id = {memory_id}")


def has_index(lancedb_dir: Path | None = None) -> bool:
    """檢查向量索引是否存在且有資料。"""
    db = _get_db(lancedb_dir)
    if TABLE_NAME not in db.table_names():
        return False
    table = db.open_table(TABLE_NAME)
    return table.count_rows() > 0
