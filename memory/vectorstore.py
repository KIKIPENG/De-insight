"""LanceDB 記憶向量資料庫 — 語意檢索記憶與知識。"""

import asyncio
import json
from pathlib import Path

import lancedb
import pyarrow as pa

DB_DIR = Path(__file__).parent.parent / "data" / "lancedb"
DB_DIR.mkdir(parents=True, exist_ok=True)

TABLE_NAME = "memories"

def _make_schema(dim: int = 1024) -> pa.Schema:
    return pa.schema([
        pa.field("id", pa.int64()),
        pa.field("type", pa.utf8()),
        pa.field("content", pa.utf8()),
        pa.field("topic", pa.utf8()),
        pa.field("source", pa.utf8()),
        pa.field("created_at", pa.utf8()),
        pa.field("vector", pa.list_(pa.float32(), dim)),
    ])

_db = None
_embed_fn = None
_embed_dim = None


def _get_db():
    global _db
    if _db is None:
        _db = lancedb.connect(str(DB_DIR))
    return _db


def _get_or_create_table(dim: int = 1024):
    db = _get_db()
    if TABLE_NAME in db.table_names():
        return db.open_table(TABLE_NAME)
    return db.create_table(TABLE_NAME, schema=_make_schema(dim))


async def _get_embedding_fn():
    """取得 embedding 函數，使用 Settings 設定的 provider。"""
    global _embed_fn, _embed_dim
    if _embed_fn is not None:
        return _embed_fn, _embed_dim

    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))
    from settings import load_env

    env = load_env()

    # 讀取 embed 設定（跟 knowledge_graph.py 同邏輯）
    embed_model = env.get("EMBED_MODEL", "")
    if embed_model:
        embed_key = env.get("EMBED_API_KEY", "") or env.get("JINA_API_KEY", "") or env.get("OPENAI_API_KEY", "")
        embed_base = env.get("EMBED_API_BASE", "https://api.openai.com/v1")
        _embed_dim = int(env.get("EMBED_DIM", "1024"))
        if env.get("EMBED_PROVIDER", "").startswith("ollama"):
            embed_key = "ollama"
            embed_base = "http://localhost:11434/v1"
    elif env.get("JINA_API_KEY"):
        embed_model = "jina-embeddings-v3"
        embed_key = env.get("JINA_API_KEY")
        embed_base = "https://api.jina.ai/v1"
        _embed_dim = 1024
    elif env.get("LLM_MODEL", "").startswith("ollama/"):
        embed_model = "nomic-embed-text"
        embed_key = "ollama"
        embed_base = "http://localhost:11434/v1"
        _embed_dim = 768
    else:
        embed_model = "text-embedding-3-small"
        embed_key = env.get("OPENAI_API_KEY", "")
        embed_base = "https://api.openai.com/v1"
        _embed_dim = 1536

    import httpx

    async def embed(texts: list[str]) -> list[list[float]]:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                f"{embed_base}/embeddings",
                headers={"Authorization": f"Bearer {embed_key}"},
                json={"model": embed_model, "input": texts},
            )
            resp.raise_for_status()
            data = resp.json()
            return [item["embedding"] for item in data["data"]]

    _embed_fn = embed
    return _embed_fn, _embed_dim


def reset_embed_fn():
    """設定變更後重置 embedding 函數。"""
    global _embed_fn, _embed_dim
    _embed_fn = None
    _embed_dim = None


async def index_memory(memory: dict) -> None:
    """將一條記憶加入向量資料庫。"""
    embed_fn, dim = await _get_embedding_fn()
    content = memory["content"]
    vectors = await embed_fn([content])

    table = _get_or_create_table()
    table.add([{
        "id": memory.get("id", 0),
        "type": memory.get("type", ""),
        "content": content,
        "topic": memory.get("topic", ""),
        "source": memory.get("source", ""),
        "created_at": memory.get("created_at", ""),
        "vector": vectors[0],
    }])


async def index_all_memories() -> int:
    """將所有 SQLite 記憶同步到 LanceDB。回傳索引數量。"""
    from memory.store import get_memories

    all_mems = await get_memories(limit=9999)
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

    # 重建 table
    db = _get_db()
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
            "vector": vec,
        })

    db.create_table(TABLE_NAME, data=rows, schema=_make_schema(dim))
    return len(rows)


async def search_similar(query: str, limit: int = 5, topic: str = "") -> list[dict]:
    """語意搜尋記憶。回傳最相關的記憶列表。"""
    db = _get_db()
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
        search = search.where(f"topic = '{topic}'")

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


async def delete_from_index(memory_id: int) -> None:
    """從向量資料庫刪除一條記憶。"""
    db = _get_db()
    if TABLE_NAME not in db.table_names():
        return
    table = db.open_table(TABLE_NAME)
    table.delete(f"id = {memory_id}")


def has_index() -> bool:
    """檢查向量索引是否存在且有資料。"""
    db = _get_db()
    if TABLE_NAME not in db.table_names():
        return False
    table = db.open_table(TABLE_NAME)
    return table.count_rows() > 0
