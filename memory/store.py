"""SQLite 記憶系統 — async CRUD via aiosqlite."""

import json
from pathlib import Path

import aiosqlite

from paths import DATA_ROOT
from utils.db_pool import get_connection

_DEFAULT_DB = DATA_ROOT / "projects" / "default" / "memories.db"


def _resolve_db(db_path: Path | None) -> Path:
    return db_path or _DEFAULT_DB


_CREATE_TABLE = """\
CREATE TABLE IF NOT EXISTS memories (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    type       TEXT NOT NULL,
    content    TEXT NOT NULL,
    source     TEXT,
    topic      TEXT DEFAULT '',
    created_at DATETIME DEFAULT (datetime('now', 'localtime')),
    tags       TEXT DEFAULT '[]'
);
"""

_CREATE_PENDING_TABLE = """\
CREATE TABLE IF NOT EXISTS pending_memories (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    type       TEXT,
    content    TEXT,
    source     TEXT,
    created_at DATETIME DEFAULT (datetime('now', 'localtime'))
);
"""

_MIGRATE_TOPIC = "ALTER TABLE memories ADD COLUMN topic TEXT DEFAULT ''"
_MIGRATE_PROJECT_ID = "ALTER TABLE memories ADD COLUMN project_id TEXT DEFAULT NULL"
_MIGRATE_CATEGORY = "ALTER TABLE memories ADD COLUMN category TEXT DEFAULT ''"
_MIGRATE_PENDING_INDEX = "ALTER TABLE memories ADD COLUMN pending_index INTEGER DEFAULT 0"


async def _get_db(db_path: Path | None = None) -> aiosqlite.Connection:
    path = _resolve_db(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    db = await aiosqlite.connect(str(path))
    db.row_factory = aiosqlite.Row
    await db.execute(_CREATE_TABLE)
    await db.execute(_CREATE_PENDING_TABLE)
    # Migrate: add topic column if missing
    try:
        await db.execute(_MIGRATE_TOPIC)
    except Exception:
        pass  # already exists
    # Migrate: add project_id column if missing
    try:
        await db.execute(_MIGRATE_PROJECT_ID)
    except Exception:
        pass  # already exists
    # Migrate: add category column if missing
    try:
        await db.execute(_MIGRATE_CATEGORY)
    except Exception:
        pass  # already exists
    # Migrate: add pending_index column if missing
    try:
        await db.execute(_MIGRATE_PENDING_INDEX)
    except Exception:
        pass  # already exists
    await db.commit()
    return db


def _row_to_dict(row: aiosqlite.Row) -> dict[str, int | str | list[str]]:
    d = dict(row)
    d["tags"] = json.loads(d.get("tags") or "[]")
    return d


def _jaccard_similarity(text1: str, text2: str) -> float:
    """Calculate Jaccard similarity between two texts based on word tokens."""
    import re

    def tokenize(text: str) -> set[str]:
        # Simple word tokenization
        words = re.findall(r'\w+', text.lower())
        return set(words)

    set1 = tokenize(text1)
    set2 = tokenize(text2)

    if not set1 or not set2:
        return 0.0

    intersection = len(set1 & set2)
    union = len(set1 | set2)

    return intersection / union if union > 0 else 0.0


async def check_duplicate(
    content: str, threshold: float = 0.8, db_path: Path | None = None
) -> dict | None:
    """
    Check if similar memory already exists (Jaccard similarity > threshold).
    Returns the duplicate memory dict if found, None otherwise.
    """
    path = _resolve_db(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    async with get_connection(str(path)) as db:
        # Get recent memories (last 50) to check against
        cursor = await db.execute(
            "SELECT id, content FROM memories ORDER BY created_at DESC LIMIT 50"
        )
        rows = await cursor.fetchall()

        for row in rows:
            existing_content = row["content"]
            similarity = _jaccard_similarity(content, existing_content)

            if similarity > threshold:
                # Found a duplicate, fetch full record
                cursor = await db.execute(
                    "SELECT * FROM memories WHERE id = ?", (row["id"],)
                )
                dup_row = await cursor.fetchone()
                return _row_to_dict(dup_row) if dup_row else None

    return None


async def add_memory(
    type: str, content: str, source: str = "", topic: str = "",
    category: str = "",
    tags: list[str] | None = None, project_id: str | None = None,
    db_path: Path | None = None,
) -> int:
    tags_json = json.dumps(tags or [], ensure_ascii=False)
    path = _resolve_db(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    async with get_connection(str(path)) as db:
        cursor = await db.execute(
            "INSERT INTO memories (type, content, source, topic, category, tags, project_id) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (type, content, source, topic, category, tags_json, project_id),
        )
        await db.commit()
        mem_id = cursor.lastrowid

        # 同步到向量資料庫，失敗時標記待補跑
        lancedb_dir = db_path.parent / "lancedb" if db_path else None
        try:
            from memory.vectorstore import index_memory
            await index_memory({
                "id": mem_id, "type": type, "content": content,
                "topic": topic, "source": source, "created_at": "",
            }, lancedb_dir=lancedb_dir)
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(f"index_memory failed id={mem_id}: {e}")
            await _mark_pending_index(mem_id, db_path=db_path)

        return mem_id


async def get_memories(type: str | None = None, limit: int = 20, project_id: str | None = None, category: str | None = None, db_path: Path | None = None) -> list[dict]:
    path = _resolve_db(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    async with get_connection(str(path)) as db:
        conditions = []
        params = []
        if type:
            conditions.append("type = ?")
            params.append(type)
        if category:
            conditions.append("category = ?")
            params.append(category)
        where = (" WHERE " + " AND ".join(conditions)) if conditions else ""
        params.append(limit)
        cursor = await db.execute(
            f"SELECT * FROM memories{where} ORDER BY created_at DESC LIMIT ?",
            tuple(params),
        )
        rows = await cursor.fetchall()
        return [_row_to_dict(r) for r in rows]


async def delete_memory(id: int, db_path: Path | None = None) -> None:
    path = _resolve_db(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    async with get_connection(str(path)) as db:
        await db.execute("DELETE FROM memories WHERE id = ?", (id,))
        await db.commit()

    # 同步刪除向量索引
    lancedb_dir = db_path.parent / "lancedb" if db_path else None
    try:
        from memory.vectorstore import delete_from_index
        await delete_from_index(id, lancedb_dir=lancedb_dir)
    except Exception:
        pass


async def search_memories(query: str, limit: int = 5, db_path: Path | None = None) -> list[dict]:
    # 優先用向量搜尋
    lancedb_dir = db_path.parent / "lancedb" if db_path else None
    try:
        from memory.vectorstore import search_similar, has_index
        if has_index(lancedb_dir=lancedb_dir):
            results = await search_similar(query, limit=limit, lancedb_dir=lancedb_dir)
            if results:
                return results
    except Exception:
        pass

    # Fallback: SQLite LIKE
    path = _resolve_db(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    async with get_connection(str(path)) as db:
        cursor = await db.execute(
            "SELECT * FROM memories WHERE content LIKE ? ORDER BY created_at DESC LIMIT ?",
            (f"%{query}%", limit),
        )
        rows = await cursor.fetchall()
        return [_row_to_dict(r) for r in rows]


async def get_topics(db_path: Path | None = None) -> list[str]:
    """取得所有不重複的 topic。"""
    path = _resolve_db(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    async with get_connection(str(path)) as db:
        cursor = await db.execute(
            "SELECT DISTINCT topic FROM memories WHERE topic != '' ORDER BY topic"
        )
        rows = await cursor.fetchall()
        return [r["topic"] for r in rows]


async def get_memories_by_topic(topic: str, limit: int = 50, db_path: Path | None = None) -> list[dict]:
    path = _resolve_db(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    async with get_connection(str(path)) as db:
        cursor = await db.execute(
            "SELECT * FROM memories WHERE topic = ? ORDER BY created_at DESC LIMIT ?",
            (topic, limit),
        )
        rows = await cursor.fetchall()
        return [_row_to_dict(r) for r in rows]


async def update_memory_topic(id: int, topic: str, db_path: Path | None = None) -> None:
    path = _resolve_db(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    async with get_connection(str(path)) as db:
        await db.execute("UPDATE memories SET topic = ? WHERE id = ?", (topic, id))
        await db.commit()


async def get_memory_stats(project_id: str | None = None, db_path: Path | None = None) -> dict:
    """取得記憶統計：類型數量、主題數量、總數。"""
    path = _resolve_db(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    async with get_connection(str(path)) as db:
        cursor = await db.execute("SELECT COUNT(*) as total FROM memories")
        total = (await cursor.fetchone())["total"]
        cursor = await db.execute(
            "SELECT type, COUNT(*) as cnt FROM memories GROUP BY type"
        )
        by_type = {r["type"]: r["cnt"] for r in await cursor.fetchall()}
        cursor = await db.execute(
            "SELECT topic, COUNT(*) as cnt FROM memories WHERE topic != '' GROUP BY topic ORDER BY cnt DESC"
        )
        by_topic = {r["topic"]: r["cnt"] for r in await cursor.fetchall()}
        return {"total": total, "by_type": by_type, "by_topic": by_topic}


async def _mark_pending_index(memory_id: int, db_path: Path | None = None) -> None:
    """標記此記憶的向量索引待補跑。"""
    path = _resolve_db(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    async with get_connection(str(path)) as db:
        await db.execute("UPDATE memories SET pending_index = 1 WHERE id = ?", (memory_id,))
        await db.commit()


async def _clear_pending_index(memory_id: int, db_path: Path | None = None) -> None:
    """補跑成功後清除標記。"""
    path = _resolve_db(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    async with get_connection(str(path)) as db:
        await db.execute("UPDATE memories SET pending_index = 0 WHERE id = ?", (memory_id,))
        await db.commit()


async def reindex_pending(db_path: Path | None = None, lancedb_dir: Path | None = None) -> int:
    """啟動時呼叫，補跑所有 pending_index=1 的記憶。回傳補跑數量。"""
    db = await _get_db(db_path)
    try:
        cursor = await db.execute("SELECT * FROM memories WHERE pending_index = 1")
        rows = await cursor.fetchall()
    finally:
        await db.close()

    if not rows:
        return 0

    from memory.vectorstore import index_memory
    _lancedb_dir = lancedb_dir or (db_path.parent / "lancedb" if db_path else None)
    count = 0
    for row in rows:
        mem = _row_to_dict(row)
        try:
            await index_memory(mem, lancedb_dir=_lancedb_dir)
            await _clear_pending_index(mem["id"], db_path=db_path)
            count += 1
        except Exception:
            pass  # 繼續下一條
    return count


async def save_pending_memories(items: list[dict], db_path: Path | None = None) -> None:
    """Save pending memory candidates to database so they survive app restarts."""
    if not items:
        return
    path = _resolve_db(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    async with get_connection(str(path)) as db:
        for item in items:
            await db.execute(
                "INSERT INTO pending_memories (type, content, source) VALUES (?, ?, ?)",
                (item.get("type"), item.get("content"), item.get("source", "")),
            )
        await db.commit()


async def load_pending_memories(db_path: Path | None = None) -> list[dict]:
    """Load pending memory candidates from database."""
    path = _resolve_db(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    async with get_connection(str(path)) as db:
        cursor = await db.execute("SELECT type, content, source FROM pending_memories ORDER BY id")
        rows = await cursor.fetchall()
        return [{"type": r["type"], "content": r["content"], "source": r["source"]} for r in rows]


async def clear_pending_memories(db_path: Path | None = None) -> None:
    """Clear all pending memory candidates from database."""
    path = _resolve_db(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    async with get_connection(str(path)) as db:
        await db.execute("DELETE FROM pending_memories")
        await db.commit()
