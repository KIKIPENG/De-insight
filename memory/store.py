"""SQLite 記憶系統 — async CRUD via aiosqlite."""

import json
from pathlib import Path

import aiosqlite

DB_PATH = Path(__file__).parent.parent / "data" / "memories.db"
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

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

_MIGRATE_TOPIC = "ALTER TABLE memories ADD COLUMN topic TEXT DEFAULT ''"
_MIGRATE_PROJECT_ID = "ALTER TABLE memories ADD COLUMN project_id TEXT DEFAULT NULL"
_MIGRATE_CATEGORY = "ALTER TABLE memories ADD COLUMN category TEXT DEFAULT ''"


async def _get_db() -> aiosqlite.Connection:
    db = await aiosqlite.connect(str(DB_PATH))
    db.row_factory = aiosqlite.Row
    await db.execute(_CREATE_TABLE)
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
    await db.commit()
    return db


def _row_to_dict(row: aiosqlite.Row) -> dict:
    d = dict(row)
    d["tags"] = json.loads(d.get("tags") or "[]")
    return d


async def add_memory(
    type: str, content: str, source: str = "", topic: str = "",
    category: str = "",
    tags: list[str] | None = None, project_id: str | None = None,
) -> int:
    tags_json = json.dumps(tags or [], ensure_ascii=False)
    db = await _get_db()
    try:
        cursor = await db.execute(
            "INSERT INTO memories (type, content, source, topic, category, tags, project_id) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (type, content, source, topic, category, tags_json, project_id),
        )
        await db.commit()
        mem_id = cursor.lastrowid

        # 同步到向量資料庫（背景，不阻塞）
        try:
            from memory.vectorstore import index_memory
            await index_memory({
                "id": mem_id, "type": type, "content": content,
                "topic": topic, "source": source, "created_at": "",
            })
        except Exception:
            pass  # embedding 未設定時不影響主流程

        return mem_id
    finally:
        await db.close()


async def get_memories(type: str | None = None, limit: int = 20, project_id: str | None = None, category: str | None = None) -> list[dict]:
    db = await _get_db()
    try:
        conditions = []
        params = []
        if type:
            conditions.append("type = ?")
            params.append(type)
        if project_id:
            conditions.append("project_id = ?")
            params.append(project_id)
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
    finally:
        await db.close()


async def delete_memory(id: int) -> None:
    db = await _get_db()
    try:
        await db.execute("DELETE FROM memories WHERE id = ?", (id,))
        await db.commit()
    finally:
        await db.close()
    # 同步刪除向量索引
    try:
        from memory.vectorstore import delete_from_index
        await delete_from_index(id)
    except Exception:
        pass


async def search_memories(query: str, limit: int = 5) -> list[dict]:
    # 優先用向量搜尋
    try:
        from memory.vectorstore import search_similar, has_index
        if has_index():
            results = await search_similar(query, limit=limit)
            if results:
                return results
    except Exception:
        pass

    # Fallback: SQLite LIKE
    db = await _get_db()
    try:
        cursor = await db.execute(
            "SELECT * FROM memories WHERE content LIKE ? ORDER BY created_at DESC LIMIT ?",
            (f"%{query}%", limit),
        )
        rows = await cursor.fetchall()
        return [_row_to_dict(r) for r in rows]
    finally:
        await db.close()


async def get_topics() -> list[str]:
    """取得所有不重複的 topic。"""
    db = await _get_db()
    try:
        cursor = await db.execute(
            "SELECT DISTINCT topic FROM memories WHERE topic != '' ORDER BY topic"
        )
        rows = await cursor.fetchall()
        return [r["topic"] for r in rows]
    finally:
        await db.close()


async def get_memories_by_topic(topic: str, limit: int = 50) -> list[dict]:
    db = await _get_db()
    try:
        cursor = await db.execute(
            "SELECT * FROM memories WHERE topic = ? ORDER BY created_at DESC LIMIT ?",
            (topic, limit),
        )
        rows = await cursor.fetchall()
        return [_row_to_dict(r) for r in rows]
    finally:
        await db.close()


async def update_memory_topic(id: int, topic: str) -> None:
    db = await _get_db()
    try:
        await db.execute("UPDATE memories SET topic = ? WHERE id = ?", (topic, id))
        await db.commit()
    finally:
        await db.close()


async def get_memory_stats() -> dict:
    """取得記憶統計：類型數量、主題數量、總數。"""
    db = await _get_db()
    try:
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
    finally:
        await db.close()
