"""對話歷史持久化。"""

import uuid
from pathlib import Path
import aiosqlite

DB_PATH = Path("data/conversations.db")


class ConversationStore:

    async def _ensure_db(self) -> None:
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        async with aiosqlite.connect(DB_PATH) as db:
            db.row_factory = aiosqlite.Row
            await db.executescript("""
                CREATE TABLE IF NOT EXISTS conversations (
                    id         TEXT PRIMARY KEY,
                    project_id TEXT DEFAULT NULL,
                    title      TEXT NOT NULL DEFAULT '未命名對話',
                    created_at DATETIME DEFAULT (datetime('now', 'localtime')),
                    updated_at DATETIME DEFAULT (datetime('now', 'localtime'))
                );
                CREATE TABLE IF NOT EXISTS messages (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    conversation_id TEXT NOT NULL,
                    role            TEXT NOT NULL,
                    content         TEXT NOT NULL,
                    created_at      DATETIME DEFAULT (datetime('now', 'localtime'))
                );
            """)
            await db.commit()

    async def create_conversation(self, project_id: str | None = None) -> str:
        await self._ensure_db()
        cid = str(uuid.uuid4())
        async with aiosqlite.connect(DB_PATH) as db:
            await db.execute(
                "INSERT INTO conversations (id, project_id) VALUES (?, ?)",
                (cid, project_id)
            )
            await db.commit()
        return cid

    async def add_message(self, conversation_id: str, role: str, content: str) -> None:
        async with aiosqlite.connect(DB_PATH) as db:
            await db.execute(
                "INSERT INTO messages (conversation_id, role, content) VALUES (?, ?, ?)",
                (conversation_id, role, content)
            )
            await db.execute(
                "UPDATE conversations SET updated_at = datetime('now','localtime') WHERE id = ?",
                (conversation_id,)
            )
            await db.commit()

    async def set_title(self, conversation_id: str, title: str) -> None:
        async with aiosqlite.connect(DB_PATH) as db:
            await db.execute(
                "UPDATE conversations SET title = ? WHERE id = ?",
                (title, conversation_id)
            )
            await db.commit()

    async def get_messages(self, conversation_id: str) -> list[dict]:
        await self._ensure_db()
        async with aiosqlite.connect(DB_PATH) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT role, content FROM messages WHERE conversation_id = ? ORDER BY id ASC",
                (conversation_id,)
            ) as cur:
                return [dict(r) for r in await cur.fetchall()]

    async def list_conversations(self, project_id: str | None = None) -> list[dict]:
        """列出對話，依 updated_at 倒序。project_id=None 回傳全部。"""
        await self._ensure_db()
        async with aiosqlite.connect(DB_PATH) as db:
            db.row_factory = aiosqlite.Row
            if project_id:
                async with db.execute(
                    "SELECT * FROM conversations WHERE project_id = ? ORDER BY updated_at DESC",
                    (project_id,)
                ) as cur:
                    return [dict(r) for r in await cur.fetchall()]
            else:
                async with db.execute(
                    "SELECT * FROM conversations ORDER BY updated_at DESC"
                ) as cur:
                    return [dict(r) for r in await cur.fetchall()]

    async def delete_conversation(self, conversation_id: str) -> None:
        async with aiosqlite.connect(DB_PATH) as db:
            await db.execute("DELETE FROM messages WHERE conversation_id = ?", (conversation_id,))
            await db.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))
            await db.commit()
