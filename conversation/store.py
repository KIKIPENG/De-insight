"""對話歷史持久化 + 文獻管理。"""

import json
import uuid
from pathlib import Path
import aiosqlite


class ConversationStore:

    def __init__(self, db_path: Path | None = None) -> None:
        from paths import DATA_ROOT
        self._db_path = db_path or (DATA_ROOT / "projects" / "default" / "conversations.db")

    async def _ensure_db(self) -> None:
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        async with aiosqlite.connect(self._db_path) as db:
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
                CREATE TABLE IF NOT EXISTS documents (
                    id          TEXT PRIMARY KEY,
                    project_id  TEXT NOT NULL DEFAULT 'default',
                    title       TEXT NOT NULL,
                    source_path TEXT,
                    source_type TEXT NOT NULL DEFAULT 'pdf',
                    file_size   INTEGER DEFAULT 0,
                    page_count  INTEGER DEFAULT 0,
                    tags        TEXT DEFAULT '[]',
                    imported_at DATETIME DEFAULT (datetime('now', 'localtime')),
                    cite_count  INTEGER DEFAULT 0
                );
            """)
            # Migrate: add note column if missing
            try:
                await db.execute("ALTER TABLE documents ADD COLUMN note TEXT DEFAULT ''")
            except Exception:
                pass  # already exists
            await db.commit()

    async def create_conversation(self, project_id: str | None = None) -> str:
        await self._ensure_db()
        cid = str(uuid.uuid4())
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(
                "INSERT INTO conversations (id, project_id) VALUES (?, ?)",
                (cid, project_id)
            )
            await db.commit()
        return cid

    async def add_message(self, conversation_id: str, role: str, content: str) -> None:
        async with aiosqlite.connect(self._db_path) as db:
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
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(
                "UPDATE conversations SET title = ? WHERE id = ?",
                (title, conversation_id)
            )
            await db.commit()

    async def get_messages(self, conversation_id: str) -> list[dict]:
        await self._ensure_db()
        async with aiosqlite.connect(self._db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT role, content FROM messages WHERE conversation_id = ? ORDER BY id ASC",
                (conversation_id,)
            ) as cur:
                return [dict(r) for r in await cur.fetchall()]

    async def get_conversation(self, conversation_id: str) -> dict | None:
        """取得單筆 conversation row（含 id, project_id, title 等）。"""
        await self._ensure_db()
        async with aiosqlite.connect(self._db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT * FROM conversations WHERE id = ?",
                (conversation_id,)
            ) as cur:
                row = await cur.fetchone()
                return dict(row) if row else None

    async def list_conversations(self, project_id: str | None = None) -> list[dict]:
        """列出對話，依 updated_at 倒序。project_id=None 回傳全部。"""
        await self._ensure_db()
        async with aiosqlite.connect(self._db_path) as db:
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
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute("DELETE FROM messages WHERE conversation_id = ?", (conversation_id,))
            await db.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))
            await db.commit()

    # ── Document CRUD ──

    async def add_document(
        self,
        title: str,
        source_path: str = "",
        source_type: str = "pdf",
        file_size: int = 0,
        page_count: int = 0,
        project_id: str = "default",
    ) -> str:
        await self._ensure_db()
        doc_id = str(uuid.uuid4())
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(
                "INSERT INTO documents (id, project_id, title, source_path, source_type, file_size, page_count) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (doc_id, project_id, title, source_path, source_type, file_size, page_count),
            )
            await db.commit()
        return doc_id

    async def list_documents(self, project_id: str = "default") -> list[dict]:
        await self._ensure_db()
        async with aiosqlite.connect(self._db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT * FROM documents WHERE project_id = ? ORDER BY imported_at DESC",
                (project_id,),
            ) as cur:
                rows = await cur.fetchall()
                return [dict(r) for r in rows]

    async def delete_document(self, doc_id: str) -> None:
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
            await db.commit()

    async def update_document_tags(self, doc_id: str, tags: list[str]) -> None:
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(
                "UPDATE documents SET tags = ? WHERE id = ?",
                (json.dumps(tags, ensure_ascii=False), doc_id),
            )
            await db.commit()

    async def update_document_meta(
        self, doc_id: str,
        title: str | None = None,
        tags: list[str] | None = None,
        note: str | None = None,
    ) -> None:
        await self._ensure_db()
        sets, params = [], []
        if title is not None:
            sets.append("title = ?")
            params.append(title)
        if tags is not None:
            sets.append("tags = ?")
            params.append(json.dumps(tags, ensure_ascii=False))
        if note is not None:
            sets.append("note = ?")
            params.append(note)
        if not sets:
            return
        params.append(doc_id)
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(
                f"UPDATE documents SET {', '.join(sets)} WHERE id = ?",
                tuple(params),
            )
            await db.commit()
