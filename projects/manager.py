import uuid
from pathlib import Path
import aiosqlite

from paths import app_db_path, ensure_project_dirs, project_root, GLOBAL_PROJECT_ID


class ProjectManager:

    async def _ensure_db(self) -> None:
        db_path = app_db_path()
        async with aiosqlite.connect(db_path) as db:
            await db.execute("""
                CREATE TABLE IF NOT EXISTS projects (
                    id TEXT PRIMARY KEY, name TEXT NOT NULL,
                    description TEXT DEFAULT '', color TEXT DEFAULT '#6b7280',
                    is_global INTEGER DEFAULT 0,
                    created_at DATETIME DEFAULT (datetime('now','localtime')),
                    last_active DATETIME DEFAULT (datetime('now','localtime'))
                )
            """)
            # 確保 is_global 欄位存在（舊資料庫遷移）
            try:
                await db.execute("SELECT is_global FROM projects LIMIT 1")
            except Exception:
                await db.execute("ALTER TABLE projects ADD COLUMN is_global INTEGER DEFAULT 0")
            await db.commit()
        # 確保全局文獻庫存在
        await self._ensure_global_library()

    async def _ensure_global_library(self) -> None:
        """確保全局文獻庫專案存在。"""
        async with aiosqlite.connect(app_db_path()) as db:
            async with db.execute(
                "SELECT id FROM projects WHERE id=?", (GLOBAL_PROJECT_ID,)
            ) as cur:
                if await cur.fetchone():
                    return
            await db.execute(
                "INSERT INTO projects (id, name, description, color, is_global) "
                "VALUES (?, ?, ?, ?, 1)",
                (GLOBAL_PROJECT_ID, "全局文獻庫", "跨專案的基礎思考奠定", "#f59e0b"),
            )
            await db.commit()
        ensure_project_dirs(GLOBAL_PROJECT_ID)

    async def create_project(self, name: str, description: str = "") -> dict:
        await self._ensure_db()
        pid = str(uuid.uuid4())
        async with aiosqlite.connect(app_db_path()) as db:
            await db.execute(
                "INSERT INTO projects (id,name,description) VALUES (?,?,?)",
                (pid, name, description))
            await db.commit()
        ensure_project_dirs(pid)
        return await self.get_project(pid)

    async def list_projects(self) -> list[dict]:
        """列出所有專案，全局文獻庫固定排在最前面。"""
        await self._ensure_db()
        async with aiosqlite.connect(app_db_path()) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT * FROM projects ORDER BY is_global DESC, last_active DESC"
            ) as cur:
                return [dict(r) for r in await cur.fetchall()]

    async def get_project(self, project_id: str) -> dict | None:
        async with aiosqlite.connect(app_db_path()) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT * FROM projects WHERE id=?", (project_id,)
            ) as cur:
                row = await cur.fetchone()
                return dict(row) if row else None

    async def delete_project(self, project_id: str) -> None:
        if project_id == GLOBAL_PROJECT_ID:
            raise ValueError("全局文獻庫不可刪除")
        import shutil
        async with aiosqlite.connect(app_db_path()) as db:
            await db.execute("DELETE FROM projects WHERE id=?", (project_id,))
            await db.commit()
        d = project_root(project_id)
        if d.exists():
            shutil.rmtree(d)

    def is_global_project(self, project_id: str) -> bool:
        return project_id == GLOBAL_PROJECT_ID

    async def touch_project(self, project_id: str) -> None:
        async with aiosqlite.connect(app_db_path()) as db:
            await db.execute(
                "UPDATE projects SET last_active=datetime('now','localtime') WHERE id=?",
                (project_id,))
            await db.commit()

    def get_project_data_dir(self, project_id: str) -> Path:
        return project_root(project_id)
