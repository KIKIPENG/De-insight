import uuid
from pathlib import Path
import aiosqlite

from paths import app_db_path, ensure_project_dirs, project_root


class ProjectManager:

    async def _ensure_db(self) -> None:
        db_path = app_db_path()
        async with aiosqlite.connect(db_path) as db:
            await db.execute("""
                CREATE TABLE IF NOT EXISTS projects (
                    id TEXT PRIMARY KEY, name TEXT NOT NULL,
                    description TEXT DEFAULT '', color TEXT DEFAULT '#6b7280',
                    created_at DATETIME DEFAULT (datetime('now','localtime')),
                    last_active DATETIME DEFAULT (datetime('now','localtime'))
                )
            """)
            await db.commit()

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
        await self._ensure_db()
        async with aiosqlite.connect(app_db_path()) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT * FROM projects ORDER BY last_active DESC"
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
        import shutil
        async with aiosqlite.connect(app_db_path()) as db:
            await db.execute("DELETE FROM projects WHERE id=?", (project_id,))
            await db.commit()
        d = project_root(project_id)
        if d.exists():
            shutil.rmtree(d)

    async def touch_project(self, project_id: str) -> None:
        async with aiosqlite.connect(app_db_path()) as db:
            await db.execute(
                "UPDATE projects SET last_active=datetime('now','localtime') WHERE id=?",
                (project_id,))
            await db.commit()

    def get_project_data_dir(self, project_id: str) -> Path:
        return project_root(project_id)
