"""ThoughtUnit store - SQLite-backed persistence for ThoughtUnit entities."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import aiosqlite

from core.schemas import ThoughtStatus, ThoughtUnit


def _resolve_db_path(project_id: str) -> Path:
    """Resolve database path for a project."""
    try:
        from paths import project_root
        return project_root(project_id) / "core_thoughts.db"
    except ImportError:
        from paths import DATA_ROOT, APP_HOME
        return APP_HOME / DATA_ROOT.name / "projects" / project_id / "core_thoughts.db"


async def _get_db(db_path: Path) -> aiosqlite.Connection:
    """Get database connection with schema initialization."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    db = await aiosqlite.connect(str(db_path))
    db.row_factory = aiosqlite.Row

    await db.execute("""
        CREATE TABLE IF NOT EXISTS thought_units (
            thought_id TEXT PRIMARY KEY,
            project_id TEXT NOT NULL,
            title TEXT NOT NULL,
            summary TEXT NOT NULL,
            core_claim_ids TEXT DEFAULT '[]',
            value_axes TEXT DEFAULT '[]',
            recurring_patterns TEXT DEFAULT '[]',
            supporting_claim_ids TEXT DEFAULT '[]',
            status TEXT NOT NULL,
            last_updated_at TEXT NOT NULL
        )
    """)

    await db.execute("""
        CREATE INDEX IF NOT EXISTS idx_thoughts_project
        ON thought_units(project_id)
    """)

    await db.commit()
    return db


class ThoughtStore:
    """SQLite-backed store for ThoughtUnit entities.

    Provides CRUD operations for tracking user thoughts across conversations.
    """

    def __init__(self, project_id: str = "default", db_path: Path | None = None):
        """Initialize store for a project.

        Args:
            project_id: Project identifier for isolation
            db_path: Optional custom database path
        """
        self.project_id = project_id
        self._db_path = db_path or _resolve_db_path(project_id)

    async def add(self, thought: ThoughtUnit) -> ThoughtUnit:
        """Add a new thought unit.

        Args:
            thought: ThoughtUnit to add

        Returns:
            The added thought
        """
        db = await _get_db(self._db_path)
        try:
            await db.execute(
                """
                INSERT INTO thought_units (
                    thought_id, project_id, title, summary,
                    core_claim_ids, value_axes, recurring_patterns,
                    supporting_claim_ids, status, last_updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    thought.thought_id,
                    thought.project_id,
                    thought.title,
                    thought.summary,
                    json.dumps(thought.core_claim_ids, ensure_ascii=False),
                    json.dumps(thought.value_axes, ensure_ascii=False),
                    json.dumps(thought.recurring_patterns, ensure_ascii=False),
                    json.dumps(thought.supporting_claim_ids, ensure_ascii=False),
                    thought.status.value if isinstance(thought.status, ThoughtStatus) else thought.status,
                    thought.last_updated_at.isoformat(),
                ),
            )
            await db.commit()
        finally:
            await db.close()

        return thought

    async def get(self, thought_id: str) -> ThoughtUnit | None:
        """Retrieve a thought by ID.

        Args:
            thought_id: Thought identifier

        Returns:
            ThoughtUnit if found, None otherwise
        """
        db = await _get_db(self._db_path)
        try:
            async with db.execute(
                "SELECT * FROM thought_units WHERE thought_id = ?",
                (thought_id,)
            ) as cursor:
                row = await cursor.fetchone()
                if row is None:
                    return None
                return self._row_to_thought(dict(row))
        finally:
            await db.close()

    async def list_by_project(
        self,
        project_id: str | None = None,
        status: ThoughtStatus | None = None,
        limit: int = 50,
    ) -> list[ThoughtUnit]:
        """List thought units for a project.

        Args:
            project_id: Filter by project (uses default if None)
            status: Optional status filter
            limit: Maximum number of results

        Returns:
            List of thought units
        """
        project_id = project_id or self.project_id
        db = await _get_db(self._db_path)
        try:
            query = "SELECT * FROM thought_units WHERE project_id = ?"
            params: list[Any] = [project_id]

            if status is not None:
                query += " AND status = ?"
                status_val = status.value if isinstance(status, ThoughtStatus) else status
                params.append(status_val)

            query += " ORDER BY last_updated_at DESC LIMIT ?"
            params.append(limit)

            async with db.execute(query, params) as cursor:
                rows = await cursor.fetchall()
                return [self._row_to_thought(dict(row)) for row in rows]
        finally:
            await db.close()

    async def update(self, thought: ThoughtUnit) -> ThoughtUnit:
        """Update an existing thought unit.

        Args:
            thought: ThoughtUnit with updated fields

        Returns:
            Updated thought
        """
        db = await _get_db(self._db_path)
        try:
            await db.execute(
                """
                UPDATE thought_units SET
                    title = ?,
                    summary = ?,
                    core_claim_ids = ?,
                    value_axes = ?,
                    recurring_patterns = ?,
                    supporting_claim_ids = ?,
                    status = ?,
                    last_updated_at = ?
                WHERE thought_id = ?
                """,
                (
                    thought.title,
                    thought.summary,
                    json.dumps(thought.core_claim_ids, ensure_ascii=False),
                    json.dumps(thought.value_axes, ensure_ascii=False),
                    json.dumps(thought.recurring_patterns, ensure_ascii=False),
                    json.dumps(thought.supporting_claim_ids, ensure_ascii=False),
                    thought.status.value if isinstance(thought.status, ThoughtStatus) else thought.status,
                    thought.last_updated_at.isoformat(),
                    thought.thought_id,
                ),
            )
            await db.commit()
        finally:
            await db.close()

        return thought

    async def delete(self, thought_id: str) -> bool:
        """Delete a thought unit by ID.

        Args:
            thought_id: Thought identifier

        Returns:
            True if deleted, False if not found
        """
        db = await _get_db(self._db_path)
        try:
            cursor = await db.execute(
                "DELETE FROM thought_units WHERE thought_id = ?",
                (thought_id,)
            )
            await db.commit()
            return cursor.rowcount > 0
        finally:
            await db.close()

    async def find_by_claim(self, claim_id: str) -> list[ThoughtUnit]:
        """Find thought units that reference a claim.

        Args:
            claim_id: Claim identifier to search for

        Returns:
            List of thought units containing the claim
        """
        db = await _get_db(self._db_path)
        try:
            async with db.execute(
                """
                SELECT * FROM thought_units
                WHERE project_id = ?
                AND (core_claim_ids LIKE ? OR supporting_claim_ids LIKE ?)
                ORDER BY last_updated_at DESC
                """,
                (self.project_id, f'%"{claim_id}"%', f'%"{claim_id}"%')
            ) as cursor:
                rows = await cursor.fetchall()
                return [self._row_to_thought(dict(row)) for row in rows]
        finally:
            await db.close()

    @staticmethod
    def _row_to_thought(row: dict[str, Any]) -> ThoughtUnit:
        """Convert database row to ThoughtUnit model."""
        return ThoughtUnit(
            thought_id=row["thought_id"],
            project_id=row["project_id"],
            title=row["title"],
            summary=row["summary"],
            core_claim_ids=json.loads(row.get("core_claim_ids", "[]")),
            value_axes=json.loads(row.get("value_axes", "[]")),
            recurring_patterns=json.loads(row.get("recurring_patterns", "[]")),
            supporting_claim_ids=json.loads(row.get("supporting_claim_ids", "[]")),
            status=ThoughtStatus(row["status"]),
            last_updated_at=datetime.fromisoformat(row["last_updated_at"]),
        )
