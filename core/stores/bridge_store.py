"""Bridge store - SQLite-backed persistence for Bridge entities."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import aiosqlite

from core.schemas import Bridge, BridgeType


def _resolve_db_path(project_id: str) -> Path:
    """Resolve database path for a project."""
    try:
        from paths import project_root
        return project_root(project_id) / "core_bridges.db"
    except ImportError:
        from paths import DATA_ROOT, APP_HOME
        return APP_HOME / DATA_ROOT.name / "projects" / project_id / "core_bridges.db"


async def _get_db(db_path: Path) -> aiosqlite.Connection:
    """Get database connection with schema initialization."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    db = await aiosqlite.connect(str(db_path))
    db.row_factory = aiosqlite.Row

    await db.execute("""
        CREATE TABLE IF NOT EXISTS bridges (
            bridge_id TEXT PRIMARY KEY,
            project_id TEXT NOT NULL,
            source_claim_id TEXT NOT NULL,
            target_claim_id TEXT NOT NULL,
            bridge_type TEXT NOT NULL,
            reason_summary TEXT NOT NULL,
            shared_patterns TEXT DEFAULT '[]',
            confidence REAL DEFAULT 0.5,
            created_at TEXT NOT NULL
        )
    """)

    await db.execute("""
        CREATE INDEX IF NOT EXISTS idx_bridges_project
        ON bridges(project_id)
    """)

    await db.execute("""
        CREATE INDEX IF NOT EXISTS idx_bridges_claims
        ON bridges(source_claim_id, target_claim_id)
    """)

    await db.commit()
    return db


class BridgeStore:
    """SQLite-backed store for Bridge entities.

    Provides CRUD operations for cross-domain and structural relationships
    between claims.
    """

    def __init__(self, project_id: str = "default", db_path: Path | None = None):
        """Initialize store for a project.

        Args:
            project_id: Project identifier for isolation
            db_path: Optional custom database path
        """
        self.project_id = project_id
        self._db_path = db_path or _resolve_db_path(project_id)

    async def add(self, bridge: Bridge) -> Bridge:
        """Add a new bridge.

        Args:
            bridge: Bridge to add

        Returns:
            The added bridge
        """
        db = await _get_db(self._db_path)
        try:
            await db.execute(
                """
                INSERT INTO bridges (
                    bridge_id, project_id, source_claim_id, target_claim_id,
                    bridge_type, reason_summary, shared_patterns,
                    confidence, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    bridge.bridge_id,
                    bridge.project_id,
                    bridge.source_claim_id,
                    bridge.target_claim_id,
                    bridge.bridge_type.value if isinstance(bridge.bridge_type, BridgeType) else bridge.bridge_type,
                    bridge.reason_summary,
                    json.dumps(bridge.shared_patterns, ensure_ascii=False),
                    bridge.confidence,
                    bridge.created_at.isoformat(),
                ),
            )
            await db.commit()
        finally:
            await db.close()

        return bridge

    async def add_many(self, bridges: list[Bridge]) -> list[Bridge]:
        """Add multiple bridges in a batch.

        Args:
            bridges: List of bridges to add

        Returns:
            The added bridges
        """
        if not bridges:
            return bridges

        db = await _get_db(self._db_path)
        try:
            for bridge in bridges:
                await db.execute(
                    """
                    INSERT INTO bridges (
                        bridge_id, project_id, source_claim_id, target_claim_id,
                        bridge_type, reason_summary, shared_patterns,
                        confidence, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        bridge.bridge_id,
                        bridge.project_id,
                        bridge.source_claim_id,
                        bridge.target_claim_id,
                        bridge.bridge_type.value if isinstance(bridge.bridge_type, BridgeType) else bridge.bridge_type,
                        bridge.reason_summary,
                        json.dumps(bridge.shared_patterns, ensure_ascii=False),
                        bridge.confidence,
                        bridge.created_at.isoformat(),
                    ),
                )
            await db.commit()
        finally:
            await db.close()

        return bridges

    async def get(self, bridge_id: str) -> Bridge | None:
        """Retrieve a bridge by ID.

        Args:
            bridge_id: Bridge identifier

        Returns:
            Bridge if found, None otherwise
        """
        db = await _get_db(self._db_path)
        try:
            async with db.execute(
                "SELECT * FROM bridges WHERE bridge_id = ?",
                (bridge_id,)
            ) as cursor:
                row = await cursor.fetchone()
                if row is None:
                    return None
                return self._row_to_bridge(dict(row))
        finally:
            await db.close()

    async def list_by_project(
        self,
        project_id: str | None = None,
        bridge_type: BridgeType | None = None,
        min_confidence: float = 0.0,
        limit: int = 50,
    ) -> list[Bridge]:
        """List bridges for a project.

        Args:
            project_id: Filter by project (uses default if None)
            bridge_type: Optional type filter
            min_confidence: Minimum confidence threshold
            limit: Maximum number of results

        Returns:
            List of bridges
        """
        project_id = project_id or self.project_id
        db = await _get_db(self._db_path)
        try:
            query = "SELECT * FROM bridges WHERE project_id = ? AND confidence >= ?"
            params: list[Any] = [project_id, min_confidence]

            if bridge_type is not None:
                bridge_type_val = bridge_type.value if isinstance(bridge_type, BridgeType) else bridge_type
                query += " AND bridge_type = ?"
                params.append(bridge_type_val)

            query += " ORDER BY confidence DESC LIMIT ?"
            params.append(limit)

            async with db.execute(query, params) as cursor:
                rows = await cursor.fetchall()
                return [self._row_to_bridge(dict(row)) for row in rows]
        finally:
            await db.close()

    async def find_by_claim(
        self,
        claim_id: str,
        limit: int = 20,
    ) -> list[Bridge]:
        """Find bridges connected to a specific claim.

        Args:
            claim_id: Claim identifier (as source or target)
            limit: Maximum results

        Returns:
            List of connected bridges
        """
        db = await _get_db(self._db_path)
        try:
            async with db.execute(
                """
                SELECT * FROM bridges
                WHERE project_id = ?
                AND (source_claim_id = ? OR target_claim_id = ?)
                ORDER BY confidence DESC
                LIMIT ?
                """,
                (self.project_id, claim_id, claim_id, limit)
            ) as cursor:
                rows = await cursor.fetchall()
                return [self._row_to_bridge(dict(row)) for row in rows]
        finally:
            await db.close()

    async def find_between_claims(
        self,
        source_claim_id: str,
        target_claim_id: str,
    ) -> Bridge | None:
        """Find a bridge connecting two specific claims.

        Args:
            source_claim_id: Source claim ID
            target_claim_id: Target claim ID

        Returns:
            Bridge if found, None otherwise
        """
        db = await _get_db(self._db_path)
        try:
            async with db.execute(
                """
                SELECT * FROM bridges
                WHERE project_id = ?
                AND (
                    (source_claim_id = ? AND target_claim_id = ?)
                    OR (source_claim_id = ? AND target_claim_id = ?)
                )
                """,
                (self.project_id, source_claim_id, target_claim_id, target_claim_id, source_claim_id)
            ) as cursor:
                row = await cursor.fetchone()
                if row is None:
                    return None
                return self._row_to_bridge(dict(row))
        finally:
            await db.close()

    async def delete(self, bridge_id: str) -> bool:
        """Delete a bridge by ID.

        Args:
            bridge_id: Bridge identifier

        Returns:
            True if deleted, False if not found
        """
        db = await _get_db(self._db_path)
        try:
            cursor = await db.execute(
                "DELETE FROM bridges WHERE bridge_id = ?",
                (bridge_id,)
            )
            await db.commit()
            return cursor.rowcount > 0
        finally:
            await db.close()

    async def delete_by_claim(self, claim_id: str) -> int:
        """Delete all bridges connected to a claim.

        Args:
            claim_id: Claim identifier

        Returns:
            Number of deleted bridges
        """
        db = await _get_db(self._db_path)
        try:
            cursor = await db.execute(
                """
                DELETE FROM bridges
                WHERE project_id = ?
                AND (source_claim_id = ? OR target_claim_id = ?)
                """,
                (self.project_id, claim_id, claim_id)
            )
            await db.commit()
            return cursor.rowcount
        finally:
            await db.close()

    @staticmethod
    def _row_to_bridge(row: dict[str, Any]) -> Bridge:
        """Convert database row to Bridge model."""
        return Bridge(
            bridge_id=row["bridge_id"],
            project_id=row["project_id"],
            source_claim_id=row["source_claim_id"],
            target_claim_id=row["target_claim_id"],
            bridge_type=BridgeType(row["bridge_type"]),
            reason_summary=row["reason_summary"],
            shared_patterns=json.loads(row.get("shared_patterns", "[]")),
            confidence=row.get("confidence", 0.5),
            created_at=datetime.fromisoformat(row["created_at"]),
        )
