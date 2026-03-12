"""Claim store - SQLite-backed persistence for Claim entities."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import aiosqlite

from core.schemas import Claim, SourceKind


def _resolve_db_path(project_id: str) -> Path:
    """Resolve database path for a project."""
    try:
        from paths import project_root
        return project_root(project_id) / "core_claims.db"
    except ImportError:
        # Fallback if paths not available
        from paths import DATA_ROOT, APP_HOME
        return APP_HOME / DATA_ROOT.name / "projects" / project_id / "core_claims.db"


async def _get_db(db_path: Path) -> aiosqlite.Connection:
    """Get database connection with schema initialization."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    db = await aiosqlite.connect(str(db_path))
    db.row_factory = aiosqlite.Row

    await db.execute("""
        CREATE TABLE IF NOT EXISTS claims (
            claim_id TEXT PRIMARY KEY,
            project_id TEXT NOT NULL,
            source_kind TEXT NOT NULL,
            source_id TEXT,
            core_claim TEXT NOT NULL,
            critique_target TEXT DEFAULT '[]',
            value_axes TEXT DEFAULT '[]',
            materiality_axes TEXT DEFAULT '[]',
            labor_time_axes TEXT DEFAULT '[]',
            abstract_patterns TEXT DEFAULT '[]',
            theory_hints TEXT DEFAULT '[]',
            confidence REAL DEFAULT 0.5,
            created_at TEXT NOT NULL
        )
    """)

    await db.execute("""
        CREATE INDEX IF NOT EXISTS idx_claims_project
        ON claims(project_id)
    """)

    await db.commit()
    return db


class ClaimStore:
    """SQLite-backed store for Claim entities.

    Provides CRUD operations for claims extracted from user utterances,
    document passages, and image summaries.
    """

    def __init__(self, project_id: str = "default", db_path: Path | None = None):
        """Initialize store for a project.

        Args:
            project_id: Project identifier for isolation
            db_path: Optional custom database path
        """
        self.project_id = project_id
        self._db_path = db_path or _resolve_db_path(project_id)

    async def add(self, claim: Claim) -> Claim:
        """Add a new claim to the store.

        Args:
            claim: Claim to add

        Returns:
            The added claim with updated fields
        """
        db = await _get_db(self._db_path)
        try:
            await db.execute(
                """
                INSERT INTO claims (
                    claim_id, project_id, source_kind, source_id, core_claim,
                    critique_target, value_axes, materiality_axes, labor_time_axes,
                    abstract_patterns, theory_hints, confidence, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    claim.claim_id,
                    claim.project_id,
                    claim.source_kind.value if isinstance(claim.source_kind, SourceKind) else claim.source_kind,
                    claim.source_id,
                    claim.core_claim,
                    json.dumps(claim.critique_target, ensure_ascii=False),
                    json.dumps(claim.value_axes, ensure_ascii=False),
                    json.dumps(claim.materiality_axes, ensure_ascii=False),
                    json.dumps(claim.labor_time_axes, ensure_ascii=False),
                    json.dumps(claim.abstract_patterns, ensure_ascii=False),
                    json.dumps(claim.theory_hints, ensure_ascii=False),
                    claim.confidence,
                    claim.created_at.isoformat(),
                ),
            )
            await db.commit()
        finally:
            await db.close()

        return claim

    async def get(self, claim_id: str) -> Claim | None:
        """Retrieve a claim by ID.

        Args:
            claim_id: Claim identifier

        Returns:
            Claim if found, None otherwise
        """
        db = await _get_db(self._db_path)
        try:
            async with db.execute(
                "SELECT * FROM claims WHERE claim_id = ?",
                (claim_id,)
            ) as cursor:
                row = await cursor.fetchone()
                if row is None:
                    return None
                return self._row_to_claim(dict(row))
        finally:
            await db.close()

    async def list_by_project(
        self,
        project_id: str | None = None,
        limit: int = 100,
    ) -> list[Claim]:
        """List claims for a project.

        Args:
            project_id: Filter by project (uses default if None)
            limit: Maximum number of results

        Returns:
            List of claims
        """
        project_id = project_id or self.project_id
        db = await _get_db(self._db_path)
        try:
            async with db.execute(
                """
                SELECT * FROM claims
                WHERE project_id = ?
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (project_id, limit)
            ) as cursor:
                rows = await cursor.fetchall()
                return [self._row_to_claim(dict(row)) for row in rows]
        finally:
            await db.close()

    async def search_by_text(
        self,
        query: str,
        limit: int = 10,
    ) -> list[Claim]:
        """Search claims by text content (simple LIKE match).

        Args:
            query: Search query
            limit: Maximum results

        Returns:
            Matching claims
        """
        db = await _get_db(self._db_path)
        try:
            async with db.execute(
                """
                SELECT * FROM claims
                WHERE core_claim LIKE ? AND project_id = ?
                ORDER BY confidence DESC
                LIMIT ?
                """,
                (f"%{query}%", self.project_id, limit)
            ) as cursor:
                rows = await cursor.fetchall()
                return [self._row_to_claim(dict(row)) for row in rows]
        finally:
            await db.close()

    async def search_by_structure(
        self,
        value_axes: list[str] | None = None,
        abstract_patterns: list[str] | None = None,
        theory_hints: list[str] | None = None,
        critique_target: list[str] | None = None,
        limit: int = 10,
    ) -> list[Claim]:
        """Search claims by structural dimensions using JSON field matching.

        Finds claims that share structural patterns, value axes, or theory
        connections — regardless of surface-level text similarity.

        Args:
            value_axes: Value dimensions to match
            abstract_patterns: Structural patterns to match
            theory_hints: Theory directions to match
            critique_target: Critique targets to match
            limit: Maximum results

        Returns:
            Claims ranked by number of matching dimensions
        """
        db = await _get_db(self._db_path)
        try:
            search_terms: list[str] = []
            if value_axes:
                search_terms.extend(value_axes)
            if abstract_patterns:
                search_terms.extend(abstract_patterns)
            if theory_hints:
                search_terms.extend(theory_hints)
            if critique_target:
                search_terms.extend(critique_target)

            if not search_terms:
                return []

            # Build OR conditions across structural columns
            conditions = []
            params: list[str] = [self.project_id]
            for term in search_terms:
                term_lower = term.lower()
                conditions.append(
                    "(LOWER(value_axes) LIKE ? OR LOWER(abstract_patterns) LIKE ? "
                    "OR LOWER(theory_hints) LIKE ? OR LOWER(critique_target) LIKE ?)"
                )
                params.extend([f"%{term_lower}%"] * 4)

            where_clause = " OR ".join(conditions)
            query = f"""
                SELECT * FROM claims
                WHERE project_id = ? AND ({where_clause})
                ORDER BY confidence DESC
                LIMIT ?
            """
            params.append(str(limit))

            async with db.execute(query, params) as cursor:
                rows = await cursor.fetchall()
                return [self._row_to_claim(dict(row)) for row in rows]
        finally:
            await db.close()

    async def update(self, claim: Claim) -> Claim:
        """Update an existing claim.

        Args:
            claim: Claim with updated fields

        Returns:
            Updated claim
        """
        db = await _get_db(self._db_path)
        try:
            await db.execute(
                """
                UPDATE claims SET
                    core_claim = ?,
                    critique_target = ?,
                    value_axes = ?,
                    materiality_axes = ?,
                    labor_time_axes = ?,
                    abstract_patterns = ?,
                    theory_hints = ?,
                    confidence = ?
                WHERE claim_id = ?
                """,
                (
                    claim.core_claim,
                    json.dumps(claim.critique_target, ensure_ascii=False),
                    json.dumps(claim.value_axes, ensure_ascii=False),
                    json.dumps(claim.materiality_axes, ensure_ascii=False),
                    json.dumps(claim.labor_time_axes, ensure_ascii=False),
                    json.dumps(claim.abstract_patterns, ensure_ascii=False),
                    json.dumps(claim.theory_hints, ensure_ascii=False),
                    claim.confidence,
                    claim.claim_id,
                ),
            )
            await db.commit()
        finally:
            await db.close()

        return claim

    async def delete(self, claim_id: str) -> bool:
        """Delete a claim by ID.

        Args:
            claim_id: Claim identifier

        Returns:
            True if deleted, False if not found
        """
        db = await _get_db(self._db_path)
        try:
            cursor = await db.execute(
                "DELETE FROM claims WHERE claim_id = ?",
                (claim_id,)
            )
            await db.commit()
            return cursor.rowcount > 0
        finally:
            await db.close()

    @staticmethod
    def _row_to_claim(row: dict[str, Any]) -> Claim:
        """Convert database row to Claim model."""
        return Claim(
            claim_id=row["claim_id"],
            project_id=row["project_id"],
            source_kind=SourceKind(row["source_kind"]),
            source_id=row["source_id"] or "",
            core_claim=row["core_claim"],
            critique_target=json.loads(row.get("critique_target", "[]")),
            value_axes=json.loads(row.get("value_axes", "[]")),
            materiality_axes=json.loads(row.get("materiality_axes", "[]")),
            labor_time_axes=json.loads(row.get("labor_time_axes", "[]")),
            abstract_patterns=json.loads(row.get("abstract_patterns", "[]")),
            theory_hints=json.loads(row.get("theory_hints", "[]")),
            confidence=row.get("confidence", 0.5),
            created_at=datetime.fromisoformat(row["created_at"]),
        )
