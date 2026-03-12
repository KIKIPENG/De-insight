"""ConceptMapping store - SQLite-backed persistence for ConceptMapping entities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import aiosqlite

from core.schemas import ConceptMapping, OwnerKind, VocabSource


def _resolve_db_path(project_id: str) -> Path:
    """Resolve database path for a project."""
    try:
        from paths import project_root
        return project_root(project_id) / "core_concepts.db"
    except ImportError:
        from paths import DATA_ROOT, APP_HOME
        return APP_HOME / DATA_ROOT.name / "projects" / project_id / "core_concepts.db"


async def _get_db(db_path: Path) -> aiosqlite.Connection:
    """Get database connection with schema initialization."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    db = await aiosqlite.connect(str(db_path))
    db.row_factory = aiosqlite.Row

    await db.execute("""
        CREATE TABLE IF NOT EXISTS concept_mappings (
            mapping_id TEXT PRIMARY KEY,
            project_id TEXT NOT NULL,
            owner_kind TEXT NOT NULL,
            owner_id TEXT NOT NULL,
            vocab_source TEXT NOT NULL,
            concept_id TEXT NOT NULL,
            preferred_label TEXT NOT NULL,
            alt_labels TEXT DEFAULT '[]',
            broader_terms TEXT DEFAULT '[]',
            related_terms TEXT DEFAULT '[]',
            confidence REAL DEFAULT 0.5
        )
    """)

    await db.execute("""
        CREATE INDEX IF NOT EXISTS idx_concepts_project
        ON concept_mappings(project_id)
    """)

    await db.execute("""
        CREATE INDEX IF NOT EXISTS idx_concepts_owner
        ON concept_mappings(owner_kind, owner_id)
    """)

    await db.commit()
    return db


class ConceptStore:
    """SQLite-backed store for ConceptMapping entities.

    Provides CRUD operations for mapping claims, passages, and thought units
    to controlled concepts (AAT or internal vocabulary).
    """

    def __init__(self, project_id: str = "default", db_path: Path | None = None):
        """Initialize store for a project.

        Args:
            project_id: Project identifier for isolation
            db_path: Optional custom database path
        """
        self.project_id = project_id
        self._db_path = db_path or _resolve_db_path(project_id)

    async def add(self, mapping: ConceptMapping) -> ConceptMapping:
        """Add a new concept mapping.

        Args:
            mapping: ConceptMapping to add

        Returns:
            The added mapping
        """
        db = await _get_db(self._db_path)
        try:
            await db.execute(
                """
                INSERT INTO concept_mappings (
                    mapping_id, project_id, owner_kind, owner_id,
                    vocab_source, concept_id, preferred_label,
                    alt_labels, broader_terms, related_terms, confidence
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    mapping.mapping_id,
                    mapping.project_id,
                    mapping.owner_kind.value if isinstance(mapping.owner_kind, OwnerKind) else mapping.owner_kind,
                    mapping.owner_id,
                    mapping.vocab_source.value if isinstance(mapping.vocab_source, VocabSource) else mapping.vocab_source,
                    mapping.concept_id,
                    mapping.preferred_label,
                    json.dumps(mapping.alt_labels, ensure_ascii=False),
                    json.dumps(mapping.broader_terms, ensure_ascii=False),
                    json.dumps(mapping.related_terms, ensure_ascii=False),
                    mapping.confidence,
                ),
            )
            await db.commit()
        finally:
            await db.close()

        return mapping

    async def add_many(self, mappings: list[ConceptMapping]) -> list[ConceptMapping]:
        """Add multiple concept mappings in a batch.

        Args:
            mappings: List of ConceptMappings to add

        Returns:
            The added mappings
        """
        if not mappings:
            return mappings

        db = await _get_db(self._db_path)
        try:
            for mapping in mappings:
                await db.execute(
                    """
                    INSERT INTO concept_mappings (
                        mapping_id, project_id, owner_kind, owner_id,
                        vocab_source, concept_id, preferred_label,
                        alt_labels, broader_terms, related_terms, confidence
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        mapping.mapping_id,
                        mapping.project_id,
                        mapping.owner_kind.value if isinstance(mapping.owner_kind, OwnerKind) else mapping.owner_kind,
                        mapping.owner_id,
                        mapping.vocab_source.value if isinstance(mapping.vocab_source, VocabSource) else mapping.vocab_source,
                        mapping.concept_id,
                        mapping.preferred_label,
                        json.dumps(mapping.alt_labels, ensure_ascii=False),
                        json.dumps(mapping.broader_terms, ensure_ascii=False),
                        json.dumps(mapping.related_terms, ensure_ascii=False),
                        mapping.confidence,
                    ),
                )
            await db.commit()
        finally:
            await db.close()

        return mappings

    async def get(self, mapping_id: str) -> ConceptMapping | None:
        """Retrieve a mapping by ID.

        Args:
            mapping_id: Mapping identifier

        Returns:
            ConceptMapping if found, None otherwise
        """
        db = await _get_db(self._db_path)
        try:
            async with db.execute(
                "SELECT * FROM concept_mappings WHERE mapping_id = ?",
                (mapping_id,)
            ) as cursor:
                row = await cursor.fetchone()
                if row is None:
                    return None
                return self._row_to_mapping(dict(row))
        finally:
            await db.close()

    async def list_by_owner(
        self,
        owner_kind: OwnerKind,
        owner_id: str,
    ) -> list[ConceptMapping]:
        """List mappings for a specific owner.

        Args:
            owner_kind: Type of owner (claim, passage, thought_unit)
            owner_id: Owner identifier

        Returns:
            List of concept mappings
        """
        db = await _get_db(self._db_path)
        try:
            owner_kind_val = owner_kind.value if isinstance(owner_kind, OwnerKind) else owner_kind
            async with db.execute(
                """
                SELECT * FROM concept_mappings
                WHERE project_id = ? AND owner_kind = ? AND owner_id = ?
                ORDER BY confidence DESC
                """,
                (self.project_id, owner_kind_val, owner_id)
            ) as cursor:
                rows = await cursor.fetchall()
                return [self._row_to_mapping(dict(row)) for row in rows]
        finally:
            await db.close()

    async def list_by_concept(
        self,
        concept_id: str,
        limit: int = 50,
    ) -> list[ConceptMapping]:
        """List mappings for a specific concept.

        Args:
            concept_id: Concept identifier
            limit: Maximum results

        Returns:
            List of concept mappings
        """
        db = await _get_db(self._db_path)
        try:
            async with db.execute(
                """
                SELECT * FROM concept_mappings
                WHERE project_id = ? AND concept_id = ?
                ORDER BY confidence DESC
                LIMIT ?
                """,
                (self.project_id, concept_id, limit)
            ) as cursor:
                rows = await cursor.fetchall()
                return [self._row_to_mapping(dict(row)) for row in rows]
        finally:
            await db.close()

    async def delete_by_owner(
        self,
        owner_kind: OwnerKind,
        owner_id: str,
    ) -> int:
        """Delete all mappings for an owner.

        Args:
            owner_kind: Type of owner
            owner_id: Owner identifier

        Returns:
            Number of deleted mappings
        """
        db = await _get_db(self._db_path)
        try:
            owner_kind_val = owner_kind.value if isinstance(owner_kind, OwnerKind) else owner_kind
            cursor = await db.execute(
                """
                DELETE FROM concept_mappings
                WHERE project_id = ? AND owner_kind = ? AND owner_id = ?
                """,
                (self.project_id, owner_kind_val, owner_id)
            )
            await db.commit()
            return cursor.rowcount
        finally:
            await db.close()

    @staticmethod
    def _row_to_mapping(row: dict[str, Any]) -> ConceptMapping:
        """Convert database row to ConceptMapping model."""
        return ConceptMapping(
            mapping_id=row["mapping_id"],
            project_id=row["project_id"],
            owner_kind=OwnerKind(row["owner_kind"]),
            owner_id=row["owner_id"],
            vocab_source=VocabSource(row["vocab_source"]),
            concept_id=row["concept_id"],
            preferred_label=row["preferred_label"],
            alt_labels=json.loads(row.get("alt_labels", "[]")),
            broader_terms=json.loads(row.get("broader_terms", "[]")),
            related_terms=json.loads(row.get("related_terms", "[]")),
            confidence=row.get("confidence", 0.5),
        )
