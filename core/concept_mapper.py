"""Concept mapping module for De-insight v2 Core.

This module maps claims and text to structured concepts using controlled
vocabularies (AAT or internal). It provides the foundation for hybrid
retrieval and bridge ranking.

References:
- Tech spec: deinsight_v2_codex_technical_spec.md
"""

from __future__ import annotations

import json
import re
from typing import Any, Callable

from core.schemas import (
    Claim,
    ConceptMapping,
    OwnerKind,
    VocabSource,
)
from core.stores import ConceptStore


# Prompt for concept mapping
MAP_CONCEPTS_PROMPT = """\
你是概念映射器。

任務：從文本中識別並映射相關概念。

輸入文本：
{input_text}

請回傳 JSON 格式的概念映射：

{{
  "concepts": [
    {{
      "concept_id": "概念ID（使用英文小寫，底線分隔）",
      "preferred_label": "人類可讀的標籤",
      "vocab_source": "internal 或 aat",
      "confidence": 0.0-1.0
    }}
  ]
}}

只回傳 JSON，不要加任何說明。
"""


def _clean_json(text: str) -> str:
    """Clean LLM JSON output."""
    text = text.strip()

    # Remove markdown fences
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()

    # Find first { or [
    start_idx = -1
    for i, c in enumerate(text):
        if c in "[{":
            start_idx = i
            break
    if start_idx >= 0:
        text = text[start_idx:]

    # Remove trailing comma
    text = re.sub(r',\s*([\]}])', r'\1', text)

    return text.strip()


class ConceptMapper:
    """Maps claims and text to structured concepts.

    This is the v2 replacement for inline concept extraction in ThoughtExtractor.
    It provides dedicated concept mapping with normalization and enrichment.

    Usage:
        mapper = ConceptMapper(llm_callable, project_id, concept_store)
        mappings = mapper.map_text_to_concepts(text, OwnerKind.CLAIM, claim_id)
    """

    def __init__(
        self,
        llm_callable: Callable[[str], str] | None = None,
        project_id: str = "default",
        concept_store: ConceptStore | None = None,
    ):
        """Initialize the concept mapper.

        Args:
            llm_callable: Callable that takes a prompt and returns text
            project_id: Project identifier for isolation
            concept_store: Optional ConceptStore instance
        """
        self._llm = llm_callable
        self.project_id = project_id
        self._store = concept_store

    @property
    def concept_store(self) -> ConceptStore:
        """Get or create concept store."""
        if self._store is None:
            self._store = ConceptStore(project_id=self.project_id)
        return self._store

    def map_claim_to_concepts(self, claim: Claim) -> list[ConceptMapping]:
        """Map a claim to concepts.

        Args:
            claim: Claim to map

        Returns:
            List of ConceptMapping objects
        """
        if not claim.core_claim:
            return []

        return self.map_text_to_concepts(
            text=claim.core_claim,
            owner_kind=OwnerKind.CLAIM,
            owner_id=claim.claim_id,
        )

    def map_text_to_concepts(
        self,
        text: str,
        owner_kind: OwnerKind,
        owner_id: str,
    ) -> list[ConceptMapping]:
        """Map text to concepts using LLM.

        Args:
            text: Text to map
            owner_kind: Type of owner (claim, passage, thought_unit)
            owner_id: Owner identifier

        Returns:
            List of ConceptMapping objects
        """
        # Handle empty/short text
        if not text or len(text.strip()) < 2:
            return []

        # Use LLM if available
        if self._llm is None:
            return []

        try:
            prompt = MAP_CONCEPTS_PROMPT.format(input_text=text[:2000])
            response = self._llm(prompt)
            cleaned = _clean_json(response)
            data = json.loads(cleaned)

            if not isinstance(data, dict):
                return []

            concepts_data = data.get("concepts", [])
            if not concepts_data:
                return []

            mappings = []
            for item in concepts_data:
                if not isinstance(item, dict):
                    continue
                if not item.get("concept_id"):
                    continue

                # Validate and normalize vocab_source
                vocab_source = item.get("vocab_source", "internal")
                if vocab_source not in ("internal", "aat"):
                    vocab_source = "internal"

                mapping = ConceptMapping(
                    project_id=self.project_id,
                    owner_kind=owner_kind,
                    owner_id=owner_id,
                    vocab_source=VocabSource(vocab_source),
                    concept_id=item.get("concept_id", ""),
                    preferred_label=item.get("preferred_label", item.get("concept_id", "")),
                    confidence=item.get("confidence", 0.5),
                )
                mappings.append(mapping)

            # Normalize mappings
            return self.normalize_concepts(mappings)

        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            return []

    def normalize_concepts(
        self,
        mappings: list[ConceptMapping],
    ) -> list[ConceptMapping]:
        """Normalize concepts by removing duplicates.

        Case-insensitive deduplication based on concept_id.

        Args:
            mappings: List of ConceptMapping objects

        Returns:
            Deduplicated list
        """
        if not mappings:
            return []

        seen: dict[str, ConceptMapping] = {}
        for m in mappings:
            key = m.concept_id.lower()
            if key not in seen:
                seen[key] = m
            else:
                # Keep the one with higher confidence
                if m.confidence > seen[key].confidence:
                    seen[key] = m

        return list(seen.values())

    def enrich_concept(self, mapping: ConceptMapping) -> ConceptMapping:
        """Enrich a concept mapping with additional fields.

        First-pass implementation:
        - Ensures preferred_label is set
        - Ensures mapping_id is generated

        Args:
            mapping: ConceptMapping to enrich

        Returns:
            Enriched ConceptMapping
        """
        # Ensure preferred_label
        if not mapping.preferred_label and mapping.concept_id:
            mapping.preferred_label = mapping.concept_id.replace("_", " ").title()

        # Ensure vocab_source is valid
        if not mapping.vocab_source:
            mapping.vocab_source = VocabSource.INTERNAL

        # Ensure confidence bounds
        if mapping.confidence < 0:
            mapping.confidence = 0.0
        elif mapping.confidence > 1:
            mapping.confidence = 1.0

        return mapping


async def map_text_async(
    text: str,
    owner_kind: OwnerKind,
    owner_id: str,
    llm_callable: Callable[[str], Any],
    project_id: str = "default",
    concept_store: ConceptStore | None = None,
) -> list[ConceptMapping]:
    """Convenience async function for concept mapping.

    Args:
        text: Text to map
        owner_kind: Type of owner
        owner_id: Owner identifier
        llm_callable: Async LLM callable
        project_id: Project identifier
        concept_store: Optional ConceptStore

    Returns:
        List of ConceptMapping objects
    """
    # Wrap sync callable if needed
    def sync_callable(prompt: str) -> str:
        import asyncio
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(llm_callable(prompt))

    mapper = ConceptMapper(
        llm_callable=sync_callable,
        project_id=project_id,
        concept_store=concept_store,
    )

    return mapper.map_text_to_concepts(text, owner_kind, owner_id)
