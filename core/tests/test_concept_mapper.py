"""Tests for ConceptMapper module.

References:
- Tech spec: deinsight_v2_codex_technical_spec.md
"""

import sys
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.schemas import Claim, ConceptMapping, OwnerKind, VocabSource
from core.stores import ConceptStore


class TestConceptMapperBasic:
    """Basic mapping tests."""

    @pytest.fixture
    def concept_store(self, tmp_path):
        """Create a ConceptStore for testing."""
        return ConceptStore(project_id="test", db_path=tmp_path / "concepts.db")

    def test_map_text_to_concepts_basic(self, concept_store):
        """Test basic text to concept mapping."""
        from core.concept_mapper import ConceptMapper

        def mock_llm(prompt: str) -> str:
            return '{"concepts": [{"concept_id": "minimalism", "preferred_label": "Minimalism", "vocab_source": "internal", "confidence": 0.9}]}'

        mapper = ConceptMapper(
            llm_callable=mock_llm,
            project_id="test",
            concept_store=concept_store,
        )

        result = mapper.map_text_to_concepts(
            text="我喜歡極簡主義的設計",
            owner_kind=OwnerKind.CLAIM,
            owner_id="test-claim-1",
        )

        assert result is not None
        assert len(result) > 0
        assert result[0].preferred_label == "Minimalism"

    def test_map_claim_to_concepts(self, concept_store):
        """Test claim to concept mapping."""
        from core.concept_mapper import ConceptMapper

        def mock_llm(prompt: str) -> str:
            return '{"concepts": [{"concept_id": "function_form", "preferred_label": "Function-Form Relationship", "vocab_source": "internal", "confidence": 0.85}]}'

        mapper = ConceptMapper(
            llm_callable=mock_llm,
            project_id="test",
            concept_store=concept_store,
        )

        claim = Claim(
            project_id="test",
            core_claim="Form follows function",
            value_axes=["functionalism"],
        )

        result = mapper.map_claim_to_concepts(claim)

        assert result is not None
        assert len(result) > 0


class TestConceptMapperEdgeCases:
    """Edge case handling tests."""

    @pytest.fixture
    def concept_store(self, tmp_path):
        return ConceptStore(project_id="test", db_path=tmp_path / "concepts.db")

    def test_empty_text_handling(self, concept_store):
        """Test handling of empty text input."""
        from core.concept_mapper import ConceptMapper

        mapper = ConceptMapper(
            llm_callable=lambda x: "",
            project_id="test",
            concept_store=concept_store,
        )

        result = mapper.map_text_to_concepts(
            text="",
            owner_kind=OwnerKind.CLAIM,
            owner_id="test",
        )

        # Should return empty list, not crash
        assert result == []

    def test_short_text_handling(self, concept_store):
        """Test handling of very short text."""
        from core.concept_mapper import ConceptMapper

        mapper = ConceptMapper(
            llm_callable=lambda x: "",
            project_id="test",
            concept_store=concept_store,
        )

        # Short text (less than 10 chars) should be handled gracefully
        result = mapper.map_text_to_concepts(
            text="hi",
            owner_kind=OwnerKind.CLAIM,
            owner_id="test",
        )

        assert isinstance(result, list)

    def test_malformed_llm_output(self, concept_store):
        """Test handling of malformed LLM output."""
        from core.concept_mapper import ConceptMapper

        # Malformed JSON
        def mock_malformed(prompt: str) -> str:
            return "這不是JSON格式"

        mapper = ConceptMapper(
            llm_callable=mock_malformed,
            project_id="test",
            concept_store=concept_store,
        )

        result = mapper.map_text_to_concepts(
            text="測試文字",
            owner_kind=OwnerKind.CLAIM,
            owner_id="test",
        )

        # Should return empty list, not crash
        assert result == []

    def test_invalid_json_output(self, concept_store):
        """Test handling of invalid JSON in output."""
        from core.concept_mapper import ConceptMapper

        def mock_invalid_json(prompt: str) -> str:
            return '{"concepts": [{"concept_id": '  # Truncated

        mapper = ConceptMapper(
            llm_callable=mock_invalid_json,
            project_id="test",
            concept_store=concept_store,
        )

        result = mapper.map_text_to_concepts(
            text="測試文字",
            owner_kind=OwnerKind.CLAIM,
            owner_id="test",
        )

        # Should return empty list, not crash
        assert result == []


class TestConceptMapperNormalization:
    """Concept normalization tests."""

    @pytest.fixture
    def concept_store(self, tmp_path):
        return ConceptStore(project_id="test", db_path=tmp_path / "concepts.db")

    def test_duplicate_normalization(self, concept_store):
        """Test that duplicates are normalized."""
        from core.concept_mapper import ConceptMapper

        mappings = [
            ConceptMapping(
                project_id="test",
                owner_kind=OwnerKind.CLAIM,
                owner_id="test-1",
                concept_id="minimalism",
                preferred_label="Minimalism",
            ),
            ConceptMapping(
                project_id="test",
                owner_kind=OwnerKind.CLAIM,
                owner_id="test-2",
                concept_id="MINIMALISM",  # Same but uppercase
                preferred_label="minimalism",  # Same but different case
            ),
        ]

        mapper = ConceptMapper(
            llm_callable=lambda x: "",
            project_id="test",
            concept_store=concept_store,
        )

        normalized = mapper.normalize_concepts(mappings)

        # Should reduce duplicates
        assert len(normalized) <= len(mappings)

    def test_case_insensitive_deduplication(self, concept_store):
        """Test case-insensitive deduplication."""
        from core.concept_mapper import ConceptMapper

        mappings = [
            ConceptMapping(
                project_id="test",
                concept_id="design",
                preferred_label="Design",
            ),
            ConceptMapping(
                project_id="test",
                concept_id="DESIGN",
                preferred_label="design",
            ),
        ]

        mapper = ConceptMapper(
            llm_callable=lambda x: "",
            project_id="test",
            concept_store=concept_store,
        )

        normalized = mapper.normalize_concepts(mappings)

        # Should deduplicate
        concept_ids = [m.concept_id.lower() for m in normalized]
        assert len(set(concept_ids)) == len(concept_ids)


class TestConceptMapperEnrichment:
    """Concept enrichment tests."""

    @pytest.fixture
    def concept_store(self, tmp_path):
        return ConceptStore(project_id="test", db_path=tmp_path / "concepts.db")

    def test_enrichment_field_completeness(self, concept_store):
        """Test that enrichment adds expected fields."""
        from core.concept_mapper import ConceptMapper

        mapper = ConceptMapper(
            llm_callable=lambda x: "",
            project_id="test",
            concept_store=concept_store,
        )

        mapping = ConceptMapping(
            project_id="test",
            concept_id="minimalism",
            preferred_label="Minimalism",
            vocab_source=VocabSource.INTERNAL,
        )

        enriched = mapper.enrich_concept(mapping)

        # Should have additional fields filled
        assert enriched.preferred_label != ""  # Should be preserved or enhanced
        assert enriched.mapping_id != ""  # Should have ID

    def test_enrichment_preserves_original(self, concept_store):
        """Test that enrichment preserves original data."""
        from core.concept_mapper import ConceptMapper

        mapper = ConceptMapper(
            llm_callable=lambda x: "",
            project_id="test",
            concept_store=concept_store,
        )

        original = ConceptMapping(
            project_id="test",
            concept_id="test-id",
            preferred_label="Test Label",
            owner_kind=OwnerKind.CLAIM,
            owner_id="owner-1",
            confidence=0.9,
        )

        enriched = mapper.enrich_concept(original)

        # Core fields should be preserved
        assert enriched.concept_id == original.concept_id
        assert enriched.owner_id == original.owner_id


class TestConceptStoreRoundTrip:
    """ConceptStore round-trip tests."""

    @pytest.mark.asyncio
    async def test_mapper_to_store_roundtrip(self, tmp_path):
        """Test that mapper output can be written to store and read back."""
        from core.concept_mapper import ConceptMapper

        store = ConceptStore(project_id="test", db_path=tmp_path / "roundtrip.db")

        def mock_llm(prompt: str) -> str:
            return '{"concepts": [{"concept_id": "roundtrip_test", "preferred_label": "Roundtrip Test", "vocab_source": "internal", "confidence": 0.95}]}'

        mapper = ConceptMapper(
            llm_callable=mock_llm,
            project_id="test",
            concept_store=store,
        )

        # Map text to concepts
        mappings = mapper.map_text_to_concepts(
            text="Testing roundtrip",
            owner_kind=OwnerKind.CLAIM,
            owner_id="test-owner",
        )

        assert len(mappings) > 0
        mapping = mappings[0]

        # Write to store
        await store.add(mapping)

        # Read from store
        fetched = await store.get(mapping.mapping_id)

        assert fetched is not None
        assert fetched.concept_id == mapping.concept_id
        assert fetched.preferred_label == mapping.preferred_label
        assert fetched.confidence == mapping.confidence

    @pytest.mark.asyncio
    async def test_multiple_mappings_roundtrip(self, tmp_path):
        """Test roundtrip with multiple mappings."""
        from core.concept_mapper import ConceptMapper

        store = ConceptStore(project_id="test", db_path=tmp_path / "multi.db")

        def mock_llm(prompt: str) -> str:
            return '{"concepts": [{"concept_id": "concept1", "preferred_label": "Concept One", "vocab_source": "internal", "confidence": 0.8}, {"concept_id": "concept2", "preferred_label": "Concept Two", "vocab_source": "internal", "confidence": 0.7}]}'

        mapper = ConceptMapper(
            llm_callable=mock_llm,
            project_id="test",
            concept_store=store,
        )

        mappings = mapper.map_text_to_concepts(
            text="Multiple concepts",
            owner_kind=OwnerKind.CLAIM,
            owner_id="test",
        )

        assert len(mappings) >= 2

        # Write all to store
        for m in mappings:
            await store.add(m)

        # Read all back
        all_mappings = await store.list_by_owner(OwnerKind.CLAIM, "test")

        assert len(all_mappings) >= 2
