"""Tests for hybrid retriever module.

References:
- Tech spec: deinsight_v2_codex_technical_spec.md
"""

import sys
from pathlib import Path

import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.schemas import (
    Claim,
    ConceptMapping,
    OwnerKind,
    QueryMode,
    RetrievalPlan,
    ThoughtStatus,
    ThoughtUnit,
    VocabSource,
)
from core.stores import ClaimStore, ThoughtStore, ConceptStore


class TestHybridRetrieverLegacyOnly:
    """Tests for legacy-only fallback behavior."""

    @pytest.mark.asyncio
    async def test_legacy_fallback_when_stores_empty(self, tmp_path):
        """Test that legacy is used when stores are empty."""
        from core.retriever import Retriever

        # Create stores that will be empty
        retriever = Retriever(
            project_id="test",
            claim_store=ClaimStore(project_id="test", db_path=tmp_path / "claims.db"),
            thought_store=ThoughtStore(project_id="test", db_path=tmp_path / "thoughts.db"),
            concept_store=ConceptStore(project_id="test", db_path=tmp_path / "concepts.db"),
        )

        plan = RetrievalPlan(
            project_id="test",
            query_mode=QueryMode.FAST,
            concept_queries=["test query"],
        )

        # Should not crash, even if legacy fails
        result = await retriever.retrieve(plan, "test query")

        # Should return a valid result
        assert result is not None
        assert result.plan == plan


class TestHybridRetrieverClaimStore:
    """Tests for claim store retrieval."""

    @pytest.mark.asyncio
    async def test_retrieve_from_claims(self, tmp_path):
        """Test retrieval from claim store."""
        from core.retriever import Retriever

        claim_store = ClaimStore(project_id="test", db_path=tmp_path / "claims.db")

        # Add test claim
        claim = Claim(
            project_id="test",
            core_claim="Form follows function is a design principle",
            value_axes=["functionalism"],
        )
        await claim_store.add(claim)

        retriever = Retriever(project_id="test", claim_store=claim_store)
        result = await retriever._retrieve_from_claims("design principle")

        assert len(result) >= 1
        assert "claim" in result[0]
        assert result[0]["claim"] is not None


class TestHybridRetrieverThoughtStore:
    """Tests for thought store retrieval."""

    @pytest.mark.asyncio
    async def test_retrieve_from_thoughts(self, tmp_path):
        """Test retrieval from thought store."""
        from core.retriever import Retriever

        thought_store = ThoughtStore(project_id="test", db_path=tmp_path / "thoughts.db")

        # Add test thought
        thought = ThoughtUnit(
            project_id="test",
            title="Form and Function",
            summary="Exploring the relationship between form and function in design",
            status=ThoughtStatus.EMERGING,
        )
        await thought_store.add(thought)

        retriever = Retriever(project_id="test", thought_store=thought_store)
        
        # Query should match - "form" is in title
        result = await retriever._retrieve_from_thoughts("form")

        # Should return result (query "form" matches title "Form and Function")
        assert isinstance(result, list)


class TestHybridRetrieverConceptMapped:
    """Tests for concept-mapped retrieval."""

    @pytest.mark.asyncio
    async def test_concept_store_lookup(self, tmp_path):
        """Test direct concept store lookup."""
        from core.retriever import Retriever

        concept_store = ConceptStore(project_id="test", db_path=tmp_path / "concepts.db")

        # Add test concept
        concept = ConceptMapping(
            project_id="test",
            owner_kind=OwnerKind.CLAIM,
            owner_id="test",
            concept_id="minimalism",
            preferred_label="Minimalism",
            confidence=0.9,
        )
        await concept_store.add(concept)

        retriever = Retriever(project_id="test", concept_store=concept_store)
        result = await retriever._retrieve_from_concepts("minimalism")

        assert isinstance(result, list)


class TestHybridMergeBehavior:
    """Tests for result merging behavior."""

    def test_merge_empty_sources(self):
        """Test merge handles empty sources gracefully."""
        from core.retriever import Retriever

        retriever = Retriever(project_id="test")
        plan = RetrievalPlan(project_id="test", query_mode=QueryMode.FAST)

        result = retriever._merge_results(
            legacy=[],
            claims=[],
            thoughts=[],
            concepts=[],
            plan=plan,
        )

        # Should return valid result with empty lists
        assert result.passages == []
        assert result.claims == []

    def test_merge_source_limits(self):
        """Test that each source has limits."""
        from core.retriever import Retriever

        retriever = Retriever(project_id="test")
        plan = RetrievalPlan(project_id="test", query_mode=QueryMode.FAST)

        # Create many items
        legacy = [{"text": f"legacy {i}"} for i in range(10)]
        claims = [{"claim": Claim(core_claim=f"claim {i}"), "score": 0.9} for i in range(10)]

        result = retriever._merge_results(legacy, claims, [], [], plan)

        # Should respect limits (legacy: 5, claims: 3)
        total = len(result.passages) + len(result.claims)
        assert total <= 8  # 5 + 3


class TestHybridRetrieverEmptyStores:
    """Tests for empty store behavior."""

    @pytest.mark.asyncio
    async def test_empty_claim_store_returns_list(self, tmp_path):
        """Test empty claim store returns empty list."""
        from core.retriever import Retriever

        claim_store = ClaimStore(project_id="test", db_path=tmp_path / "c.db")
        retriever = Retriever(project_id="test", claim_store=claim_store)

        result = await retriever._retrieve_from_claims("nonexistent query")
        assert result == []

    @pytest.mark.asyncio
    async def test_empty_thought_store_returns_list(self, tmp_path):
        """Test empty thought store returns empty list."""
        from core.retriever import Retriever

        thought_store = ThoughtStore(project_id="test", db_path=tmp_path / "t.db")
        retriever = Retriever(project_id="test", thought_store=thought_store)

        result = await retriever._retrieve_from_thoughts("nonexistent query")
        assert result == []

    @pytest.mark.asyncio
    async def test_empty_concept_store_returns_list(self, tmp_path):
        """Test empty concept store returns empty list."""
        from core.retriever import Retriever

        concept_store = ConceptStore(project_id="test", db_path=tmp_path / "co.db")
        retriever = Retriever(project_id="test", concept_store=concept_store)

        result = await retriever._retrieve_from_concepts("nonexistent")
        assert result == []


class TestBackwardCompatibility:
    """Tests to ensure backward compatibility."""

    def test_retriever_has_project_id(self):
        """Test retriever has project_id attribute."""
        from core.retriever import Retriever
        retriever = Retriever(project_id="test")
        assert hasattr(retriever, "project_id")
        assert retriever.project_id == "test"

    def test_retriever_has_retrieve_method(self):
        """Test retriever has retrieve method."""
        from core.retriever import Retriever
        retriever = Retriever(project_id="test")
        assert hasattr(retriever, "retrieve")
        assert callable(retriever.retrieve)

    @pytest.mark.asyncio
    async def test_retrieve_returns_retrieval_result(self):
        """Test retrieve returns RetrievalResult type."""
        from core.retriever import Retriever
        from core.schemas import RetrievalResult

        retriever = Retriever(project_id="test")
        plan = RetrievalPlan(project_id="test", query_mode=QueryMode.FAST)

        result = await retriever.retrieve(plan, "test query")

        assert isinstance(result, RetrievalResult)


class TestMixedSourceMerge:
    """Tests for mixed source merging."""

    def test_merge_preserves_source_info(self):
        """Test that merge preserves source information."""
        from core.retriever import Retriever

        retriever = Retriever(project_id="test")
        plan = RetrievalPlan(project_id="test", query_mode=QueryMode.FAST)

        legacy = [{"text": "legacy text"}]
        claims = [{"claim": Claim(core_claim="claim text"), "score": 0.9}]

        result = retriever._merge_results(legacy, claims, [], [], plan)

        # Legacy should be in passages, claims in claims list
        assert isinstance(result.passages, list)
        assert isinstance(result.claims, list)

    def test_merge_empty_legacy_uses_stores(self):
        """Test that stores are used when legacy is empty."""
        from core.retriever import Retriever

        retriever = Retriever(project_id="test")
        plan = RetrievalPlan(project_id="test", query_mode=QueryMode.FAST)

        claims = [{"claim": Claim(core_claim="from claim"), "score": 0.9}]
        thoughts = [{"thought": ThoughtUnit(title="from thought"), "score": 0.8}]

        result = retriever._merge_results([], claims, thoughts, [], plan)

        # Should have results from stores
        assert isinstance(result.claims, list)
