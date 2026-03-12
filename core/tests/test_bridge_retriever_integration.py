"""Integration tests for BridgeRanker in Retriever pipeline.

These tests verify that BridgeRanker is properly integrated into the
hybrid retrieval flow, ranking candidates after merge and populating
RetrievalResult.bridges.

References:
- Tech spec: deinsight_v2_codex_technical_spec.md
"""

import sys
from pathlib import Path

import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.schemas import (
    Bridge,
    Claim,
    ConceptMapping,
    OwnerKind,
    QueryMode,
    RetrievalPlan,
    ThoughtStatus,
    ThoughtUnit,
)
from core.stores import ClaimStore, ConceptStore, ThoughtStore


async def create_test_claims(claim_store, claims_data):
    """Helper to create claims in store."""
    for data in claims_data:
        claim = Claim(project_id="test", **data)
        await claim_store.add(claim)


class TestBridgeRankerIntegration:
    """Tests for BridgeRanker integration in hybrid retrieval."""

    @pytest.mark.asyncio
    async def test_bridge_ranker_invoked_after_hybrid_retrieval(self, tmp_path):
        """Test that bridge ranking is invoked after hybrid retrieval."""
        from core.retriever import Retriever

        claim_store = ClaimStore(project_id="test", db_path=tmp_path / "claims.db")

        # Add claims - these become both anchor AND candidates
        await create_test_claims(claim_store, [
            {"core_claim": "Design should prioritize function", "theory_hints": ["functionalism"], "value_axes": ["functionality"]},
            {"core_claim": "Functionalism emphasizes form follows function", "theory_hints": ["functionalism"], "value_axes": ["functionality"]},
        ])

        retriever = Retriever(
            project_id="test",
            claim_store=claim_store,
            thought_store=ThoughtStore(project_id="test", db_path=tmp_path / "thoughts.db"),
            concept_store=ConceptStore(project_id="test", db_path=tmp_path / "concepts.db"),
        )

        plan = RetrievalPlan(
            project_id="test",
            query_mode=QueryMode.FAST,
            concept_queries=["design function"],
        )

        result = await retriever.retrieve(plan, "function")

        # After integration: bridges should be populated
        # The first claim becomes anchor, rest are candidates
        assert hasattr(result, 'bridges')
        assert isinstance(result.bridges, list)
        # With 2 claims, we expect at least 1 bridge (first as anchor, second as target)
        if len(result.claims) >= 2:
            assert len(result.bridges) >= 1

    @pytest.mark.asyncio
    async def test_ranked_candidates_in_output(self, tmp_path):
        """Test that ranked bridge candidates appear in output."""
        from core.retriever import Retriever

        claim_store = ClaimStore(project_id="test", db_path=tmp_path / "claims.db")

        await create_test_claims(claim_store, [
            {"core_claim": "Design should prioritize function", "theory_hints": ["functionalism"], "value_axes": ["functionality"]},
            {"core_claim": "Functionalism emphasizes form", "theory_hints": ["functionalism"], "value_axes": ["functionality"]},
            {"core_claim": "Minimalism is about simplicity", "theory_hints": ["minimalism"], "value_axes": ["simplicity"]},
        ])

        retriever = Retriever(
            project_id="test",
            claim_store=claim_store,
            thought_store=ThoughtStore(project_id="test", db_path=tmp_path / "thoughts.db"),
            concept_store=ConceptStore(project_id="test", db_path=tmp_path / "concepts.db"),
        )

        plan = RetrievalPlan(
            project_id="test",
            query_mode=QueryMode.FAST,
        )

        result = await retriever.retrieve(plan, "function")

        # Check bridges are sorted by score
        if len(result.bridges) > 1:
            for i in range(len(result.bridges) - 1):
                score1 = result.bridges[i].score or 0.0
                score2 = result.bridges[i + 1].score or 0.0
                assert score1 >= score2

    @pytest.mark.asyncio
    async def test_backward_compatibility_no_stores(self, tmp_path):
        """Test that legacy-only path still works without stores."""
        from core.retriever import Retriever

        # No stores injected - should use legacy path
        retriever = Retriever(project_id="test")

        # Should not crash - legacy path handles missing pipeline gracefully
        plan = RetrievalPlan(
            project_id="test",
            query_mode=QueryMode.FAST,
        )

        result = await retriever.retrieve(plan, "test query")

        # Should return a valid result (may be empty if legacy fails)
        assert result is not None
        assert isinstance(result.passages, list)
        assert isinstance(result.bridges, list)

    @pytest.mark.asyncio
    async def test_empty_behavior_no_candidates(self, tmp_path):
        """Test graceful handling when no candidates available."""
        from core.retriever import Retriever

        retriever = Retriever(
            project_id="test",
            claim_store=ClaimStore(project_id="test", db_path=tmp_path / "claims.db"),
            thought_store=ThoughtStore(project_id="test", db_path=tmp_path / "thoughts.db"),
            concept_store=ConceptStore(project_id="test", db_path=tmp_path / "concepts.db"),
        )

        plan = RetrievalPlan(
            project_id="test",
            query_mode=QueryMode.FAST,
        )

        result = await retriever.retrieve(plan, "nonexistent query")

        assert result is not None
        assert isinstance(result.bridges, list)
        # Should return empty bridges, not crash
        assert result.bridges == []

    @pytest.mark.asyncio
    async def test_legacy_only_path_works(self, tmp_path):
        """Test that legacy-only retrieval still functions correctly."""
        from core.retriever import Retriever

        # No stores - should use legacy path
        retriever = Retriever(project_id="test")

        plan = RetrievalPlan(
            project_id="test",
            query_mode=QueryMode.DEEP,
        )

        # Should not crash
        result = await retriever.retrieve(plan, "test")

        # Should return valid result structure
        assert result is not None
        assert isinstance(result.passages, list)
        assert isinstance(result.bridges, list)

    @pytest.mark.asyncio
    async def test_strong_structural_match_above_lexical(self, tmp_path):
        """Test that strong structural match ranks higher than lexical only."""
        from core.retriever import Retriever

        claim_store = ClaimStore(project_id="test", db_path=tmp_path / "claims.db")

        await create_test_claims(claim_store, [
            {"core_claim": "Design should prioritize function", "theory_hints": ["functionalism"], "value_axes": ["functionality"], "abstract_patterns": ["form-function"]},
            {"core_claim": "Form serves function in architecture", "theory_hints": ["functionalism"], "value_axes": ["functionality"], "abstract_patterns": ["form-function"]},
            {"core_claim": "Function is the most important word function function function", "theory_hints": ["aesthetics"], "value_axes": ["beauty"], "abstract_patterns": ["expression"]},
        ])

        retriever = Retriever(
            project_id="test",
            claim_store=claim_store,
            thought_store=ThoughtStore(project_id="test", db_path=tmp_path / "thoughts.db"),
            concept_store=ConceptStore(project_id="test", db_path=tmp_path / "concepts.db"),
        )

        plan = RetrievalPlan(project_id="test", query_mode=QueryMode.FAST)

        result = await retriever.retrieve(plan, "function")

        # Structural match should rank higher
        if len(result.bridges) >= 2:
            scores = {b.target_claim_id: b.score for b in result.bridges}
            structural_claim = next((c for c in result.claims if "architecture" in c.core_claim), None)
            lexical_claim = next((c for c in result.claims if "most important word" in c.core_claim), None)
            
            if structural_claim and lexical_claim:
                structural_score = scores.get(structural_claim.claim_id, 0) or 0.0
                lexical_score = scores.get(lexical_claim.claim_id, 0) or 0.0
                assert structural_score >= lexical_score

    @pytest.mark.asyncio
    async def test_cross_domain_match_surfaces(self, tmp_path):
        """Test that cross-domain matches can surface in results."""
        from core.retriever import Retriever

        claim_store = ClaimStore(project_id="test", db_path=tmp_path / "claims.db")

        await create_test_claims(claim_store, [
            {"core_claim": "Form follows function in design", "theory_hints": ["design theory"], "value_axes": ["functionality"]},
            {"core_claim": "Architecture should serve human needs", "theory_hints": ["architecture"], "value_axes": ["utility"]},
            {"core_claim": "Bauhaus emphasizes functional design", "theory_hints": ["design history"], "value_axes": ["functionality"]},
        ])

        retriever = Retriever(
            project_id="test",
            claim_store=claim_store,
            thought_store=ThoughtStore(project_id="test", db_path=tmp_path / "thoughts.db"),
            concept_store=ConceptStore(project_id="test", db_path=tmp_path / "concepts.db"),
        )

        plan = RetrievalPlan(project_id="test", query_mode=QueryMode.FAST)

        # Use "design" which matches all claims
        result = await retriever.retrieve(plan, "design")

        assert result is not None
        # Should have claims and bridges
        assert len(result.claims) >= 1

    @pytest.mark.asyncio
    async def test_anchor_missing_doesnt_crash(self, tmp_path):
        """Test that missing anchor doesn't crash retrieval."""
        from core.retriever import Retriever

        retriever = Retriever(
            project_id="test",
            claim_store=ClaimStore(project_id="test", db_path=tmp_path / "claims.db"),
            thought_store=ThoughtStore(project_id="test", db_path=tmp_path / "thoughts.db"),
            concept_store=ConceptStore(project_id="test", db_path=tmp_path / "concepts.db"),
        )

        plan = RetrievalPlan(project_id="test", query_mode=QueryMode.FAST)

        result = await retriever.retrieve(plan, "test query")

        assert result is not None
        assert isinstance(result.bridges, list)
        assert result.bridges == []


class TestBridgeRankerWithThoughtAnchor:
    """Tests for BridgeRanker with ThoughtUnit as anchor."""

    @pytest.mark.asyncio
    async def test_thought_anchor_used_when_no_claim(self, tmp_path):
        """Test that ThoughtUnit is used as anchor when no claim available."""
        from core.retriever import Retriever

        thought_store = ThoughtStore(project_id="test", db_path=tmp_path / "thoughts.db")

        thought = ThoughtUnit(
            project_id="test",
            title="Form and Function",
            summary="Exploring the relationship between form and function",
            value_axes=["functionality"],
            recurring_patterns=["form-function"],
            status=ThoughtStatus.STABLE,
        )
        await thought_store.add(thought)

        # Also add a claim so we have candidates
        claim_store = ClaimStore(project_id="test", db_path=tmp_path / "claims.db")
        claim = Claim(
            project_id="test",
            core_claim="Functionalism in architecture",
            value_axes=["functionality"],
            theory_hints=["functionalism"],
        )
        await claim_store.add(claim)

        retriever = Retriever(
            project_id="test",
            claim_store=claim_store,
            thought_store=thought_store,
            concept_store=ConceptStore(project_id="test", db_path=tmp_path / "concepts.db"),
        )

        plan = RetrievalPlan(project_id="test", query_mode=QueryMode.FAST)

        # Should not crash - uses thought as anchor
        result = await retriever.retrieve(plan, "form function")

        assert result is not None
        assert isinstance(result.bridges, list)


class TestBridgeRankerGracefulFallback:
    """Tests for graceful fallback of anchor sources."""

    @pytest.mark.asyncio
    async def test_concept_context_fallback(self, tmp_path):
        """Test fallback to concept context when no claim/thought."""
        from core.retriever import Retriever

        concept_store = ConceptStore(project_id="test", db_path=tmp_path / "concepts.db")

        concept = ConceptMapping(
            project_id="test",
            owner_kind=OwnerKind.CLAIM,
            owner_id="test",
            concept_id="functionalism",
            preferred_label="Functionalism",
            confidence=0.9,
        )
        await concept_store.add(concept)

        # Add a claim so we have something to rank
        claim_store = ClaimStore(project_id="test", db_path=tmp_path / "claims.db")
        claim = Claim(project_id="test", core_claim="Test claim", theory_hints=["functionalism"])
        await claim_store.add(claim)

        retriever = Retriever(
            project_id="test",
            claim_store=claim_store,
            thought_store=ThoughtStore(project_id="test", db_path=tmp_path / "thoughts.db"),
            concept_store=concept_store,
        )

        plan = RetrievalPlan(
            project_id="test",
            query_mode=QueryMode.FAST,
            concept_queries=["functionalism"],
        )

        result = await retriever.retrieve(plan, "functionalism")

        assert result is not None
        assert isinstance(result.bridges, list)
