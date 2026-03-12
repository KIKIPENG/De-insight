"""Tests for bridge ranker module.

References:
- Tech spec: deinsight_v2_codex_technical_spec.md
"""

import sys
from pathlib import Path

import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.schemas import Bridge, BridgeType, Claim, ConceptMapping, OwnerKind, ThoughtStatus, ThoughtUnit, VocabSource


class TestBridgeRankerConceptOverlap:
    """Tests for concept overlap ranking."""

    def test_concept_overlap_ranking(self):
        """Test that higher concept overlap leads to higher ranking."""
        from core.bridge_ranker import BridgeRanker

        ranker = BridgeRanker()

        # Anchor with concepts
        anchor = Claim(
            project_id="test",
            core_claim="Design should prioritize function over form",
            theory_hints=["functionalism", "modernism"],
        )

        # Candidate 1: high concept overlap
        candidate1 = Claim(
            project_id="test",
            core_claim="Functionalism emphasizes that form follows function",
            theory_hints=["functionalism"],
        )

        # Candidate 2: low concept overlap
        candidate2 = Claim(
            project_id="test",
            core_claim="Minimalism is about visual simplicity",
            theory_hints=["minimalism"],
        )

        results = ranker.rank_candidates(anchor, [candidate1, candidate2])

        # Candidate 1 should rank higher due to concept overlap
        assert len(results) == 2
        assert results[0].candidate_id == candidate1.claim_id
        assert results[0].score >= results[1].score


class TestBridgeRankerValueAxisOverlap:
    """Tests for value-axis overlap ranking."""

    def test_value_axis_overlap_ranking(self):
        """Test that value-axis overlap increases ranking."""
        from core.bridge_ranker import BridgeRanker

        ranker = BridgeRanker()

        anchor = Claim(
            project_id="test",
            core_claim="Design should prioritize function",
            value_axes=["functionality", "practicality"],
        )

        # High value-axis overlap
        candidate1 = Claim(
            project_id="test",
            core_claim="Functionality is the primary value",
            value_axes=["functionality", "efficiency"],
        )

        # No value-axis overlap
        candidate2 = Claim(
            project_id="test",
            core_claim="Aesthetics matter most",
            value_axes=["beauty", "form"],
        )

        results = ranker.rank_candidates(anchor, [candidate1, candidate2])

        assert results[0].candidate_id == candidate1.claim_id
        assert results[0].score >= results[1].score


class TestBridgeRankerPatternOverlap:
    """Tests for abstract-pattern overlap ranking."""

    def test_pattern_overlap_ranking(self):
        """Test that abstract-pattern overlap increases ranking."""
        from core.bridge_ranker import BridgeRanker

        ranker = BridgeRanker()

        anchor = Claim(
            project_id="test",
            core_claim="Form follows function",
            abstract_patterns=["form-function", "causation"],
        )

        # High pattern overlap
        candidate1 = Claim(
            project_id="test",
            core_claim="Structure determines behavior",
            abstract_patterns=["form-function", "system"],
        )

        # No pattern overlap
        candidate2 = Claim(
            project_id="test",
            core_claim="Color is expressive",
            abstract_patterns=["expression", "emotion"],
        )

        results = ranker.rank_candidates(anchor, [candidate1, candidate2])

        assert results[0].candidate_id == candidate1.claim_id


class TestBridgeRankerCrossDomainBonus:
    """Tests for cross-domain bonus."""

    def test_cross_domain_bonus(self):
        """Test that cross-domain matches get a bonus."""
        from core.bridge_ranker import BridgeRanker

        ranker = BridgeRanker()

        # Anchor: design domain
        anchor = Claim(
            project_id="test",
            core_claim="Form follows function in design",
            theory_hints=["design theory"],
        )

        # Same domain
        candidate1 = Claim(
            project_id="test",
            core_claim="Bauhaus design principles",
            theory_hints=["design history"],
        )

        # Different domain (architecture)
        candidate2 = Claim(
            project_id="test",
            core_claim="Architecture should serve function",
            theory_hints=["architecture", "functionalism"],
        )

        results = ranker.rank_candidates(anchor, [candidate1, candidate2])

        # Cross-domain should get bonus
        assert results[0].score > 0


class TestBridgeRankerLexicalWeakMatch:
    """Tests for lexical-only weak matches."""

    def test_lexical_only_not_dominate(self):
        """Test that pure lexical match doesn't dominate conceptual matches."""
        from core.bridge_ranker import BridgeRanker

        ranker = BridgeRanker()

        anchor = Claim(
            project_id="test",
            core_claim="Design should prioritize function",
            value_axes=["functionality"],
        )

        # High lexical overlap but no conceptual match
        candidate1 = Claim(
            project_id="test",
            core_claim="Function is the most important word in this sentence about design",
            value_axes=["aesthetics"],
        )

        # Moderate lexical but conceptual match
        candidate2 = Claim(
            project_id="test",
            core_claim="Form serves function",
            value_axes=["functionality"],
        )

        results = ranker.rank_candidates(anchor, [candidate1, candidate2])

        # Conceptual match should rank higher
        assert results[0].candidate_id == candidate2.claim_id


class TestBridgeRankerEmptyInput:
    """Tests for empty input handling."""

    def test_empty_candidates(self):
        """Test that empty candidates returns empty list."""
        from core.bridge_ranker import BridgeRanker

        ranker = BridgeRanker()

        anchor = Claim(
            project_id="test",
            core_claim="Test claim",
        )

        results = ranker.rank_candidates(anchor, [])

        assert results == []

    def test_none_anchor(self):
        """Test that None anchor returns empty list."""
        from core.bridge_ranker import BridgeRanker

        ranker = BridgeRanker()

        candidate = Claim(project_id="test", core_claim="Test")
        results = ranker.rank_candidates(None, [candidate])

        # Should handle gracefully
        assert isinstance(results, list)


class TestBridgeRankerMixedCandidates:
    """Tests for mixed candidate ranking scenarios."""

    def test_mixed_candidate_ranking(self):
        """Test ranking with multiple signal types."""
        from core.bridge_ranker import BridgeRanker

        ranker = BridgeRanker()

        anchor = Claim(
            project_id="test",
            core_claim="Minimalism emphasizes simplicity",
            value_axes=["simplicity", "restraint"],
            abstract_patterns=["minimal-form"],
            theory_hints=["minimalism", "modernism"],
        )

        candidates = [
            # High multi-signal match
            Claim(
                project_id="test",
                core_claim="Simple forms define minimalist design",
                value_axes=["simplicity", "clarity"],
                abstract_patterns=["minimal-form", "reduction"],
                theory_hints=["minimalism"],
            ),
            # Concept only
            Claim(
                project_id="test",
                core_claim="Modernism changed everything",
                theory_hints=["modernism"],
            ),
            # Lexical only
            Claim(
                project_id="test",
                core_claim="The word minimal appears here many times minimal minimal",
            ),
            # Value only
            Claim(
                project_id="test",
                core_claim="Restraint in architecture",
                value_axes=["restraint", "space"],
            ),
        ]

        results = ranker.rank_candidates(anchor, candidates)

        assert len(results) == 4
        # First should be multi-signal match
        assert results[0].candidate_id == candidates[0].claim_id
        # All should have positive scores
        for r in results:
            assert r.score >= 0

    def test_score_breakdown_visible(self):
        """Test that score breakdown is visible."""
        from core.bridge_ranker import BridgeRanker

        ranker = BridgeRanker()

        anchor = Claim(
            project_id="test",
            core_claim="Test",
            value_axes=["value1"],
            theory_hints=["theory1"],
        )

        candidate = Claim(
            project_id="test",
            core_claim="Test",
            value_axes=["value1"],
            theory_hints=["theory1"],
        )

        results = ranker.rank_candidates(anchor, [candidate])

        assert len(results) > 0
        # Check that breakdown exists
        result = results[0]
        assert hasattr(result, 'score_breakdown')
        assert isinstance(result.score_breakdown, dict)
