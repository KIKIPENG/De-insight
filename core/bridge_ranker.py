"""Bridge ranking module for De-insight v2 Core.

This module ranks retrieved candidates based on bridge-worthiness signals.
It identifies which candidates are most likely to be useful conceptual bridges
rather than merely textually relevant items.

References:
- Tech spec: deinsight_v2_codex_technical_spec.md
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from core.schemas import Claim, ConceptMapping, ThoughtUnit


# Score weights
WEIGHT_CONCEPT = 0.25
WEIGHT_VALUE_AXES = 0.25
WEIGHT_PATTERN = 0.20
WEIGHT_CRITIQUE = 0.15
WEIGHT_CROSS_DOMAIN = 0.15


@dataclass
class RankedBridgeCandidate:
    """A candidate with bridge-worthiness ranking.

    Attributes:
        candidate_id: Unique identifier of the candidate
        candidate: The original candidate object (Claim or ThoughtUnit)
        score: Overall bridge-worthiness score (0.0-1.0)
        score_breakdown: Detailed score breakdown by signal type
        reason: Human-readable explanation of why this candidate ranks high
    """

    candidate_id: str
    candidate: Any
    score: float
    score_breakdown: dict[str, float]
    reason: str = ""


class BridgeRanker:
    """Ranks candidates based on bridge-worthiness signals.

    This is a first-pass implementation that scores candidates based on:
    - Concept overlap (theory_hints)
    - Value-axis overlap
    - Abstract-pattern overlap
    - Critique-target overlap
    - Cross-domain novelty bonus

    The ranker prefers structurally meaningful matches over lexical overlap.
    """

    def rank_candidates(
        self,
        anchor: Claim | ThoughtUnit | ConceptMapping | None,
        candidates: list[Claim | ThoughtUnit],
    ) -> list[RankedBridgeCandidate]:
        """Rank candidates by bridge-worthiness.

        Args:
            anchor: User-side anchor (claim, thought, or concept)
            candidates: Retrieved candidates to rank

        Returns:
            List of RankedBridgeCandidate sorted by score (descending)
        """
        if not candidates:
            return []

        if anchor is None:
            # Return candidates with default low score
            return [
                RankedBridgeCandidate(
                    candidate_id=self._get_id(c),
                    candidate=c,
                    score=0.1,
                    score_breakdown={"default": 0.1},
                    reason="No anchor provided",
                )
                for c in candidates
            ]

        results = []
        for candidate in candidates:
            score_breakdown = {}

            # Calculate individual scores
            concept_score = self._score_concept_overlap(anchor, candidate)
            value_score = self._score_value_axes_overlap(anchor, candidate)
            pattern_score = self._score_pattern_overlap(anchor, candidate)
            critique_score = self._score_critique_overlap(anchor, candidate)
            cross_domain_score = self._score_cross_domain_bonus(anchor, candidate)

            score_breakdown["concept"] = concept_score
            score_breakdown["value_axes"] = value_score
            score_breakdown["pattern"] = pattern_score
            score_breakdown["critique"] = critique_score
            score_breakdown["cross_domain"] = cross_domain_score

            # Calculate weighted total
            total_score = (
                concept_score * WEIGHT_CONCEPT +
                value_score * WEIGHT_VALUE_AXES +
                pattern_score * WEIGHT_PATTERN +
                critique_score * WEIGHT_CRITIQUE +
                cross_domain_score * WEIGHT_CROSS_DOMAIN
            )

            # Generate reason
            reason = self._generate_reason(score_breakdown)

            results.append(RankedBridgeCandidate(
                candidate_id=self._get_id(candidate),
                candidate=candidate,
                score=total_score,
                score_breakdown=score_breakdown,
                reason=reason,
            ))

        # Sort by score descending
        results.sort(key=lambda x: x.score, reverse=True)

        return results

    def _get_id(self, obj: Any) -> str:
        """Get unique ID from candidate object."""
        if isinstance(obj, Claim):
            return obj.claim_id
        elif isinstance(obj, ThoughtUnit):
            return obj.thought_id
        return str(id(obj))

    def _score_concept_overlap(
        self,
        anchor: Claim | ThoughtUnit | ConceptMapping,
        candidate: Claim | ThoughtUnit,
    ) -> float:
        """Score concept/theory overlap.

        Args:
            anchor: Source anchor
            candidate: Candidate to score

        Returns:
            Score 0.0-1.0
        """
        anchor_concepts = self._get_theory_hints(anchor)
        candidate_concepts = self._get_theory_hints(candidate)

        if not anchor_concepts or not candidate_concepts:
            return 0.0

        return self._calculate_overlap(anchor_concepts, candidate_concepts)

    def _score_value_axes_overlap(
        self,
        anchor: Claim | ThoughtUnit | ConceptMapping,
        candidate: Claim | ThoughtUnit,
    ) -> float:
        """Score value-axis overlap.

        Args:
            anchor: Source anchor
            candidate: Candidate to score

        Returns:
            Score 0.0-1.0
        """
        anchor_values = self._get_value_axes(anchor)
        candidate_values = self._get_value_axes(candidate)

        if not anchor_values or not candidate_values:
            return 0.0

        return self._calculate_overlap(anchor_values, candidate_values)

    def _score_pattern_overlap(
        self,
        anchor: Claim | ThoughtUnit | ConceptMapping,
        candidate: Claim | ThoughtUnit,
    ) -> float:
        """Score abstract-pattern overlap.

        Args:
            anchor: Source anchor
            candidate: Candidate to score

        Returns:
            Score 0.0-1.0
        """
        anchor_patterns = self._get_abstract_patterns(anchor)
        candidate_patterns = self._get_abstract_patterns(candidate)

        if not anchor_patterns or not candidate_patterns:
            return 0.0

        return self._calculate_overlap(anchor_patterns, candidate_patterns)

    def _score_critique_overlap(
        self,
        anchor: Claim | ThoughtUnit | ConceptMapping,
        candidate: Claim | ThoughtUnit,
    ) -> float:
        """Score critique-target overlap.

        Args:
            anchor: Source anchor
            candidate: Candidate to score

        Returns:
            Score 0.0-1.0
        """
        anchor_critiques = self._get_critique_targets(anchor)
        candidate_critiques = self._get_critique_targets(candidate)

        if not anchor_critiques or not candidate_critiques:
            return 0.0

        return self._calculate_overlap(anchor_critiques, candidate_critiques)

    def _score_cross_domain_bonus(
        self,
        anchor: Claim | ThoughtUnit | ConceptMapping,
        candidate: Claim | ThoughtUnit,
    ) -> float:
        """Score cross-domain bonus.

        If candidate is in a different domain than anchor, give a small bonus.

        Args:
            anchor: Source anchor
            candidate: Candidate to score

        Returns:
            Bonus score 0.0-0.15
        """
        # Simple heuristic: if theory hints are different but related
        anchor_concepts = set(self._get_theory_hints(anchor))
        candidate_concepts = set(self._get_theory_hints(candidate))

        if not anchor_concepts or not candidate_concepts:
            return 0.0

        # If there's some overlap but not complete match, it's cross-domain
        overlap = anchor_concepts & candidate_concepts
        if overlap and overlap != anchor_concepts:
            return 0.1  # Small bonus for partial cross-domain

        # If completely different domains
        if not overlap:
            return 0.05  # Tiny bonus

        return 0.0

    def _get_theory_hints(self, obj: Any) -> list[str]:
        """Get theory hints from object."""
        if isinstance(obj, Claim):
            return obj.theory_hints or []
        elif isinstance(obj, ThoughtUnit):
            return obj.recurring_patterns or []
        elif isinstance(obj, ConceptMapping):
            return [obj.preferred_label]
        return []

    def _get_value_axes(self, obj: Any) -> list[str]:
        """Get value axes from object."""
        if isinstance(obj, Claim):
            return obj.value_axes or []
        elif isinstance(obj, ThoughtUnit):
            return obj.value_axes or []
        return []

    def _get_abstract_patterns(self, obj: Any) -> list[str]:
        """Get abstract patterns from object."""
        if isinstance(obj, Claim):
            return obj.abstract_patterns or []
        elif isinstance(obj, ThoughtUnit):
            return obj.recurring_patterns or []
        return []

    def _get_critique_targets(self, obj: Any) -> list[str]:
        """Get critique targets from object."""
        if isinstance(obj, Claim):
            return obj.critique_target or []
        return []

    def _calculate_overlap(self, list1: list[str], list2: list[str]) -> float:
        """Calculate normalized overlap between two lists.

        Args:
            list1: First list
            list2: Second list

        Returns:
            Overlap score 0.0-1.0
        """
        if not list1 or not list2:
            return 0.0

        set1 = set(s.lower() for s in list1)
        set2 = set(s.lower() for s in list2)

        intersection = len(set1 & set2)
        union = len(set1 | set2)

        if union == 0:
            return 0.0

        return intersection / union

    def _generate_reason(self, score_breakdown: dict[str, float]) -> str:
        """Generate human-readable reason for ranking.

        Args:
            score_breakdown: Score breakdown dictionary

        Returns:
            Reason string
        """
        reasons = []

        if score_breakdown.get("concept", 0) > 0.3:
            reasons.append("concept match")
        if score_breakdown.get("value_axes", 0) > 0.3:
            reasons.append("value alignment")
        if score_breakdown.get("pattern", 0) > 0.3:
            reasons.append("pattern similarity")
        if score_breakdown.get("critique", 0) > 0.3:
            reasons.append("shared critique")
        if score_breakdown.get("cross_domain", 0) > 0:
            reasons.append("cross-domain")

        if reasons:
            return ", ".join(reasons)
        return "low signal match"
