"""Retriever module for De-insight v2 Core.

This module executes retrieval based on a RetrievalPlan.
Phase 2 adds hybrid retrieval that combines legacy pipeline results
with store-backed results from ClaimStore, ThoughtStore, and ConceptStore.

References:
- Tech spec: deinsight_v2_codex_technical_spec.md
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from core.schemas import (
    Claim,
    ConceptMapping,
    QueryMode,
    RetrievalPlan,
    RetrievalResult,
    ThoughtUnit,
)
from core.stores import ClaimStore, ConceptStore, ThoughtStore


# Merge configuration
LEGACY_MAX = 5
CLAIM_MAX = 3
THOUGHT_MAX = 3

# Anchor quality thresholds
ANCHOR_QUALITY_THRESHOLD = 10  # Minimum score out of 25
MIN_VALUE_AXES = 2
MIN_THEORY_HINTS = 1


def assess_anchor_quality(anchor: Any) -> dict:
    """Assess anchor quality for bridge ranking.
    
    Returns dict with:
    - quality_score: 0-25 based on dimensions
    - is_thin: bool indicating if anchor needs enrichment
    - dimensions: per-dimension scores
    """
    if anchor is None:
        return {"quality_score": 0, "is_thin": True, "dimensions": {}}
    
    score = 0
    dimensions = {}
    
    # Core claim presence
    has_core = hasattr(anchor, 'core_claim') and anchor.core_claim
    if has_core:
        claim_len = len(anchor.core_claim)
        if claim_len > 30:
            score += 5
        elif claim_len > 15:
            score += 3
        else:
            score += 1
        dimensions["core_claim"] = 5 if claim_len > 30 else 3 if claim_len > 15 else 1
    else:
        dimensions["core_claim"] = 0
    
    # Value axes richness
    value_axes = getattr(anchor, 'value_axes', []) or []
    if len(value_axes) >= 4:
        score += 5
        dimensions["value_axes"] = 5
    elif len(value_axes) >= 3:
        score += 4
        dimensions["value_axes"] = 4
    elif len(value_axes) >= 2:
        score += 3
        dimensions["value_axes"] = 3
    elif len(value_axes) >= 1:
        score += 2
        dimensions["value_axes"] = 2
    else:
        dimensions["value_axes"] = 1
    
    # Theory hints usefulness
    theory_hints = getattr(anchor, 'theory_hints', []) or []
    if len(theory_hints) >= 3:
        score += 5
        dimensions["theory_hints"] = 5
    elif len(theory_hints) >= 2:
        score += 4
        dimensions["theory_hints"] = 4
    elif len(theory_hints) >= 1:
        score += 3
        dimensions["theory_hints"] = 3
    else:
        dimensions["theory_hints"] = 1
    
    # Abstract patterns
    patterns = getattr(anchor, 'abstract_patterns', []) or []
    if len(patterns) >= 2:
        score += 5
        dimensions["abstract_patterns"] = 5
    elif len(patterns) >= 1:
        score += 3
        dimensions["abstract_patterns"] = 3
    else:
        dimensions["abstract_patterns"] = 1
    
    # Critique targets
    critiques = getattr(anchor, 'critique_target', []) or []
    if len(critiques) >= 2:
        score += 5
        dimensions["critique_targets"] = 5
    elif len(critiques) >= 1:
        score += 3
        dimensions["critique_targets"] = 3
    else:
        dimensions["critique_targets"] = 1
    
    is_thin = score < ANCHOR_QUALITY_THRESHOLD or len(value_axes) < MIN_VALUE_AXES or len(theory_hints) < MIN_THEORY_HINTS
    
    return {
        "quality_score": score,
        "percentage": (score / 25) * 100,
        "is_thin": is_thin,
        "dimensions": dimensions,
    }


def enrich_thin_anchor(anchor: Claim, plan: RetrievalPlan) -> Claim:
    """Enrich a thin anchor with additional context from the retrieval plan.
    
    This provides a lightweight fallback when the anchor is too thin
    to produce meaningful bridges.
    """
    # Try to enrich with concept queries
    if plan.concept_queries:
        existing_values = list(anchor.value_axes) if anchor.value_axes else []
        existing_hints = list(anchor.theory_hints) if anchor.theory_hints else []
        
        # Add concept queries as value axes if we have few
        if len(existing_values) < MIN_VALUE_AXES:
            for cq in plan.concept_queries[:3]:
                if cq not in existing_values:
                    existing_values.append(cq)
        
        # Add concept queries as theory hints if we have few
        if len(existing_hints) < MIN_THEORY_HINTS:
            for cq in plan.concept_queries[:2]:
                if cq not in existing_hints:
                    existing_hints.append(cq)
        
        # Create enriched anchor
        return Claim(
            project_id=anchor.project_id,
            core_claim=anchor.core_claim or plan.concept_queries[0] if plan.concept_queries else "enriched anchor",
            value_axes=existing_values,
            theory_hints=existing_hints,
            abstract_patterns=list(anchor.abstract_patterns) if anchor.abstract_patterns else [],
            critique_target=list(anchor.critique_target) if anchor.critique_target else [],
        )
    
    return anchor
CONCEPT_MAX = 3


@dataclass
class RetrievalItem:
    """Internal retrieval item with metadata.

    Used for merging results from different sources.
    """
    item: Any
    source: str  # "legacy" | "claim" | "thought" | "concept"
    score: float = 0.0
    reason: str = ""


class Retriever:
    """Executes retrieval based on a RetrievalPlan.

    Phase 2 adds hybrid retrieval that combines:
    - Legacy rag.pipeline results
    - ClaimStore results
    - ThoughtStore results
    - ConceptStore results (via ConceptMapper)

    The results are merged with simple deduplication and limits.
    """

    def __init__(
        self,
        project_id: str = "default",
        claim_store: ClaimStore | None = None,
        thought_store: ThoughtStore | None = None,
        concept_store: ConceptStore | None = None,
    ):
        """Initialize the retriever.

        Args:
            project_id: Project identifier
            claim_store: Optional ClaimStore instance
            thought_store: Optional ThoughtStore instance
            concept_store: Optional ConceptStore instance
        """
        self.project_id = project_id
        self._claim_store = claim_store
        self._thought_store = thought_store
        self._concept_store = concept_store

    @property
    def claim_store(self) -> ClaimStore:
        """Get or create claim store."""
        if self._claim_store is None:
            self._claim_store = ClaimStore(project_id=self.project_id)
        return self._claim_store

    @property
    def thought_store(self) -> ThoughtStore:
        """Get or create thought store."""
        if self._thought_store is None:
            self._thought_store = ThoughtStore(project_id=self.project_id)
        return self._thought_store

    @property
    def concept_store(self) -> ConceptStore:
        """Get or create concept store."""
        if self._concept_store is None:
            self._concept_store = ConceptStore(project_id=self.project_id)
        return self._concept_store

    async def retrieve(
        self,
        plan: RetrievalPlan,
        user_query: str,
    ) -> RetrievalResult:
        """Execute retrieval based on the plan.

        Args:
            plan: RetrievalPlan with configured strategy
            user_query: Original user query

        Returns:
            RetrievalResult with passages, claims, bridges, and sources
        """
        # Try hybrid retrieval if stores are available
        if self._has_stores():
            return await self._retrieve_hybrid(plan, user_query)

        # Fallback to legacy-only
        if plan.query_mode == QueryMode.FAST:
            return await self._retrieve_fast(plan, user_query)
        else:
            return await self._retrieve_deep(plan, user_query)

    def _has_stores(self) -> bool:
        """Check if any store is available."""
        # Check if stores have been injected or can be created
        return (
            self._claim_store is not None
            or self._thought_store is not None
            or self._concept_store is not None
        )

    async def _retrieve_hybrid(
        self,
        plan: RetrievalPlan,
        user_query: str,
    ) -> RetrievalResult:
        """Execute hybrid retrieval combining stores and legacy.

        Args:
            plan: RetrievalPlan with configured strategy
            user_query: Original user query

        Returns:
            RetrievalResult with merged results and ranked bridges
        """
        # Get legacy results
        legacy_results = await self._retrieve_legacy(plan, user_query)

        # Get store results
        claims = await self._retrieve_from_claims(user_query)
        thoughts = await self._retrieve_from_thoughts(user_query)
        concepts = await self._retrieve_from_concepts(user_query)

        # Merge results
        result = self._merge_results(
            legacy=legacy_results.get("passages", []),
            claims=claims,
            thoughts=thoughts,
            concepts=concepts,
            plan=plan,
        )

        # Run bridge ranking
        result = await self._rank_bridges(result, plan, user_query)

        return result

    async def _rank_bridges(
        self,
        result: RetrievalResult,
        plan: RetrievalPlan,
        user_query: str,
    ) -> RetrievalResult:
        """Rank bridge candidates and populate result.bridges.

        Anchor priority: Claim > ThoughtUnit > concept context
        Gracefully handles missing anchor or no candidates.

        Args:
            result: Merged RetrievalResult
            plan: Original retrieval plan
            user_query: Original user query

        Returns:
            RetrievalResult with populated bridges
        """
        # Build anchor and candidates with graceful fallback
        anchor = None
        candidates = []

        # Priority 1: Use first claim as anchor
        if result.claims:
            anchor = result.claims[0]
            candidates = result.claims[1:]  # Rest are candidates

        # Priority 2: Use first thought as anchor
        if anchor is None and plan.thought_summary:
            # Use thought summary as anchor concept
            from core.schemas import ConceptMapping, OwnerKind
            anchor = ConceptMapping(
                project_id=self.project_id,
                owner_kind=OwnerKind.CLAIM,
                owner_id="thought_anchor",
                concept_id=plan.thought_summary[:50],
                preferred_label=plan.thought_summary,
            )

        # Priority 3: Use concept queries as context
        if anchor is None and plan.concept_queries:
            from core.schemas import ConceptMapping, OwnerKind
            anchor = ConceptMapping(
                project_id=self.project_id,
                owner_kind=OwnerKind.CLAIM,
                owner_id="concept_anchor",
                concept_id=plan.concept_queries[0] if plan.concept_queries else "",
                preferred_label=plan.concept_queries[0] if plan.concept_queries else "",
            )

        # Assess anchor quality
        quality = assess_anchor_quality(anchor)
        
        # Enrich thin anchor if needed
        if quality["is_thin"] and isinstance(anchor, Claim):
            anchor = enrich_thin_anchor(anchor, plan)
            # Re-assess after enrichment
            quality = assess_anchor_quality(anchor)
        
        # Run bridge ranker if we have anchor and candidates
        if anchor and candidates:
            try:
                from core.bridge_ranker import BridgeRanker
                from typing import cast

                ranker = BridgeRanker()
                ranked = ranker.rank_candidates(anchor, cast(list, candidates))

                # Convert to Bridge objects
                from core.schemas import Bridge, BridgeType

                bridges = []
                for rc in ranked:
                    bridge = Bridge(
                        project_id=self.project_id,
                        source_claim_id=getattr(anchor, 'claim_id', ''),
                        target_claim_id=rc.candidate_id,
                        bridge_type=BridgeType.VALUE_STRUCTURE_MATCH,
                        reason_summary=rc.reason,
                        confidence=rc.score,
                        score=rc.score,
                        score_breakdown=rc.score_breakdown,
                    )
                    bridges.append(bridge)

                result.bridges = bridges

            except Exception:
                # Graceful fallback: keep empty bridges
                result.bridges = []
        else:
            # No anchor or no candidates - return empty bridges
            result.bridges = []

        return result

    async def _retrieve_from_claims(self, query: str) -> list[dict]:
        """Retrieve from ClaimStore.

        Args:
            query: Search query

        Returns:
            List of claim items with metadata
        """
        try:
            # Use text search if store available
            if self._claim_store is not None:
                results = await self.claim_store.search_by_text(query, limit=10)
                return [
                    {"claim": claim, "score": 0.8, "reason": "text_match"}
                    for claim in results
                ]
        except Exception:
            pass
        return []

    async def _retrieve_from_thoughts(self, query: str) -> list[dict]:
        """Retrieve from ThoughtStore.

        Args:
            query: Search query

        Returns:
            List of thought items with metadata
        """
        try:
            if self._thought_store is not None:
                # Get recent thoughts (simple approach)
                thoughts = await self.thought_store.list_by_project(
                    project_id=self.project_id,
                    limit=10,
                )
                # Simple relevance: check if query appears in title/summary
                matched = []
                query_lower = query.lower()
                for thought in thoughts:
                    score = 0.5
                    reason = "recent"
                    if query_lower in thought.title.lower():
                        score = 0.9
                        reason = "title_match"
                    elif query_lower in thought.summary.lower():
                        score = 0.7
                        reason = "summary_match"
                    if score > 0.5:
                        matched.append({
                            "thought": thought,
                            "score": score,
                            "reason": reason,
                        })
                return matched
        except Exception:
            pass
        return []

    async def _retrieve_from_concepts(self, query: str) -> list[dict]:
        """Retrieve from ConceptStore using ConceptMapper.

        Args:
            query: Search query

        Returns:
            List of concept items with metadata
        """
        try:
            if self._concept_store is not None:
                # First use ConceptMapper to normalize query
                from core.concept_mapper import ConceptMapper

                # Create simple sync wrapper for async mapper
                def sync_mapper(prompt: str) -> str:
                    # Return empty for now - we'll do direct lookup
                    return '{"concepts": []}'

                mapper = ConceptMapper(
                    llm_callable=sync_mapper,
                    project_id=self.project_id,
                    concept_store=self.concept_store,
                )

                # Try to map query to concepts
                try:
                    from core.schemas import OwnerKind
                    
                    concept_mappings = mapper.map_text_to_concepts(
                        text=query,
                        owner_kind=OwnerKind.CLAIM,  # Use CLAIM as default
                        owner_id="hybrid_retrieval",
                    )

                    if concept_mappings:
                        # Query store for each concept
                        results = []
                        for cm in concept_mappings:
                            store_results = await self.concept_store.list_by_concept(
                                cm.concept_id,
                                limit=5,
                            )
                            for sr in store_results:
                                results.append({
                                    "concept": sr,
                                    "score": cm.confidence,
                                    "reason": f"concept_match:{cm.concept_id}",
                                })
                        return results
                except Exception:
                    pass

                # Fallback: direct text search in concepts
                # (ConceptStore doesn't have text search, so just return recent)
                recent = await self.concept_store.list_by_concept(
                    "",
                    limit=10,
                )
                return [
                    {"concept": c, "score": 0.5, "reason": "recent"}
                    for c in recent
                ]
        except Exception:
            pass
        return []

    async def _retrieve_legacy(
        self,
        plan: RetrievalPlan,
        user_query: str,
    ) -> dict:
        """Retrieve from legacy pipeline.

        NOTE: We intentionally return empty results here to avoid circular
        recursion. The pipeline has already fetched sources before calling
        core.retriever, so we don't need to re-run the pipeline here.

        Args:
            plan: RetrievalPlan
            user_query: Original query

        Returns:
            Dict with passages and sources (empty to avoid recursion)
        """
        return {"passages": [], "sources": []}

    def _merge_results(
        self,
        legacy: list[dict],
        claims: list[dict],
        thoughts: list[dict],
        concepts: list[dict],
        plan: RetrievalPlan,
    ) -> RetrievalResult:
        """Merge results from multiple sources.

        Merge rules:
        - Duplicate: same core_claim/title (case-insensitive)
        - Limits: legacy: 5, claims: 3, thoughts: 3, concepts: 3
        - When store empty, still return legacy
        - Sort by source weight then confidence

        Args:
            legacy: Legacy pipeline results
            claims: Claim store results
            thoughts: Thought store results
            concepts: Concept store results
            plan: Original retrieval plan

        Returns:
            Merged RetrievalResult
        """
        # Apply limits
        limited_legacy = legacy[:LEGACY_MAX]
        limited_claims = claims[:CLAIM_MAX]
        limited_thoughts = thoughts[:THOUGHT_MAX]
        limited_concepts = concepts[:CONCEPT_MAX]

        # Deduplicate
        seen_texts = set()

        # Process legacy
        final_passages = []
        for item in limited_legacy:
            text = item.get("text", "")
            if text and text.lower() not in seen_texts:
                seen_texts.add(text.lower())
                final_passages.append(item)

        # Process claims - extract core_claim for deduplication
        final_claims = []
        for item in limited_claims:
            claim = item.get("claim")
            if claim and isinstance(claim, Claim):
                key = claim.core_claim.lower() if claim.core_claim else ""
                if key and key not in seen_texts:
                    seen_texts.add(key)
                    final_claims.append(claim)

        # Build result
        return RetrievalResult(
            plan=plan,
            passages=final_passages,
            claims=final_claims,
            bridges=[],
            sources=legacy[0].get("sources", []) if legacy else [],
        )

    async def _retrieve_fast(
        self,
        plan: RetrievalPlan,
        user_query: str,
    ) -> RetrievalResult:
        """Execute fast mode retrieval (legacy only).

        Args:
            plan: Fast mode retrieval plan
            user_query: Original query

        Returns:
            RetrievalResult
        """
        try:
            from rag.pipeline import run_thinking_pipeline

            result = await run_thinking_pipeline(
                user_input=user_query,
                project_id=self.project_id,
                mode="fast",
            )

            return RetrievalResult(
                plan=plan,
                passages=result.get("passages", []),
                claims=[],
                bridges=[],
                sources=result.get("sources", []),
            )
        except Exception:
            return RetrievalResult(
                plan=plan,
                passages=[],
                claims=[],
                bridges=[],
                sources=[],
            )

    async def _retrieve_deep(
        self,
        plan: RetrievalPlan,
        user_query: str,
    ) -> RetrievalResult:
        """Execute deep mode retrieval (legacy only).

        Args:
            plan: Deep mode retrieval plan
            user_query: Original query

        Returns:
            RetrievalResult
        """
        try:
            from rag.pipeline import run_thinking_pipeline

            result = await run_thinking_pipeline(
                user_input=user_query,
                project_id=self.project_id,
                mode="deep",
            )

            return RetrievalResult(
                plan=plan,
                passages=result.get("passages", []),
                claims=[],
                bridges=[],
                sources=result.get("sources", []),
            )
        except Exception:
            return RetrievalResult(
                plan=plan,
                passages=[],
                claims=[],
                bridges=[],
                sources=[],
            )


async def retrieve_with_plan(
    plan: RetrievalPlan,
    user_query: str,
    project_id: str = "default",
) -> RetrievalResult:
    """Convenience function for retrieval execution.

    Args:
        plan: RetrievalPlan
        user_query: Original query
        project_id: Project identifier

    Returns:
        RetrievalResult
    """
    retriever = Retriever(project_id)
    return await retriever.retrieve(plan, user_query)
