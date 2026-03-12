"""Retrieval planning module for De-insight v2 Core.

This module creates structured retrieval plans based on classified queries.
It generates concept queries, supporting paths, and analogy paths for
multi-route retrieval.

References:
- Tech spec: deinsight_v2_codex_technical_spec.md
"""

from __future__ import annotations

from dataclasses import dataclass

from core.query_classifier import ClassificationResult, QueryClassifier
from core.schemas import QueryMode, RetrievalPlan


# Default max passages per path
DEFAULT_MAX_PASSAGES = 3
DEEP_MAX_PASSAGES = 5


@dataclass
class PlanningInput:
    """Input for retrieval planning.

    Attributes:
        query: Original user query
        classification: Classification result from QueryClassifier
        context: Optional conversation context for deeper understanding
        thought_summary: Optional summary of user's current thinking
    """

    query: str
    classification: ClassificationResult
    context: list[dict] | None = None
    thought_summary: str | None = None


class RetrievalPlanner:
    """Plans retrieval strategies based on query classification.

    The planner creates RetrievalPlan objects that define:
    - concept_queries: Key concept-based search queries
    - supporting_paths: Paths for finding supporting evidence
    - analogy_paths: Paths for finding analogical connections
    - max_passages_per_path: Maximum passages to retrieve per path

    Fast mode:
    - Single concept query
    - Simple semantic search
    - max_passages_per_path = 3

    Deep mode:
    - Multiple concept queries (extracted from query structure)
    - Supporting paths + analogy paths
    - max_passages_per_path = 5
    - Multi-route retrieval
    """

    def __init__(self):
        """Initialize the retrieval planner."""
        self._classifier = QueryClassifier()

    def create_plan(
        self,
        query: str,
        context: list[dict] | None = None,
        project_id: str = "default",
    ) -> RetrievalPlan:
        """Create a retrieval plan for the given query.

        Args:
            query: User query
            context: Optional conversation context
            project_id: Project identifier

        Returns:
            RetrievalPlan with configured retrieval strategy
        """
        # First classify the query if not provided
        classification = self._classifier.classify(query, context)

        return self._build_plan(
            query=query,
            classification=classification,
            context=context,
            project_id=project_id,
        )

    def _build_plan(
        self,
        query: str,
        classification: ClassificationResult,
        context: list[dict] | None,
        project_id: str,
    ) -> RetrievalPlan:
        """Build a retrieval plan based on classification.

        Args:
            query: User query
            classification: Classification result
            context: Conversation context
            project_id: Project identifier

        Returns:
            RetrievalPlan
        """
        mode = classification.mode

        if mode == QueryMode.FAST:
            return self._build_fast_plan(query, project_id)
        else:
            return self._build_deep_plan(
                query=query,
                classification=classification,
                context=context,
                project_id=project_id,
            )

    def _build_fast_plan(self, query: str, project_id: str) -> RetrievalPlan:
        """Build a fast mode retrieval plan.

        Fast mode uses a single query with simple semantic search.

        Args:
            query: User query
            project_id: Project identifier

        Returns:
            RetrievalPlan for fast mode
        """
        # Extract key concepts for the query (simple approach)
        concept_queries = self._extract_concepts_simple(query)

        return RetrievalPlan(
            project_id=project_id,
            query_mode=QueryMode.FAST,
            why_deep=None,
            thought_summary="",
            concept_queries=concept_queries,
            supporting_paths=[],
            analogy_paths=[],
            max_passages_per_path=DEFAULT_MAX_PASSAGES,
        )

    def _build_deep_plan(
        self,
        query: str,
        classification: ClassificationResult,
        context: list[dict] | None,
        project_id: str,
    ) -> RetrievalPlan:
        """Build a deep mode retrieval plan.

        Deep mode uses:
        - Multiple concept queries (extracted from query structure)
        - Supporting paths for finding evidence
        - Analogy paths for cross-domain connections

        Args:
            query: User query
            classification: Classification result
            context: Conversation context
            project_id: Project identifier

        Returns:
            RetrievalPlan for deep mode
        """
        # Extract concepts and generate queries
        concept_queries = self._extract_concepts_deep(query, context)

        # Generate supporting paths (for finding evidence)
        supporting_paths = self._generate_supporting_paths(query)

        # Generate analogy paths (for cross-domain connections)
        analogy_paths = self._generate_analogy_paths(query)

        return RetrievalPlan(
            project_id=project_id,
            query_mode=QueryMode.DEEP,
            why_deep=classification.why_deep,
            thought_summary=classification.signals[0] if classification.signals else "",
            concept_queries=concept_queries,
            supporting_paths=supporting_paths,
            analogy_paths=analogy_paths,
            max_passages_per_path=DEEP_MAX_PASSAGES,
        )

    def _extract_concepts_simple(self, query: str) -> list[str]:
        """Extract key concepts from query (simple approach).

        Args:
            query: User query

        Returns:
            List of concept queries
        """
        # Simple approach: use query as-is and add variations
        queries = [query]

        # Add a shorter version focusing on key terms
        words = query.replace("?", "").replace("？", "").split()
        if len(words) > 3:
            # Keep significant words (exclude common particles)
            stop_words = {"的", "是", "在", "和", "與", "了", "有", "沒有", "這", "那", "什麼", "怎麼", "如何"}
            key_words = [w for w in words if w not in stop_words]
            if key_words:
                queries.append(" ".join(key_words[:4]))

        return queries[:2]

    def _extract_concepts_deep(
        self,
        query: str,
        context: list[dict] | None,
    ) -> list[str]:
        """Extract concepts for deep mode.

        Generates multiple search queries based on:
        - The original query
        - Conceptual structure derived from the query
        - Potential theoretical frameworks

        Args:
            query: User query
            context: Conversation context

        Returns:
            List of concept queries
        """
        queries = [query]

        # Generate abstract reformulation of the query
        abstract_query = self._abstractize_query(query)
        if abstract_query:
            queries.append(abstract_query)

        # Extract domain-independent structural query
        structural_query = self._extract_structural_query(query)
        if structural_query:
            queries.append(structural_query)

        # Add conceptual variations
        conceptual_variations = self._generate_conceptual_variations(query)
        queries.extend(conceptual_variations)

        # Deduplicate and limit
        seen = set()
        unique_queries = []
        for q in queries:
            q_normalized = q.lower().strip()
            if q_normalized and q_normalized not in seen:
                seen.add(q_normalized)
                unique_queries.append(q)

        return unique_queries[:4]

    def _abstractize_query(self, query: str) -> str | None:
        """Convert query to its abstract structural form.

        This creates a domain-independent version of the query
        for finding analogies across different fields.

        Args:
            query: User query

        Returns:
            Abstracted query or None
        """
        # This is a simplified heuristic approach
        # A full implementation would use an LLM for better results

        # Replace specific domain terms with generic ones
        replacements = [
            ("設計", "實踐"),
            ("藝術", "創作"),
            ("建築", "空間"),
            ("作品", "產出"),
        ]

        abstracted = query
        for old, new in replacements:
            abstracted = abstracted.replace(old, new)

        if abstracted != query:
            return abstracted

        return None

    def _extract_structural_query(self, query: str) -> str | None:
        """Extract the structural/logical pattern from the query.

        Args:
            query: User query

        Returns:
            Structural query or None
        """
        # Look for structural keywords
        if any(kw in query for kw in ["關係", "結構", "邏輯", "因為", "所以"]):
            # Keep the logical part
            return query

        return None

    def _generate_conceptual_variations(self, query: str) -> list[str]:
        """Generate conceptual variations of the query.

        Args:
            query: User query

        Returns:
            List of conceptual variations
        """
        variations = []

        # Add "為什麼" variation if not present
        if "為什麼" not in query and "怎麼" not in query:
            variations.append(f"這是什麼？{query}")

        return variations

    def _generate_supporting_paths(self, query: str) -> list[str]:
        """Generate paths for finding supporting evidence.

        Args:
            query: User query

        Returns:
            List of supporting path descriptions
        """
        paths = [
            "理論脈絡",
            "歷史脈絡",
            "案例支持",
        ]
        return paths

    def _generate_analogy_paths(self, query: str) -> list[str]:
        """Generate paths for finding analogical connections.

        Args:
            query: User query

        Returns:
            List of analogy path descriptions
        """
        paths = [
            "跨領域類比",
            "結構相似性",
            "價值取向對比",
        ]
        return paths


def create_retrieval_plan(
    query: str,
    context: list[dict] | None = None,
    project_id: str = "default",
) -> RetrievalPlan:
    """Convenience function for creating a retrieval plan.

    Args:
        query: User query
        context: Optional conversation context
        project_id: Project identifier

    Returns:
        RetrievalPlan
    """
    planner = RetrievalPlanner()
    return planner.create_plan(query, context, project_id)
