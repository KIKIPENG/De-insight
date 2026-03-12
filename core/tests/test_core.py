"""Tests for De-insight v2 Core modules.

This test suite covers:
- Schema models
- Query classification
- Retrieval planning
- Thought extraction (basic)
- Compatibility layer

References:
- Tech spec: deinsight_v2_codex_technical_spec.md
"""

import sys
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.schemas import (
    Bridge,
    BridgeType,
    Claim,
    ConceptMapping,
    OwnerKind,
    QueryMode,
    RetrievalPlan,
    SourceKind,
    ThoughtStatus,
    ThoughtUnit,
    VocabSource,
)
from core.query_classifier import QueryClassifier, classify_query
from core.retrieval_planner import RetrievalPlanner, create_retrieval_plan
from core.compat import (
    enable_core_pipeline,
    disable_core_pipeline,
    is_core_enabled,
)


class TestSchemas:
    """Test schema models and their validation."""

    def test_claim_creation(self):
        """Test Claim model creation with defaults."""
        claim = Claim(
            project_id="test-project",
            core_claim="Test claim about design",
            critique_target=["functionality"],
            value_axes=["aesthetics"],
        )

        assert claim.claim_id is not None
        assert claim.project_id == "test-project"
        assert claim.core_claim == "Test claim about design"
        assert claim.critique_target == ["functionality"]
        assert claim.value_axes == ["aesthetics"]
        assert claim.confidence == 0.5
        assert claim.source_kind == SourceKind.USER_UTTERANCE

    def test_claim_confidence_bounds(self):
        """Test confidence field is bounded 0-1."""
        claim = Claim(core_claim="Test", confidence=0.8)
        assert claim.confidence == 0.8

        with pytest.raises(ValueError):
            Claim(core_claim="Test", confidence=1.5)

        with pytest.raises(ValueError):
            Claim(core_claim="Test", confidence=-0.1)

    def test_thought_unit_creation(self):
        """Test ThoughtUnit model creation."""
        thought = ThoughtUnit(
            project_id="test-project",
            title="On Function and Form",
            summary="Exploring the relationship between function and form in design",
            core_claim_ids=["abc123", "def456"],
            status=ThoughtStatus.EMERGING,
        )

        assert thought.thought_id is not None
        assert thought.title == "On Function and Form"
        assert len(thought.core_claim_ids) == 2
        assert thought.status == ThoughtStatus.EMERGING

    def test_thought_unit_status_values(self):
        """Test ThoughtStatus enum values."""
        assert ThoughtStatus.EMERGING.value == "emerging"
        assert ThoughtStatus.STABLE.value == "stable"
        assert ThoughtStatus.CONTESTED.value == "contested"

    def test_concept_mapping_creation(self):
        """Test ConceptMapping model."""
        mapping = ConceptMapping(
            project_id="test",
            owner_kind=OwnerKind.CLAIM,
            owner_id="claim123",
            vocab_source=VocabSource.AAT,
            concept_id="aat_12345",
            preferred_label="Minimalism",
            confidence=0.9,
        )

        assert mapping.owner_kind == OwnerKind.CLAIM
        assert mapping.vocab_source == VocabSource.AAT
        assert mapping.confidence == 0.9

    def test_bridge_creation(self):
        """Test Bridge model."""
        bridge = Bridge(
            project_id="test",
            source_claim_id="claim1",
            target_claim_id="claim2",
            bridge_type=BridgeType.ANALOGY,
            reason_summary="Both discuss function-form relationship",
            shared_patterns=["function", "form", "minimalism"],
            confidence=0.75,
        )

        assert bridge.bridge_type == BridgeType.ANALOGY
        assert len(bridge.shared_patterns) == 3

    def test_retrieval_plan_creation(self):
        """Test RetrievalPlan model."""
        plan = RetrievalPlan(
            project_id="test",
            query_mode=QueryMode.DEEP,
            why_deep="理論框架相關",
            concept_queries=["function form relationship", "minimalism aesthetics"],
            supporting_paths=["理論脈絡", "歷史脈絡"],
            analogy_paths=["跨領域類比", "結構相似性"],
            max_passages_per_path=5,
        )

        assert plan.query_mode == QueryMode.DEEP
        assert plan.why_deep == "理論框架相關"
        assert len(plan.concept_queries) == 2
        assert plan.max_passages_per_path == 5


class TestQueryClassifier:
    """Test query classification for fast/deep mode."""

    def setup_method(self):
        """Set up classifier for each test."""
        self.classifier = QueryClassifier()

    def test_fact_query_fast_mode(self):
        """Test factual questions are classified as fast."""
        result = self.classifier.classify("什麼是包豪斯？")

        assert result.mode == QueryMode.FAST
        assert result.confidence >= 0.8

    def test_theory_query_deep_mode(self):
        """Test theory-related questions are classified as deep."""
        result = self.classifier.classify("包豪斯的理論框架是什麼？")

        assert result.mode == QueryMode.DEEP
        assert result.why_deep is not None

    def test_structural_similarity_deep(self):
        """Test structural similarity queries use deep mode."""
        result = self.classifier.classify(
            "這和其他設計運動的結構模式有什麼相似之處？"
        )

        assert result.mode == QueryMode.DEEP

    def test_analogy_query_deep(self):
        """Test analogy queries use deep mode."""
        result = self.classifier.classify(
            "這像建築中的什麼？兩者有什麼相通之處？"
        )

        assert result.mode == QueryMode.DEEP

    def test_concept_query_deep(self):
        """Test conceptual interpretation uses deep mode."""
        result = self.classifier.classify("這個概念是什麼意思？")

        assert result.mode == QueryMode.DEEP

    def test_empty_query_default_fast(self):
        """Test empty query defaults to fast mode."""
        result = self.classifier.classify("")

        assert result.mode == QueryMode.FAST

    def test_short_query_fast(self):
        """Test very short queries may use fast mode."""
        result = self.classifier.classify("包豪斯")

        # Short queries may go either way, but shouldn't error
        assert result.mode in [QueryMode.FAST, QueryMode.DEEP]


class TestQueryClassifierConvenience:
    """Test convenience function for query classification."""

    def test_classify_query_function(self):
        """Test classify_query convenience function."""
        result = classify_query("這個設計的理論基礎是什麼？")

        assert isinstance(result.mode, QueryMode)
        assert isinstance(result.confidence, float)


class TestRetrievalPlanner:
    """Test retrieval planning."""

    def setup_method(self):
        """Set up planner for each test."""
        self.planner = RetrievalPlanner()

    def test_fast_plan_structure(self):
        """Test fast mode plan has correct structure."""
        plan = self.planner.create_plan(
            query="什麼是包豪斯？",
            project_id="test",
        )

        assert plan.query_mode == QueryMode.FAST
        assert plan.why_deep is None
        assert len(plan.concept_queries) >= 1
        assert plan.max_passages_per_path == 3

    def test_deep_plan_structure(self):
        """Test deep mode plan has correct structure."""
        plan = self.planner.create_plan(
            query="這個設計理念和哲學有什麼關係？",
            project_id="test",
        )

        assert plan.query_mode == QueryMode.DEEP
        assert plan.why_deep is not None
        assert len(plan.concept_queries) >= 1
        assert plan.max_passages_per_path == 5
        assert len(plan.supporting_paths) >= 1
        assert len(plan.analogy_paths) >= 1

    def test_plan_with_context(self):
        """Test plan creation includes context."""
        context = [
            {"role": "user", "content": "我對包豪斯有興趣"},
            {"role": "assistant", "content": "請說"},
        ]

        plan = self.planner.create_plan(
            query="它的理論框架是什麼？",
            context=context,
            project_id="test",
        )

        # Context may influence classification
        assert plan.project_id == "test"


class TestRetrievalPlannerConvenience:
    """Test convenience function for retrieval planning."""

    def test_create_retrieval_plan_function(self):
        """Test create_retrieval_plan convenience function."""
        plan = create_retrieval_plan(
            query="設計和藝術有什麼關係？",
            project_id="test",
        )

        assert isinstance(plan, RetrievalPlan)


class TestCompatibilityLayer:
    """Test compatibility layer."""

    def setup_method(self):
        """Reset core state before each test."""
        disable_core_pipeline()

    def teardown_method(self):
        """Reset core state after each test."""
        disable_core_pipeline()

    def test_core_disabled_by_default(self):
        """Test core pipeline is disabled by default."""
        assert is_core_enabled() is False

    def test_enable_core_pipeline(self):
        """Test enabling core pipeline."""
        enable_core_pipeline()
        assert is_core_enabled() is True

    def test_disable_core_pipeline(self):
        """Test disabling core pipeline."""
        enable_core_pipeline()
        disable_core_pipeline()
        assert is_core_enabled() is False


class TestModuleImports:
    """Test that all public exports are available."""

    def test_schemas_exported(self):
        """Test schema models are exported."""
        from core import (
            Bridge,
            Claim,
            ConceptMapping,
            QueryMode,
            RetrievalPlan,
            ThoughtUnit,
        )

        assert Claim is not None
        assert ThoughtUnit is not None
        assert ConceptMapping is not None
        assert Bridge is not None
        assert RetrievalPlan is not None
        assert QueryMode is not None

    def test_classifiers_exported(self):
        """Test classifier is exported."""
        from core import QueryClassifier, classify_query

        assert QueryClassifier is not None
        assert classify_query is not None

    def test_planners_exported(self):
        """Test planner is exported."""
        from core import RetrievalPlanner, create_retrieval_plan

        assert RetrievalPlanner is not None
        assert create_retrieval_plan is not None

    def test_stores_exported(self):
        """Test stores are exported."""
        from core import ClaimStore, ThoughtStore, ConceptStore, BridgeStore

        assert ClaimStore is not None
        assert ThoughtStore is not None
        assert ConceptStore is not None
        assert BridgeStore is not None
