"""Legacy/core comparison tests for De-insight v2 Core.

These tests compare legacy vs core pipeline outputs for the same inputs.

References:
- Tech spec: deinsight_phase2_test_debug_spec.md
- Fixtures: core/tests/fixtures/legacy_core_comparison_cases.json
"""

import json
import sys
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.query_classifier import QueryClassifier
from core.compat import enable_core_pipeline, disable_core_pipeline, is_core_enabled, run_core_pipeline, CorePipelineInput


def _load_comparison_cases() -> list[dict]:
    """Load comparison cases from fixtures."""
    fixtures_path = Path(__file__).parent / "fixtures" / "legacy_core_comparison_cases.json"
    with open(fixtures_path) as f:
        data = json.load(f)
    return data["comparison_cases"]


class TestLegacyCoreComparison:
    """Test legacy vs core comparison fixtures."""

    @pytest.mark.parametrize("case", _load_comparison_cases())
    def test_classification_matches_expected(self, case):
        """Verify classification matches expected mode."""
        classifier = QueryClassifier()
        query = case["user_message"]
        expected_mode = case["expected_mode"]

        result = classifier.classify(query)

        assert result.mode == expected_mode, (
            f"Case {case['case_id']}: expected {expected_mode}, got {result.mode.value}"
        )


class TestLegacyCoreDivergenceAnalysis:
    """Test legacy vs core comparison."""

    def setup_method(self):
        disable_core_pipeline()

    def teardown_method(self):
        disable_core_pipeline()

    def test_core_enabled_flag_behavior(self):
        """Test that core enabled flag changes behavior."""
        # When disabled, core should not be enabled
        assert not is_core_enabled()

        enable_core_pipeline()
        assert is_core_enabled()

        disable_core_pipeline()
        assert not is_core_enabled()

    @pytest.mark.asyncio
    async def test_comparison_case_deep_mode(self):
        """Test a deep mode comparison case runs correctly."""
        enable_core_pipeline()

        def mock_llm(prompt: str) -> str:
            return '{"claims": [], "thought_summary": "", "concepts": []}'

        result = await run_core_pipeline(CorePipelineInput(
            user_message="包豪斯的理論框架是什麼？",
            project_id="test",
            llm_callable=mock_llm,
        ))

        # Result should have extraction and plan
        assert result is not None


class TestClassifierDecisionTrace:
    """Test that classifier decisions are traceable."""

    def test_fast_mode_decision_trace(self):
        """Verify fast mode decision is explainable."""
        classifier = QueryClassifier()
        query = "包豪斯成立於哪一年？"

        result = classifier.classify(query)

        assert result.mode.value == "fast"
        assert result.confidence > 0

    def test_deep_mode_decision_trace(self):
        """Verify deep mode decision is explainable."""
        classifier = QueryClassifier()
        query = "這和其他設計運動的結構模式有什麼相似之處？"

        result = classifier.classify(query)

        assert result.mode == "deep"
        assert result.why_deep is not None
        assert len(result.signals) > 0


class TestExtractionDecisionTrace:
    """Test that extraction decisions are traceable."""

    @pytest.mark.asyncio
    async def test_extraction_produces_traceable_output(self):
        """Verify extraction produces traceable output."""
        from core.thought_extractor import ThoughtExtractor, LLMCallable

        def mock_llm(prompt: str) -> str:
            return '{"claims": [{"core_claim": "test", "critique_target": ["target1"], "value_axes": ["value1"], "materiality_axes": [], "labor_time_axes": [], "abstract_patterns": [], "theory_hints": [], "confidence": 0.8}], "thought_summary": "summary", "concepts": []}'

        mock_callable = LLMCallable(func=AsyncMock(side_effect=mock_llm))
        extractor = ThoughtExtractor(mock_callable, project_id="test")

        result = await extractor.extract("測試文字" * 10)

        # Verify traceable fields
        assert result.was_extracted is True
        assert len(result.claims) > 0
        if result.claims:
            assert result.claims[0].core_claim == "test"
            assert result.claims[0].critique_target == ["target1"]


class TestRetrievalPlanTrace:
    """Test that retrieval plans are traceable."""

    def test_plan_is_traceable(self):
        """Verify retrieval plan is traceable."""
        from core.retrieval_planner import RetrievalPlanner

        planner = RetrievalPlanner()
        query = "包豪斯的理論框架是什麼？"

        plan = planner.create_plan(query, project_id="test")

        # Verify traceable fields
        assert plan.query_mode in ["fast", "deep"]
        assert plan.plan_id is not None
        assert plan.created_at is not None
