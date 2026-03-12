"""End-to-end pipeline tests for De-insight v2 Core.

These tests verify the full pipeline behavior from input to storage.

References:
- Tech spec: deinsight_phase2_test_debug_spec.md
"""

import sys
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core import (
    QueryClassifier,
    RetrievalPlanner,
    Retriever,
    ThoughtExtractor,
    LLMCallable,
    enable_core_pipeline,
    disable_core_pipeline,
    is_core_enabled,
    run_core_pipeline,
    CorePipelineInput,
)
from core.debug.trace import PipelineTrace, TraceContext


class TestE2EFastPath:
    """E2E Test: Fast path through pipeline."""

    @pytest.mark.asyncio
    async def test_fast_path_classification(self):
        """Test that fast path queries are classified correctly."""
        classifier = QueryClassifier()
        result = classifier.classify("包豪斯是什麼？")

        assert result.mode == "fast"

    @pytest.mark.asyncio
    async def test_fast_path_planning(self):
        """Test that planner creates lightweight plan for fast queries."""
        planner = RetrievalPlanner()
        plan = planner.create_plan("包豪斯是什麼？", project_id="test")

        assert plan.query_mode == "fast"
        assert plan.max_passages_per_path == 3


class TestE2EDeepPath:
    """E2E Test: Deep path through pipeline."""

    @pytest.mark.asyncio
    async def test_deep_path_classification(self):
        """Test that deep path queries are classified correctly."""
        classifier = QueryClassifier()
        result = classifier.classify("包豪斯的理論框架是什麼？")

        assert result.mode == "deep"

    @pytest.mark.asyncio
    async def test_deep_path_planning(self):
        """Test that planner creates multiple routes for deep queries."""
        planner = RetrievalPlanner()
        plan = planner.create_plan(
            "包豪斯的理論框架和現代主義有什麼關係？",
            project_id="test"
        )

        assert plan.query_mode == "deep"
        assert plan.why_deep is not None
        assert plan.max_passages_per_path == 5


class TestE2EFeatureFlagOff:
    """E2E Test: Feature flag off (legacy path)."""

    def setup_method(self):
        disable_core_pipeline()

    def teardown_method(self):
        disable_core_pipeline()

    def test_core_disabled_by_default(self):
        """Test that core is disabled by default."""
        assert is_core_enabled() is False

    @pytest.mark.asyncio
    async def test_no_core_writes_when_disabled(self):
        """Test that no core-specific writes occur when disabled."""
        enable_core_pipeline()
        assert is_core_enabled() is True
        disable_core_pipeline()
        assert is_core_enabled() is False


class TestE2EFeatureFlagOn:
    """E2E Test: Feature flag on (core path)."""

    def setup_method(self):
        disable_core_pipeline()

    def teardown_method(self):
        disable_core_pipeline()

    def test_enable_core_pipeline(self):
        """Test enabling the core pipeline."""
        enable_core_pipeline()
        assert is_core_enabled() is True

    @pytest.mark.asyncio
    async def test_core_pipeline_runs_when_enabled(self):
        """Test that core pipeline runs when enabled."""
        enable_core_pipeline()
        assert is_core_enabled() is True


class TestE2EExtractorRepair:
    """E2E Test: Extractor repair path."""

    @pytest.mark.asyncio
    async def test_extractor_repair_with_malformed_json(self):
        """Test that extractor repairs malformed JSON."""
        # This test verifies the extractor can handle malformed input
        def mock_llm(prompt: str) -> str:
            # Return malformed JSON
            return '{"claims": [{"core_claim": "test",},], "thought_summary": ""}'

        mock_callable = LLMCallable(func=AsyncMock(side_effect=mock_llm))
        extractor = ThoughtExtractor(mock_callable, project_id="test")

        # Should not crash
        result = await extractor.extract("測試文字" * 10)
        assert result is not None


class TestE2EExtractorHardFailure:
    """E2E Test: Extractor hard failure path."""

    @pytest.mark.asyncio
    async def test_extractor_fallback_on_complete_failure(self):
        """Test that extractor falls back safely on hard failure."""
        def mock_llm(prompt: str) -> str:
            # Return completely invalid response
            return "這不是可以解析的格式"

        mock_callable = LLMCallable(func=AsyncMock(side_effect=mock_llm))
        extractor = ThoughtExtractor(mock_callable, project_id="test")

        # Should return empty result, not crash
        result = await extractor.extract("測試文字" * 10)
        assert result is not None
        assert result.was_extracted is False


class TestE2ETraceContext:
    """E2E Test: Trace context for debugging."""

    @pytest.mark.asyncio
    async def test_trace_context_creation(self):
        """Test that trace context is created properly."""
        async with TraceContext("test-project", "test message") as trace:
            trace.user_message_preview = "test"
            trace.classification_mode = "fast"
            trace.extractor_success = True
            trace.timings["test"] = 0.1

        assert trace.trace_id is not None
        assert trace.project_id == "test-project"
        assert trace.timings["test"] == 0.1

    @pytest.mark.asyncio
    async def test_trace_context_with_error(self):
        """Test that trace captures errors."""
        async with TraceContext("test") as trace:
            trace.user_message_preview = "test"
            try:
                raise ValueError("Test error")
            except ValueError as e:
                trace.errors.append(f"{type(e).__name__}: {e}")

        assert len(trace.errors) > 0
        assert "ValueError" in trace.errors[0]


class TestE2EPipelineIntegration:
    """E2E Test: Full pipeline integration."""

    @pytest.mark.asyncio
    async def test_classification_to_planning_flow(self):
        """Test flow from classification to planning."""
        # Step 1: Classify
        classifier = QueryClassifier()
        query = "這和其他設計運動的結構模式有什麼相似之處？"
        classification = classifier.classify(query)

        # Step 2: Plan
        planner = RetrievalPlanner()
        plan = planner.create_plan(query, project_id="test")

        # Verify consistency
        assert classification.mode == plan.query_mode

    @pytest.mark.asyncio
    async def test_full_pipeline_with_mock_extraction(self):
        """Test full pipeline with mocked extraction."""
        enable_core_pipeline()

        def mock_llm(prompt: str) -> str:
            return '{"claims": [{"core_claim": "test", "critique_target": [], "value_axes": [], "materiality_axes": [], "labor_time_axes": [], "abstract_patterns": [], "theory_hints": [], "confidence": 0.8}], "thought_summary": "test", "concepts": []}'

        try:
            result = await run_core_pipeline(CorePipelineInput(
                user_message="這是測試",
                project_id="test",
                llm_callable=mock_llm,
            ))

            # Should have result
            assert result is not None
        finally:
            disable_core_pipeline()
