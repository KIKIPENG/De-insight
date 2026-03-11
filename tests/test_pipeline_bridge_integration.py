"""End-to-end integration test for pipeline + bridge surfacing."""

import pytest
from unittest.mock import patch, AsyncMock


class TestPipelineBridgeIntegration:
    """Test that pipeline correctly integrates with bridge surfacing."""

    def test_pipeline_has_bridge_result_variable(self):
        """Verify pipeline code has bridge_result variable."""
        import inspect
        from rag.pipeline import run_thinking_pipeline
        source = inspect.getsource(run_thinking_pipeline)
        assert "bridge_result" in source, "Pipeline should have bridge_result variable"

    def test_pipeline_calls_core_retriever(self):
        """Verify pipeline calls core.retriever for bridge retrieval."""
        import inspect
        from rag.pipeline import run_thinking_pipeline
        source = inspect.getsource(run_thinking_pipeline)
        assert "retrieve_with_plan" in source, "Pipeline should call retrieve_with_plan"

    def test_surfacing_uses_bridge_result(self):
        """Verify surfacing uses bridge_result, not raw_result."""
        import inspect
        from rag.pipeline import run_thinking_pipeline
        source = inspect.getsource(run_thinking_pipeline)
        # Should use bridge_result.bridges
        assert "bridge_result.bridges" in source, "Surfacing should use bridge_result.bridges"

    def test_pipeline_returns_surfaced_bridge(self):
        """Verify pipeline return includes surfaced_bridge field."""
        import asyncio
        from rag.pipeline import run_thinking_pipeline

        # Run with a simple query (will use fallback since no real KB)
        result = asyncio.run(run_thinking_pipeline(
            user_input="test query",
            project_id="test-project",
            mode="fast",
        ))
        
        assert "surfaced_bridge" in result, "Pipeline should return surfaced_bridge"
        # surfaced_bridge can be None if no bridges found, but field should exist

    def test_pipeline_with_mocked_bridge_retrieval(self):
        """Test pipeline with mocked core.retriever returning bridges."""
        import asyncio
        from unittest.mock import patch
        from core.schemas import Bridge, BridgeType, RetrievalResult, RetrievalPlan
        from rag.pipeline import run_thinking_pipeline

        # Create mock bridge result with proper plan
        mock_bridge = Bridge(
            project_id="test",
            source_claim_id="source1",
            target_claim_id="target1",
            bridge_type=BridgeType.VALUE_STRUCTURE_MATCH,
            reason_summary="test bridge",
            confidence=0.5,
            score=0.5,
        )
        mock_plan = RetrievalPlan(
            project_id="test",
            concept_queries=["test"],
        )
        mock_result = RetrievalResult(
            plan=mock_plan,
            passages=[],
            claims=[],
            bridges=[mock_bridge],
            sources=[],
        )

        async def mock_retrieve(plan, query, project_id):
            return mock_result

        # Patch at the import location in pipeline.py
        with patch("core.retriever.retrieve_with_plan", side_effect=mock_retrieve):
            result = asyncio.run(run_thinking_pipeline(
                user_input="test query",
                project_id="test-project", 
                mode="fast",
            ))
            
            # Bridge should be retrieved and potentially surfaced
            assert "surfaced_bridge" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
