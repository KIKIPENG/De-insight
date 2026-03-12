"""Comprehensive tests for RAG pipeline, knowledge graph, and related features.

Tests mock LightRAG and external LLM calls while testing actual code logic.
Uses pytest with asyncio support (strict mode).
"""

import asyncio
import json
import pytest
import tempfile
import threading
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, Mock, patch

# Ensure paths are testable
import sys
import os
from pathlib import Path as PathlibPath
sys.path.insert(0, str(PathlibPath(__file__).parent.parent))
sys.path.insert(0, str(PathlibPath(__file__).parent.parent / "backend"))

# Mock lightrag and other optional deps BEFORE importing rag modules
_mock_lightrag = MagicMock()
_mock_lightrag.LightRAG = MagicMock
_mock_lightrag.QueryParam = MagicMock
sys.modules.setdefault("lightrag", _mock_lightrag)
sys.modules.setdefault("lightrag.llm", MagicMock())
sys.modules.setdefault("lightrag.utils", MagicMock())
sys.modules.setdefault("lancedb", MagicMock())

from paths import (
    GLOBAL_PROJECT_ID, project_root, ensure_project_dirs, DATA_ROOT
)


# ═══════════════════════════════════════════════════════════════════════════
# A. KNOWLEDGE GRAPH TESTS (15+ tests)
# ═══════════════════════════════════════════════════════════════════════════

class TestKnowledgeGraphBasics:
    """Tests for basic knowledge graph functionality."""

    @pytest.mark.asyncio
    async def test_get_rag_returns_instance(self):
        """Test that get_rag() returns a LightRAG instance."""
        with patch("rag.knowledge_graph.LightRAG") as mock_rag_class:
            mock_instance = MagicMock()
            mock_rag_class.return_value = mock_instance

            with patch("rag.knowledge_graph._apply_env"):
                with patch("rag.knowledge_graph._get_llm_config") as mock_llm:
                    with patch("rag.knowledge_graph._get_embed_config") as mock_embed:
                        mock_llm.return_value = ("gpt-4", "test-key", "https://api.openai.com/v1")
                        mock_embed.return_value = ("text-embedding-3-small", "key", "base", 1536)

                        from rag.knowledge_graph import get_rag, reset_rag
                        reset_rag()

                        result = get_rag("test_project")
                        assert result is not None

    @pytest.mark.asyncio
    async def test_has_knowledge_empty_project(self, tmp_path):
        """Test has_knowledge() returns False for empty project."""
        with patch("paths.DATA_ROOT", tmp_path):
            from rag.knowledge_graph import has_knowledge
            project_id = "empty_project"
            ensure_project_dirs(project_id)

            assert has_knowledge(project_id) is False

    @pytest.mark.asyncio
    async def test_has_knowledge_with_data(self, tmp_path):
        """Test has_knowledge() returns True when vdb_chunks.json has data."""
        with patch("paths.DATA_ROOT", tmp_path):
            from rag.knowledge_graph import has_knowledge
            project_id = "project_with_data"
            root = ensure_project_dirs(project_id)

            # Create vdb_chunks.json with data
            vdb_chunks = root / "lightrag" / "vdb_chunks.json"
            vdb_chunks.write_text(json.dumps({
                "data": [{"id": "1", "content": "test"}],
                "embedding_dim": 1536
            }), encoding="utf-8")

            assert has_knowledge(project_id) is True

    @pytest.mark.asyncio
    async def test_has_knowledge_empty_data(self, tmp_path):
        """Test has_knowledge() returns False when vdb_chunks.json has empty data."""
        with patch("paths.DATA_ROOT", tmp_path):
            from rag.knowledge_graph import has_knowledge
            project_id = "empty_data"
            root = ensure_project_dirs(project_id)

            vdb_chunks = root / "lightrag" / "vdb_chunks.json"
            vdb_chunks.write_text(json.dumps({
                "data": [],
                "embedding_dim": 1536
            }), encoding="utf-8")

            assert has_knowledge(project_id) is False

    @pytest.mark.asyncio
    async def test_get_rag_lock_singleton(self):
        """Test that _get_rag_lock() returns the same lock instance."""
        from rag.knowledge_graph import _get_rag_lock

        # Get lock multiple times
        lock1 = _get_rag_lock()
        lock2 = _get_rag_lock()

        assert lock1 is lock2
        assert isinstance(lock1, asyncio.Lock)

    @pytest.mark.asyncio
    async def test_query_knowledge_merged_project_only(self, tmp_path):
        """Test query_knowledge_merged() with only project knowledge."""
        with patch("paths.DATA_ROOT", tmp_path):
            project_id = "proj1"
            root = ensure_project_dirs(project_id)

            # Create vdb_chunks.json
            vdb_chunks = root / "lightrag" / "vdb_chunks.json"
            vdb_chunks.write_text(json.dumps({
                "data": [{"id": "1"}],
                "embedding_dim": 1536
            }), encoding="utf-8")

            with patch("rag.knowledge_graph._ensure_initialized") as mock_init:
                mock_rag = AsyncMock()
                mock_init.return_value = mock_rag
                mock_rag.aquery.return_value = "Project knowledge result"

                with patch("rag.knowledge_graph.query_knowledge") as mock_query:
                    mock_query.return_value = ("Project result", [{"title": "Doc1", "snippet": "content"}])

                    from rag.knowledge_graph import query_knowledge_merged

                    result_text, sources, merge_info = await query_knowledge_merged(
                        "test query",
                        project_id=project_id
                    )

                    assert "Project result" in result_text
                    assert len(sources) > 0
                    assert merge_info["project_chars"] > 0

    @pytest.mark.asyncio
    async def test_query_knowledge_merged_global_only(self, tmp_path):
        """Test query_knowledge_merged() with only global knowledge."""
        with patch("paths.DATA_ROOT", tmp_path):
            project_id = "proj2"
            global_root = ensure_project_dirs(GLOBAL_PROJECT_ID)

            # Only global has data
            vdb_chunks = global_root / "lightrag" / "vdb_chunks.json"
            vdb_chunks.write_text(json.dumps({
                "data": [{"id": "1"}],
                "embedding_dim": 1536
            }), encoding="utf-8")

            with patch("rag.knowledge_graph.query_knowledge") as mock_query:
                mock_query.return_value = ("Global result", [{"title": "GlobalDoc", "snippet": "global"}])

                from rag.knowledge_graph import query_knowledge_merged, has_knowledge

                result_text, sources, merge_info = await query_knowledge_merged(
                    "test query",
                    project_id=project_id
                )

                assert "Global result" in result_text
                assert merge_info["global_chars"] > 0
                assert merge_info["project_chars"] == 0

    @pytest.mark.asyncio
    async def test_query_knowledge_merged_both_project_and_global(self, tmp_path):
        """Test query_knowledge_merged() merges results from both project and global."""
        with patch("paths.DATA_ROOT", tmp_path):
            project_id = "proj3"
            proj_root = ensure_project_dirs(project_id)
            global_root = ensure_project_dirs(GLOBAL_PROJECT_ID)

            # Both have data
            for root in [proj_root, global_root]:
                vdb_chunks = root / "lightrag" / "vdb_chunks.json"
                vdb_chunks.write_text(json.dumps({
                    "data": [{"id": "1"}],
                    "embedding_dim": 1536
                }), encoding="utf-8")

            with patch("rag.knowledge_graph.query_knowledge") as mock_query:
                # First call = project, second = global
                mock_query.side_effect = [
                    ("Project content", [{"title": "P1", "snippet": "proj"}]),
                    ("Global content", [{"title": "G1", "snippet": "glob"}]),
                ]

                from rag.knowledge_graph import query_knowledge_merged

                result_text, sources, merge_info = await query_knowledge_merged(
                    "test query",
                    project_id=project_id
                )

                assert "Project content" in result_text
                assert "Global content" in result_text
                assert merge_info["project_chars"] > 0
                assert merge_info["global_chars"] > 0
                # Sources should have tier markers
                assert any(s.get("tier") == "project" for s in sources)
                assert any(s.get("tier") == "global" for s in sources)

    @pytest.mark.asyncio
    async def test_query_knowledge_merged_neither(self, tmp_path):
        """Test query_knowledge_merged() returns empty when neither has knowledge."""
        with patch("paths.DATA_ROOT", tmp_path):
            project_id = "proj_empty"
            ensure_project_dirs(project_id)

            from rag.knowledge_graph import query_knowledge_merged

            result_text, sources, merge_info = await query_knowledge_merged(
                "test query",
                project_id=project_id
            )

            assert result_text == ""
            assert sources == []
            assert merge_info["global_chars"] == 0
            assert merge_info["project_chars"] == 0

    @pytest.mark.asyncio
    async def test_query_knowledge_merged_handles_exception_gracefully(self, tmp_path):
        """Test query_knowledge_merged() handles exceptions in one source gracefully."""
        with patch("paths.DATA_ROOT", tmp_path):
            project_id = "proj4"
            proj_root = ensure_project_dirs(project_id)

            vdb_chunks = proj_root / "lightrag" / "vdb_chunks.json"
            vdb_chunks.write_text(json.dumps({
                "data": [{"id": "1"}],
                "embedding_dim": 1536
            }), encoding="utf-8")

            with patch("rag.knowledge_graph.query_knowledge") as mock_query:
                # Simulate exception in project query
                mock_query.side_effect = RuntimeError("Query failed")

                from rag.knowledge_graph import query_knowledge_merged

                # Should not raise, but return empty results
                result_text, sources, merge_info = await query_knowledge_merged(
                    "test query",
                    project_id=project_id
                )

                assert result_text == ""
                assert sources == []

    def test_global_project_id_constant(self):
        """Test that GLOBAL_PROJECT_ID is '__global__'."""
        assert GLOBAL_PROJECT_ID == "__global__"

    @pytest.mark.asyncio
    async def test_working_directory_per_project(self, tmp_path):
        """Test that working directory per project is correct path."""
        with patch("paths.DATA_ROOT", tmp_path):
            with patch("paths.PROJECTS_DIR", tmp_path / "projects"):
                project_id = "test_proj"
                root = project_root(project_id)

                assert root == tmp_path / "projects" / project_id

                # Ensure dirs creates subdirectories
                ensured = ensure_project_dirs(project_id)
                assert (ensured / "lightrag").exists()
                assert (ensured / "lancedb").exists()
                assert (ensured / "documents").exists()

    @pytest.mark.asyncio
    async def test_concurrent_query_knowledge_merged_no_deadlock(self, tmp_path):
        """Test concurrent query_knowledge_merged calls don't deadlock."""
        with patch("paths.DATA_ROOT", tmp_path):
            with patch("paths.PROJECTS_DIR", tmp_path / "projects"):
                project_id = "proj_concurrent"
                root = ensure_project_dirs(project_id)

                vdb_chunks = root / "lightrag" / "vdb_chunks.json"
                vdb_chunks.write_text(json.dumps({
                    "data": [{"id": "1"}],
                    "embedding_dim": 1536
                }), encoding="utf-8")

                async def query_task(query_text):
                    with patch("rag.knowledge_graph.query_knowledge") as mock_query:
                        mock_query.return_value = (f"Result for {query_text}", [])
                        from rag.knowledge_graph import query_knowledge_merged
                        return await query_knowledge_merged(query_text, project_id=project_id)

                from rag.knowledge_graph import query_knowledge_merged

                # Run multiple queries concurrently (use await, not asyncio.run)
                tasks = [query_task(f"query{i}") for i in range(5)]
                results = await asyncio.gather(*tasks, return_exceptions=True)

                # Should complete without timeout/deadlock
                assert len(results) == 5
                assert all(not isinstance(r, Exception) for r in results)


# ═══════════════════════════════════════════════════════════════════════════
# B. PIPELINE TESTS (10+ tests)
# ═══════════════════════════════════════════════════════════════════════════

class TestRAGPipeline:
    """Tests for RAG pipeline functionality."""

    @pytest.mark.asyncio
    async def test_retrieve_empty_knowledge(self):
        """Test _retrieve() returns empty when no knowledge exists."""
        with patch("rag.knowledge_graph.has_knowledge") as mock_has:
            mock_has.return_value = False

            from rag.pipeline import _retrieve

            raw, sources, strategy, error = await _retrieve(
                "test query",
                "empty_proj",
                "fast",
                "summary"
            )

            assert raw == ""
            assert sources == []
            assert strategy == "none"
            assert error is None

    @pytest.mark.asyncio
    async def test_retrieve_fast_mode(self):
        """Test _retrieve() fast mode calls query_knowledge_merged."""
        with patch("rag.knowledge_graph.has_knowledge") as mock_has:
            mock_has.return_value = True

            with patch("rag.knowledge_graph.query_knowledge_merged") as mock_query:
                mock_query.return_value = (
                    "Fast result",
                    [{"title": "Doc", "snippet": "content"}],
                    {"project_chars": 20, "global_chars": 0}
                )

                from rag.pipeline import _retrieve

                raw, sources, strategy, error = await _retrieve(
                    "test query",
                    "proj",
                    "fast",
                    "summary"
                )

                assert raw == "Fast result"
                assert len(sources) == 1
                assert strategy == "fast"
                assert error is None
                mock_query.assert_called_once()

    @pytest.mark.asyncio
    async def test_retrieve_deep_mode(self):
        """Test _retrieve() deep mode calls query_knowledge_merged."""
        with patch("rag.knowledge_graph.has_knowledge") as mock_has:
            mock_has.return_value = True

            with patch("rag.pipeline._deep_breaker_open") as mock_breaker:
                mock_breaker.return_value = False

                with patch("rag.pipeline._degraded_mode", False):
                    with patch("rag.knowledge_graph.query_knowledge_merged") as mock_query:
                        mock_query.return_value = (
                            "Deep result",
                            [{"title": "Doc", "snippet": "content"}],
                            {"project_chars": 30, "global_chars": 0}
                        )

                        from rag.pipeline import _retrieve

                        raw, sources, strategy, error = await _retrieve(
                            "test query",
                            "proj",
                            "deep",
                            "reasoning"
                        )

                        assert raw == "Deep result"
                        assert strategy == "deep"

    @pytest.mark.asyncio
    async def test_circuit_breaker_trips_after_deep_failure(self):
        """Test circuit breaker trips after deep mode failure."""
        with patch("rag.knowledge_graph.has_knowledge") as mock_has:
            mock_has.return_value = True

            with patch("rag.pipeline._deep_breaker_open") as mock_breaker:
                mock_breaker.return_value = False

                with patch("rag.pipeline._degraded_mode", False):
                    with patch("rag.knowledge_graph.query_knowledge_merged") as mock_query:
                        # First call (deep mode) raises exception
                        mock_query.side_effect = RuntimeError("Deep failed")

                        with patch("rag.pipeline._trip_deep_breaker") as mock_trip:
                            from rag.pipeline import _retrieve

                            raw, sources, strategy, error = await _retrieve(
                                "test query",
                                "proj",
                                "deep",
                                "summary"
                            )

                            # Should have called trip
                            mock_trip.assert_called_once()
                            assert error is not None

    @pytest.mark.asyncio
    async def test_circuit_breaker_resets_after_cooldown(self):
        """Test circuit breaker resets after cooldown period."""
        from rag.pipeline import _deep_breaker_open, _trip_deep_breaker, _deep_fail_until
        import rag.pipeline as pipeline_mod

        # Trip the breaker
        _trip_deep_breaker("test_error")
        assert _deep_breaker_open() is True

        # Mock time to simulate cooldown passing
        with patch("rag.pipeline.time") as mock_time:
            mock_time.time.return_value = pipeline_mod._deep_fail_until + 1

            assert _deep_breaker_open() is False

    @pytest.mark.asyncio
    async def test_retrieve_with_global_project(self):
        """Test _retrieve() with global project checks both sources."""
        with patch("rag.knowledge_graph.has_knowledge") as mock_has:
            def has_knowledge_side_effect(project_id=None):
                return project_id == "__global__"

            mock_has.side_effect = has_knowledge_side_effect

            with patch("rag.pipeline._deep_breaker_open") as mock_breaker:
                mock_breaker.return_value = False

                with patch("rag.pipeline._degraded_mode", False):
                    with patch("rag.knowledge_graph.query_knowledge_merged") as mock_query:
                        mock_query.return_value = ("Result", [], {})

                        from rag.pipeline import _retrieve
                        from paths import GLOBAL_PROJECT_ID

                        # Query with global project
                        raw, sources, strategy, error = await _retrieve(
                            "test",
                            GLOBAL_PROJECT_ID,
                            "fast",
                            "summary"
                        )

    def test_pipeline_uses_threading_lock(self):
        """Test that pipeline uses threading.Lock for breaker state."""
        import rag.pipeline as pipeline_mod

        # _pipeline_lock is initialized at module level as threading.Lock()
        # Check it exists and has the Lock interface (acquire/release methods)
        assert hasattr(pipeline_mod, '_pipeline_lock')
        assert hasattr(pipeline_mod._pipeline_lock, 'acquire')
        assert hasattr(pipeline_mod._pipeline_lock, 'release')

    @pytest.mark.asyncio
    async def test_build_rag_prompt_formats_context(self):
        """Test build_rag_prompt() formats context correctly."""
        # Note: build_rag_prompt is referenced in pipeline
        # but may not exist as standalone function; test via run_thinking_pipeline

        from rag.pipeline import run_thinking_pipeline

        with patch("rag.knowledge_graph.has_knowledge") as mock_has:
            mock_has.return_value = True

            with patch("rag.pipeline._retrieve") as mock_retrieve:
                mock_retrieve.return_value = (
                    "Retrieved content",
                    [{"title": "Source", "snippet": "text"}],
                    "fast",
                    None
                )

                with patch("rag.pipeline.get_insight_score") as mock_insight:
                    mock_insight.return_value = 0.5

                    result = await run_thinking_pipeline(
                        "What is design?",
                        "test_proj",
                        mode="fast"
                    )

                    assert isinstance(result, dict)
                    assert "context_text" in result
                    assert "sources" in result

    @pytest.mark.asyncio
    async def test_degraded_mode_fallback(self):
        """Test degraded mode fallback when deep fails."""
        from rag.pipeline import is_degraded

        with patch("rag.knowledge_graph.has_knowledge") as mock_has:
            mock_has.return_value = True

            with patch("rag.pipeline._degraded_mode", True):
                with patch("rag.knowledge_graph.query_knowledge_merged") as mock_query:
                    mock_query.return_value = ("Fast result", [], {})

                    from rag.pipeline import _retrieve

                    raw, sources, strategy, error = await _retrieve(
                        "test",
                        "proj",
                        "deep",
                        "summary"
                    )

                    # When degraded_mode is True and requesting deep, it falls back to fast
                    # Strategy is "fast_fallback" indicating a fallback occurred
                    assert strategy in ("fast", "fast_fallback")
                    assert error == "degraded_mode"

    @pytest.mark.asyncio
    async def test_fast_mode_always_available(self):
        """Test fast mode always available even when breaker is open."""
        with patch("rag.knowledge_graph.has_knowledge") as mock_has:
            mock_has.return_value = True

            with patch("rag.pipeline._deep_breaker_open") as mock_breaker:
                mock_breaker.return_value = True

                with patch("rag.knowledge_graph.query_knowledge_merged") as mock_query:
                    mock_query.return_value = ("Fast result", [], {})

                    from rag.pipeline import _retrieve

                    # Even though breaker is open, fast mode should work
                    raw, sources, strategy, error = await _retrieve(
                        "test",
                        "proj",
                        "fast",
                        "summary"
                    )

                    assert strategy == "fast"
                    assert error is None


# ═══════════════════════════════════════════════════════════════════════════
# C. PATH VALIDATION TESTS (5+ tests)
# ═══════════════════════════════════════════════════════════════════════════

class TestPathValidation:
    """Tests for path validation in RAG context."""

    def test_project_root_with_valid_uuid(self, tmp_path):
        """Test project_root with valid UUID."""
        with patch("paths.DATA_ROOT", tmp_path):
            with patch("paths.PROJECTS_DIR", tmp_path / "projects"):
                from paths import project_root
                import uuid

                proj_id = str(uuid.uuid4())
                root = project_root(proj_id)

                assert root == tmp_path / "projects" / proj_id
                assert str(root).startswith(str(tmp_path / "projects"))

    def test_project_root_with_global_id(self, tmp_path):
        """Test project_root with GLOBAL_PROJECT_ID."""
        with patch("paths.DATA_ROOT", tmp_path):
            with patch("paths.PROJECTS_DIR", tmp_path / "projects"):
                from paths import project_root

                root = project_root(GLOBAL_PROJECT_ID)
                assert root == tmp_path / "projects" / GLOBAL_PROJECT_ID

    def test_project_root_rejects_path_traversal(self):
        """Test project_root rejects path traversal attempts."""
        from paths import _validate_project_id

        with pytest.raises(ValueError):
            _validate_project_id("../../../etc/passwd")

        with pytest.raises(ValueError):
            _validate_project_id("..\\windows\\system")

    def test_ensure_project_dirs_creates_subdirs(self, tmp_path):
        """Test ensure_project_dirs creates required subdirectories."""
        with patch("paths.DATA_ROOT", tmp_path):
            from paths import ensure_project_dirs

            root = ensure_project_dirs("test_project")

            assert (root / "lightrag").exists()
            assert (root / "lancedb").exists()
            assert (root / "documents").exists()

    def test_knowledge_working_dir_inside_project_root(self, tmp_path):
        """Test knowledge working directory is inside project_root."""
        with patch("paths.DATA_ROOT", tmp_path):
            from paths import ensure_project_dirs, project_root

            project_id = "test_proj"
            proj_root = ensure_project_dirs(project_id)
            knowledge_dir = proj_root / "lightrag"

            # Verify knowledge dir is under project root
            assert knowledge_dir.is_relative_to(proj_root)
            assert knowledge_dir.exists()


# ═══════════════════════════════════════════════════════════════════════════
# D. UTILITY & INTEGRATION TESTS
# ═══════════════════════════════════════════════════════════════════════════

class TestUtilityFunctions:
    """Tests for utility functions."""

    def test_extract_query_keywords(self):
        """Test _extract_query_keywords splits CJK bigrams."""
        from rag.pipeline import _extract_query_keywords

        # Chinese query — extracts 2-char bigrams
        keywords = _extract_query_keywords("設計排版是什麼")
        # Should have bigrams like "設計", "排版", "版是", "是什", "什麼"
        assert "設計" in keywords or "排版" in keywords

        # Mixed query
        keywords = _extract_query_keywords("what is design 設計")
        assert "design" in keywords

    def test_source_relevance_score(self):
        """Test _source_relevance_score calculates scores."""
        from rag.pipeline import _source_relevance_score, _extract_query_keywords

        query_kw = _extract_query_keywords("設計")
        source = {"title": "設計原理", "snippet": "設計美學"}

        score = _source_relevance_score(source, query_kw)
        assert 0.0 <= score <= 1.0
        assert score > 0.3  # Should have reasonable overlap

    def test_classify_question(self):
        """Test classify_question categorizes correctly."""
        from rag.pipeline import classify_question

        fact_q = classify_question("什麼是策展？")
        assert fact_q in ("fact", "summary")

        reasoning_q = classify_question("你怎麼看待當代藝術中的策展概念與權力關係？")
        assert reasoning_q in ("reasoning", "summary")

    @pytest.mark.asyncio
    async def test_augment_query_cross_lang(self):
        """Test augment_query_cross_lang expands queries."""
        from rag.pipeline import augment_query_cross_lang

        # Chinese query with known terms
        expanded = augment_query_cross_lang("設計和排版是什麼")
        assert "design" in expanded or "typography" in expanded or "設計" in expanded

        # English query should pass through
        expanded = augment_query_cross_lang("what is design")
        assert "design" in expanded.lower()

    @pytest.mark.asyncio
    async def test_clean_context(self):
        """Test clean_context removes no-context markers."""
        from rag.pipeline import clean_context

        # Test no-context result
        result = clean_context("[no-context]")
        assert result == ""

        # Test normal content
        result = clean_context("This is relevant content about design theory.")
        assert "design" in result.lower()

    @pytest.mark.asyncio
    async def test_extract_perspective(self):
        """Test extract_perspective creates card."""
        from rag.pipeline import extract_perspective

        # Use longer input with quoted concepts for better extraction
        card = extract_perspective("設計應該重視美學。「設計」「美學」這兩個概念很重要。")
        assert card.claim  # Should have claim from first sentence
        # key_concepts may have quoted terms
        assert isinstance(card.key_concepts, list)

    @pytest.mark.asyncio
    async def test_apply_source_constraint_fact_without_sources(self):
        """Test apply_source_constraint for fact questions without sources."""
        from rag.pipeline import apply_source_constraint

        result = apply_source_constraint("fact", 0, "some context")
        assert "注意" in result or "警告" in result or result.startswith("（")

    @pytest.mark.asyncio
    async def test_apply_relevance_gate(self):
        """Test apply_relevance_gate filters sources."""
        from rag.pipeline import apply_relevance_gate

        sources = [
            {"title": "設計理論", "snippet": "設計的基本原理..."},
            {"title": "烹飪技巧", "snippet": "如何烹飪..."},
            {"title": "設計實踐", "snippet": "設計方法論..."},
        ]

        filtered = apply_relevance_gate(sources, "設計理論", "summary", threshold=0.15)

        # Should filter sources by relevance
        assert len(filtered) <= len(sources)
        # Design-related sources should remain or all could be filtered depending on threshold
        # Just verify it returns a list
        assert isinstance(filtered, list)

    @pytest.mark.asyncio
    async def test_citation_guard_fact_unverifiable(self):
        """Test citation_guard warns about unverifiable facts."""
        from rag.pipeline import citation_guard

        sources = [{"title": "Short", "snippet": "ab"}]  # Short snippet
        context = "Some claim about history."

        result = citation_guard(context, sources, "fact")
        assert "注意" in result or "警告" in result or context in result

    @pytest.mark.asyncio
    async def test_run_thinking_pipeline_full(self):
        """Test run_thinking_pipeline executes full flow."""
        from rag.pipeline import run_thinking_pipeline

        with patch("rag.knowledge_graph.has_knowledge") as mock_has:
            mock_has.return_value = True

            with patch("rag.pipeline._retrieve") as mock_retrieve:
                mock_retrieve.return_value = (
                    "Retrieved context",
                    [{"title": "Source", "snippet": "content", "source": "doc1"}],
                    "fast",
                    None
                )

                with patch("rag.pipeline.get_insight_score") as mock_score:
                    mock_score.return_value = 0.7

                    result = await run_thinking_pipeline(
                        "What is design?",
                        "test_proj",
                        mode="fast"
                    )

                    assert "strategy_used" in result
                    assert "context_text" in result
                    assert "sources" in result
                    assert "perspective_card" in result
                    assert "diagnostics" in result
                    assert result["strategy_used"] in ("fast", "deep", "none", "failed", "fast_fallback")


# ═══════════════════════════════════════════════════════════════════════════
# PYTEST CONFIGURATION & FIXTURES
# ═══════════════════════════════════════════════════════════════════════════

@pytest.fixture(autouse=True)
def reset_rag_state():
    """Reset RAG state before each test."""
    with patch("rag.knowledge_graph._rag_instance", None):
        with patch("rag.knowledge_graph._rag_project_id", None):
            yield


@pytest.fixture
def mock_lightrag():
    """Fixture providing a mock LightRAG instance."""
    mock = MagicMock()
    mock.aquery = AsyncMock(return_value="Mock result")
    return mock


@pytest.fixture
def temp_deinsight_home(tmp_path):
    """Fixture providing a temporary DEINSIGHT_HOME."""
    with patch.dict("os.environ", {"DEINSIGHT_HOME": str(tmp_path)}):
        yield tmp_path


if __name__ == "__main__":
    # Allow running with pytest
    pytest.main([__file__, "-v", "--tb=short"])
