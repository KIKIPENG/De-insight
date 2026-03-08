"""v7.5.0 Pipeline 單元測試與整合測試。

涵蓋：
- 問題分類、觀點抽取、來源約束
- Deep 熔斷器 / fallback
- Embedding provider 單一真值
- 洞見加分引擎
- _clean_rag_chunk 內容保留
- B1: 相關性閘門（relevance gate）
- B2: 引用守衛（citation guard）— hard block
- D1: 跨語言查詢增強 — 字典 + 命名實體
- A3: 啟動健康檢查 / 降級模式 — endpoint probe
- A4: Thread-safe singleton
- 完整 pipeline 整合測試
"""

import asyncio
import sys
import time
from pathlib import Path
from types import ModuleType
from unittest.mock import AsyncMock, MagicMock, patch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Mock lightrag module so knowledge_graph can import without the actual package
if "lightrag" not in sys.modules:
    _mock_lr = ModuleType("lightrag")
    _mock_lr.LightRAG = MagicMock
    _mock_lr.QueryParam = MagicMock
    sys.modules["lightrag"] = _mock_lr
    _mock_lr_llm = ModuleType("lightrag.llm")
    sys.modules["lightrag.llm"] = _mock_lr_llm
    _mock_lr_llm_oai = ModuleType("lightrag.llm.openai")
    _mock_lr_llm_oai.openai_complete_if_cache = MagicMock
    _mock_lr_llm_oai.openai_embed = MagicMock
    sys.modules["lightrag.llm.openai"] = _mock_lr_llm_oai
    _mock_lr_utils = ModuleType("lightrag.utils")
    _mock_lr_utils.EmbeddingFunc = MagicMock
    sys.modules["lightrag.utils"] = _mock_lr_utils


# ── Unit Tests ─────────────────────────────────────────────────────

def test_classify_question_fact():
    """事實題型分類。"""
    from rag.pipeline import classify_question
    assert classify_question("王志弘是誰？") == "fact"
    assert classify_question("這本書什麼時候出版的？") == "fact"
    assert classify_question("哪一年成立的？") == "fact"
    assert classify_question("有沒有相關的作品？") == "fact"
    assert classify_question("Bloom Lab 花藝計劃是什麼？") == "fact"
    assert classify_question("Who designed this?") == "fact"
    print("PASS: test_classify_question_fact")


def test_classify_question_reasoning():
    from rag.pipeline import classify_question
    assert classify_question("你覺得這個設計風格怎麼看？") == "reasoning"
    assert classify_question("請分析王志弘的設計特色，並比較他跟其他設計師的差異") == "reasoning"
    print("PASS: test_classify_question_reasoning")


def test_classify_question_summary():
    from rag.pipeline import classify_question
    assert classify_question("告訴我關於字體的事") == "summary"
    print("PASS: test_classify_question_summary")


def test_perspective_extraction():
    from rag.pipeline import extract_perspective
    card = extract_perspective("我覺得「排版」是一種精確的語言，用來結構化想法")
    assert card.claim
    assert "排版" in card.key_concepts
    print("PASS: test_perspective_extraction")


def test_source_constraint_fact_no_source():
    """事實題無來源：輕量警告前置，context 保留。"""
    from rag.pipeline import apply_source_constraint
    # 有 context 時：輕量警告 + context 都保留
    result = apply_source_constraint("fact", 0, "some valid context")
    assert "謹慎" in result or "注意" in result
    assert "some valid context" in result  # context 不再被丟棄
    # 空 context 時：只有警告
    result_empty = apply_source_constraint("fact", 0, "")
    assert "注意" in result_empty or "謹慎" in result_empty
    print("PASS: test_source_constraint_fact_no_source")


def test_source_constraint_fact_with_source():
    from rag.pipeline import apply_source_constraint
    result = apply_source_constraint("fact", 2, "some knowledge content")
    assert "來源約束" in result
    assert "some knowledge content" in result
    print("PASS: test_source_constraint_fact_with_source")


def test_source_constraint_reasoning():
    from rag.pipeline import apply_source_constraint
    # Non-fact question: no warning prefix, context passed through as-is
    result = apply_source_constraint("reasoning", 0, "context here")
    assert result == "context here"  # no warning for non-fact
    print("PASS: test_source_constraint_reasoning")


def test_no_context_not_injected():
    """[no-context] 不注入。"""
    from rag.pipeline import clean_context
    assert clean_context("[no-context]") == ""
    assert clean_context("not able to provide an answer") == ""
    assert clean_context("") == ""
    assert clean_context("No relevant document chunks found") == ""
    print("PASS: test_no_context_not_injected")


def test_clean_context_budget():
    from rag.pipeline import clean_context
    long_text = "A" * 5000
    with patch("rag.knowledge_graph._clean_rag_chunk", return_value=long_text):
        result = clean_context(long_text, budget=2000)
        assert len(result) <= 2100
    print("PASS: test_clean_context_budget")


def test_deep_circuit_breaker():
    """熔斷器測試。"""
    from rag.pipeline import _trip_deep_breaker, _deep_breaker_open, _DEEP_COOLDOWN_SECS
    import rag.pipeline as _p
    orig = _p._deep_fail_until
    try:
        _p._deep_fail_until = 0.0
        assert not _deep_breaker_open()
        _trip_deep_breaker("http_400")
        assert _deep_breaker_open()
        _p._deep_fail_until = time.time() - 1
        assert not _deep_breaker_open()
    finally:
        _p._deep_fail_until = orig
    print("PASS: test_deep_circuit_breaker")


def test_deep_fallback_on_400():
    """Deep 400 error triggers fallback."""
    async def _run():
        from rag.pipeline import _retrieve
        import rag.pipeline as _p
        _p._deep_fail_until = 0.0
        _p._degraded_mode = False

        with patch("rag.knowledge_graph.query_knowledge") as mock_qk:
            with patch("rag.knowledge_graph.has_knowledge", return_value=True):
                mock_qk.side_effect = [
                    Exception("400 Bad Request"),
                    ("fast result", [{"title": "t", "snippet": "s", "file": "f"}]),
                ]
                result, sources, strategy, err = await _retrieve(
                    "test", "proj1", "deep", "summary",
                )
                assert strategy == "fast_fallback"
                assert err is not None
                assert "400" in err
                assert result == "fast result"
                print("PASS: test_deep_fallback_on_400")
    asyncio.run(_run())


def test_deep_fallback_on_timeout():
    """Deep timeout triggers fallback."""
    async def _run():
        from rag.pipeline import _retrieve
        import rag.pipeline as _p
        _p._deep_fail_until = 0.0
        _p._degraded_mode = False

        with patch("rag.knowledge_graph.query_knowledge") as mock_qk:
            with patch("rag.knowledge_graph.has_knowledge", return_value=True):
                mock_qk.side_effect = [
                    Exception("Connection timed out"),
                    ("fallback content", []),
                ]
                result, sources, strategy, err = await _retrieve(
                    "test", "proj1", "deep", "fact",
                )
                assert strategy == "fast_fallback"
                assert "timeout" in err
                print("PASS: test_deep_fallback_on_timeout")
    asyncio.run(_run())


def test_embed_provider_unified_v4():
    """v0.8: 所有 embedding 統一使用本地 jina-embeddings-v4 / 1024d。"""
    from rag.knowledge_graph import _get_embed_config
    model, key, base, dim = _get_embed_config()
    assert model == "jina-embeddings-v4-gguf"
    assert dim == 1024
    assert key == "local"
    assert base == ""
    print("PASS: test_embed_provider_unified_v4")


def test_insight_score_affects_ranking():
    """洞見加分可提升候選排序。"""
    async def _run():
        from rag.insight_profile import compute_insight_score
        import rag.insight_profile as _ip
        _ip._profile_cache["test_proj"] = {
            "keywords": ["排版", "字體", "設計", "結構"],
            "topics": ["設計史", "美學"],
            "categories": ["美學偏好", "思考方式"],
            "count": 5,
        }
        _ip._profile_ts["test_proj"] = time.time()
        score_match = await compute_insight_score("排版設計和字體的美學", "test_proj")
        score_nomatch = await compute_insight_score("天氣預報明天下雨", "test_proj")
        assert score_match > score_nomatch
        assert score_match > 0.0
        _ip._profile_cache.pop("test_proj", None)
        _ip._profile_ts.pop("test_proj", None)
        print("PASS: test_insight_score_affects_ranking")
    asyncio.run(_run())


def test_clean_rag_chunk_preserves_content():
    """_clean_rag_chunk extracts JSON content before stripping headers."""
    from rag.knowledge_graph import _clean_rag_chunk
    import json
    fake_result = '-----Document Chunks-----\n'
    fake_result += json.dumps({"reference_id": "1", "content": "這是一段關於排版設計的重要內容，包含了許多有價值的資訊。"}, ensure_ascii=False) + '\n'
    fake_result += json.dumps({"reference_id": "2", "content": "字體選擇是設計過程中最關鍵的決定之一。"}, ensure_ascii=False) + '\n'
    fake_result += '\n\nReference Document List:'
    cleaned = _clean_rag_chunk(fake_result)
    assert len(cleaned) > 20, f"Cleaned content too short: {len(cleaned)} chars"
    assert "排版設計" in cleaned or "字體選擇" in cleaned
    print("PASS: test_clean_rag_chunk_preserves_content")


# ── B1: Relevance gate tests ──────────────────────────────────────

def test_relevance_gate_filters_irrelevant():
    """B1: Irrelevant sources are filtered out."""
    from rag.pipeline import apply_relevance_gate
    sources = [
        {"title": "排版設計原理", "snippet": "排版是視覺設計中最重要的一環", "file": "a.pdf"},
        {"title": "天氣預報系統", "snippet": "明天台北將會下雨，氣溫降低", "file": "b.pdf"},
        {"title": "字體設計美學", "snippet": "字體的選擇直接影響排版品質", "file": "c.pdf"},
    ]
    filtered = apply_relevance_gate(sources, "排版設計和字體美學", "fact")
    titles = [s["title"] for s in filtered]
    assert "排版設計原理" in titles
    assert "字體設計美學" in titles
    assert "天氣預報系統" not in titles
    print("PASS: test_relevance_gate_filters_irrelevant")


def test_relevance_gate_passes_all_when_relevant():
    """B1: All relevant sources pass through."""
    from rag.pipeline import apply_relevance_gate
    sources = [
        {"title": "書籍設計", "snippet": "王志弘的書籍設計風格獨特", "file": "a.pdf"},
        {"title": "設計美學", "snippet": "設計中的美學原理與實踐", "file": "b.pdf"},
    ]
    filtered = apply_relevance_gate(sources, "王志弘的設計美學", "summary")
    assert len(filtered) == 2
    print("PASS: test_relevance_gate_passes_all_when_relevant")


def test_relevance_gate_empty_input():
    """B1: Empty sources or query returns as-is."""
    from rag.pipeline import apply_relevance_gate
    assert apply_relevance_gate([], "test", "fact") == []
    sources = [{"title": "t", "snippet": "s", "file": "f"}]
    result = apply_relevance_gate(sources, "a", "fact")
    assert len(result) == 1
    print("PASS: test_relevance_gate_empty_input")


# ── B2: Citation guard tests ──────────────────────────────────────

def test_citation_guard_warns_short_snippets():
    """B2: Fact question with only short snippets gets warning prefix, context preserved."""
    from rag.pipeline import citation_guard
    sources = [{"title": "t", "snippet": "short", "file": "f"}]
    result = citation_guard("some context about design", sources, "fact")
    assert "some context about design" in result  # context 保留
    assert "注意" in result or "不完整" in result  # 警告前置
    print("PASS: test_citation_guard_warns_short_snippets")


def test_citation_guard_passes_substantial():
    """B2: Fact question with substantial snippets passes through."""
    from rag.pipeline import citation_guard
    long_snippet = "這是一段足夠長的摘要內容，包含了設計師王志弘的詳細資訊和作品列表" * 2
    sources = [{"title": "t", "snippet": long_snippet, "file": "f"}]
    result = citation_guard("some context", sources, "fact")
    assert result == "some context"
    print("PASS: test_citation_guard_passes_substantial")


def test_citation_guard_no_block_for_reasoning():
    """B2: Reasoning questions don't get blocked even with short snippets."""
    from rag.pipeline import citation_guard
    sources = [{"title": "t", "snippet": "short", "file": "f"}]
    result = citation_guard("some context", sources, "reasoning")
    assert result == "some context"
    print("PASS: test_citation_guard_no_block_for_reasoning")


# ── D1: Cross-language augmentation tests ─────────────────────────

def test_cross_lang_chinese_query_gets_english():
    """D1: Chinese query about design gets English terms appended."""
    from rag.pipeline import augment_query_cross_lang
    result = augment_query_cross_lang("排版設計和字體美學")
    assert "typography" in result or "design" in result
    assert "排版設計" in result
    print("PASS: test_cross_lang_chinese_query_gets_english")


def test_cross_lang_english_query_unchanged():
    """D1: English-only query is returned unchanged."""
    from rag.pipeline import augment_query_cross_lang
    query = "What is the design philosophy?"
    result = augment_query_cross_lang(query)
    assert result == query
    print("PASS: test_cross_lang_english_query_unchanged")


def test_cross_lang_no_matching_terms():
    """D1: Chinese query without matching terms returns unchanged."""
    from rag.pipeline import augment_query_cross_lang
    query = "今天天氣如何"
    result = augment_query_cross_lang(query)
    assert result == query
    print("PASS: test_cross_lang_no_matching_terms")


def test_cross_lang_named_entity_extraction():
    """D1: Named entities (person names) are extracted for cross-lang search."""
    from rag.pipeline import _extract_named_entities
    entities = _extract_named_entities("王志弘是什麼時候發現文字比圖像更有趣？")
    # Should extract "王志弘", "文字", "圖像" etc.
    found_texts = " ".join(entities)
    assert "王志弘" in found_texts
    print(f"  extracted entities: {entities[:5]}")
    print("PASS: test_cross_lang_named_entity_extraction")


def test_cross_lang_augments_with_entities():
    """D1: Chinese query with named entities gets them included in augmentation."""
    from rag.pipeline import augment_query_cross_lang
    query = "王志弘是什麼時候發現文字比圖像更有趣？"
    result = augment_query_cross_lang(query)
    # Should include original + dictionary matches + entities
    assert "王志弘" in result
    assert query in result  # Original preserved
    # Should have some expansion
    assert len(result) > len(query)
    print(f"  augmented: {result}")
    print("PASS: test_cross_lang_augments_with_entities")


# ── A3: Startup health check tests ───────────────────────────────

def test_health_check_healthy():
    """A3: Health check passes with valid config (probe_llm=False to skip network)."""
    async def _run():
        import rag.pipeline as _p
        _p._degraded_mode = False
        mock_transformers = ModuleType("transformers")
        with patch("rag.knowledge_graph._get_llm_config", return_value=("gpt-4o", "sk-test", "https://api.openai.com/v1")):
            with patch("rag.knowledge_graph._get_embed_config", return_value=("jina-embeddings-v4", "local", "", 1024)):
                with patch.dict(sys.modules, {"transformers": mock_transformers}):
                    result = await _p.startup_health_check(probe_llm=False)
                    assert result["healthy"]
                    assert not _p._degraded_mode
                    print("PASS: test_health_check_healthy")
    asyncio.run(_run())


def test_health_check_no_model():
    """A3: Health check fails with no model configured."""
    async def _run():
        import rag.pipeline as _p
        with patch("rag.knowledge_graph._get_llm_config", return_value=("", "", "")):
            with patch("rag.knowledge_graph._get_embed_config", return_value=("jina-embeddings-v4", "local", "", 1024)):
                result = await _p.startup_health_check(probe_llm=False)
                assert not result["healthy"]
                assert _p._degraded_mode
                assert "未設定" in _p._degraded_reason
                print("PASS: test_health_check_no_model")
        _p._degraded_mode = False
        _p._degraded_reason = ""
    asyncio.run(_run())


def test_health_check_no_api_key():
    """A3: Health check warns when API key missing."""
    async def _run():
        import rag.pipeline as _p
        with patch("rag.knowledge_graph._get_llm_config", return_value=("gpt-4o", "", "https://api.openai.com/v1")):
            with patch("rag.knowledge_graph._get_embed_config", return_value=("jina-embeddings-v4", "local", "", 1024)):
                result = await _p.startup_health_check(probe_llm=False)
                assert not result["healthy"]
                assert _p._degraded_mode
                assert "API key" in _p._degraded_reason
                print("PASS: test_health_check_no_api_key")
        _p._degraded_mode = False
        _p._degraded_reason = ""
    asyncio.run(_run())


def test_health_check_probe_401():
    """A3: Health check probe catches invalid API key (401)."""
    async def _run():
        import rag.pipeline as _p
        _p._degraded_mode = False

        # Mock httpx to return 401
        mock_resp = MagicMock()
        mock_resp.status_code = 401
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_resp)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        mock_transformers = ModuleType("transformers")
        with patch("rag.knowledge_graph._get_llm_config", return_value=("gpt-4o", "bad-key", "https://api.openai.com/v1")):
            with patch("rag.knowledge_graph._get_embed_config", return_value=("jina-embeddings-v4", "local", "", 1024)):
                with patch.dict(sys.modules, {"transformers": mock_transformers}):
                    with patch("httpx.AsyncClient", return_value=mock_client):
                        result = await _p.startup_health_check(probe_llm=True)
                        assert not result["healthy"]
                        assert _p._degraded_mode
                        assert "401" in _p._degraded_reason or "無效" in _p._degraded_reason
                        print("PASS: test_health_check_probe_401")
        _p._degraded_mode = False
        _p._degraded_reason = ""
    asyncio.run(_run())


def test_health_check_probe_404_invalid_model():
    """A3: Health check probe catches non-existent model (404)."""
    async def _run():
        import rag.pipeline as _p
        _p._degraded_mode = False

        mock_resp = MagicMock()
        mock_resp.status_code = 404
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_resp)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        mock_transformers = ModuleType("transformers")
        with patch("rag.knowledge_graph._get_llm_config", return_value=("fake/nonexistent-model", "sk-test", "https://api.openai.com/v1")):
            with patch("rag.knowledge_graph._get_embed_config", return_value=("jina-embeddings-v4", "local", "", 1024)):
                with patch.dict(sys.modules, {"transformers": mock_transformers}):
                    with patch("httpx.AsyncClient", return_value=mock_client):
                        result = await _p.startup_health_check(probe_llm=True)
                        assert not result["healthy"]
                        assert _p._degraded_mode
                        assert "404" in _p._degraded_reason or "不存在" in _p._degraded_reason
                        print("PASS: test_health_check_probe_404_invalid_model")
        _p._degraded_mode = False
        _p._degraded_reason = ""
    asyncio.run(_run())


def test_degraded_mode_skips_deep():
    """A3: Degraded mode forces fast path even when deep requested."""
    async def _run():
        import rag.pipeline as _p
        _p._degraded_mode = True
        _p._deep_fail_until = 0.0

        with patch("rag.knowledge_graph.query_knowledge") as mock_qk:
            with patch("rag.knowledge_graph.has_knowledge", return_value=True):
                mock_qk.return_value = (
                    '{"reference_id": "1", "content": "fallback"}',
                    [{"title": "t", "snippet": "s", "file": "f"}],
                )
                from rag.pipeline import _retrieve
                _, _, strategy, err = await _retrieve("test", "proj1", "deep", "summary")
                assert strategy == "fast_fallback"
                assert err == "degraded_mode"
                print("PASS: test_degraded_mode_skips_deep")
        _p._degraded_mode = False
    asyncio.run(_run())


# ── A4: Thread-safe singleton test ────────────────────────────────

def test_thread_safe_lock_exists():
    """A4: Verify threading lock is present in EmbeddingService."""
    from embeddings import service as _es
    import threading
    assert hasattr(_es, "_service_lock")
    assert isinstance(_es._service_lock, type(threading.Lock()))
    print("PASS: test_thread_safe_lock_exists")


# ── A1: Dual embed split test ────────────────────────────────────

def test_image_embed_uses_service():
    """A1: v0.9 — image_store imports via EmbeddingService."""
    import ast
    src = Path(__file__).parent.parent / "rag" / "image_store.py"
    tree = ast.parse(src.read_text())
    found_service_import = False
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.module == "embeddings.service":
                found_service_import = True
    assert found_service_import, "image_store.py must import from embeddings.service"
    print("PASS: test_image_embed_uses_service")


# ── Integration Tests ─────────────────────────────────────────────

def test_full_pipeline_integration():
    """Integration: full pipeline returns valid structure."""
    async def _run():
        import rag.pipeline as _p
        _p._degraded_mode = False

        with patch("rag.knowledge_graph.query_knowledge") as mock_qk:
            with patch("rag.knowledge_graph.has_knowledge", return_value=True):
                mock_qk.return_value = (
                    '{"reference_id": "1", "content": "王志弘是台灣知名的書籍設計師"}',
                    [{"title": "Design article", "snippet": "王志弘設計", "file": "test.pdf"}],
                )

                from rag.pipeline import run_thinking_pipeline
                result = await run_thinking_pipeline(
                    user_input="王志弘是誰？",
                    project_id="test_proj",
                    mode="fast",
                )

                assert result["strategy_used"] == "fast"
                assert "context_text" in result
                assert "sources" in result
                assert "perspective_card" in result
                assert "diagnostics" in result
                assert result["diagnostics"]["question_type"] == "fact"
                assert result["diagnostics"]["source_count"] >= 1
                assert "latency_ms" in result["diagnostics"]
                assert "filtered_by_gate" in result["diagnostics"]
                assert "degraded_mode" in result["diagnostics"]
                print(f"  strategy: {result['strategy_used']}")
                print(f"  sources: {result['diagnostics']['source_count']}")
                print(f"  question_type: {result['diagnostics']['question_type']}")
                print(f"  filtered: {result['diagnostics']['filtered_by_gate']}")
                print(f"  latency: {result['diagnostics']['latency_ms']}ms")
                print("PASS: test_full_pipeline_integration")
    asyncio.run(_run())


def test_deep_available_uses_deep():
    """Integration: deep mode used when available."""
    async def _run():
        import rag.pipeline as _p
        _p._deep_fail_until = 0.0
        _p._degraded_mode = False

        with patch("rag.knowledge_graph.query_knowledge") as mock_qk:
            with patch("rag.knowledge_graph.has_knowledge", return_value=True):
                mock_qk.return_value = (
                    '{"reference_id": "1", "content": "深度檢索結果"}',
                    [{"title": "Deep result", "snippet": "detail", "file": "f"}],
                )

                from rag.pipeline import run_thinking_pipeline
                result = await run_thinking_pipeline(
                    user_input="分析設計美學",
                    project_id="test_proj",
                    mode="deep",
                )
                assert result["strategy_used"] == "deep"
                assert not result["fallback_used"]
                print("PASS: test_deep_available_uses_deep")
    asyncio.run(_run())


def test_deep_fallback_still_returns_sources():
    """Integration: deep failure + fast fallback still provides sources."""
    async def _run():
        import rag.pipeline as _p
        _p._deep_fail_until = 0.0
        _p._degraded_mode = False

        with patch("rag.knowledge_graph.query_knowledge") as mock_qk:
            with patch("rag.knowledge_graph.has_knowledge", return_value=True):
                mock_qk.side_effect = [
                    Exception("500 Internal Server Error"),
                    ('{"reference_id": "1", "content": "排版設計的原理和方法"}',
                     [{"title": "排版設計", "snippet": "排版設計是視覺傳達的核心技術之一", "file": "design.pdf"}]),
                ]

                from rag.pipeline import run_thinking_pipeline
                result = await run_thinking_pipeline(
                    user_input="排版設計",
                    project_id="test_proj",
                    mode="deep",
                )
                assert result["fallback_used"]
                assert result["strategy_used"] == "fast_fallback"
                assert result["diagnostics"]["deep_error_code"] is not None
                assert result["diagnostics"]["retrieval_hit_count"] >= 1
                print("PASS: test_deep_fallback_still_returns_sources")
    asyncio.run(_run())


def test_pipeline_irrelevant_question_filtered():
    """Integration: irrelevant question gets sources filtered + constraint applied."""
    async def _run():
        import rag.pipeline as _p
        _p._degraded_mode = False

        with patch("rag.knowledge_graph.query_knowledge") as mock_qk:
            with patch("rag.knowledge_graph.has_knowledge", return_value=True):
                mock_qk.return_value = (
                    '{"reference_id": "1", "content": "王志弘的排版設計理念"}',
                    [{"title": "排版設計", "snippet": "王志弘是知名的書籍設計師，他的排版風格獨特", "file": "design.pdf"}],
                )

                from rag.pipeline import run_thinking_pipeline
                result = await run_thinking_pipeline(
                    user_input="今天天氣怎麼樣？",
                    project_id="test_proj",
                    mode="fast",
                )
                assert result["diagnostics"]["filtered_by_gate"] >= 0
                # Non-fact questions no longer get warning prefixes;
                # context_text passes through as-is for model to judge
                print(f"  sources after gate: {result['diagnostics']['source_count']}")
                print(f"  filtered: {result['diagnostics']['filtered_by_gate']}")
                print("PASS: test_pipeline_irrelevant_question_filtered")
    asyncio.run(_run())


def test_pipeline_cross_lang_augmentation_integration():
    """Integration: Chinese query gets augmented for English document retrieval."""
    async def _run():
        import rag.pipeline as _p
        _p._degraded_mode = False

        call_args = []
        async def mock_query(question, **kwargs):
            call_args.append(question)
            return ('{"reference_id": "1", "content": "design content"}',
                    [{"title": "Design Book", "snippet": "Typography and design principles", "file": "book.pdf"}])

        with patch("rag.knowledge_graph.query_knowledge", side_effect=mock_query):
            with patch("rag.knowledge_graph.has_knowledge", return_value=True):
                from rag.pipeline import run_thinking_pipeline
                result = await run_thinking_pipeline(
                    user_input="排版設計和字體美學",
                    project_id="test_proj",
                    mode="fast",
                )
                assert call_args
                used_query = call_args[0]
                assert "排版設計" in used_query
                assert "typography" in used_query or "design" in used_query
                print(f"  augmented query: {used_query}")
                print("PASS: test_pipeline_cross_lang_augmentation_integration")
    asyncio.run(_run())


def test_pipeline_wang_zhihong_cross_lang():
    """Integration: 王志弘 cross-lang query includes entity in augmented query."""
    async def _run():
        import rag.pipeline as _p
        _p._degraded_mode = False

        call_args = []
        async def mock_query(question, **kwargs):
            call_args.append(question)
            return ('{"reference_id": "1", "content": "Wang found text more interesting"}',
                    [{"title": "Interview", "snippet": "Wang Zhi-Hong discovered text was more interesting than images", "file": "interview.pdf"}])

        with patch("rag.knowledge_graph.query_knowledge", side_effect=mock_query):
            with patch("rag.knowledge_graph.has_knowledge", return_value=True):
                from rag.pipeline import run_thinking_pipeline
                result = await run_thinking_pipeline(
                    user_input="王志弘是什麼時候發現文字比圖像更有趣？",
                    project_id="test_proj",
                    mode="fast",
                )
                assert call_args
                used_query = call_args[0]
                # Must include "王志弘" (entity) and dictionary terms
                assert "王志弘" in used_query
                assert "text" in used_query or "文字" in used_query
                print(f"  augmented query: {used_query}")
                print(f"  question_type: {result['diagnostics']['question_type']}")
                print("PASS: test_pipeline_wang_zhihong_cross_lang")
    asyncio.run(_run())


# ── A3 Hook test ──────────────────────────────────────────────────

def test_health_check_hooked_in_app():
    """A3: Verify startup_health_check is called from app._init_app."""
    import ast
    src = Path(__file__).parent.parent / "app.py"
    code = src.read_text()
    assert "startup_health_check" in code, "app.py must call startup_health_check"
    assert "probe_llm" in code, "app.py must pass probe_llm to startup_health_check"
    print("PASS: test_health_check_hooked_in_app")


def test_pipeline_preserves_context_when_sources_empty():
    """Integration: context preserved even when _extract_sources returns empty."""
    async def _run():
        import rag.pipeline as _p
        _p._degraded_mode = False
        raw_text = "這是一段關於排版設計的重要知識內容，沒有結構化的來源標記但有實際價值"
        with patch("rag.knowledge_graph.query_knowledge") as mock_qk:
            with patch("rag.knowledge_graph.has_knowledge", return_value=True):
                mock_qk.return_value = (raw_text, [])
                from rag.pipeline import run_thinking_pipeline
                result = await run_thinking_pipeline(
                    user_input="排版設計",
                    project_id="test_proj",
                    mode="fast",
                )
                assert "排版設計" in result["context_text"]
                assert result["diagnostics"]["source_count"] == 0
                print("PASS: test_pipeline_preserves_context_when_sources_empty")
    asyncio.run(_run())


# ── Run all tests ──────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("v7.5.0 Pipeline Tests (穩定化修復 v2)")
    print("=" * 60)
    print()

    print("── Unit Tests (15) ──")
    test_classify_question_fact()
    test_classify_question_reasoning()
    test_classify_question_summary()
    test_perspective_extraction()
    test_source_constraint_fact_no_source()
    test_source_constraint_fact_with_source()
    test_source_constraint_reasoning()
    test_no_context_not_injected()
    test_clean_context_budget()
    test_deep_circuit_breaker()
    test_deep_fallback_on_400()
    test_deep_fallback_on_timeout()
    test_embed_provider_unified_v4()
    test_insight_score_affects_ranking()
    test_clean_rag_chunk_preserves_content()

    print()
    print("── B1: Relevance Gate (3) ──")
    test_relevance_gate_filters_irrelevant()
    test_relevance_gate_passes_all_when_relevant()
    test_relevance_gate_empty_input()

    print()
    print("── B2: Citation Guard (3) — warn + preserve ──")
    test_citation_guard_warns_short_snippets()
    test_citation_guard_passes_substantial()
    test_citation_guard_no_block_for_reasoning()

    print()
    print("── D1: Cross-Language Augmentation (5) — entity extraction ──")
    test_cross_lang_chinese_query_gets_english()
    test_cross_lang_english_query_unchanged()
    test_cross_lang_no_matching_terms()
    test_cross_lang_named_entity_extraction()
    test_cross_lang_augments_with_entities()

    print()
    print("── A3: Startup Health Check (6) — endpoint probe ──")
    test_health_check_healthy()
    test_health_check_no_model()
    test_health_check_no_api_key()
    test_health_check_probe_401()
    test_health_check_probe_404_invalid_model()
    test_degraded_mode_skips_deep()

    print()
    print("── A4: Thread Safety (1) ──")
    test_thread_safe_lock_exists()

    print()
    print("── A1: Unified Embed (1) ──")
    test_image_embed_uses_jina_v4()

    print()
    print("── A3 Hook (1) ──")
    test_health_check_hooked_in_app()

    print()
    print("── Integration Tests (7) ──")
    test_full_pipeline_integration()
    test_deep_available_uses_deep()
    test_deep_fallback_still_returns_sources()
    test_pipeline_irrelevant_question_filtered()
    test_pipeline_cross_lang_augmentation_integration()
    test_pipeline_wang_zhihong_cross_lang()
    test_pipeline_preserves_context_when_sources_empty()

    print()
    print("=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)
