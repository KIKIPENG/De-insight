"""Jina Reader web fetch 測試 — 覆蓋 insert_url 的 reader/fallback/legacy 路徑。"""

import asyncio
import os
import sys
import tempfile
from pathlib import Path
from types import ModuleType
from unittest.mock import AsyncMock, MagicMock, patch

# Project root
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

os.environ.setdefault("DEINSIGHT_HOME", tempfile.mkdtemp())
os.environ.setdefault("DEINSIGHT_DATA_VERSION", "test_jina")

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


# ── Helpers ──────────────────────────────────────────────────────────

def _run(coro):
    """Run async test in sync context."""
    return asyncio.run(coro)


def _make_html_response(html: str = "<html><title>Test</title><body>Hello World</body></html>"):
    """Create a mock httpx.Response for HTML content."""
    resp = MagicMock()
    resp.status_code = 200
    resp.headers = {"content-type": "text/html; charset=utf-8"}
    resp.text = html
    resp.content = html.encode("utf-8")
    resp.raise_for_status = MagicMock()
    return resp


def _make_pdf_response():
    """Create a mock httpx.Response for PDF content."""
    resp = MagicMock()
    resp.status_code = 200
    resp.headers = {"content-type": "application/pdf"}
    resp.text = ""
    resp.content = b"%PDF-1.4 fake content"
    resp.raise_for_status = MagicMock()
    return resp


# ── A. Unit Tests ────────────────────────────────────────────────────


def test_insert_url_reader_success():
    """mock reader 回傳有效正文 → _llm_clean_web_content 未被呼叫，fetch_method=jina_reader。"""
    reader_text = "這是一篇很長的正文內容，超過五十個字符，用來測試 Jina Reader 是否成功抓取了網頁的正文內容並直接匯入知識庫。"

    async def run():
        with patch("settings.load_env", return_value={"WEB_FETCH_PROVIDER": "auto"}), \
             patch("rag.knowledge_graph.load_env", return_value={"WEB_FETCH_PROVIDER": "auto"}), \
             patch("rag.knowledge_graph._fetch_with_jina_reader", new_callable=AsyncMock) as mock_reader, \
             patch("rag.knowledge_graph._llm_clean_web_content", new_callable=AsyncMock) as mock_legacy, \
             patch("rag.knowledge_graph.insert_text", new_callable=AsyncMock) as mock_insert, \
             patch("httpx.AsyncClient") as mock_client_cls:

            # Mock HTTP client for initial URL fetch
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.get = AsyncMock(return_value=_make_html_response())
            mock_client_cls.return_value = mock_client

            # Mock reader success
            mock_reader.return_value = (reader_text, {"status_code": 200, "latency_ms": 150, "reader_url": "https://r.jina.ai/https://example.com"})

            # Mock insert_text
            mock_insert.return_value = ""

            from rag.knowledge_graph import insert_url
            result = await insert_url("https://example.com", project_id="default")

            assert result["fetch_method"] == "jina_reader", f"Expected jina_reader, got {result['fetch_method']}"
            mock_legacy.assert_not_called()
            mock_reader.assert_called_once()

    _run(run())
    print("  PASS  test_insert_url_reader_success")


def test_insert_url_reader_fallback_to_legacy():
    """mock reader 失敗，legacy 成功 → fetch_method=legacy_html_clean。"""
    legacy_text = "這是 legacy 路徑清理後的正文內容，用來測試當 Jina Reader 失敗時是否正確回退到 legacy 清理流程。"

    async def run():
        with patch("settings.load_env", return_value={"WEB_FETCH_PROVIDER": "auto"}), \
             patch("rag.knowledge_graph.load_env", return_value={"WEB_FETCH_PROVIDER": "auto"}), \
             patch("rag.knowledge_graph._fetch_with_jina_reader", new_callable=AsyncMock) as mock_reader, \
             patch("rag.knowledge_graph._llm_clean_web_content", new_callable=AsyncMock) as mock_legacy, \
             patch("rag.knowledge_graph.insert_text", new_callable=AsyncMock) as mock_insert, \
             patch("httpx.AsyncClient") as mock_client_cls:

            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.get = AsyncMock(return_value=_make_html_response())
            mock_client_cls.return_value = mock_client

            # Reader fails
            mock_reader.side_effect = RuntimeError("Jina Reader HTTP 503")
            # Legacy succeeds
            mock_legacy.return_value = legacy_text
            mock_insert.return_value = ""

            from rag.knowledge_graph import insert_url
            result = await insert_url("https://example.com", project_id="default")

            assert result["fetch_method"] == "legacy_html_clean", f"Expected legacy_html_clean, got {result['fetch_method']}"
            mock_legacy.assert_called_once()
            assert "fallback_reason" in result

    _run(run())
    print("  PASS  test_insert_url_reader_fallback_to_legacy")


def test_insert_url_reader_short_text_fallback():
    """reader 回傳 <50 字，視為失敗並回退 legacy。"""
    short_text = "太短了"
    legacy_text = "這是 legacy 路徑清理後的正文內容，用來測試當 Jina Reader 回傳太短的文字時是否正確回退到 legacy 清理流程。"

    async def run():
        with patch("settings.load_env", return_value={"WEB_FETCH_PROVIDER": "auto"}), \
             patch("rag.knowledge_graph.load_env", return_value={"WEB_FETCH_PROVIDER": "auto"}), \
             patch("rag.knowledge_graph._fetch_with_jina_reader", new_callable=AsyncMock) as mock_reader, \
             patch("rag.knowledge_graph._llm_clean_web_content", new_callable=AsyncMock) as mock_legacy, \
             patch("rag.knowledge_graph.insert_text", new_callable=AsyncMock) as mock_insert, \
             patch("httpx.AsyncClient") as mock_client_cls:

            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.get = AsyncMock(return_value=_make_html_response())
            mock_client_cls.return_value = mock_client

            # Reader raises because text < 50
            mock_reader.side_effect = RuntimeError("Jina Reader 回傳內容太短（3 字）for https://example.com")
            mock_legacy.return_value = legacy_text
            mock_insert.return_value = ""

            from rag.knowledge_graph import insert_url
            result = await insert_url("https://example.com", project_id="default")

            assert result["fetch_method"] == "legacy_html_clean"
            mock_legacy.assert_called_once()

    _run(run())
    print("  PASS  test_insert_url_reader_short_text_fallback")


def test_insert_url_reader_only_mode_fails():
    """WEB_FETCH_PROVIDER=reader 且 reader 失敗 → raise 且錯誤可讀。"""

    async def run():
        with patch("settings.load_env", return_value={"WEB_FETCH_PROVIDER": "reader"}), \
             patch("rag.knowledge_graph.load_env", return_value={"WEB_FETCH_PROVIDER": "reader"}), \
             patch("rag.knowledge_graph._fetch_with_jina_reader", new_callable=AsyncMock) as mock_reader, \
             patch("rag.knowledge_graph._llm_clean_web_content", new_callable=AsyncMock) as mock_legacy, \
             patch("httpx.AsyncClient") as mock_client_cls:

            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.get = AsyncMock(return_value=_make_html_response())
            mock_client_cls.return_value = mock_client

            mock_reader.side_effect = RuntimeError("Jina Reader HTTP 500")

            from rag.knowledge_graph import insert_url
            try:
                await insert_url("https://example.com", project_id="default")
                assert False, "Should have raised RuntimeError"
            except RuntimeError as e:
                assert "WEB_FETCH_PROVIDER=reader" in str(e), f"Error message not descriptive: {e}"
                assert "不回退" in str(e)

            mock_legacy.assert_not_called()

    _run(run())
    print("  PASS  test_insert_url_reader_only_mode_fails")


def test_insert_url_legacy_mode_skips_reader():
    """WEB_FETCH_PROVIDER=legacy → reader 不被呼叫。"""
    legacy_text = "這是 legacy 路徑清理後的正文內容，用來測試當設為 legacy 模式時是否完全跳過 Jina Reader 直接走原有清理流程。"

    async def run():
        with patch("settings.load_env", return_value={"WEB_FETCH_PROVIDER": "legacy"}), \
             patch("rag.knowledge_graph.load_env", return_value={"WEB_FETCH_PROVIDER": "legacy"}), \
             patch("rag.knowledge_graph._fetch_with_jina_reader", new_callable=AsyncMock) as mock_reader, \
             patch("rag.knowledge_graph._llm_clean_web_content", new_callable=AsyncMock) as mock_legacy, \
             patch("rag.knowledge_graph.insert_text", new_callable=AsyncMock) as mock_insert, \
             patch("httpx.AsyncClient") as mock_client_cls:

            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.get = AsyncMock(return_value=_make_html_response())
            mock_client_cls.return_value = mock_client

            mock_legacy.return_value = legacy_text
            mock_insert.return_value = ""

            from rag.knowledge_graph import insert_url
            result = await insert_url("https://example.com", project_id="default")

            assert result["fetch_method"] == "legacy_html_clean"
            mock_reader.assert_not_called()

    _run(run())
    print("  PASS  test_insert_url_legacy_mode_skips_reader")


# ── B. Integration Tests ─────────────────────────────────────────────


def test_insert_url_preserves_pdf_path():
    """URL 指向 PDF 時仍走 insert_pdf，fetch_method=pdf。"""

    async def run():
        with patch("settings.load_env", return_value={"WEB_FETCH_PROVIDER": "auto"}), \
             patch("rag.knowledge_graph.load_env", return_value={"WEB_FETCH_PROVIDER": "auto"}), \
             patch("rag.knowledge_graph.insert_pdf", new_callable=AsyncMock) as mock_pdf, \
             patch("rag.knowledge_graph._fetch_with_jina_reader", new_callable=AsyncMock) as mock_reader, \
             patch("httpx.AsyncClient") as mock_client_cls:

            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.get = AsyncMock(return_value=_make_pdf_response())
            mock_client_cls.return_value = mock_client

            mock_pdf.return_value = {"title": "test.pdf", "page_count": 3, "file_size": 1000}

            from rag.knowledge_graph import insert_url
            result = await insert_url("https://example.com/paper.pdf", project_id="default")

            assert result["fetch_method"] == "pdf"
            mock_pdf.assert_called_once()
            mock_reader.assert_not_called()

    _run(run())
    print("  PASS  test_insert_url_preserves_pdf_path")


def test_insert_url_result_shape():
    """檢查 result 欄位完整（含 fetch_method）。"""
    reader_text = "這是一篇很長的正文內容，超過五十個字符，用來測試結果欄位是否完整包含所有必要的 fetch_method 等欄位資訊。"

    async def run():
        with patch("settings.load_env", return_value={"WEB_FETCH_PROVIDER": "auto"}), \
             patch("rag.knowledge_graph.load_env", return_value={"WEB_FETCH_PROVIDER": "auto"}), \
             patch("rag.knowledge_graph._fetch_with_jina_reader", new_callable=AsyncMock) as mock_reader, \
             patch("rag.knowledge_graph.insert_text", new_callable=AsyncMock) as mock_insert, \
             patch("httpx.AsyncClient") as mock_client_cls:

            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.get = AsyncMock(return_value=_make_html_response(
                "<html><title>測試頁面</title><body>content</body></html>"
            ))
            mock_client_cls.return_value = mock_client

            mock_reader.return_value = (reader_text, {"status_code": 200, "latency_ms": 100, "reader_url": "https://r.jina.ai/x"})
            mock_insert.return_value = ""

            from rag.knowledge_graph import insert_url
            result = await insert_url("https://example.com/article", project_id="default")

            # 必須有這些欄位
            assert "title" in result, "Missing 'title'"
            assert "page_count" in result, "Missing 'page_count'"
            assert "file_size" in result, "Missing 'file_size'"
            assert "fetch_method" in result, "Missing 'fetch_method'"
            assert result["fetch_method"] in ("jina_reader", "legacy_html_clean", "pdf"), \
                f"Unexpected fetch_method: {result['fetch_method']}"

    _run(run())
    print("  PASS  test_insert_url_result_shape")


def test_fetch_with_jina_reader_unit():
    """直接測試 _fetch_with_jina_reader 的成功/失敗邏輯。"""
    from rag.knowledge_graph import _fetch_with_jina_reader

    # Success case
    async def run_success():
        good_text = "A" * 100  # > 50 chars

        with patch("httpx.AsyncClient") as mock_cls:
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.text = good_text

            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.get = AsyncMock(return_value=mock_resp)
            mock_cls.return_value = mock_client

            text, meta = await _fetch_with_jina_reader("https://example.com")
            assert len(text) >= 50
            assert meta["status_code"] == 200
            assert "reader_url" in meta

    _run(run_success())

    # Short text → error
    async def run_short():
        with patch("httpx.AsyncClient") as mock_cls:
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.text = "短"

            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.get = AsyncMock(return_value=mock_resp)
            mock_cls.return_value = mock_client

            try:
                await _fetch_with_jina_reader("https://example.com")
                assert False, "Should have raised"
            except RuntimeError as e:
                assert "太短" in str(e)

    _run(run_short())

    # HTTP error → error
    async def run_http_err():
        with patch("httpx.AsyncClient") as mock_cls:
            mock_resp = MagicMock()
            mock_resp.status_code = 503
            mock_resp.text = "Service Unavailable"

            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.get = AsyncMock(return_value=mock_resp)
            mock_cls.return_value = mock_client

            try:
                await _fetch_with_jina_reader("https://example.com")
                assert False, "Should have raised"
            except RuntimeError as e:
                assert "503" in str(e)

    _run(run_http_err())

    print("  PASS  test_fetch_with_jina_reader_unit")


# ── Runner ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    passed = 0
    failed = 0
    for name, func in list(globals().items()):
        if name.startswith("test_") and callable(func):
            try:
                func()
                passed += 1
            except Exception as e:
                print(f"  FAIL  {name}: {e}")
                import traceback
                traceback.print_exc()
                failed += 1

    print(f"\n{'='*50}")
    print(f"Results: {passed} passed, {failed} failed")
    if failed > 0:
        sys.exit(1)
