"""v0.9.3 測試：_quick_llm_call retry 邏輯 + evolution 觸發。"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

import pytest
from unittest.mock import AsyncMock, patch, MagicMock


# ── T1: _extract_retry_after 單元測試 ──────────────────────────

class TestExtractRetryAfter:
    """測試 _extract_retry_after 靜態方法。"""

    def _get_method(self):
        from mixins.chat import ChatMixin
        return ChatMixin._extract_retry_after

    def test_retry_after_colon(self):
        exc = Exception("Rate limited. Retry-After: 30")
        assert self._get_method()(exc) == 30.0

    def test_retry_after_equals(self):
        exc = Exception("retry_after=15.5 seconds")
        assert self._get_method()(exc) == 15.5

    def test_retry_after_hyphen(self):
        exc = Exception("Please retry-after 60 seconds")
        assert self._get_method()(exc) == 60.0

    def test_no_retry_after(self):
        exc = Exception("Something went wrong")
        assert self._get_method()(exc) is None

    def test_empty_message(self):
        exc = Exception("")
        assert self._get_method()(exc) is None


# ── T2: _quick_llm_call retry 邏輯 ────────────────────────────

class TestQuickLlmCallRetry:
    """測試 _quick_llm_call 的重試邏輯。"""

    def _make_mixin(self):
        """建立一個最小的 ChatMixin 實例來測試。"""
        from mixins.chat import ChatMixin
        obj = object.__new__(ChatMixin)
        obj.log = MagicMock()
        return obj

    @pytest.mark.asyncio
    async def test_success_first_try(self):
        mixin = self._make_mixin()
        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock()]
        mock_resp.choices[0].message.content = "result"

        with patch("mixins.chat.load_env", return_value={"LLM_MODEL": "gemini/gemini-2.5-flash"}):
            with patch("litellm.acompletion", new_callable=AsyncMock, return_value=mock_resp):
                result = await mixin._quick_llm_call("test prompt")
                assert result == "result"

    @pytest.mark.asyncio
    async def test_retry_on_rate_limit(self):
        from rag.rate_guard import RateLimitError
        mixin = self._make_mixin()
        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock()]
        mock_resp.choices[0].message.content = "ok"

        with patch("mixins.chat.load_env", return_value={"LLM_MODEL": "gemini/gemini-2.5-flash"}):
            with patch("litellm.acompletion", new_callable=AsyncMock,
                       side_effect=[RateLimitError("429 rate limit"), mock_resp]):
                result = await mixin._quick_llm_call("test", max_retries=3)
                assert result == "ok"

    @pytest.mark.asyncio
    async def test_retry_on_generic_error(self):
        mixin = self._make_mixin()
        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock()]
        mock_resp.choices[0].message.content = "recovered"

        with patch("mixins.chat.load_env", return_value={"LLM_MODEL": "gemini/gemini-2.5-flash"}):
            with patch("litellm.acompletion", new_callable=AsyncMock,
                       side_effect=[ConnectionError("timeout"), mock_resp]):
                result = await mixin._quick_llm_call("test", max_retries=3)
                assert result == "recovered"

    @pytest.mark.asyncio
    async def test_all_retries_exhausted(self):
        mixin = self._make_mixin()

        with patch("mixins.chat.load_env", return_value={"LLM_MODEL": "gemini/gemini-2.5-flash"}):
            with patch("litellm.acompletion", new_callable=AsyncMock,
                       side_effect=ConnectionError("always fails")):
                with pytest.raises(ConnectionError, match="always fails"):
                    await mixin._quick_llm_call("test", max_retries=2)

    @pytest.mark.asyncio
    async def test_rate_limit_all_retries_exhausted(self):
        from rag.rate_guard import RateLimitError
        mixin = self._make_mixin()

        with patch("mixins.chat.load_env", return_value={"LLM_MODEL": "gemini/gemini-2.5-flash"}):
            with patch("litellm.acompletion", new_callable=AsyncMock,
                       side_effect=RateLimitError("429")):
                with pytest.raises(RateLimitError):
                    await mixin._quick_llm_call("test", max_retries=2)

    @pytest.mark.asyncio
    async def test_retry_after_extracted(self):
        """rate limit 有 retry-after 時應使用該值。"""
        from rag.rate_guard import RateLimitError
        mixin = self._make_mixin()
        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock()]
        mock_resp.choices[0].message.content = "ok"

        with patch("mixins.chat.load_env", return_value={"LLM_MODEL": "gemini/gemini-2.5-flash"}):
            with patch("litellm.acompletion", new_callable=AsyncMock,
                       side_effect=[RateLimitError("Retry-After: 1"), mock_resp]):
                with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
                    result = await mixin._quick_llm_call("test", max_retries=3)
                    assert result == "ok"
                    # Should have used retry-after value of 1
                    mock_sleep.assert_called_once_with(1.0)


# ── T3: evolution 觸發測試 ─────────────────────────────────────

class TestEvolutionTrigger:
    """測試 _save_confirmed_memories 中的 evolution 觸發邏輯。"""

    def test_evolution_wiring_exists(self):
        """確認 _check_insight_evolution 方法存在。"""
        from mixins.memory import MemoryMixin
        assert hasattr(MemoryMixin, "_check_insight_evolution")

    def test_save_confirmed_memories_has_evolution_check(self):
        """確認 _save_confirmed_memories 中有呼叫 evolution 檢查。"""
        import inspect
        from mixins.memory import MemoryMixin
        source = inspect.getsource(MemoryMixin._save_confirmed_memories)
        assert "_check_insight_evolution" in source
        assert 'type' in source and 'insight' in source

    def test_check_evolution_passes_db_path(self):
        """確認 _check_insight_evolution 有傳遞 db_path 給 check_for_evolution。"""
        import inspect
        from mixins.memory import MemoryMixin
        source = inspect.getsource(MemoryMixin._check_insight_evolution)
        assert "db_path" in source
        assert "_memory_db_path" in source

    def test_check_for_evolution_accepts_db_path(self):
        """確認 check_for_evolution 接受 db_path 參數。"""
        import inspect
        from memory.thought_tracker import check_for_evolution
        sig = inspect.signature(check_for_evolution)
        assert "db_path" in sig.parameters

    def test_exception_handler_has_debug_logging(self):
        """確認例外處理不是靜默 pass，有 DEBUG 模式 logging。"""
        import inspect
        from mixins.memory import MemoryMixin
        source = inspect.getsource(MemoryMixin._check_insight_evolution)
        assert "DEBUG" in source
        assert "pass" not in source.split("except")[-1] or "traceback" in source
