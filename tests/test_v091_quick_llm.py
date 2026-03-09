"""v0.9.1: _quick_llm_call 應優先使用 RAG_LLM_MODEL。"""

import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import patch, AsyncMock, MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

# Pre-load litellm mock so patch() can find it
if "litellm" not in sys.modules:
    _mock_litellm = ModuleType("litellm")
    _mock_litellm.acompletion = AsyncMock()
    sys.modules["litellm"] = _mock_litellm

import pytest


@pytest.fixture
def mock_env_rag_local():
    """模擬 RAG_LLM_MODEL=ollama/phi4-mini, LLM_MODEL=openai/gpt-4o"""
    return {
        "RAG_LLM_MODEL": "ollama/phi4-mini",
        "LLM_MODEL": "openai/gpt-4o",
        "OPENAI_API_KEY": "sk-test",
    }


@pytest.fixture
def mock_env_no_rag():
    """模擬只設了 LLM_MODEL，沒有 RAG_LLM_MODEL"""
    return {
        "LLM_MODEL": "openai/gpt-4o",
        "OPENAI_API_KEY": "sk-test",
    }


@pytest.fixture
def mock_env_rag_empty():
    """模擬 RAG_LLM_MODEL 設為空字串"""
    return {
        "RAG_LLM_MODEL": "",
        "LLM_MODEL": "openai/gpt-4o",
        "OPENAI_API_KEY": "sk-test",
    }


@pytest.mark.asyncio
async def test_quick_llm_uses_rag_model_when_set(mock_env_rag_local):
    """設了 RAG_LLM_MODEL 時，_quick_llm_call 應該用它。"""
    captured_model = {}

    async def fake_acompletion(model, **kwargs):
        captured_model["model"] = model
        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock()]
        mock_resp.choices[0].message.content = "test response"
        return mock_resp

    with patch("mixins.chat.load_env", return_value=mock_env_rag_local), \
         patch("litellm.acompletion", side_effect=fake_acompletion):
        from mixins.chat import ChatMixin
        mixin = ChatMixin()
        result = await mixin._quick_llm_call("test prompt")

    assert captured_model["model"] == "ollama/phi4-mini", \
        f"Expected ollama/phi4-mini, got {captured_model['model']}"


@pytest.mark.asyncio
async def test_quick_llm_falls_back_when_no_rag(mock_env_no_rag):
    """沒設 RAG_LLM_MODEL 時，應該 fallback 到 LLM_MODEL。"""
    captured_model = {}

    async def fake_acompletion(model, **kwargs):
        captured_model["model"] = model
        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock()]
        mock_resp.choices[0].message.content = "test response"
        return mock_resp

    with patch("mixins.chat.load_env", return_value=mock_env_no_rag), \
         patch("litellm.acompletion", side_effect=fake_acompletion):
        from mixins.chat import ChatMixin
        mixin = ChatMixin()
        result = await mixin._quick_llm_call("test prompt")

    assert captured_model["model"] == "openai/gpt-4o", \
        f"Expected openai/gpt-4o, got {captured_model['model']}"


@pytest.mark.asyncio
async def test_quick_llm_falls_back_when_rag_empty(mock_env_rag_empty):
    """RAG_LLM_MODEL 設為空字串時，應該 fallback 到 LLM_MODEL。"""
    captured_model = {}

    async def fake_acompletion(model, **kwargs):
        captured_model["model"] = model
        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock()]
        mock_resp.choices[0].message.content = "test response"
        return mock_resp

    with patch("mixins.chat.load_env", return_value=mock_env_rag_empty), \
         patch("litellm.acompletion", side_effect=fake_acompletion):
        from mixins.chat import ChatMixin
        mixin = ChatMixin()
        result = await mixin._quick_llm_call("test prompt")

    assert captured_model["model"] == "openai/gpt-4o", \
        f"Expected openai/gpt-4o, got {captured_model['model']}"
