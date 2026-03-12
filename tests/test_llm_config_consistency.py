"""Test LLM provider configuration consistency.

Verifies that OpenRouter configuration is properly passed to litellm
regardless of whether env vars are inherited from parent process.
"""

import pytest
from unittest.mock import patch, MagicMock, AsyncMock
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))


@pytest.fixture
def mock_openrouter_config():
    """Simulate OpenRouter configuration in .env."""
    return {
        "LLM_MODEL": "openai/deepseek/deepseek-chat-v3-0324",
        "OPENAI_API_BASE": "https://openrouter.ai/api/v1",
        "OPENAI_API_KEY": "sk-or-v1-test-openai-key",
        "OPENROUTER_API_KEY": "sk-or-v1-test-router-key",
    }


def test_chat_completion_passes_api_key_for_openrouter(mock_openrouter_config):
    """Verify chat_completion passes API key to litellm for OpenRouter models."""
    from backend.services.llm import chat_completion, _is_codex_model
    
    captured_kwargs = {}
    
    async def fake_acompletion(model, **kwargs):
        captured_kwargs.update(kwargs)
        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock()]
        mock_resp.choices[0].delta = MagicMock()
        mock_resp.choices[0].delta.content = "test"
        
        # Return an async iterator
        async def gen():
            yield mock_resp
        return gen()
    
    with patch("backend.services.llm.get_config_service") as mock_cfg, \
         patch("backend.services.llm.litellm.acompletion", side_effect=fake_acompletion) as mock_comp, \
         patch("backend.services.llm._is_codex_model", return_value=False):  # Skip codex path
        
        mock_cfg.return_value.get.side_effect = lambda key, default="": mock_openrouter_config.get(key, default)
        
        # Run the async function - collect yielded values
        import asyncio
        
        async def run():
            result_chunks = []
            try:
                async for chunk in chat_completion([{"role": "user", "content": "test"}]):
                    result_chunks.append(chunk)
            except Exception as e:
                # May fail due to mock, but we care about kwargs passed
                pass
            return result_chunks
        
        asyncio.run(run())
        
        # Verify api_key was passed
        mock_comp.assert_called()
        call_kwargs = mock_comp.call_args[1]
        
        assert "api_key" in call_kwargs, "API key must be passed to litellm"
        assert call_kwargs["api_key"] in ["sk-or-v1-test-openai-key", "sk-or-v1-test-router-key"], \
            "API key should be one of the configured keys"


def test_chat_completion_passes_api_base_for_openrouter(mock_openrouter_config):
    """Verify chat_completion passes API base to litellm for openai/* models."""
    from backend.services.llm import chat_completion
    
    async def fake_acompletion(model, **kwargs):
        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock()]
        mock_resp.choices[0].delta = MagicMock()
        mock_resp.choices[0].delta.content = ""
        
        async def gen():
            yield mock_resp
        return gen()
    
    with patch("backend.services.llm.get_config_service") as mock_cfg, \
         patch("backend.services.llm.litellm.acompletion", side_effect=fake_acompletion) as mock_comp, \
         patch("backend.services.llm._is_codex_model", return_value=False):
        
        mock_cfg.return_value.get.side_effect = lambda key, default="": mock_openrouter_config.get(key, default)
        
        import asyncio
        async def run():
            async for _ in chat_completion([{"role": "user", "content": "test"}]):
                pass
        asyncio.run(run())
        
        mock_comp.assert_called()
        call_kwargs = mock_comp.call_args[1]
        
        assert "api_base" in call_kwargs, "API base must be passed to litellm for openai/* models"
        assert call_kwargs["api_base"] == "https://openrouter.ai/api/v1"


def test_config_service_has_required_openrouter_keys():
    """Verify that config service properly reads OpenRouter keys from .env."""
    from config.service import get_config_service
    
    cfg = get_config_service()
    
    # These should be present in .env
    llm_model = cfg.get("LLM_MODEL", "")
    
    # If using OpenRouter (model starts with openai/)
    if llm_model.startswith("openai/"):
        # Either direct OpenAI key or OpenRouter key should be available
        has_base = bool(cfg.get("OPENAI_API_BASE", ""))
        has_key = bool(cfg.get("OPENAI_API_KEY", "") or cfg.get("OPENROUTER_API_KEY", ""))
        
        assert has_base, "OPENAI_API_BASE must be set for OpenRouter"
        assert has_key, "Either OPENAI_API_KEY or OPENROUTER_API_KEY must be set"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
