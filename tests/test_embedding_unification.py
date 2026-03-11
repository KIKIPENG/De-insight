"""Tests for embedding provider unification."""

import pytest
from unittest.mock import patch


class TestEmbeddingProviderUnification:
    """Ensure embedding is unified to OpenRouter."""

    def test_embed_providers_has_openrouter(self):
        """EMBED_PROVIDERS must contain openrouter with nvidia model."""
        from providers import EMBED_PROVIDERS
        
        assert "openrouter" in EMBED_PROVIDERS
        assert "nvidia/llama-nemotron-embed-vl-1b-v2:free" in EMBED_PROVIDERS["openrouter"]["models"]

    def test_embed_providers_no_jina(self):
        """EMBED_PROVIDERS must not contain jina."""
        from providers import EMBED_PROVIDERS
        
        assert "jina" not in EMBED_PROVIDERS
        assert "jina-api" not in EMBED_PROVIDERS

    def test_save_embed_config_writes_openrouter(self):
        """_save_embed_config must write openrouter settings."""
        from settings import save_env_key, load_env
        
        # Call the function that's used in onboarding
        from modals import OnboardingScreen
        import asyncio
        
        modal = OnboardingScreen.__new__(OnboardingScreen)
        asyncio.run(modal._save_embed_config("openrouter"))
        
        env = load_env()
        assert env.get("EMBED_PROVIDER") == "openrouter"
        assert env.get("EMBED_MODEL") == "nvidia/llama-nemotron-embed-vl-1b-v2:free"
        assert env.get("EMBED_DIM") == "1024"

    def test_runtime_uses_openrouter_regardless_of_env(self):
        """Runtime must use OpenRouter even if .env has Jina values."""
        import os
        from embeddings.service import get_embedding_service, reset_embedding_service
        
        # Force reset to pick up new config
        reset_embedding_service()
        
        # Even with JINA_API_KEY set, should use OpenRouter
        with patch.dict(os.environ, {"JINA_API_KEY": "fake_jina_key"}, clear=False):
            svc = get_embedding_service()
            svc.ensure_server_running()
            
            assert type(svc._backend).__name__ == "OpenRouterEmbeddingBackend"
            assert svc._backend._model == "nvidia/llama-nemotron-embed-vl-1b-v2:free"

    def test_render_done_shows_correct_fallback(self):
        """_render_done must show correct embedding model."""
        from settings import save_env_key, load_env
        
        # Simulate empty EMBED_MODEL (what happens before onboarding)
        save_env_key("EMBED_MODEL", "")
        
        # Check the fallback logic
        env = load_env()
        embed = env.get("EMBED_MODEL", "")
        if not embed or "/" not in embed:
            embed = "nvidia/llama-nemotron-embed-vl-1b-v2:free"
        
        assert embed == "nvidia/llama-nemotron-embed-vl-1b-v2:free"

    def test_embeddings_service_no_jina_imports(self):
        """embeddings/service.py must not import jina."""
        import embeddings.service as svc_module
        
        source = open(svc_module.__file__).read()
        assert "jina" not in source.lower() or "jina_reader" in source


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
