"""De-insight — Provider / Service 定義（從 settings.py 抽出）"""

# ── Service definitions ──────────────────────────────────────────────
# 三種服務：Chat LLM、Embedding、RAG LLM（知識庫抽取用）

SERVICES = {
    "chat": {
        "label": "聊天模型",
        "desc": "主要對話用的 LLM",
        "env_model": "LLM_MODEL",
    },
    "embedding": {
        "label": "Embedding 模型",
        "desc": "知識庫向量化用",
        "env_model": "EMBED_PROVIDER",
    },
    "rag_llm": {
        "label": "知識庫 LLM",
        "desc": "知識圖譜實體抽取用",
        "env_model": "RAG_LLM_MODEL",
    },
}

# ── Chat LLM providers ──

CHAT_PROVIDERS = {
    "openai": {
        "label": "OpenAI",
        "models": ["gpt-4o", "gpt-4o-mini", "gpt-4.1", "gpt-4.1-mini", "gpt-4.1-nano", "o3", "o4-mini"],
        "key_env": "OPENAI_API_KEY",
        "base_env": "OPENAI_API_BASE",
        "model_prefix": "openai/",
        "default_base": "",
        "auth_type": "api_key",
    },
    "codex": {
        "label": "OpenAI Codex (API)",
        "models": ["codex-mini-latest", "gpt-5.3-codex-medium"],
        "key_env": "CODEX_API_KEY",
        "base_env": "",
        "model_prefix": "codex/",
        "default_base": "",
        "auth_type": "api_key",
    },
    "codex-cli": {
        "label": "Codex CLI (OAuth)",
        "models": [],
        "key_env": "",
        "base_env": "",
        "model_prefix": "codex-cli/",
        "default_base": "",
        "auth_type": "oauth",
    },
    "anthropic": {
        "label": "Anthropic",
        "models": [
            "claude-sonnet-4-20250514",
            "claude-haiku-4-5-20251001",
            "claude-opus-4-20250514",
        ],
        "key_env": "ANTHROPIC_API_KEY",
        "base_env": "",
        "model_prefix": "",
        "default_base": "",
        "auth_type": "api_key",
    },
    "deepseek": {
        "label": "DeepSeek",
        "models": [
            "deepseek-chat",
            "deepseek-reasoner",
        ],
        "key_env": "DEEPSEEK_API_KEY",
        "base_env": "DEEPSEEK_API_BASE",
        "model_prefix": "openai/",
        "default_base": "https://api.deepseek.com/v1",
        "auth_type": "api_key",
    },
    "minimax": {
        "label": "MiniMax",
        "models": ["MiniMax-M2.5", "MiniMax-Text-01"],
        "key_env": "MINIMAX_API_KEY",
        "base_env": "MINIMAX_API_BASE",
        "model_prefix": "openai/",
        "default_base": "https://api.minimaxi.chat/v1",
        "auth_type": "api_key",
    },
    "openrouter": {
        "label": "OpenRouter",
        "models": [
            "deepseek/deepseek-chat-v3-0324",
            "deepseek/deepseek-r1",
            "google/gemini-2.5-pro-preview",
            "google/gemini-2.5-flash-preview",
            "anthropic/claude-sonnet-4-20250514",
            "anthropic/claude-3.5-haiku",
            "openai/gpt-4o",
            "openai/gpt-4.1-mini",
            "meta-llama/llama-4-maverick",
        ],
        "key_env": "OPENROUTER_API_KEY",
        "base_env": "",
        "model_prefix": "openai/",
        "default_base": "https://openrouter.ai/api/v1",
        "auth_type": "api_key",
    },
    "ollama": {
        "label": "Ollama (本地)",
        "models": ["llama3.2", "qwen2.5", "mistral", "deepseek-r1"],
        "key_env": "",
        "base_env": "",
        "model_prefix": "ollama/",
        "default_base": "",
        "auth_type": "none",
    },
}

# ── Embedding providers ──

EMBED_PROVIDERS = {
    "local": {
        "label": "本地 jina-clip-v1 (免費)",
        "models": ["jina-clip-v1"],
        "key_env": "",
        "base_env": "",
        "default_base": "",
        "auth_type": "none",
        "dims": {"jina-clip-v1": 512},
    },
    "jina": {
        "label": "Jina AI",
        "models": ["jina-embeddings-v3", "jina-embeddings-v2-base-en"],
        "key_env": "JINA_API_KEY",
        "base_env": "",
        "default_base": "https://api.jina.ai/v1",
        "auth_type": "api_key",
        "dims": {"jina-embeddings-v3": 1024, "jina-embeddings-v2-base-en": 768},
    },
    "openai-embed": {
        "label": "OpenAI",
        "models": ["text-embedding-3-small", "text-embedding-3-large"],
        "key_env": "OPENAI_API_KEY",
        "base_env": "",
        "default_base": "https://api.openai.com/v1",
        "auth_type": "api_key",
        "dims": {"text-embedding-3-small": 1536, "text-embedding-3-large": 3072},
    },
    "ollama-embed": {
        "label": "Ollama (本地)",
        "models": ["nomic-embed-text", "mxbai-embed-large"],
        "key_env": "",
        "base_env": "",
        "default_base": "http://localhost:11434/v1",
        "auth_type": "none",
        "dims": {"nomic-embed-text": 768, "mxbai-embed-large": 1024},
    },
}

# ── RAG LLM providers (reuse chat providers but store separately) ──

RAG_LLM_PROVIDERS = {
    "same-as-chat": {
        "label": "跟聊天模型相同（含 Codex CLI）",
        "models": [],
        "key_env": "",
        "base_env": "",
        "model_prefix": "",
        "default_base": "",
        "auth_type": "none",
    },
    "openai-rag": {
        "label": "OpenAI",
        "models": ["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini", "gpt-4.1-nano"],
        "key_env": "OPENAI_API_KEY",
        "base_env": "",
        "model_prefix": "",
        "default_base": "https://api.openai.com/v1",
        "auth_type": "api_key",
    },
    "anthropic-rag": {
        "label": "Anthropic",
        "models": ["claude-haiku-4-5-20251001", "claude-sonnet-4-20250514"],
        "key_env": "ANTHROPIC_API_KEY",
        "base_env": "",
        "model_prefix": "",
        "default_base": "",
        "auth_type": "api_key",
    },
    "minimax-rag": {
        "label": "MiniMax",
        "models": ["MiniMax-Text-01", "MiniMax-M2.5"],
        "key_env": "MINIMAX_API_KEY",
        "base_env": "MINIMAX_API_BASE",
        "model_prefix": "",
        "default_base": "https://api.minimaxi.chat/v1",
        "auth_type": "api_key",
    },
    "openrouter-rag": {
        "label": "OpenRouter",
        "models": [
            "google/gemini-2.5-flash-preview",
            "openai/gpt-4.1-mini",
            "openai/gpt-4o-mini",
            "deepseek/deepseek-r1",
            "meta-llama/llama-4-maverick",
        ],
        "key_env": "OPENROUTER_API_KEY",
        "base_env": "",
        "model_prefix": "",
        "default_base": "https://openrouter.ai/api/v1",
        "auth_type": "api_key",
    },
    "deepseek-rag": {
        "label": "DeepSeek",
        "models": ["deepseek-chat"],
        "key_env": "DEEPSEEK_API_KEY",
        "base_env": "DEEPSEEK_API_BASE",
        "model_prefix": "openai/",
        "default_base": "https://api.deepseek.com/v1",
        "auth_type": "api_key",
    },
    "ollama-rag": {
        "label": "Ollama (本地)",
        "models": ["llama3.2", "qwen2.5", "mistral"],
        "key_env": "",
        "base_env": "",
        "model_prefix": "ollama/",
        "default_base": "",
        "auth_type": "none",
    },
}

# Keep backward compat
PROVIDERS = CHAT_PROVIDERS
