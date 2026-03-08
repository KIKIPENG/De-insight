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
    "google": {
        "label": "Google AI Studio",
        "models": ["gemini-2.5-flash", "gemini-2.5-pro"],
        "key_env": "GOOGLE_API_KEY",
        "base_env": "",
        "model_prefix": "openai/",
        "default_base": "https://generativelanguage.googleapis.com/v1beta/openai",
        "auth_type": "api_key",
    },
    "openrouter": {
        "label": "OpenRouter",
        "models": [
            "deepseek/deepseek-chat-v3-0324",
            "deepseek/deepseek-r1",
            "google/gemini-2.5-pro-preview",
            "google/gemini-2.5-flash",
            "anthropic/claude-sonnet-4-20250514",
            "anthropic/claude-3.5-haiku",
            "openai/gpt-4o",
            "openai/gpt-4.1-mini",
            "meta-llama/llama-4-maverick",
            "qwen/qwen2.5-vl-72b-instruct",
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
    "gguf": {
        "label": "jina-embeddings-v4 GGUF (本地 llama-server)",
        "models": ["jina-embeddings-v4-gguf"],
        "key_env": "",
        "base_env": "GGUF_SERVER_HOST",
        "default_base": "127.0.0.1",
        "auth_type": "none",
        "dims": {"jina-embeddings-v4-gguf": 1024},
        "env_extras": {
            "GGUF_SERVER_PORT": "8999",
            "GGUF_AUTO_INSTALL": "1",
        },
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
    "google-rag": {
        "label": "Google AI Studio",
        "models": ["gemini-2.5-flash", "gemini-2.5-pro"],
        "key_env": "GOOGLE_API_KEY",
        "base_env": "",
        "model_prefix": "",
        "default_base": "https://generativelanguage.googleapis.com/v1beta/openai",
        "auth_type": "api_key",
    },
    "openrouter-rag": {
        "label": "OpenRouter",
        "models": [
            "google/gemini-2.5-flash",
            "openai/gpt-4.1-mini",
            "openai/gpt-4o-mini",
            "deepseek/deepseek-r1",
            "meta-llama/llama-4-maverick",
            "qwen/qwen2.5-vl-72b-instruct",
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
