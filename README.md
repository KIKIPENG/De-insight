**[繁體中文](README.zh-TW.md)** | English

# De-insight

A terminal-based AI thinking partner for visual artists and designers.

Not the kind of AI that says "sure, let me organize that for you."
More like a friend who pushes back — when you say something vague, it asks what you actually mean.

Ideas are always yours. It just helps you find the structure underneath.

---

## Install

```bash
curl -fsSL https://raw.githubusercontent.com/KIKIPENG/De-insight/main/install.sh | bash
```

Then run:

```bash
de-insight
```

Update / uninstall:

```bash
de-insight --update
de-insight --uninstall
```

> Requires **Python 3.11+**, **git**, and **macOS / Linux**.

---

## Features

**v0.8 — Current**

- **Curator dialogue** — emotional / rational mode switching, interactive prompts (`<<SELECT>>`, `<<CONFIRM>>`, etc.)
- **Memory system** — auto-extracts insights from conversation, requires user confirmation before saving
- **Knowledge base** — import PDFs/URLs/DOI, builds a knowledge graph (LightRAG), auto-referenced in conversation
- **Project isolation** — each project has its own memories, conversations, knowledge graph, and images
- **Local GGUF embedding** — jina-embeddings-v4 (Q4_K_M, dim=1024) via llama-server, auto-installed on first run
- **Image gallery** — web-based upload/search/select, multimodal embedding, `@mention` to attach images to conversation
- **Ingestion pipeline** — background job queue with rate limiting, retry, and post-insert verification
- **Multi-provider** — OpenAI, Anthropic, DeepSeek, OpenRouter, MiniMax, Ollama, Codex CLI

**Known limitations**

- TUI does **not** render images inline. Image features work via text retrieval, path/caption references, and external viewing.

---

## Architecture Overview

```
project-root/
├── tui.py                     # Entry point
├── app.py                     # DeInsightApp (CSS/BINDINGS/compose/on_mount)
├── providers.py               # Provider/Service definitions
├── settings.py                # SettingsScreen + ENV helpers
├── widgets.py                 # ChatInput, MenuBar, Chatbox, etc.
├── modals.py                  # All ModalScreens
├── paths.py                   # Single source of truth for all data paths
│
├── mixins/                    # App mixin modules
│   ├── chat.py                # Dialogue flow, slash commands, RAG injection
│   ├── memory.py              # Memory CRUD, auto-extraction
│   ├── rag.py                 # Knowledge import/search
│   ├── project.py             # Project switching
│   └── ui.py                  # UI state, settings, gallery
│
├── embeddings/
│   ├── service.py             # EmbeddingService facade (singleton)
│   ├── backend.py             # Abstract EmbeddingBackend interface
│   ├── gguf_backend.py        # GGUFMultimodalBackend (llama-server API)
│   ├── llama_server.py        # LlamaServerManager (lifecycle, PID tracking)
│   └── gguf_installer.py      # Auto cmake + build + model download
│
├── memory/
│   ├── store.py               # SQLite async CRUD (aiosqlite)
│   ├── thought_tracker.py     # Memory extraction candidates
│   └── vectorstore.py         # LanceDB vector index
│
├── rag/
│   ├── knowledge_graph.py     # LightRAG wrapper
│   ├── image_store.py         # Image knowledge base (LanceDB, multimodal)
│   ├── pipeline.py            # RAG pipeline probe + readiness
│   ├── ingestion_service.py   # Background ingestion job queue
│   ├── job_repository.py      # SQLite job persistence
│   ├── rate_guard.py          # Rate limiting + retry
│   └── repair.py              # Index repair policies
│
├── backend/
│   ├── main.py                # FastAPI entry point
│   ├── routers/chat.py        # /api/chat SSE streaming
│   ├── routers/images.py      # Image gallery API
│   └── services/llm.py        # LiteLLM wrapper
│
├── frontend/
│   └── index.html             # Web gallery (/gallery)
│
└── ~/.deinsight/               # User data
    ├── v0.7/                   # Data (versioned)
    │   ├── app.db
    │   └── projects/{id}/
    │       ├── memories.db
    │       ├── conversations.db
    │       ├── lancedb/
    │       ├── lightrag/
    │       ├── images/
    │       └── documents/
    └── gguf/                   # Embedding model + llama-server
        ├── llama.cpp/
        ├── models/
        └── logs/
```

---

## Configuration

Edit `~/.deinsight/app/.env` (or `Ctrl+S` in TUI):

### Required

| Variable | Description | Example |
|----------|-------------|---------|
| `LLM_MODEL` | Chat model identifier | `openai/gpt-4o` |

### Provider API Keys

| Variable | Provider |
|----------|----------|
| `ANTHROPIC_API_KEY` | Anthropic |
| `OPENAI_API_KEY` | OpenAI |
| `OPENROUTER_API_KEY` | OpenRouter |
| `DEEPSEEK_API_KEY` | DeepSeek |
| `MINIMAX_API_KEY` | MiniMax |

No key needed for **Ollama** (local) or **Codex CLI** (OAuth).

### Optional

| Variable | Description |
|----------|-------------|
| `OPENAI_API_BASE` | Custom API base URL (for OpenRouter, MiniMax, etc.) |
| `RAG_LLM_MODEL` | Separate model for knowledge graph extraction |
| `DEINSIGHT_HOME` | Override data directory (default: `~/.deinsight`) |
| `GGUF_AUTO_INSTALL` | Auto-install GGUF embedding on first run (default: `1`) |

---

## Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `Ctrl+S` | Settings |
| `Ctrl+E` | Toggle emotional/rational mode |
| `Ctrl+N` | New conversation |
| `Ctrl+P` | Project management |
| `Ctrl+F` | Import PDF/URL |
| `Ctrl+K` | Search knowledge base |
| `Ctrl+M` | Memory management |
| `Ctrl+G` | View memory relations |
| `Ctrl+L` | Open image gallery (browser) |
| `Ctrl+D` | Document management |
| `Ctrl+B` | Bulk import |

Type `/help` in chat input for slash commands.

---

## Testing

```bash
# Using the installed venv
~/.deinsight/app/.venv/bin/python -m pytest -q tests/

# Or if developing locally
backend/.venv/bin/python -m pytest -q tests/
```

---

## Troubleshooting

### GGUF embedding build fails

Requires Xcode Command Line Tools (macOS) or `cmake` + C++ compiler (Linux):

```bash
xcode-select --install   # macOS
# or
sudo apt install cmake build-essential   # Ubuntu/Debian
```

### Backend connection error in TUI

The backend auto-starts with the TUI. If it fails, check port 8000:

```bash
curl -m 3 -sS http://127.0.0.1:8000/api/health
```

### Gallery not loading

Open `http://localhost:8000/gallery` in browser. Ensure TUI is running (backend starts with it).

---

## Contributing

1. Fork the repo and create a feature branch
2. Make changes — read relevant files before modifying
3. Run `python -m pytest tests/` to verify no regressions
4. Submit a pull request

---

## License

TODO — License not yet defined.
