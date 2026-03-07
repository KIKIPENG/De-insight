**[繁體中文](README.zh-TW.md)** | English

# De-insight

A terminal-based AI thinking partner for visual artists and designers.

Not the kind of AI that says "sure, let me organize that for you."
More like a friend who pushes back — when you say something vague, it asks what you actually mean.

Ideas are always yours. It just helps you find the structure underneath.

---

## Features

**v0.7 — Current**

- **Curator dialogue** — emotional / rational mode switching, interactive prompts (`<<SELECT>>`, `<<CONFIRM>>`, etc.)
- **Memory system** — auto-extracts insights from conversation, requires user confirmation before saving
- **Knowledge base** — import PDFs/URLs, builds a knowledge graph (LightRAG), auto-referenced in conversation
- **Project isolation** — each project has its own memories, conversations, knowledge graph, and images
- **Local embedding** — jina-clip-v1 (dim=512), supports both text and image semantic search
- **Image gallery** — web-based upload/search/select, `@mention` to attach images to conversation
- **Onboarding** — first-launch setup wizard for LLM provider and embedding mode
- **Multi-provider** — OpenAI, Anthropic, DeepSeek, OpenRouter, MiniMax, Ollama, Codex CLI

**Known limitations**

- TUI does **not** render images inline. Image features work via text retrieval, path/caption references, and external viewing.
- `test_rag_switch.py` requires LightRAG and external API access; runs in backend venv only.

---

## Architecture Overview

```
project-root/
├── tui.py                     # Entry point (3 lines)
├── app.py                     # DeInsightApp (CSS/BINDINGS/compose/on_mount)
├── providers.py               # Provider/Service definitions
├── settings.py                # SettingsScreen + ENV helpers
├── widgets.py                 # ChatInput, MenuBar, Chatbox, etc.
├── modals.py                  # All ModalScreens (Onboarding, Project, Memory, etc.)
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
│   └── local.py               # jina-clip-v1 local embedding (dim=512)
│
├── memory/
│   ├── store.py               # SQLite async CRUD (aiosqlite)
│   ├── thought_tracker.py     # Memory extraction candidates
│   └── vectorstore.py         # LanceDB vector index
│
├── rag/
│   ├── knowledge_graph.py     # LightRAG wrapper
│   └── image_store.py         # Image knowledge base (LanceDB)
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
└── ~/.deinsight/v0.6/         # User data (physical directory isolation)
    ├── app.db                 # Project list
    ├── selected.json          # Gallery selection state
    └── projects/{project_id}/
        ├── memories.db
        ├── conversations.db
        ├── lancedb/           # Vector index (memories + images tables)
        ├── lightrag/          # Knowledge graph
        ├── images/            # Uploaded image files
        └── documents/         # Imported documents
```

---

## Requirements

- **Python** 3.10+
- **macOS / Linux** (tested on macOS; Linux should work, Windows untested)
- **Disk space** for local embedding: ~1 GB for jina-clip-v1 model download (optional, only if `EMBED_MODE=local`)

---

## Quick Start

```bash
# 1. Clone and enter the repo
git clone <repo-url>
cd De-insight

# 2. Set up backend virtual environment
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cd ..

# 3. Create .env with your API key
echo 'LLM_MODEL=anthropic/claude-sonnet-4-20250514' > .env
echo 'ANTHROPIC_API_KEY=sk-ant-...' >> .env

# 4. Start the backend
source backend/.venv/bin/activate
cd backend && uvicorn main:app --reload &
cd ..

# 5. Start the TUI
python3 tui.py
```

On first launch with no existing projects, the **Onboarding wizard** will guide you through provider and embedding setup.

---

## Configuration

All configuration lives in `.env` at the project root.

### Required

| Variable | Description | Example |
|----------|-------------|---------|
| `LLM_MODEL` | Chat model identifier | `anthropic/claude-sonnet-4-20250514` |

### Provider API Keys (set the one matching your `LLM_MODEL`)

| Variable | Provider |
|----------|----------|
| `ANTHROPIC_API_KEY` | Anthropic |
| `OPENAI_API_KEY` | OpenAI |
| `OPENROUTER_API_KEY` | OpenRouter |
| `DEEPSEEK_API_KEY` | DeepSeek |
| `MINIMAX_API_KEY` | MiniMax |
| `CODEX_API_KEY` | OpenAI Codex API |

No key needed for **Ollama** (local) or **Codex CLI** (OAuth).

### Embedding

| Variable | Description | Default |
|----------|-------------|---------|
| `EMBED_MODE` | `local` for jina-clip-v1, omit for API-based | _(unset)_ |
| `EMBED_PROVIDER` | Embedding provider ID (e.g. `local`, `jina`, `openai-embed`) | _(unset)_ |
| `EMBED_DIM` | Embedding dimension | `512` for local |

### Optional

| Variable | Description |
|----------|-------------|
| `OPENAI_API_BASE` | Custom API base URL (for OpenRouter, MiniMax, etc.) |
| `EMBED_API_BASE` | Custom embedding API base URL |
| `RAG_LLM_MODEL` | Separate model for knowledge graph extraction |
| `DEINSIGHT_HOME` | Override data directory (default: `~/.deinsight`) |

---

## Run

### Backend

```bash
cd backend
source .venv/bin/activate
uvicorn main:app --reload
```

The backend serves:
- `http://localhost:8000/api/health` — health check
- `http://localhost:8000/api/images` — image gallery API
- `http://localhost:8000/gallery` — web gallery UI

### TUI

```bash
source backend/.venv/bin/activate
python3 tui.py
```

### Keyboard Shortcuts

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

### Slash Commands

Type `/help` in the chat input for the full list. Key commands: `/import`, `/search`, `/memory`, `/save`, `/mode`, `/project`, `/reindex`.

---

## Testing

```bash
# Activate the backend venv first
source backend/.venv/bin/activate

# Conversation and memory isolation tests (no external dependencies)
python3 -m unittest tests/test_conversation_isolation.py -v

# Prompt parser tests
python3 -m unittest tests/test_prompt_parser.py -v

# RAG switch tests (requires lightrag + external embedding API)
# Must run from backend venv with lightrag installed
python3 -m unittest tests/test_rag_switch.py -v
```

> **Note:** `test_rag_switch.py` requires `lightrag` (installed in `backend/.venv`) and a working embedding API. It will skip gracefully if the API is unavailable.

---

## Data & Compatibility

### Storage location

All user data is stored in `~/.deinsight/v0.6/` (override with `DEINSIGHT_HOME` env var).

Each project gets its own directory under `~/.deinsight/v0.6/projects/{project_id}/` with isolated databases, vector indices, knowledge graphs, and images.

### v0.6 → v0.7 Incompatibility

- **v0.7 and v0.6 data formats are not compatible.**
- **No automatic migration is provided.**
- Recommended: start fresh with a new data directory when upgrading to v0.7.

To use a separate data directory:

```bash
export DEINSIGHT_HOME=~/.deinsight-v07
python3 tui.py
```

---

## Troubleshooting

### `ModuleNotFoundError: No module named 'lightrag'`

You're running outside the backend venv. Activate it first:

```bash
source backend/.venv/bin/activate
```

### Backend connection error in TUI

The backend must be running before or alongside the TUI:

```bash
cd backend && source .venv/bin/activate && uvicorn main:app --reload
```

### `torch` / `transformers` not found (local embedding)

Install the full requirements in the backend venv:

```bash
cd backend && source .venv/bin/activate
pip install -r requirements.txt
```

### Embedding dimension mismatch

If you switch `EMBED_MODE` between `local` (dim=512) and an API provider (different dim), the vector index will be automatically rebuilt on next access. Existing memories are re-indexed; no data is lost.

### Gallery not loading at `/gallery`

Ensure the `frontend/` directory exists at the project root and the backend is running. The gallery is served as static files from `frontend/index.html`.

---

## Contributing

1. Fork the repo and create a feature branch
2. Make changes — read relevant files before modifying
3. Run `python3 -m compileall <changed files>` to check for syntax errors
4. Run `python3 -m unittest tests/test_conversation_isolation.py` to verify no regressions
5. Submit a pull request

### Rules

- Don't hardcode API keys
- Background workers must log exceptions (no silent `except: pass`)
- New modals go in `modals.py`, not inline
- Test with `python3 tui.py` after changes to verify no import errors

---

## License

TODO — License not yet defined.
