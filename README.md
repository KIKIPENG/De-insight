<h4 align="center">
  <strong>English</strong> | <a href="https://github.com/KIKIPENG/De-insight/blob/main/README.zh-TW.md">繁體中文</a>
</h4>

<h1 align="center">◈ De-insight</h1>

<p align="center">
  An AI thinking partner for visual artists and designers.<br>
  It remembers what you said, sees the images you collect, and tells you when your ideas evolve.
</p>

<pre align="center">curl -fsSL https://raw.githubusercontent.com/KIKIPENG/De-insight/main/install.sh | bash</pre>

---

## What is this

When developing an artist statement or design thesis, you have readings, observations, and gut reactions scattered across months. Turning all of that into a coherent text is a long road.

De-insight tries to shorten that road.

It's not a search engine, not a writing assistant, not a general chatbot. It's closer to a long-term collaborator with curatorial sensibility — it remembers your last conversation, has read the texts you fed it, has seen the reference images you collected, and pushes back when you're being vague.

When your thinking quietly shifts over three months, it tells you. When you say you like minimalism but your image collection is all rough, handmade stuff, it points that out.

When you're ready to write, it helps you organize from your own language first. Theory comes last — as support, not the main character.

> This is a graduation project. Built for one person's workflow, open to anyone who thinks similarly.

---

## What it does

### Conversation

Two modes. **Emotional mode** pays attention to the hesitation behind your word choices, doesn't rush to analyze. **Rational mode** demands precision — if a word is vague, it asks what you mean.

The curator speaks directly, briefly, with a position. It's not neutral — ask what it thinks and it'll tell you.

When you say "I want to write my statement now," it walks you through three steps: distill your thesis, find evidence from the knowledge base, draft the text. Each step waits for your confirmation.

### Knowledge base

Feed it what you've been reading: PDFs, web links, DOIs, arXiv, plain text. It builds a knowledge graph in the background. You can see progress in the MenuBar.

During conversation, it automatically retrieves relevant passages. The curator can only use what's in the knowledge base — it can't invent theorists that aren't there, and must say so when evidence is insufficient.

Two retrieval modes: **fast** (vector search, semantic matching) and **deep** (graph reasoning — can find structural connections across texts, like the link between Foucault's "panopticon" and Berger's "ways of seeing").

### Memory

After each conversation, the system analyzes what you said and extracts things worth keeping — insights, questions, reactions to artworks. These are candidates; they only get saved after you confirm.

Memories carry topic tags and category labels. Next conversation, the curator speaks with these memories in context.

**Thought evolution detection**: When you save a new insight, the system compares it against past insights to detect shifts or contradictions. "Three months ago you believed X, but what you're saying now implies Y."

### Images

Open the image gallery in your browser and upload reference images. Each image gets a three-part analysis:

- **CONTENT**: Object identification — book title, author, and the relationship between content and design language
- **STYLE_TAGS**: Style coordinates — making attitude, density, color tendency, era, cultural axis, stance
- **DESCRIPTION**: Curatorial vocabulary description

**Visual preference extraction**: After 5+ images, the system automatically identifies your style tendencies. The result is injected into conversation — the curator can say "your collection leans toward rough, handmade, experimental work."

**Cross-modal detection**: When your image preferences contradict your text-based insights, the system flags it — "you say you like minimalism, but your images tell a different story."

### Projects

Each project has its own conversations, memories, knowledge base, and images. For separating different creative contexts.

---

## Install

```bash
curl -fsSL https://raw.githubusercontent.com/KIKIPENG/De-insight/main/install.sh | bash
```

```bash
de-insight           # launch
de-insight --update  # update
de-insight --uninstall  # remove
```

First launch walks you through onboarding to choose an LLM provider. The embedding model (~1.5 GB) downloads and compiles automatically on first use.

Requires **Python 3.11+**, **git**, and **macOS / Linux**.

---

## Configuration

Edit `~/.deinsight/app/.env`, or press `Ctrl+S` in the TUI.

```
# Main chat model
LLM_MODEL=openai/deepseek/deepseek-chat-v3-0324
OPENROUTER_API_KEY=<key>
OPENAI_API_BASE=https://openrouter.ai/api/v1

# Knowledge graph + image analysis (Gemini recommended — free and reliable)
RAG_LLM_MODEL=gemini-2.5-flash
VISION_MODEL=gemini-2.5-flash
GOOGLE_API_KEY=<key>
```

Embedding runs entirely locally. No API key needed.

---

## System requirements

| | Minimum | Recommended |
|-|---------|-------------|
| Chip | Any | Apple M1+ (Metal acceleration) |
| RAM | 8 GB | 16 GB |
| Disk | 4 GB | 10 GB+ |
| Python | 3.11+ | — |

macOS and Linux only need a working Python 3.11+ environment for the default remote embedding setup.

---

## Architecture

```
tui.py → app.py (Textual TUI + FastAPI backend)

Chat         → DeepSeek V3 via OpenRouter
Knowledge    → Gemini via Google AI Studio
Vision       → Gemini via Google AI Studio
Embedding    → NVIDIA Llama Nemotron Embed VL via OpenRouter

Knowledge DB → LightRAG (JSON/NetworkX)
Vector index → LanceDB
Memory       → SQLite (aiosqlite)
Image gallery→ LanceDB + web frontend
```

```
~/.deinsight/
├── app/                  source + venv
├── v0.7/projects/{id}/   project data
│   ├── memories.db       insights, questions, reactions, preferences
│   ├── conversations.db  conversation history
│   ├── lancedb/          vector indices
│   ├── lightrag/         knowledge graph
│   ├── images/           image files
│   └── documents/        PDFs
```

---

## Troubleshooting

**Embedding setup fails**: confirm `OPENROUTER_API_KEY` is set and the selected embedding model is reachable.

**Backend won't connect**: It starts automatically with the TUI. Check port 8000: `curl -m 3 -sS http://127.0.0.1:8000/api/health`.

**Gallery won't load**: Make sure the TUI is running, then open `http://localhost:8000/gallery`.

---

## Known limitations

- The terminal can't render images inline. Image features work through text descriptions and semantic retrieval.
- First launch requires compiling llama.cpp and downloading the model — takes a few minutes.
- The curator speaks Traditional Chinese only.

---

## Tests

```bash
~/.deinsight/app/.venv/bin/python -m pytest -q tests/
```

---

## License

Not yet defined.

---

<sub>made by KIKI PENG — art & design graduation project</sub>
