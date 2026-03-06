# CLAUDE.md — De-insight 專案說明

給 Claude Code 的入職說明。每次開始新任務前請先讀這份文件。

---

## 這個專案是什麼

De-insight 是一個在終端機裡跑的 AI 思想對話工具。
人格設定是策展人——懂得聆聽、說話直接、不急著給答案的思想夥伴。

核心體驗：把還說不清楚的東西說清楚。想法來自使用者，策展人幫他找到骨架。

**當前版本：v0.3.5**

---

## 實際架構（以這個為準，不要參考其他文件的舊描述）

```
project-root/
├── tui.py                    # Textual TUI 主體，App class + 對話流程
├── modals.py                 # 所有 ModalScreen：ProjectModal、MemoryConfirmModal
├── panels.py                 # ResearchPanel、MemoryPanel widgets
├── settings.py               # SettingsScreen + Provider/ENV 管理
├── codex_client.py           # Codex CLI OAuth 串接（不要動）
├── .env                      # API keys（不進 git）
├── requirements.txt
│
├── backend/
│   ├── main.py               # FastAPI 入口
│   ├── routers/chat.py       # /api/chat SSE 串流
│   ├── services/llm.py       # LiteLLM 封裝
│   └── prompts/
│       ├── curator.py        # 策展人人格（主要 system prompt）
│       ├── foucault.py       # re-export curator，向後相容用
│       └── memory_prompts.py # 記憶抽取提示詞
│
├── memory/
│   ├── store.py              # SQLite async CRUD（aiosqlite）
│   ├── thought_tracker.py    # 記憶抽取候選，不直接存入
│   └── vectorstore.py        # LanceDB 向量索引
│
├── rag/
│   └── knowledge_graph.py   # LightRAG 封裝，per-project working dir
│
├── projects/
│   └── manager.py            # 專案 CRUD（data/app.db）
│
├── conversation/
│   └── store.py              # 對話歷史持久化（data/conversations.db）
│
├── interaction/
│   └── prompt_parser.py      # 解析 <<SELECT>> 等互動標記
│
├── docs/                     # 規格文件（不影響執行）
│
├── tests/                    # 測試腳本
│   ├── test_prompt_parser.py
│   ├── test_rag_switch.py
│   └── test_stability.py
│
└── data/                     # 執行時產生，不進 git
    ├── app.db                # 專案清單
    ├── memories.db           # 記憶（跨專案）
    ├── conversations.db      # 對話歷史
    ├── lancedb/              # 向量索引
    └── projects/
        └── {project_id}/
            └── lightrag/     # per-project 知識圖譜
```

---

## 三條對話路徑

`_stream_response()` 根據設定走不同路徑：

1. **Codex CLI 路徑**：`LLM_MODEL` 以 `codex-cli/` 開頭
2. **直接 API 路徑**：`LLM_MODEL` 以 `openai/` 開頭且有 `OPENAI_API_BASE`
3. **FastAPI 後端路徑**：其餘情況，走 `/api/chat` SSE

---

## 支援的 LLM Provider

| Provider | 設定方式 |
|---------|---------|
| Anthropic | `LLM_MODEL=anthropic/claude-sonnet-4-6` |
| OpenAI | `LLM_MODEL=openai/gpt-4o` |
| OpenRouter | `LLM_MODEL=openai/deepseek/...` + `OPENAI_API_BASE=https://openrouter.ai/api/v1` |
| MiniMax | 直接 API 路徑 |
| Codex CLI | OAuth，本機執行 |
| Ollama | 本地模型 |

---

## 狀態管理（AppState）

所有重要狀態集中在 `AppState` dataclass，透過 `self.state` 存取：

```python
self.state.current_project        # 當前專案 dict | None
self.state.pending_memories       # 待確認記憶候選 list[dict]
self.state.interactive_depth      # 互動提問遞迴深度（上限 3）
self.state.current_conversation_id # 當前對話 ID | None
self.state.cached_memory_count    # MenuBar 顯示用的記憶數快取
self.state.current_interactive_block # 當前互動提問區塊
```

`self.messages` 是當前對話的訊息 list，不在 AppState 裡。

---

## 記憶流程

1. 對話結束後，`_auto_extract_memories()` 背景抽取候選
2. 候選存入 `self.state.pending_memories`，MenuBar 顯示「💡 N 待確認」
3. 使用者點擊 → `MemoryConfirmModal` → 確認後才呼叫 `add_memory()`
4. **不自動存入**，必須使用者確認

---

## 互動提問流程

策展人回應可以嵌入 `<<SELECT>>`、`<<MULTI>>`、`<<INPUT>>`、`<<CONFIRM>>` 標記。
TUI 解析後把選項嵌入 ChatInput，使用者用鍵盤選擇，選擇結果自動送入對話。

遞迴深度由 `self.state.interactive_depth` 控制，超過 3 層停止。

---

## 絕對不要動的檔案

- `codex_client.py`
- `settings.py`（除非明確被要求）
- `backend/routers/chat.py`
- `backend/services/llm.py`
- `memory/vectorstore.py`

---

## 啟動方式

```bash
# 安裝依賴
pip install -r requirements.txt

# 設定 API key
cp .env.example .env
# 編輯 .env 填入 key

# 啟動
python tui.py
```

後端 FastAPI 由 TUI 自動在背景啟動，不需要手動跑。

---

## 開發規則

1. 只做被要求的改動，不要自作主張新增功能
2. 修改前先讀相關檔案，確認現有實作再動手
3. 不要把 API key 硬編碼進任何檔案
4. 每次修改後跑一次 `python tui.py` 確認無 import error
5. 背景 worker 的 exception 要 log，不要 `except: pass` 靜默吞掉
6. `tui.py` 超過 2000 行，新增的 ModalScreen 一律放 `modals.py`
