# Claude Code 指示：v0.2 升級

## 讀這份文件前先讀

- `CLAUDE.md` — 專案背景和開發規則
- `tui.py` — 現有 TUI 主體（**實際架構，不是 CLAUDE.md 說的 Next.js**）
- `main.py` — FastAPI 後端入口
- `settings.py` — 設定 modal
- `requirements.txt` — 目前依賴

---

## 現有架構摘要（給你定向）

```
project-root/          ← 所有檔案都在這層，不是 backend/ 子目錄
├── tui.py             ← Textual TUI 主體（入口）
├── main.py            ← FastAPI 後端
├── settings.py        ← 設定 ModalScreen
├── codex_client.py    ← Codex CLI 串接
├── requirements.txt
└── backend/           ← sys.path 被插入這裡，包含：
    ├── routers/chat.py
    ├── services/llm.py
    └── prompts/foucault.py
```

現有 TUI 快捷鍵：
- `ctrl+e` — 感性/理性切換
- `ctrl+n` — 新對話
- `ctrl+s` — 設定

現有 TUI 佈局（單欄）：
```
ModeIndicator（頂部，顯示模式）
VerticalScroll #chat-scroll → Vertical #messages（對話訊息）
StatusBar（顯示 model / mode / msg count）
Vertical #input-box → Vertical #input-frame → Input #chat-input
Footer（顯示快捷鍵）
```

---

## v0.2 任務範圍（Phase 1 MVP）

### 一句話目標
加入右側面板（記憶 + 研究）、LightRAG 知識庫、SQLite 記憶系統，以及基礎思維追蹤。

**只做 Phase 1，不要做 Phase 2（FalkorDB、Graphiti）。**

---

## 任務一：安裝新依賴

`requirements.txt` 新增：
```
lightrag-hku>=1.0.0
aiosqlite>=0.20.0
```

不要安裝：cognee、kuzu、neo4j、falkordb、graphiti、mem0。

---

## 任務二：新建檔案結構

```
project-root/
├── rag/
│   └── knowledge_graph.py    ← 新建
├── memory/
│   ├── store.py              ← 新建
│   └── thought_tracker.py    ← 新建
└── data/
    └── lightrag/             ← LightRAG working dir（git ignore）
```

在 `.gitignore` 加入 `data/` 。

---

## 任務三：LightRAG 整合（`rag/knowledge_graph.py`）

```python
"""LightRAG 知識圖譜封裝。"""

from lightrag import LightRAG, QueryParam
from lightrag.llm.litellm import litellm_model_complete, litellm_embed
from lightrag.utils import EmbeddingFunc
import os
from pathlib import Path

WORKING_DIR = Path(__file__).parent.parent / "data" / "lightrag"
WORKING_DIR.mkdir(parents=True, exist_ok=True)

# 藝術領域客製化 entity 類型（加入 LightRAG 的 extraction prompt）
ART_ENTITY_TYPES = [
    "藝術家", "設計師", "建築師", "理論家", "批評家",
    "藝術運動", "設計風格", "藝術流派",
    "媒材", "技法", "創作方法",
    "哲學概念", "批判理論",
    "藝術機構", "美術館", "畫廊",
    "展覽", "作品", "著作", "批評文本",
]

def get_rag() -> LightRAG:
    """取得 LightRAG 實例。從環境變數讀取 API key 和模型。"""
    # 實作時從 settings.py 的 load_env() 取得目前 model 設定
    # 用 litellm 統一介面，和現有後端一致
    ...

async def insert_text(text: str, source: str = "") -> None:
    """插入文字到知識庫。"""
    ...

async def insert_pdf(path: str) -> None:
    """讀取 PDF 並插入知識庫。需要 pypdf。"""
    ...

async def query_knowledge(question: str, mode: str = "hybrid") -> str:
    """查詢知識庫。mode: local / global / hybrid / naive"""
    ...
```

**重要**：LightRAG 初始化時，客製化 entity extraction prompt，把 `ART_ENTITY_TYPES` 注入，讓圖譜抽取藝術領域的實體而非通用實體。

LLM 和 embedding 設定從 `settings.py` 的 `load_env()` 讀取，**不要硬編碼**，和現有 `services/llm.py` 的邏輯保持一致。

---

## 任務四：SQLite 記憶系統（`memory/store.py`）

Schema：
```sql
CREATE TABLE IF NOT EXISTS memories (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    type      TEXT NOT NULL,           -- 'insight' | 'question' | 'reaction'
    content   TEXT NOT NULL,
    source    TEXT,                    -- 觸發這條記憶的對話原文（截短）
    created_at DATETIME DEFAULT (datetime('now', 'localtime')),
    tags      TEXT DEFAULT '[]'        -- JSON array
);
```

提供 async 函數（用 `aiosqlite`）：
- `async def add_memory(type, content, source="", tags=[]) -> int`
- `async def get_memories(type=None, limit=20) -> list[dict]`
- `async def delete_memory(id: int) -> None`
- `async def search_memories(query: str, limit=5) -> list[dict]` — 用 LIKE

DB 路徑：`data/memories.db`

---

## 任務五：基礎思維追蹤（`memory/thought_tracker.py`）

**Phase 1 只用 SQLite + LLM prompt，不用 Graphiti。**

```python
async def check_for_evolution(
    new_insight: str,
    store: MemoryStore,
    llm_call: callable,  # 傳入 LLM 呼叫函數，保持解耦
) -> dict | None:
    """
    新增洞見後呼叫。
    從歷史取出最相關的 5 條洞見，讓 LLM 判斷有無演變或矛盾。
    
    回傳 None（無顯著變化）
    或 {"type": "evolution"|"contradiction", "summary": str, "old": str, "new": str}
    """
```

LLM prompt 要點：
- 語氣：中性觀察者，不評判，只描述「偵測到什麼」
- 區分「演變」（自然成長，正向）vs「矛盾」（同時持有互斥觀點）
- 回傳格式要求 JSON，方便解析

---

## 任務六：TUI 改版（`tui.py`）

### 6.1 新佈局：加入右側面板

把現有的單欄佈局改為左右分割：

```
┌─────────────────────────┬──────────────────┐
│  ModeIndicator（頂部）                       │
├─────────────────────────┬──────────────────┤
│                         │  #research-panel  │
│  #chat-scroll（65%）    │  研究面板（右上）  │
│                         ├──────────────────┤
│                         │  #memory-panel    │
│                         │  記憶面板（右下）  │
└─────────────────────────┴──────────────────┘
  StatusBar（底部）
  #input-box（底部，dock）
  Footer（底部）
```

用 Textual 的 `Horizontal` 容器包住 `#chat-scroll` 和右側 `Vertical`（包含兩個面板）。

右側總寬度：`35%`（或固定 `40` 字元）

### 6.2 記憶面板（`#memory-panel`）

顯示最新 5 條記憶，格式：
```
💡 [洞見] 包豪斯預設了功能的穩定性    2h
❓ [問題] 無媒材的概念藝術說的是什麼   1d
💭 [反應] Judd 讓我感到安靜但不空洞   3d
```

- 每條記憶點擊後展開顯示完整內容
- 面板標題：`◇ Memories`
- 有新記憶時短暫高亮（加 CSS class `.-new`，1 秒後移除）

### 6.3 研究面板（`#research-panel`）

顯示最近一次 LightRAG 查詢的來源摘要（若知識庫為空則顯示提示）。
面板標題：`◇ Knowledge`

格式範例：
```
◇ 3 sources found
─────────────────
[1] Foucault - 規訓與懲罰 (p.42)
    「規訓製造了馴服的身體...」

[2] Berger - 觀看之道 (p.7)
    ...
```

知識庫為空時顯示：
```
◇ Knowledge  [empty]
─────────────────
ctrl+f  匯入 PDF
ctrl+k  搜尋
```

### 6.4 新增快捷鍵

在現有 `BINDINGS` 加入：

```python
Binding("ctrl+k", "search_knowledge", "搜尋知識庫", priority=True),
Binding("ctrl+f", "import_document", "匯入文件", priority=True),
Binding("ctrl+m", "manage_memories", "記憶管理", priority=True),
```

（保留原有的 `ctrl+e`/`ctrl+n`/`ctrl+s`）

**`ctrl+f`**：彈出簡單 modal，讓使用者輸入 PDF 路徑，呼叫 `insert_pdf()`，顯示進度通知。

**`ctrl+k`**：彈出搜尋 modal，輸入關鍵字，顯示 LightRAG 查詢結果。

**`ctrl+m`**：彈出記憶管理 modal，列出所有記憶，可刪除，可篩選類型。

### 6.5 對話流程整合

在 `_stream_response()` 結束後，加入背景工作：

```python
# 1. 若知識庫非空，在送出前先查詢 LightRAG，把結果注入 system context
# 2. 對話結束後，呼叫 LLM 自動抽取記憶
# 3. 若抽到洞見，呼叫 thought_tracker.check_for_evolution()
# 4. 若偵測到演變/矛盾，在記憶面板顯示特殊提示
```

記憶自動抽取 prompt 要點：
- **只抽使用者說的話**，不抽 AI 回應
- 只抽「洞見/問題/反應」三類，有則抽，無則回傳空
- 不要把閒聊或問候存成記憶

---

## CSS 補充（加入 `tui.py` 的 CSS 字串）

```css
/* ── layout ── */
#main-horizontal {
    height: 1fr;
}

#right-panel {
    width: 35%;
    border-left: solid #2a2a2a;
}

/* ── research panel ── */
#research-panel {
    height: 60%;
    border-bottom: solid #2a2a2a;
    padding: 0 1;
    overflow-y: auto;
}

/* ── memory panel ── */
#memory-panel {
    height: 40%;
    padding: 0 1;
    overflow-y: auto;
}

.memory-item {
    height: auto;
    margin: 0 0 1 0;
    padding: 0 1;
    color: #8b949e;
}

.memory-item.-new {
    color: #fafafa;
    background: #1a1a1a;
}

.memory-item:hover {
    color: #fafafa;
    background: #111111;
    cursor: pointer;
}
```

---

## 驗收標準

完成後應能做到：

1. `python tui.py` 啟動，左右分割佈局正確顯示
2. 右側兩個面板有標題，記憶面板顯示「空」的初始狀態
3. `ctrl+f` 彈出 modal → 輸入 PDF 路徑 → 匯入成功通知 → 研究面板出現來源列表
4. 匯入後問一個相關問題 → 回應明顯引用了文件內容 → 研究面板更新
5. 對話幾輪後記憶面板自動出現條目（💡/❓/💭）
6. `ctrl+m` 彈出記憶管理 → 可以刪除單條記憶
7. 原有功能完整保留：感性/理性切換、設定頁面、Codex CLI 模式

---

## 不要做的事

- 不要重構現有的 `Chatbox`、`ThinkingIndicator`、`WelcomeBlock` widget
- 不要更動 `settings.py` 的設定邏輯
- 不要安裝或整合 FalkorDB / Graphiti / mem0（Phase 2）
- LightRAG 後端**保持 JSON/NetworkX**（預設），不要換 Neo4j
- 不要把 `tui.py` 拆成多個檔案（除非超過 400 行）
- 不要動 `codex_client.py`
