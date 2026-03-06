# De-insight v0.3 技術規格文件

> 給 Claude Code 的實作指南。**閱讀完整文件再動工。先讀完，再寫第一行程式碼。**

---

## 零、讀這份文件前先讀

- `tui.py` — TUI 主體（1600+ 行，入口點）
- `panels.py` — ResearchPanel、MemoryPanel widgets
- `memory/store.py` — SQLite async CRUD
- `rag/knowledge_graph.py` — LightRAG 封裝
- `settings.py` — Provider/ENV 管理
- `backend/prompts/curator.py` — 策展人人格提示詞

v0.2 已完成的功能（不要重做）：
- 策展人人格
- 右側面板（ResearchPanel + MemoryPanel）
- LightRAG 知識庫、LanceDB 向量索引
- SQLite 記憶系統
- 三條對話路徑（Codex CLI / 直接 API / FastAPI 後端）

---

## 一、v0.3 範圍說明

### 本版本：核心對話體驗

| 功能 | 主要新增檔案 | 修改檔案 |
|------|------------|---------|
| A. 專案系統 | `projects/manager.py` | `tui.py`、`memory/store.py`、`rag/knowledge_graph.py` |
| E. 記憶確認流程 | 無新增 | `tui.py`、`memory/thought_tracker.py` |
| H. 策展人互動提問 | `interaction/prompt_parser.py` | `tui.py`、`backend/prompts/curator.py` |
| F. Markdown 渲染改善 | 無新增 | `tui.py`（Chatbox）、`backend/prompts/curator.py` |

**不做（推到 v0.4）**：批量文獻匯入（B）、Obsidian 雙向連結（C）、關聯圖 UI（D）。

**永不做**：多使用者架構（G）——專案系統已滿足需求，使用者系統只增加複雜度。

### v0.4 預留：知識庫擴充

B、C、D 三個功能一起做，它們共享資料結構（文獻 → 連結 → 圖）。

---

## 二、目錄結構（v0.3 後）

```
project-root/
├── tui.py                    # App class + 對話流程（目標壓在 1200 行以內）
├── modals.py                 # 新建：所有 ModalScreen 集中在這裡
├── panels.py                 # 原有，v0.3 不動
├── settings.py
├── codex_client.py
├── .env
├── backend/
│   ├── main.py
│   ├── routers/chat.py
│   ├── services/llm.py
│   └── prompts/
│       ├── curator.py        # 修改：加互動提問格式 + Callout 語法指引
│       ├── foucault.py
│       └── memory_prompts.py
├── memory/
│   ├── store.py              # 修改：加 project_id 欄位
│   └── thought_tracker.py   # 修改：extract_memories() 只回傳候選
├── rag/
│   └── knowledge_graph.py   # 修改：get_rag() 加 project_id
├── projects/
│   ├── __init__.py
│   └── manager.py            # 新建
├── interaction/
│   ├── __init__.py
│   └── prompt_parser.py      # 新建
└── data/
    ├── app.db                # 全域：專案清單（無使用者表）
    ├── memories.db           # 記憶（跨專案）
    ├── lancedb/              # 向量索引（跨專案）
    └── projects/
        └── {project_id}/
            └── lightrag/     # per-project 知識圖譜
```

### 關於 tui.py 拆分

目前 1600+ 行，v0.3 預計再加 600 行。原規則「不拆 tui.py」**本版起解除**。

**允許**：所有 ModalScreen 子類別移到 `modals.py`。
**不允許**：不要動 Chatbox、ThinkingIndicator、WelcomeBlock；不要拆 `_stream_response()`。

---

## 三、動工前必做：LightRAG 熱切換驗證

**在寫任何 UI 之前，先執行這個測試。測試通過才繼續。**

```python
# test_rag_switch.py
import asyncio, gc
from rag.knowledge_graph import get_rag

async def test():
    rag_a = get_rag(project_id="test_proj_a")
    await rag_a.ainsert("包豪斯預設了功能的穩定性")
    del rag_a; gc.collect()

    # B 是空的，不應該找到 A 的資料
    rag_b = get_rag(project_id="test_proj_b")
    result_b = await rag_b.aquery("包豪斯")
    assert "包豪斯" not in result_b, "隔離失敗：B 讀到了 A 的資料"
    del rag_b; gc.collect()

    # A 應該還在
    rag_a2 = get_rag(project_id="test_proj_a")
    result_a = await rag_a2.aquery("包豪斯")
    assert "包豪斯" in result_a, "持久化失敗：A 的資料消失了"

    print("LightRAG 熱切換測試通過")

asyncio.run(test())
```

測試失敗的處理方案：

| 失敗原因 | 解法 |
|---------|------|
| LightRAG process 層快取 | `gc.collect()` + 等 0.5s 再重試 |
| 無法在 process 內隔離 | 把 LightRAG 移到 FastAPI 後端，透過 API 切換 |
| 以上都失敗 | 切換專案時 `os.execv()` 重啟 TUI 進程 |

---

## 四、功能 A：專案系統

### A1. app.db Schema

```sql
CREATE TABLE IF NOT EXISTS projects (
    id          TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    description TEXT DEFAULT '',
    color       TEXT DEFAULT '#6b7280',
    created_at  DATETIME DEFAULT (datetime('now', 'localtime')),
    last_active DATETIME DEFAULT (datetime('now', 'localtime'))
);
```

注意：**沒有 users 表**。

### A2. projects/manager.py（完整實作）

```python
import uuid
from pathlib import Path
import aiosqlite

APP_DB = Path("data/app.db")

class ProjectManager:

    async def _ensure_db(self) -> None:
        APP_DB.parent.mkdir(parents=True, exist_ok=True)
        async with aiosqlite.connect(APP_DB) as db:
            await db.execute("""
                CREATE TABLE IF NOT EXISTS projects (
                    id TEXT PRIMARY KEY, name TEXT NOT NULL,
                    description TEXT DEFAULT '', color TEXT DEFAULT '#6b7280',
                    created_at DATETIME DEFAULT (datetime('now','localtime')),
                    last_active DATETIME DEFAULT (datetime('now','localtime'))
                )
            """)
            await db.commit()

    async def create_project(self, name: str, description: str = "") -> dict:
        await self._ensure_db()
        pid = str(uuid.uuid4())
        async with aiosqlite.connect(APP_DB) as db:
            await db.execute(
                "INSERT INTO projects (id,name,description) VALUES (?,?,?)",
                (pid, name, description))
            await db.commit()
        self.get_project_data_dir(pid).mkdir(parents=True, exist_ok=True)
        return await self.get_project(pid)

    async def list_projects(self) -> list[dict]:
        await self._ensure_db()
        async with aiosqlite.connect(APP_DB) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT * FROM projects ORDER BY last_active DESC"
            ) as cur:
                return [dict(r) for r in await cur.fetchall()]

    async def get_project(self, project_id: str) -> dict | None:
        async with aiosqlite.connect(APP_DB) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT * FROM projects WHERE id=?", (project_id,)
            ) as cur:
                row = await cur.fetchone()
                return dict(row) if row else None

    async def delete_project(self, project_id: str) -> None:
        import shutil
        async with aiosqlite.connect(APP_DB) as db:
            await db.execute("DELETE FROM projects WHERE id=?", (project_id,))
            await db.commit()
        d = self.get_project_data_dir(project_id)
        if d.exists():
            shutil.rmtree(d)

    async def touch_project(self, project_id: str) -> None:
        async with aiosqlite.connect(APP_DB) as db:
            await db.execute(
                "UPDATE projects SET last_active=datetime('now','localtime') WHERE id=?",
                (project_id,))
            await db.commit()

    def get_project_data_dir(self, project_id: str) -> Path:
        return Path(f"data/projects/{project_id}")
```

### A3. rag/knowledge_graph.py 修改

```python
def get_rag(project_id: str = "default") -> LightRAG:
    working_dir = Path(f"data/projects/{project_id}/lightrag")
    working_dir.mkdir(parents=True, exist_ok=True)
    # 其餘初始化邏輯不變，只換 working_dir
    ...
```

### A4. memory/store.py 修改

`memories` 表加欄位（啟動時 migration）：

```python
async with db.execute(
    "ALTER TABLE memories ADD COLUMN project_id TEXT DEFAULT NULL"
):
    pass  # 若欄位已存在會拋例外，catch OperationalError 忽略即可
```

`get_memories()` 加 `project_id` 過濾參數：

```python
async def get_memories(type=None, limit=20, project_id=None) -> list[dict]:
    ...
```

### A5. TUI 整合（tui.py）

```python
# App.__init__
self.current_project: dict | None = None
self._project_manager = ProjectManager()
```

新快捷鍵 `ctrl+p` → 開啟 `ProjectModal`。

切換專案：

```python
async def _switch_project(self, project: dict) -> None:
    import gc
    self.current_project = project
    await self._project_manager.touch_project(project["id"])
    self._rag_instance = None
    gc.collect()
    self.messages = []
    await self.query_one("#messages").remove_children()
    await self._show_welcome()
    self._refresh_memory_panel()
    self._refresh_research_panel()
    self._update_menubar()
    self.notify(f"已切換到：{project['name']}", timeout=2)
```

啟動時偵測 v0.2 舊資料並提示（on_mount 開頭）：

```python
from pathlib import Path
if Path("data/lightrag").exists() and not Path("data/projects").exists():
    self.notify(
        "偵測到 v0.2 知識庫（data/lightrag/），請手動搬移或重新匯入。",
        severity="warning", timeout=10)
```

### A6. ProjectModal（modals.py）

```
┌──────────────────────────────────────┐
│ ◇ 專案                         [新增] │
├──────────────────────────────────────┤
│ ● 背面 The Back Side            剛剛  │
│   攝影系列                      3天前  │
│   建築論述草稿                   1週前  │
├──────────────────────────────────────┤
│                    [切換]  [刪除]      │
└──────────────────────────────────────┘
  Escape 關閉
```

使用 `ListView` + `ListItem`，`dismiss()` 回傳選中的 project dict。

---

## 五、功能 E：記憶確認流程

### E1. 設計

1. 對話結束後背景呼叫 `extract_memories()`
2. 候選存入 `self._pending_memories`，MenuBar 顯示計數
3. 使用者點擊計數 → `MemoryConfirmModal` → 確認後才 `add_memory()`

### E2. MenuBar 提示

```
de-insight  ● 背面  感性  claude-sonnet-4-6  💡 3 待確認
```

### E3. MemoryConfirmModal（modals.py）

```
┌────────────────────────────────────────────┐
│ 這段對話有什麼你想留下來的嗎？                │
├────────────────────────────────────────────┤
│ ☑  [洞見] 功能主義預設了使用者永遠面向正面    │
│    來源：「我覺得椅子的背面從來沒有被設計...」 │
│ ☑  [問題] 如果背面被設計，使用者會看嗎？     │
│ ☐  [反應] 對包豪斯有些不滿但說不清楚        │
├────────────────────────────────────────────┤
│          [全部儲存]  [只存勾選]  [略過]      │
└────────────────────────────────────────────┘
```

使用 `SelectionList`，預設全勾。

### E4. tui.py 修改

```python
# App.__init__
self._pending_memories: list[dict] = []

# _auto_extract_memories 改版
@work(exclusive=False)
async def _auto_extract_memories(self, user_text: str) -> None:
    try:
        from memory.thought_tracker import extract_memories
        items = await extract_memories(user_text, self._quick_llm_call)
        if not items:
            return
        self._pending_memories.extend(items)
        self.call_from_thread(self._update_menubar_pending_count)
    except Exception:
        pass
```

`memory/thought_tracker.py`：`extract_memories()` 移除 `add_memory()` 呼叫，只回傳候選清單。

---

## 六、功能 H：策展人互動提問

### H1. 設計原則

策展人在回應末尾嵌入互動標記，TUI 解析後彈出 Modal，使用者的選擇自動送入對話。

**遞迴深度限制**：`self._interactive_depth` 計數器，超過 3 層停止。

### H2. 語法規格

```
<<SELECT: 問題>>
- 選項A
- 選項B
<</SELECT>>

<<MULTI: 問題>>
- 項目A
- 項目B
<</MULTI>>

<<INPUT: 提示文字>>
<</INPUT>>

<<CONFIRM: 確認句子>>
<</CONFIRM>>
```

### H3. interaction/prompt_parser.py（新建）

```python
import re
from dataclasses import dataclass, field

@dataclass
class InteractiveBlock:
    type: str
    prompt: str
    choices: list[str] = field(default_factory=list)
    start: int = 0
    end: int = 0

SELECT_RE  = re.compile(r'<<SELECT:\s*(.+?)>>(.*?)<</SELECT>>',  re.DOTALL)
MULTI_RE   = re.compile(r'<<MULTI:\s*(.+?)>>(.*?)<</MULTI>>',    re.DOTALL)
INPUT_RE   = re.compile(r'<<INPUT:\s*(.+?)>>\s*<</INPUT>>',       re.DOTALL)
CONFIRM_RE = re.compile(r'<<CONFIRM:\s*(.+?)>>\s*<</CONFIRM>>', re.DOTALL)

def parse_interactive_blocks(text: str) -> tuple[str, list[InteractiveBlock]]:
    blocks = []

    def _choices(raw):
        return [l.lstrip('- ').strip() for l in raw.strip().splitlines()
                if l.strip().startswith('-') and l.strip() != '-']

    for m in SELECT_RE.finditer(text):
        blocks.append(InteractiveBlock('select', m.group(1).strip(),
            _choices(m.group(2)), m.start(), m.end()))
    for m in MULTI_RE.finditer(text):
        blocks.append(InteractiveBlock('multi', m.group(1).strip(),
            _choices(m.group(2)), m.start(), m.end()))
    for m in INPUT_RE.finditer(text):
        blocks.append(InteractiveBlock('input', m.group(1).strip(),
            start=m.start(), end=m.end()))
    for m in CONFIRM_RE.finditer(text):
        blocks.append(InteractiveBlock('confirm', m.group(1).strip(),
            start=m.start(), end=m.end()))

    blocks.sort(key=lambda b: b.start)
    clean = CONFIRM_RE.sub('', INPUT_RE.sub('', MULTI_RE.sub('', SELECT_RE.sub('', text))))
    return clean.strip(), blocks
```

加單元測試（`test_prompt_parser.py`）：

```python
from interaction.prompt_parser import parse_interactive_blocks

def test_select():
    text = "正文\n<<SELECT: 問題>>\n- 選A\n- 選B\n<</SELECT>>"
    clean, blocks = parse_interactive_blocks(text)
    assert clean == "正文"
    assert blocks[0].type == "select"
    assert blocks[0].choices == ["選A", "選B"]

def test_no_blocks():
    clean, blocks = parse_interactive_blocks("普通回應")
    assert clean == "普通回應" and blocks == []
```

### H4. InteractivePromptModal（modals.py）

```python
class InteractivePromptModal(ModalScreen):
    DEFAULT_CSS = """
    InteractivePromptModal { align: center middle; }
    InteractivePromptModal > Vertical {
        width: 64; height: auto; max-height: 85%;
        background: #0d1117; border: solid #30363d;
        border-title-color: #7dd3fc; padding: 1 2;
    }
    InteractivePromptModal Label.prompt {
        color: #f0f6fc; text-style: bold; margin-bottom: 1; width: 100%;
    }
    InteractivePromptModal RadioSet { border: none; height: auto; max-height: 16; margin-bottom: 1; }
    InteractivePromptModal SelectionList { border: solid #1e2d3d; height: auto; max-height: 16; margin-bottom: 1; }
    InteractivePromptModal Input { border: solid #30363d; margin: 1 0; }
    InteractivePromptModal .buttons { height: 3; align: right middle; margin-top: 1; }
    InteractivePromptModal Button { margin-left: 1; min-width: 8; }
    """

    def __init__(self, block) -> None:
        super().__init__()
        self.block = block

    def compose(self) -> ComposeResult:
        b = self.block
        with Vertical() as v:
            v.border_title = "◇ 策展人"
            yield Label(b.prompt, classes='prompt')
            yield Rule()
            if b.type == 'select':
                with RadioSet(id='radio'):
                    for c in b.choices: yield RadioButton(c)
            elif b.type == 'multi':
                yield SelectionList(*[(c, c) for c in b.choices], id='multi')
            elif b.type == 'input':
                yield Input(placeholder='輸入你的回應...', id='text-input')
            yield Rule()
            with Horizontal(classes='buttons'):
                if b.type == 'confirm':
                    yield Button('是，繼續', variant='primary', id='yes')
                    yield Button('不，等等', variant='default', id='no')
                else:
                    yield Button('確認', variant='primary', id='confirm')
                    yield Button('略過', variant='default', id='skip')

    @on(Button.Pressed, '#confirm')
    def do_confirm(self) -> None:
        b = self.block
        if b.type == 'select':
            idx = self.query_one(RadioSet).pressed_index
            self.dismiss(b.choices[idx] if idx is not None else None)
        elif b.type == 'multi':
            self.dismiss(list(self.query_one(SelectionList).selected) or None)
        elif b.type == 'input':
            val = self.query_one(Input).value.strip()
            self.dismiss(val or None)

    @on(Button.Pressed, '#yes')
    def do_yes(self)  -> None: self.dismiss('yes')
    @on(Button.Pressed, '#no')
    def do_no(self)   -> None: self.dismiss('no')
    @on(Button.Pressed, '#skip')
    def do_skip(self) -> None: self.dismiss(None)

    def on_key(self, event) -> None:
        if event.key == 'escape': self.dismiss(None)
        elif event.key == 'enter' and self.block.type != 'input': self.do_confirm()
```

### H5. tui.py 整合

```python
# App.__init__
self._interactive_depth: int = 0
```

在 `_stream_response()` 的 `self.messages.append(...)` 之後：

```python
from interaction.prompt_parser import parse_interactive_blocks
import re

clean_content, interactive_blocks = parse_interactive_blocks(full_content)

if re.search(r'<<\w+', clean_content):  # 格式漂移偵測
    self.notify("策展人格式未完整解析", severity="warning", timeout=4)

if interactive_blocks:
    bubble.stream_update(clean_content)
    await bubble.finalize_stream()
    if self._interactive_depth < 3:
        self._handle_interactive_blocks(interactive_blocks)
    else:
        self._interactive_depth = 0
        self.notify("互動提問深度上限，請直接輸入", timeout=3)
```

```python
@work(exclusive=True)
async def _handle_interactive_blocks(self, blocks: list) -> None:
    collected = []
    for block in blocks:
        result = await self.app.push_screen_wait(InteractivePromptModal(block))
        if result is None: continue
        if isinstance(result, list):   user_reply = '、'.join(result)
        elif result == 'yes':          user_reply = '是，繼續。'
        elif result == 'no':           user_reply = '不，我還想調整。'
        else:                          user_reply = result
        collected.append(user_reply)
    if collected:
        await self._send_as_user('\n'.join(collected))

async def _send_as_user(self, text: str) -> None:
    if not text.strip(): return
    self._interactive_depth += 1
    try:
        self._add_message_bubble('user', text)
        self.messages.append({'role': 'user', 'content': text})
        await self._stream_response(text)
    finally:
        self._interactive_depth -= 1
```

### H6. curator.py Prompt 更新

新增兩個常數並加入 `get_system_prompt()`：

```python
INTERACTIVE_PROMPTS_GUIDE = """
# 互動提問格式

需要使用者選擇或輸入時，把標記放在回應最末尾：

單選：<<SELECT: 問題>>\n- 選項\n<</SELECT>>
多選：<<MULTI: 問題>>\n- 項目\n<</MULTI>>
輸入：<<INPUT: 提示>><</INPUT>>
確認：<<CONFIRM: 確認句>><</CONFIRM>>

使用時機：
- 「我想寫論述了」三步驟 → 每步用 CONFIRM 確認
- 使用者方向不明確 → 用 SELECT 給選項
- 需要使用者說出某個想法 → 用 INPUT
- 感性模式少用，一次最多一個標記
""".strip()

CALLOUT_GUIDE = """
# 回應格式（選用）

強調洞見：> [!INSIGHT]\n> 內容
深度問題：> [!QUESTION]\n> 內容
引用理論：> [!THEORY]\n> 內容

不要濫用，有意義時才用。
""".strip()

def get_system_prompt(mode="emotional", memory_summary="", knowledge_content="") -> str:
    mode_block = EMOTIONAL_MODE if mode == "emotional" else RATIONAL_MODE
    prompt = (CURATOR_BASE + "\n\n" + INTERACTIVE_PROMPTS_GUIDE
              + "\n\n" + CALLOUT_GUIDE + "\n\n" + mode_block)
    if memory_summary:
        prompt += "\n\n" + MEMORY_INJECTION.format(memory_summary=memory_summary)
    if knowledge_content:
        prompt += "\n\n" + KNOWLEDGE_INJECTION.format(knowledge_content=knowledge_content)
    return prompt
```

---

## 七、功能 F：Markdown 渲染改善

### F1. 方案

保留 Textual 5.0 `Markdown` widget，加深度 CSS 客製化。
策展人用 `> [!INSIGHT]` 等 Callout blockquote 格式（Obsidian 相容）。

### F2. CSS（加入 tui.py）

```css
Markdown { background: transparent; padding: 0 1; }
Markdown H1 { color: #f0f6fc; text-style: bold; border-bottom: solid #30363d; margin-bottom: 1; }
Markdown H2 { color: #c9d1d9; text-style: bold; margin-top: 1; }
Markdown H3 { color: #8b949e; text-style: bold italic; }
Markdown CodeBlock { background: #161b22; border-left: thick #30363d; padding: 0 1; margin: 1 0; }
Markdown :inline-code { color: #7dd3fc; background: #1e2d3d; }
Markdown Blockquote { border-left: thick #30363d; color: #8b949e; padding: 0 1; }
Markdown :link { color: #7dd3fc; text-style: underline; }
Markdown :strong { color: #f0f6fc; text-style: bold; }
Markdown HorizontalRule { color: #30363d; }
```

---

## 八、資料遷移（v0.2 → v0.3）

| 資料 | v0.2 路徑 | v0.3 路徑 |
|------|----------|----------|
| 記憶 | `data/memories.db` | `data/memories.db`（不變） |
| LightRAG | `data/lightrag/` | `data/projects/{id}/lightrag/`（路徑有變） |
| 向量索引 | `data/lancedb/` | `data/lancedb/`（不變） |

不做自動遷移，啟動時偵測舊路徑並顯示警告提示。

---

## 九、新增斜線指令

```python
"/project": "切換專案",
"/memory":  "記憶待確認",
```

---

## 十、修改清單（精確）

### 新建檔案

| 檔案 | 說明 |
|------|------|
| `modals.py` | ProjectModal、MemoryConfirmModal、InteractivePromptModal |
| `projects/__init__.py` | 空 |
| `projects/manager.py` | 專案 CRUD |
| `interaction/__init__.py` | 空 |
| `interaction/prompt_parser.py` | 互動標記解析 |
| `test_rag_switch.py` | LightRAG 熱切換驗證腳本 |
| `test_prompt_parser.py` | parser 單元測試 |

### 修改檔案

| 檔案 | 修改內容 |
|------|---------|
| `tui.py` | 加 current_project、_pending_memories、_interactive_depth；加 _switch_project、_handle_interactive_blocks、_send_as_user、_update_menubar_pending_count；加 Markdown CSS；import modals |
| `backend/prompts/curator.py` | 加 INTERACTIVE_PROMPTS_GUIDE、CALLOUT_GUIDE；更新 get_system_prompt() |
| `memory/store.py` | memories 表加 project_id；get_memories() 加過濾參數 |
| `memory/thought_tracker.py` | extract_memories() 移除 add_memory() 呼叫 |
| `rag/knowledge_graph.py` | get_rag() 加 project_id；working dir per-project |

### 絕對不要動的檔案

- `codex_client.py`
- `panels.py`（v0.4 才動）
- `settings.py`
- `backend/routers/chat.py`
- `backend/services/llm.py`
- `memory/vectorstore.py`

---

## 十一、驗收標準

1. `test_rag_switch.py` 輸出「LightRAG 熱切換測試通過」
2. `python tui.py` 啟動正常，佈局和 v0.2 一致
3. `ctrl+p` 開啟專案管理 → 新增 → 切換 → MenuBar 更新、對話清空
4. 切換到新專案後問問題，知識庫回應是空的
5. 對話後 MenuBar 出現「💡 N 待確認」→ 點擊 → 勾選 → 儲存 → 記憶面板更新
6. 說「我想寫論述了」→ 策展人觸發 `<<CONFIRM>>` → 彈出確認框 → 「是，繼續」→ 繼續第二步
7. 策展人觸發 `<<SELECT>>` → RadioSet 彈出 → 選一個 → 自動送入對話繼續
8. 連續三次互動後，第四次 `<<CONFIRM>>` 不彈出（觸發深度上限通知）
9. Markdown 的 H2 有視覺分量，`> [!INSIGHT]` 渲染成帶色 blockquote
10. 原有功能完整保留：感性/理性切換、設定頁面、三條對話路徑

---

## 十二、執行順序

**每步完成後跑一次 `python tui.py` 確認無 import error 再往下。**

**Step 0：LightRAG 熱切換驗證**
執行 `test_rag_switch.py`。通過才繼續，失敗先解決。

**Step 1：curator.py prompt 更新**
加 `INTERACTIVE_PROMPTS_GUIDE` 和 `CALLOUT_GUIDE`。
**人工測試**：對話說「我想寫論述了」，確認 Claude 輸出 `<<CONFIRM>>` 格式。
Claude 不遵守格式就調整 prompt，直到通過再做後面的 UI。

**Step 2：H 互動提問**
寫 `prompt_parser.py`，跑 `test_prompt_parser.py`。
寫 `InteractivePromptModal`，加到 `modals.py`，接入 `_stream_response()`。

**Step 3：E 記憶確認**
修改 `thought_tracker.py`（移除 `add_memory()`）。
加 `_pending_memories`、MenuBar 計數、`MemoryConfirmModal`。

**Step 4：A 專案系統**
手動測試 `ProjectManager` CRUD。
修改 `knowledge_graph.py` 和 `store.py`。
寫 `ProjectModal`，接入 `_switch_project()`。

**Step 5：F Markdown CSS**
加 CSS，視覺確認。完全獨立，任何時候都可以插入做。
