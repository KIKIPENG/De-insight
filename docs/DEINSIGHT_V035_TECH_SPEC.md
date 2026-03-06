# De-insight v0.3.5 技術規格文件

> 給 Claude Code 的實作指南。**閱讀完整文件再動工。先讀完，再寫第一行程式碼。**

---

## 零、前提

v0.3 已完成。v0.3.5 只做兩件事：

1. **ConversationStore** — 對話歷史持久化
2. **AppState** — 散落狀態集中管理

不加任何新功能，不動 UI 佈局，不動 panels.py、modals.py、settings.py。

---

## 一、範圍

| 任務 | 新增檔案 | 修改檔案 |
|------|---------|---------|
| ConversationStore | `conversation/store.py` | `tui.py`、`WelcomeBlock` |
| AppState | 無 | `tui.py` |

---

## 二、ConversationStore

### 2.1 Schema（`data/conversations.db`）

```sql
CREATE TABLE IF NOT EXISTS conversations (
    id         TEXT PRIMARY KEY,
    project_id TEXT DEFAULT NULL,
    title      TEXT NOT NULL DEFAULT '未命名對話',
    created_at DATETIME DEFAULT (datetime('now', 'localtime')),
    updated_at DATETIME DEFAULT (datetime('now', 'localtime'))
);

CREATE TABLE IF NOT EXISTS messages (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id TEXT NOT NULL,
    role            TEXT NOT NULL,
    content         TEXT NOT NULL,
    created_at      DATETIME DEFAULT (datetime('now', 'localtime')),
    FOREIGN KEY (conversation_id) REFERENCES conversations(id)
);
```

### 2.2 `conversation/store.py`（新建）

```python
"""對話歷史持久化。"""

import uuid
from pathlib import Path
import aiosqlite

DB_PATH = Path("data/conversations.db")

class ConversationStore:

    async def _ensure_db(self) -> None:
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        async with aiosqlite.connect(DB_PATH) as db:
            db.row_factory = aiosqlite.Row
            await db.executescript("""
                CREATE TABLE IF NOT EXISTS conversations (
                    id         TEXT PRIMARY KEY,
                    project_id TEXT DEFAULT NULL,
                    title      TEXT NOT NULL DEFAULT '未命名對話',
                    created_at DATETIME DEFAULT (datetime('now', 'localtime')),
                    updated_at DATETIME DEFAULT (datetime('now', 'localtime'))
                );
                CREATE TABLE IF NOT EXISTS messages (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    conversation_id TEXT NOT NULL,
                    role            TEXT NOT NULL,
                    content         TEXT NOT NULL,
                    created_at      DATETIME DEFAULT (datetime('now', 'localtime'))
                );
            """)
            await db.commit()

    async def create_conversation(self, project_id: str | None = None) -> str:
        await self._ensure_db()
        cid = str(uuid.uuid4())
        async with aiosqlite.connect(DB_PATH) as db:
            await db.execute(
                "INSERT INTO conversations (id, project_id) VALUES (?, ?)",
                (cid, project_id)
            )
            await db.commit()
        return cid

    async def add_message(self, conversation_id: str, role: str, content: str) -> None:
        async with aiosqlite.connect(DB_PATH) as db:
            await db.execute(
                "INSERT INTO messages (conversation_id, role, content) VALUES (?, ?, ?)",
                (conversation_id, role, content)
            )
            await db.execute(
                "UPDATE conversations SET updated_at = datetime('now','localtime') WHERE id = ?",
                (conversation_id,)
            )
            await db.commit()

    async def set_title(self, conversation_id: str, title: str) -> None:
        async with aiosqlite.connect(DB_PATH) as db:
            await db.execute(
                "UPDATE conversations SET title = ? WHERE id = ?",
                (title, conversation_id)
            )
            await db.commit()

    async def get_messages(self, conversation_id: str) -> list[dict]:
        async with aiosqlite.connect(DB_PATH) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT role, content FROM messages WHERE conversation_id = ? ORDER BY id ASC",
                (conversation_id,)
            ) as cur:
                return [dict(r) for r in await cur.fetchall()]

    async def list_conversations(self, project_id: str | None = None) -> list[dict]:
        """列出對話，依 updated_at 倒序。project_id=None 回傳全部。"""
        async with aiosqlite.connect(DB_PATH) as db:
            db.row_factory = aiosqlite.Row
            if project_id:
                async with db.execute(
                    "SELECT * FROM conversations WHERE project_id = ? ORDER BY updated_at DESC",
                    (project_id,)
                ) as cur:
                    return [dict(r) for r in await cur.fetchall()]
            else:
                async with db.execute(
                    "SELECT * FROM conversations ORDER BY updated_at DESC"
                ) as cur:
                    return [dict(r) for r in await cur.fetchall()]

    async def delete_conversation(self, conversation_id: str) -> None:
        async with aiosqlite.connect(DB_PATH) as db:
            await db.execute("DELETE FROM messages WHERE conversation_id = ?", (conversation_id,))
            await db.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))
            await db.commit()
```

### 2.3 目錄結構

```
conversation/
├── __init__.py
└── store.py
```

---

## 三、AppState

### 3.1 定義（加在 `tui.py` 頂部，App class 之前）

```python
from dataclasses import dataclass, field

@dataclass
class AppState:
    current_project: dict | None = None
    pending_memories: list[dict] = field(default_factory=list)
    interactive_depth: int = 0
    current_conversation_id: str | None = None
    cached_memory_count: int = 0
    current_interactive_block: object = None  # InteractiveBlock | None
```

### 3.2 替換規則（tui.py 全域搜尋替換）

| 舊寫法 | 新寫法 |
|--------|--------|
| `self.current_project` | `self.state.current_project` |
| `self._pending_memories` | `self.state.pending_memories` |
| `self._interactive_depth` | `self.state.interactive_depth` |
| `self._cached_memory_count` | `self.state.cached_memory_count` |
| `self._current_interactive_block` | `self.state.current_interactive_block` |

`self.messages` **保留不動**，它屬於對話流程，v0.4 再處理。

### 3.3 App.__init__ 修改

```python
# 舊
self.current_project: dict | None = None
self._pending_memories: list[dict] = []
self._interactive_depth: int = 0
self._cached_memory_count: int = 0
self._current_interactive_block = None

# 新
self.state = AppState()
```

---

## 四、TUI 整合

### 4.1 App.__init__ 新增

```python
from conversation.store import ConversationStore
self._conv_store = ConversationStore()
```

### 4.2 新對話流程（`action_new_chat`）

```python
def action_new_chat(self) -> None:
    self.messages.clear()
    self.state.current_conversation_id = None   # 新對話，ID 留空，第一條訊息時才建立
    # 其餘原有邏輯不變
    ...
```

### 4.3 送出訊息時建立／延續對話（`_submit_chat`）

在 `self.messages.append({"role": "user", ...})` 之後加：

```python
# 第一條使用者訊息：建立對話記錄 + 自動生成標題
if self.state.current_conversation_id is None:
    project_id = self.state.current_project["id"] if self.state.current_project else None
    self.state.current_conversation_id = await self._conv_store.create_conversation(project_id)
    # 用前 30 字作為標題
    title = text[:30].strip().replace("\n", " ")
    await self._conv_store.set_title(self.state.current_conversation_id, title)

await self._conv_store.add_message(self.state.current_conversation_id, "user", text)
```

### 4.4 AI 回應後存入（`_stream_response`）

在 `self.messages.append({"role": "assistant", "content": ...})` 之後加：

```python
if self.state.current_conversation_id:
    await self._conv_store.add_message(
        self.state.current_conversation_id, "assistant", full_content
    )
```

三條路徑（codex-cli / 直接 API / FastAPI 後端）都要加，位置是各自的 `self.messages.append` 之後。

### 4.5 切換專案時重置 conversation_id

`_do_switch_project()` 裡加：

```python
self.state.current_conversation_id = None
```

### 4.6 載入歷史對話

新增 method：

```python
async def _load_conversation(self, conversation_id: str) -> None:
    """從 ConversationStore 載入對話歷史，替換目前的 messages。"""
    messages = await self._conv_store.get_messages(conversation_id)
    if not messages:
        return

    self.state.current_conversation_id = conversation_id
    self.messages = list(messages)

    # 清空畫面，重新渲染
    msg_container = self.query_one("#messages")
    await msg_container.remove_children()
    for m in messages:
        bubble = self._add_message_bubble(m["role"], m["content"])
    self._scroll_to_bottom()
```

---

## 五、WelcomeBlock 改版

### 5.1 新的開始畫面

```
◆ De-insight  你的思想策展人

  把還說不清楚的東西說清楚。
  想法來自你，它幫你找到骨架。

────────────────────────────────

◇ 功能

  △ 策展人對話　感性／理性模式切換（ctrl+e）
  △ 知識庫　　　匯入文獻，對話時自動引用
  △ 記憶系統　　留下洞見、問題、感性反應
  △ 專案管理　　不同創作脈絡分開（ctrl+p）

────────────────────────────────

◇ 一個創作者的工作路徑

  累積文獻與閱讀　→　和策展人反覆對話
        ↓
  沉澱洞見與問題　→　知識庫建立連結
        ↓
      準備好了　→　寫出一份論述

────────────────────────────────

◇ 最近的對話

  2025-03-06 14:32  背面與功能主義的關係
  2025-03-05 22:10  字體安裝的儀式感
  2025-03-04 18:44  包豪斯預設了什麼

  輸入 /help 查看所有指令

────────────────────────────────

  made by KIKI PEGN with love
```

### 5.2 實作方式

`WelcomeBlock` 改為 async 載入對話歷史：

```python
class WelcomeBlock(Vertical):

    def compose(self) -> ComposeResult:
        # 靜態內容（功能、工作路徑）照舊
        ...
        # 對話歷史區塊：先放一個空的容器
        yield Vertical(id="welcome-history")
        ...

    async def on_mount(self) -> None:
        await self._load_recent_conversations()

    async def _load_recent_conversations(self) -> None:
        from conversation.store import ConversationStore
        store = ConversationStore()
        # 取最近 10 條，不過濾 project
        conversations = await store.list_conversations()
        conversations = conversations[:10]

        container = self.query_one("#welcome-history")
        if not conversations:
            await container.mount(
                Static("[dim #484f58]尚無對話記錄[/]")
            )
            return

        for c in conversations:
            updated = c.get("updated_at", "")[:16]  # 截到分鐘
            title = c.get("title", "未命名對話")
            entry = Static(
                f"  [#484f58]{updated}[/]  [#8b949e]{title}[/]",
                classes="history-entry",
                name=c["id"],
            )
            await container.mount(entry)
```

點擊歷史條目：

```python
    def on_click(self, event) -> None:
        widget = event.widget if hasattr(event, 'widget') else None
        if widget and isinstance(widget, Static) and widget.has_class("history-entry"):
            conv_id = widget.name
            if conv_id:
                # 通知 App 載入這個對話
                self.app.call_after_refresh(
                    lambda: self.app._load_conversation(conv_id)
                )
```

CSS 加入：

```css
.history-entry {
    height: 1;
    padding: 0 1;
    color: #8b949e;
}
.history-entry:hover {
    color: #fafafa;
    background: #111111;
    cursor: pointer;
}
```

---

## 六、_send_as_user bug 修正

這個 bug 在 v0.3 就存在，順手修掉：

```python
# 舊（interactive_depth 不會減回來）
async def _send_as_user(self, text: str) -> None:
    self._interactive_depth += 1
    self._scroll_to_bottom()
    self.messages.append({'role': 'user', 'content': text})
    self._stream_response()

# 新
async def _send_as_user(self, text: str) -> None:
    self.state.interactive_depth += 1
    try:
        self._scroll_to_bottom()
        self.messages.append({'role': 'user', 'content': text})
        await self._stream_response()
    finally:
        self.state.interactive_depth -= 1
```

---

## 七、新增目錄結構

```
conversation/
├── __init__.py
└── store.py
```

---

## 八、修改清單（精確）

### 新建檔案

| 檔案 | 說明 |
|------|------|
| `conversation/__init__.py` | 空 |
| `conversation/store.py` | 對話歷史 CRUD |

### 修改檔案

| 檔案 | 修改內容 |
|------|---------|
| `tui.py` | 加 AppState dataclass；App.__init__ 換成 self.state；全域替換五個狀態存取；加 ConversationStore 整合；WelcomeBlock 加歷史對話顯示；修 _send_as_user bug |

### 不要動的檔案

- `modals.py`
- `panels.py`
- `settings.py`
- `memory/store.py`
- `rag/knowledge_graph.py`
- `projects/manager.py`
- `backend/` 底下所有檔案

---

## 九、驗收標準

1. `python tui.py` 啟動，開始畫面下方出現「最近的對話」區塊
2. 進行一段對話後關閉 TUI，重新啟動，開始畫面出現剛才那段對話的標題和時間
3. 點擊歷史條目，對話內容完整載入，可以繼續聊
4. `ctrl+n` 新對話，開始畫面歷史清單更新
5. 切換專案後，`state.current_conversation_id` 重置為 None
6. 連續互動提問三次後，第四次不再觸發（depth 保護有效）
7. 所有 v0.3 功能正常：記憶確認、專案切換、知識庫匯入

---

## 十、執行順序

**Step 1：AppState**
在 tui.py 頂部加 AppState dataclass，App.__init__ 換成 `self.state = AppState()`，全域搜尋替換五個狀態存取。跑一次 `python tui.py` 確認無 import error 和功能異常。

**Step 2：_send_as_user bug 修正**
順手一起做，一行的事。

**Step 3：ConversationStore**
新建 `conversation/store.py`，手動測試 CRUD：
```python
import asyncio
from conversation.store import ConversationStore

async def test():
    store = ConversationStore()
    cid = await store.create_conversation()
    await store.add_message(cid, "user", "測試訊息")
    await store.set_title(cid, "測試對話")
    msgs = await store.get_messages(cid)
    assert msgs[0]["content"] == "測試訊息"
    convs = await store.list_conversations()
    assert any(c["id"] == cid for c in convs)
    print("ConversationStore 測試通過")

asyncio.run(test())
```

**Step 4：接入 _submit_chat 和 _stream_response**
三條路徑都要加存訊息的呼叫，仔細確認位置正確。

**Step 5：WelcomeBlock 改版**
加歷史對話顯示和點擊載入。最後做，因為它依賴 Step 3 的 store。
