"""De-insight v0.2 — 右側面板 Widget 與 Modal Screen"""

from textual import work
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Static, TextArea

from memory.store import get_memories, delete_memory, get_memory_stats


# ── Right Panel Widgets ──────────────────────────────────────────────


class MemoryItem(Static):
    """單條記憶顯示，可點擊閱覽完整內容。"""

    def __init__(self, mem: dict, **kwargs) -> None:
        self.mem = mem
        icons = {"insight": "💡", "question": "❓", "reaction": "💭"}
        icon = icons.get(mem["type"], "◇")
        label = {"insight": "洞見", "question": "問題", "reaction": "反應"}.get(
            mem["type"], mem["type"]
        )
        content = mem["content"]
        if len(content) > 30:
            content = content[:30] + "…"
        super().__init__(
            f"{icon} [{label}] {content}",
            classes="memory-item",
            **kwargs,
        )

    def on_click(self) -> None:
        def on_result(result: str | None) -> None:
            if result and result.startswith("discuss:"):
                content = result.removeprefix("discuss:")
                self.app._start_discussion_from_memory(content)
            elif result and result.startswith("__insert__:"):
                content = result[len("__insert__:"):]
                self.app.fill_input(content)

        self.app.push_screen(MemoryDetailModal(self.mem), callback=on_result)


class ResearchPanel(VerticalScroll):
    """右上：知識庫查詢結果面板。"""

    def compose(self) -> ComposeResult:
        yield Static(
            "[dim #484f58]ctrl+f  匯入 PDF\nctrl+k  搜尋知識庫[/]",
            id="research-content",
        )


class MemoryPanel(VerticalScroll):
    """右下：記憶面板。"""

    def compose(self) -> ComposeResult:
        yield Static(
            "[dim #484f58]對話後自動記錄洞見[/]",
            id="memory-content",
        )


# ── Modal Screens ────────────────────────────────────────────────────


class ImportModal(ModalScreen[str | None]):
    """匯入 PDF 的 Modal。"""

    CSS = """
    ImportModal { align: center middle; }
    #import-box {
        width: 56; height: auto; padding: 1 2;
        border: round #3a3a3a; background: #0a0a0a;
        border-title-color: #fafafa;
    }
    #import-input {
        margin: 1 0; background: #111111; color: #fafafa;
        border: tall #3a3a3a; padding: 0 1;
    }
    """

    def compose(self) -> ComposeResult:
        box = Vertical(id="import-box")
        box.border_title = "◇ 匯入文件"
        with box:
            yield Static("[#8b949e]輸入檔案路徑或網址[/]")
            yield Input(placeholder="/path/to/file.pdf 或 https://...", id="import-input")
            yield Static("[dim #484f58]支援 PDF 檔案、網頁 URL[/]")
            yield Button("← 回到對話", classes="back-btn")

    def on_mount(self) -> None:
        self.query_one("#import-input", Input).focus()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        path = event.value.strip()
        if path:
            self.dismiss(path)
        else:
            self.dismiss(None)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.has_class("back-btn"):
            self.dismiss(None)

    def on_key(self, event) -> None:
        if event.key == "escape":
            self.dismiss(None)


class SearchModal(ModalScreen[str | None]):
    """搜尋知識庫的 Modal。"""

    CSS = """
    SearchModal { align: center middle; }
    #search-box {
        width: 56; height: auto; padding: 1 2;
        border: round #3a3a3a; background: #0a0a0a;
        border-title-color: #fafafa;
    }
    #search-input {
        margin: 1 0; background: #111111; color: #fafafa;
        border: tall #3a3a3a; padding: 0 1;
    }
    """

    def compose(self) -> ComposeResult:
        box = Vertical(id="search-box")
        box.border_title = "◇ 搜尋知識庫"
        with box:
            yield Static("[#8b949e]輸入搜尋關鍵字[/]")
            yield Input(placeholder="搜尋…", id="search-input")
            yield Button("← 回到對話", classes="back-btn")

    def on_mount(self) -> None:
        self.query_one("#search-input", Input).focus()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        query = event.value.strip()
        if query:
            self.dismiss(query)
        else:
            self.dismiss(None)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.has_class("back-btn"):
            self.dismiss(None)

    def on_key(self, event) -> None:
        if event.key == "escape":
            self.dismiss(None)


class MemoryDetailModal(ModalScreen[str | None]):
    """單條記憶的完整閱覽，可開啟討論。"""

    BINDINGS = [("escape", "close", "關閉")]

    CSS = """
    MemoryDetailModal { align: center middle; }
    #mem-detail-box {
        width: 64; height: auto; max-height: 70%; padding: 1 2;
        border: round #3a3a3a; background: #0a0a0a;
        border-title-color: #fafafa;
    }
    #mem-detail-content {
        height: auto; max-height: 100%; padding: 0 1;
        color: #c9d1d9;
    }
    #mem-detail-meta {
        height: auto; margin: 1 0 0 0; padding: 0 1;
        color: #484f58;
    }
    .detail-actions {
        height: 1; margin: 1 0 0 0;
    }
    .detail-action-btn {
        background: transparent; color: #6e7681;
        border: none; height: 1; min-width: 0;
        margin: 0 1 0 0; padding: 0 1;
    }
    .detail-action-btn:hover { color: #fafafa; }
    #btn-discuss {
        color: #fafafa; background: #1a1a1a;
    }
    """

    def __init__(self, mem: dict) -> None:
        super().__init__()
        self._mem = mem

    def compose(self) -> ComposeResult:
        icons = {"insight": "💡", "question": "❓", "reaction": "💭"}
        labels = {"insight": "洞見", "question": "問題", "reaction": "反應"}
        mem = self._mem
        icon = icons.get(mem["type"], "◇")
        label = labels.get(mem["type"], mem["type"])
        topic = mem.get("topic", "")
        box = Vertical(id="mem-detail-box")
        box.border_title = f"{icon} {label}" + (f"  #{topic}" if topic else "")
        with box:
            yield Static(mem["content"], id="mem-detail-content")
            meta_parts = []
            if mem.get("category"):
                meta_parts.append(f"分類: {mem['category']}")
            if topic:
                meta_parts.append(f"主題: {topic}")
            if mem.get("source"):
                meta_parts.append(f"來源: {mem['source'][:60]}")
            if mem.get("created_at"):
                meta_parts.append(f"時間: {mem['created_at']}")
            if meta_parts:
                yield Static(
                    "[dim]" + "\n".join(meta_parts) + "[/]",
                    id="mem-detail-meta",
                )
            with Horizontal(classes="detail-actions"):
                yield Button(
                    "◆ 用這條開啟討論", id="btn-discuss",
                    classes="detail-action-btn",
                )
                yield Button(
                    "插入對話", id="detail-insert",
                    classes="detail-action-btn",
                )
                yield Button(
                    "[刪除]", id="btn-detail-del",
                    classes="detail-action-btn",
                )
                yield Button("← 回到對話", classes="back-btn detail-action-btn")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.has_class("back-btn"):
            self.dismiss(None)
            return
        bid = event.button.id or ""
        if bid == "btn-discuss":
            # 回傳記憶內容，讓 TUI 開啟討論
            self.dismiss(f"discuss:{self._mem['content']}")
        elif bid == "detail-insert":
            content = self._mem.get("content", "")
            self.dismiss(f"__insert__:{content}")
        elif bid == "btn-detail-del":
            self._do_delete()

    @work(exclusive=True)
    async def _do_delete(self) -> None:
        await delete_memory(self._mem["id"])
        self.notify("已刪除")
        self.dismiss(None)

    def action_close(self) -> None:
        self.dismiss(None)


class MemoryManageModal(ModalScreen[str | None]):
    """記憶/知識資料管理 — 分類、主題、時間，有資料感。"""

    BINDINGS = [("escape", "close", "關閉")]

    CSS = """
    MemoryManageModal { align: center middle; }
    #mem-manage-box {
        width: 72; height: auto; max-height: 85%; padding: 1 2;
        border: round #3a3a3a; background: #0a0a0a;
        border-title-color: #fafafa;
    }
    /* stats bar */
    #mem-stats {
        height: auto; margin: 0 0 1 0; padding: 0;
        color: #484f58;
    }
    /* filter row */
    #mem-filter-row {
        height: 1; margin: 0 0 1 0;
    }
    .filter-btn {
        background: transparent; color: #484f58;
        border: none; height: 1; min-width: 0;
        margin: 0 1 0 0; padding: 0 1;
    }
    .filter-btn:hover { color: #fafafa; }
    .filter-btn.-active { color: #fafafa; background: #1a1a1a; }
    /* category filter row */
    #mem-category-row {
        height: 1; margin: 0 0 1 0;
    }
    .cat-btn {
        background: transparent; color: #484f58;
        border: none; height: 1; min-width: 0;
        margin: 0 1 0 0; padding: 0 1;
    }
    .cat-btn:hover { color: #fafafa; }
    .cat-btn.-active { color: #c9d1d9; background: #1a1a1a; }
    /* separator */
    .mm-sep {
        height: 1; margin: 0; color: #2a2a2a;
    }
    /* memory list */
    #mem-scroll {
        height: auto; max-height: 100%;
    }
    .mem-entry {
        height: auto; margin: 0 0 0 0; padding: 0 1;
        color: #8b949e;
    }
    .mem-entry:hover {
        color: #fafafa; background: #111111;
    }
    .mem-entry-time {
        width: 12; height: 1; color: #484f58;
        margin: 0; padding: 0;
    }
    .mem-entry-topic {
        width: auto; height: 1; color: #6e7681;
        margin: 0 1 0 0; padding: 0;
    }
    .mem-entry-text {
        width: 1fr; height: 1; color: #8b949e;
        margin: 0; padding: 0;
    }
    .mem-del-btn {
        background: transparent; color: #484f58;
        border: none; height: 1; min-width: 3;
        margin: 0; padding: 0;
    }
    .mem-del-btn:hover { color: #ff6b6b; }
    /* time group header */
    .time-header {
        height: 1; margin: 1 0 0 0; padding: 0;
        color: #6e7681;
    }
    """

    def __init__(self) -> None:
        super().__init__()
        self._filter_type: str = ""  # "" = all
        self._filter_category: str = ""  # "" = all
        self._all_mems: list[dict] = []

    def compose(self) -> ComposeResult:
        box = Vertical(id="mem-manage-box")
        box.border_title = "◇ 記憶 / 知識庫"
        with box:
            yield Static("", id="mem-stats")
            with Horizontal(id="mem-filter-row"):
                yield Button("全部", id="flt-all", classes="filter-btn -active", name="")
                yield Button("💡洞見", id="flt-insight", classes="filter-btn", name="insight")
                yield Button("❓問題", id="flt-question", classes="filter-btn", name="question")
                yield Button("💭反應", id="flt-reaction", classes="filter-btn", name="reaction")
            with Horizontal(id="mem-category-row"):
                yield Button("#全部", id="cat-all", classes="cat-btn -active", name="")
                yield Button("#思考方式", id="cat-thinking", classes="cat-btn", name="思考方式")
                yield Button("#美學偏好", id="cat-aesthetic", classes="cat-btn", name="美學偏好")
                yield Button("#創作問題", id="cat-creation", classes="cat-btn", name="創作問題")
                yield Button("#理論連結", id="cat-theory", classes="cat-btn", name="理論連結")
            yield Static("[dim #2a2a2a]" + "─" * 66 + "[/]", classes="mm-sep")
            yield VerticalScroll(id="mem-scroll")
            yield Button("← 回到對話", classes="back-btn")

    def on_mount(self) -> None:
        self._load_all()

    @work(exclusive=True)
    async def _load_all(self) -> None:
        # Stats
        stats = await get_memory_stats()
        total = stats["total"]
        by_type = stats["by_type"]
        by_topic = stats["by_topic"]
        stats_text = f"[#6e7681]{total} 條記憶[/]"
        for t, cnt in by_type.items():
            icons = {"insight": "💡", "question": "❓", "reaction": "💭"}
            stats_text += f"  {icons.get(t, '◇')}{cnt}"
        try:
            self.query_one("#mem-stats", Static).update(stats_text)
        except Exception:
            pass

        # Load memories
        self._all_mems = await get_memories(limit=100)
        await self._render_list()

    async def _render_list(self) -> None:
        try:
            scroll = self.query_one("#mem-scroll", VerticalScroll)
        except Exception:
            return
        await scroll.remove_children()

        mems = self._all_mems
        # Apply filters
        if self._filter_type:
            mems = [m for m in mems if m["type"] == self._filter_type]
        if self._filter_category:
            mems = [m for m in mems if m.get("category") == self._filter_category]

        if not mems:
            await scroll.mount(Static("[dim #484f58]沒有符合的記憶[/]"))
            return

        icons = {"insight": "💡", "question": "❓", "reaction": "💭"}
        # Group by date
        current_date = ""
        for m in mems:
            date_str = (m.get("created_at") or "")[:10]
            if date_str != current_date:
                current_date = date_str
                await scroll.mount(
                    Static(f"[#6e7681]── {date_str or '未知日期'} ──[/]", classes="time-header")
                )
            icon = icons.get(m["type"], "◇")
            time_str = (m.get("created_at") or "")[11:16]  # HH:MM
            topic = m.get("topic", "")
            content = m["content"][:36] + ("…" if len(m["content"]) > 36 else "")
            # Build entry as a clickable row
            row = _MemEntry(m)
            await scroll.mount(row)
            topic_display = f"[#6e7681]#{topic}[/] " if topic else ""
            await row.mount(Static(f"[#484f58]{time_str}[/]", classes="mem-entry-time"))
            await row.mount(Static(f"{icon} {topic_display}{content}",
                                   classes="mem-entry-text"))
            await row.mount(Button("✗", classes="mem-del-btn", name=str(m["id"])))

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.has_class("back-btn"):
            self.dismiss(None)
            return
        bid = event.button.id or ""
        name = event.button.name or ""

        # Type filters
        if bid.startswith("flt-"):
            self._filter_type = name
            for btn in self.query(".filter-btn"):
                btn.remove_class("-active")
            event.button.add_class("-active")
            self._apply_filter()
            return

        # Category filters
        if event.button.has_class("cat-btn"):
            self._filter_category = name
            for btn in self.query(".cat-btn"):
                btn.remove_class("-active")
            event.button.add_class("-active")
            self._apply_filter()
            return

        # Delete
        if event.button.has_class("mem-del-btn"):
            mem_id = int(name or "0")
            if mem_id:
                self._delete_and_reload(mem_id)

    @work(exclusive=True)
    async def _apply_filter(self) -> None:
        await self._render_list()

    @work(exclusive=True)
    async def _delete_and_reload(self, mem_id: int) -> None:
        await delete_memory(mem_id)
        self._all_mems = [m for m in self._all_mems if m["id"] != mem_id]
        await self._render_list()
        self.notify("已刪除")

    def action_close(self) -> None:
        self.dismiss(None)


class _MemEntry(Horizontal):
    """記憶列表中可點擊的單條記錄。"""

    def __init__(self, mem: dict, **kwargs) -> None:
        super().__init__(classes="mem-entry", **kwargs)
        self._mem = mem

    def on_click(self) -> None:
        def on_result(result: str | None) -> None:
            if result and (result.startswith("discuss:") or result.startswith("__insert__:")):
                # Bubble up to MemoryManageModal → TUI
                modal = self.screen
                if isinstance(modal, MemoryManageModal):
                    modal.dismiss(result)
        self.app.push_screen(MemoryDetailModal(self._mem), callback=on_result)


class InsightConfirmModal(ModalScreen[dict | None]):
    """LLM 整理後的洞見確認 Modal。使用者可編輯後儲存或取消。"""

    BINDINGS = [("escape", "cancel", "取消")]

    CSS = """
    InsightConfirmModal { align: center middle; }
    #insight-box {
        width: 60; height: auto; max-height: 80%; padding: 1 2;
        border: round #3a3a3a; background: #0a0a0a;
        border-title-color: #fafafa;
    }
    #insight-type-row {
        height: 1; margin: 1 0 0 0;
    }
    .type-btn {
        background: transparent; color: #484f58;
        border: none; height: 1; min-width: 0;
        margin: 0 1 0 0; padding: 0 1;
    }
    .type-btn:hover { color: #fafafa; }
    .type-btn.-selected { color: #fafafa; background: #1a1a1a; }
    #insight-editor {
        height: 6; margin: 1 0;
        background: #111111; color: #fafafa;
        border: tall #3a3a3a;
    }
    #insight-editor:focus { border: tall #666666; }
    #insight-save-row {
        height: 1; margin: 1 0 0 0;
    }
    #btn-insight-save {
        background: #fafafa; color: #0a0a0a; border: none;
        min-width: 12; margin: 0 1 0 0;
    }
    #btn-insight-cancel {
        background: transparent; color: #484f58; border: none;
        min-width: 8; margin: 0;
    }
    """

    def __init__(self, draft: str, insight_type: str = "insight") -> None:
        super().__init__()
        self._draft = draft
        self._type = insight_type

    def compose(self) -> ComposeResult:
        box = Vertical(id="insight-box")
        box.border_title = "◇ 確認洞見"
        with box:
            yield Static("[#8b949e]LLM 整理了以下洞見，你可以編輯後儲存：[/]")
            with Horizontal(id="insight-type-row"):
                for t, label in [("insight", "洞見"), ("question", "問題"), ("reaction", "反應")]:
                    cls = "type-btn" + (" -selected" if t == self._type else "")
                    yield Button(f"[{label}]", id=f"itype-{t}", classes=cls, name=t)
            yield TextArea(self._draft, id="insight-editor")
            with Horizontal(id="insight-save-row"):
                yield Button("儲存", id="btn-insight-save")
                yield Button("[取消]", id="btn-insight-cancel")
                yield Button("← 回到對話", classes="back-btn")

    def on_mount(self) -> None:
        self.query_one("#insight-editor", TextArea).focus()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.has_class("back-btn"):
            self.dismiss(None)
            return
        bid = event.button.id or ""
        if bid.startswith("itype-"):
            self._type = event.button.name or "insight"
            for btn in self.query(".type-btn"):
                btn.remove_class("-selected")
            event.button.add_class("-selected")
        elif bid == "btn-insight-save":
            content = self.query_one("#insight-editor", TextArea).text.strip()
            if content:
                self.dismiss({"type": self._type, "content": content})
            else:
                self.notify("內容不能為空")
        elif bid == "btn-insight-cancel":
            self.dismiss(None)

    def action_cancel(self) -> None:
        self.dismiss(None)


class MemorySaveModal(ModalScreen[dict | None]):
    """快速儲存記憶點 Modal。"""

    BINDINGS = [("escape", "cancel", "取消")]

    CSS = """
    MemorySaveModal { align: center middle; }
    #mem-box {
        width: 60; height: auto; max-height: 80%; padding: 1 2;
        border: round #3a3a3a; background: #0a0a0a;
        border-title-color: #fafafa;
    }
    #mem-type-row { height: 1; margin: 1 0 0 0; }
    .mtype-btn {
        background: transparent; color: #484f58;
        border: none; height: 1; min-width: 0;
        margin: 0 1 0 0; padding: 0 1;
    }
    .mtype-btn:hover { color: #fafafa; }
    .mtype-btn.-selected { color: #fafafa; background: #1a1a1a; }
    #mem-editor {
        height: 6; margin: 1 0;
        background: #111111; color: #fafafa;
        border: tall #3a3a3a;
    }
    #mem-editor:focus { border: tall #666666; }
    #mem-topic-input {
        margin: 0; background: #111111; color: #fafafa;
        border: tall #3a3a3a; padding: 0 1;
    }
    #mem-topic-input:focus { border: tall #666666; }
    #mem-save-row { height: 1; margin: 1 0 0 0; }
    #btn-mem-save {
        background: #fafafa; color: #0a0a0a; border: none;
        min-width: 12; margin: 0 1 0 0;
    }
    #btn-mem-cancel {
        background: transparent; color: #484f58; border: none;
        min-width: 8; margin: 0;
    }
    """

    def __init__(self, content: str, mem_type: str = "thought") -> None:
        super().__init__()
        self._content = content
        self._type = mem_type

    def compose(self) -> ComposeResult:
        box = Vertical(id="mem-box")
        box.border_title = "◇ 儲存記憶"
        with box:
            yield Static("[#8b949e]選擇要記住的內容，可以編輯：[/]")
            with Horizontal(id="mem-type-row"):
                for t, label in [("thought", "想法"), ("insight", "洞見"), ("question", "問題"), ("reaction", "反應"), ("reference", "參考")]:
                    cls = "mtype-btn" + (" -selected" if t == self._type else "")
                    yield Button(f"[{label}]", id=f"mtype-{t}", classes=cls, name=t)
            yield TextArea(self._content, id="mem-editor")
            yield Static("[dim #484f58]主題標籤（選填）：[/]")
            yield Input(placeholder="例：字體設計、工具", id="mem-topic-input")
            with Horizontal(id="mem-save-row"):
                yield Button("儲存記憶", id="btn-mem-save")
                yield Button("[取消]", id="btn-mem-cancel")
                yield Button("← 回到對話", classes="back-btn")

    def on_mount(self) -> None:
        self.query_one("#mem-editor", TextArea).focus()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.has_class("back-btn"):
            self.dismiss(None)
            return
        bid = event.button.id or ""
        if bid.startswith("mtype-"):
            self._type = event.button.name or "thought"
            for btn in self.query(".mtype-btn"):
                btn.remove_class("-selected")
            event.button.add_class("-selected")
        elif bid == "btn-mem-save":
            content = self.query_one("#mem-editor", TextArea).text.strip()
            topic = self.query_one("#mem-topic-input", Input).value.strip()
            if content:
                self.dismiss({"type": self._type, "content": content, "topic": topic})
            else:
                self.notify("內容不能為空")
        elif bid == "btn-mem-cancel":
            self.dismiss(None)

    def action_cancel(self) -> None:
        self.dismiss(None)
