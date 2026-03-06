"""De-insight TUI — Elia 風格的對話介面"""

import json
import sys
import time
from pathlib import Path

# Allow importing from backend/
sys.path.insert(0, str(Path(__file__).parent / "backend"))

import httpx
from rich.markup import escape
from rich.text import Text
from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.css.query import NoMatches
from textual.reactive import reactive
from textual.timer import Timer
from textual.widgets import Input, Markdown, OptionList, Static, TextArea
from textual.widgets.option_list import Option

from codex_client import codex_stream, is_codex_available
from memory.store import add_memory, get_memories
from panels import (
    ImportModal, InsightConfirmModal, MemoryDetailModal, MemoryManageModal,
    MemoryItem, MemoryPanel, MemorySaveModal, ResearchPanel, SearchModal,
)
from settings import SettingsScreen, load_env

SPINNER_FRAMES = ["|", "/", "—", "\\"]

LANCEDB_DIR = Path(__file__).parent / "data" / "lancedb"


def _get_reindex_age() -> str:
    """取得距離上次 reindex 的時間描述。回傳空字串表示尚無索引。"""
    if not LANCEDB_DIR.exists():
        return ""
    # 找 LanceDB 目錄下最新的檔案修改時間
    latest = 0.0
    for f in LANCEDB_DIR.rglob("*"):
        if f.is_file():
            mt = f.stat().st_mtime
            if mt > latest:
                latest = mt
    if latest == 0.0:
        return ""
    elapsed = time.time() - latest
    if elapsed < 60:
        return "剛剛"
    elif elapsed < 3600:
        return f"{int(elapsed // 60)}分鐘前"
    elif elapsed < 86400:
        return f"{int(elapsed // 3600)}小時前"
    else:
        return f"{int(elapsed // 86400)}天前"


# ── Widgets ──────────────────────────────────────────────────────────


class ChatInput(TextArea):
    """Enter 送出，Shift+Enter 換行，Escape 清空。"""

    BINDINGS = [
        Binding("enter", "submit", "送出", show=False, priority=True),
        Binding("escape", "clear_input", "清空", show=False),
    ]

    class Submitted(TextArea.Changed):
        """使用者按下 Enter 送出。"""

    async def _on_key(self, event) -> None:
        if event.key == "enter":
            event.prevent_default()
            event.stop()
            self.post_message(self.Submitted(self))
            return
        # 不攔截其他按鍵（讓 Ctrl+C 等正常運作）
        await super()._on_key(event)

    def action_submit(self) -> None:
        self.post_message(self.Submitted(self))

    def action_clear_input(self) -> None:
        self.text = ""


class MenuBar(Static):
    """Mac 風格頂部功能列，單行可點擊。"""

    # Each item: (label, action, start_col, end_col) — populated on render
    _ITEMS = [
        ("New", "new_chat"),
        ("Import", "import_document"),
        ("Search", "search_knowledge"),
        ("Memory", "manage_memories"),
        ("Settings", "open_settings"),
    ]

    _mode: str = "emotional"
    _model: str = ""
    _memory_count: int = 0
    _has_kg: bool = False
    _rag_mode: str = "fast"
    _last_reindex: str = ""

    def __init__(self, **kwargs) -> None:
        super().__init__("", **kwargs)
        self._regions: list[tuple[int, int, str]] = []  # (start, end, action)

    def set_state(
        self,
        mode: str = "emotional",
        model: str = "",
        memory_count: int = 0,
        has_kg: bool = False,
        rag_mode: str = "fast",
        last_reindex: str = "",
    ) -> None:
        """Update mode/model/status and trigger re-render."""
        self._mode = mode
        self._model = model
        self._memory_count = memory_count
        self._has_kg = has_kg
        self._rag_mode = rag_mode
        self._last_reindex = last_reindex
        self._regions.clear()
        self.refresh()

    def render(self) -> Text:
        """Build menu bar text with click regions tracked by column position."""
        text = Text()
        col = 1  # after left padding (CSS padding: 0 1)
        self._regions.clear()

        for i, (label, action) in enumerate(self._ITEMS):
            display = f"[{label}]"
            start = col
            end = col + len(display)
            self._regions.append((start, end, action))
            text.append(display, style="#6e7681")
            if i < len(self._ITEMS) - 1:
                text.append(" ")
                col = end + 1
            else:
                col = end

        text.append(" ")
        col += 1
        text.append("│", style="#2a2a2a")
        col += 1
        text.append(" ")
        col += 1

        # Mode: 感性
        e_start = col
        text.append("感性", style="bold #fafafa" if self._mode == "emotional" else "#484f58")
        col += 4  # 2 CJK chars = 4 columns in terminal
        self._regions.append((e_start, col, "_mode_emotional"))

        text.append(" ")
        col += 1

        # Mode: 理性
        r_start = col
        text.append("理性", style="bold #fafafa" if self._mode == "rational" else "#484f58")
        col += 4
        self._regions.append((r_start, col, "_mode_rational"))

        # Separator + status info
        text.append("  ")
        col += 2
        text.append("│", style="#2a2a2a")
        col += 1
        text.append(" ", style="")
        col += 1

        # Memory count
        mem_style = "#8b949e" if self._memory_count > 0 else "#484f58"
        mem_label = f"記憶:{self._memory_count}"
        text.append(mem_label, style=mem_style)
        col += sum(2 if ord(c) > 0x7F else 1 for c in mem_label)

        text.append("  ")
        col += 2

        # Knowledge base status
        kg_label = "知識庫:✓" if self._has_kg else "知識庫:—"
        kg_style = "#8b949e" if self._has_kg else "#484f58"
        text.append(kg_label, style=kg_style)
        col += sum(2 if ord(c) > 0x7F else 1 for c in kg_label)

        text.append("  ")
        col += 2

        # RAG mode
        rag_label = "RAG:快速" if self._rag_mode == "fast" else "RAG:深度"
        rag_start = col
        rag_style = "#6e7681"
        text.append(rag_label, style=rag_style)
        col += sum(2 if ord(c) > 0x7F else 1 for c in rag_label)
        self._regions.append((rag_start, col, "toggle_rag_mode"))

        # Last reindex time
        if self._last_reindex:
            text.append("  ")
            col += 2
            idx_label = f"索引:{self._last_reindex}"
            text.append(idx_label, style="#484f58")
            col += sum(2 if ord(c) > 0x7F else 1 for c in idx_label)

        # Model name (dimmed, at the end)
        if self._model:
            text.append("  ")
            text.append(self._model, style="#484f58")

        return text

    def on_click(self, event) -> None:
        x = event.x
        for start, end, action in self._regions:
            if start <= x < end:
                if action == "_mode_emotional":
                    self.app.mode = "emotional"
                elif action == "_mode_rational":
                    self.app.mode = "rational"
                else:
                    self.app.action_from_menu(action)
                return


class WelcomeBlock(Vertical):
    """初始歡迎畫面。"""

    def compose(self) -> ComposeResult:
        yield Static(
            "[bold #fafafa]◈ De-insight[/]  [dim]批判性對話者[/]",
        )
        yield Static(
            "[dim #6e7681]────────────────────────────[/]",
        )
        yield Static(
            "[#8b949e]△ 我不是助手。我是挑戰者。\n"
            "△ 質疑你的視覺決策背後的權力結構。\n"
            "△ 基於 Foucault 規訓理論框架。[/]",
        )
        yield Static(
            "[dim #484f58]輸入 /help 查看所有指令[/]",
        )


class ThinkingIndicator(Static):
    """斜線旋轉思考指示器。"""

    def __init__(self, **kwargs) -> None:
        super().__init__("", **kwargs)
        self._frame = 0
        self._timer: Timer | None = None

    def on_mount(self) -> None:
        self._timer = self.set_interval(0.1, self._spin)

    def _spin(self) -> None:
        ch = SPINNER_FRAMES[self._frame % len(SPINNER_FRAMES)]
        self.update(Text(f" {ch} ", style="#fafafa"))
        self._frame += 1

    def stop(self) -> None:
        if self._timer:
            self._timer.stop()


class ActionLink(Static):
    """可點擊的操作連結。"""

    def __init__(self, label: str, action_name: str, **kwargs) -> None:
        super().__init__(Text(f"[{label}]", style="#484f58"), **kwargs)
        self._action_name = action_name

    def on_click(self) -> None:
        # 往上找到最近的 Chatbox 祖先
        node = self.parent
        while node is not None and not isinstance(node, Chatbox):
            node = node.parent
        if node is None:
            return
        if self._action_name == "save_insight":
            self.app.action_save_insight_from_chat(node)
        elif self._action_name == "save_memory":
            self.app.action_save_memory_from_chat(node)
        elif self._action_name == "copy":
            self.app.action_copy_chatbox(node)


class ChatboxActions(Horizontal):
    """訊息下方的操作列。"""

    def __init__(self, show_insight: bool = True, **kwargs) -> None:
        super().__init__(**kwargs)
        self._show_insight = show_insight

    def compose(self) -> ComposeResult:
        yield ActionLink("複製", "copy", classes="action-link")
        if self._show_insight:
            yield ActionLink("save insight", "save_insight", classes="action-link")
        yield ActionLink("記憶", "save_memory", classes="action-link")


class Chatbox(Vertical):
    """Elia 風格的訊息框：圓角邊框 + 標題。"""

    # Breathing animation: cycle border between dim and bright
    _BREATH_COLORS = [
        "#2a2a2a", "#3a3a3a", "#4a4a4a", "#5a5a5a",
        "#6a6a6a", "#7a7a7a", "#8a8a8a", "#9a9a9a",
        "#aaaaaa", "#bbbbbb", "#cccccc", "#dddddd",
        "#eeeeee", "#fafafa",
        "#eeeeee", "#dddddd", "#cccccc", "#bbbbbb",
        "#aaaaaa", "#9a9a9a", "#8a8a8a", "#7a7a7a",
        "#6a6a6a", "#5a5a5a", "#4a4a4a", "#3a3a3a",
    ]

    def __init__(self, role: str, content: str = "", system: bool = False,
                 streaming: bool = False, **kwargs) -> None:
        super().__init__(**kwargs)
        self.role = role
        self._system = system
        self._streaming = streaming
        self._content = content
        self.add_class(f"chatbox-{role}")
        self._breath_timer: Timer | None = None
        self._breath_frame = 0

    def on_mount(self) -> None:
        if self.role == "user":
            self.border_title = "◇ You"
        else:
            self.border_title = "◆ De-insight"

    def compose(self) -> ComposeResult:
        if self._streaming:
            yield Static("", classes="chatbox-body stream-body", id="stream-text")
        else:
            yield Markdown(self._content, classes="chatbox-body")
        if not self._system:
            if self.role == "assistant":
                yield ChatboxActions(show_insight=True, classes="chatbox-actions")
            elif self.role == "user":
                yield ChatboxActions(show_insight=False, classes="chatbox-actions")

    def stream_update(self, content: str) -> None:
        """Update with streaming content using lightweight Static."""
        self._content = content
        try:
            self.query_one("#stream-text", Static).update(content)
        except NoMatches:
            pass

    async def finalize_stream(self) -> None:
        """Switch from streaming Static to full Markdown rendering."""
        self._streaming = False
        try:
            stream_el = self.query_one("#stream-text", Static)
            md = Markdown(self._content, classes="chatbox-body")
            await self.mount(md, before=stream_el)
            await stream_el.remove()
        except NoMatches:
            pass

    def _breathe(self) -> None:
        color = self._BREATH_COLORS[self._breath_frame % len(self._BREATH_COLORS)]
        self.styles.border = ("round", color)
        self.styles.border_title_color = color
        self._breath_frame += 1

    def set_responding(self, responding: bool) -> None:
        if responding:
            self.border_title = "◆ De-insight"
            self.add_class("responding")
            self._breath_frame = 0
            if self.is_mounted:
                self._breath_timer = self.set_interval(0.08, self._breathe)
        else:
            if self._breath_timer:
                self._breath_timer.stop()
                self._breath_timer = None
            self.border_title = "◆ De-insight"
            self.remove_class("responding")
            self.styles.border = ("round", "#666666")
            self.styles.border_title_color = "#fafafa"


class StatusBar(Static):
    """底部狀態列。"""

    pass


# ── App ──────────────────────────────────────────────────────────────


class DeInsightApp(App):
    TITLE = "De-insight"
    CSS = """
    Screen {
        background: #0a0a0a;
        color: #fafafa;
    }

    /* ── menu bar ── */
    MenuBar {
        dock: top;
        height: auto;
        padding: 0 1;
        background: #111111;
        color: #6e7681;
        border-bottom: solid #2a2a2a;
    }

    MenuBar:hover {
        color: #fafafa;
    }

    /* ── chatbox actions ── */
    .chatbox-actions {
        height: 1;
        margin: 0;
        padding: 0;
    }

    ActionLink {
        width: auto;
        height: 1;
        padding: 0;
        margin: 0 1 0 0;
        color: #484f58;
    }

    ActionLink:hover {
        color: #8b949e;
    }

    /* ── chat scroll area ── */
    #chat-scroll {
        background: #0a0a0a;
        scrollbar-size: 1 1;
        scrollbar-color: #2a2a2a;
        scrollbar-color-hover: #484f58;
        scrollbar-color-active: #6e7681;
    }

    #messages {
        padding: 1 2;
        height: auto;
    }

    /* ── welcome ── */
    WelcomeBlock {
        padding: 1 2;
        margin: 0 1;
        height: auto;
        border: round #3a3a3a;
        border-title-color: #8b949e;
    }

    /* ── chatbox ── */
    Chatbox {
        height: auto;
        margin: 1 1 0 1;
        padding: 0 2;
        border: round #2a2a2a;
        border-title-color: #6e7681;
    }

    .chatbox-user {
        border: round #3a3a3a;
        border-title-color: #8b949e;
    }

    .chatbox-assistant {
        border: round #fafafa 40%;
        border-title-color: #fafafa;
    }

    .chatbox-assistant.responding {
        background: #fafafa 3%;
    }

    .chatbox-body {
        margin: 0;
        padding: 0;
        height: auto;
        color: #fafafa;
    }

    .chatbox-user .chatbox-body {
        color: #c9d1d9;
    }

    .stream-body {
        margin: 0;
        padding: 0;
        height: auto;
        color: #fafafa;
    }

    /* ── markdown overrides ── */
    Markdown {
        margin: 0;
        padding: 0;
        background: transparent;
    }

    MarkdownH1, MarkdownH2, MarkdownH3 {
        margin: 0;
        padding: 0;
        background: transparent;
        color: #fafafa;
    }

    MarkdownFence {
        margin: 1 0;
        padding: 1 2;
        background: #111111;
        color: #e6edf3;
    }

    MarkdownBlockQuote {
        margin: 0;
        padding: 0 0 0 2;
        border-left: tall #fafafa;
        background: transparent;
        color: #8b949e;
    }

    MarkdownBulletList, MarkdownOrderedList {
        margin: 0;
        padding: 0 0 0 2;
    }

    /* ── thinking indicator ── */
    ThinkingIndicator {
        height: 1;
        margin: 1 1 0 1;
        padding: 0 2;
        border: round #fafafa 40%;
        border-title-color: #fafafa;
    }

    /* ── input area ── */
    #input-box {
        dock: bottom;
        height: auto;
        max-height: 8;
        padding: 0 2 0 2;
        margin: 0;
        background: #0a0a0a;
    }

    #input-frame {
        height: auto;
        margin: 0 1;
        padding: 0 1;
        border: round #3a3a3a;
        border-title-color: #6e7681;
        background: #111111;
    }

    #input-frame:focus-within {
        border: round #fafafa 40%;
        border-title-color: #fafafa;
    }

    #chat-input {
        background: transparent;
        color: #fafafa;
        border: none;
        padding: 0;
        margin: 0;
        height: auto;
        min-height: 1;
        max-height: 8;
    }

    /* ── status bar ── */
    StatusBar {
        dock: bottom;
        height: 1;
        padding: 0 2;
        background: #111111;
        color: #6e7681;
    }

    /* ── layout ── */
    #main-horizontal {
        height: 1fr;
    }

    #chat-column {
        width: 1fr;
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
    }

    /* ── memory panel ── */
    #memory-panel {
        height: 40%;
        padding: 0 1;
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
    }

    /* ── slash hint popup ── */
    #slash-hints {
        display: none;
        dock: bottom;
        layer: overlay;
        height: auto;
        max-height: 12;
        width: 40;
        margin: 0 0 0 3;
        background: #1a1a1a;
        border: round #3a3a3a;
        color: #c9d1d9;
        scrollbar-size: 1 1;
        scrollbar-color: #2a2a2a;
    }
    #slash-hints > .option-list--option-highlighted {
        background: #2a2a2a;
        color: #fafafa;
    }
    #slash-hints.-visible {
        display: block;
    }

    /* ── back button (shared across modals) ── */
    .back-btn {
        background: transparent;
        color: #484f58;
        border: none;
        height: 1;
        min-width: 0;
        margin: 1 0 0 0;
        padding: 0 1;
    }
    .back-btn:hover {
        color: #fafafa;
    }

    """

    BINDINGS = [
        Binding("ctrl+s", "open_settings", "設定", show=False, priority=True),
        Binding("ctrl+e", "toggle_mode", "感性/理性", show=False, priority=True),
        Binding("ctrl+n", "new_chat", "新對話", show=False, priority=True),
        Binding("ctrl+k", "search_knowledge", "搜尋知識庫", show=False, priority=True),
        Binding("ctrl+f", "import_document", "匯入文件", show=False, priority=True),
        Binding("ctrl+m", "manage_memories", "記憶管理", show=False, priority=True),
        Binding("ctrl+c", "quit", "退出", show=False),
    ]

    mode: reactive[str] = reactive("emotional")
    is_loading: reactive[bool] = reactive(False)
    rag_mode: str = "fast"  # "fast" = naive+context_only, "deep" = hybrid+LLM合成

    def __init__(self) -> None:
        super().__init__()
        self.messages: list[dict] = []
        self.api_base = "http://localhost:8000"

    def compose(self) -> ComposeResult:
        yield MenuBar(id="menu-bar")
        with Horizontal(id="main-horizontal"):
            with Vertical(id="chat-column"):
                yield VerticalScroll(
                    Vertical(id="messages"),
                    id="chat-scroll",
                )
            with Vertical(id="right-panel"):
                rp = ResearchPanel(id="research-panel")
                rp.border_title = "◇ Knowledge"
                yield rp
                mp = MemoryPanel(id="memory-panel")
                mp.border_title = "◇ Memories"
                yield mp
        yield StatusBar(id="status-bar")
        yield OptionList(id="slash-hints")
        ta = ChatInput(id="chat-input")
        ta.show_line_numbers = False
        input_frame = Vertical(ta, id="input-frame")
        input_frame.border_title = "⌨ Message"
        yield Vertical(input_frame, id="input-box")

    async def on_mount(self) -> None:
        self._update_menu_bar()
        self._update_status()
        container = self.query_one("#messages", Vertical)
        welcome = WelcomeBlock()
        welcome.border_title = "◈ De-insight v0.2"
        await container.mount(welcome)
        self.query_one("#chat-input", ChatInput).focus()
        self._refresh_memory_panel()

    # ── status ──

    def watch_mode(self) -> None:
        if self.is_mounted:
            self._update_menu_bar()
            self._update_status()

    def watch_is_loading(self) -> None:
        if self.is_mounted:
            self._update_status()

    def _update_menu_bar(self) -> None:
        try:
            env = load_env()
            model = env.get("LLM_MODEL", "?")
            menu = self.query_one("#menu-bar", MenuBar)
            # Get memory count (sync-safe: read cached value)
            mem_count = getattr(self, "_cached_memory_count", 0)
            # Knowledge base status
            try:
                from rag.knowledge_graph import has_knowledge
                has_kg = has_knowledge()
            except Exception:
                has_kg = False
            menu.set_state(
                mode=self.mode,
                model=model,
                memory_count=mem_count,
                has_kg=has_kg,
                rag_mode=self.rag_mode,
                last_reindex=_get_reindex_age(),
            )
        except NoMatches:
            pass

    def _update_status(self) -> None:
        loading = "  [italic #fafafa]⟳ thinking…[/]" if self.is_loading else ""
        msg_count = len(self.messages)
        try:
            self.query_one("#status-bar", StatusBar).update(
                f"[#484f58]{msg_count} msgs[/]"
                f"{loading}"
            )
        except NoMatches:
            pass

    # ── menu bar click handler ──

    def action_from_menu(self, action: str) -> None:
        """Called by MenuItem on click."""
        method = getattr(self, f"action_{action}", None)
        if method:
            method()

    # ── actions ──

    def action_toggle_mode(self) -> None:
        self.mode = "rational" if self.mode == "emotional" else "emotional"

    def action_open_settings(self) -> None:
        def on_dismiss(result: str | None) -> None:
            if result:
                self._update_menu_bar()
                self._update_status()
                self._reload_backend_env()

        self.push_screen(SettingsScreen(), callback=on_dismiss)

    @work(thread=True)
    async def _reload_backend_env(self) -> None:
        """通知後端重新載入 .env。"""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.post(f"{self.api_base}/api/reload-env")
                if resp.status_code == 200:
                    data = resp.json()
                    model = data.get("model", "?")
                    self.notify(f"⚙ 設定已儲存 ∙ 模型: {model}")
                else:
                    self.notify("⚙ 設定已儲存（後端重載失敗，請手動重啟）")
        except Exception:
            self.notify("⚙ 設定已儲存到 .env（後端未運行）")

    # ── knowledge & memory actions ──

    def action_import_document(self) -> None:
        def on_dismiss(path: str | None) -> None:
            if path:
                self._do_import(path)

        self.push_screen(ImportModal(), callback=on_dismiss)

    @work(exclusive=True)
    async def _do_import(self, source: str) -> None:
        is_url = source.startswith("http://") or source.startswith("https://")
        label = "網頁" if is_url else "檔案"
        self.notify(f"匯入{label}中…")
        try:
            from rag.knowledge_graph import insert_pdf, insert_url, reset_rag
            reset_rag()
            if is_url:
                await insert_url(source)
            else:
                await insert_pdf(source)
            self.notify("匯入完成")
            await self._update_research_panel("匯入完成，知識庫已更新")
        except Exception as e:
            self.notify(f"匯入失敗: {e}")

    def action_search_knowledge(self) -> None:
        def on_dismiss(query: str | None) -> None:
            if query:
                self._do_search(query)

        self.push_screen(SearchModal(), callback=on_dismiss)

    @work(exclusive=True)
    async def _do_search(self, query: str) -> None:
        try:
            from rag.knowledge_graph import query_knowledge, has_knowledge
            if not has_knowledge():
                self.notify("知識庫為空，請先匯入文件 (ctrl+f)")
                return
            result = await query_knowledge(query)
            await self._update_research_panel(result)
        except Exception as e:
            self.notify(f"搜尋失敗: {e}")

    def action_manage_memories(self) -> None:
        def on_dismiss(result: str | None) -> None:
            self._refresh_memory_panel()
            if result and result.startswith("discuss:"):
                content = result.removeprefix("discuss:")
                self._start_discussion_from_memory(content)

        self.push_screen(MemoryManageModal(), callback=on_dismiss)

    def _start_discussion_from_memory(self, memory_content: str) -> None:
        """用一條記憶開啟新的討論。"""
        prompt = f"我之前有一個想法：「{memory_content}」\n\n我想進一步討論這個概念。"
        inp = self.query_one("#chat-input", ChatInput)
        inp.text = prompt
        self._submit_chat()

    @staticmethod
    def _clean_rag_display(raw: str) -> str:
        """清理 LightRAG 原始輸出，提取可讀的中文內容。"""
        import re
        # 移除 LightRAG 格式標頭
        text = re.sub(r"Document Chunks.*?Reference Document List[`'\s)]*:", "", raw, flags=re.DOTALL)
        text = re.sub(r"```json\s*", "", text)
        text = re.sub(r"```\s*", "", text)
        # 提取 JSON content 欄位
        contents = re.findall(r'"content"\s*:\s*"((?:[^"\\]|\\.)*)"', text)
        if contents:
            parts = []
            for c in contents:
                c = c.replace("\\n", "\n").replace("\\t", " ").strip()
                if c and len(c) > 20:
                    parts.append(c)
            if parts:
                # 優先排序含中文的片段
                def cn_ratio(s):
                    cn = len(re.findall(r'[\u4e00-\u9fff]', s))
                    return cn / max(len(s), 1)
                parts.sort(key=cn_ratio, reverse=True)
                # 只取中文內容較多的段落，用分隔線連接
                result = "\n\n---\n\n".join(parts[:3])
                # 移除 [ＯＫ p.X] 等來源標記，簡化顯示
                result = re.sub(r"\[.*?p\.\d+\]\s*", "", result)
                return result
        # Fallback
        text = re.sub(r'\{["\s]*reference_id["\s]*:.*?\}', "", text)
        text = re.sub(r"Reference Document List.*", "", text, flags=re.DOTALL)
        return text.strip()

    async def _update_research_panel(self, content: str) -> None:
        try:
            panel = self.query_one("#research-content", Static)
            if content:
                display = self._clean_rag_display(content)
                if not display:
                    display = content
                display = display[:800] + ("…" if len(display) > 800 else "")
                panel.update(f"[#c9d1d9]{escape(display)}[/]")
            else:
                panel.update("[dim #484f58]無結果[/]")
        except NoMatches:
            pass

    @work(exclusive=True)
    async def _refresh_memory_panel(self) -> None:
        try:
            panel = self.query_one("#memory-panel", MemoryPanel)
        except NoMatches:
            return
        # 移除舊的 memory-content 和 MemoryItem
        await panel.remove_children()
        mems = await get_memories(limit=10)
        # Update cached memory count for status bar
        try:
            from memory.store import get_memory_stats
            stats = await get_memory_stats()
            self._cached_memory_count = stats.get("total", 0)
        except Exception:
            self._cached_memory_count = len(mems) if mems else 0
        self._update_menu_bar()
        if not mems:
            await panel.mount(
                Static("[dim #484f58]對話後自動記錄洞見[/]", id="memory-content")
            )
            return
        for m in mems:
            await panel.mount(MemoryItem(m))

    def action_new_chat(self) -> None:
        self.messages.clear()
        container = self.query_one("#messages", Vertical)
        container.remove_children()
        self.call_after_refresh(self._mount_welcome)
        self.query_one("#chat-input", ChatInput).focus()

    async def _mount_welcome(self) -> None:
        container = self.query_one("#messages", Vertical)
        welcome = WelcomeBlock()
        welcome.border_title = "◈ De-insight v0.2"
        await container.mount(welcome)

    # ── slash commands ──

    SLASH_COMMANDS = {
        "/help": "show_help",
        "/new": "new_chat",
        "/import": "import_document",
        "/search": "search_knowledge",
        "/memory": "manage_memories",
        "/settings": "open_settings",
        "/mode": "toggle_mode",
        "/save": "save_insight_manual",
        "/reindex": "reindex_memories",
        "/ragmode": "toggle_rag_mode",
    }

    SLASH_HINTS: list[tuple[str, str]] = [
        ("/help", "顯示所有指令說明"),
        ("/new", "開啟新對話"),
        ("/import", "匯入 PDF 或網頁到知識庫"),
        ("/search", "搜尋知識庫"),
        ("/memory", "管理記憶 / 知識"),
        ("/settings", "開啟設定"),
        ("/mode", "切換感性 / 理性模式"),
        ("/save", "儲存當前對話的洞見"),
        ("/reindex", "重建記憶向量索引"),
        ("/ragmode", "切換知識檢索：快速 / 深度"),
    ]

    def _handle_slash_command(self, text: str) -> bool:
        """Handle slash commands. Returns True if handled."""
        cmd = text.split()[0].lower()
        if cmd in self.SLASH_COMMANDS:
            action = self.SLASH_COMMANDS[cmd]
            method = getattr(self, f"action_{action}", None)
            if method:
                method()
            return True
        if cmd.startswith("/"):
            self.notify(f"未知指令: {cmd}  輸入 /help 查看")
            return True
        return False

    def action_show_help(self) -> None:
        help_text = (
            "**可用指令**\n\n"
            "| 指令 | 功能 |\n"
            "|------|------|\n"
            "| `/new` | 新對話 |\n"
            "| `/import` | 匯入 PDF 或網頁到知識庫 |\n"
            "| `/search` | 搜尋知識庫 |\n"
            "| `/memory` | 管理記憶 |\n"
            "| `/save` | 儲存當前對話的洞見 |\n"
            "| `/reindex` | 重建記憶向量索引 |\n"
            "| `/settings` | 開啟設定 |\n"
            "| `/mode` | 切換感性/理性 |\n"
            "| `/ragmode` | 切換知識檢索：快速/深度 |\n"
            "| `/help` | 顯示此說明 |\n"
        )
        self.call_after_refresh(lambda: self._show_system_message(help_text))

    def action_reindex_memories(self) -> None:
        self._do_reindex()

    @work(exclusive=True)
    async def _do_reindex(self) -> None:
        self.notify("正在重建記憶向量索引...")
        try:
            from memory.vectorstore import index_all_memories
            count = await index_all_memories()
            self.notify(f"索引完成：{count} 條記憶已向量化")
            self._update_menu_bar()
        except Exception as e:
            self.notify(f"索引失敗: {e}")

    def action_toggle_rag_mode(self) -> None:
        if self.rag_mode == "fast":
            self.rag_mode = "deep"
            self.notify("知識檢索：深度模式（圖譜推理，較慢）")
        else:
            self.rag_mode = "fast"
            self.notify("知識檢索：快速模式（向量搜尋，<1秒）")
        self._update_menu_bar()

    async def _show_system_message(self, content: str) -> None:
        container = self.query_one("#messages", Vertical)
        box = Chatbox("assistant", content, system=True)
        await container.mount(box)
        box.styles.border = ("round", "#3a3a3a")
        self._scroll_to_bottom()

    # ── save insight ──

    def action_save_insight_manual(self) -> None:
        """From /save command — use last exchange."""
        if len(self.messages) < 2:
            self.notify("至少需要一輪對話才能儲存洞見")
            return
        # Get last user + assistant exchange
        user_msg = ""
        ai_msg = ""
        for m in reversed(self.messages):
            if m["role"] == "assistant" and not ai_msg:
                ai_msg = m["content"]
            elif m["role"] == "user" and not user_msg:
                user_msg = m["content"]
            if user_msg and ai_msg:
                break
        if user_msg:
            self._prepare_insight(user_msg, ai_msg)

    def action_save_insight_from_chat(self, chatbox: "Chatbox") -> None:
        """From [save insight] button on a chatbox."""
        ai_msg = chatbox._content
        user_msg = ""
        for m in reversed(self.messages):
            if m["role"] == "user":
                user_msg = m["content"]
                break
        if user_msg or ai_msg:
            self._prepare_insight(user_msg, ai_msg)

    def action_copy_chatbox(self, chatbox: "Chatbox") -> None:
        """複製訊息內容到系統剪貼板。"""
        import subprocess
        content = chatbox._content
        if not content:
            return
        try:
            subprocess.run(["pbcopy"], input=content.encode(), check=True)
            self.notify("已複製到剪貼板")
        except Exception:
            # Fallback: pyperclip or just notify
            self.notify("複製失敗，請手動選取")

    def action_save_memory_from_chat(self, chatbox: "Chatbox") -> None:
        """From [記憶] button on a chatbox."""
        content = chatbox._content
        mem_type = "thought" if chatbox.role == "user" else "reference"

        def on_confirm(result: dict | None) -> None:
            if result:
                self._do_save_memory(result)

        self.push_screen(MemorySaveModal(content, mem_type), callback=on_confirm)

    @work(exclusive=True)
    async def _do_save_memory(self, data: dict) -> None:
        """儲存記憶到 SQLite + LanceDB。"""
        try:
            from memory.store import add_memory
            await add_memory(
                mem_type=data["type"],
                content=data["content"],
                topic=data.get("topic", ""),
                source="manual",
            )
            self.notify(f"已儲存記憶：{data['content'][:30]}…")
            self._refresh_memory_panel()
        except Exception as e:
            self.notify(f"儲存失敗: {e}")

    @work(exclusive=True)
    async def _prepare_insight(self, user_msg: str, ai_msg: str) -> None:
        """Use LLM to draft an insight from the exchange, then confirm."""
        self.notify("整理洞見中…")
        try:
            prompt = (
                "以下是一段對話。請從使用者的發言中提取一個核心洞見，"
                "用一到兩句精簡的繁體中文概括。只回傳洞見本身，不要加任何解釋。\n\n"
                f"使用者：{user_msg}\n\nAI：{ai_msg}"
            )
            draft = await self._quick_llm_call(prompt, max_tokens=200)
            draft = draft.strip()

            if not draft:
                self.notify("無法提取洞見")
                return

            source = user_msg[:100]

            def on_confirm(result: dict | None) -> None:
                if result:
                    self._save_confirmed_insight(result, source)

            self.app.push_screen(
                InsightConfirmModal(draft, "insight"),
                callback=on_confirm,
            )
        except Exception as e:
            self.notify(f"整理失敗: {e}")

    @work()
    async def _save_confirmed_insight(self, data: dict, source: str) -> None:
        await add_memory(
            type=data["type"],
            content=data["content"],
            source=source,
        )
        self.notify(f"已儲存 [{data['type']}]")
        self._refresh_memory_panel()

    @work(exclusive=False)
    async def _auto_extract_memories(self, user_text: str) -> None:
        """對話結束後，背景自動從使用者發言抽取記憶。"""
        try:
            from memory.thought_tracker import extract_memories
            items = await extract_memories(user_text, self._quick_llm_call)
            if not items:
                return
            from memory.store import add_memory as _add_memory
            for item in items:
                await _add_memory(
                    type=item["type"],
                    content=item["content"],
                    source=user_text[:80],
                    topic=item.get("topic", ""),
                )
            self._refresh_memory_panel()
            # 若有新洞見，檢查思維演變
            insights = [i for i in items if i["type"] == "insight"]
            if insights:
                from memory.thought_tracker import check_for_evolution
                evolution = await check_for_evolution(
                    insights[0]["content"], self._quick_llm_call
                )
                if evolution and evolution.get("type") in ("evolution", "contradiction"):
                    etype = "演變" if evolution["type"] == "evolution" else "矛盾"
                    self.notify(f"偵測到思維{etype}", timeout=6)
        except Exception:
            pass  # 記憶抽取失敗不影響主流程

    # ── drag-to-import ──

    def _clean_dropped_path(self, text: str) -> str | None:
        """清理拖放進來的文字，回傳有效檔案路徑或 None。"""
        from urllib.parse import unquote, urlparse
        t = text.strip().strip("'\"")
        if not t:
            return None
        # file:// URI → 本地路徑
        if t.startswith("file://"):
            t = unquote(urlparse(t).path)
        # URL 編碼的空格
        if "%20" in t:
            t = unquote(t)
        # 終端拖放的反斜線轉義空格
        t = t.replace("\\ ", " ")
        # 檢查檔案是否存在
        if Path(t).exists() and Path(t).is_file():
            return t
        return None

    def on_paste(self, event) -> None:
        """攔截貼上事件，偵測檔案路徑或 URL 自動匯入。"""
        text = event.text.strip()
        if not text:
            return
        # 偵測 URL
        clean = text.strip("'\"")
        is_url = clean.startswith("http://") or clean.startswith("https://")
        if is_url:
            event.prevent_default()
            self.notify("偵測到網址，開始匯入…")
            self._do_import(clean)
            return
        # 偵測檔案路徑
        path = self._clean_dropped_path(text)
        if path:
            event.prevent_default()
            if path.lower().endswith(".pdf"):
                self.notify(f"偵測到 PDF，開始匯入…")
                self._do_import(path)
            else:
                # 非 PDF 檔案：顯示路徑在輸入框
                inp = self.query_one("#chat-input", ChatInput)
                inp.text = path
                inp.focus()
                self.notify(f"檔案路徑已填入輸入框")

    # ── slash hints ──

    def _update_slash_hints(self) -> None:
        """根據輸入框內容顯示/隱藏指令提示。"""
        try:
            ta = self.query_one("#chat-input", ChatInput)
            hints = self.query_one("#slash-hints", OptionList)
        except NoMatches:
            return
        text = ta.text.strip()
        if text.startswith("/") and "\n" not in text:
            query = text.lower()
            matches = [
                (cmd, desc) for cmd, desc in self.SLASH_HINTS
                if cmd.startswith(query)
            ]
            if matches:
                hints.clear_options()
                for cmd, desc in matches:
                    hints.add_option(Option(f"{cmd}  {desc}", id=cmd))
                hints.add_class("-visible")
                return
        hints.remove_class("-visible")

    def on_text_area_changed(self, event: TextArea.Changed) -> None:
        if event.text_area.id == "chat-input":
            self._update_slash_hints()

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        if event.option_list.id == "slash-hints":
            cmd = event.option.id
            ta = self.query_one("#chat-input", ChatInput)
            ta.text = cmd
            event.option_list.remove_class("-visible")
            ta.focus()
            self._submit_chat()

    # ── chat ──

    def on_chat_input_submitted(self, event: ChatInput.Submitted) -> None:
        """ChatInput 按 Enter 送出。"""
        self._submit_chat()

    @work(exclusive=False, thread=False)
    async def _submit_chat(self) -> None:
        ta = self.query_one("#chat-input", ChatInput)
        text = ta.text.strip()
        if not text or self.is_loading:
            return

        ta.text = ""

        # 隱藏 slash hints
        try:
            self.query_one("#slash-hints", OptionList).remove_class("-visible")
        except NoMatches:
            pass

        # 先偵測檔案路徑或 URL（優先於 slash command，因為路徑也以 / 開頭）
        path = self._clean_dropped_path(text)
        if path and path.lower().endswith(".pdf"):
            self.notify("偵測到 PDF，開始匯入…")
            self._do_import(path)
            return
        clean_text = text.strip("'\"")
        if (clean_text.startswith("http://") or clean_text.startswith("https://")) and " " not in clean_text:
            self.notify("偵測到網址，開始匯入…")
            self._do_import(clean_text)
            return

        # Handle slash commands
        if text.startswith("/"):
            self._handle_slash_command(text)
            return

        self.messages.append({"role": "user", "content": text})

        container = self.query_one("#messages", Vertical)

        for w in container.query("WelcomeBlock"):
            await w.remove()

        await container.mount(Chatbox("user", text))
        self._scroll_to_bottom()
        self._stream_response()

    def _is_codex_cli_mode(self) -> bool:
        """檢查目前是否使用 Codex CLI 模式。"""
        env = load_env()
        return env.get("LLM_MODEL", "").startswith("codex-cli/")

    def _is_direct_api_mode(self) -> bool:
        """檢查是否需要直接呼叫 API（不經 LiteLLM）。"""
        env = load_env()
        model = env.get("LLM_MODEL", "")
        # openai/ prefix with custom base = OpenAI-compatible API (MiniMax, etc.)
        return model.startswith("openai/") and env.get("OPENAI_API_BASE", "")

    @staticmethod
    def _strip_reasoning(text: str) -> str:
        """移除 reasoning model 輸出的英文思考過程，只保留最終回應。"""
        import re
        # 偵測模式：一段英文推理 + 緊接中文回應
        # 找到第一個中文字元的位置（包含中文標點）
        m = re.search(r'[\u4e00-\u9fff\u3000-\u303f\uff00-\uffef]', text)
        if not m:
            # 整篇沒有中文 — 檢查是否以 **bold** 思考標記開頭（reasoning 特徵）
            if re.match(r'\s*\*\*[A-Z]', text):
                return text  # 無法修復，原樣回傳
            return text
        cn_start = m.start()
        if cn_start == 0:
            return text  # 開頭就是中文，沒有推理前綴
        # 檢查中文前的部分是否像英文推理（至少 50 字元的英文段落）
        prefix = text[:cn_start]
        if len(prefix) > 50 and prefix.count('.') >= 2:
            return text[cn_start:]
        return text

    @work(exclusive=True)
    async def _stream_response(self) -> None:
        self.is_loading = True
        container = self.query_one("#messages", Vertical)

        # 立即建立 AI 回覆框（streaming 模式），呼吸燈開始
        bubble = Chatbox("assistant", streaming=True)
        await container.mount(bubble)
        bubble.set_responding(True)
        self._scroll_to_bottom()

        full_content = ""

        try:
            if self._is_codex_cli_mode():
                # ── Codex CLI 路徑 ──
                if not is_codex_available():
                    raise RuntimeError("codex CLI 未安裝。執行: npm i -g @openai/codex")

                from prompts.foucault import get_system_prompt
                sys_prompt = get_system_prompt(self.mode)
                user_msg = self.messages[-1]["content"]

                # 注入記憶上下文（向量搜尋）
                memory_context = ""
                try:
                    from memory.vectorstore import search_similar, has_index
                    if has_index():
                        mem_results = await search_similar(user_msg, limit=3)
                        if mem_results:
                            mem_lines = "\n".join(
                                f"- [{m['type']}] {m['content']}" + (f" (#{m['topic']})" if m.get('topic') else "")
                                for m in mem_results
                            )
                            memory_context = f"\n\n[使用者過去的想法]\n{mem_lines}"
                except Exception:
                    pass

                # 注入知識庫上下文
                rag_context = ""
                try:
                    from rag.knowledge_graph import query_knowledge, has_knowledge
                    if has_knowledge():
                        is_deep = self.rag_mode == "deep"
                        result = await query_knowledge(
                            user_msg,
                            mode="hybrid" if is_deep else "naive",
                            context_only=not is_deep,
                        )
                        if result and len(result.strip()) > 10:
                            rag_context = f"\n\n[知識庫參考]\n{result[:2000]}"
                            await self._update_research_panel(result)
                except Exception:
                    pass

                if memory_context or rag_context:
                    sys_prompt += memory_context + rag_context
                    sys_prompt += "\n\n（以上為參考資料，請務必用繁體中文回覆使用者。）"

                # Include conversation context
                context = ""
                for m in self.messages[:-1]:
                    role = "You" if m["role"] == "user" else "De-insight"
                    context += f"{role}: {m['content']}\n\n"

                full_prompt = f"{context}You: {user_msg}" if context else user_msg

                env = load_env()
                codex_model = env.get("LLM_MODEL", "").removeprefix("codex-cli/")

                async for chunk in codex_stream(full_prompt, sys_prompt, model=codex_model):
                    full_content += chunk
                    bubble.stream_update(full_content)
                    self._scroll_to_bottom()
            elif self._is_direct_api_mode():
                # ── 直接 API 路徑（MiniMax 等 OpenAI-compatible）──
                import re as _re
                from prompts.foucault import get_system_prompt as _get_sp
                env = load_env()
                api_base = env.get("OPENAI_API_BASE", "")
                api_key = env.get("OPENAI_API_KEY", "")
                model = env.get("LLM_MODEL", "").removeprefix("openai/")

                sys_prompt = _get_sp(self.mode)
                send_messages = [{"role": "system", "content": sys_prompt}]

                # 注入 RAG + 記憶上下文
                user_msg = self.messages[-1]["content"]
                ctx_parts = []
                try:
                    from memory.vectorstore import search_similar, has_index
                    if has_index():
                        mem_results = await search_similar(user_msg, limit=3)
                        if mem_results:
                            mem_lines = "\n".join(
                                f"- [{m['type']}] {m['content']}" + (f" (#{m['topic']})" if m.get('topic') else "")
                                for m in mem_results
                            )
                            ctx_parts.append(f"[使用者過去的想法]\n{mem_lines}")
                except Exception:
                    pass
                try:
                    from rag.knowledge_graph import query_knowledge, has_knowledge
                    if has_knowledge():
                        is_deep = self.rag_mode == "deep"
                        result = await query_knowledge(
                            user_msg,
                            mode="hybrid" if is_deep else "naive",
                            context_only=not is_deep,
                        )
                        if result and len(result.strip()) > 10:
                            ctx_parts.append(f"[知識庫參考]\n{result[:2000]}")
                            await self._update_research_panel(result)
                except Exception:
                    pass
                if ctx_parts:
                    send_messages[0]["content"] += "\n\n" + "\n\n".join(ctx_parts)
                    send_messages[0]["content"] += "\n\n（以上為參考資料，請務必用繁體中文回覆使用者。）"

                for m in self.messages:
                    send_messages.append({"role": m["role"], "content": m["content"]})

                async with httpx.AsyncClient(timeout=120.0) as client:
                    async with client.stream(
                        "POST",
                        f"{api_base}/chat/completions",
                        headers={
                            "Authorization": f"Bearer {api_key}",
                            "Content-Type": "application/json",
                            "HTTP-Referer": "https://github.com/De-insight",
                            "X-Title": "De-insight",
                        },
                        json={"model": model, "messages": send_messages, "stream": True},
                    ) as response:
                        async for line in response.aiter_lines():
                            if not line.startswith("data: "):
                                continue
                            data_str = line[6:]
                            if data_str == "[DONE]":
                                break
                            try:
                                event = json.loads(data_str)
                                choices = event.get("choices", [])
                                if not choices:
                                    continue
                                delta = choices[0].get("delta", {}).get("content", "")
                                if delta:
                                    full_content += delta
                                    # Strip <think> tags in real-time
                                    display = _re.sub(r"<think>[\s\S]*?</think>\s*", "", full_content)
                                    if "<think>" in full_content and "</think>" not in full_content:
                                        display = ""  # Still thinking, don't show yet
                                    bubble.stream_update(display)
                                    self._scroll_to_bottom()
                            except (json.JSONDecodeError, ValueError):
                                pass
                # Final cleanup of think tags
                full_content = _re.sub(r"<think>[\s\S]*?</think>\s*", "", full_content)
            else:
                # ── FastAPI 後端路徑 ──
                send_messages = await self._inject_rag_context(self.messages)
                async with httpx.AsyncClient(timeout=120.0) as client:
                    async with client.stream(
                        "POST",
                        f"{self.api_base}/api/chat",
                        json={"messages": send_messages, "mode": self.mode},
                    ) as response:
                        async for line in response.aiter_lines():
                            if line.startswith("0:"):
                                try:
                                    chunk = json.loads(line[2:])
                                    full_content += chunk
                                    bubble.stream_update(full_content)
                                    self._scroll_to_bottom()
                                except (json.JSONDecodeError, ValueError):
                                    pass
                            elif line.startswith("3:"):
                                try:
                                    err = json.loads(line[2:])
                                    raise RuntimeError(err.get("error", "未知錯誤"))
                                except json.JSONDecodeError:
                                    raise RuntimeError(line[2:])
                            elif line.startswith("d:"):
                                break

            # 過濾 reasoning model 的思考過程（英文 chain-of-thought）
            full_content = self._strip_reasoning(full_content)
            bubble.set_responding(False)
            bubble.stream_update(full_content)
            await bubble.finalize_stream()
            self.messages.append({"role": "assistant", "content": full_content})

            # 背景：自動抽取記憶（不阻塞 UI）
            user_content = self.messages[-2]["content"] if len(self.messages) >= 2 else ""
            if user_content and len(user_content.strip()) >= 15:
                self._auto_extract_memories(user_content)

        except httpx.ConnectError:
            bubble.set_responding(False)
            bubble.stream_update(
                "**連線錯誤** — 後端未啟動\n\n"
                "cd backend && .venv/bin/python3 -m uvicorn main:app --reload"
            )
            await bubble.finalize_stream()
        except Exception as e:
            bubble.set_responding(False)
            bubble.stream_update(f"**錯誤** — {escape(str(e))}")
            await bubble.finalize_stream()
        finally:
            self.is_loading = False
            self._update_status()
            self.query_one("#chat-input", ChatInput).focus()

    def _scroll_to_bottom(self) -> None:
        self.query_one("#chat-scroll", VerticalScroll).scroll_end(animate=False)

    async def _quick_llm_call(self, prompt: str, max_tokens: int = 500) -> str:
        """共用的快速 LLM 呼叫，用於記憶抽取、洞見整理等。"""
        env = load_env()
        model = env.get("LLM_MODEL", "ollama/llama3.2")

        # codex-cli 模式：透過 Codex CLI 執行（用 OAuth，不需 API key）
        if model.startswith("codex-cli/"):
            codex_model = model.removeprefix("codex-cli/")
            result = ""
            async for chunk in codex_stream(prompt, model=codex_model):
                result += chunk
            return result

        # codex API 模式：用底層模型名稱
        if model.startswith("codex/"):
            model = "openai/" + model.removeprefix("codex/")

        import litellm
        resp = await litellm.acompletion(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content or ""


    async def _inject_rag_context(self, messages: list[dict]) -> list[dict]:
        """Inject knowledge base + memory vector search context."""
        user_msg = messages[-1]["content"] if messages else ""
        if not user_msg:
            return messages

        augmented = list(messages)
        insert_idx = 1 if (augmented and augmented[0]["role"] == "system") else 0

        # 1. 記憶向量搜尋
        try:
            from memory.vectorstore import search_similar, has_index
            if has_index():
                mem_results = await search_similar(user_msg, limit=3)
                if mem_results:
                    mem_lines = "\n".join(
                        f"- [{m['type']}] {m['content']}" + (f" (#{m['topic']})" if m.get('topic') else "")
                        for m in mem_results
                    )
                    augmented.insert(insert_idx, {
                        "role": "system",
                        "content": f"使用者過去的想法（語意相關）：\n{mem_lines}",
                    })
                    insert_idx += 1
        except Exception:
            pass

        # 2. 知識庫 RAG
        try:
            from rag.knowledge_graph import query_knowledge, has_knowledge
            if has_knowledge():
                is_deep = self.rag_mode == "deep"
                result = await query_knowledge(
                    user_msg,
                    mode="hybrid" if is_deep else "naive",
                    context_only=not is_deep,
                )
                if result and len(result.strip()) > 10:
                    await self._update_research_panel(result)
                    augmented.insert(insert_idx, {
                        "role": "system",
                        "content": f"知識庫相關資訊：\n{result[:2000]}\n\n（以上為參考資料，請務必用繁體中文回覆使用者。）",
                    })
        except Exception:
            pass

        return augmented


if __name__ == "__main__":
    DeInsightApp().run()
