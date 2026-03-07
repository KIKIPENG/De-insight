"""De-insight TUI — App 主體"""

import sys
from pathlib import Path

# Allow importing from backend/
sys.path.insert(0, str(Path(__file__).parent / "backend"))

from widgets import (
    AppState, ChatInput, MenuBar, WelcomeBlock, StatusBar,
)
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.reactive import reactive
from textual.widgets import OptionList

from panels import MemoryPanel, ResearchPanel
from conversation.store import ConversationStore
from projects.manager import ProjectManager

from mixins.chat import ChatMixin
from mixins.memory import MemoryMixin
from mixins.rag import RAGMixin
from mixins.project import ProjectMixin
from mixins.ui import UIMixin


class DeInsightApp(ChatMixin, MemoryMixin, RAGMixin, ProjectMixin, UIMixin, App):
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
        background: #0d0d0d;
        color: #6e7681;
        border-bottom: solid #1e1e1e;
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
        color: #2a2a2a;
    }

    ActionLink:hover {
        color: #d4a27a;
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
        border: round #1e1e1e;
        border-title-color: #d4a27a;
    }

    /* ── chatbox ── */
    Chatbox {
        height: auto;
        margin: 1 1 0 1;
        padding: 0 2;
        border: round #1e1e1e;
        border-title-color: #484f58;
    }

    .chatbox-user {
        border: round #2a2a2a;
        border-title-color: #6e7681;
    }

    .chatbox-assistant {
        border: round #2a2a2a;
        border-title-color: #d4a27a;
    }

    .chatbox-assistant.responding {
        background: #d4a27a 4%;
        border-title-color: #d4a27a;
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
        padding: 0 1;
        background: transparent;
    }

    Markdown MarkdownH1 {
        color: #f0f6fc;
        text-style: bold;
        border-bottom: solid #30363d;
        margin-bottom: 1;
        padding: 0;
        background: transparent;
    }

    Markdown MarkdownH2 {
        color: #c9d1d9;
        text-style: bold;
        margin-top: 1;
        padding: 0;
        background: transparent;
    }

    Markdown MarkdownH3 {
        color: #8b949e;
        text-style: bold italic;
        padding: 0;
        background: transparent;
    }

    MarkdownFence {
        margin: 1 0;
        padding: 0 1;
        background: #161b22;
        border-left: thick #30363d;
        color: #e6edf3;
    }

    MarkdownBlockQuote {
        margin: 0;
        padding: 0 1;
        border-left: thick #30363d;
        background: transparent;
        color: #8b949e;
    }

    MarkdownBulletList, MarkdownOrderedList {
        margin: 0;
        padding: 0 0 0 2;
    }

    Markdown .inline-code {
        color: #7dd3fc;
        background: #1e2d3d;
    }

    Markdown MarkdownHorizontalRule {
        color: #30363d;
    }

    /* ── knowledge concept link ── */
    ActionLink.source-link {
        color: #484f58;
    }
    ActionLink.source-link:hover {
        color: #d4a27a;
    }

    /* ── thinking indicator ── */
    ThinkingIndicator {
        height: 1;
        margin: 1 1 0 1;
        padding: 0 2;
        border: round #d4a27a 50%;
        border-title-color: #d4a27a;
    }

    /* ── input area ── */
    #input-box {
        dock: bottom;
        height: auto;
        max-height: 16;
        padding: 0 1;
        margin: 0;
        background: #0a0a0a;
    }

    #input-frame {
        height: auto;
        margin: 0 1;
        padding: 0 1;
        border: round #2a2a2a;
        border-title-color: #484f58;
        background: #0d0d0d;
    }

    #input-frame:focus-within {
        border: round #d4a27a 70%;
        border-title-color: #d4a27a;
    }

    #chat-input {
        background: transparent;
        color: #fafafa;
        border: none;
        padding: 0;
        margin: 0;
        height: auto;
        min-height: 3;
        max-height: 14;
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
        border-top: solid #2a2a2a;
        border-title-color: #6e7681;
        padding: 0 1;
    }

    .kb-actions-row {
        height: 1;
        margin: 1 0 1 0;
    }
    .kb-action {
        width: auto;
        min-width: 0;
        padding: 0 1;
        color: #6e7681;
    }
    .kb-action:hover {
        color: #fafafa;
    }
    #kb-status-line {
        height: 1;
        color: #484f58;
        margin: 0;
    }
    .kb-divider {
        height: 1;
        color: #2a2a2a;
        margin: 0 0 1 0;
    }
    .doc-item {
        height: auto;
        padding: 0 1;
        color: #8b949e;
    }
    .doc-item:hover {
        color: #fafafa;
        background: #111111;
    }

    /* ── memory panel ── */
    #memory-panel {
        height: 40%;
        border-top: solid #2a2a2a;
        border-title-color: #6e7681;
        padding: 0 1;
    }

    .memory-item {
        height: auto;
        margin: 0 0 1 0;
        padding: 0 1;
        color: #6e7681;
    }

    .memory-item.-new {
        color: #d4a27a;
        background: #3d2a1a;
    }

    .memory-item:hover {
        color: #fafafa;
        background: #1a1a1a;
    }

    /* ── history entries (WelcomeBlock) ── */
    .history-entry {
        height: 1;
        padding: 0 1;
        color: #8b949e;
    }
    .history-entry:hover {
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
        Binding("ctrl+p", "open_project_modal", "專案管理", show=False, priority=True),
        Binding("ctrl+d", "manage_documents", "文獻管理", show=False, priority=True),
        Binding("ctrl+b", "bulk_import", "批量匯入", show=False, priority=True),
        Binding("ctrl+g", "view_relations", "記憶關聯", show=False, priority=True),
        Binding("ctrl+l", "open_gallery", "圖片庫", show=False, priority=True),
        Binding("ctrl+u", "update_document", "更新文件", show=False, priority=True),
        Binding("ctrl+c", "quit", "退出", show=False),
    ]

    mode: reactive[str] = reactive("emotional")
    is_loading: reactive[bool] = reactive(False)
    rag_mode: str = "fast"

    def __init__(self) -> None:
        super().__init__()
        self.messages: list[dict] = []
        self.api_base = "http://localhost:8000"
        self.state = AppState()
        self._project_manager = ProjectManager()
        self._conv_store = ConversationStore()

    def compose(self) -> ComposeResult:
        yield MenuBar(id="menu-bar")
        with Horizontal(id="main-horizontal"):
            with Vertical(id="chat-column"):
                yield VerticalScroll(
                    Vertical(id="messages"),
                    id="chat-scroll",
                )
                yield OptionList(id="slash-hints")
                ta = ChatInput(id="chat-input")
                ta.show_line_numbers = False
                input_frame = Vertical(ta, id="input-frame")
                input_frame.border_title = "⌨ Message"
                yield Vertical(input_frame, id="input-box")
            with Vertical(id="right-panel"):
                rp = ResearchPanel(id="research-panel")
                rp.border_title = "◇ Research 研究面板"
                yield rp
                mp = MemoryPanel(id="memory-panel")
                mp.border_title = "◇ Memory 記憶面板"
                yield mp
        yield StatusBar(id="status-bar")

    async def on_mount(self) -> None:
        projects = []
        try:
            projects = await self._project_manager.list_projects()
        except Exception:
            pass

        needs_onboarding = not projects
        if needs_onboarding:
            from modals import OnboardingScreen
            await self.push_screen(OnboardingScreen(), callback=self._on_onboarding_done)
            return

        await self._init_app(projects)

    async def _on_onboarding_done(self, result: str | None) -> None:
        """Onboarding 完成後初始化。"""
        projects = await self._project_manager.list_projects()
        if not projects:
            proj = await self._project_manager.create_project("My Project")
            projects = [proj]
        await self._init_app(projects)

    async def _init_app(self, projects: list[dict]) -> None:
        """正常初始化流程（onboarding 後或直接啟動）。"""
        if projects:
            self.state.current_project = projects[0]
            from paths import project_root, ensure_project_dirs
            from conversation.store import ConversationStore
            pid = projects[0]["id"]
            ensure_project_dirs(pid)
            self._conv_store = ConversationStore(
                db_path=project_root(pid) / "conversations.db"
            )
        self._update_menu_bar()
        self._update_status()
        container = self.query_one("#messages", Vertical)
        welcome = WelcomeBlock()
        welcome.border_title = "[#d4a27a]◈[/] De-insight v0.7"
        await container.mount(welcome)
        self.query_one("#chat-input", ChatInput).focus()
        self._refresh_memory_panel()
        self._refresh_knowledge_panel()
        self._reindex_pending_memories()
