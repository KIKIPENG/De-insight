"""De-insight TUI — Elia 風格的對話介面"""

import json
import sys
import time
from dataclasses import dataclass, field
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
from modals import MemoryConfirmModal, ProjectModal
from panels import (
    ImportModal, InsightConfirmModal, MemoryDetailModal, MemoryManageModal,
    MemoryItem, MemoryPanel, MemorySaveModal, ResearchPanel, SearchModal,
)
from conversation.store import ConversationStore
from projects.manager import ProjectManager
from settings import SettingsScreen, load_env

SPINNER_FRAMES = ["|", "/", "—", "\\"]


@dataclass
class AppState:
    current_project: dict | None = None
    pending_memories: list[dict] = field(default_factory=list)
    interactive_depth: int = 0
    current_conversation_id: str | None = None
    cached_memory_count: int = 0
    current_interactive_block: object = None  # InteractiveBlock | None
    last_rag_sources: list[dict] = field(default_factory=list)

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
    """Enter 送出，Shift+Enter 換行，Tab 選擇提示，Escape 清空。"""

    BINDINGS = [
        Binding("enter", "submit", "送出", show=False, priority=True),
        Binding("escape", "clear_input", "清空", show=False),
    ]

    # 互動選項模式：choices 列表 + 當前高亮 index
    _choices: list[str] = []
    _choice_idx: int = 0

    class Submitted(TextArea.Changed):
        """使用者按下 Enter 送出。"""

    @property
    def in_choice_mode(self) -> bool:
        return bool(self._choices)

    def set_choices(self, choices: list[str]) -> None:
        """進入選擇模式，將選項寫入輸入框。"""
        self._choices = list(choices)
        self._choice_idx = 0
        self.read_only = True
        self._render_choices()

    def clear_choices(self) -> None:
        """退出選擇模式。"""
        self._choices = []
        self._choice_idx = 0
        self.read_only = False
        self.text = ""

    def _render_choices(self) -> None:
        """根據 _choice_idx 渲染選項文字到輸入框。"""
        lines = []
        for i, c in enumerate(self._choices):
            prefix = ">" if i == self._choice_idx else " "
            lines.append(f"{prefix} {c}")
        self.text = "\n".join(lines)

    async def _on_key(self, event) -> None:
        if event.key == "tab":
            event.prevent_default()
            event.stop()
            if self._choices:
                # 選擇模式：Tab 循環選項
                self._choice_idx = (self._choice_idx + 1) % len(self._choices)
                self._render_choices()
            else:
                self._cycle_slash_hints()
            return
        if event.key == "enter":
            if not self.text.strip():
                event.prevent_default()
                event.stop()
                return
            event.prevent_default()
            event.stop()
            self.post_message(self.Submitted(self))
            return
        if self._choices:
            # 選擇模式下攔截所有其他按鍵（不允許編輯）
            event.prevent_default()
            event.stop()
            return
        # 不攔截其他按鍵（讓 Ctrl+C 等正常運作）
        await super()._on_key(event)

    def _cycle_slash_hints(self) -> None:
        """Tab 循環選擇 slash-hints 中的選項，填入輸入框。"""
        try:
            hints = self.app.query_one("#slash-hints", OptionList)
        except Exception:
            return
        if not hints.has_class("-visible"):
            return
        count = hints.option_count
        if count == 0:
            return
        current = hints.highlighted
        next_idx = 0 if current is None else (current + 1) % count
        hints.highlighted = next_idx
        option = hints.get_option_at_index(next_idx)
        if option and option.id:
            self.text = option.id

    def action_submit(self) -> None:
        self.post_message(self.Submitted(self))

    def action_clear_input(self) -> None:
        if self._choices:
            self.clear_choices()
        else:
            self.text = ""


class MenuBar(Static):
    """Mac 風格頂部功能列，單行可點擊。"""

    # Each item: (label, action, start_col, end_col) — populated on render
    _ITEMS = [
        ("New", "new_chat"),
        ("Project", "open_project_modal"),
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
    _project_name: str | None = None
    _pending_count: int = 0

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
        project_name: str | None = None,
        pending_count: int = 0,
    ) -> None:
        """Update mode/model/status and trigger re-render."""
        self._mode = mode
        self._model = model
        self._memory_count = memory_count
        self._has_kg = has_kg
        self._rag_mode = rag_mode
        self._last_reindex = last_reindex
        self._project_name = project_name
        self._pending_count = pending_count
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

        # Project name
        if self._project_name:
            text.append("  ")
            col += 2
            text.append("│", style="#2a2a2a")
            col += 1
            text.append(" ", style="")
            col += 1
            proj_label = f"● {self._project_name}"
            text.append(proj_label, style="bold #7dd3fc")
            col += sum(2 if ord(c) > 0x7F else 1 for c in proj_label)

        # Pending memories count
        if self._pending_count > 0:
            text.append("  ")
            col += 2
            pending_label = f"💡 {self._pending_count} 待確認"
            pending_start = col
            text.append(pending_label, style="bold #f0c674")
            col += sum(2 if ord(c) > 0x7F else 1 for c in pending_label)
            self._regions.append((pending_start, col, "confirm_pending_memories"))

        return text

    def on_click(self, event) -> None:
        x = event.x
        for start, end, action in self._regions:
            if start <= x < end:
                if action == "_mode_emotional":
                    self.app.mode = "emotional"
                elif action == "_mode_rational":
                    self.app.mode = "rational"
                elif action == "confirm_pending_memories":
                    self.app.action_confirm_pending_memories()
                else:
                    self.app.action_from_menu(action)
                return


class WelcomeBlock(Vertical):
    """初始歡迎畫面，含最近對話歷史。"""

    def compose(self) -> ComposeResult:
        yield Static(
            "[bold #fafafa]◆ De-insight[/]  [dim]你的思想策展人[/]",
        )
        yield Static(
            "[#8b949e]  把還說不清楚的東西說清楚。\n"
            "  想法來自你，它幫你找到骨架。[/]",
        )
        yield Static(
            "[dim #6e7681]────────────────────────────────[/]",
        )
        yield Static(
            "[dim #fafafa]◇ 功能[/]\n\n"
            "[#8b949e]  △ 策展人對話　感性／理性模式切換（ctrl+e）\n"
            "  △ 知識庫　　　匯入文獻，對話時自動引用\n"
            "  △ 記憶系統　　留下洞見、問題、感性反應\n"
            "  △ 專案管理　　不同創作脈絡分開（ctrl+p）[/]",
        )
        yield Static(
            "[dim #6e7681]────────────────────────────────[/]",
        )
        yield Static(
            "[dim #fafafa]◇ 一個創作者的工作路徑[/]\n\n"
            "[#8b949e]  累積文獻與閱讀　→　和策展人反覆對話\n"
            "        ↓\n"
            "  沉澱洞見與問題　→　知識庫建立連結\n"
            "        ↓\n"
            "      準備好了　→　寫出一份論述[/]",
        )
        yield Static(
            "[dim #6e7681]────────────────────────────────[/]",
        )
        yield Static("[dim #fafafa]◇ 最近的對話[/]")
        yield Vertical(id="welcome-history")
        yield Static(
            "[dim #6e7681]────────────────────────────────[/]",
        )
        yield Static(
            "[dim #484f58]  made by KIKI PENG with love[/]",
        )

    async def on_mount(self) -> None:
        await self._load_recent_conversations()

    async def _load_recent_conversations(self) -> None:
        store = ConversationStore()
        conversations = await store.list_conversations()
        conversations = conversations[:10]
        container = self.query_one("#welcome-history")
        if not conversations:
            await container.mount(
                Static("[dim #484f58]  尚無對話記錄[/]")
            )
            return
        for c in conversations:
            updated = c.get("updated_at", "")[:16]
            title = c.get("title", "未命名對話")
            entry = Static(
                f"  [#484f58]{updated}[/]  [#8b949e]{title}[/]",
                classes="history-entry",
                name=c["id"],
            )
            await container.mount(entry)

    def on_click(self, event) -> None:
        widget = event.widget if hasattr(event, 'widget') else None
        if widget and isinstance(widget, Static) and widget.has_class("history-entry"):
            conv_id = widget.name
            if conv_id:
                self.app._load_conversation(conv_id)


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
        elif self._action_name == "view_sources":
            self.app.action_view_sources()


class ChatboxActions(Horizontal):
    """訊息下方的操作列。"""

    def __init__(self, show_insight: bool = True, has_sources: bool = False, **kwargs) -> None:
        super().__init__(**kwargs)
        self._show_insight = show_insight
        self._has_sources = has_sources

    def compose(self) -> ComposeResult:
        yield ActionLink("複製", "copy", classes="action-link")
        if self._show_insight:
            yield ActionLink("save insight", "save_insight", classes="action-link")
        yield ActionLink("記憶", "save_memory", classes="action-link")
        if self._has_sources:
            yield ActionLink("查看出處", "view_sources", classes="action-link source-link")


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
        self._concepts: list[str] = []
        self._has_sources: bool = False
        self.add_class(f"chatbox-{role}")
        self._breath_timer: Timer | None = None
        self._breath_frame = 0

    def on_mount(self) -> None:
        if self.role == "user":
            self.border_title = "◇ You"
        else:
            self.border_title = "◆ De-insight"

    @staticmethod
    def _clean_callouts(text: str) -> str:
        import re
        # 移除 callout 標記
        text = re.sub(r'^>?\s*\[!(INSIGHT|THEORY|QUESTION|QUOTE|CONFIRM)\]\s*\n?', '', text, flags=re.MULTILINE)
        # 移除互動標記 <<SELECT: ...>>, <<CONFIRM: ...>>, <<INPUT: ...>>, <<MULTI: ...>>
        text = re.sub(r'<<(SELECT|CONFIRM|INPUT|MULTI)[:：].*?>>', '', text, flags=re.DOTALL)
        return text

    @staticmethod
    def _parse_concept_marks(text: str) -> tuple[str, list[str], bool]:
        """解析 [[概念]] 標記。回傳 (rendered_text, concepts_list, has_concepts)。
        有概念時用 Rich markup（天藍色底線），無概念時原文不動。
        """
        import re
        concepts = []
        has = bool(re.search(r'\[\[.+?\]\]', text))
        def replace(m):
            concept = m.group(1)
            concepts.append(concept)
            return f"[underline #7dd3fc]{concept}[/underline #7dd3fc]"
        rendered = re.sub(r'\[\[(.+?)\]\]', replace, text)
        return rendered, concepts, has

    def compose(self) -> ComposeResult:
        if self._streaming:
            yield Static("", classes="chatbox-body stream-body", id="stream-text")
        else:
            cleaned = self._clean_callouts(self._content)
            rendered, self._concepts, has_concepts = self._parse_concept_marks(cleaned)
            if has_concepts:
                yield Static(rendered, classes="chatbox-body")
            else:
                yield Markdown(rendered, classes="chatbox-body")
        if not self._system:
            if self.role == "assistant":
                self._has_sources = bool(
                    getattr(self.app, 'state', None) and self.app.state.last_rag_sources
                )
                yield ChatboxActions(
                    show_insight=True, has_sources=self._has_sources,
                    classes="chatbox-actions",
                )
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
            cleaned = self._clean_callouts(self._content)
            rendered, self._concepts, has_concepts = self._parse_concept_marks(cleaned)
            if has_concepts:
                body = Static(rendered, classes="chatbox-body")
            else:
                body = Markdown(rendered, classes="chatbox-body")
            await self.mount(body, before=stream_el)
            await stream_el.remove()
            # Add source button if sources available
            if self.role == "assistant" and not self._has_sources:
                self._has_sources = bool(
                    getattr(self.app, 'state', None) and self.app.state.last_rag_sources
                )
                if self._has_sources:
                    try:
                        actions = self.query_one(ChatboxActions)
                        await actions.mount(
                            ActionLink("查看出處", "view_sources", classes="action-link source-link")
                        )
                    except NoMatches:
                        pass
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
        color: #7dd3fc;
    }
    ActionLink.source-link:hover {
        color: #bae6fd;
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
        max-height: 16;
        padding: 0 1;
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
        Binding("ctrl+c", "quit", "退出", show=False),
    ]

    mode: reactive[str] = reactive("emotional")
    is_loading: reactive[bool] = reactive(False)
    rag_mode: str = "fast"  # "fast" = naive+context_only, "deep" = hybrid+LLM合成

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
                rp.border_title = "◇ Knowledge"
                yield rp
                mp = MemoryPanel(id="memory-panel")
                mp.border_title = "◇ Memories"
                yield mp
        yield StatusBar(id="status-bar")

    async def on_mount(self) -> None:
        # v0.2 舊資料偵測
        if Path("data/lightrag").exists() and not Path("data/projects").exists():
            self.notify(
                "偵測到 v0.2 知識庫（data/lightrag/），請手動搬移或重新匯入。",
                severity="warning", timeout=10)
        self._update_menu_bar()
        self._update_status()
        container = self.query_one("#messages", Vertical)
        welcome = WelcomeBlock()
        welcome.border_title = "◈ De-insight v0.3"
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
            project_id = self.state.current_project["id"] if self.state.current_project else "default"
            try:
                from rag.knowledge_graph import has_knowledge
                has_kg = has_knowledge(project_id=project_id)
            except Exception:
                has_kg = False
            project_name = self.state.current_project["name"] if self.state.current_project else None
            pending_count = len(self.state.pending_memories)
            menu.set_state(
                mode=self.mode,
                model=model,
                memory_count=mem_count,
                has_kg=has_kg,
                rag_mode=self.rag_mode,
                last_reindex=_get_reindex_age(),
                project_name=project_name,
                pending_count=pending_count,
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
            from rag.knowledge_graph import insert_pdf, insert_url, reset_rag, get_rag
            _pid = self.state.current_project["id"] if self.state.current_project else "default"
            reset_rag()
            get_rag(project_id=_pid)  # prime cache for correct project
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
            _pid = self.state.current_project["id"] if self.state.current_project else "default"
            if not has_knowledge(project_id=_pid):
                self.notify("知識庫為空，請先匯入文件 (ctrl+f)")
                return
            result, sources = await query_knowledge(query)
            if sources:
                self.state.last_rag_sources = sources
            await self._update_research_panel(result)
        except Exception as e:
            self.notify(f"搜尋失敗: {e}")

    def action_manage_memories(self) -> None:
        def on_dismiss(result: str | None) -> None:
            self._refresh_memory_panel()
            if result and result.startswith("discuss:"):
                content = result.removeprefix("discuss:")
                self._start_discussion_from_memory(content)
            elif result and result.startswith("__insert__:"):
                content = result[len("__insert__:"):]
                self.action_close_modals()
                self.fill_input(content)

        self.push_screen(MemoryManageModal(), callback=on_dismiss)

    def action_close_modals(self) -> None:
        """關閉所有開啟的 modal，回到對話。"""
        while len(self.screen_stack) > 1:
            self.pop_screen()

    def action_view_sources(self) -> None:
        """顯示知識庫來源 Modal。"""
        sources = self.state.last_rag_sources
        if sources:
            from modals import SourceModal
            self.push_screen(SourceModal(sources))
        else:
            self.notify("這則回應沒有知識庫來源", timeout=2)

    def fill_input(self, text: str) -> None:
        """把文字填入 ChatInput，保持游標在最後，不送出。"""
        try:
            chat_input = self.query_one("#chat-input", ChatInput)
            chat_input.clear()
            chat_input.insert(text)
            chat_input.focus()
        except Exception:
            pass

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
            self.state.cached_memory_count = stats.get("total", 0)
        except Exception:
            self.state.cached_memory_count = len(mems) if mems else 0
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
        self.state.current_conversation_id = None
        container = self.query_one("#messages", Vertical)
        container.remove_children()
        self.call_after_refresh(self._mount_welcome)
        self.query_one("#chat-input", ChatInput).focus()

    async def _mount_welcome(self) -> None:
        container = self.query_one("#messages", Vertical)
        welcome = WelcomeBlock()
        welcome.border_title = "◈ De-insight v0.3"
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
        "/project": "open_project_modal",
        "/pending": "confirm_pending_memories",
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
        ("/project", "切換專案"),
        ("/pending", "記憶待確認"),
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
            "| `/project` | 切換專案 |\n"
            "| `/pending` | 記憶待確認 |\n"
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
                type=data["type"],
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
        """對話結束後，背景自動從使用者發言抽取記憶候選。"""
        try:
            from memory.thought_tracker import extract_memories
            items = await extract_memories(user_text, self._quick_llm_call)
            if not items:
                return
            for item in items:
                item["source"] = user_text[:80]
            self.state.pending_memories.extend(items)
            self._update_menu_bar()
        except Exception:
            pass  # 記憶抽取失敗不影響主流程

    def _update_menubar_pending_count(self) -> None:
        self._update_menu_bar()

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
            ta = self.query_one("#chat-input", ChatInput)
            if not ta.in_choice_mode:
                self._update_slash_hints()

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        if event.option_list.id == "slash-hints":
            opt_id = event.option.id
            # Normal slash command
            ta = self.query_one("#chat-input", ChatInput)
            ta.text = opt_id
            event.option_list.remove_class("-visible")
            ta.focus()
            self._submit_chat()

    # ── chat ──

    def on_chat_input_submitted(self, event: ChatInput.Submitted) -> None:
        """ChatInput 按 Enter 送出。"""
        ta = self.query_one("#chat-input", ChatInput)
        # 選擇模式：Enter 確認當前選項
        if ta.in_choice_mode:
            self._resolve_inline_choice()
            return
        self._submit_chat()

    @work(exclusive=True, group="submit_chat", thread=False)
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

        # 如果有活躍的互動提問，把輸入當作回應送出
        if self.state.current_interactive_block:
            self.state.current_interactive_block = None
            self.query_one("#input-frame", Vertical).border_title = "⌨ Message"
            self._send_as_user(text)
            return

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

        # 對話持久化：第一條訊息時建立對話記錄
        if self.state.current_conversation_id is None:
            project_id = self.state.current_project["id"] if self.state.current_project else None
            self.state.current_conversation_id = await self._conv_store.create_conversation(project_id)
            title = text[:30].strip().replace("\n", " ")
            await self._conv_store.set_title(self.state.current_conversation_id, title)
        await self._conv_store.add_message(self.state.current_conversation_id, "user", text)

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
        self.state.last_rag_sources = []  # Clear stale sources
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
                _pid = self.state.current_project["id"] if self.state.current_project else "default"
                try:
                    from rag.knowledge_graph import query_knowledge, has_knowledge
                    if has_knowledge(project_id=_pid):
                        is_deep = self.rag_mode == "deep"
                        result, sources = await query_knowledge(
                            user_msg,
                            mode="hybrid" if is_deep else "naive",
                            context_only=not is_deep,
                        )
                        if result and len(result.strip()) > 10:
                            rag_context = f"\n\n[知識庫參考]\n{result[:2000]}"
                            await self._update_research_panel(result)
                            if sources:
                                self.state.last_rag_sources = sources
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
                _pid = self.state.current_project["id"] if self.state.current_project else "default"
                try:
                    from rag.knowledge_graph import query_knowledge, has_knowledge
                    if has_knowledge(project_id=_pid):
                        is_deep = self.rag_mode == "deep"
                        result, sources = await query_knowledge(
                            user_msg,
                            mode="hybrid" if is_deep else "naive",
                            context_only=not is_deep,
                        )
                        if result and len(result.strip()) > 10:
                            ctx_parts.append(f"[知識庫參考]\n{result[:2000]}")
                            await self._update_research_panel(result)
                            if sources:
                                self.state.last_rag_sources = sources
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

            # 解析互動提問標記
            import re as _re_interactive
            from interaction.prompt_parser import parse_interactive_blocks
            clean_content, interactive_blocks = parse_interactive_blocks(full_content)

            if _re_interactive.search(r'<<\w+', clean_content):
                self.notify("策展人格式未完整解析", severity="warning", timeout=4)

            if interactive_blocks:
                bubble.set_responding(False)
                bubble.stream_update(clean_content)
                await bubble.finalize_stream()
                self.messages.append({"role": "assistant", "content": clean_content})
                if self.state.current_conversation_id:
                    await self._conv_store.add_message(
                        self.state.current_conversation_id, "assistant", clean_content)
                if self.state.interactive_depth < 3:
                    self._handle_interactive_blocks(interactive_blocks)
                else:
                    self.state.interactive_depth = 0
                    self.notify("互動提問深度上限，請直接輸入", timeout=3)
            else:
                bubble.set_responding(False)
                bubble.stream_update(full_content)
                await bubble.finalize_stream()
                self.messages.append({"role": "assistant", "content": full_content})
                if self.state.current_conversation_id:
                    await self._conv_store.add_message(
                        self.state.current_conversation_id, "assistant", full_content)

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

    # ── project management ──

    def action_open_project_modal(self) -> None:
        self._load_and_show_project_modal()

    @work(exclusive=True)
    async def _load_and_show_project_modal(self) -> None:
        projects = await self._project_manager.list_projects()
        current_id = self.state.current_project["id"] if self.state.current_project else None

        def on_dismiss(result) -> None:
            if result is None:
                return
            action, data = result
            if action == 'create':
                self._create_and_switch_project(data)
            elif action == 'switch':
                self._switch_project(data)

        self.app.push_screen(ProjectModal(projects, current_id=current_id), callback=on_dismiss)

    @work(exclusive=True)
    async def _create_and_switch_project(self, name: str) -> None:
        project = await self._project_manager.create_project(name)
        await self._do_switch_project(project)

    @work(exclusive=True)
    async def _switch_project(self, project: dict) -> None:
        await self._do_switch_project(project)

    async def _do_switch_project(self, project: dict) -> None:
        import gc
        self.state.current_project = project
        self.state.current_conversation_id = None
        await self._project_manager.touch_project(project["id"])
        from rag.knowledge_graph import reset_rag
        reset_rag()
        gc.collect()
        self.messages = []
        container = self.query_one("#messages", Vertical)
        await container.remove_children()
        welcome = WelcomeBlock()
        welcome.border_title = "◈ De-insight v0.3"
        await container.mount(welcome)
        self._refresh_memory_panel()
        self._update_menu_bar()
        self.notify(f"已切換到：{project['name']}", timeout=2)

    @work(exclusive=True)
    async def _delete_project(self, project: dict) -> None:
        await self._project_manager.delete_project(project["id"])
        if self.state.current_project and self.state.current_project["id"] == project["id"]:
            self.state.current_project = None
            self._update_menu_bar()
        self.notify(f"已刪除專案：{project['name']}", timeout=2)

    # ── pending memory confirmation ──

    def action_confirm_pending_memories(self) -> None:
        if not self.state.pending_memories:
            self.notify("目前沒有待確認的記憶")
            return

        def on_dismiss(result) -> None:
            if result is None:
                self.state.pending_memories.clear()
                self._update_menu_bar()
                return
            self._save_confirmed_memories(result)

        self.push_screen(MemoryConfirmModal(self.state.pending_memories), callback=on_dismiss)

    @work(exclusive=True)
    async def _save_confirmed_memories(self, items: list) -> None:
        project_id = self.state.current_project["id"] if self.state.current_project else None
        for item in items:
            if isinstance(item, dict):
                await add_memory(
                    type=item["type"],
                    content=item["content"],
                    source=item.get("source", ""),
                    topic=item.get("topic", ""),
                    category=item.get("category", ""),
                    project_id=project_id,
                )
        self.state.pending_memories.clear()
        self._refresh_memory_panel()
        self._update_menu_bar()
        self.notify(f"已儲存 {len(items)} 條記憶")

    # ── interactive prompt handling ──

    def _handle_interactive_blocks(self, blocks: list) -> None:
        """把互動選項直接顯示在輸入框內，Tab 切換，Enter 選擇。"""
        if not blocks:
            return
        block = blocks[0]  # 一次處理一個
        self.state.current_interactive_block = block

        ta = self.query_one("#chat-input", ChatInput)
        frame = self.query_one("#input-frame", Vertical)
        frame.border_title = f"◇ {block.prompt}"

        if block.type == 'confirm':
            ta.set_choices(["是，繼續", "不，我還想調整"])
        elif block.type == 'select':
            ta.set_choices(block.choices)
        elif block.type == 'multi':
            ta.set_choices(block.choices)
        elif block.type == 'input':
            pass  # 自由輸入，不需要選項

        ta.focus()

    def _resolve_inline_choice(self) -> None:
        """從輸入框的選擇模式中取得選中項目並送出。"""
        ta = self.query_one("#chat-input", ChatInput)
        if not ta.in_choice_mode:
            return
        idx = ta._choice_idx
        choices = ta._choices
        text = choices[idx] if idx < len(choices) else ""
        ta.clear_choices()
        self.state.current_interactive_block = None
        self.query_one("#input-frame", Vertical).border_title = "⌨ Message"
        if text:
            self._send_as_user(text)

    @work(exclusive=False, thread=False)
    async def _send_as_user(self, text: str) -> None:
        from textual.worker import WorkerCancelled
        if not text.strip():
            return
        self.state.interactive_depth += 1
        try:
            container = self.query_one("#messages", Vertical)
            await container.mount(Chatbox("user", text))
            self._scroll_to_bottom()
            self.messages.append({'role': 'user', 'content': text})
            # 對話持久化：互動回應也要存入
            if self.state.current_conversation_id:
                await self._conv_store.add_message(
                    self.state.current_conversation_id, "user", text)
            worker = self._stream_response()
            try:
                await worker.wait()
            except WorkerCancelled:
                pass
        finally:
            self.state.interactive_depth -= 1

    def _scroll_to_bottom(self) -> None:
        self.query_one("#chat-scroll", VerticalScroll).scroll_end(animate=False)

    @work(exclusive=True, thread=False)
    async def _load_conversation(self, conversation_id: str) -> None:
        """從 ConversationStore 載入對話歷史。"""
        messages = await self._conv_store.get_messages(conversation_id)
        if not messages:
            return
        self.state.current_conversation_id = conversation_id
        self.messages = list(messages)
        container = self.query_one("#messages", Vertical)
        await container.remove_children()
        for m in messages:
            await container.mount(Chatbox(m["role"], m["content"]))
        self._scroll_to_bottom()

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
        _pid = self.state.current_project["id"] if self.state.current_project else "default"
        try:
            from rag.knowledge_graph import query_knowledge, has_knowledge
            if has_knowledge(project_id=_pid):
                is_deep = self.rag_mode == "deep"
                result, sources = await query_knowledge(
                    user_msg,
                    mode="hybrid" if is_deep else "naive",
                    context_only=not is_deep,
                )
                if result and len(result.strip()) > 10:
                    await self._update_research_panel(result)
                    if sources:
                        self.state.last_rag_sources = sources
                    augmented.insert(insert_idx, {
                        "role": "system",
                        "content": f"知識庫相關資訊：\n{result[:2000]}\n\n（以上為參考資料，請務必用繁體中文回覆使用者。）",
                    })
        except Exception:
            pass

        return augmented


if __name__ == "__main__":
    DeInsightApp().run()
