"""De-insight — Widget 子類別。"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from io import StringIO
from pathlib import Path

from rich.console import Console
from rich.markdown import Markdown as RichMarkdown
from rich.style import Style
from rich.text import Text
from rich.theme import Theme
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.css.query import NoMatches
from textual.timer import Timer
from textual.widgets import Markdown, OptionList, Static, TextArea

from conversation.store import ConversationStore


DE_INSIGHT_THEME = Theme({
    "markdown.h1":          Style(color="#d4a27a", bold=True),
    "markdown.h2":          Style(color="#c4925a", bold=True),
    "markdown.h3":          Style(color="#b88a50", bold=True, italic=True),
    "markdown.h1.border":   Style(color="#3d2a1a"),
    "markdown.h2.border":   Style(color="#2a2a2a"),
    "markdown.code":        Style(color="#d4a27a", bgcolor="#1a1a1a"),
    "markdown.code_block":  Style(color="#c9d1d9", bgcolor="#111111"),
    "markdown.block_quote": Style(color="#8b949e"),
    "markdown.link":        Style(color="#d4a27a", underline=True),
    "markdown.link_url":    Style(color="#6e7681"),
    "markdown.bullet":      Style(color="#d4a27a"),
    "markdown.hr":          Style(color="#2a2a2a"),
    "markdown.item.bullet": Style(color="#d4a27a"),
    "markdown.strong":      Style(bold=True, color="#fafafa"),
    "markdown.em":          Style(italic=True, color="#c9d1d9"),
})


def _render_markdown(text: str) -> Text:
    """用琥珀主題渲染 markdown，回傳 Rich Text 物件。"""
    sio = StringIO()
    console = Console(
        file=sio,
        theme=DE_INSIGHT_THEME,
        highlight=False,
        width=80,
        no_color=False,
    )
    console.print(RichMarkdown(text))
    return Text.from_ansi(sio.getvalue())


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
    last_imported_source: str | None = None
    pending_images: list = field(default_factory=list)

LANCEDB_DIR = Path(__file__).parent / "data" / "lancedb"


def _get_reindex_age() -> str:
    """取得距離上次 reindex 的時間描述。回傳空字串表示尚無索引。"""
    if not LANCEDB_DIR.exists():
        return ""
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

    _choices: list[str] = []
    _choice_idx: int = 0

    class Submitted(TextArea.Changed):
        """使用者按下 Enter 送出。"""

    @property
    def in_choice_mode(self) -> bool:
        return bool(self._choices)

    def set_choices(self, choices: list[str]) -> None:
        self._choices = list(choices)
        self._choice_idx = 0
        self.read_only = True
        self._render_choices()

    def clear_choices(self) -> None:
        self._choices = []
        self._choice_idx = 0
        self.read_only = False
        self.text = ""

    def _render_choices(self) -> None:
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
            event.prevent_default()
            event.stop()
            return
        # @ mention: load gallery selected images
        if event.character == "@":
            self._load_gallery_selected()
        await super()._on_key(event)

    def _load_gallery_selected(self) -> None:
        """讀取 selected.json，把選取的圖片載入 pending_images。"""
        try:
            from paths import DATA_ROOT
            import json as _json
            sel_path = DATA_ROOT / "selected.json"
            if not sel_path.exists():
                return
            ids = _json.loads(sel_path.read_text())
            if not ids:
                return
            state = getattr(self.app, "state", None)
            if not state or not state.current_project:
                return
            pid = state.current_project["id"]
            from paths import project_root
            img_dir = project_root(pid) / "images"
            # Read image metadata from selected.json ids
            # For simplicity, just add paths for images that exist
            pending = []
            import lancedb
            lance_dir = project_root(pid) / "lancedb"
            if not lance_dir.exists():
                return
            db = lancedb.connect(str(lance_dir))
            if "images" not in db.table_names():
                return
            tbl = db.open_table("images")
            df = tbl.to_pandas().drop(columns=["vector"], errors="ignore")
            selected = df[df["id"].isin(ids)]
            for _, row in selected.iterrows():
                fpath = img_dir / row.get("filename", "")
                if fpath.exists():
                    pending.append(str(fpath))
            if pending:
                state.pending_images = pending
                self.app._update_menu_bar()
                self.app.notify(f"{len(pending)} 張圖片已帶入")
                # Clear selected.json
                sel_path.write_text("[]")
        except Exception:
            pass

    def _cycle_slash_hints(self) -> None:
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

    _ITEMS = [
        ("New", "new_chat"),
        ("Project", "open_project_modal"),
        ("Knowledge", "manage_documents"),
        ("Memory", "manage_memories"),
        ("Gallery", "open_gallery"),
        ("Settings", "open_settings"),
    ]

    _mode: str = "emotional"
    _model: str = ""
    _memory_count: int = 0
    _has_kg: bool = False
    _rag_mode: str = "fast"
    _project_name: str | None = None
    _pending_count: int = 0
    _gallery_selected: int = 0

    def __init__(self, **kwargs) -> None:
        super().__init__("", **kwargs)
        self._regions: list[tuple[int, int, str]] = []

    def set_state(
        self,
        mode: str = "emotional",
        model: str = "",
        memory_count: int = 0,
        has_kg: bool = False,
        rag_mode: str = "fast",
        project_name: str | None = None,
        pending_count: int = 0,
        gallery_selected: int = 0,
    ) -> None:
        self._mode = mode
        self._model = model
        self._memory_count = memory_count
        self._has_kg = has_kg
        self._rag_mode = rag_mode
        self._project_name = project_name
        self._pending_count = pending_count
        self._gallery_selected = gallery_selected
        self._regions.clear()
        self.refresh()

    def render(self) -> Text:
        text = Text()
        col = 1
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

        e_start = col
        text.append("感性", style="bold #d4a27a" if self._mode == "emotional" else "#484f58")
        col += 4
        self._regions.append((e_start, col, "_mode_emotional"))

        text.append(" ")
        col += 1

        r_start = col
        text.append("理性", style="bold #d4a27a" if self._mode == "rational" else "#484f58")
        col += 4
        self._regions.append((r_start, col, "_mode_rational"))

        text.append("  ")
        col += 2
        text.append("│", style="#2a2a2a")
        col += 1
        text.append(" ", style="")
        col += 1

        mem_style = "#8b949e" if self._memory_count > 0 else "#484f58"
        mem_label = f"記憶:{self._memory_count}"
        text.append(mem_label, style=mem_style)
        col += sum(2 if ord(c) > 0x7F else 1 for c in mem_label)

        text.append("  ")
        col += 2

        kg_label = "知識庫:✓" if self._has_kg else "知識庫:—"
        kg_style = "#8b949e" if self._has_kg else "#484f58"
        text.append(kg_label, style=kg_style)
        col += sum(2 if ord(c) > 0x7F else 1 for c in kg_label)

        text.append("  ")
        col += 2

        rag_label = "RAG:快速" if self._rag_mode == "fast" else "RAG:深度"
        rag_start = col
        text.append(rag_label, style="#6e7681")
        col += sum(2 if ord(c) > 0x7F else 1 for c in rag_label)
        self._regions.append((rag_start, col, "toggle_rag_mode"))

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

        if self._pending_count > 0:
            text.append("  ")
            col += 2
            pending_label = f"💡 {self._pending_count} 待確認"
            pending_start = col
            text.append(pending_label, style="bold #f0c674")
            col += sum(2 if ord(c) > 0x7F else 1 for c in pending_label)
            self._regions.append((pending_start, col, "confirm_pending_memories"))

        if self._gallery_selected > 0:
            text.append("  ")
            col += 2
            gal_label = f"🖼 {self._gallery_selected} 圖"
            gal_start = col
            text.append(gal_label, style="bold #d4a27a")
            col += sum(2 if ord(c) > 0x7F else 1 for c in gal_label)
            self._regions.append((gal_start, col, "open_gallery"))

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
            "[bold #fafafa][#d4a27a]◆[/] De-insight[/]  [dim]你的思想策展人[/]",
        )
        yield Static(
            "[#8b949e]  把還說不清楚的東西說清楚。\n"
            "  想法來自你，它幫你找到骨架。[/]",
        )
        yield Static(
            "[dim #6e7681]────────────────────────────────[/]",
        )
        yield Static(
            "[dim #fafafa][#d4a27a]◇[/] 功能[/]\n\n"
            "[#8b949e]  [#d4a27a]△[/] 策展人對話　感性／理性模式切換（ctrl+e）\n"
            "  [#d4a27a]△[/] 知識庫　　　匯入文獻，對話時自動引用\n"
            "  [#d4a27a]△[/] 記憶系統　　留下洞見、問題、感性反應\n"
            "  [#d4a27a]△[/] 專案管理　　不同創作脈絡分開（ctrl+p）[/]",
        )
        yield Static(
            "[dim #6e7681]────────────────────────────────[/]",
        )
        yield Static(
            "[dim #fafafa][#d4a27a]◇[/] 一個創作者的工作路徑[/]\n\n"
            "[#8b949e]  累積文獻與閱讀　→　和策展人反覆對話\n"
            "        ↓\n"
            "  沉澱洞見與問題　→　知識庫建立連結\n"
            "        ↓\n"
            "      準備好了　→　寫出一份論述[/]",
        )
        yield Static(
            "[dim #6e7681]────────────────────────────────[/]",
        )
        yield Static("[dim #fafafa][#d4a27a]◇[/] 最近的對話[/]")
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
        project_id = self.app.state.current_project["id"] if self.app.state.current_project else None
        conversations = await store.list_conversations(project_id=project_id)
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

    _BREATH_COLORS = [
        "#2a2a2a", "#2e2620", "#332b1a", "#3d2a1a",
        "#4a3520", "#5a4025", "#6a4d2a", "#7a5a30",
        "#8a6635", "#9a7340", "#a87d48", "#b88a50",
        "#c4925a", "#d4a27a",
        "#c4925a", "#b88a50", "#a87d48", "#9a7340",
        "#8a6635", "#7a5a30", "#6a4d2a", "#5a4025",
        "#4a3520", "#3d2a1a", "#332b1a", "#2e2620",
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
            self.border_title = "[#d4a27a]◆[/] De-insight"

    @staticmethod
    def _clean_callouts(text: str) -> str:
        import re
        text = re.sub(r'^```(?:markdown|md)\s*\n', '', text.strip())
        text = re.sub(r'\n```\s*$', '', text)
        text = re.sub(r'^>?\s*\[!(INSIGHT|THEORY|QUESTION|QUOTE|CONFIRM)\]\s*\n?', '', text, flags=re.MULTILINE)
        text = re.sub(r'<<(SELECT|CONFIRM|INPUT|MULTI)[:：].*?>>', '', text, flags=re.DOTALL)
        return text

    @staticmethod
    def _parse_concept_marks(text: str) -> tuple[str, list[str], bool]:
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
            elif self.role == "assistant":
                yield Static(_render_markdown(cleaned), classes="chatbox-body")
            else:
                yield Static(cleaned, classes="chatbox-body")
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
        self._content = content
        try:
            self.query_one("#stream-text", Static).update(content)
        except NoMatches:
            pass

    async def finalize_stream(self) -> None:
        self._streaming = False
        try:
            stream_el = self.query_one("#stream-text", Static)
            cleaned = self._clean_callouts(self._content)
            rendered, self._concepts, has_concepts = self._parse_concept_marks(cleaned)
            if has_concepts:
                stream_el.update(rendered)
            elif self.role == "assistant":
                stream_el.update(_render_markdown(cleaned))
            else:
                stream_el.update(cleaned)
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
            self.border_title = "[#d4a27a]◆[/] De-insight"
            self.add_class("responding")
            self._breath_frame = 0
            if self.is_mounted:
                self._breath_timer = self.set_interval(0.08, self._breathe)
        else:
            if self._breath_timer:
                self._breath_timer.stop()
                self._breath_timer = None
            self.border_title = "[#d4a27a]◆[/] De-insight"
            self.remove_class("responding")
            self.styles.border = ("round", "#444444")
            self.styles.border_title_color = "#d4a27a"


class StatusBar(Static):
    """底部狀態列。"""

    pass
