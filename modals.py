"""v0.7 Modals — all ModalScreen subclasses live here."""

from textual import work
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Label, ProgressBar, Static, TextArea

from memory.store import delete_memory, get_memories, get_memory_stats


def _get_project_paths(app) -> "tuple[Path | None, Path | None]":
    """回傳 (memories_db_path, lancedb_dir)。"""
    from pathlib import Path
    from paths import project_root
    state = getattr(app, 'state', None)
    pid = state.current_project["id"] if state and state.current_project else None
    if not pid:
        return None, None
    root = project_root(pid)
    return root / "memories.db", root / "lancedb"


# ── ProjectModal ────────────────────────────────────────────────────

class ProjectModal(ModalScreen):
    """專案管理 — 風格與 MemoryManageModal 統一。"""

    BINDINGS = [("escape", "close", "關閉")]

    CSS = """
    ProjectModal { align: center middle; }
    #project-box {
        width: 60; height: auto; max-height: 85%; padding: 1 2;
        border: round #3a3a3a; background: #0a0a0a;
        border-title-color: #fafafa;
    }
    #project-scroll {
        height: auto; max-height: 60%;
    }
    .proj-entry {
        height: auto; padding: 0 1; color: #8b949e;
    }
    .proj-entry:hover {
        color: #fafafa; background: #111111;
    }
    .proj-entry.-active {
        color: #7dd3fc; background: #111111;
    }
    .proj-sep {
        height: 1; margin: 0; color: #2a2a2a;
    }
    #project-new-input {
        margin: 1 0; background: #111111; color: #fafafa;
        border: tall #3a3a3a; padding: 0 1;
    }
    .proj-actions { height: 1; margin: 1 0 0 0; }
    .proj-btn {
        background: transparent; color: #484f58;
        border: none; height: 1; min-width: 0;
        margin: 0 1 0 0; padding: 0 1;
    }
    .proj-btn:hover { color: #fafafa; }
    .proj-del-btn {
        background: transparent; color: #484f58;
        border: none; height: 1; min-width: 0;
        margin: 0; padding: 0;
    }
    .proj-del-btn:hover { color: #ff6b6b; }
    """

    def __init__(self, projects: list[dict], current_id: str | None = None) -> None:
        super().__init__()
        self._projects = projects
        self._current_id = current_id

    def compose(self) -> ComposeResult:
        box = Vertical(id="project-box")
        box.border_title = "◇ 專案管理"
        with box:
            if self._projects:
                yield Static(f"[#6e7681]{len(self._projects)} 個專案[/]")
                yield Static("[dim #2a2a2a]" + "─" * 54 + "[/]", classes="proj-sep")
                with VerticalScroll(id="project-scroll"):
                    for p in self._projects:
                        is_active = p["id"] == self._current_id if self._current_id else False
                        marker = "●" if is_active else " "
                        last = p.get("last_active", "")
                        entry = Static(
                            f"[#7dd3fc]{marker}[/] {p['name']}  [dim #484f58]{last}[/]",
                            classes="proj-entry" + (" -active" if is_active else ""),
                            name=p["id"],
                        )
                        yield entry
                yield Static("[dim #2a2a2a]" + "─" * 54 + "[/]", classes="proj-sep")
            else:
                yield Static("[dim #484f58]尚無專案，在下方輸入名稱新增[/]")
            yield Input(placeholder="新專案名稱…", id="project-new-input")
            with Horizontal(classes="proj-actions"):
                yield Button("新增", id="proj-create", classes="proj-btn")
            yield Button("← 回到對話", classes="back-btn")

    def on_mount(self) -> None:
        try:
            self.query_one("#project-new-input", Input).focus()
        except Exception:
            pass

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id == "project-new-input":
            name = event.value.strip()
            if name:
                self.dismiss(("create", name))

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "proj-create":
            name = self.query_one("#project-new-input", Input).value.strip()
            if name:
                self.dismiss(("create", name))
        elif event.button.has_class("back-btn"):
            self.dismiss(None)

    def on_click(self, event) -> None:
        # Find if clicked on a proj-entry Static
        widget = event.widget if hasattr(event, 'widget') else None
        if widget and isinstance(widget, Static) and widget.has_class("proj-entry"):
            pid = widget.name
            if pid:
                for p in self._projects:
                    if p["id"] == pid:
                        self.dismiss(("switch", p))
                        return

    def action_close(self) -> None:
        self.dismiss(None)


# ── MemoryConfirmModal ──────────────────────────────────────────────

class MemoryConfirmModal(ModalScreen):
    """記憶確認 — 用 checkbox Static 代替 SelectionList 避免 dict unhashable。"""

    BINDINGS = [("escape", "close", "關閉")]

    CSS = """
    MemoryConfirmModal { align: center middle; }
    #memconf-box {
        width: 72; height: auto; max-height: 85%; padding: 1 2;
        border: round #3a3a3a; background: #0a0a0a;
        border-title-color: #fafafa;
    }
    #memconf-scroll {
        height: auto; max-height: 60%;
    }
    .memconf-entry {
        height: auto; padding: 0 1; color: #8b949e;
    }
    .memconf-entry:hover {
        color: #fafafa; background: #111111;
    }
    .memconf-entry.-checked {
        color: #c9d1d9;
    }
    .memconf-sep {
        height: 1; margin: 0; color: #2a2a2a;
    }
    .memconf-actions { height: 1; margin: 1 0 0 0; }
    .memconf-btn {
        background: transparent; color: #484f58;
        border: none; height: 1; min-width: 0;
        margin: 0 1 0 0; padding: 0 1;
    }
    .memconf-btn:hover { color: #fafafa; }
    """

    def __init__(self, pending_memories: list[dict]) -> None:
        super().__init__()
        self._pending = pending_memories
        self._checked: list[bool] = [True] * len(pending_memories)

    def compose(self) -> ComposeResult:
        box = Vertical(id="memconf-box")
        box.border_title = "◇ 記憶確認"
        with box:
            yield Static("[#8b949e]這段對話有什麼你想留下來的嗎？[/]")
            yield Static("[dim #2a2a2a]" + "─" * 66 + "[/]", classes="memconf-sep")
            with VerticalScroll(id="memconf-scroll"):
                for i, m in enumerate(self._pending):
                    icons = {"insight": "💡", "question": "❓", "reaction": "💭"}
                    icon = icons.get(m.get("type", ""), "◇")
                    label = f"☑ {icon} [{m['type']}] {m['content']}"
                    yield Static(label, classes="memconf-entry -checked", name=str(i))
            yield Static("[dim #2a2a2a]" + "─" * 66 + "[/]", classes="memconf-sep")
            with Horizontal(classes="memconf-actions"):
                yield Button("全部儲存", id="memconf-save-all", classes="memconf-btn")
                yield Button("只存勾選", id="memconf-save-sel", classes="memconf-btn")
                yield Button("略過", id="memconf-skip", classes="memconf-btn")
            yield Button("← 回到對話", classes="back-btn")

    def on_click(self, event) -> None:
        widget = event.widget if hasattr(event, 'widget') else None
        if widget and isinstance(widget, Static) and widget.has_class("memconf-entry"):
            idx = int(widget.name)
            self._checked[idx] = not self._checked[idx]
            m = self._pending[idx]
            icons = {"insight": "💡", "question": "❓", "reaction": "💭"}
            icon = icons.get(m.get("type", ""), "◇")
            if self._checked[idx]:
                widget.update(f"☑ {icon} [{m['type']}] {m['content']}")
                widget.add_class("-checked")
            else:
                widget.update(f"☐ {icon} [{m['type']}] {m['content']}")
                widget.remove_class("-checked")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "memconf-save-all":
            self.dismiss(list(self._pending))
        elif event.button.id == "memconf-save-sel":
            selected = [m for i, m in enumerate(self._pending) if self._checked[i]]
            self.dismiss(selected if selected else None)
        elif event.button.id == "memconf-skip":
            self.dismiss(None)
        elif event.button.has_class("back-btn"):
            self.dismiss(None)

    def action_close(self) -> None:
        self.dismiss(None)


# ── SourceModal ───────────────────────────────────────────────────

class SourceModal(ModalScreen):
    """知識庫來源檢視。"""

    BINDINGS = [("escape", "close", "關閉")]

    CSS = """
    SourceModal { align: center middle; }
    #source-box {
        width: 72; height: auto; max-height: 85%; padding: 1 2;
        border: round #3a3a3a; background: #0a0a0a;
        border-title-color: #7dd3fc;
    }
    .source-item { height: auto; margin: 1 0; }
    .source-title { color: #7dd3fc; }
    .source-snippet {
        color: #8b949e; padding: 0 0 0 2;
        height: auto;
        border-left: solid #2a2a2a;
    }
    .source-meta { color: #484f58; }
    .src-sep { height: 1; color: #2a2a2a; }
    .src-cite-btn {
        background: transparent; color: #484f58;
        border: none; height: 1; min-width: 0; padding: 0;
    }
    .src-cite-btn:hover { color: #7dd3fc; }
    """

    def __init__(self, sources: list[dict]) -> None:
        super().__init__()
        self._sources = sources

    def compose(self) -> ComposeResult:
        box = Vertical(id="source-box")
        box.border_title = "◇ 知識庫出處"
        with box:
            yield Static(f"[#6e7681]{len(self._sources)} 個來源[/]")
            yield Static("[dim #2a2a2a]" + "─" * 66 + "[/]", classes="src-sep")
            with VerticalScroll():
                for i, s in enumerate(self._sources):
                    with Vertical(classes="source-item"):
                        yield Static(
                            f"[{i+1}] {s.get('title', '未知來源')}",
                            classes="source-title",
                        )
                        if s.get("snippet"):
                            yield Static(s["snippet"], classes="source-snippet")
                        if s.get("file"):
                            yield Static(f"來源：{s['file']}", classes="source-meta")
                        yield Button("▸ 引用到對話", classes="src-cite-btn", name=str(i))
                    if i < len(self._sources) - 1:
                        yield Static("[dim #2a2a2a]" + "─" * 66 + "[/]", classes="src-sep")
            yield Static("[dim #2a2a2a]" + "─" * 66 + "[/]", classes="src-sep")
            yield Button("← 回到對話", classes="back-btn")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.has_class("back-btn"):
            self.dismiss(None)
            return
        if event.button.has_class("src-cite-btn"):
            idx = int(event.button.name)
            s = self._sources[idx]
            snippet = s.get("snippet", "")[:300]
            title = s.get("title", "未知來源")
            formatted = f"\u300c{snippet}\u300d\u2014 {title}"
            self.dismiss(f"__cite__:{formatted}")

    def action_close(self) -> None:
        self.dismiss(None)


# ── DocReaderModal ───────────────────────────────────────────────


class _Paragraph(Static):
    """可點擊選取的段落。"""

    def __init__(self, text: str, index: int, **kwargs) -> None:
        super().__init__(text, **kwargs)
        self._index = index
        self._selected = False

    def on_click(self) -> None:
        self._selected = not self._selected
        if self._selected:
            self.add_class("para-selected")
        else:
            self.remove_class("para-selected")
        # Update selection count in parent modal
        modal = self.screen
        if isinstance(modal, DocReaderModal):
            modal._update_selection_hint()


class DocReaderModal(ModalScreen):
    """文獻全文閱讀 + 段落選取引用。"""

    BINDINGS = [("escape", "close", "關閉")]

    CSS = """
    DocReaderModal { align: center middle; }
    #doc-reader-box {
        width: 90; height: auto; max-height: 92%; padding: 1 2;
        border: round #3a3a3a; background: #0a0a0a;
        border-title-color: #7dd3fc;
    }
    #doc-reader-scroll { max-height: 75%; }
    .doc-para {
        color: #c9d1d9; padding: 0 1; margin: 0 0 1 0;
        height: auto;
    }
    .doc-para:hover { background: #1a1a1a; }
    .para-selected { background: #2a1f14; border-left: thick #d4a27a; }
    .para-selected:hover { background: #3d2a1a; }
    #doc-reader-hint { color: #484f58; height: 1; margin: 0; }
    .doc-reader-actions { height: auto; margin: 1 0 0 0; }
    """

    def __init__(self, doc: dict, project_id: str = "default") -> None:
        super().__init__()
        self._doc = doc
        self._project_id = project_id
        self._full_text = ""
        self._paragraphs: list[str] = []

    def compose(self) -> ComposeResult:
        title = self._doc.get("title", "未知文獻")
        box = Vertical(id="doc-reader-box")
        box.border_title = f"◇ {title}"
        with box:
            yield VerticalScroll(
                Static("載入中…", id="doc-reader-text"),
                id="doc-reader-scroll",
            )
            yield Static("[dim #484f58]點擊段落以選取，再點取消選取[/]", id="doc-reader-hint")
            yield Static("[dim #2a2a2a]" + "─" * 84 + "[/]")
            with Horizontal(classes="doc-reader-actions"):
                yield Button("▸ 引用選取段落", classes="src-cite-btn", name="cite")
                yield Button("▸ 引用全文摘要", classes="src-cite-btn", name="cite_all")
                yield Button("← 關閉", classes="back-btn")

    def on_mount(self) -> None:
        self._load_content()

    @work(exclusive=True)
    async def _load_content(self) -> None:
        text = await self._extract_text()
        self._full_text = text
        if not text:
            try:
                self.query_one("#doc-reader-text", Static).update(
                    "[dim #484f58]無法讀取內容[/]"
                )
            except Exception:
                pass
            return

        # Split into paragraphs (double newline or page breaks)
        import re
        raw_paras = re.split(r'\n{2,}', text)
        self._paragraphs = [p.strip() for p in raw_paras if p.strip()]

        # Replace loading text with paragraph widgets
        try:
            scroll = self.query_one("#doc-reader-scroll", VerticalScroll)
            await scroll.remove_children()
            for i, para in enumerate(self._paragraphs):
                display = para
                if len(display) > 500:
                    display = display[:500] + "…"
                await scroll.mount(_Paragraph(display, i, classes="doc-para"))
        except Exception:
            pass

    def _set_loading_status(self, msg: str) -> None:
        try:
            self.query_one("#doc-reader-text", Static).update(f"[dim #484f58]{msg}[/]")
        except Exception:
            pass

    async def _extract_text(self) -> str:
        """Extract full text from PDF (PyMuPDF), URL (Jina Reader), or LightRAG kv store."""
        import json as _json
        from pathlib import Path
        title = self._doc.get("title", "")
        source_path = self._doc.get("source_path", "")
        source_type = self._doc.get("source_type", "pdf")

        # 1) Local PDF — use PyMuPDF
        if source_path and Path(source_path).exists() and source_path.lower().endswith(".pdf"):
            self._set_loading_status("正在讀取 PDF…")
            try:
                import fitz  # PyMuPDF
                doc = fitz.open(source_path)
                pages = []
                for page in doc:
                    pages.append(page.get_text())
                doc.close()
                return "\n\n".join(pages)
            except ImportError:
                pass
            except Exception:
                pass

        # 2) URL source — use Jina Reader
        is_url = source_type in ("url", "doi") or (
            source_path.startswith("http://") or source_path.startswith("https://")
        )
        if is_url and source_path.startswith("http"):
            self._set_loading_status("正在透過 Jina Reader 抓取網頁…")
            try:
                from rag.knowledge_graph import _fetch_with_jina_reader
                text, _meta = await _fetch_with_jina_reader(source_path)
                if text and len(text) >= 50:
                    return text
            except Exception:
                pass
            # Jina Reader failed — try direct HTTP fetch + basic HTML strip
            self._set_loading_status("Jina Reader 失敗，嘗試直接抓取…")
            try:
                import httpx
                async with httpx.AsyncClient(timeout=20, follow_redirects=True) as client:
                    resp = await client.get(source_path)
                    if resp.status_code < 400:
                        ct = resp.headers.get("content-type", "")
                        if "html" in ct:
                            text = self._strip_html(resp.text)
                            if text and len(text) >= 50:
                                return text
                        elif "text" in ct:
                            return resp.text.strip()
            except Exception:
                pass

        # 3) Try LightRAG kv_store_full_docs (cached ingestion content)
        self._set_loading_status("正在搜尋已建圖內容…")
        try:
            from paths import project_root
            kv_path = project_root(self._project_id) / "lightrag" / "kv_store_full_docs.json"
            if kv_path.exists():
                data = _json.loads(kv_path.read_text(encoding="utf-8"))
                # Search for matching title
                for key, val in data.items():
                    content = ""
                    if isinstance(val, dict):
                        content = val.get("content", "") or val.get("original_content", "")
                    elif isinstance(val, str):
                        content = val
                    if content and title and title.lower() in content[:200].lower():
                        return content
                # If no title match, try source_path match
                for key, val in data.items():
                    if source_path and source_path in key:
                        if isinstance(val, dict):
                            return val.get("content", "") or val.get("original_content", "")
                        elif isinstance(val, str):
                            return val
        except Exception:
            pass

        return ""

    @staticmethod
    def _strip_html(html: str) -> str:
        """Basic HTML → plain text fallback."""
        import re
        # Remove script/style blocks
        text = re.sub(r'<(script|style)[^>]*>.*?</\1>', '', html, flags=re.DOTALL | re.IGNORECASE)
        # Remove tags
        text = re.sub(r'<[^>]+>', ' ', text)
        # Collapse whitespace
        text = re.sub(r'[ \t]+', ' ', text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        # Decode common entities
        import html as _html
        text = _html.unescape(text)
        return text.strip()

    def _get_selected_paragraphs(self) -> list[str]:
        """Return full text of all selected paragraphs in order."""
        selected = []
        try:
            for widget in self.query(_Paragraph):
                if widget._selected:
                    idx = widget._index
                    if idx < len(self._paragraphs):
                        selected.append(self._paragraphs[idx])
        except Exception:
            pass
        return selected

    def _update_selection_hint(self) -> None:
        count = len(self._get_selected_paragraphs())
        try:
            hint = self.query_one("#doc-reader-hint", Static)
            if count > 0:
                hint.update(f"[#d4a27a]已選取 {count} 個段落[/]  [dim #484f58]點擊段落以選取/取消[/]")
            else:
                hint.update("[dim #484f58]點擊段落以選取，再點取消選取[/]")
        except Exception:
            pass

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.has_class("back-btn"):
            self.dismiss(None)
            return
        if event.button.name == "cite":
            selected = self._get_selected_paragraphs()
            if not selected:
                self.notify("請先點擊選取要引用的段落", severity="warning", timeout=3)
                return
            title = self._doc.get("title", "未知文獻")
            excerpt = "\n\n".join(selected)
            self.dismiss(f"__cite__:[{title}]\n{excerpt}")
        elif event.button.name == "cite_all":
            title = self._doc.get("title", "未知文獻")
            excerpt = self._full_text[:2000] if self._full_text else ""
            if excerpt:
                self.dismiss(f"__cite__:[{title}]\n{excerpt}")
            else:
                self.notify("無內容可引用", timeout=2)

    def action_close(self) -> None:
        self.dismiss(None)


# ── DocumentManageModal ───────────────────────────────────────────

class DocumentManageModal(ModalScreen):
    """文獻管理 — 列出已匯入文獻，可刪除記錄。"""

    BINDINGS = [("escape", "close", "關閉")]

    CSS = """
    DocumentManageModal { align: center middle; }
    #doc-box {
        width: 78; height: auto; max-height: 85%; padding: 1 2;
        border: round #3a3a3a; background: #0a0a0a;
        border-title-color: #fafafa;
    }
    #doc-scroll { height: auto; max-height: 70%; }
    .doc-entry { height: auto; padding: 0 1; color: #8b949e; }
    .doc-entry:hover { color: #fafafa; background: #111111; }
    .doc-sep { height: 1; color: #2a2a2a; }
    .doc-del-btn {
        background: transparent; color: #484f58;
        border: none; height: 1; min-width: 3;
        margin: 0; padding: 0;
    }
    .doc-del-btn:hover { color: #ff6b6b; }
    .doc-actions { height: 1; margin: 1 0 0 0; }
    .doc-action-btn {
        background: transparent; color: #484f58;
        border: none; height: 1; min-width: 0;
        margin: 0 1 0 0; padding: 0 1;
    }
    .doc-action-btn:hover { color: #fafafa; }
    """

    def __init__(self, project_id: str = "default") -> None:
        super().__init__()
        self._project_id = project_id
        self._docs: list[dict] = []

    def compose(self) -> ComposeResult:
        box = Vertical(id="doc-box")
        box.border_title = "◇ 文獻管理"
        with box:
            yield Static("", id="doc-count")
            yield Static("[dim #2a2a2a]" + "─" * 72 + "[/]", classes="doc-sep")
            yield VerticalScroll(id="doc-scroll")
            yield Static("[dim #2a2a2a]" + "─" * 72 + "[/]", classes="doc-sep")
            yield Static("[dim #484f58]刪除僅移除記錄，知識庫內容保留[/]")
            with Horizontal(classes="doc-actions"):
                yield Button("+ 匯入新文獻 (ctrl+f)", id="doc-import", classes="doc-action-btn")
                yield Button("← 回到對話", classes="back-btn doc-action-btn")

    def on_mount(self) -> None:
        self._load_docs()

    @work(exclusive=True)
    async def _load_docs(self) -> None:
        from conversation.store import ConversationStore
        from paths import project_root
        store = ConversationStore(db_path=project_root(self._project_id) / "conversations.db")
        self._docs = await store.list_documents(self._project_id)
        try:
            self.query_one("#doc-count", Static).update(
                f"[#6e7681]{len(self._docs)} 份文獻[/]"
            )
        except Exception:
            pass
        await self._render_list()

    async def _render_list(self) -> None:
        try:
            scroll = self.query_one("#doc-scroll", VerticalScroll)
        except Exception:
            return
        await scroll.remove_children()
        if not self._docs:
            await scroll.mount(Static("[dim #484f58]尚無匯入文獻，按 ctrl+f 匯入[/]"))
            return
        for doc in self._docs:
            time_str = (doc.get("imported_at") or "")[:16]
            title = doc.get("title", "未知")
            if len(title) > 40:
                title = title[:40] + "…"
            row = Horizontal(classes="doc-entry")
            await scroll.mount(row)
            await row.mount(Static(f"[#484f58]{time_str}[/]  {title}"))
            await row.mount(Button("✗", classes="doc-del-btn", name=doc["id"]))

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.has_class("back-btn"):
            self.dismiss(None)
            return
        if event.button.id == "doc-import":
            self.dismiss("__import__")
            return
        if event.button.has_class("doc-del-btn"):
            doc_id = event.button.name
            if doc_id:
                self._delete_doc(doc_id)

    @work(exclusive=True)
    async def _delete_doc(self, doc_id: str) -> None:
        from conversation.store import ConversationStore
        from paths import project_root
        store = ConversationStore(db_path=project_root(self._project_id) / "conversations.db")
        await store.delete_document(doc_id)
        self._docs = [d for d in self._docs if d["id"] != doc_id]
        try:
            self.query_one("#doc-count", Static).update(
                f"[#6e7681]{len(self._docs)} 份文獻[/]"
            )
        except Exception:
            pass
        await self._render_list()
        self.notify("已刪除文獻記錄")

    def action_close(self) -> None:
        self.dismiss(None)


# ── BulkImportModal ───────────────────────────────────────────────

class BulkImportModal(ModalScreen):
    """批量匯入文獻。"""

    BINDINGS = [("escape", "close_if_done", "關閉")]

    CSS = """
    BulkImportModal { align: center middle; }
    #bulk-box {
        width: 72; height: auto; max-height: 85%; padding: 1 2;
        border: round #3a3a3a; background: #0a0a0a;
        border-title-color: #fafafa;
    }
    #bulk-input {
        height: 8; margin: 1 0;
        background: #111111; color: #fafafa;
        border: tall #3a3a3a;
    }
    #bulk-input:focus { border: tall #666666; }
    #bulk-progress { height: auto; max-height: 50%; }
    .bulk-item { height: 1; padding: 0 1; color: #8b949e; }
    .bulk-item.-done { color: #7dd3fc; }
    .bulk-item.-error { color: #ff6b6b; }
    .bulk-item.-running { color: #fafafa; }
    .bulk-actions { height: 1; margin: 1 0 0 0; }
    .bulk-btn {
        background: transparent; color: #484f58;
        border: none; height: 1; min-width: 0;
        margin: 0 1 0 0; padding: 0 1;
    }
    .bulk-btn:hover { color: #fafafa; }
    """

    def __init__(self) -> None:
        super().__init__()
        self._done = False
        self._items: list[Static] = []

    def compose(self) -> ComposeResult:
        box = Vertical(id="bulk-box")
        box.border_title = "◇ 批量匯入"
        with box:
            yield Static("[#8b949e]每行一個來源（PDF 路徑、URL、DOI）[/]")
            yield TextArea("", id="bulk-input")
            yield Static("", id="bulk-status")
            yield VerticalScroll(id="bulk-progress")
            with Horizontal(classes="bulk-actions"):
                yield Button("開始匯入", id="bulk-start", classes="bulk-btn")
                yield Button("← 回到對話", classes="back-btn bulk-btn")

    def on_mount(self) -> None:
        try:
            self.query_one("#bulk-input", TextArea).focus()
        except Exception:
            pass

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.has_class("back-btn"):
            if self._done or not self._items:
                self.dismiss(None)
            return
        if event.button.id == "bulk-start":
            text = self.query_one("#bulk-input", TextArea).text
            sources = [s.strip() for s in text.strip().split("\n") if s.strip()]
            if sources:
                self._start_imports(sources)

    @work(exclusive=True)
    async def _start_imports(self, sources: list[str]) -> None:
        # Build progress list
        try:
            scroll = self.query_one("#bulk-progress", VerticalScroll)
        except Exception:
            return
        await scroll.remove_children()
        self._items = []
        for src in sources:
            label = src[:60] + ("…" if len(src) > 60 else "")
            item = Static(f"○  {label}", classes="bulk-item")
            await scroll.mount(item)
            self._items.append(item)

        _pid = self.app.state.current_project["id"] if self.app.state.current_project else "default"
        done_count = 0
        for i, src in enumerate(sources):
            self._items[i].update(f"⏳ {src[:60]}")
            self._items[i].add_class("-running")
            self._update_status(done_count, len(sources))
            try:
                await self.app._import_one(src, _pid)
                self._items[i].update(f"✅ {src[:60]}")
                self._items[i].remove_class("-running")
                self._items[i].add_class("-done")
                done_count += 1
            except Exception as e:
                self._items[i].update(f"❌ {src[:60]}  {str(e)[:30]}")
                self._items[i].remove_class("-running")
                self._items[i].add_class("-error")
            self._update_status(done_count, len(sources))

        self._done = True
        if sources:
            self.app.state.last_imported_source = sources[-1]
        self._update_status(done_count, len(sources), finished=True)

    def _update_status(self, done: int, total: int, finished: bool = False) -> None:
        try:
            status = self.query_one("#bulk-status", Static)
            if finished:
                status.update(f"[#7dd3fc]全部完成 {done}/{total}  按 ESC 關閉[/]")
            else:
                status.update(f"[#8b949e]匯入中 {done}/{total}[/]")
        except Exception:
            pass

    def action_close_if_done(self) -> None:
        if self._done or not self._items:
            self.dismiss(None)


# ── RelationModal ─────────────────────────────────────────────────

class RelationModal(ModalScreen):
    """記憶關聯 — 顯示雙向向量相似的記憶對。"""

    BINDINGS = [("escape", "close", "關閉")]

    CSS = """
    RelationModal { align: center middle; }
    #rel-box {
        width: 72; height: auto; max-height: 85%; padding: 1 2;
        border: round #3a3a3a; background: #0a0a0a;
        border-title-color: #fafafa;
    }
    #rel-scroll { height: auto; max-height: 70%; }
    .rel-pair { height: auto; padding: 0 1; margin: 0 0 1 0; color: #8b949e; }
    .rel-pair.-highlight { border-left: thick #7dd3fc; }
    .rel-sep { height: 1; color: #2a2a2a; }
    """

    def compose(self) -> ComposeResult:
        box = Vertical(id="rel-box")
        box.border_title = "◇ 記憶關聯"
        with box:
            yield Static("", id="rel-count")
            yield Static("[dim #2a2a2a]" + "─" * 66 + "[/]", classes="rel-sep")
            yield VerticalScroll(id="rel-scroll")
            yield Static("[dim #2a2a2a]" + "─" * 66 + "[/]", classes="rel-sep")
            yield Button("← 回到對話", classes="back-btn")

    def on_mount(self) -> None:
        self._find_relations()

    @work(exclusive=True)
    async def _find_relations(self) -> None:
        try:
            scroll = self.query_one("#rel-scroll", VerticalScroll)
        except Exception:
            return
        await scroll.remove_children()
        await scroll.mount(Static("[dim #484f58]搜尋中…[/]"))

        try:
            from memory.store import get_memories
            from memory.vectorstore import search_similar, has_index
            _db_path, _lancedb_dir = _get_project_paths(self.app)
            if not has_index(lancedb_dir=_lancedb_dir):
                await scroll.remove_children()
                await scroll.mount(Static("[dim #484f58]向量索引為空[/]"))
                return

            mems = await get_memories(limit=100, db_path=_db_path)
            if len(mems) < 2:
                await scroll.remove_children()
                await scroll.mount(Static("[dim #484f58]記憶不足，需要至少 2 條[/]"))
                return

            # Build similarity map: for each memory, find top-6 similar
            sim_map: dict[int, list[tuple[int, float]]] = {}
            for m in mems:
                results = await search_similar(m["content"], limit=7, lancedb_dir=_lancedb_dir)
                sim_map[m["id"]] = [
                    (r["id"], r.get("score", 0))
                    for r in results if r["id"] != m["id"]
                ][:6]

            # Find bidirectional pairs with score >= 0.55
            pairs = []
            seen_pairs = set()
            mem_by_id = {m["id"]: m for m in mems}
            for a_id, a_sims in sim_map.items():
                for b_id, score_ab in a_sims:
                    if score_ab < 0.55:
                        continue
                    pair_key = (min(a_id, b_id), max(a_id, b_id))
                    if pair_key in seen_pairs:
                        continue
                    # Check bidirectional
                    b_sims = sim_map.get(b_id, [])
                    score_ba = next((s for bid, s in b_sims if bid == a_id), 0)
                    if score_ba >= 0.55:
                        avg_score = (score_ab + score_ba) / 2
                        seen_pairs.add(pair_key)
                        pairs.append((a_id, b_id, avg_score))

            pairs.sort(key=lambda x: x[2], reverse=True)

            await scroll.remove_children()
            if not pairs:
                await scroll.mount(Static("[dim #484f58]未找到足夠相似的記憶對[/]"))
                self.query_one("#rel-count", Static).update("[#6e7681]0 組關聯[/]")
                return

            self.query_one("#rel-count", Static).update(f"[#6e7681]{len(pairs)} 組關聯[/]")
            icons = {"insight": "💡", "question": "❓", "reaction": "💭"}
            for a_id, b_id, score in pairs[:20]:
                a = mem_by_id.get(a_id, {})
                b = mem_by_id.get(b_id, {})
                pct = int(score * 100)
                highlight = " -highlight" if pct >= 80 else ""
                icon_a = icons.get(a.get("type", ""), "◇")
                icon_b = icons.get(b.get("type", ""), "◇")
                text = (
                    f"{icon_a} {a.get('content', '?')[:40]}\n"
                    f"  ↔  {icon_b} {b.get('content', '?')[:40]}\n"
                    f"  [#6e7681]相似度 {pct}%[/]"
                )
                await scroll.mount(Static(text, classes=f"rel-pair{highlight}"))
        except Exception as e:
            await scroll.remove_children()
            await scroll.mount(Static(f"[#ff6b6b]載入失敗: {e}[/]"))

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.has_class("back-btn"):
            self.dismiss(None)

    def action_close(self) -> None:
        self.dismiss(None)


# ── ImportModal ──────────────────────────────────────────────────

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
            yield Input(placeholder="/path/to/file.pdf  或  https://...  或  10.1234/doi", id="import-input")
            yield Static("[dim #484f58]支援：PDF 路徑、URL、arXiv 連結、DOI（10.xxxx/...）[/]")
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


# ── SearchModal ──────────────────────────────────────────────────

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


# ── MemoryDetailModal ────────────────────────────────────────────

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
            self.dismiss(f"discuss:{self._mem['content']}")
        elif bid == "detail-insert":
            content = self._mem.get("content", "")
            self.dismiss(f"__insert__:{content}")
        elif bid == "btn-detail-del":
            self._do_delete()

    @work(exclusive=True)
    async def _do_delete(self) -> None:
        _db_path, _ = _get_project_paths(self.app)
        await delete_memory(self._mem["id"], db_path=_db_path)
        self.notify("已刪除")
        self.dismiss(None)

    def action_close(self) -> None:
        self.dismiss(None)


# ── MemoryManageModal ────────────────────────────────────────────

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

    def __init__(self, project_id: str | None = None) -> None:
        super().__init__()
        self._project_id = project_id
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
        _db_path, _ = _get_project_paths(self.app)
        stats = await get_memory_stats(db_path=_db_path)
        total = stats["total"]
        by_type = stats["by_type"]
        stats_text = f"[#6e7681]{total} 條記憶[/]"
        for t, cnt in by_type.items():
            icons = {"insight": "💡", "question": "❓", "reaction": "💭"}
            stats_text += f"  {icons.get(t, '◇')}{cnt}"
        try:
            self.query_one("#mem-stats", Static).update(stats_text)
        except Exception:
            pass

        self._all_mems = await get_memories(limit=100, db_path=_db_path)
        await self._render_list()

    async def _render_list(self) -> None:
        try:
            scroll = self.query_one("#mem-scroll", VerticalScroll)
        except Exception:
            return
        await scroll.remove_children()

        mems = self._all_mems
        if self._filter_type:
            mems = [m for m in mems if m["type"] == self._filter_type]
        if self._filter_category:
            mems = [m for m in mems if m.get("category") == self._filter_category]

        if not mems:
            await scroll.mount(Static("[dim #484f58]沒有符合的記憶[/]"))
            return

        icons = {"insight": "💡", "question": "❓", "reaction": "💭"}
        current_date = ""
        for m in mems:
            date_str = (m.get("created_at") or "")[:10]
            if date_str != current_date:
                current_date = date_str
                await scroll.mount(
                    Static(f"[#6e7681]── {date_str or '未知日期'} ──[/]", classes="time-header")
                )
            icon = icons.get(m["type"], "◇")
            time_str = (m.get("created_at") or "")[11:16]
            topic = m.get("topic", "")
            content = m["content"][:36] + ("…" if len(m["content"]) > 36 else "")
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

        if bid.startswith("flt-"):
            self._filter_type = name
            for btn in self.query(".filter-btn"):
                btn.remove_class("-active")
            event.button.add_class("-active")
            self._apply_filter()
            return

        if event.button.has_class("cat-btn"):
            self._filter_category = name
            for btn in self.query(".cat-btn"):
                btn.remove_class("-active")
            event.button.add_class("-active")
            self._apply_filter()
            return

        if event.button.has_class("mem-del-btn"):
            mem_id = int(name or "0")
            if mem_id:
                self._delete_and_reload(mem_id)

    @work(exclusive=True)
    async def _apply_filter(self) -> None:
        await self._render_list()

    @work(exclusive=True)
    async def _delete_and_reload(self, mem_id: int) -> None:
        _db_path, _ = _get_project_paths(self.app)
        await delete_memory(mem_id, db_path=_db_path)
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
                modal = self.screen
                if isinstance(modal, MemoryManageModal):
                    modal.dismiss(result)
        self.app.push_screen(MemoryDetailModal(self._mem), callback=on_result)


# ── InsightConfirmModal ──────────────────────────────────────────

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


# ── MemorySaveModal ──────────────────────────────────────────────

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


# ── KnowledgeModal ───────────────────────────────────────────────

class KnowledgeModal(ModalScreen):
    """知識庫管理 — 匯入 / 搜尋 / 批量 / 文獻管理 四合一。"""

    BINDINGS = [("escape", "close", "關閉")]

    CSS = """
    KnowledgeModal { align: center middle; }
    #kb-box {
        width: 78; height: auto; max-height: 85%; padding: 1 2;
        border: round #3a3a3a; background: #0a0a0a;
        border-title-color: #fafafa;
    }
    .kb-tabs { height: 1; margin: 0 0 1 0; }
    .kb-tab {
        background: transparent; color: #484f58;
        border: none; height: 1; min-width: 0;
        margin: 0; padding: 0 1;
    }
    .kb-tab:hover { color: #fafafa; }
    .kb-tab.-active { color: #fafafa; text-style: bold; }
    #kb-content { height: auto; max-height: 65%; }
    #kb-input {
        margin: 1 0; background: #111111; color: #fafafa;
        border: tall #3a3a3a; padding: 0 1;
    }
    #kb-bulk-input {
        height: 5; margin: 0 0 1 0;
        background: #111111; color: #fafafa;
        border: tall #3a3a3a;
    }
    #kb-bulk-input:focus { border: tall #666666; }
    #kb-paste-content {
        height: 10; margin: 1 0;
        background: #111111; color: #fafafa;
        border: tall #3a3a3a;
    }
    #kb-paste-content:focus { border: tall #666666; }
    .kb-doc-entry { height: 1; padding: 0 1; color: #8b949e; }
    .kb-doc-entry:hover { color: #fafafa; background: #111111; }
    .kb-doc-title { width: 1fr; height: 1; overflow: hidden; }
    .kb-del-btn {
        background: transparent; color: #484f58;
        border: none; height: 1; width: 6;
        margin: 0; padding: 0;
    }
    .kb-del-btn:hover { color: #ff6b6b; }
    .kb-read-btn {
        background: transparent; color: #7dd3fc;
        border: none; height: 1; width: 6;
        margin: 0; padding: 0;
    }
    .kb-read-btn:hover { color: #fafafa; }
    .kb-sep { height: 1; color: #2a2a2a; }
    .kb-progress { height: auto; max-height: 50%; }
    .kb-bulk-item { height: 1; padding: 0 1; color: #8b949e; }
    .kb-bulk-item.-done { color: #7dd3fc; }
    .kb-bulk-item.-error { color: #ff6b6b; }
    .kb-bulk-item.-running { color: #fafafa; }
    .kb-action-btn {
        background: transparent; color: #484f58;
        border: none; height: 1; min-width: 0;
        margin: 1 0 0 0; padding: 0 1;
    }
    .kb-action-btn:hover { color: #fafafa; }
    """

    _TAB_NAMES = ["匯入", "貼上", "搜尋", "批量", "文獻"]
    _TAB_IDS = ["import", "paste", "search", "bulk", "docs"]

    def __init__(self, project_id: str = "default", initial_tab: str = "import") -> None:
        super().__init__()
        self._project_id = project_id
        self._initial_tab = initial_tab
        self._current_tab = initial_tab
        self._docs: list[dict] = []
        self._bulk_done = False
        self._bulk_items: list[Static] = []
        self._bulk_parsed_items: list[dict] = []
        self._bulk_batch_meta: dict = {}

    def compose(self) -> ComposeResult:
        box = Vertical(id="kb-box")
        box.border_title = "◇ 知識庫"
        with box:
            with Horizontal(classes="kb-tabs"):
                for name, tid in zip(self._TAB_NAMES, self._TAB_IDS):
                    cls = "kb-tab -active" if tid == self._initial_tab else "kb-tab"
                    yield Button(name, id=f"kb-tab-{tid}", classes=cls)
            yield Static("[dim #2a2a2a]" + "─" * 72 + "[/]", classes="kb-sep")
            yield VerticalScroll(id="kb-content")
            yield Button("← 回到對話", classes="back-btn kb-action-btn")

    def on_mount(self) -> None:
        self._show_tab(self._initial_tab)

    def _show_tab(self, tab: str) -> None:
        self._current_tab = tab
        for tid in self._TAB_IDS:
            try:
                btn = self.query_one(f"#kb-tab-{tid}", Button)
                if tid == tab:
                    btn.add_class("-active")
                else:
                    btn.remove_class("-active")
            except Exception:
                pass
        self._render_tab()

    @work(exclusive=True)
    async def _render_tab(self) -> None:
        try:
            content = self.query_one("#kb-content", VerticalScroll)
        except Exception:
            return
        await content.remove_children()

        if self._current_tab == "import":
            await self._render_import(content)
        elif self._current_tab == "paste":
            await self._render_paste(content)
        elif self._current_tab == "search":
            await self._render_search(content)
        elif self._current_tab == "bulk":
            await self._render_bulk(content)
        elif self._current_tab == "docs":
            await self._render_docs(content)

    async def _render_import(self, container: VerticalScroll) -> None:
        await container.mount(Static("[#8b949e]輸入檔案路徑或網址[/]"))
        await container.mount(
            Input(placeholder="/path/to/file.pdf  或  https://...  或  10.1234/doi", id="kb-input")
        )
        await container.mount(
            Input(placeholder="自訂標題（選填，留空則自動擷取）", id="kb-import-title")
        )
        await container.mount(
            Static("[dim #484f58]支援：PDF 路徑、URL、arXiv 連結、DOI（10.xxxx/...）[/]")
        )
        try:
            self.query_one("#kb-input", Input).focus()
        except Exception:
            pass

    async def _render_search(self, container: VerticalScroll) -> None:
        await container.mount(Static("[#8b949e]輸入搜尋關鍵字[/]"))
        await container.mount(Input(placeholder="搜尋知識庫…", id="kb-input"))
        try:
            self.query_one("#kb-input", Input).focus()
        except Exception:
            pass

    async def _render_paste(self, container: VerticalScroll) -> None:
        await container.mount(Static("[#8b949e]手動貼上文章標題與內文[/]"))
        await container.mount(Input(placeholder="文章標題（必填）", id="kb-paste-title"))
        await container.mount(TextArea("", id="kb-paste-content"))
        await container.mount(Static("[dim #484f58]貼上完整內文後按 Enter（在標題欄）或按下匯入按鈕[/]"))
        await container.mount(Button("匯入貼上內容", id="kb-paste-submit", classes="kb-action-btn"))
        try:
            self.query_one("#kb-paste-title", Input).focus()
        except Exception:
            pass

    async def _render_bulk(self, container: VerticalScroll) -> None:
        self._bulk_done = False
        self._bulk_items = []
        await container.mount(Static(
            "[#8b949e]每行一個來源，或貼入 JSON 陣列[/]\n"
            "[dim #484f58]JSON 格式：[\"url\", ...] 或 [{\"url\": \"...\", \"title\": \"...\", \"tags\": [...]}][/]"
        ))
        await container.mount(TextArea("", id="kb-bulk-input"))
        await container.mount(Button("開始匯入", id="kb-bulk-start", classes="kb-action-btn"))
        await container.mount(Static("", id="kb-bulk-status"))
        await container.mount(VerticalScroll(id="kb-bulk-progress", classes="kb-progress"))
        try:
            self.query_one("#kb-bulk-input", TextArea).focus()
        except Exception:
            pass

    async def _render_docs(self, container: VerticalScroll) -> None:
        await container.mount(Static("[dim #484f58]載入中…[/]"))
        from conversation.store import ConversationStore
        from paths import project_root
        store = ConversationStore(db_path=project_root(self._project_id) / "conversations.db")
        self._docs = await store.list_documents(self._project_id)
        await container.remove_children()
        await container.mount(Static(f"[#6e7681]{len(self._docs)} 份文獻[/]"))
        await container.mount(Static("[dim #2a2a2a]" + "─" * 70 + "[/]", classes="kb-sep"))
        if not self._docs:
            await container.mount(Static("[dim #484f58]尚無匯入文獻[/]"))
        else:
            for doc in self._docs:
                time_str = (doc.get("imported_at") or "")[:16]
                title = doc.get("title", "未知")
                if len(title) > 40:
                    title = title[:40] + "…"
                row = Horizontal(classes="kb-doc-entry")
                await container.mount(row)
                await row.mount(Static(f"[#484f58]{time_str}[/]  {title}", classes="kb-doc-title"))
                await row.mount(Button("[閱讀]", classes="kb-read-btn", name=doc["id"]))
                await row.mount(Button("[刪除]", classes="kb-del-btn", name=doc["id"]))
        await container.mount(Static("[dim #484f58]刪除僅移除記錄，知識庫內容保留[/]"))

    # ── event handlers ──

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.has_class("back-btn"):
            self.dismiss(None)
            return
        if event.button.id and event.button.id.startswith("kb-tab-"):
            tab = event.button.id.removeprefix("kb-tab-")
            if tab != self._current_tab:
                self._show_tab(tab)
            return
        if event.button.id == "kb-bulk-start":
            try:
                text = self.query_one("#kb-bulk-input", TextArea).text.strip()
            except Exception:
                return
            if not text:
                return
            items, batch_meta = self._parse_bulk_input(text)
            if items:
                self._bulk_parsed_items = items
                self._bulk_batch_meta = batch_meta
                self._show_bulk_preview(items, batch_meta)
            return
        if event.button.id == "kb-bulk-confirm":
            self._run_bulk(self._bulk_parsed_items)
            return
        if event.button.id == "kb-bulk-reset":
            self._show_tab("bulk")
            return
        if event.button.id == "kb-paste-submit":
            try:
                title = self.query_one("#kb-paste-title", Input).value.strip()
                content = self.query_one("#kb-paste-content", TextArea).text.strip()
            except Exception:
                return
            if not title:
                self.notify("請輸入標題", severity="warning")
                return
            if not content:
                self.notify("請貼上內文", severity="warning")
                return
            self.dismiss(("import_text", {"title": title, "content": content}))
            return
        if event.button.has_class("kb-read-btn"):
            doc_id = event.button.name
            doc = next((d for d in self._docs if d.get("id") == doc_id), None)
            if doc:
                def _on_reader_result(result):
                    if result and isinstance(result, str) and result.startswith("__cite__:"):
                        self.dismiss(f"__fill__:{result[len('__cite__:'):]}")
                self.app.push_screen(DocReaderModal(doc, self._project_id), callback=_on_reader_result)
            return
        if event.button.has_class("kb-del-btn"):
            doc_id = event.button.name
            if doc_id:
                self._delete_doc(doc_id)

    def on_input_submitted(self, event: Input.Submitted) -> None:
        value = event.value.strip()
        if not value:
            return
        if self._current_tab == "import":
            # Capture optional title from second input
            title = ""
            try:
                title = self.query_one("#kb-import-title", Input).value.strip()
            except Exception:
                pass
            self.dismiss(("import", {"source": value, "title": title}))
        elif self._current_tab == "paste":
            try:
                title = self.query_one("#kb-paste-title", Input).value.strip()
                content = self.query_one("#kb-paste-content", TextArea).text.strip()
            except Exception:
                return
            if not title:
                self.notify("請輸入標題", severity="warning")
                return
            if not content:
                self.notify("請貼上內文", severity="warning")
                return
            self.dismiss(("import_text", {"title": title, "content": content}))
        elif self._current_tab == "search":
            self.dismiss(("search", value))

    def _parse_bulk_input(self, text: str) -> tuple[list[dict], dict]:
        """Parse bulk input. Returns (items, batch_meta).
        items: list of dicts with 'src' key and optional metadata.
        batch_meta: top-level metadata from wrapper JSON (project, description, etc.)
        """
        import json as _json
        stripped = text.strip()
        batch_meta: dict = {}
        # Try JSON if starts with [ or {
        if stripped.startswith("[") or stripped.startswith("{"):
            try:
                data = _json.loads(stripped)
                # Wrapper object: {"project": ..., "items": [...]}
                if isinstance(data, dict):
                    if "items" in data:
                        batch_meta = {k: v for k, v in data.items() if k != "items"}
                        data = data["items"]
                    else:
                        # Single item object
                        data = [data]
                if not isinstance(data, list):
                    raise ValueError("JSON 必須是陣列或含 items 的物件")
                items = []
                for entry in data:
                    if isinstance(entry, str):
                        items.append({"src": entry.strip()})
                    elif isinstance(entry, dict):
                        src = entry.get("url") or entry.get("path") or ""
                        if not src:
                            continue
                        meta = {"src": src.strip()}
                        if entry.get("title"):
                            meta["title"] = entry["title"]
                        if entry.get("tags"):
                            meta["tags"] = entry["tags"]
                        if entry.get("layer"):
                            meta["layer"] = entry["layer"]
                        # Collect note from note/author/source/layer
                        note_parts = []
                        if entry.get("note"):
                            note_parts.append(entry["note"])
                        for key in ("author", "source", "layer"):
                            if entry.get(key):
                                note_parts.append(f"{key}: {entry[key]}")
                        if note_parts:
                            meta["note"] = "\n".join(note_parts)
                        items.append(meta)
                if not items:
                    self.notify("JSON 中未找到有效來源", severity="error")
                return items, batch_meta
            except (_json.JSONDecodeError, ValueError) as e:
                self.notify(f"JSON 解析失敗：{e}", severity="error")
                return [], {}
        # Plain text: one source per line
        return [{"src": line.strip()} for line in text.split("\n") if line.strip()], {}

    @work(exclusive=True)
    async def _show_bulk_preview(self, items: list[dict], batch_meta: dict) -> None:
        """Stage 2: show parsed items with metadata for review before importing."""
        try:
            content = self.query_one("#kb-content", VerticalScroll)
        except Exception:
            return
        await content.remove_children()

        # Batch metadata header
        if batch_meta:
            header_parts = []
            if batch_meta.get("project"):
                header_parts.append(f"[bold #fafafa]{batch_meta['project']}[/]")
            if batch_meta.get("description"):
                desc = batch_meta["description"]
                if len(desc) > 100:
                    desc = desc[:100] + "…"
                header_parts.append(f"[#8b949e]{desc}[/]")
            if header_parts:
                await content.mount(Static("\n".join(header_parts)))
                await content.mount(Static("[dim #2a2a2a]" + "─" * 70 + "[/]", classes="kb-sep"))

        await content.mount(Static(f"[#6e7681]{len(items)} 筆來源[/]"))
        await content.mount(Static("[dim #2a2a2a]" + "─" * 70 + "[/]", classes="kb-sep"))

        # Item list with metadata
        for i, item in enumerate(items):
            src = item["src"]
            title = item.get("title", "")
            label = title if title else src
            if len(label) > 55:
                label = label[:55] + "…"

            parts = [f"[#c9d1d9]{i+1}. {label}[/]"]
            if title:
                short_src = src if len(src) < 50 else src[:50] + "…"
                parts.append(f"   [dim #484f58]{short_src}[/]")
            meta_bits = []
            if item.get("layer"):
                meta_bits.append(f"[#7dd3fc]{item['layer']}[/]")
            if item.get("tags"):
                tags_str = ", ".join(item["tags"][:4])
                if len(item["tags"]) > 4:
                    tags_str += " …"
                meta_bits.append(f"[#484f58]{tags_str}[/]")
            if meta_bits:
                parts.append("   " + "  ".join(meta_bits))
            await content.mount(Static("\n".join(parts), classes="kb-bulk-item"))

        await content.mount(Static("[dim #2a2a2a]" + "─" * 70 + "[/]", classes="kb-sep"))
        await content.mount(Button(f"確認匯入 {len(items)} 筆", id="kb-bulk-confirm", classes="kb-action-btn"))

    @work(exclusive=True)
    async def _run_bulk(self, items: list[dict]) -> None:
        try:
            content = self.query_one("#kb-content", VerticalScroll)
        except Exception:
            return
        await content.remove_children()

        await content.mount(Static("", id="kb-bulk-status"))
        self._bulk_items = []
        for item in items:
            title = item.get("title", "")
            src = item["src"]
            label = title if title else src
            if len(label) > 60:
                label = label[:60] + "…"
            widget = Static(f"○  {label}", classes="kb-bulk-item")
            await content.mount(widget)
            self._bulk_items.append(widget)

        spin_frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
        self._spin_idx = 0
        self._spin_label = ""
        self._spin_widget = None

        def _spin() -> None:
            if self._spin_widget is not None:
                ch = spin_frames[self._spin_idx % len(spin_frames)]
                self._spin_widget.update(f"{ch} {self._spin_label}")
                self._spin_idx += 1

        timer = self.set_interval(0.1, _spin)

        done_count = 0
        for i, item in enumerate(items):
            src = item["src"]
            title = item.get("title", "")
            label = title if title else src
            if len(label) > 60:
                label = label[:60] + "…"
            self._spin_label = label
            self._spin_widget = self._bulk_items[i]
            self._spin_idx = 0
            self._bulk_items[i].add_class("-running")
            self._update_bulk_status(done_count, len(items))
            try:
                meta = await self.app._import_one(src, self._project_id, title=item.get("title", ""))
                doc_id = meta.get("doc_id")
                if doc_id and any(k in item for k in ("title", "tags", "note")):
                    from conversation.store import ConversationStore
                    from paths import project_root
                    store = ConversationStore(db_path=project_root(self._project_id) / "conversations.db")
                    await store.update_document_meta(
                        doc_id,
                        title=item.get("title"),
                        tags=item.get("tags"),
                        note=item.get("note"),
                    )
                self._bulk_items[i].update(f"✅ {label}")
                self._bulk_items[i].remove_class("-running")
                self._bulk_items[i].add_class("-done")
                done_count += 1
            except Exception as e:
                self._bulk_items[i].update(f"❌ {label}  [#ff6b6b]{str(e)[:30]}[/]")
                self._bulk_items[i].remove_class("-running")
                self._bulk_items[i].add_class("-error")
            self._update_bulk_status(done_count, len(items))

        self._spin_widget = None
        timer.stop()

        self._bulk_done = True
        if items:
            self.app.state.last_imported_source = items[-1]["src"]
        self._update_bulk_status(done_count, len(items), finished=True)
        try:
            await content.mount(Button("重新匯入", id="kb-bulk-reset", classes="kb-action-btn"))
        except Exception:
            pass

    def _update_bulk_status(self, done: int, total: int, finished: bool = False) -> None:
        try:
            status = self.query_one("#kb-bulk-status", Static)
            if finished:
                status.update(f"[#7dd3fc]全部完成 {done}/{total}[/]")
            else:
                status.update(f"[#8b949e]匯入中 {done}/{total}[/]")
        except Exception:
            pass

    @work(exclusive=True)
    async def _delete_doc(self, doc_id: str) -> None:
        from conversation.store import ConversationStore
        from paths import project_root
        store = ConversationStore(db_path=project_root(self._project_id) / "conversations.db")
        await store.delete_document(doc_id)
        self._docs = [d for d in self._docs if d["id"] != doc_id]
        self.notify("已刪除文獻記錄")
        self._show_tab("docs")

    def action_close(self) -> None:
        self.dismiss(None)


# ── OnboardingScreen ───────────────────────────────────────────────


class OnboardingScreen(ModalScreen[str | None]):
    """首次啟動三步驟引導：Chat Provider → Embedding → 完成。"""

    CSS = """
    OnboardingScreen { align: center middle; }
    #ob-box {
        width: 62; height: auto; max-height: 85%; padding: 1 2;
        border: round #3a3a3a; background: #0a0a0a;
        border-title-color: #d4a27a;
    }
    #ob-scroll { height: auto; max-height: 100%; }
    .ob-sep { height: 1; margin: 0; color: #2a2a2a; }
    .ob-title { height: 1; margin: 0; padding: 0; color: #d4a27a; text-style: bold; }
    .ob-hint { height: auto; margin: 0; padding: 0 1; color: #484f58; }
    .ob-prov-btn {
        background: transparent; color: #8b949e;
        border: none; height: 1; margin: 0; padding: 0 2;
        min-width: 0; width: 100%;
        text-align: left; content-align: left middle;
    }
    .ob-prov-btn:hover { background: #1a1a1a; color: #fafafa; }
    #ob-key-input {
        margin: 0; background: #111111; color: #fafafa;
        border: tall #3a3a3a; padding: 0 1;
    }
    #ob-key-input:focus { border: tall #fafafa 40%; }
    #ob-model-search {
        margin: 0; background: #111111; color: #fafafa;
        border: tall #3a3a3a; padding: 0 1;
    }
    #ob-model-search:focus { border: tall #fafafa 40%; }
    #ob-model-scroll { height: 16; max-height: 45%; }
    #ob-key-section { height: auto; }
    .ob-btn-row { height: auto; margin: 1 0 0 0; }
    .ob-btn-row Button { margin: 0 1 0 0; }
    #btn-ob-save {
        background: #fafafa; color: #0a0a0a; border: none;
        margin: 0; min-width: 16;
    }
    #btn-ob-skip {
        background: #2a2a2a; color: #8b949e; border: none;
        margin: 0; min-width: 12;
    }
    #btn-ob-next {
        background: #fafafa; color: #0a0a0a; border: none;
        margin: 0; min-width: 16;
    }
    #btn-ob-back {
        background: #2a2a2a; color: #8b949e; border: none;
        margin: 0; min-width: 12;
    }
    #ob-download-status { height: auto; margin: 0; padding: 0 1; color: #6e7681; }
    #ob-progress-bar { margin: 0 2; height: 1; }
    #ob-progress-bar Bar { color: #d4a27a; }
    """

    def __init__(self) -> None:
        super().__init__()
        self._step = "chat_provider"
        self._selected_chat_pid: str | None = None
        self._selected_embed_pid: str | None = None
        self._selected_rag_pid: str | None = None
        self._selected_vision_pid: str | None = None
        self._model_search_query: str = ""
        self._model_candidates: list[str] = []
        self._model_button_kind: str | None = None

    def compose(self) -> ComposeResult:
        box = Vertical(id="ob-box")
        box.border_title = "De-insight v0.7 Setup"
        with box:
            with VerticalScroll(id="ob-scroll"):
                yield Vertical(id="ob-content")

    async def on_mount(self) -> None:
        await self._render_step()

    async def _render_step(self) -> None:
        container = self.query_one("#ob-content", Vertical)
        await container.remove_children()

        if self._step == "chat_provider":
            await self._render_chat_provider(container)
        elif self._step == "chat_setup":
            await self._render_chat_setup(container)
        elif self._step == "embed":
            await self._render_embed(container)
        elif self._step == "embed_setup":
            await self._render_embed_setup(container)
        elif self._step == "embed_download":
            await self._render_embed_download(container)
        elif self._step == "rag_llm":
            await self._render_rag_llm(container)
        elif self._step == "rag_setup":
            await self._render_rag_setup(container)
        elif self._step == "vision_provider":
            await self._render_vision_provider(container)
        elif self._step == "vision_setup":
            await self._render_vision_setup(container)
        elif self._step == "done":
            await self._render_done(container)

    def _filter_models(self, models: list[str]) -> list[str]:
        q = self._model_search_query.strip().lower()
        if not q:
            return models
        return [m for m in models if q in m.lower()]

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id == "ob-model-search":
            self._model_search_query = event.value
            self.run_worker(self._refresh_model_buttons(), exclusive=True)

    async def _refresh_model_buttons(self) -> None:
        """Refresh only the model button list to avoid full-screen flicker."""
        try:
            scroll = self.query_one("#ob-model-scroll", VerticalScroll)
        except Exception:
            return

        await scroll.remove_children()
        models = self._filter_models(self._model_candidates)
        kind = self._model_button_kind
        for idx, model in enumerate(models):
            if kind == "chat":
                btn_id = f"ob-model-{idx}"
            elif kind == "rag":
                btn_id = (
                    "ob-ragmodel-"
                    + model.replace("/", "_").replace(":", "-").replace(".", "_")
                )
            elif kind == "vision":
                btn_id = f"ob-vmodel-{idx}"
            else:
                continue

            await scroll.mount(
                Button(
                    f"  {model}",
                    id=btn_id,
                    classes="ob-prov-btn",
                    name=model,
                )
            )

    async def _render_chat_provider(self, container: Vertical) -> None:
        from providers import CHAT_PROVIDERS
        await container.mount(
            Static("  Step 1/5: Chat Provider", classes="ob-title"),
            Static("[dim #2a2a2a]" + "-" * 56 + "[/]", classes="ob-sep"),
            Static("  選擇聊天用的 LLM 服務", classes="ob-hint"),
        )
        for pid, pinfo in CHAT_PROVIDERS.items():
            await container.mount(
                Button(
                    f"  {pinfo['label']}",
                    id=f"ob-chat-{pid}",
                    classes="ob-prov-btn",
                    name=pid,
                )
            )

    async def _render_chat_setup(self, container: Vertical) -> None:
        from providers import CHAT_PROVIDERS
        from config.service import get_config_service
        from model_registry import resolve_dynamic_models
        pinfo = CHAT_PROVIDERS.get(self._selected_chat_pid, {})
        auth_type = pinfo.get("auth_type", "api_key")

        await container.mount(
            Static(f"  Step 1/5: {pinfo.get('label', '')} Setup", classes="ob-title"),
            Static("[dim #2a2a2a]" + "-" * 56 + "[/]", classes="ob-sep"),
        )

        # A3: auth_type in ("none", "oauth") -> hide key input
        if auth_type not in ("none", "oauth"):
            key_env = pinfo.get("key_env", "")
            await container.mount(
                Static(f"  請輸入 {key_env}", classes="ob-hint"),
                Vertical(
                    Input(placeholder="API Key", id="ob-key-input", password=True),
                    id="ob-key-section",
                ),
            )
        else:
            if auth_type == "oauth":
                label = "OAuth 登入（稍後在設定中完成）"
            else:
                label = "無需 API Key"
            await container.mount(Static(f"  {label}", classes="ob-hint"))

        # model selection
        models = await resolve_dynamic_models(
            provider_id=self._selected_chat_pid or "",
            service="chat",
            fallback=list(pinfo.get("models", [])),
            env=get_config_service().snapshot(include_process=True),
        )
        self._model_candidates = list(models)
        self._model_button_kind = "chat"
        models = self._filter_models(self._model_candidates)
        if models:
            await container.mount(
                Static("[dim #2a2a2a]" + "-" * 56 + "[/]", classes="ob-sep"),
                Static("  選擇模型", classes="ob-hint"),
                Input(
                    value=self._model_search_query,
                    placeholder="搜尋模型...",
                    id="ob-model-search",
                ),
            )
            scroll = VerticalScroll(id="ob-model-scroll")
            await container.mount(scroll)
            for idx, model in enumerate(models):
                await scroll.mount(
                    Button(
                        f"  {model}",
                        id=f"ob-model-{idx}",
                        classes="ob-prov-btn",
                        name=model,
                    )
                )
        else:
            await container.mount(
                Horizontal(
                    Button("下一步 ->", id="btn-ob-next"),
                    Button("<- 返回", id="btn-ob-back"),
                    classes="ob-btn-row",
                ),
            )

    async def _render_embed(self, container: Vertical) -> None:
        from providers import EMBED_PROVIDERS
        await container.mount(
            Static("  Step 2/5: Embedding Model", classes="ob-title"),
            Static("[dim #2a2a2a]" + "-" * 56 + "[/]", classes="ob-sep"),
            Static("  選擇文字向量化的方式", classes="ob-hint"),
            Static("", classes="ob-sep"),
        )
        for pid, pinfo in EMBED_PROVIDERS.items():
            await container.mount(
                Button(
                    f"  {pinfo['label']}",
                    id=f"ob-embed-provider-{pid}",
                    classes="ob-prov-btn",
                    name=pid,
                )
            )
        await container.mount(
            Horizontal(Button("<- 返回", id="btn-ob-back"), classes="ob-btn-row"),
        )

    async def _render_embed_setup(self, container: Vertical) -> None:
        """Jina API Key 輸入步驟。"""
        from settings import load_env
        existing_key = load_env().get("JINA_API_KEY", "")
        hint = "  已有 JINA_API_KEY（可直接下一步，或輸入新的）" if existing_key else "  請輸入 Jina API Key（到 https://jina.ai/ 免費取得）"
        await container.mount(
            Static("  Step 2/5: Jina Embedding API", classes="ob-title"),
            Static("[dim #2a2a2a]" + "-" * 56 + "[/]", classes="ob-sep"),
            Static(hint, classes="ob-hint"),
            Vertical(
                Input(
                    placeholder="jina_xxxx（留空則沿用現有）" if existing_key else "jina_xxxx",
                    id="ob-key-input",
                    password=True,
                ),
                id="ob-key-section",
            ),
            Horizontal(
                Button("<- 返回", id="btn-ob-back"),
                Button("儲存並繼續 ->", id="btn-ob-jina-save"),
                classes="ob-btn-row",
            ),
        )

    _DL_PHASES = [
        "正在檢查環境...",
        "正在編譯 llama-server（Metal ON）...",
        "正在下載 GGUF 模型（Q4_K_M）...",
        "正在下載 mmproj-f16...",
        "即將完成，請稍候...",
    ]

    async def _render_embed_download(self, container: Vertical) -> None:
        self._dl_phase_idx = 0
        await container.mount(
            Static("  Step 2/5: 安裝 GGUF Embedding 環境", classes="ob-title"),
            Static("[dim #2a2a2a]" + "-" * 56 + "[/]", classes="ob-sep"),
            ProgressBar(total=None, id="ob-progress-bar"),
            Static(f"  {self._DL_PHASES[0]}", id="ob-download-status"),
        )
        self._dl_timer = self.set_interval(15.0, self._tick_dl_phase)
        self._run_model_download()

    def _tick_dl_phase(self) -> None:
        self._dl_phase_idx = min(
            self._dl_phase_idx + 1, len(self._DL_PHASES) - 1
        )
        try:
            status = self.query_one("#ob-download-status", Static)
            status.update(f"  {self._DL_PHASES[self._dl_phase_idx]}")
        except Exception:
            pass

    @work(exclusive=True, thread=True)
    def _run_model_download(self) -> None:
        import traceback
        from embeddings.local import ensure_model_downloaded
        try:
            def _progress(desc: str, pct: float) -> None:
                try:
                    # 更新 phase 文字
                    idx = min(int(pct * len(self._DL_PHASES)), len(self._DL_PHASES) - 1)
                    self._dl_phase_idx = idx
                except Exception:
                    pass

            ensure_model_downloaded(download_if_missing=True, progress_callback=_progress)
            self.app.call_from_thread(self._on_download_complete)
        except Exception as exc:
            self.app.log.error("GGUF install failed:\n%s", traceback.format_exc())
            msg = str(exc)
            self.app.call_from_thread(self._on_download_error, msg)

    def _on_download_complete(self) -> None:
        if hasattr(self, "_dl_timer"):
            self._dl_timer.stop()
        self._step = "rag_llm"
        self.run_worker(self._render_step(), exclusive=True)

    def _on_download_error(self, msg: str) -> None:
        if hasattr(self, "_dl_timer"):
            self._dl_timer.stop()
        try:
            status = self.query_one("#ob-download-status", Static)
            status.update(f"  [red]下載失敗：{msg}[/]")
        except Exception:
            pass

    # ── Step 3: RAG LLM ──

    async def _render_rag_llm(self, container: Vertical) -> None:
        from providers import RAG_LLM_PROVIDERS
        await container.mount(
            Static("  Step 3/5: 知識庫建圖 LLM", classes="ob-title"),
            Static("[dim #2a2a2a]" + "-" * 56 + "[/]", classes="ob-sep"),
            Static("  匯入文件時需要 LLM 抽取知識圖譜", classes="ob-hint"),
            Static("  建議選低成本 API（如 Google AI Studio 免費額度充足）", classes="ob-hint"),
            Static("", classes="ob-sep"),
        )
        for pid, pinfo in RAG_LLM_PROVIDERS.items():
            if pid == "ollama" or pid == "ollama-local":
                continue
            await container.mount(
                Button(
                    f"  {pinfo['label']}",
                    id=f"ob-rag-{pid}",
                    classes="ob-prov-btn",
                    name=pid,
                ),
            )

    async def _render_rag_setup(self, container: Vertical) -> None:
        from providers import RAG_LLM_PROVIDERS
        from config.service import get_config_service
        from model_registry import resolve_dynamic_models
        pinfo = RAG_LLM_PROVIDERS.get(self._selected_rag_pid, {})
        auth_type = pinfo.get("auth_type", "api_key")

        await container.mount(
            Static(f"  Step 3/5: {pinfo.get('label', '')} RAG Setup", classes="ob-title"),
            Static("[dim #2a2a2a]" + "-" * 56 + "[/]", classes="ob-sep"),
        )

        key_env = pinfo.get("key_env", "")
        if auth_type == "api_key" and key_env:
            from settings import load_env
            existing_key = load_env().get(key_env, "")
            hint = f"  已有 {key_env}（可直接選模型，或輸入新的 Key）" if existing_key else f"  請輸入 {key_env}"
            await container.mount(
                Static(hint, classes="ob-hint"),
                Vertical(
                    Input(
                        placeholder="API Key（留空則沿用現有）" if existing_key else "API Key",
                        id="ob-key-input",
                        password=True,
                    ),
                    id="ob-key-section",
                ),
            )

        models = await resolve_dynamic_models(
            provider_id=self._selected_rag_pid or "",
            service="rag_llm",
            fallback=list(pinfo.get("models", [])),
            env=get_config_service().snapshot(include_process=True),
        )
        self._model_candidates = list(models)
        self._model_button_kind = "rag"
        models = self._filter_models(self._model_candidates)
        if models:
            await container.mount(
                Static("  選擇模型：", classes="ob-hint"),
                Input(
                    value=self._model_search_query,
                    placeholder="搜尋模型...",
                    id="ob-model-search",
                ),
            )
            scroll = VerticalScroll(id="ob-model-scroll")
            await container.mount(scroll)
            for m in models:
                await scroll.mount(
                    Button(
                        f"  {m}",
                        id=f"ob-ragmodel-{m.replace('/', '_').replace(':', '-').replace('.', '_')}",
                        classes="ob-prov-btn",
                        name=m,
                    ),
                )
        await container.mount(
            Horizontal(Button("<- 返回", id="btn-ob-back"), classes="ob-btn-row"),
        )

    async def _save_rag_config(self, model_name: str) -> None:
        from providers import RAG_LLM_PROVIDERS
        from settings import save_env_key

        pid = self._selected_rag_pid
        if not pid:
            return
        pinfo = RAG_LLM_PROVIDERS.get(pid, {})

        key_env = pinfo.get("key_env", "")
        if pinfo.get("auth_type") == "api_key" and key_env:
            # Always try to read the input field; use new value if provided
            try:
                key_input = self.query_one("#ob-key-input", Input)
                key_val = key_input.value.strip()
                if key_val:
                    save_env_key(key_env, key_val)
            except Exception:
                pass

        prefix = pinfo.get("model_prefix", "")
        save_env_key("RAG_LLM_MODEL", prefix + model_name)

        if pinfo.get("default_base"):
            save_env_key("RAG_API_BASE", pinfo["default_base"])

        # Save RAG_API_KEY = same key as the provider key
        if key_env:
            from settings import load_env
            key_val = load_env().get(key_env, "")
            if key_val:
                save_env_key("RAG_API_KEY", key_val)

    # ── Step 4: Vision Model ──

    async def _render_vision_provider(self, container: Vertical) -> None:
        from providers import VISION_PROVIDERS
        await container.mount(
            Static("  Step 4/5: 圖片描述模型 (Vision)", classes="ob-title"),
            Static("[dim #2a2a2a]" + "-" * 56 + "[/]", classes="ob-sep"),
            Static("  圖片匯入時自動生成描述，需要支援 Vision 的模型", classes="ob-hint"),
            Static("", classes="ob-sep"),
        )
        for pid, pinfo in VISION_PROVIDERS.items():
            await container.mount(
                Button(
                    f"  {pinfo['label']}",
                    id=f"ob-vision-{pid}",
                    classes="ob-prov-btn",
                    name=pid,
                ),
            )

    async def _render_vision_setup(self, container: Vertical) -> None:
        from providers import VISION_PROVIDERS
        from config.service import get_config_service
        from model_registry import resolve_dynamic_models
        pinfo = VISION_PROVIDERS.get(self._selected_vision_pid, {})
        auth_type = pinfo.get("auth_type", "api_key")

        await container.mount(
            Static(f"  Step 4/5: {pinfo.get('label', '')} Vision Setup", classes="ob-title"),
            Static("[dim #2a2a2a]" + "-" * 56 + "[/]", classes="ob-sep"),
        )

        # Check if key already exists from chat provider setup
        key_env = pinfo.get("key_env", "")
        if auth_type == "api_key" and key_env:
            from settings import load_env
            existing_key = load_env().get(key_env, "")
            hint = f"  已有 {key_env}（可直接選模型，或輸入新的 Key）" if existing_key else f"  請輸入 {key_env}"
            await container.mount(
                Static(hint, classes="ob-hint"),
                Vertical(
                    Input(
                        placeholder="API Key（留空則沿用現有）" if existing_key else "API Key",
                        id="ob-key-input",
                        password=True,
                    ),
                    id="ob-key-section",
                ),
            )

        # Model selection
        models = await resolve_dynamic_models(
            provider_id=self._selected_vision_pid or "",
            service="vision",
            fallback=list(pinfo.get("models", [])),
            env=get_config_service().snapshot(include_process=True),
        )
        self._model_candidates = list(models)
        self._model_button_kind = "vision"
        models = self._filter_models(self._model_candidates)
        if models:
            await container.mount(
                Static("[dim #2a2a2a]" + "-" * 56 + "[/]", classes="ob-sep"),
                Static("  選擇 Vision 模型", classes="ob-hint"),
                Input(
                    value=self._model_search_query,
                    placeholder="搜尋模型...",
                    id="ob-model-search",
                ),
            )
            scroll = VerticalScroll(id="ob-model-scroll")
            await container.mount(scroll)
            for idx, model in enumerate(models):
                await scroll.mount(
                    Button(
                        f"  {model}",
                        id=f"ob-vmodel-{idx}",
                        classes="ob-prov-btn",
                        name=model,
                    ),
                )
        await container.mount(
            Horizontal(
                Button("<- 返回", id="btn-ob-back"),
                classes="ob-btn-row",
            ),
        )

    async def _render_done(self, container: Vertical) -> None:
        from settings import load_env
        env = load_env()
        model = env.get("LLM_MODEL", "(未設定)")
        embed = "jina-embeddings-v4 GGUF (Q4_K_M, 1024d)"
        rag_llm = env.get("RAG_LLM_MODEL", "")
        rag_display = rag_llm if rag_llm else "跟聊天模型相同"
        vision = env.get("VISION_MODEL", "")
        vision_display = vision if vision else "(未設定)"

        await container.mount(
            Static("  Step 5/5: 設定完成!", classes="ob-title"),
            Static("[dim #2a2a2a]" + "-" * 56 + "[/]", classes="ob-sep"),
            Static(f"  Chat: {model}", classes="ob-hint"),
            Static(f"  Embedding: {embed}", classes="ob-hint"),
            Static(f"  知識庫 LLM: {rag_display}", classes="ob-hint"),
            Static(f"  Vision: {vision_display}", classes="ob-hint"),
            Static("", classes="ob-sep"),
            Static("  可隨時用 Ctrl+S 開啟設定修改", classes="ob-hint"),
            Horizontal(
                Button("開始使用 ->", id="btn-ob-finish"),
                classes="ob-btn-row",
            ),
        )

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        btn_id = event.button.id or ""
        btn_name = event.button.name or ""

        # Step 1: chat provider selection
        if btn_id.startswith("ob-chat-"):
            self._selected_chat_pid = btn_name
            self._model_search_query = ""
            self._model_candidates = []
            self._model_button_kind = None
            self._step = "chat_setup"
            await self._render_step()
            return

        # Step 1 setup: model selection
        if btn_id.startswith("ob-model-"):
            await self._save_chat_config(btn_name)
            self._step = "embed"
            await self._render_step()
            return

        # Step 1 setup: next (for providers without model list)
        if btn_id == "btn-ob-next":
            await self._save_chat_config(None)
            self._step = "embed"
            await self._render_step()
            return

        # Step 3: RAG LLM provider selection
        if btn_id.startswith("ob-rag-"):
            pid = btn_name
            if pid == "same-as-chat":
                from settings import save_env_key
                save_env_key("RAG_LLM_MODEL", "")
                self._step = "vision_provider"
                await self._render_step()
                return
            self._selected_rag_pid = pid
            self._model_search_query = ""
            self._model_candidates = []
            self._model_button_kind = None
            self._step = "rag_setup"
            await self._render_step()
            return

        # Step 4: Vision provider selection
        if btn_id.startswith("ob-vision-"):
            pid = btn_name
            if pid == "skip-vision":
                self._step = "done"
                await self._render_step()
                return
            self._selected_vision_pid = pid
            self._model_search_query = ""
            self._model_candidates = []
            self._model_button_kind = None
            self._step = "vision_setup"
            await self._render_step()
            return

        # Step 4: Vision model selection
        if btn_id.startswith("ob-vmodel-"):
            await self._save_vision_config(btn_name)
            self._step = "done"
            await self._render_step()
            return

        # Step 3: RAG model selection
        if btn_id.startswith("ob-ragmodel-"):
            await self._save_rag_config(btn_name)
            self._step = "vision_provider"
            await self._render_step()
            return

        # Back button
        if btn_id == "btn-ob-back":
            if self._step == "chat_setup":
                self._model_search_query = ""
                self._step = "chat_provider"
            elif self._step == "embed":
                self._step = "chat_provider"
            elif self._step == "embed_setup":
                self._step = "embed"
            elif self._step == "rag_llm":
                self._step = "embed"
            elif self._step == "rag_setup":
                self._model_search_query = ""
                self._step = "rag_llm"
            elif self._step == "vision_provider":
                self._step = "rag_llm"
            elif self._step == "vision_setup":
                self._model_search_query = ""
                self._step = "vision_provider"
            await self._render_step()
            return

        # Step 2: embed provider selection
        if btn_id.startswith("ob-embed-provider-"):
            embed_pid = btn_name
            if embed_pid == "jina-api":
                self._selected_embed_pid = "jina-api"
                self._step = "embed_setup"
                await self._render_step()
                return
            elif embed_pid == "gguf":
                from settings import save_env_key
                save_env_key("EMBED_PROVIDER", "gguf")
                save_env_key("EMBED_MODEL", "jina-embeddings-v4-gguf")
                save_env_key("EMBED_DIM", "1024")
                save_env_key("GGUF_AUTO_INSTALL", "1")
                self._step = "embed_download"
                await self._render_step()
                return

        # Step 2: Jina API key save
        if btn_id == "btn-ob-jina-save":
            await self._save_jina_config()
            self._step = "rag_llm"
            await self._render_step()
            return

        # Step 2: embed — download or skip (GGUF path)
        if btn_id == "ob-embed-download":
            from settings import save_env_key
            save_env_key("EMBED_PROVIDER", "gguf")
            save_env_key("EMBED_MODEL", "jina-embeddings-v4-gguf")
            save_env_key("EMBED_DIM", "1024")
            save_env_key("GGUF_AUTO_INSTALL", "1")
            self._step = "embed_download"
            await self._render_step()
            return

        if btn_id == "ob-embed-skip":
            from settings import save_env_key
            save_env_key("EMBED_PROVIDER", "gguf")
            save_env_key("EMBED_MODEL", "jina-embeddings-v4-gguf")
            save_env_key("EMBED_DIM", "1024")
            save_env_key("GGUF_AUTO_INSTALL", "1")
            self._step = "rag_llm"
            await self._render_step()
            return

        # Finish
        if btn_id == "btn-ob-finish":
            self.dismiss("done")
            return

    async def _save_chat_config(self, model_name: str | None) -> None:
        from providers import CHAT_PROVIDERS
        from settings import save_env_key

        pid = self._selected_chat_pid
        if not pid:
            return

        pinfo = CHAT_PROVIDERS.get(pid, {})

        # Save API key if provided
        if pinfo.get("auth_type") not in ("none", "oauth"):
            try:
                key_input = self.query_one("#ob-key-input", Input)
                key_val = key_input.value.strip()
                if key_val and pinfo.get("key_env"):
                    save_env_key(pinfo["key_env"], key_val)
            except Exception:
                pass

        # Save base URL if provider has default
        if pinfo.get("default_base"):
            base_env = pinfo.get("base_env", "OPENAI_API_BASE")
            if base_env:
                save_env_key(base_env, pinfo["default_base"])

        # Save model
        prefix = pinfo.get("model_prefix", "")
        if model_name:
            full_model = prefix + model_name
            save_env_key("LLM_MODEL", full_model)
        elif prefix:
            # No model list (e.g. codex-cli/) — use prefix + default
            default = "codex-mini-latest" if "codex-cli" in pid else "default"
            save_env_key("LLM_MODEL", prefix + default)

    async def _save_vision_config(self, model_name: str) -> None:
        from providers import VISION_PROVIDERS
        from settings import save_env_key

        pid = self._selected_vision_pid
        if not pid:
            return
        pinfo = VISION_PROVIDERS.get(pid, {})

        # Save API key if provided (new value overrides existing)
        key_env = pinfo.get("key_env", "")
        if pinfo.get("auth_type") == "api_key" and key_env:
            try:
                key_input = self.query_one("#ob-key-input", Input)
                key_val = key_input.value.strip()
                if key_val:
                    save_env_key(key_env, key_val)
            except Exception:
                pass

        # Save VISION_MODEL with prefix
        prefix = pinfo.get("model_prefix", "")
        save_env_key("VISION_MODEL", prefix + model_name)

        # Save VISION_API_BASE if provider has a default
        if pinfo.get("default_base"):
            save_env_key("VISION_API_BASE", pinfo["default_base"])

    async def _save_embed_config(self, embed_pid: str) -> None:
        # v0.7: 統一 jina-embeddings-v4，此方法僅作向後相容
        from settings import save_env_key
        save_env_key("EMBED_PROVIDER", "jina")
        save_env_key("EMBED_MODEL", "jina-embeddings-v3")
        save_env_key("EMBED_DIM", "1024")

    async def _save_jina_config(self) -> None:
        """儲存 Jina API 設定。"""
        from settings import save_env_key
        try:
            key_input = self.query_one("#ob-key-input", Input)
            key_val = key_input.value.strip()
            if key_val:
                save_env_key("JINA_API_KEY", key_val)
        except Exception:
            pass
        save_env_key("EMBED_PROVIDER", "jina")
        save_env_key("EMBED_MODEL", "jina-embeddings-v3")
        save_env_key("EMBED_DIM", "1024")
