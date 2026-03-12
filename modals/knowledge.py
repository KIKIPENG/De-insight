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



class KnowledgeModal(ModalScreen):
    """知識庫管理 — 匯入 / 貼上 / 搜尋 / 文獻管理。"""

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
    .kb-action-btn {
        background: transparent; color: #484f58;
        border: none; height: 1; min-width: 0;
        margin: 1 0 0 0; padding: 0 1;
    }
    .kb-action-btn:hover { color: #fafafa; }
    """

    _TAB_NAMES = ["匯入", "貼上", "搜尋", "文獻"]
    _TAB_IDS = ["import", "paste", "search", "docs"]

    def __init__(self, project_id: str = "default", initial_tab: str = "import") -> None:
        super().__init__()
        self._project_id = project_id
        self._initial_tab = initial_tab
        self._current_tab = initial_tab
        self._docs: list[dict] = []

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
        elif self._current_tab == "docs":
            await self._render_docs(content)

    async def _render_import(self, container: VerticalScroll) -> None:
        await container.mount(Static("[#8b949e]輸入檔案路徑或網址（拖放僅在此頁生效）[/]"))
        await container.mount(
            Input(placeholder="/path/to/file.pdf|txt|md  或  https://...  或  10.1234/doi", id="kb-input")
        )
        await container.mount(
            Input(placeholder="文獻標題（必填）", id="kb-import-title")
        )
        await container.mount(
            Static("[dim #484f58]支援：PDF/TXT/MD 路徑、URL、arXiv 連結、DOI（10.xxxx/...）[/]")
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
                note = (doc.get("note") or "").strip().lower()
                if "warning:" in note:
                    title = f"⚠ {title}"
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
        if self._current_tab == "import":
            source = ""
            if event.input.id == "kb-input":
                source = value
            else:
                try:
                    source = self.query_one("#kb-input", Input).value.strip()
                except Exception:
                    source = ""
            title = ""
            try:
                title = self.query_one("#kb-import-title", Input).value.strip()
            except Exception:
                pass
            if not source:
                self.notify("請輸入來源（檔案路徑 / URL / DOI）", severity="warning")
                return
            if not title:
                self.notify("請輸入文獻標題（必填）", severity="warning")
                return
            self.dismiss(("import", {"source": source, "title": title}))
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
        elif self._current_tab == "search" and value:
            self.dismiss(("search", value))

    def on_paste(self, event) -> None:
        """只允許在匯入頁使用拖放/貼上來源。"""
        if self._current_tab != "import":
            return
        raw = (event.text or "").strip()
        if not raw:
            return
        source = self._normalize_import_source(raw)
        if not source:
            return
        event.prevent_default()
        try:
            self.query_one("#kb-input", Input).value = source
            self.query_one("#kb-import-title", Input).focus()
        except Exception:
            pass

    def _normalize_import_source(self, text: str) -> str:
        from pathlib import Path
        from urllib.parse import unquote, urlparse
        t = text.strip().strip("'\"")
        if not t:
            return ""
        if t.startswith("http://") or t.startswith("https://"):
            return t
        if t.startswith("10."):
            return t
        if t.startswith("file://"):
            t = unquote(urlparse(t).path)
        if "%20" in t:
            t = unquote(t)
        t = t.replace("\\ ", " ")
        p = Path(t)
        if not p.exists() or not p.is_file():
            return ""
        if not t.lower().endswith((".pdf", ".txt", ".md")):
            self.notify("僅支援 PDF、TXT、MD 檔案", severity="warning")
            return ""
        return str(p)

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

