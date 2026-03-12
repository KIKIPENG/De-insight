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


# ── RelationModal ─────────────────────────────────────────────────
