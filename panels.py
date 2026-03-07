"""De-insight v0.2 — 右側面板 Widget"""

from textual import work
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import Button, Input, Static

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
        from modals import MemoryDetailModal

        def on_result(result: str | None) -> None:
            if result and result.startswith("discuss:"):
                content = result.removeprefix("discuss:")
                self.app._start_discussion_from_memory(content)
            elif result and result.startswith("__insert__:"):
                content = result[len("__insert__:"):]
                self.app.fill_input(content)

        self.app.push_screen(MemoryDetailModal(self.mem), callback=on_result)


class KnowledgeActionLink(Static):
    """知識庫面板內的可點擊操作。"""

    def __init__(self, label: str, action: str, **kwargs) -> None:
        super().__init__(label, **kwargs)
        self._action = action

    def on_click(self) -> None:
        method = getattr(self.app, f"action_{self._action}", None)
        if method:
            method()


class DocumentItem(Static):
    """知識庫文獻項目，可點擊管理。"""

    def __init__(self, doc: dict, **kwargs) -> None:
        self.doc = doc
        title = doc.get("title", "未知文獻")
        if len(title) > 28:
            title = title[:28] + "…"
        source_type = doc.get("source_type", "")
        icon = {"pdf": "📄", "url": "🔗", "doi": "📑", "arxiv": "📐"}.get(source_type, "📄")
        super().__init__(f"{icon} {title}", classes="doc-item", **kwargs)

    def on_click(self) -> None:
        self.app.action_manage_documents()


class ResearchPanel(VerticalScroll):
    """右上：知識庫面板，含文獻列表 + 操作 + 查詢結果。"""

    def compose(self) -> ComposeResult:
        yield Horizontal(
            KnowledgeActionLink("[#6e7681]匯入[/]", "import_document", classes="kb-action"),
            KnowledgeActionLink("[#6e7681]搜尋[/]", "search_knowledge", classes="kb-action"),
            KnowledgeActionLink("[#6e7681]批量[/]", "bulk_import", classes="kb-action"),
            KnowledgeActionLink("[#6e7681]文獻[/]", "manage_documents", classes="kb-action"),
            classes="kb-actions-row",
        )
        yield Static("", id="kb-status-line")
        yield Static("[dim #484f58]─[/]" * 30, classes="kb-divider")
        yield Vertical(id="kb-doc-list")
        yield Static("", id="research-content")


class MemoryPanel(VerticalScroll):
    """右下：記憶面板。"""

    def compose(self) -> ComposeResult:
        yield Static(
            "[dim #484f58]對話後自動記錄洞見[/]",
            id="memory-content",
        )
