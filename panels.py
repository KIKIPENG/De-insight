"""De-insight v0.2 — 右側面板 Widget"""

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import Button, Static


# ── Right Panel Widgets ──────────────────────────────────────────────


class MemoryItem(Static):
    """單條記憶顯示，可點擊閱覽完整內容。"""

    def __init__(self, mem: dict, **kwargs) -> None:
        self.mem = mem
        icons = {
            "insight": "💡", "question": "❓", "reaction": "💭",
            "evolution": "🔄", "contradiction": "⚡",
        }
        icon = icons.get(mem["type"], "◇")
        label = {
            "insight": "洞見", "question": "問題", "reaction": "反應",
            "evolution": "演變", "contradiction": "矛盾",
        }.get(mem["type"], mem["type"])
        content = mem["content"]
        if len(content) > 30:
            content = content[:30] + "…"
        extra_classes = "memory-item"
        if mem["type"] in ("evolution", "contradiction"):
            extra_classes += f" -{mem['type']}"
        super().__init__(
            f"{icon} [{label}] {content}",
            classes=extra_classes,
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


class KnowledgeActionButton(Button):
    """知識庫面板操作按鈕。"""

    def __init__(self, label: str, action: str, **kwargs) -> None:
        super().__init__(label, **kwargs)
        self._action = action

    def on_button_pressed(self, event: Button.Pressed) -> None:
        event.stop()
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


class ResearchPanel(Vertical):
    """右上：知識庫面板，含文獻列表 + 操作 + 查詢結果。"""

    def compose(self) -> ComposeResult:
        yield Horizontal(
            KnowledgeActionButton("匯入", "import_document", classes="kb-action"),
            KnowledgeActionButton("搜尋", "search_knowledge", classes="kb-action"),
            KnowledgeActionButton("批量", "bulk_import", classes="kb-action"),
            KnowledgeActionButton("文獻", "manage_documents", classes="kb-action"),
            classes="kb-actions-row",
        )
        yield Static("[dim #484f58]─[/]" * 30, classes="kb-divider")
        yield Static("[dim #6e7681]文獻[/]", classes="kb-section-label")
        with VerticalScroll(id="kb-doc-scroll"):
            yield Vertical(id="kb-doc-list")
        yield Static("[dim #6e7681]研究結果[/]", classes="kb-section-label")
        with VerticalScroll(id="research-result-scroll"):
            yield Static("[dim #484f58]尚未檢索[/]", id="research-content")
        yield ResearchCiteLink("引用到對話", id="research-cite", disabled=True)


class ResearchCiteLink(Button):
    """知識庫引用按鈕，將 RAG 摘要注入聊天。"""

    def on_button_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        method = getattr(self.app, "action_cite_research", None)
        if method:
            method()


class MemoryPanel(VerticalScroll):
    """右下：記憶面板。"""

    def compose(self) -> ComposeResult:
        yield Static(
            "[dim #484f58]對話後自動記錄洞見[/]",
            id="memory-content",
        )
