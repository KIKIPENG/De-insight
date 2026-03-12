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
