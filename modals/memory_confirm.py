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
