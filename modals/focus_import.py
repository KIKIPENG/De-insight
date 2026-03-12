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



class FocusImportModal(ModalScreen[str | None]):
    """輸入 Markdown 檔案路徑並匯入問題意識。"""

    CSS = """
    FocusImportModal { align: center middle; }
    #focus-import-container {
        width: 60; height: auto;
        background: #111111; border: round #2a2a2a; padding: 2 3;
    }
    #focus-import-title { color: #d4a27a; margin-bottom: 1; }
    #focus-import-input { margin-bottom: 1; }
    #focus-import-buttons { height: auto; }
    """

    BINDINGS = [("escape", "dismiss(None)", "取消")]

    def compose(self) -> ComposeResult:
        with Vertical(id="focus-import-container"):
            yield Static("匯入問題意識", id="focus-import-title")
            yield Input(
                placeholder="輸入 .md 檔案的完整路徑",
                id="focus-import-input",
            )
            with Horizontal(id="focus-import-buttons"):
                yield Button("匯入", id="btn-do-import", variant="primary")
                yield Button("取消", id="btn-cancel-import", variant="default")

    def on_mount(self) -> None:
        try:
            self.query_one("#focus-import-input", Input).focus()
        except Exception:
            pass

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-cancel-import":
            self.dismiss(None)
        elif event.button.id == "btn-do-import":
            path = self.query_one("#focus-import-input", Input).value.strip()
            self.dismiss(path if path else None)

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id == "focus-import-input":
            path = event.value.strip()
            self.dismiss(path if path else None)


# ── SearchModal ──────────────────────────────────────────────────
