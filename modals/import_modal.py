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


# ── FocusImportModal ──────────────────────────────────────────────
