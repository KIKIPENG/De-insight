"""Generic delete confirmation modal."""

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Static


class DeleteConfirmModal(ModalScreen):
    """通用刪除確認對話框。"""

    CSS = """
    DeleteConfirmModal { align: center middle; }
    #delete-box {
        width: 50; height: auto; padding: 1 2;
        border: round #3a3a3a; background: #0a0a0a;
        border-title-color: #e06c75;
    }
    .delete-btn {
        margin: 0 1 0 0;
        background: transparent;
        color: #8b949e;
        border: none;
        height: 1;
        min-width: 10;
    }
    .delete-btn:hover { color: #fafafa; }
    .delete-btn.-danger { color: #e06c75; }
    """

    def __init__(self, message: str = "確定要刪除?") -> None:
        super().__init__()
        self.message = message

    def compose(self) -> ComposeResult:
        box = Vertical(id="delete-box")
        box.border_title = "◇ 確認刪除"
        with box:
            yield Static(f"[#8b949e]{self.message}[/]")
            yield Static("")
            with Horizontal():
                yield Button("是，刪除", id="del-yes", classes="delete-btn -danger")
                yield Button("取消", id="del-no", classes="delete-btn")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "del-yes":
            self.dismiss(True)
        else:
            self.dismiss(False)
