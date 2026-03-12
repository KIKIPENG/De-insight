"""Exit confirmation modal."""

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Static


class ExitConfirmModal(ModalScreen):
    """確認離開對話框。"""

    CSS = """
    ExitConfirmModal { align: center middle; }
    #exit-box {
        width: 50; height: auto; padding: 1 2;
        border: round #3a3a3a; background: #0a0a0a;
        border-title-color: #d4a27a;
    }
    .exit-btn {
        margin: 0 1 0 0;
        background: transparent;
        color: #8b949e;
        border: none;
        height: 1;
        min-width: 10;
    }
    .exit-btn:hover { color: #fafafa; }
    .exit-btn.-danger { color: #e06c75; }
    """

    def compose(self) -> ComposeResult:
        box = Vertical(id="exit-box")
        box.border_title = "◇ 確定要離開?"
        with box:
            yield Static("[#8b949e]確定要離開嗎？未儲存的對話將會保留。[/]")
            yield Static("")
            with Horizontal():
                yield Button("是，離開", id="exit-yes", classes="exit-btn -danger")
                yield Button("取消", id="exit-no", classes="exit-btn")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "exit-yes":
            self.dismiss(True)
        else:
            self.dismiss(False)
