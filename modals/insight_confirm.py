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



class InsightConfirmModal(ModalScreen[dict | None]):
    """LLM 整理後的洞見確認 Modal。使用者可編輯後儲存或取消。"""

    BINDINGS = [("escape", "cancel", "取消")]

    CSS = """
    InsightConfirmModal { align: center middle; }
    #insight-box {
        width: 60; height: auto; max-height: 80%; padding: 1 2;
        border: round #3a3a3a; background: #0a0a0a;
        border-title-color: #fafafa;
    }
    #insight-type-row {
        height: 1; margin: 1 0 0 0;
    }
    .type-btn {
        background: transparent; color: #484f58;
        border: none; height: 1; min-width: 0;
        margin: 0 1 0 0; padding: 0 1;
    }
    .type-btn:hover { color: #fafafa; }
    .type-btn.-selected { color: #fafafa; background: #1a1a1a; }
    #insight-editor {
        height: 6; margin: 1 0;
        background: #111111; color: #fafafa;
        border: tall #3a3a3a;
    }
    #insight-editor:focus { border: tall #666666; }
    #insight-save-row {
        height: 1; margin: 1 0 0 0;
    }
    #btn-insight-save {
        background: #fafafa; color: #0a0a0a; border: none;
        min-width: 12; margin: 0 1 0 0;
    }
    #btn-insight-cancel {
        background: transparent; color: #484f58; border: none;
        min-width: 8; margin: 0;
    }
    """

    def __init__(self, draft: str, insight_type: str = "insight") -> None:
        super().__init__()
        self._draft = draft
        self._type = insight_type

    def compose(self) -> ComposeResult:
        box = Vertical(id="insight-box")
        box.border_title = "◇ 確認洞見"
        with box:
            yield Static("[#8b949e]LLM 整理了以下洞見，你可以編輯後儲存：[/]")
            with Horizontal(id="insight-type-row"):
                for t, label in [("insight", "洞見"), ("question", "問題"), ("reaction", "反應")]:
                    cls = "type-btn" + (" -selected" if t == self._type else "")
                    yield Button(f"[{label}]", id=f"itype-{t}", classes=cls, name=t)
            yield TextArea(self._draft, id="insight-editor")
            with Horizontal(id="insight-save-row"):
                yield Button("儲存", id="btn-insight-save")
                yield Button("[取消]", id="btn-insight-cancel")
                yield Button("← 回到對話", classes="back-btn")

    def on_mount(self) -> None:
        self.query_one("#insight-editor", TextArea).focus()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.has_class("back-btn"):
            self.dismiss(None)
            return
        bid = event.button.id or ""
        if bid.startswith("itype-"):
            self._type = event.button.name or "insight"
            for btn in self.query(".type-btn"):
                btn.remove_class("-selected")
            event.button.add_class("-selected")
        elif bid == "btn-insight-save":
            content = self.query_one("#insight-editor", TextArea).text.strip()
            if content:
                self.dismiss({"type": self._type, "content": content})
            else:
                self.notify("內容不能為空")
        elif bid == "btn-insight-cancel":
            self.dismiss(None)

    def action_cancel(self) -> None:
        self.dismiss(None)


# ── MemorySaveModal ──────────────────────────────────────────────
