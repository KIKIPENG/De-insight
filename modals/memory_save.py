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



class MemorySaveModal(ModalScreen[dict | None]):
    """快速儲存記憶點 Modal。"""

    BINDINGS = [("escape", "cancel", "取消")]

    CSS = """
    MemorySaveModal { align: center middle; }
    #mem-box {
        width: 60; height: auto; max-height: 80%; padding: 1 2;
        border: round #3a3a3a; background: #0a0a0a;
        border-title-color: #fafafa;
    }
    #mem-type-row { height: 1; margin: 1 0 0 0; }
    .mtype-btn {
        background: transparent; color: #484f58;
        border: none; height: 1; min-width: 0;
        margin: 0 1 0 0; padding: 0 1;
    }
    .mtype-btn:hover { color: #fafafa; }
    .mtype-btn.-selected { color: #fafafa; background: #1a1a1a; }
    #mem-editor {
        height: 6; margin: 1 0;
        background: #111111; color: #fafafa;
        border: tall #3a3a3a;
    }
    #mem-editor:focus { border: tall #666666; }
    #mem-topic-input {
        margin: 0; background: #111111; color: #fafafa;
        border: tall #3a3a3a; padding: 0 1;
    }
    #mem-topic-input:focus { border: tall #666666; }
    #mem-save-row { height: 1; margin: 1 0 0 0; }
    #btn-mem-save {
        background: #fafafa; color: #0a0a0a; border: none;
        min-width: 12; margin: 0 1 0 0;
    }
    #btn-mem-cancel {
        background: transparent; color: #484f58; border: none;
        min-width: 8; margin: 0;
    }
    """

    def __init__(self, content: str, mem_type: str = "thought") -> None:
        super().__init__()
        self._content = content
        self._type = mem_type

    def compose(self) -> ComposeResult:
        box = Vertical(id="mem-box")
        box.border_title = "◇ 儲存記憶"
        with box:
            yield Static("[#8b949e]選擇要記住的內容，可以編輯：[/]")
            with Horizontal(id="mem-type-row"):
                for t, label in [("thought", "想法"), ("insight", "洞見"), ("question", "問題"), ("reaction", "反應"), ("reference", "參考")]:
                    cls = "mtype-btn" + (" -selected" if t == self._type else "")
                    yield Button(f"[{label}]", id=f"mtype-{t}", classes=cls, name=t)
            yield TextArea(self._content, id="mem-editor")
            yield Static("[dim #484f58]主題標籤（選填）：[/]")
            yield Input(placeholder="例：字體設計、工具", id="mem-topic-input")
            with Horizontal(id="mem-save-row"):
                yield Button("儲存記憶", id="btn-mem-save")
                yield Button("[取消]", id="btn-mem-cancel")
                yield Button("← 回到對話", classes="back-btn")

    def on_mount(self) -> None:
        self.query_one("#mem-editor", TextArea).focus()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.has_class("back-btn"):
            self.dismiss(None)
            return
        bid = event.button.id or ""
        if bid.startswith("mtype-"):
            self._type = event.button.name or "thought"
            for btn in self.query(".mtype-btn"):
                btn.remove_class("-selected")
            event.button.add_class("-selected")
        elif bid == "btn-mem-save":
            content = self.query_one("#mem-editor", TextArea).text.strip()
            topic = self.query_one("#mem-topic-input", Input).value.strip()
            if content:
                self.dismiss({"type": self._type, "content": content, "topic": topic})
            else:
                self.notify("內容不能為空")
        elif bid == "btn-mem-cancel":
            self.dismiss(None)

    def action_cancel(self) -> None:
        self.dismiss(None)


# ── KnowledgeModal ───────────────────────────────────────────────
