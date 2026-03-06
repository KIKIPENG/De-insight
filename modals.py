"""v0.3 Modals — ProjectModal, MemoryConfirmModal (style matches panels.py)."""

from textual import work
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Label, Static


# ── ProjectModal ────────────────────────────────────────────────────

class ProjectModal(ModalScreen):
    """專案管理 — 風格與 MemoryManageModal 統一。"""

    BINDINGS = [("escape", "close", "關閉")]

    CSS = """
    ProjectModal { align: center middle; }
    #project-box {
        width: 60; height: auto; max-height: 85%; padding: 1 2;
        border: round #3a3a3a; background: #0a0a0a;
        border-title-color: #fafafa;
    }
    #project-scroll {
        height: auto; max-height: 60%;
    }
    .proj-entry {
        height: auto; padding: 0 1; color: #8b949e;
    }
    .proj-entry:hover {
        color: #fafafa; background: #111111;
    }
    .proj-entry.-active {
        color: #7dd3fc; background: #111111;
    }
    .proj-sep {
        height: 1; margin: 0; color: #2a2a2a;
    }
    #project-new-input {
        margin: 1 0; background: #111111; color: #fafafa;
        border: tall #3a3a3a; padding: 0 1;
    }
    .proj-actions { height: 1; margin: 1 0 0 0; }
    .proj-btn {
        background: transparent; color: #484f58;
        border: none; height: 1; min-width: 0;
        margin: 0 1 0 0; padding: 0 1;
    }
    .proj-btn:hover { color: #fafafa; }
    .proj-del-btn {
        background: transparent; color: #484f58;
        border: none; height: 1; min-width: 0;
        margin: 0; padding: 0;
    }
    .proj-del-btn:hover { color: #ff6b6b; }
    """

    def __init__(self, projects: list[dict], current_id: str | None = None) -> None:
        super().__init__()
        self._projects = projects
        self._current_id = current_id

    def compose(self) -> ComposeResult:
        box = Vertical(id="project-box")
        box.border_title = "◇ 專案管理"
        with box:
            if self._projects:
                yield Static(f"[#6e7681]{len(self._projects)} 個專案[/]")
                yield Static("[dim #2a2a2a]" + "─" * 54 + "[/]", classes="proj-sep")
                with VerticalScroll(id="project-scroll"):
                    for p in self._projects:
                        is_active = p["id"] == self._current_id if self._current_id else False
                        marker = "●" if is_active else " "
                        last = p.get("last_active", "")
                        entry = Static(
                            f"[#7dd3fc]{marker}[/] {p['name']}  [dim #484f58]{last}[/]",
                            classes="proj-entry" + (" -active" if is_active else ""),
                            name=p["id"],
                        )
                        yield entry
                yield Static("[dim #2a2a2a]" + "─" * 54 + "[/]", classes="proj-sep")
            else:
                yield Static("[dim #484f58]尚無專案，在下方輸入名稱新增[/]")
            yield Input(placeholder="新專案名稱…", id="project-new-input")
            with Horizontal(classes="proj-actions"):
                yield Button("新增", id="proj-create", classes="proj-btn")
            yield Button("← 回到對話", classes="back-btn")

    def on_mount(self) -> None:
        try:
            self.query_one("#project-new-input", Input).focus()
        except Exception:
            pass

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id == "project-new-input":
            name = event.value.strip()
            if name:
                self.dismiss(("create", name))

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "proj-create":
            name = self.query_one("#project-new-input", Input).value.strip()
            if name:
                self.dismiss(("create", name))
        elif event.button.has_class("back-btn"):
            self.dismiss(None)

    def on_static_click(self, event: Static.Click) -> None:
        """Click on a project entry to switch."""
        widget = event.widget if hasattr(event, 'widget') else event.static
        if not hasattr(widget, 'name') or not widget.name:
            return
        pid = widget.name
        for p in self._projects:
            if p["id"] == pid:
                self.dismiss(("switch", p))
                return

    def on_click(self, event) -> None:
        # Find if clicked on a proj-entry Static
        widget = event.widget if hasattr(event, 'widget') else None
        if widget and isinstance(widget, Static) and widget.has_class("proj-entry"):
            pid = widget.name
            if pid:
                for p in self._projects:
                    if p["id"] == pid:
                        self.dismiss(("switch", p))
                        return

    def action_close(self) -> None:
        self.dismiss(None)


# ── MemoryConfirmModal ──────────────────────────────────────────────

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
