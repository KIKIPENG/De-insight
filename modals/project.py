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
        from paths import GLOBAL_PROJECT_ID
        box = Vertical(id="project-box")
        box.border_title = "◇ 專案管理"
        with box:
            # 全局文獻庫固定在最上方
            global_proj = next((p for p in self._projects if p.get("id") == GLOBAL_PROJECT_ID or p.get("is_global")), None)
            regular_projects = [p for p in self._projects if p.get("id") != GLOBAL_PROJECT_ID and not p.get("is_global")]

            if global_proj:
                is_active = global_proj["id"] == self._current_id if self._current_id else False
                marker = "●" if is_active else "◆"
                yield Static(
                    f"[#f59e0b]{marker}[/] [#f59e0b]{global_proj['name']}[/]  [dim #484f58]跨專案基礎文獻[/]",
                    classes="proj-entry" + (" -active" if is_active else ""),
                    name=global_proj["id"],
                )
                yield Static("[dim #2a2a2a]" + "─" * 54 + "[/]", classes="proj-sep")

            if regular_projects:
                yield Static(f"[#6e7681]{len(regular_projects)} 個專案[/]")
                yield Static("[dim #2a2a2a]" + "─" * 54 + "[/]", classes="proj-sep")
                with VerticalScroll(id="project-scroll"):
                    for p in regular_projects:
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
            elif not global_proj:
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
