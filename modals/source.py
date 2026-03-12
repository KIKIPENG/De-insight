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



class SourceModal(ModalScreen):
    """知識庫來源檢視。"""

    BINDINGS = [("escape", "close", "關閉")]

    CSS = """
    SourceModal { align: center middle; }
    #source-box {
        width: 72; height: auto; max-height: 85%; padding: 1 2;
        border: round #3a3a3a; background: #0a0a0a;
        border-title-color: #7dd3fc;
    }
    .source-item { height: auto; margin: 1 0; }
    .source-title { color: #7dd3fc; }
    .source-snippet {
        color: #8b949e; padding: 0 0 0 2;
        height: auto;
        border-left: solid #2a2a2a;
    }
    .source-meta { color: #484f58; }
    .src-sep { height: 1; color: #2a2a2a; }
    .src-cite-btn {
        background: transparent; color: #484f58;
        border: none; height: 1; min-width: 0; padding: 0;
    }
    .src-cite-btn:hover { color: #7dd3fc; }
    """

    def __init__(self, sources: list[dict]) -> None:
        super().__init__()
        self._sources = sources

    def compose(self) -> ComposeResult:
        box = Vertical(id="source-box")
        box.border_title = "◇ 知識庫出處"
        with box:
            yield Static(f"[#6e7681]{len(self._sources)} 個來源[/]")
            yield Static("[dim #2a2a2a]" + "─" * 66 + "[/]", classes="src-sep")
            with VerticalScroll():
                for i, s in enumerate(self._sources):
                    with Vertical(classes="source-item"):
                        yield Static(
                            f"[{i+1}] {s.get('title', '未知來源')}",
                            classes="source-title",
                        )
                        if s.get("snippet"):
                            yield Static(s["snippet"], classes="source-snippet")
                        if s.get("file"):
                            yield Static(f"來源：{s['file']}", classes="source-meta")
                        yield Button("▸ 引用到對話", classes="src-cite-btn", name=str(i))
                    if i < len(self._sources) - 1:
                        yield Static("[dim #2a2a2a]" + "─" * 66 + "[/]", classes="src-sep")
            yield Static("[dim #2a2a2a]" + "─" * 66 + "[/]", classes="src-sep")
            yield Button("← 回到對話", classes="back-btn")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.has_class("back-btn"):
            self.dismiss(None)
            return
        if event.button.has_class("src-cite-btn"):
            idx = int(event.button.name)
            s = self._sources[idx]
            snippet = s.get("snippet", "")[:300]
            title = s.get("title", "未知來源")
            formatted = f"\u300c{snippet}\u300d\u2014 {title}"
            self.dismiss(f"__cite__:{formatted}")

    def action_close(self) -> None:
        self.dismiss(None)


# ── DocReaderModal ───────────────────────────────────────────────

