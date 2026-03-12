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



class MemoryManageModal(ModalScreen[str | None]):
    """記憶/知識資料管理 — 分類、主題、時間，有資料感。"""

    BINDINGS = [("escape", "close", "關閉")]

    CSS = """
    MemoryManageModal { align: center middle; }
    #mem-manage-box {
        width: 72; height: auto; max-height: 85%; padding: 1 2;
        border: round #3a3a3a; background: #0a0a0a;
        border-title-color: #fafafa;
    }
    /* stats bar */
    #mem-stats {
        height: auto; margin: 0 0 1 0; padding: 0;
        color: #484f58;
    }
    /* filter row */
    #mem-filter-row {
        height: 1; margin: 0 0 1 0;
    }
    .filter-btn {
        background: transparent; color: #484f58;
        border: none; height: 1; min-width: 0;
        margin: 0 1 0 0; padding: 0 1;
    }
    .filter-btn:hover { color: #fafafa; }
    .filter-btn.-active { color: #fafafa; background: #1a1a1a; }
    /* category filter row */
    #mem-category-row {
        height: 1; margin: 0 0 1 0;
    }
    .cat-btn {
        background: transparent; color: #484f58;
        border: none; height: 1; min-width: 0;
        margin: 0 1 0 0; padding: 0 1;
    }
    .cat-btn:hover { color: #fafafa; }
    .cat-btn.-active { color: #c9d1d9; background: #1a1a1a; }
    /* separator */
    .mm-sep {
        height: 1; margin: 0; color: #2a2a2a;
    }
    /* memory list */
    #mem-scroll {
        height: auto; max-height: 100%;
    }
    .mem-entry {
        height: auto; margin: 0 0 0 0; padding: 0 1;
        color: #8b949e;
    }
    .mem-entry:hover {
        color: #fafafa; background: #111111;
    }
    .mem-entry-time {
        width: 12; height: 1; color: #484f58;
        margin: 0; padding: 0;
    }
    .mem-entry-topic {
        width: auto; height: 1; color: #6e7681;
        margin: 0 1 0 0; padding: 0;
    }
    .mem-entry-text {
        width: 1fr; height: 1; color: #8b949e;
        margin: 0; padding: 0;
    }
    .mem-del-btn {
        background: transparent; color: #484f58;
        border: none; height: 1; min-width: 3;
        margin: 0; padding: 0;
    }
    .mem-del-btn:hover { color: #ff6b6b; }
    /* time group header */
    .time-header {
        height: 1; margin: 1 0 0 0; padding: 0;
        color: #6e7681;
    }
    """

    def __init__(self, project_id: str | None = None) -> None:
        super().__init__()
        self._project_id = project_id
        self._filter_type: str = ""  # "" = all
        self._filter_category: str = ""  # "" = all
        self._all_mems: list[dict] = []
        self._current_page: int = 0
        self._page_size: int = 50
        self._total_count: int = 0

    def compose(self) -> ComposeResult:
        box = Vertical(id="mem-manage-box")
        box.border_title = "◇ 記憶 / 知識庫"
        with box:
            yield Static("", id="mem-stats")
            with Horizontal(id="mem-filter-row"):
                yield Button("全部", id="flt-all", classes="filter-btn -active", name="")
                yield Button("💡洞見", id="flt-insight", classes="filter-btn", name="insight")
                yield Button("❓問題", id="flt-question", classes="filter-btn", name="question")
                yield Button("💭反應", id="flt-reaction", classes="filter-btn", name="reaction")
            with Horizontal(id="mem-category-row"):
                yield Button("#全部", id="cat-all", classes="cat-btn -active", name="")
                yield Button("#思考方式", id="cat-thinking", classes="cat-btn", name="思考方式")
                yield Button("#美學偏好", id="cat-aesthetic", classes="cat-btn", name="美學偏好")
                yield Button("#創作問題", id="cat-creation", classes="cat-btn", name="創作問題")
                yield Button("#理論連結", id="cat-theory", classes="cat-btn", name="理論連結")
            yield Static("[dim #2a2a2a]" + "─" * 66 + "[/]", classes="mm-sep")
            yield VerticalScroll(id="mem-scroll")
            with Horizontal(id="mem-pagination"):
                yield Button("← 上一頁", id="btn-prev-page", classes="filter-btn")
                yield Static("", id="page-info")
                yield Button("下一頁 →", id="btn-next-page", classes="filter-btn")
            yield Button("← 回到對話", classes="back-btn")

    def on_mount(self) -> None:
        self._load_all()

    @work(exclusive=True)
    async def _load_all(self) -> None:
        _db_path, _ = _get_project_paths(self.app)
        stats = await get_memory_stats(db_path=_db_path)
        total = stats["total"]
        by_type = stats["by_type"]
        self._total_count = total
        stats_text = f"[#6e7681]{total} 條記憶[/]"
        for t, cnt in by_type.items():
            icons = {"insight": "💡", "question": "❓", "reaction": "💭"}
            stats_text += f"  {icons.get(t, '◇')}{cnt}"
        try:
            self.query_one("#mem-stats", Static).update(stats_text)
        except Exception:
            pass

        self._all_mems = await get_memories(limit=1000, db_path=_db_path)
        self._current_page = 0
        await self._render_list()

    async def _render_list(self) -> None:
        try:
            scroll = self.query_one("#mem-scroll", VerticalScroll)
        except Exception:
            return
        await scroll.remove_children()

        mems = self._all_mems
        if self._filter_type:
            mems = [m for m in mems if m["type"] == self._filter_type]
        if self._filter_category:
            mems = [m for m in mems if m.get("category") == self._filter_category]

        if not mems:
            await scroll.mount(Static("[dim #484f58]沒有符合的記憶[/]"))
            self._update_pagination_info(0, 0)
            return

        # Calculate pagination
        total_pages = (len(mems) + self._page_size - 1) // self._page_size
        self._current_page = min(self._current_page, max(0, total_pages - 1))
        start_idx = self._current_page * self._page_size
        end_idx = start_idx + self._page_size
        page_mems = mems[start_idx:end_idx]

        icons = {"insight": "💡", "question": "❓", "reaction": "💭"}
        current_date = ""
        for m in page_mems:
            date_str = (m.get("created_at") or "")[:10]
            if date_str != current_date:
                current_date = date_str
                await scroll.mount(
                    Static(f"[#6e7681]── {date_str or '未知日期'} ──[/]", classes="time-header")
                )
            icon = icons.get(m["type"], "◇")
            time_str = (m.get("created_at") or "")[11:16]
            topic = m.get("topic", "")
            content = m["content"][:36] + ("…" if len(m["content"]) > 36 else "")
            row = _MemEntry(m)
            await scroll.mount(row)
            topic_display = f"[#6e7681]#{topic}[/] " if topic else ""
            await row.mount(Static(f"[#484f58]{time_str}[/]", classes="mem-entry-time"))
            await row.mount(Static(f"{icon} {topic_display}{content}",
                                   classes="mem-entry-text"))
            await row.mount(Button("✗", classes="mem-del-btn", name=str(m["id"])))

        # Update pagination info
        self._update_pagination_info(self._current_page + 1, total_pages)

    def _update_pagination_info(self, current_page: int, total_pages: int) -> None:
        """Update pagination display."""
        try:
            page_info = self.query_one("#page-info", Static)
            if total_pages > 1:
                page_info.update(f"[#6e7681]第 {current_page} / {total_pages} 頁[/]")
            else:
                page_info.update("")
        except Exception:
            pass

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.has_class("back-btn"):
            self.dismiss(None)
            return
        bid = event.button.id or ""
        name = event.button.name or ""

        # Pagination buttons
        if bid == "btn-prev-page":
            if self._current_page > 0:
                self._current_page -= 1
                self._apply_filter()
            return

        if bid == "btn-next-page":
            # Calculate total pages
            mems = self._all_mems
            if self._filter_type:
                mems = [m for m in mems if m["type"] == self._filter_type]
            if self._filter_category:
                mems = [m for m in mems if m.get("category") == self._filter_category]
            total_pages = (len(mems) + self._page_size - 1) // self._page_size
            if self._current_page < total_pages - 1:
                self._current_page += 1
                self._apply_filter()
            return

        if bid.startswith("flt-"):
            self._filter_type = name
            self._current_page = 0  # Reset to first page on filter change
            for btn in self.query(".filter-btn"):
                btn.remove_class("-active")
            event.button.add_class("-active")
            self._apply_filter()
            return

        if event.button.has_class("cat-btn"):
            self._filter_category = name
            self._current_page = 0  # Reset to first page on filter change
            for btn in self.query(".cat-btn"):
                btn.remove_class("-active")
            event.button.add_class("-active")
            self._apply_filter()
            return

        if event.button.has_class("mem-del-btn"):
            mem_id = int(name or "0")
            if mem_id:
                self._delete_and_reload(mem_id)

    @work(exclusive=True)
    async def _apply_filter(self) -> None:
        await self._render_list()

    @work(exclusive=True)
    async def _delete_and_reload(self, mem_id: int) -> None:
        _db_path, _ = _get_project_paths(self.app)
        await delete_memory(mem_id, db_path=_db_path)
        self._all_mems = [m for m in self._all_mems if m["id"] != mem_id]
        await self._render_list()
        self.notify("已刪除")

    def action_close(self) -> None:
        self.dismiss(None)


class _MemEntry(Horizontal):
    """記憶列表中可點擊的單條記錄。"""

    def __init__(self, mem: dict, **kwargs) -> None:
        super().__init__(classes="mem-entry", **kwargs)
        self._mem = mem

    def on_click(self) -> None:
        def on_result(result: str | None) -> None:
            if result and (result.startswith("discuss:") or result.startswith("__insert__:")):
                modal = self.screen
                if isinstance(modal, MemoryManageModal):
                    modal.dismiss(result)
        self.app.push_screen(MemoryDetailModal(self._mem), callback=on_result)


# ── InsightConfirmModal ──────────────────────────────────────────
