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



class RelationModal(ModalScreen):
    """記憶關聯 — 顯示雙向向量相似的記憶對。"""

    BINDINGS = [("escape", "close", "關閉")]

    CSS = """
    RelationModal { align: center middle; }
    #rel-box {
        width: 72; height: auto; max-height: 85%; padding: 1 2;
        border: round #3a3a3a; background: #0a0a0a;
        border-title-color: #fafafa;
    }
    #rel-scroll { height: auto; max-height: 70%; }
    .rel-pair { height: auto; padding: 0 1; margin: 0 0 1 0; color: #8b949e; }
    .rel-pair.-highlight { border-left: thick #7dd3fc; }
    .rel-sep { height: 1; color: #2a2a2a; }
    """

    def compose(self) -> ComposeResult:
        box = Vertical(id="rel-box")
        box.border_title = "◇ 記憶關聯"
        with box:
            yield Static("", id="rel-count")
            yield Static("[dim #2a2a2a]" + "─" * 66 + "[/]", classes="rel-sep")
            yield VerticalScroll(id="rel-scroll")
            yield Static("[dim #2a2a2a]" + "─" * 66 + "[/]", classes="rel-sep")
            yield Button("← 回到對話", classes="back-btn")

    def on_mount(self) -> None:
        self._find_relations()

    @work(exclusive=True)
    async def _find_relations(self) -> None:
        try:
            scroll = self.query_one("#rel-scroll", VerticalScroll)
        except Exception:
            return
        await scroll.remove_children()
        await scroll.mount(Static("[dim #484f58]搜尋中…[/]"))

        try:
            from memory.store import get_memories
            from memory.vectorstore import search_similar, has_index
            _db_path, _lancedb_dir = _get_project_paths(self.app)
            if not has_index(lancedb_dir=_lancedb_dir):
                await scroll.remove_children()
                await scroll.mount(Static("[dim #484f58]向量索引為空[/]"))
                return

            mems = await get_memories(limit=100, db_path=_db_path)
            if len(mems) < 2:
                await scroll.remove_children()
                await scroll.mount(Static("[dim #484f58]記憶不足，需要至少 2 條[/]"))
                return

            # Build similarity map: for each memory, find top-6 similar
            sim_map: dict[int, list[tuple[int, float]]] = {}
            for m in mems:
                results = await search_similar(m["content"], limit=7, lancedb_dir=_lancedb_dir)
                sim_map[m["id"]] = [
                    (r["id"], r.get("score", 0))
                    for r in results if r["id"] != m["id"]
                ][:6]

            # Find bidirectional pairs with score >= 0.55
            pairs = []
            seen_pairs = set()
            mem_by_id = {m["id"]: m for m in mems}
            for a_id, a_sims in sim_map.items():
                for b_id, score_ab in a_sims:
                    if score_ab < 0.55:
                        continue
                    pair_key = (min(a_id, b_id), max(a_id, b_id))
                    if pair_key in seen_pairs:
                        continue
                    # Check bidirectional
                    b_sims = sim_map.get(b_id, [])
                    score_ba = next((s for bid, s in b_sims if bid == a_id), 0)
                    if score_ba >= 0.55:
                        avg_score = (score_ab + score_ba) / 2
                        seen_pairs.add(pair_key)
                        pairs.append((a_id, b_id, avg_score))

            pairs.sort(key=lambda x: x[2], reverse=True)

            await scroll.remove_children()
            if not pairs:
                await scroll.mount(Static("[dim #484f58]未找到足夠相似的記憶對[/]"))
                self.query_one("#rel-count", Static).update("[#6e7681]0 組關聯[/]")
                return

            self.query_one("#rel-count", Static).update(f"[#6e7681]{len(pairs)} 組關聯[/]")
            icons = {"insight": "💡", "question": "❓", "reaction": "💭"}
            for a_id, b_id, score in pairs[:20]:
                a = mem_by_id.get(a_id, {})
                b = mem_by_id.get(b_id, {})
                pct = int(score * 100)
                highlight = " -highlight" if pct >= 80 else ""
                icon_a = icons.get(a.get("type", ""), "◇")
                icon_b = icons.get(b.get("type", ""), "◇")
                text = (
                    f"{icon_a} {a.get('content', '?')[:40]}\n"
                    f"  ↔  {icon_b} {b.get('content', '?')[:40]}\n"
                    f"  [#6e7681]相似度 {pct}%[/]"
                )
                await scroll.mount(Static(text, classes=f"rel-pair{highlight}"))
        except Exception as e:
            await scroll.remove_children()
            await scroll.mount(Static(f"[#ff6b6b]載入失敗: {e}[/]"))

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.has_class("back-btn"):
            self.dismiss(None)

    def action_close(self) -> None:
        self.dismiss(None)


# ── ImportModal ──────────────────────────────────────────────────
