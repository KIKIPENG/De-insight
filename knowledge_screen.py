"""Knowledge 全屏管理介面 — 全局文獻 + 專案文獻兩層架構。

功能：
- 全局/專案分頁，各自獨立的文獻列表
- 文獻卡片：標題、來源類型、匯入時間、大小
- 匯入：PDF / URL / DOI / 貼上文字
- 文獻重命名
- 匯入進度 ETA
- 知識圖譜統計（entities / relations / chunks）
"""

from __future__ import annotations

import json as _json
import logging
import time
from pathlib import Path

from textual import work
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Static, TextArea

from paths import GLOBAL_PROJECT_ID, project_root

log = logging.getLogger(__name__)


# ── 工具函式 ──────────────────────────────────────────────────────

def _format_size(size: int) -> str:
    if size <= 0:
        return ""
    if size < 1024:
        return f"{size}B"
    if size < 1024 * 1024:
        return f"{size / 1024:.0f}KB"
    return f"{size / (1024 * 1024):.1f}MB"


def _format_time_ago(dt_str: str) -> str:
    """將 datetime 字串轉為相對時間。"""
    if not dt_str:
        return ""
    try:
        from datetime import datetime
        dt = datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
        now = datetime.now(dt.tzinfo) if dt.tzinfo else datetime.now()
        delta = now - dt
        if delta.days > 30:
            return dt.strftime("%Y-%m-%d")
        if delta.days > 0:
            return f"{delta.days}天前"
        hours = delta.seconds // 3600
        if hours > 0:
            return f"{hours}小時前"
        mins = delta.seconds // 60
        if mins > 0:
            return f"{mins}分鐘前"
        return "剛剛"
    except Exception:
        return dt_str[:16] if len(dt_str) > 16 else dt_str


def _source_icon(source_type: str) -> str:
    return {"pdf": "📄", "url": "🔗", "doi": "📑", "arxiv": "📐", "text": "📝", "txt": "📃"}.get(source_type, "📄")


async def _get_graph_stats(project_id: str) -> dict:
    """取得知識圖譜統計（entities, relations, chunks）。"""
    working_dir = project_root(project_id) / "lightrag"
    stats = {"entities": 0, "relations": 0, "chunks": 0}
    try:
        vdb_chunks = working_dir / "vdb_chunks.json"
        if vdb_chunks.exists():
            data = _json.loads(vdb_chunks.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                stats["chunks"] = len(data.get("data", []))

        vdb_entities = working_dir / "vdb_entities.json"
        if vdb_entities.exists():
            data = _json.loads(vdb_entities.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                stats["entities"] = len(data.get("data", []))

        vdb_relations = working_dir / "vdb_relationships.json"
        if vdb_relations.exists():
            data = _json.loads(vdb_relations.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                stats["relations"] = len(data.get("data", []))
    except Exception:
        pass
    return stats


# ── KnowledgeScreen ModalScreen ───────────────────────────────────

class KnowledgeScreen(ModalScreen):
    """知識庫全屏管理 — 全局 / 專案兩層文獻管理。"""

    BINDINGS = [
        ("escape", "close", "關閉"),
        ("tab", "switch_scope", "切換"),
    ]

    CSS = """
    KnowledgeScreen { align: center middle; }
    #ks-main {
        width: 90; height: 85%; padding: 1 2;
        border: round #3a3a3a; background: #0a0a0a;
        border-title-color: #fafafa;
    }
    /* 頂部範圍切換 */
    .ks-scope-bar { height: 1; margin: 0 0 1 0; }
    .ks-scope-btn {
        background: transparent; color: #484f58;
        border: none; height: 1; min-width: 0;
        margin: 0; padding: 0 2;
    }
    .ks-scope-btn:hover { color: #fafafa; }
    .ks-scope-btn.-active { color: #f59e0b; text-style: bold; }
    .ks-scope-btn.-project.-active { color: #7dd3fc; text-style: bold; }
    /* 統計欄 */
    .ks-stats { height: 1; margin: 0 0 1 0; }
    /* 操作按鈕列 */
    .ks-actions { height: 1; margin: 0 0 1 0; }
    .ks-action-btn {
        background: transparent; color: #6e7681;
        border: none; height: 1; min-width: 0;
        margin: 0; padding: 0 1;
    }
    .ks-action-btn:hover { color: #fafafa; }
    /* 分隔線 */
    .ks-sep { height: 1; color: #2a2a2a; }
    /* 文獻列表 */
    #ks-doc-list { height: 1fr; }
    .ks-doc-card {
        height: auto; padding: 0 1; margin: 0 0 0 0;
        color: #c9d1d9;
    }
    .ks-doc-card:hover { background: #111111; }
    .ks-doc-meta { height: 1; color: #484f58; }
    .ks-doc-actions { height: 1; }
    .ks-small-btn {
        background: transparent; color: #484f58;
        border: none; height: 1; min-width: 0;
        margin: 0; padding: 0;
    }
    .ks-small-btn:hover { color: #7dd3fc; }
    .ks-del-btn:hover { color: #ff6b6b; }
    /* 匯入面板 */
    #ks-import-panel {
        height: auto; max-height: 12; margin: 1 0 0 0;
        padding: 1 1; border: tall #2a2a2a; background: #0d0d0d;
    }
    #ks-import-input {
        margin: 0; background: #111111; color: #fafafa;
        border: tall #3a3a3a; padding: 0 1;
    }
    #ks-import-title-input {
        margin: 0; background: #111111; color: #8b949e;
        border: tall #2a2a2a; padding: 0 1;
    }
    #ks-import-status { height: auto; margin: 0; }
    /* 重命名 input */
    .ks-rename-input {
        background: #111111; color: #fafafa;
        border: tall #3a3a3a; padding: 0 1; height: 1;
    }
    /* 底部 */
    .ks-footer { height: 1; margin: 1 0 0 0; }
    .ks-back-btn {
        background: transparent; color: #484f58;
        border: none; height: 1; min-width: 0;
        padding: 0 1;
    }
    .ks-back-btn:hover { color: #fafafa; }
    """

    def __init__(
        self,
        project_id: str = "default",
        initial_scope: str = "project",
    ) -> None:
        super().__init__()
        self._project_id = project_id
        self._scope = initial_scope  # "global" or "project"
        self._docs: list[dict] = []
        self._stats: dict = {}
        self._importing = False
        self._import_start_time: float = 0
        self._renaming_doc_id: str | None = None

    @property
    def _active_project_id(self) -> str:
        return GLOBAL_PROJECT_ID if self._scope == "global" else self._project_id

    def compose(self) -> ComposeResult:
        main = Vertical(id="ks-main")
        main.border_title = "◇ 知識庫管理"
        with main:
            # 範圍切換
            with Horizontal(classes="ks-scope-bar"):
                yield Button(
                    "◆ 全局文獻", id="ks-scope-global",
                    classes="ks-scope-btn" + (" -active" if self._scope == "global" else ""),
                )
                yield Button(
                    "◇ 專案文獻", id="ks-scope-project",
                    classes="ks-scope-btn -project" + (" -active" if self._scope == "project" else ""),
                )
                yield Static("", id="ks-scope-label")

            # 統計
            yield Static("", id="ks-stats", classes="ks-stats")

            yield Static("[dim #2a2a2a]" + "─" * 84 + "[/]", classes="ks-sep")

            # 操作按鈕
            with Horizontal(classes="ks-actions"):
                yield Button("+ 匯入", id="ks-btn-import", classes="ks-action-btn")
                yield Button("🔍 搜尋", id="ks-btn-search", classes="ks-action-btn")
                yield Button("📋 批量", id="ks-btn-bulk", classes="ks-action-btn")

            # 文獻列表
            yield VerticalScroll(id="ks-doc-list")

            # 匯入面板（初始隱藏）
            import_panel = Vertical(id="ks-import-panel")
            import_panel.display = False
            with import_panel:
                yield Static("[#8b949e]輸入檔案路徑、URL、DOI[/]")
                yield Input(
                    placeholder="/path/to/file.pdf  或  https://...  或  10.xxxx/...",
                    id="ks-import-input",
                )
                yield Input(
                    placeholder="自訂標題（選填）",
                    id="ks-import-title-input",
                )
                yield Static("", id="ks-import-status")

            # 底部
            with Horizontal(classes="ks-footer"):
                yield Button("← 回到對話", id="ks-back", classes="ks-back-btn")

    def on_mount(self) -> None:
        self._refresh_view()

    # ── 資料載入 ─────────────────────────────────────────────────

    @work(exclusive=True, group="ks_refresh")
    async def _refresh_view(self) -> None:
        """載入文獻列表 + 統計，更新 UI。"""
        pid = self._active_project_id

        # 載入文獻
        try:
            from conversation.store import ConversationStore
            store = ConversationStore(db_path=project_root(pid) / "conversations.db")
            self._docs = await store.list_documents(pid)
        except Exception:
            self._docs = []

        # 統計
        self._stats = await _get_graph_stats(pid)

        # 更新 scope label
        try:
            label = self.query_one("#ks-scope-label", Static)
            if self._scope == "global":
                label.update("[#f59e0b]跨專案基礎思考文獻[/]")
            else:
                label.update(f"[#7dd3fc]當前專案文獻[/]")
        except Exception:
            pass

        # 更新統計
        try:
            stats_w = self.query_one("#ks-stats", Static)
            s = self._stats
            doc_count = len(self._docs)
            total_size = sum(d.get("file_size", 0) for d in self._docs)
            stats_w.update(
                f"[#6e7681]文獻 [#c9d1d9]{doc_count}[/] 篇"
                f"  ·  實體 [#c9d1d9]{s['entities']}[/]"
                f"  ·  關係 [#c9d1d9]{s['relations']}[/]"
                f"  ·  片段 [#c9d1d9]{s['chunks']}[/]"
                f"  ·  總計 [#c9d1d9]{_format_size(total_size)}[/]"
                "[/]"
            )
        except Exception:
            pass

        # 更新文獻列表
        await self._render_doc_list()

    async def _render_doc_list(self) -> None:
        try:
            container = self.query_one("#ks-doc-list", VerticalScroll)
        except Exception:
            return
        await container.remove_children()

        if not self._docs:
            scope_name = "全局文獻庫" if self._scope == "global" else "專案文獻庫"
            await container.mount(
                Static(f"[dim #484f58]{scope_name}尚無文獻，按「+ 匯入」開始[/]")
            )
            return

        for doc in self._docs:
            card = await self._build_doc_card(doc)
            await container.mount(card)

    async def _build_doc_card(self, doc: dict) -> Vertical:
        """建構單一文獻卡片。"""
        doc_id = doc.get("id", "")
        title = doc.get("title", "未知文獻")
        source_type = doc.get("source_type", "")
        icon = _source_icon(source_type)
        file_size = _format_size(doc.get("file_size", 0))
        page_count = doc.get("page_count", 0)
        imported_at = doc.get("imported_at", "")
        time_ago = _format_time_ago(imported_at)
        time_abs = imported_at[:16] if imported_at else ""

        # 標題行
        display_title = title if len(title) <= 60 else title[:57] + "…"
        title_line = f"{icon} {display_title}"

        # 元資料行
        meta_parts = []
        if time_ago:
            meta_parts.append(time_ago)
        if time_abs and time_ago != time_abs:
            meta_parts.append(f"({time_abs})")
        if file_size:
            meta_parts.append(file_size)
        if page_count:
            meta_parts.append(f"{page_count}頁")
        if source_type:
            meta_parts.append(source_type.upper())
        meta_line = "  ·  ".join(meta_parts)

        card = Vertical(classes="ks-doc-card", name=doc_id)
        await card.mount(Static(title_line))
        await card.mount(Static(f"[dim #484f58]{meta_line}[/]", classes="ks-doc-meta"))

        # 操作按鈕
        actions = Horizontal(classes="ks-doc-actions")
        await card.mount(actions)
        await actions.mount(Button("重命名", classes="ks-small-btn ks-rename-btn", name=doc_id))
        await actions.mount(Button("閱讀", classes="ks-small-btn ks-read-btn", name=doc_id))
        await actions.mount(Button("刪除", classes="ks-small-btn ks-del-btn", name=doc_id))

        # 分隔線
        await card.mount(Static("[dim #1a1a1a]" + "─" * 82 + "[/]"))

        return card

    # ── 事件處理 ─────────────────────────────────────────────────

    def on_button_pressed(self, event: Button.Pressed) -> None:
        btn = event.button

        # 範圍切換
        if btn.id == "ks-scope-global":
            self._switch_scope("global")
            return
        if btn.id == "ks-scope-project":
            self._switch_scope("project")
            return

        # 返回
        if btn.id == "ks-back":
            self.dismiss(None)
            return

        # 匯入面板切換
        if btn.id == "ks-btn-import":
            self._toggle_import_panel()
            return

        # 搜尋 — 回到舊 modal
        if btn.id == "ks-btn-search":
            self.dismiss(("open_search", self._active_project_id))
            return

        # 批量 — 回到舊 modal
        if btn.id == "ks-btn-bulk":
            self.dismiss(("open_bulk", self._active_project_id))
            return

        # 文獻操作
        if btn.has_class("ks-rename-btn") and btn.name:
            self._start_rename(btn.name)
            return
        if btn.has_class("ks-read-btn") and btn.name:
            doc = next((d for d in self._docs if d.get("id") == btn.name), None)
            if doc:
                self.dismiss(("read_doc", doc))
            return
        if btn.has_class("ks-del-btn") and btn.name:
            self._delete_doc(btn.name)
            return

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id == "ks-import-input":
            value = event.value.strip()
            if value and not self._importing:
                title = ""
                try:
                    title = self.query_one("#ks-import-title-input", Input).value.strip()
                except Exception:
                    pass
                self._do_import(value, title)
            return

        # 重命名確認
        if event.input.has_class("ks-rename-input") and self._renaming_doc_id:
            new_title = event.value.strip()
            if new_title:
                self._save_rename(self._renaming_doc_id, new_title)
            self._renaming_doc_id = None
            return

    # ── 範圍切換 ─────────────────────────────────────────────────

    def _switch_scope(self, scope: str) -> None:
        if scope == self._scope:
            return
        self._scope = scope
        # 更新按鈕樣式
        for s in ("global", "project"):
            try:
                btn = self.query_one(f"#ks-scope-{s}", Button)
                if s == scope:
                    btn.add_class("-active")
                else:
                    btn.remove_class("-active")
            except Exception:
                pass
        self._refresh_view()

    def action_switch_scope(self) -> None:
        self._switch_scope("project" if self._scope == "global" else "global")

    # ── 匯入 ────────────────────────────────────────────────────

    def _toggle_import_panel(self) -> None:
        try:
            panel = self.query_one("#ks-import-panel", Vertical)
            panel.display = not panel.display
            if panel.display:
                self.query_one("#ks-import-input", Input).focus()
        except Exception:
            pass

    @work(exclusive=True, group="ks_import")
    async def _do_import(self, source: str, title: str = "") -> None:
        """執行匯入，帶 ETA 進度更新。"""
        self._importing = True
        self._import_start_time = time.time()
        pid = self._active_project_id

        status = self.query_one("#ks-import-status", Static)

        try:
            status.update("[#f59e0b]⟳ 正在提交匯入任務…[/]")

            # 啟動進度更新 timer
            self._update_import_progress("正在處理…")

            result = await self.app._import_one(source, pid, title=title)

            elapsed = time.time() - self._import_start_time
            doc_title = result.get("title", source)
            warning = result.get("warning", "")

            if warning:
                status.update(
                    f"[#f59e0b]⚠ {doc_title} — {warning} ({elapsed:.0f}s)[/]"
                )
            else:
                status.update(
                    f"[#7dd3fc]✓ {doc_title} 匯入完成 ({elapsed:.0f}s)[/]"
                )

            # 清空輸入
            try:
                self.query_one("#ks-import-input", Input).value = ""
                self.query_one("#ks-import-title-input", Input).value = ""
            except Exception:
                pass

            # 刷新列表
            self._refresh_view()

        except Exception as e:
            elapsed = time.time() - self._import_start_time
            status.update(f"[#ff6b6b]✗ 匯入失敗 ({elapsed:.0f}s): {e}[/]")
        finally:
            self._importing = False

    def _update_import_progress(self, stage: str) -> None:
        """更新匯入進度與 ETA。"""
        if not self._importing:
            return
        elapsed = time.time() - self._import_start_time
        try:
            status = self.query_one("#ks-import-status", Static)
            # 根據經驗估算 ETA
            if elapsed < 5:
                eta_str = "預計 30-60 秒"
            elif elapsed < 15:
                eta_str = "預計 20-40 秒"
            elif elapsed < 30:
                eta_str = "預計 10-30 秒"
            else:
                eta_str = "接近完成…"
            status.update(
                f"[#f59e0b]⟳ {stage}  ·  已用 {elapsed:.0f}s  ·  {eta_str}[/]"
            )
        except Exception:
            pass

    # ── 重命名 ───────────────────────────────────────────────────

    def _start_rename(self, doc_id: str) -> None:
        """在文獻卡片內嵌入重命名 Input。"""
        self._renaming_doc_id = doc_id
        doc = next((d for d in self._docs if d.get("id") == doc_id), None)
        if not doc:
            return
        self._show_rename_input(doc_id, doc.get("title", ""))

    @work(exclusive=True, group="ks_rename")
    async def _show_rename_input(self, doc_id: str, current_title: str) -> None:
        try:
            container = self.query_one("#ks-doc-list", VerticalScroll)
        except Exception:
            return
        # 找到對應的卡片，在其中插入 input
        for card in container.query(".ks-doc-card"):
            if card.name == doc_id:
                rename_input = Input(
                    value=current_title,
                    placeholder="新標題…",
                    classes="ks-rename-input",
                    name=doc_id,
                )
                await card.mount(rename_input, before=1)
                rename_input.focus()
                break

    @work(exclusive=True, group="ks_rename_save")
    async def _save_rename(self, doc_id: str, new_title: str) -> None:
        pid = self._active_project_id
        try:
            from conversation.store import ConversationStore
            store = ConversationStore(db_path=project_root(pid) / "conversations.db")
            await store.update_document_meta(doc_id, title=new_title)
            self.notify(f"已重命名為「{new_title[:30]}」")
        except Exception as e:
            self.notify(f"重命名失敗: {e}", severity="error")
        self._refresh_view()

    # ── 刪除 ────────────────────────────────────────────────────

    @work(exclusive=True, group="ks_delete")
    async def _delete_doc(self, doc_id: str) -> None:
        pid = self._active_project_id
        try:
            from conversation.store import ConversationStore
            store = ConversationStore(db_path=project_root(pid) / "conversations.db")
            await store.delete_document(doc_id)
            self.notify("文獻記錄已刪除")
        except Exception as e:
            self.notify(f"刪除失敗: {e}", severity="error")
        self._refresh_view()

    # ── 快捷鍵 ───────────────────────────────────────────────────

    def action_close(self) -> None:
        self.dismiss(None)
