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



class _Paragraph(Static):
    """可點擊選取的段落。"""

    def __init__(self, text: str, index: int, **kwargs) -> None:
        super().__init__(text, **kwargs)
        self._index = index
        self._selected = False

    def on_click(self) -> None:
        self._selected = not self._selected
        if self._selected:
            self.add_class("para-selected")
        else:
            self.remove_class("para-selected")
        # Update selection count in parent modal
        modal = self.screen
        if isinstance(modal, DocReaderModal):
            modal._update_selection_hint()


class DocReaderModal(ModalScreen):
    """文獻全文閱讀 + 段落選取引用。"""

    BINDINGS = [("escape", "close", "關閉")]

    CSS = """
    DocReaderModal { align: center middle; }
    #doc-reader-box {
        width: 90; height: auto; max-height: 92%; padding: 1 2;
        border: round #3a3a3a; background: #0a0a0a;
        border-title-color: #7dd3fc;
    }
    #doc-reader-scroll { max-height: 75%; }
    .doc-para {
        color: #c9d1d9; padding: 0 1; margin: 0 0 1 0;
        height: auto;
    }
    .doc-para:hover { background: #1a1a1a; }
    .para-selected { background: #2a1f14; border-left: thick #d4a27a; }
    .para-selected:hover { background: #3d2a1a; }
    #doc-reader-hint { color: #484f58; height: 1; margin: 0; }
    .doc-reader-actions { height: auto; margin: 1 0 0 0; }
    """

    def __init__(self, doc: dict, project_id: str = "default") -> None:
        super().__init__()
        self._doc = doc
        self._project_id = project_id
        self._full_text = ""
        self._paragraphs: list[str] = []

    def compose(self) -> ComposeResult:
        title = self._doc.get("title", "未知文獻")
        note = (self._doc.get("note") or "").strip()
        box = Vertical(id="doc-reader-box")
        box.border_title = f"◇ {title}"
        with box:
            if note.lower().startswith("warning:"):
                detail = note[len("warning:"):].strip() or note
                if len(detail) > 180:
                    detail = detail[:180] + "…"
                yield Static(
                    f"[#d4a27a]⚠ 完成（有警告）[/]\n[dim #6e7681]{detail}[/]"
                )
                yield Static("[dim #2a2a2a]" + "─" * 84 + "[/]")
            yield VerticalScroll(
                Static("載入中…", id="doc-reader-text"),
                id="doc-reader-scroll",
            )
            yield Static("[dim #484f58]點擊段落以選取，再點取消選取[/]", id="doc-reader-hint")
            yield Static("[dim #2a2a2a]" + "─" * 84 + "[/]")
            with Horizontal(classes="doc-reader-actions"):
                yield Button("▸ 引用選取段落", classes="src-cite-btn", name="cite")
                yield Button("▸ 引用全文摘要", classes="src-cite-btn", name="cite_all")
                yield Button("← 關閉", classes="back-btn")

    def on_mount(self) -> None:
        self._load_content()

    @work(exclusive=True)
    async def _load_content(self) -> None:
        text = await self._extract_text()
        self._full_text = text
        if not text:
            try:
                self.query_one("#doc-reader-text", Static).update(
                    "[dim #484f58]無法讀取內容[/]"
                )
            except Exception:
                pass
            return

        # Split into paragraphs (double newline or page breaks)
        import re
        raw_paras = re.split(r'\n{2,}', text)
        self._paragraphs = [p.strip() for p in raw_paras if p.strip()]

        # Replace loading text with paragraph widgets
        try:
            scroll = self.query_one("#doc-reader-scroll", VerticalScroll)
            await scroll.remove_children()
            for i, para in enumerate(self._paragraphs):
                display = para
                if len(display) > 500:
                    display = display[:500] + "…"
                await scroll.mount(_Paragraph(display, i, classes="doc-para"))
        except Exception:
            pass

    def _set_loading_status(self, msg: str) -> None:
        try:
            self.query_one("#doc-reader-text", Static).update(f"[dim #484f58]{msg}[/]")
        except Exception:
            pass

    async def _extract_text(self) -> str:
        """Extract full text from PDF (PyMuPDF), URL (Jina Reader), or LightRAG kv store."""
        import json as _json
        from pathlib import Path
        title = self._doc.get("title", "")
        source_path = self._doc.get("source_path", "")
        source_type = self._doc.get("source_type", "pdf")

        # 1) Local PDF — use PyMuPDF
        if source_path and Path(source_path).exists() and source_path.lower().endswith(".pdf"):
            self._set_loading_status("正在讀取 PDF…")
            try:
                import fitz  # PyMuPDF
                doc = fitz.open(source_path)
                pages = []
                for page in doc:
                    pages.append(page.get_text())
                doc.close()
                return "\n\n".join(pages)
            except ImportError:
                pass
            except Exception:
                pass

        # 2) URL source — use Jina Reader
        is_url = source_type in ("url", "doi") or (
            source_path.startswith("http://") or source_path.startswith("https://")
        )
        if is_url and source_path.startswith("http"):
            self._set_loading_status("正在透過 Jina Reader 抓取網頁…")
            try:
                from rag.knowledge_graph import _fetch_with_jina_reader
                text, _meta = await _fetch_with_jina_reader(source_path)
                if text and len(text) >= 50:
                    return text
            except Exception:
                pass
            # Jina Reader failed — try direct HTTP fetch + basic HTML strip
            self._set_loading_status("Jina Reader 失敗，嘗試直接抓取…")
            try:
                import httpx
                async with httpx.AsyncClient(timeout=20, follow_redirects=True) as client:
                    resp = await client.get(source_path)
                    if resp.status_code < 400:
                        ct = resp.headers.get("content-type", "")
                        if "html" in ct:
                            text = self._strip_html(resp.text)
                            if text and len(text) >= 50:
                                return text
                        elif "text" in ct:
                            return resp.text.strip()
            except Exception:
                pass

        # 3) Try LightRAG kv_store_full_docs (cached ingestion content)
        self._set_loading_status("正在搜尋已建圖內容…")
        try:
            from paths import project_root
            kv_path = project_root(self._project_id) / "lightrag" / "kv_store_full_docs.json"
            if kv_path.exists():
                data = _json.loads(kv_path.read_text(encoding="utf-8"))
                # Search for matching title
                for key, val in data.items():
                    content = ""
                    if isinstance(val, dict):
                        content = val.get("content", "") or val.get("original_content", "")
                    elif isinstance(val, str):
                        content = val
                    if content and title and title.lower() in content[:200].lower():
                        return content
                # If no title match, try source_path match
                for key, val in data.items():
                    if source_path and source_path in key:
                        if isinstance(val, dict):
                            return val.get("content", "") or val.get("original_content", "")
                        elif isinstance(val, str):
                            return val
        except Exception:
            pass

        return ""

    @staticmethod
    def _strip_html(html: str) -> str:
        """Basic HTML → plain text fallback."""
        import re
        # Remove script/style blocks
        text = re.sub(r'<(script|style)[^>]*>.*?</\1>', '', html, flags=re.DOTALL | re.IGNORECASE)
        # Remove tags
        text = re.sub(r'<[^>]+>', ' ', text)
        # Collapse whitespace
        text = re.sub(r'[ \t]+', ' ', text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        # Decode common entities
        import html as _html
        text = _html.unescape(text)
        return text.strip()

    def _get_selected_paragraphs(self) -> list[str]:
        """Return full text of all selected paragraphs in order."""
        selected = []
        try:
            for widget in self.query(_Paragraph):
                if widget._selected:
                    idx = widget._index
                    if idx < len(self._paragraphs):
                        selected.append(self._paragraphs[idx])
        except Exception:
            pass
        return selected

    def _update_selection_hint(self) -> None:
        count = len(self._get_selected_paragraphs())
        try:
            hint = self.query_one("#doc-reader-hint", Static)
            if count > 0:
                hint.update(f"[#d4a27a]已選取 {count} 個段落[/]  [dim #484f58]點擊段落以選取/取消[/]")
            else:
                hint.update("[dim #484f58]點擊段落以選取，再點取消選取[/]")
        except Exception:
            pass

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.has_class("back-btn"):
            self.dismiss(None)
            return
        if event.button.name == "cite":
            selected = self._get_selected_paragraphs()
            if not selected:
                self.notify("請先點擊選取要引用的段落", severity="warning", timeout=3)
                return
            title = self._doc.get("title", "未知文獻")
            excerpt = "\n\n".join(selected)
            self.dismiss(f"__cite__:[{title}]\n{excerpt}")
        elif event.button.name == "cite_all":
            title = self._doc.get("title", "未知文獻")
            excerpt = self._full_text[:2000] if self._full_text else ""
            if excerpt:
                self.dismiss(f"__cite__:[{title}]\n{excerpt}")
            else:
                self.notify("無內容可引用", timeout=2)

    def action_close(self) -> None:
        self.dismiss(None)


# ── DocumentManageModal ───────────────────────────────────────────
