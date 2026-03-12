"""De-insight v0.2 — 右側面板 Widget"""

from __future__ import annotations

from pathlib import Path

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import Button, Static, TextArea


# ── Right Panel Widgets ──────────────────────────────────────────────


class MemoryItem(Static):
    """單條記憶顯示，可點擊閱覽完整內容。"""

    def __init__(self, mem: dict, **kwargs) -> None:
        self.mem = mem
        self._memory_type = mem.get("type", "")
        icons = {
            "insight": "💡", "question": "❓", "reaction": "💭",
            "evolution": "🔄", "contradiction": "⚡",
        }
        icon = icons.get(mem["type"], "◇")
        label = {
            "insight": "洞見", "question": "問題", "reaction": "反應",
            "evolution": "演變", "contradiction": "矛盾",
        }.get(mem["type"], mem["type"])
        content = mem["content"]
        if len(content) > 30:
            content = content[:30] + "…"
        self._base_text = f"{icon} [{label}] {content}"
        extra_classes = "memory-item"
        if mem["type"] in ("evolution", "contradiction"):
            extra_classes += f" -{mem['type']}"
        super().__init__(self._base_text, classes=extra_classes, **kwargs)

    def mark_focus_tagged(self) -> None:
        self.add_class("-focus-tagged")
        if not str(self.renderable).startswith("◈ "):
            self.update(f"◈ {self._base_text}")

    def on_click(self) -> None:
        from modals import MemoryDetailModal

        def on_result(result: str | None) -> None:
            if result and result.startswith("discuss:"):
                content = result.removeprefix("discuss:")
                self.app._start_discussion_from_memory(content)
            elif result and result.startswith("__insert__:"):
                content = result[len("__insert__:"):]
                self.app.fill_input(content)

        self.app.push_screen(MemoryDetailModal(self.mem), callback=on_result)


class KnowledgeActionButton(Button):
    """知識庫面板操作按鈕。"""

    def __init__(self, label: str, action: str, **kwargs) -> None:
        super().__init__(label, **kwargs)
        self._action = action

    def on_button_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        method = getattr(self.app, f"action_{self._action}", None)
        if method:
            method()


class DocumentItem(Static):
    """知識庫文獻項目，可點擊管理。"""

    def __init__(self, doc: dict, **kwargs) -> None:
        self.doc = doc
        title = doc.get("title", "未知文獻")
        if len(title) > 28:
            title = title[:28] + "…"
        source_type = doc.get("source_type", "")
        icon = {
            "pdf": "📄",
            "txt": "📝",
            "md": "📝",
            "url": "🔗",
            "doi": "📑",
            "arxiv": "📐",
        }.get(source_type, "📄")
        super().__init__(f"{icon} {title}", classes="doc-item", **kwargs)

    def on_click(self) -> None:
        self.app.action_manage_documents()


class ResearchPanel(Vertical):
    """右上：知識庫 + 問題意識面板。"""

    def compose(self) -> ComposeResult:
        yield Horizontal(
            Button("匯入", id="tab-import", classes="kb-action -active"),
            Button("搜尋", id="tab-search", classes="kb-action"),
            Button("文獻", id="tab-docs", classes="kb-action"),
            Button("問題", id="tab-focus", classes="kb-action"),
            classes="kb-actions-row",
        )
        yield Static("[dim #484f58]─[/]" * 30, classes="kb-divider")

        with Vertical(id="research-view"):
            yield Static("[dim #6e7681]文獻[/]", classes="kb-section-label")
            with VerticalScroll(id="kb-doc-scroll"):
                yield Vertical(id="kb-doc-list")
            yield Static("[dim #6e7681]研究結果[/]", classes="kb-section-label")
            with VerticalScroll(id="research-result-scroll"):
                yield Static("[dim #484f58]尚未檢索[/]", id="research-content")
            yield ResearchCiteLink("引用到對話", id="research-cite", disabled=True)

        with Vertical(id="focus-view"):
            yield Static("使用單一 Markdown 區塊編輯（以 --- 包住）", classes="focus-label")
            yield TextArea("", id="focus-editor", classes="focus-editor")
            with Horizontal(id="focus-buttons"):
                yield Button("複製模板", id="btn-focus-template", variant="default")
                yield Button("匯入 .md", id="btn-focus-import", variant="default")
                yield Button("對焦評估", id="btn-focus-evaluate", variant="primary")
            yield Static("", id="focus-import-status")

    def on_mount(self) -> None:
        self._toggle_view(show_focus=False)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        bid = event.button.id or ""
        if bid in {"tab-import", "tab-search", "tab-docs", "tab-focus"}:
            event.stop()
            self._activate_tab(bid)
            return

        if bid == "btn-focus-template":
            event.stop()
            from focus import TEMPLATE
            import pyperclip
            try:
                pyperclip.copy(TEMPLATE)
                self.app.notify("模板已複製到剪貼簿")
            except Exception:
                self.app.notify("複製失敗，請手動複製：\n" + TEMPLATE, timeout=10)
            return

        if bid == "btn-focus-import":
            event.stop()
            from modals import FocusImportModal
            self.app.push_screen(FocusImportModal(), callback=self._on_focus_imported)
            return

        if bid == "btn-focus-evaluate":
            event.stop()
            self.app.action_focus_evaluate()

    def on_text_area_changed(self, event: TextArea.Changed) -> None:
        if (event.text_area.id or "") != "focus-editor":
            return
        self._auto_save_focus()

    def _activate_tab(self, tab_id: str) -> None:
        for bid in ("tab-import", "tab-search", "tab-docs", "tab-focus"):
            try:
                btn = self.query_one(f"#{bid}", Button)
            except Exception:
                continue
            if bid == tab_id:
                btn.add_class("-active")
            else:
                btn.remove_class("-active")

        if tab_id == "tab-focus":
            self._toggle_view(show_focus=True)
            self._load_focus_fields()
            return

        self._toggle_view(show_focus=False)
        method_name = {
            "tab-import": "action_import_document",
            "tab-search": "action_search_knowledge",
            "tab-docs": "action_open_knowledge_screen",
        }.get(tab_id)
        if method_name:
            method = getattr(self.app, method_name, None)
            if method:
                method()

    def _toggle_view(self, *, show_focus: bool) -> None:
        try:
            focus_view = self.query_one("#focus-view", Vertical)
            focus_view.styles.display = "block" if show_focus else "none"
        except Exception:
            pass
        try:
            research_view = self.query_one("#research-view", Vertical)
            research_view.styles.display = "none" if show_focus else "block"
        except Exception:
            pass

    def _get_project_root(self) -> Path | None:
        try:
            from paths import project_root
            pid = self.app.state.current_project["id"]
            return project_root(pid)
        except Exception:
            return None

    def _load_focus_fields(self) -> None:
        from focus import load_focus
        project_root = self._get_project_root()
        if not project_root:
            return
        fields = load_focus(project_root)
        try:
            self.query_one("#focus-editor", TextArea).load_text(self._render_focus_editor(fields))
        except Exception:
            pass

    def _auto_save_focus(self) -> None:
        from focus import save_focus
        project_root = self._get_project_root()
        if not project_root:
            return

        try:
            raw = self.query_one("#focus-editor", TextArea).text
        except Exception:
            raw = ""
        fields = self._parse_focus_editor(raw)
        save_focus(project_root, fields)

    def _on_focus_imported(self, path: str | None) -> None:
        if not path:
            return
        from focus import import_focus, save_focus

        p = Path(path).expanduser()
        if not p.exists():
            self.app.notify(f"找不到檔案：{path}", severity="error")
            return

        try:
            raw = p.read_text(encoding="utf-8")
        except Exception as e:
            self.app.notify(f"讀取失敗：{e}", severity="error")
            return

        fields, matched = import_focus(raw)

        try:
            self.query_one("#focus-editor", TextArea).load_text(self._render_focus_editor(fields))
        except Exception:
            pass

        project_root = self._get_project_root()
        if project_root:
            save_focus(project_root, fields)

        ok = [f for f, v in matched.items() if v]
        missing = [f for f, v in matched.items() if not v]
        ok_str = "  ".join(f"✓ {f}" for f in ok) if ok else ""
        miss_str = "  ".join(f"— {f}" for f in missing) if missing else ""
        status = f"已匯入　{ok_str}　{miss_str}".strip()

        try:
            self.query_one("#focus-import-status", Static).update(status)
        except Exception:
            pass

        self.app.notify("問題意識已匯入")

    @staticmethod
    def _render_focus_editor(fields: dict[str, str]) -> str:
        display_to_canonical = [
            ("問題意識", "問題意識"),
            ("標籤", "標籤"),
            ("作品形式", "作品形式"),
            ("目標", "目標"),
            ("限制", "限制"),
        ]
        chunks: list[str] = []
        for i, (display, canonical) in enumerate(display_to_canonical):
            value = (fields.get(canonical, "") or "").strip()
            block = f"{display}：\n{value}" if value else f"{display}："
            chunks.append(block)
            if i < len(display_to_canonical) - 1:
                chunks.append("---")
        return "\n\n".join(chunks).strip() + "\n"

    @staticmethod
    def _parse_focus_editor(raw: str) -> dict[str, str]:
        from focus import import_focus

        text = (raw or "").strip()
        result = {k: "" for k in ("問題意識", "標籤", "作品形式", "目標", "限制")}
        if not text:
            return result

        # 先解析「欄位：內容 + --- 分隔」的段落格式
        import re
        display_to_canonical = {
            "問題意識": "問題意識",
            "核心問題": "問題意識",
            "標籤": "標籤",
            "作品形式": "作品形式",
            "目標": "目標",
            "限制": "限制",
            "邊界": "限制",
        }
        heading_re = re.compile(r"^(問題意識|核心問題|標籤|作品形式|目標|限制|邊界)\s*[：:]\s*(.*)$")
        current: str | None = None
        buf: list[str] = []
        matched_any = False

        def flush() -> None:
            nonlocal current, buf
            if current is None:
                return
            result[current] = "\n".join(buf).strip()
            current = None
            buf = []

        for line in text.splitlines():
            stripped = line.strip()
            if stripped == "---":
                flush()
                continue
            m = heading_re.match(stripped)
            if m:
                matched_any = True
                flush()
                current = display_to_canonical[m.group(1)]
                inline_value = m.group(2).strip()
                buf = [inline_value] if inline_value else []
                continue
            if current is not None:
                buf.append(line.rstrip())
        flush()

        if matched_any:
            return result

        # fallback：仍支援 frontmatter / 純文字匯入
        fields, _ = import_focus(text)
        for key in result:
            result[key] = (fields.get(key, "") or "").strip()
        return result


class ResearchCiteLink(Button):
    """知識庫引用按鈕，將 RAG 摘要注入聊天。"""

    def on_button_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        method = getattr(self.app, "action_cite_research", None)
        if method:
            method()


class MemoryPanel(VerticalScroll):
    """右下：記憶面板。"""

    def compose(self) -> ComposeResult:
        yield Static(
            "[dim #484f58]對話後自動記錄洞見[/]",
            id="memory-content",
        )
