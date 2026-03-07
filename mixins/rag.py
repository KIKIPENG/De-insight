"""知識庫 / RAG 相關方法。"""
from __future__ import annotations

from textual import work
from textual.containers import Vertical
from textual.css.query import NoMatches
from textual.widgets import Static

from rich.markup import escape


class RAGMixin:
    """知識庫 / RAG 相關方法。需要混入 App 才能使用。"""

    @work(exclusive=True, group="knowledge_panel")
    async def _refresh_knowledge_panel(self) -> None:
        from panels import ResearchPanel, DocumentItem
        from widgets import _get_reindex_age
        try:
            panel = self.query_one("#research-panel", ResearchPanel)
        except NoMatches:
            return
        project_id = self.state.current_project["id"] if self.state.current_project else "default"
        try:
            from rag.knowledge_graph import has_knowledge
            has_kg = has_knowledge(project_id=project_id)
        except Exception:
            has_kg = False
        kg_icon = "✓" if has_kg else "—"
        rag_label = "快速" if self.rag_mode == "fast" else "深度"
        reindex_age = _get_reindex_age()
        idx_info = f"  索引:{reindex_age}" if reindex_age else ""
        try:
            self.query_one("#kb-status-line", Static).update(
                f"[#484f58]知識庫:{kg_icon}  RAG:{rag_label}{idx_info}[/]"
            )
        except NoMatches:
            pass
        try:
            doc_list = self.query_one("#kb-doc-list", Vertical)
        except NoMatches:
            return
        await doc_list.remove_children()
        try:
            docs = await self._conv_store.list_documents(project_id)
        except Exception:
            docs = []
        if not docs:
            await doc_list.mount(
                Static("[dim #484f58]尚無文獻，按上方「匯入」開始[/]")
            )
        else:
            for d in docs[:8]:
                await doc_list.mount(DocumentItem(d))
            if len(docs) > 8:
                await doc_list.mount(
                    Static(f"[dim #484f58]… 還有 {len(docs) - 8} 份文獻[/]")
                )

    def action_import_document(self) -> None:
        self._open_knowledge_modal("import")

    async def _import_one(self, source: str, project_id: str, title: str = "") -> dict:
        """執行單筆匯入，回傳 meta dict。DOI/arXiv/URL/PDF 判斷都在這裡。"""
        import re as _re
        from rag.knowledge_graph import reset_rag, get_rag
        source = source.strip()
        is_doi = bool(_re.match(r'^10\.\d{4,}/', source))
        is_arxiv = "arxiv.org/abs/" in source
        is_url = source.startswith("http://") or source.startswith("https://")

        reset_rag()
        get_rag(project_id=project_id)

        if is_doi:
            from rag.knowledge_graph import insert_doi
            meta = await insert_doi(source, project_id=project_id, title=title)
            source_type = "doi"
        elif is_arxiv:
            from rag.knowledge_graph import insert_url
            meta = await insert_url(source.replace("/abs/", "/pdf/"), project_id=project_id, title=title)
            source_type = "url"
        elif is_url:
            from rag.knowledge_graph import insert_url
            meta = await insert_url(source, project_id=project_id, title=title)
            source_type = "url"
        else:
            from rag.knowledge_graph import insert_pdf
            meta = await insert_pdf(source, project_id=project_id, title=title)
            source_type = "pdf"

        doc_id = await self._conv_store.add_document(
            title=meta["title"],
            source_path=source,
            source_type=source_type,
            file_size=meta.get("file_size", 0),
            page_count=meta.get("page_count", 0),
            project_id=project_id,
        )
        meta["doc_id"] = doc_id
        return meta

    @work(exclusive=True)
    async def _do_import(self, source: str) -> None:
        _pid = self.state.current_project["id"] if self.state.current_project else "default"
        self.notify("匯入中…")
        try:
            await self._import_one(source, _pid)
            self.state.last_imported_source = source
            self.notify("匯入完成　ctrl+u 可重新匯入")
            self._refresh_knowledge_panel()
            await self._update_research_panel("匯入完成，知識庫已更新")
        except Exception as e:
            self.notify(f"匯入失敗: {e}")

    def action_search_knowledge(self) -> None:
        self._open_knowledge_modal("search")

    @work(exclusive=True)
    async def _do_search(self, query: str) -> None:
        try:
            from rag.knowledge_graph import query_knowledge, has_knowledge
            _pid = self.state.current_project["id"] if self.state.current_project else "default"
            if not has_knowledge(project_id=_pid):
                self.notify("知識庫為空，請先匯入文件 (ctrl+f)")
                return
            result, sources = await query_knowledge(query, project_id=_pid)
            if sources:
                self.state.last_rag_sources = sources
            await self._update_research_panel(result)
        except Exception as e:
            self.notify(f"搜尋失敗: {e}")

    def _open_knowledge_modal(self, tab: str = "import") -> None:
        project_id = self.state.current_project["id"] if self.state.current_project else "default"
        def on_dismiss(result) -> None:
            self._refresh_knowledge_panel()
            self._update_menu_bar()
            if isinstance(result, tuple):
                action, value = result
                if action == "import":
                    self._do_import(value)
                elif action == "search":
                    self._do_search(value)
        from modals import KnowledgeModal
        self.push_screen(KnowledgeModal(project_id, initial_tab=tab), callback=on_dismiss)

    def action_manage_documents(self) -> None:
        self._open_knowledge_modal("docs")

    def action_bulk_import(self) -> None:
        self._open_knowledge_modal("bulk")

    def action_view_relations(self) -> None:
        from modals import RelationModal
        self.push_screen(RelationModal())

    def action_update_document(self) -> None:
        last = self.state.last_imported_source
        if not last:
            self.notify("尚無匯入記錄，請先用 ctrl+f 匯入文件")
            return
        self._do_import(last)

    def action_view_sources(self) -> None:
        sources = self.state.last_rag_sources
        if sources:
            from modals import SourceModal
            self.push_screen(SourceModal(sources))
        else:
            self.notify("這則回應沒有知識庫來源", timeout=2)

    def action_toggle_rag_mode(self) -> None:
        if self.rag_mode == "fast":
            self.rag_mode = "deep"
            self.notify("知識檢索：深度模式（圖譜推理，較慢）")
        else:
            self.rag_mode = "fast"
            self.notify("知識檢索：快速模式（向量搜尋，<1秒）")
        self._update_menu_bar()

    @staticmethod
    def _clean_rag_display(raw: str) -> str:
        """清理 LightRAG 原始輸出，提取可讀的中文內容。"""
        import re
        text = re.sub(r"Document Chunks.*?Reference Document List[`'\s)]*:", "", raw, flags=re.DOTALL)
        text = re.sub(r"```json\s*", "", text)
        text = re.sub(r"```\s*", "", text)
        contents = re.findall(r'"content"\s*:\s*"((?:[^"\\]|\\.)*)"', text)
        if contents:
            parts = []
            for c in contents:
                c = c.replace("\\n", "\n").replace("\\t", " ").strip()
                if c and len(c) > 20:
                    parts.append(c)
            if parts:
                def cn_ratio(s):
                    cn = len(re.findall(r'[\u4e00-\u9fff]', s))
                    return cn / max(len(s), 1)
                parts.sort(key=cn_ratio, reverse=True)
                result = "\n\n---\n\n".join(parts[:3])
                result = re.sub(r"\[.*?p\.\d+\]\s*", "", result)
                return result
        text = re.sub(r'\{["\s]*reference_id["\s]*:.*?\}', "", text)
        text = re.sub(r"Reference Document List.*", "", text, flags=re.DOTALL)
        return text.strip()

    async def _update_research_panel(self, content: str) -> None:
        try:
            panel = self.query_one("#research-content", Static)
            if content:
                display = self._clean_rag_display(content)
                if not display:
                    display = content
                display = display[:800] + ("…" if len(display) > 800 else "")
                panel.update(f"[#c9d1d9]{escape(display)}[/]")
            else:
                panel.update("[dim #484f58]無結果[/]")
        except NoMatches:
            pass
