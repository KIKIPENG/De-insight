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
        try:
            panel = self.query_one("#research-panel", ResearchPanel)
        except NoMatches:
            return
        project_id = self.state.current_project["id"] if self.state.current_project else "default"
        readiness_status = ""
        try:
            from rag.readiness import get_readiness_service
            snapshot = await get_readiness_service().get_snapshot(project_id)
            readiness_status = snapshot.status_label
        except Exception:
            readiness_status = ""

        # Document list
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
            if readiness_status == "building":
                await doc_list.mount(
                    Static("[dim #6e7681]建圖中：內容可能尚未完整[/]")
                )
            elif readiness_status == "degraded":
                await doc_list.mount(
                    Static("[dim #d4a27a]部分匯入失敗：結果可能不完整[/]")
                )
            for d in docs[:8]:
                await doc_list.mount(DocumentItem(d))
            if len(docs) > 8:
                await doc_list.mount(
                    Static(f"[dim #484f58]… 還有 {len(docs) - 8} 份文獻[/]")
                )

    def action_import_document(self) -> None:
        self._open_knowledge_modal("import")

    async def _import_one(self, source: str, project_id: str, title: str = "") -> dict:
        """Submit a single import job and wait for completion.

        All actual insert_* work runs in the worker process.
        Returns meta dict compatible with previous API (title, doc_id, warning, etc.).
        """
        import re as _re
        source = source.strip()
        is_doi = bool(_re.match(r'^10\.\d{4,}/', source))
        is_arxiv = "arxiv.org/abs/" in source
        is_url = source.startswith("http://") or source.startswith("https://")

        if is_doi:
            source_type = "doi"
        elif is_arxiv:
            source_type = "url"
            source = source.replace("/abs/", "/pdf/")
        elif is_url:
            source_type = "url"
        else:
            source_type = "pdf"

        svc = self._get_ingestion_service()
        job = await svc.submit_and_wait(project_id, source, source_type, title=title)
        result = job.get("_result", {})
        return {
            "title": result.get("title", title or source),
            "doc_id": result.get("doc_id", ""),
            "file_size": result.get("file_size", 0),
            "page_count": result.get("page_count", 0),
            "warning": result.get("warning", ""),
        }

    def _set_import_status(self, status: str) -> None:
        from widgets import MenuBar
        try:
            menu = self.query_one("#menu-bar", MenuBar)
            if status:
                menu.show_progress(status)
            else:
                menu.clear_notification()
        except Exception:
            pass

    @staticmethod
    def _clean_file_path(path: str) -> str:
        """Clean shell-escaped / URL-encoded file paths (spaces, quotes, file://)."""
        from urllib.parse import unquote, urlparse
        t = path.strip().strip("'\"")
        if t.startswith("file://"):
            t = unquote(urlparse(t).path)
        if "%20" in t:
            t = unquote(t)
        t = t.replace("\\ ", " ")
        return t

    @work(exclusive=True)
    async def _do_import(self, source: str, title: str = "") -> None:
        import re as _re
        _pid = self.state.current_project["id"] if self.state.current_project else "default"
        source = source.strip()

        # Determine source_type
        is_doi = bool(_re.match(r'^10\.\d{4,}/', source))
        is_arxiv = "arxiv.org/abs/" in source
        is_url = source.startswith("http://") or source.startswith("https://")

        if is_doi:
            source_type = "doi"
        elif is_arxiv:
            source_type = "url"
            source = source.replace("/abs/", "/pdf/")
        elif is_url:
            source_type = "url"
        else:
            source = self._clean_file_path(source)
            if source.lower().endswith(".txt"):
                source_type = "txt"
            elif source.lower().endswith(".pdf"):
                source_type = "pdf"
            else:
                self.notify("僅支援 PDF 或 TXT 檔案", severity="warning", timeout=3)
                return

        try:
            svc = self._get_ingestion_service()
            await svc.submit(_pid, source, source_type, title=title)
            self.state.last_imported_source = source
            self.notify("已排入建圖佇列")
        except Exception as e:
            self.notify(f"排入佇列失敗: {e}")

    @work(exclusive=True)
    async def _do_import_text(self, payload: dict) -> None:
        """手動貼上標題＋內文匯入知識庫。"""
        _pid = self.state.current_project["id"] if self.state.current_project else "default"
        title = (payload or {}).get("title", "").strip()
        content = (payload or {}).get("content", "").strip()
        if not content:
            self.notify("內文不能為空")
            return
        if not title:
            title = "手動貼上文獻"

        try:
            svc = self._get_ingestion_service()
            await svc.submit(
                _pid, f"manual:{title}", "text",
                title=title,
                payload={"content": content},
            )
            self.state.last_imported_source = f"manual:{title}"
            self.notify("已排入建圖佇列（手動貼上）")
        except Exception as e:
            self.notify(f"排入佇列失敗: {e}")

    def action_search_knowledge(self) -> None:
        self._open_knowledge_modal("search")

    @work(exclusive=True)
    async def _do_search(self, query: str) -> None:
        try:
            _pid = self.state.current_project["id"] if self.state.current_project else "default"
            # Prevent stale source carryover between searches.
            self.state.last_rag_sources = []

            # Use readiness gate — same rules as chat RAG
            from rag.readiness import get_readiness_service
            snapshot = await get_readiness_service().get_snapshot(_pid)

            if snapshot.status_label == "empty":
                self.notify("知識庫為空，請先匯入文件 (ctrl+f)")
                return
            if snapshot.status_label == "degraded" and not snapshot.has_ready_chunks:
                msg = "匯入曾失敗，知識庫尚無可檢索內容；請重新匯入文件"
                if snapshot.last_error:
                    msg += f"（{snapshot.last_error[:60]}）"
                self.notify(msg, severity="warning")
                return
            if snapshot.status_label == "building" and not snapshot.has_ready_chunks:
                self.notify("知識庫建構中，完成後即可搜尋")
                return

            from rag.knowledge_graph import query_knowledge
            result, sources = await query_knowledge(query, project_id=_pid)
            if not result:
                self.notify("搜尋完成，但未找到相關內容")
                await self._update_research_panel("")
                return
            self.state.last_rag_sources = sources or []
            await self._update_research_panel(result)

            # Show readiness hint if degraded/building
            if snapshot.status_label == "building":
                self.notify("建圖中，結果可能不完整", timeout=3)
            elif snapshot.status_label == "degraded":
                self.notify("部分文獻匯入失敗，搜尋結果可能不完整", timeout=3)
        except Exception as e:
            self.notify(f"搜尋失敗: {e}")

    def _open_knowledge_modal(self, tab: str = "import") -> None:
        project_id = self.state.current_project["id"] if self.state.current_project else "default"
        def on_dismiss(result) -> None:
            self._refresh_knowledge_panel()
            self._update_menu_bar()
            if isinstance(result, str) and result.startswith("__fill__:"):
                self.fill_input(result[len("__fill__:"):])
                return
            if isinstance(result, tuple):
                action, value = result
                if action == "import":
                    if isinstance(value, dict):
                        self._do_import(value["source"], title=value.get("title", ""))
                    else:
                        self._do_import(value)
                elif action == "import_text":
                    self._do_import_text(value)
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
            def _on_source_result(result):
                if result and isinstance(result, str) and result.startswith("__cite__:"):
                    self.fill_input(result[len("__cite__:"):])
            self.push_screen(SourceModal(sources), callback=_on_source_result)
        else:
            self.notify("這則回應沒有知識庫來源", timeout=2)

    @work(exclusive=True, group="backfill_captions")
    async def action_backfill_captions(self) -> None:
        """為圖片庫重新生成 caption（所有圖片）。"""
        project_id = self.state.current_project["id"] if self.state.current_project else "default"
        self.notify("開始重新生成 caption…")
        try:
            from rag.image_store import regenerate_all_captions
            total, updated = await regenerate_all_captions(
                project_id=project_id,
                only_fallback=False,
                progress_callback=lambda current, t: self.notify(
                    f"進度：{current}/{t}"
                ),
            )
            if total == 0:
                self.notify("圖片庫為空")
            else:
                self.notify(f"完成：{updated}/{total} 張圖片重新生成 caption")
        except Exception as e:
            self.notify(f"重新生成失敗: {e}")

    @work(exclusive=True, group="reindex_images")
    async def action_reindex_images(self) -> None:
        """為所有圖片生成圖片向量（/reindex-images 指令）。"""
        if not self.state.current_project:
            self.notify("請先選擇專案", severity="warning")
            return
        pid = self.state.current_project["id"]
        self.notify("開始重新索引圖片向量...")
        try:
            from rag.image_store import reindex_all_images
            result = await reindex_all_images(
                pid,
                progress_callback=lambda current, total: self.notify(
                    f"索引中：{current}/{total}", timeout=2
                ),
            )
            self.notify(
                f"索引完成：{result['updated']} 更新、{result['skipped']} 跳過、{result['failed']} 失敗"
            )
        except Exception as e:
            self.notify(f"索引失敗: {e}")

    def action_cite_research(self) -> None:
        display = getattr(self, '_last_research_display', '')
        if display:
            self.fill_input(f"[知識庫參考]\n{display}")
        else:
            self.notify("目前無知識庫內容可引用", timeout=2)

    def _get_ingestion_service(self):
        """Get or create global IngestionService singleton."""
        from rag.ingestion_service import get_ingestion_service
        return get_ingestion_service()

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
        # 先從原始文字提取 JSON content（在移除 header/footer 之前）
        contents = re.findall(r'"content"\s*:\s*"((?:[^"\\]|\\.)*)"', raw)
        if not contents:
            text = re.sub(r"Document Chunks.*?Reference Document List[`'\s)]*:", "", raw, flags=re.DOTALL)
            text = re.sub(r"```json\s*", "", text)
            text = re.sub(r"```\s*", "", text)
        else:
            text = raw
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
                self._last_research_display = display
                panel.update(f"[#c9d1d9]{escape(display)}[/]")
            else:
                self._last_research_display = ""
                panel.update("[dim #484f58]無結果[/]")
        except NoMatches:
            pass
        # Update cite link visibility
        try:
            cite_link = self.query_one("#research-cite", Static)
            if content:
                cite_link.update("[#484f58]▸ 引用到對話[/]")
            else:
                cite_link.update("")
        except NoMatches:
            pass
