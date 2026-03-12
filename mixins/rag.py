"""知識庫 / RAG 相關方法。"""
from __future__ import annotations

from textual import work
from textual.containers import Vertical
from textual.css.query import NoMatches
from textual.widgets import Button, Static

from rich.markup import escape


class RAGMixin:
    """知識庫 / RAG 相關方法。需要混入 App 才能使用。"""

    @work(exclusive=True, group="knowledge_panel")
    async def _refresh_knowledge_panel(self) -> None:
        from panels import ResearchPanel, DocumentItem
        from paths import GLOBAL_PROJECT_ID, project_root
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

        # 載入專案文獻
        try:
            docs = await self._conv_store.list_documents(project_id)
        except Exception:
            docs = []

        # 載入全局文獻（如果當前不在全局專案）
        global_docs = []
        if project_id != GLOBAL_PROJECT_ID:
            try:
                from conversation.store import ConversationStore
                global_store = ConversationStore(
                    db_path=project_root(GLOBAL_PROJECT_ID) / "conversations.db"
                )
                global_docs = await global_store.list_documents(GLOBAL_PROJECT_ID)
            except Exception:
                pass

        if not docs and not global_docs:
            await doc_list.mount(
                Static("[dim #484f58]尚無文獻，按上方「文獻」開始管理[/]")
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
            # 專案文獻
            if docs:
                for d in docs[:5]:
                    await doc_list.mount(DocumentItem(d))
                if len(docs) > 5:
                    await doc_list.mount(
                        Static(f"[dim #484f58]… 還有 {len(docs) - 5} 份專案文獻[/]")
                    )
            # 全局文獻摘要
            if global_docs:
                await doc_list.mount(
                    Static(f"[dim #f59e0b]◆ 全局文獻 {len(global_docs)} 篇[/]")
                )

    def action_import_document(self) -> None:
        self._open_knowledge_modal("import")

    async def _import_one(self, source: str, project_id: str, title: str = "") -> dict[str, int | str]:
        """Submit a single import job and wait for completion.

        All actual insert_* work runs in the worker process.
        Returns meta dict compatible with previous API (title, doc_id, warning, etc.).
        """
        import os.path as _osp
        import re as _re
        source = source.strip()
        title = (title or "").strip()
        if not title:
            raise ValueError("請先輸入文獻標題（必填）")
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
            elif source.lower().endswith(".md"):
                source_type = "md"
            else:
                source_type = "pdf"

        svc = self._get_ingestion_service()
        job = await svc.submit_and_wait(project_id, source, source_type, title=title)
        result = job.get("_result", {})
        return {
            "title": result.get("title", title or _osp.basename(source)),
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
        self.state.import_in_progress = True
        try:
            _pid = self.state.current_project["id"] if self.state.current_project else "default"
            source = source.strip()
            title = (title or "").strip()
            if not title:
                self.notify("請先輸入文獻標題（必填）", severity="warning", timeout=3)
                return

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
                elif source.lower().endswith(".md"):
                    source_type = "md"
                elif source.lower().endswith(".pdf"):
                    source_type = "pdf"
                else:
                    self.notify("僅支援 PDF、TXT 或 MD 檔案", severity="warning", timeout=3)
                    return

            try:
                svc = self._get_ingestion_service()
                await svc.submit(_pid, source, source_type, title=title)
                self.state.last_imported_source = source
                self.notify("已排入建圖佇列")
            except Exception as e:
                self.notify(f"排入佇列失敗: {e}")
        finally:
            self.state.import_in_progress = False

    @work(exclusive=True)
    async def _do_import_text(self, payload: dict[str, str]) -> None:
        """手動貼上標題＋內文匯入知識庫。"""
        self.state.import_in_progress = True
        try:
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
        finally:
            self.state.import_in_progress = False

    def action_search_knowledge(self) -> None:
        self._open_knowledge_modal("search")

    def _highlight_keywords(self, text: str, query: str) -> str:
        """Highlight search keywords in text using Rich markup."""
        import re
        if not query or not text:
            return text

        # Extract words from query (simple tokenization)
        keywords = re.findall(r'\w+', query.lower())
        if not keywords:
            return text

        # Create case-insensitive regex pattern
        result = text
        for keyword in keywords:
            # Use word boundaries to match whole words only
            pattern = rf'\b{re.escape(keyword)}\b'
            # Replace with Rich bold markup (case-insensitive)
            result = re.sub(
                pattern,
                f'[bold]{keyword}[/bold]',
                result,
                flags=re.IGNORECASE
            )

        return result

    @work(exclusive=True)
    async def _do_search(self, query: str) -> None:
        try:
            _pid = self.state.current_project["id"] if self.state.current_project else "default"
            # Prevent stale source carryover between searches.
            self.state.last_rag_sources = []
            await self._update_research_panel(
                status="loading",
                content="正在搜尋知識庫…",
                query=query,
            )

            # Use readiness gate — same rules as chat RAG
            from rag.readiness import get_readiness_service
            snapshot = await get_readiness_service().get_snapshot(_pid)

            if snapshot.status_label == "empty":
                self.notify("知識庫為空，請先匯入文件 (ctrl+f)")
                await self._update_research_panel(status="empty", content="", query=query)
                return
            if snapshot.status_label == "degraded" and not snapshot.has_ready_chunks:
                msg = "匯入曾失敗，知識庫尚無可檢索內容；請重新匯入文件"
                if snapshot.last_error:
                    msg += f"（{snapshot.last_error[:60]}）"
                self.notify(msg, severity="warning")
                await self._update_research_panel(status="error", content=msg, query=query)
                return
            if snapshot.status_label == "building" and not snapshot.has_ready_chunks:
                self.notify("知識庫建構中，完成後即可搜尋")
                await self._update_research_panel(
                    status="loading",
                    content="知識庫建構中，請稍候再試",
                    query=query,
                )
                return

            from rag.knowledge_graph import query_knowledge
            result, sources = await query_knowledge(query, project_id=_pid)
            if not result:
                self.notify("搜尋完成，但未找到相關內容")
                await self._update_research_panel(
                    status="empty",
                    content="",
                    sources=[],
                    query=query,
                )
                return

            # Highlight search keywords in result
            highlighted_result = self._highlight_keywords(result, query)

            self.state.last_rag_sources = sources or []
            await self._update_research_panel(
                status="degraded" if snapshot.status_label == "degraded" else "ready",
                content=highlighted_result,
                sources=sources or [],
                query=query,
            )

            # Show readiness hint if degraded/building
            if snapshot.status_label == "building":
                self.notify("建圖中，結果可能不完整", timeout=3)
            elif snapshot.status_label == "degraded":
                self.notify("部分文獻匯入失敗，搜尋結果可能不完整", timeout=3)
        except Exception as e:
            self.notify(f"搜尋失敗: {e}")
            await self._update_research_panel(status="error", content=f"搜尋失敗：{e}", query=query)

    def _open_knowledge_modal(self, tab: str = "import") -> None:
        project_id = self.state.current_project["id"] if self.state.current_project else "default"
        def on_dismiss(result: str | tuple[str, str] | None) -> None:
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

    def action_open_knowledge_screen(self) -> None:
        """開啟全屏 Knowledge 管理介面。"""
        project_id = self.state.current_project["id"] if self.state.current_project else "default"
        def on_dismiss(result: tuple[str, str] | None) -> None:
            self._refresh_knowledge_panel()
            self._update_menu_bar()
            if result is None:
                return
            if isinstance(result, tuple):
                action, value = result
                if action == "open_search":
                    self._open_knowledge_modal("search")
                elif action == "open_bulk":
                    self._open_knowledge_modal("bulk")
                elif action == "read_doc":
                    from modals import DocReaderModal
                    self.push_screen(DocReaderModal(value, project_id))
        from knowledge_screen import KnowledgeScreen
        self.push_screen(KnowledgeScreen(project_id=project_id), callback=on_dismiss)

    def action_manage_documents(self) -> None:
        self.action_open_knowledge_screen()

    def action_view_relations(self) -> None:
        from modals import RelationModal
        self.push_screen(RelationModal())

    def action_update_document(self) -> None:
        last = self.state.last_imported_source
        if not last:
            self.notify("尚無匯入記錄，請先用 ctrl+f 匯入文件")
            return
        self.notify("請開啟匯入視窗並重新輸入標題後再更新", severity="warning", timeout=4)
        self._open_knowledge_modal("import")

    def action_view_sources(self) -> None:
        sources = self.state.last_rag_sources
        if sources:
            from modals import SourceModal
            def _on_source_result(result: str | None) -> None:
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
        state = getattr(self, "_research_panel_state", {}) or {}
        sources = state.get("sources", []) or []
        context = state.get("raw_result", "") or state.get("display_text", "")
        if sources:
            lines: list[str] = []
            for idx, src in enumerate(sources[:8], start=1):
                title = (src.get("title", "") or src.get("file", "") or "未知來源").strip()
                snippet = (src.get("snippet", "") or "").strip()
                file_path = (src.get("file", "") or "").strip()
                head = f"{idx}. {title}"
                if file_path and file_path != title:
                    head += f" ({file_path})"
                lines.append(head)
                if snippet:
                    lines.append(f"   摘要：{snippet}")
            cite_text = "[知識庫參考來源]\n" + "\n".join(lines)
            self.fill_input(cite_text)
        elif context:
            self.fill_input(f"[知識庫參考]\n{context}")
        else:
            self.notify("目前無知識庫內容可引用", timeout=2)

    def _get_ingestion_service(self) -> object:
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
        """清理 RAG 原始輸出，保留可讀段落（最小破壞）。"""
        import re
        text = raw or ""
        text = text.replace("\\n", "\n").replace("\\t", " ").replace("\\r", "")
        text = re.sub(r"```json\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"```\s*", "", text)
        text = re.sub(r"Document Chunks.*?Reference Document List[`'\s)]*:", "", text, flags=re.DOTALL)
        text = re.sub(r"Reference Document List.*", "", text, flags=re.DOTALL)
        text = re.sub(r"\{[^{}]*\"reference_id\"[^{}]*\}", "", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    @staticmethod
    def _build_thinking_block(diagnostics: dict | None) -> str:
        """Build thinking trace block from pipeline diagnostics."""
        if not diagnostics:
            return ""
        trace = diagnostics.get("thinking_trace", "")
        if not trace:
            return ""
        from rich.markup import escape
        return f"[dim #8b949e]{escape(trace)}[/]\n\n"

    @staticmethod
    def _build_research_meta(content: str, sources: list[dict]) -> str:
        """Build compact metadata block shown before research result body."""
        import re

        # Titles
        titles: list[str] = []
        for src in sources or []:
            t = (src.get("title", "") or "").strip()
            if not t:
                continue
            if t not in titles:
                titles.append(t)
            if len(titles) >= 5:
                break
        title_text = "、".join(titles[:3]) if titles else "未識別"

        # Concepts: prefer [[概念]] markers, then source snippets fallback.
        concept_set: list[str] = []
        for c in re.findall(r"\[\[([^\[\]]+)\]\]", content or ""):
            c = c.strip()
            if c and c not in concept_set:
                concept_set.append(c)
            if len(concept_set) >= 8:
                break
        if not concept_set:
            joined = " ".join((s.get("snippet", "") or "") for s in (sources or []))
            for token in re.findall(r"[\u4e00-\u9fff]{2,8}", joined):
                if token not in concept_set:
                    concept_set.append(token)
                if len(concept_set) >= 8:
                    break
        concept_text = "、".join(concept_set[:5]) if concept_set else "未識別"

        # Pages from titles/snippets/content
        page_candidates: list[str] = []
        page_text_src = "\n".join(
            [
                content or "",
                *[(s.get("title", "") or "") for s in (sources or [])],
                *[(s.get("snippet", "") or "") for s in (sources or [])],
            ]
        )
        for p in re.findall(r"\bp\.\s*(\d+)\b", page_text_src, flags=re.IGNORECASE):
            if p not in page_candidates:
                page_candidates.append(p)
        if not page_candidates:
            for p in re.findall(r"第\s*(\d+)\s*頁", page_text_src):
                if p not in page_candidates:
                    page_candidates.append(p)
        page_text = ", ".join(page_candidates[:8]) if page_candidates else "未識別"

        return (
            f"[#6e7681]文章標題：[/]{escape(title_text)}\n"
            f"[#6e7681]關鍵字（概念）：[/]{escape(concept_text)}\n"
            f"[#6e7681]頁數：[/]{escape(page_text)}"
        )

    async def _detect_warning_sources(self, project_id: str, sources: list[dict]) -> list[str]:
        """Return warning document titles that match current sources."""
        # Simple cache: skip DB query if sources unchanged
        _sources_key = tuple(sorted((s.get("title", ""), s.get("file", "")) for s in (sources or [])))
        if hasattr(self, '_warning_cache_key') and self._warning_cache_key == _sources_key:
            return getattr(self, '_warning_cache_result', [])

        try:
            from paths import project_root
            from conversation.store import ConversationStore
            import os.path as _osp

            store = ConversationStore(db_path=project_root(project_id) / "conversations.db")
            docs = await store.list_documents(project_id)
            warned = []
            for d in docs:
                note = (d.get("note") or "").strip().lower()
                if "warning:" not in note:
                    continue
                warned.append(d)
            if not warned:
                return []

            matched: list[str] = []
            for src in sources or []:
                s_title = (src.get("title", "") or "").strip().lower()
                s_file = _osp.basename((src.get("file", "") or "").strip()).lower()
                for d in warned:
                    d_title = (d.get("title", "") or "").strip().lower()
                    d_file = _osp.basename((d.get("source_path", "") or "").strip()).lower()
                    if (s_title and d_title and (s_title in d_title or d_title in s_title)) or (
                        s_file and d_file and s_file == d_file
                    ):
                        t = d.get("title", "未知文獻")
                        if t not in matched:
                            matched.append(t)
            result = matched[:3]
            self._warning_cache_key = _sources_key
            self._warning_cache_result = result
            return result
        except Exception:
            return []

    async def _update_research_panel(
        self,
        content: str = "",
        *,
        status: str = "ready",
        sources: list[dict] | None = None,
        diagnostics: dict | None = None,
        query: str = "",
    ) -> None:
        display = ""
        if content and status not in ("loading", "error"):
            display = self._clean_rag_display(content) or content
            display = display[:1200] + ("…" if len(display) > 1200 else "")

        self._research_panel_state = {
            "status": status,
            "display_text": display,
            "raw_result": content or "",
            "sources": list(sources or []),
            "diagnostics": diagnostics or {},
            "query": query,
        }

        _pid = self.state.current_project["id"] if self.state.current_project else "default"
        warning_docs = await self._detect_warning_sources(_pid, list(sources or []))
        warning_prefix = ""
        if warning_docs:
            warning_prefix = (
                "[#d4a27a]（注意：部分來源文獻有警告，結果可能含重複節點影響）[/]\n"
                f"[dim #6e7681]來源：{escape('、'.join(warning_docs))}[/]\n\n"
            )

        try:
            panel = self.query_one("#research-content", Static)
            if status == "loading":
                # Show thinking trace if available in content
                if content and len(content) > 10 and "⟐" in content:
                    panel.update(f"[dim #8b949e]{escape(content)}[/]")
                else:
                    panel.update(f"[dim #6e7681]{escape(content or '檢索中…')}[/]")
            elif status == "error":
                panel.update(f"[#d4a27a]{escape(content or '搜尋失敗')}[/]")
            elif status == "empty":
                panel.update("[dim #484f58]無結果[/]")
            elif status == "degraded":
                prefix = "[#d4a27a]（部分匯入失敗，結果可能不完整）[/]\n\n"
                meta = self._build_research_meta(content, list(sources or []))
                thinking_block = self._build_thinking_block(diagnostics)
                panel.update(
                    prefix
                    + warning_prefix
                    + thinking_block
                    + meta
                    + "\n\n[dim #2a2a2a]"
                    + ("-" * 42)
                    + "[/]\n\n"
                    + f"[#c9d1d9]{escape(display or '無結果')}[/]"
                )
            else:
                meta = self._build_research_meta(content, list(sources or []))
                thinking_block = self._build_thinking_block(diagnostics)
                panel.update(
                    warning_prefix
                    + thinking_block
                    + meta
                    + "\n\n[dim #2a2a2a]"
                    + ("-" * 42)
                    + "[/]\n\n"
                    + f"[#c9d1d9]{escape(display or '無結果')}[/]"
                )
        except NoMatches:
            pass

        has_citable = bool((sources or []) or display)
        try:
            cite_link = self.query_one("#research-cite", Button)
            cite_link.disabled = not has_citable
            cite_link.label = "引用到對話"
        except NoMatches:
            pass
