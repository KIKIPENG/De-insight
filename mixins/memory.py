"""記憶管理相關方法。"""
from __future__ import annotations

from textual import work
from textual.css.query import NoMatches
from textual.widgets import Static

from memory.store import add_memory, get_memories, save_pending_memories, load_pending_memories, clear_pending_memories


class MemoryMixin:
    """記憶管理相關方法。需要混入 App 才能使用。"""

    def _memory_db_path(self) -> "Path | None":
        if self.state.current_project:
            from paths import project_root
            return project_root(self.state.current_project["id"]) / "memories.db"
        return None

    def _lancedb_dir(self) -> "Path | None":
        if self.state.current_project:
            from paths import project_root
            return project_root(self.state.current_project["id"]) / "lancedb"
        return None

    @work(exclusive=False)
    async def _reindex_pending_memories(self) -> None:
        try:
            from memory.store import reindex_pending
            db_path = self._memory_db_path()
            lancedb_dir = self._lancedb_dir()
            count = await reindex_pending(db_path=db_path, lancedb_dir=lancedb_dir)
            if count > 0:
                self.log.info(f"reindex_pending: fixed {count} memories")
        except Exception as e:
            self.log.warning(f"reindex_pending failed: {e}")

    def action_manage_memories(self) -> None:
        from modals import MemoryManageModal

        def on_dismiss(result: str | None) -> None:
            self._refresh_memory_panel()
            if result and result.startswith("discuss:"):
                content = result.removeprefix("discuss:")
                self._start_discussion_from_memory(content)
            elif result and result.startswith("__insert__:"):
                content = result[len("__insert__:"):]
                self.action_close_modals()
                self.fill_input(content)

        _pid = self.state.current_project["id"] if self.state.current_project else None
        self.push_screen(MemoryManageModal(project_id=_pid), callback=on_dismiss)

    @work(exclusive=True, group="memory_panel")
    async def _refresh_memory_panel(self) -> None:
        from panels import MemoryPanel, MemoryItem
        try:
            panel = self.query_one("#memory-panel", MemoryPanel)
        except NoMatches:
            return
        db_path = self._memory_db_path()
        await panel.remove_children()
        mems = await get_memories(limit=10, db_path=db_path)
        try:
            from memory.store import get_memory_stats
            stats = await get_memory_stats(db_path=db_path)
            self.state.cached_memory_count = stats.get("total", 0)
        except Exception:
            self.state.cached_memory_count = len(mems) if mems else 0
        self._update_menu_bar()
        if not mems:
            await panel.mount(
                Static("[dim #484f58]對話後自動記錄洞見[/]", id="memory-content")
            )
            return
        highlight_count = getattr(self, "_newly_saved_count", 0)
        self._newly_saved_count = 0
        for i, m in enumerate(mems):
            item = MemoryItem(m)
            await panel.mount(item)
            if (
                m.get("type") == "insight"
                and m.get("content", "") in getattr(self, "_focus_tagged_insights", set())
            ):
                item.mark_focus_tagged()
            if i < highlight_count:
                item.add_class("-new")
                self.set_timer(1.5, lambda w=item: w.remove_class("-new") if w.is_mounted else None)

    def action_reindex_memories(self) -> None:
        self._do_reindex()

    @work(exclusive=True)
    async def _do_reindex(self) -> None:
        lancedb_dir = self._lancedb_dir()
        db_path = self._memory_db_path()
        self.notify("正在重建記憶向量索引...")
        try:
            from memory.vectorstore import index_all_memories
            count = await index_all_memories(lancedb_dir=lancedb_dir, db_path=db_path)
            self.notify(f"索引完成：{count} 條記憶已向量化")
            self._update_menu_bar()
        except Exception as e:
            self.notify(f"索引失敗: {e}")

    def action_save_insight_manual(self) -> None:
        if len(self.messages) < 2:
            self.notify("至少需要一輪對話才能儲存洞見")
            return
        user_msg = ""
        ai_msg = ""
        for m in reversed(self.messages):
            if m["role"] == "assistant" and not ai_msg:
                ai_msg = m["content"]
            elif m["role"] == "user" and not user_msg:
                user_msg = m["content"]
            if user_msg and ai_msg:
                break
        if user_msg:
            self._prepare_insight(user_msg, ai_msg)

    def action_save_insight_from_chat(self, chatbox: object) -> None:
        ai_msg = chatbox._content
        user_msg = ""
        for m in reversed(self.messages):
            if m["role"] == "user":
                user_msg = m["content"]
                break
        if user_msg or ai_msg:
            self._prepare_insight(user_msg, ai_msg)

    def action_save_memory_from_chat(self, chatbox: object) -> None:
        from modals import MemorySaveModal
        content = chatbox._content
        mem_type = "thought" if chatbox.role == "user" else "reference"

        def on_confirm(result: dict | None) -> None:
            if result:
                self._do_save_memory(result)

        self.push_screen(MemorySaveModal(content, mem_type), callback=on_confirm)

    @work(exclusive=True)
    async def _do_save_memory(self, data: dict[str, str]) -> None:
        try:
            db_path = self._memory_db_path()
            await add_memory(
                type=data["type"],
                content=data["content"],
                topic=data.get("topic", ""),
                source="manual",
                db_path=db_path,
            )
            self.notify(f"已儲存記憶：{data['content'][:30]}…")
            self._refresh_memory_panel()
        except Exception as e:
            self.notify(f"儲存失敗: {e}")

    @work(exclusive=True)
    async def _prepare_insight(self, user_msg: str, ai_msg: str) -> None:
        from modals import InsightConfirmModal
        self.notify("整理洞見中…")
        try:
            prompt = (
                "以下是一段對話。請從使用者的發言中提取一個核心洞見，"
                "用一到兩句精簡的繁體中文概括。只回傳洞見本身，不要加任何解釋。\n\n"
                f"使用者：{user_msg}\n\nAI：{ai_msg}"
            )
            draft = await self._quick_llm_call(prompt, max_tokens=200)
            draft = draft.strip()

            if not draft:
                self.notify("無法提取洞見")
                return

            source = user_msg[:100]

            def on_confirm(result: dict | None) -> None:
                if result:
                    self._save_confirmed_insight(result, source)

            self.app.push_screen(
                InsightConfirmModal(draft, "insight"),
                callback=on_confirm,
            )
        except Exception as e:
            self.notify(f"整理失敗: {e}")

    @work()
    async def _save_confirmed_insight(self, data: dict[str, str], source: str) -> None:
        db_path = self._memory_db_path()
        await add_memory(
            type=data["type"],
            content=data["content"],
            source=source,
            db_path=db_path,
        )
        # Invalidate insight profile cache so future queries reflect new insights
        try:
            from rag.insight_profile import invalidate_cache
            _pid = self.state.current_project["id"] if self.state.current_project else "default"
            invalidate_cache(_pid)
        except Exception:
            pass
        self.notify(f"已儲存 [{data['type']}]")
        self._refresh_memory_panel()
        if data.get("type") == "insight" and data.get("content"):
            await self._tag_focus_relevance(data["content"])

    @work(exclusive=True)
    async def _load_pending_memories_from_db(self) -> None:
        """Load pending memories from database on startup."""
        try:
            db_path = self._memory_db_path()
            items = await load_pending_memories(db_path=db_path)
            if items:
                self.state.pending_memories.extend(items)
                self._update_menu_bar()
        except Exception as e:
            import logging
            logging.getLogger("de-insight").warning("待確認記憶載入失敗: %s", e, exc_info=True)

    @work(exclusive=True, group="auto_extract_mem")
    async def _auto_extract_memories(self, user_text: str) -> None:
        # 冷卻：距上次抽取不到 10 秒則跳過，避免快速連打時 API storm
        import time as _time
        _last = getattr(self, "_last_extract_time", 0.0)
        _now = _time.time()
        if _now - _last < 10.0:
            return
        self._last_extract_time = _now
        try:
            from memory.thought_tracker import extract_memories
            items = await extract_memories(user_text, self._quick_llm_call)
            if not items:
                return
            for item in items:
                item["source"] = user_text[:80]
            self.state.pending_memories.extend(items)
            # 上限 50 筆，超出時丟棄最舊的
            _MAX_PENDING = 50
            if len(self.state.pending_memories) > _MAX_PENDING:
                self.state.pending_memories = self.state.pending_memories[-_MAX_PENDING:]
            # Also save to database for persistence across restarts
            db_path = self._memory_db_path()
            await save_pending_memories(items, db_path=db_path)
            self._update_menu_bar()
            # 播放記憶發現動畫
            await self._show_memory_discovery_animation()
        except Exception as e:
            import logging
            logging.getLogger("de-insight").warning("記憶抽取失敗: %s", e, exc_info=True)
            # Show MenuBar notification for memory extraction failure
            try:
                from widgets import MenuBar
                menu = self.query_one("#menu-bar", MenuBar)
                menu.show_message("記憶抽取暫時不可用", severity="warning", timeout=5)
            except Exception:
                pass

    async def _show_memory_discovery_animation(self) -> None:
        """在對話區域播放記憶發現動畫。"""
        try:
            from textual.containers import Vertical
            from utils.animations import AnimatedStatic, MEMORY_FRAMES

            container = self.query_one("#messages", Vertical)
            anim = AnimatedStatic(classes="memory-discovery-anim")
            await container.mount(anim)
            self._scroll_to_bottom()

            # 非循環播放，播完後自動移除
            def _on_done():
                anim.set_timer(2.0, anim.remove)

            anim.start(MEMORY_FRAMES, interval=0.4, loop=False, on_complete=_on_done)
        except Exception:
            pass  # 動畫失敗不影響主流程

    def _update_menubar_pending_count(self) -> None:
        self._update_menu_bar()

    def action_confirm_pending_memories(self) -> None:
        from modals import MemoryConfirmModal
        if not self.state.pending_memories:
            self.notify("目前沒有待確認的記憶")
            return

        def on_dismiss(result: list[dict] | None) -> None:
            if result is None:
                # User cancelled — keep pending memories in state AND database (don't clear)
                self._update_menu_bar()
                return
            self._save_confirmed_memories(result)

        self.push_screen(MemoryConfirmModal(self.state.pending_memories), callback=on_dismiss)

    @work(exclusive=True)
    async def _save_confirmed_memories(self, items: list[dict]) -> None:
        db_path = self._memory_db_path()
        for item in items:
            if isinstance(item, dict):
                await add_memory(
                    type=item["type"],
                    content=item["content"],
                    source=item.get("source", ""),
                    topic=item.get("topic", ""),
                    category=item.get("category", ""),
                    db_path=db_path,
                )
        self.state.pending_memories.clear()
        # Clear pending memories from database after confirming
        await clear_pending_memories(db_path=db_path)
        self._newly_saved_count = len(items)
        self._refresh_memory_panel()
        self._update_menu_bar()
        self.notify(f"已儲存 {len(items)} 條記憶")

        # 只對 insight 進行演變偵測
        insights = [i for i in items if isinstance(i, dict) and i.get("type") == "insight"]
        if insights:
            self._check_insight_evolution(insights[0]["content"])
            for insight in insights:
                content = (insight.get("content") or "").strip()
                if content:
                    await self._tag_focus_relevance(content)

        # 偏好萃取：每存入一批記憶後，檢查是否需要更新視覺偏好
        self._maybe_update_visual_preference()

    async def _tag_focus_relevance(self, insight_content: str) -> None:
        """比對新洞見與問題意識，相關則標記；無關累積達門檻時排隊對焦提問。"""
        try:
            if not self.state.current_project:
                return
            from focus import load_focus
            from paths import project_root as get_project_root

            pid = self.state.current_project["id"]
            fields = load_focus(get_project_root(pid))
            focus_question = fields.get("問題意識", "").strip()
            if not focus_question:
                return

            prompt = (
                f"問題意識：{focus_question}\n"
                f"洞見：{insight_content}\n\n"
                "這條洞見和問題意識有沒有直接關聯？只回答 yes 或 no。"
            )
            result = (await self._quick_llm_call(prompt, max_tokens=8)).strip().lower()
            if "yes" in result:
                self._unrelated_insight_count = 0
                self._focus_tagged_insights.add(insight_content)
                self._mark_latest_insight_tagged()
            else:
                self._unrelated_insight_count += 1
                if self._unrelated_insight_count >= 5:
                    self._unrelated_insight_count = 0
                    self._schedule_focus_nudge()
        except Exception:
            pass

    def _mark_latest_insight_tagged(self) -> None:
        """在記憶面板最新的 insight 條目加上 focus 標記。"""
        try:
            from panels import MemoryItem
            for item in self.query(MemoryItem):
                if getattr(item, "_memory_type", None) == "insight":
                    item.mark_focus_tagged()
                    break
        except Exception:
            pass

    def _schedule_focus_nudge(self) -> None:
        self._pending_focus_nudge = True

    @work(exclusive=False)
    async def _maybe_update_visual_preference(self) -> None:
        """背景：圖片數 >= 5 且距上次萃取超過 5 張時，更新視覺偏好。"""
        try:
            if not self.state.current_project:
                return
            pid = self.state.current_project["id"]
            db_path = self._memory_db_path()

            from rag.image_store import get_image_count
            count = await get_image_count(pid)
            if count < 5:
                return

            # 檢查上次萃取時的圖片數
            prefs = await get_memories(type="preference", limit=1, db_path=db_path)
            if prefs:
                # source 欄位存了上次的圖片數
                last_count = 0
                try:
                    last_count = int(prefs[0].get("source", "0"))
                except (ValueError, TypeError):
                    pass
                if count - last_count < 5:
                    return  # 新增不到 5 張，不更新

            from rag.image_store import extract_visual_preference
            result = await extract_visual_preference(pid, llm_call=self._quick_llm_call)
            if not result or not result.get("summary"):
                return

            await add_memory(
                type="preference",
                content=result["summary"],
                source=str(count),  # 記錄當時的圖片數
                topic="美學偏好",
                category="美學偏好",
                db_path=db_path,
            )
            self._refresh_memory_panel()
            self.notify("◇ 視覺偏好已更新")

            # 交叉偵測：視覺偏好 vs 文字洞見
            try:
                from memory.thought_tracker import check_cross_modal
                cross = await check_cross_modal(
                    result["summary"], self._quick_llm_call, db_path=db_path,
                )
                if cross and cross.get("type") == "cross_modal":
                    await add_memory(
                        type="contradiction",
                        content=cross["summary"],
                        source=f"視覺：{cross.get('visual', '')}\n文字：{cross.get('textual', '')}",
                        topic="跨模態矛盾",
                        category="美學偏好",
                        db_path=db_path,
                    )
                    self._refresh_memory_panel()
                    self.notify(f"⚡ {cross['summary']}", timeout=8)
            except Exception:
                pass  # 交叉偵測失敗不影響主流程
        except Exception:
            pass  # 偏好萃取失敗不影響主流程

    @work(exclusive=False)
    async def _check_insight_evolution(self, new_insight: str) -> None:
        """背景任務：檢查洞見的思維演變，結果寫入記憶面板。"""
        try:
            from memory.thought_tracker import check_for_evolution

            db_path = self._memory_db_path()
            result = await check_for_evolution(
                new_insight, self._quick_llm_call, db_path=db_path
            )

            if result is None or result.get("type") is None:
                return

            type_label = result["type"]  # "evolution" or "contradiction"
            content = result["summary"]
            source_detail = f"舊：{result.get('old', '')}\n新：{result.get('new', '')}"

            await add_memory(
                type=type_label,
                content=content,
                source=source_detail,
                topic="思維演變",
                category="思考方式",
                db_path=db_path,
            )

            self._refresh_memory_panel()

        except Exception:
            pass
