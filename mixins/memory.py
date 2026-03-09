"""記憶管理相關方法。"""
from __future__ import annotations

from textual import work
from textual.css.query import NoMatches
from textual.widgets import Static

from memory.store import add_memory, get_memories


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

    def action_save_insight_from_chat(self, chatbox) -> None:
        ai_msg = chatbox._content
        user_msg = ""
        for m in reversed(self.messages):
            if m["role"] == "user":
                user_msg = m["content"]
                break
        if user_msg or ai_msg:
            self._prepare_insight(user_msg, ai_msg)

    def action_save_memory_from_chat(self, chatbox) -> None:
        from modals import MemorySaveModal
        content = chatbox._content
        mem_type = "thought" if chatbox.role == "user" else "reference"

        def on_confirm(result: dict | None) -> None:
            if result:
                self._do_save_memory(result)

        self.push_screen(MemorySaveModal(content, mem_type), callback=on_confirm)

    @work(exclusive=True)
    async def _do_save_memory(self, data: dict) -> None:
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
    async def _save_confirmed_insight(self, data: dict, source: str) -> None:
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

    @work(exclusive=False)
    async def _auto_extract_memories(self, user_text: str) -> None:
        try:
            from memory.thought_tracker import extract_memories
            items = await extract_memories(user_text, self._quick_llm_call)
            if not items:
                return
            for item in items:
                item["source"] = user_text[:80]
            self.state.pending_memories.extend(items)
            self._update_menu_bar()
        except Exception:
            pass

    def _update_menubar_pending_count(self) -> None:
        self._update_menu_bar()

    def action_confirm_pending_memories(self) -> None:
        from modals import MemoryConfirmModal
        if not self.state.pending_memories:
            self.notify("目前沒有待確認的記憶")
            return

        def on_dismiss(result) -> None:
            if result is None:
                self.state.pending_memories.clear()
                self._update_menu_bar()
                return
            self._save_confirmed_memories(result)

        self.push_screen(MemoryConfirmModal(self.state.pending_memories), callback=on_dismiss)

    @work(exclusive=True)
    async def _save_confirmed_memories(self, items: list) -> None:
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
        self._newly_saved_count = len(items)
        self._refresh_memory_panel()
        self._update_menu_bar()
        self.notify(f"已儲存 {len(items)} 條記憶")

        # 只對 insight 進行演變偵測
        insights = [i for i in items if isinstance(i, dict) and i.get("type") == "insight"]
        print(f"[DEBUG] _save_confirmed_memories: found {len(insights)} insights in {len(items)} items")
        if insights:
            print(f"[DEBUG] Triggering evolution check for: {insights[0]['content'][:50]}")
            self._check_insight_evolution(insights[0]["content"])

    @work(exclusive=False)
    async def _check_insight_evolution(self, new_insight: str) -> None:
        """背景任務：檢查洞見的思維演變，結果寫入記憶面板。"""
        print(f"[DEBUG] _check_insight_evolution called with: {new_insight[:50]}")
        try:
            from memory.thought_tracker import check_for_evolution

            db_path = self._memory_db_path()
            print(f"[DEBUG] db_path={db_path}")
            print(f"[DEBUG] Calling check_for_evolution...")
            result = await check_for_evolution(
                new_insight, self._quick_llm_call, db_path=db_path
            )
            print(f"[DEBUG] check_for_evolution result: {result}")

            if result is None or result.get("type") is None:
                print("[DEBUG] No evolution detected (result is None or type is None)")
                return

            if result["type"] == "evolution":
                type_label = "evolution"
            else:
                type_label = "contradiction"

            content = result["summary"]
            source_detail = f"舊：{result.get('old', '')}\n新：{result.get('new', '')}"

            print(f"[DEBUG] Saving evolution record: type={type_label}, content={content[:50]}")

            await add_memory(
                type=type_label,
                content=content,
                source=source_detail,
                topic="思維演變",
                category="思考方式",
                db_path=db_path,
            )

            print("[DEBUG] Evolution record saved, refreshing panel...")
            self._refresh_memory_panel()
            print("[DEBUG] Panel refreshed successfully")

        except Exception as e:
            print(f"[DEBUG] Exception in _check_insight_evolution: {e}")
            import traceback
            traceback.print_exc()
