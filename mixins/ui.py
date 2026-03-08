"""UI 相關方法：狀態更新、模式切換、設定、歷史等。"""
from __future__ import annotations

import subprocess

import httpx
from textual import work
from textual.containers import Vertical, VerticalScroll
from textual.css.query import NoMatches
from textual.widgets import Static

from settings import SettingsScreen, load_env


class UIMixin:
    """UI 相關方法。需要混入 App 才能使用。"""

    def watch_mode(self) -> None:
        if self.is_mounted:
            self._update_menu_bar()
            self._update_status()

    def watch_is_loading(self) -> None:
        if self.is_mounted:
            self._update_status()

    def _update_menu_bar(self) -> None:
        from widgets import MenuBar
        try:
            env = load_env()
            model = env.get("LLM_MODEL", "?")
            menu = self.query_one("#menu-bar", MenuBar)
            mem_count = getattr(self, "_cached_memory_count", 0)
            project_id = self.state.current_project["id"] if self.state.current_project else "default"
            try:
                from rag.knowledge_graph import has_knowledge
                has_kg = has_knowledge(project_id=project_id)
            except Exception:
                has_kg = False
            project_name = self.state.current_project["name"] if self.state.current_project else None
            pending_count = len(self.state.pending_memories)
            gallery_selected = len(getattr(self.state, "pending_images", None) or [])
            menu.set_state(
                mode=self.mode,
                model=model,
                memory_count=mem_count,
                has_kg=has_kg,
                rag_mode=self.rag_mode,
                project_name=project_name,
                pending_count=pending_count,
                gallery_selected=gallery_selected,
            )
        except NoMatches:
            pass

    def _update_status(self) -> None:
        from widgets import MenuBar
        try:
            env = load_env()
            llm_model = env.get("LLM_MODEL", "")
            menu = self.query_one("#menu-bar", MenuBar)
            menu.set_system_status(
                llm_model=llm_model,
                llm_ok=bool(llm_model),
                embed_label=getattr(self, "_embed_label", "jina-embeddings-v4"),
                embed_ok=getattr(self, "_embed_ok", False),
            )
        except NoMatches:
            pass

    def action_from_menu(self, action: str) -> None:
        method = getattr(self, f"action_{action}", None)
        if method:
            method()

    def action_toggle_mode(self) -> None:
        from widgets import MenuBar
        self.mode = "rational" if self.mode == "emotional" else "emotional"
        try:
            menu = self.query_one("#menu-bar", MenuBar)
            menu.styles.background = "#3d2a1a"
            self.set_timer(0.25, lambda: setattr(menu.styles, "background", "#0d0d0d"))
        except Exception:
            pass

    def action_open_settings(self) -> None:
        def on_dismiss(result: str | None) -> None:
            if result:
                self._update_menu_bar()
                self._update_status()
                self._reload_backend_env()

        self.push_screen(SettingsScreen(), callback=on_dismiss)

    @work(thread=True)
    async def _reload_backend_env(self) -> None:
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.post(f"{self.api_base}/api/reload-env")
                if resp.status_code == 200:
                    data = resp.json()
                    model = data.get("model", "?")
                    self.notify(f"⚙ 設定已儲存 ∙ 模型: {model}")
                else:
                    self.notify("⚙ 設定已儲存（後端重載失敗，請手動重啟）")
        except Exception:
            self.notify("⚙ 設定已儲存到 .env（後端未運行）")

    def action_new_chat(self) -> None:
        from widgets import WelcomeBlock, ChatInput
        self.messages.clear()
        self.state.current_conversation_id = None
        container = self.query_one("#messages", Vertical)
        container.remove_children()
        self.call_after_refresh(self._mount_welcome)
        self.query_one("#chat-input", ChatInput).focus()

    async def _mount_welcome(self) -> None:
        from widgets import WelcomeBlock
        container = self.query_one("#messages", Vertical)
        welcome = WelcomeBlock()
        welcome.border_title = "[#d4a27a]◈[/] De-insight v0.8"
        await container.mount(welcome)

    def action_show_help(self) -> None:
        help_text = (
            "**可用指令**\n\n"
            "| 指令 | 功能 |\n"
            "|------|------|\n"
            "| `/new` | 新對話 |\n"
            "| `/import` | 匯入 PDF 或網頁到知識庫 |\n"
            "| `/search` | 搜尋知識庫 |\n"
            "| `/memory` | 管理記憶 |\n"
            "| `/save` | 儲存當前對話的洞見 |\n"
            "| `/reindex` | 重建記憶向量索引 |\n"
            "| `/settings` | 開啟設定 |\n"
            "| `/mode` | 切換感性/理性 |\n"
            "| `/ragmode` | 切換知識檢索：快速/深度 |\n"
            "| `/project` | 切換專案 |\n"
            "| `/pending` | 記憶待確認 |\n"
            "| `/help` | 顯示此說明 |\n"
        )
        self.call_after_refresh(lambda: self._show_system_message(help_text))

    def action_close_modals(self) -> None:
        while len(self.screen_stack) > 1:
            self.pop_screen()

    def action_copy_chatbox(self, chatbox) -> None:
        content = chatbox._content
        if not content:
            return
        try:
            subprocess.run(["pbcopy"], input=content.encode(), check=True)
            self.notify("已複製到剪貼板")
        except Exception:
            self.notify("複製失敗，請手動選取")

    async def _show_system_message(self, content: str) -> None:
        from widgets import Chatbox
        container = self.query_one("#messages", Vertical)
        box = Chatbox("assistant", content, system=True)
        await container.mount(box)
        box.styles.border = ("round", "#3a3a3a")
        self._scroll_to_bottom()

    def _start_discussion_from_memory(self, memory_content: str) -> None:
        from widgets import ChatInput
        prompt = f"我之前有一個想法：「{memory_content}」\n\n我想進一步討論這個概念。"
        inp = self.query_one("#chat-input", ChatInput)
        inp.text = prompt
        self._submit_chat()

    @work(exclusive=True, thread=False)
    async def _load_conversation(self, conversation_id: str) -> None:
        from widgets import Chatbox
        conv = await self._conv_store.get_conversation(conversation_id)
        if not conv:
            return
        current_pid = self.state.current_project["id"] if self.state.current_project else None
        if current_pid and conv.get("project_id") != current_pid:
            self.notify("這筆對話不屬於目前專案", severity="warning")
            return
        messages = await self._conv_store.get_messages(conversation_id)
        if not messages:
            return
        self.state.current_conversation_id = conversation_id
        self.messages = list(messages)
        container = self.query_one("#messages", Vertical)
        await container.remove_children()
        for m in messages:
            await container.mount(Chatbox(m["role"], m["content"]))
        self._scroll_to_bottom()

    def action_open_gallery(self) -> None:
        """在瀏覽器開啟圖片庫。"""
        import webbrowser
        gallery_url = f"{self.api_base}/gallery"
        try:
            health_url = f"{self.api_base}/api/health"
            r = httpx.get(health_url, timeout=1.5)
            if r.status_code >= 400:
                raise RuntimeError(f"health check failed: {r.status_code}")
            webbrowser.open(gallery_url)
            self.notify("圖片庫已在瀏覽器開啟")
        except Exception:
            self.notify("後端未啟動：cd backend && .venv/bin/python -m uvicorn main:app --reload")

    def _scroll_to_bottom(self) -> None:
        self.query_one("#chat-scroll", VerticalScroll).scroll_end(animate=False)
