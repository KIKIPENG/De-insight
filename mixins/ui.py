"""UI 相關方法：狀態更新、模式切換、設定、歷史等。"""
from __future__ import annotations

import subprocess
import time

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
        # Throttle: skip update if called within last 0.3s
        now = time.time()
        if hasattr(self, '_menu_bar_last_update') and now - self._menu_bar_last_update < 0.3:
            return
        self._menu_bar_last_update = now

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
                llm_calls=getattr(self, '_llm_call_count', 0),
            )
        except NoMatches:
            pass

    def _update_status(self) -> None:
        # Throttle: skip update if called within last 0.5s
        now = time.time()
        if hasattr(self, '_status_last_update') and now - self._status_last_update < 0.5:
            return
        self._status_last_update = now

        from widgets import MenuBar
        try:
            env = load_env()
            llm_model = env.get("LLM_MODEL", "")

            # RAG LLM 狀態：有設定就算 OK（雲端 API，不做 probe）
            rag_model = env.get("RAG_LLM_MODEL", "") or llm_model
            rag_llm_ok = bool(rag_model)

            # Vision 狀態：有模型設定就算 OK
            vision_ok = False
            try:
                from rag.image_store import _resolve_vision_config
                v_model, v_key, v_base = _resolve_vision_config()
                vision_ok = bool(v_model)
            except Exception:
                pass

            menu = self.query_one("#menu-bar", MenuBar)
            menu.set_system_status(
                llm_model=llm_model,
                llm_ok=bool(llm_model),
                rag_llm_ok=rag_llm_ok,
                vision_ok=vision_ok,
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
                # 設定變更後自動重置 embedding 快取，避免模型切換後用到舊的 embedding 函數
                try:
                    from memory.vectorstore import reset_embed_fn
                    reset_embed_fn()
                except Exception:
                    pass
                # 同時清除 LanceDB query cache
                try:
                    from rag.knowledge_graph import _QUERY_CACHE
                    _QUERY_CACHE.clear()
                except Exception:
                    pass

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
        from paths import __version__ as _ver
        welcome.border_title = f"[#d4a27a]◈[/] De-insight {_ver}"
        await container.mount(welcome)

    def action_show_help(self) -> None:
        from paths import __version__
        help_text = (
            f"# De-insight 說明\n\n"
            f"**版本:** {__version__}\n\n"
            f"**關於**\n\n"
            f"De-insight 是你的思想策展人。"
            f"懂得聆聽、說話直接、不急著給答案。"
            f"幫你把還說不清楚的東西說清楚。\n\n"
            f"**可用指令**\n\n"
            f"| 指令 | 功能 |\n"
            f"|------|------|\n"
            f"| `/new` | 新對話 |\n"
            f"| `/import` | 匯入 PDF/TXT/MD 或網頁到知識庫 |\n"
            f"| `/search` | 搜尋知識庫 |\n"
            f"| `/memory` | 管理記憶 |\n"
            f"| `/save` | 儲存當前對話的洞見 |\n"
            f"| `/reindex` | 重建記憶向量索引 |\n"
            f"| `/settings` | 開啟設定 |\n"
            f"| `/mode` | 切換感性/理性 |\n"
            f"| `/ragmode` | 切換知識檢索：快速/深度 |\n"
            f"| `/project` | 切換專案 |\n"
            f"| `/pending` | 記憶待確認 |\n"
            f"| `/caption` | 為圖片庫自動生成描述 |\n"
            f"| `/reindex-images` | 重建圖片向量索引 |\n"
            f"| `/focus` | 對焦評估──比對問題意識與最近記憶 |\n"
            f"| `/help` | 顯示此說明 |\n"
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
            import platform
            system = platform.system()
            if system == "Darwin":
                subprocess.run(["pbcopy"], input=content.encode(), check=True)
            elif system == "Linux":
                # 嘗試 xclip → xsel → wl-copy
                for cmd in (["xclip", "-selection", "clipboard"], ["xsel", "--clipboard", "--input"], ["wl-copy"]):
                    try:
                        subprocess.run(cmd, input=content.encode(), check=True)
                        break
                    except FileNotFoundError:
                        continue
                else:
                    raise FileNotFoundError("No clipboard tool found (xclip/xsel/wl-copy)")
            elif system == "Windows":
                subprocess.run(["clip"], input=content.encode(), check=True)
            else:
                raise RuntimeError(f"Unsupported platform: {system}")
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
        project_id = ""
        if self.state.current_project:
            project_id = self.state.current_project.get("id", "")
        if project_id:
            gallery_url = f"{self.api_base}/gallery?project_id={project_id}"
        else:
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

    def action_show_health(self) -> None:
        """開啟系統健康監測儀板。"""
        from modals import HealthDashboardModal

        def on_dismiss(result) -> None:
            pass

        self.push_screen(HealthDashboardModal(), callback=on_dismiss)

    def _scroll_to_bottom(self) -> None:
        self.query_one("#chat-scroll", VerticalScroll).scroll_end(animate=False)
        # 長對話優化：卸載超出可見範圍的舊 Chatbox 子 widget，減少 DOM 複雜度
        self._gc_old_chatboxes()

    _GC_KEEP_RECENT = 40  # 保留最近 N 個 Chatbox

    def _gc_old_chatboxes(self) -> None:
        """將最近 N 個以外的 Chatbox 子 widget 折疊為空 Static，降低 DOM 開銷。"""
        from widgets import Chatbox
        try:
            container = self.query_one("#messages", Vertical)
        except NoMatches:
            return
        chatboxes = list(container.query(Chatbox))
        if len(chatboxes) <= self._GC_KEEP_RECENT:
            return
        for box in chatboxes[:-self._GC_KEEP_RECENT]:
            if getattr(box, "_gc_collapsed", False):
                continue
            # 保留 border_title 讓使用者能看到是誰的訊息，但移除子 widget
            box._gc_collapsed = True
            box.remove_children()
            # 用簡短標記取代完整內容
            role_label = "You" if box.role == "user" else "De-insight"
            preview = (box._content or "")[:40].replace("\n", " ")
            box.mount(Static(f"[dim]…{role_label}: {preview}…[/dim]", classes="chatbox-body"))
