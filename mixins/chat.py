"""對話相關方法：送出、串流、互動提問等。"""
from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import unquote, urlparse

import httpx
from rich.markup import escape
from textual import work
from textual.containers import Vertical, VerticalScroll
from textual.css.query import NoMatches
from textual.widgets import OptionList, Static, TextArea
from textual.widgets.option_list import Option

from codex_client import codex_stream, is_codex_available
from settings import load_env


class ChatMixin:
    """對話相關方法。需要混入 App 才能使用。"""

    # ── slash commands ──

    SLASH_COMMANDS = {
        "/help": "show_help",
        "/new": "new_chat",
        "/import": "import_document",
        "/search": "search_knowledge",
        "/memory": "manage_memories",
        "/settings": "open_settings",
        "/mode": "toggle_mode",
        "/save": "save_insight_manual",
        "/reindex": "reindex_memories",
        "/ragmode": "toggle_rag_mode",
        "/project": "open_project_modal",
        "/pending": "confirm_pending_memories",
    }

    SLASH_HINTS: list[tuple[str, str]] = [
        ("/help", "顯示所有指令說明"),
        ("/new", "開啟新對話"),
        ("/import", "匯入 PDF 或網頁到知識庫"),
        ("/search", "搜尋知識庫"),
        ("/memory", "管理記憶 / 知識"),
        ("/settings", "開啟設定"),
        ("/mode", "切換感性 / 理性模式"),
        ("/save", "儲存當前對話的洞見"),
        ("/reindex", "重建記憶向量索引"),
        ("/ragmode", "切換知識檢索：快速 / 深度"),
        ("/project", "切換專案"),
        ("/pending", "記憶待確認"),
    ]

    def on_chat_input_submitted(self, event) -> None:
        from widgets import ChatInput
        ta = self.query_one("#chat-input", ChatInput)
        if ta.in_choice_mode:
            self._resolve_inline_choice()
            return
        self._submit_chat()

    def on_text_area_changed(self, event: TextArea.Changed) -> None:
        from widgets import ChatInput
        if event.text_area.id == "chat-input":
            ta = self.query_one("#chat-input", ChatInput)
            if not ta.in_choice_mode:
                self._update_slash_hints()

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        from widgets import ChatInput
        if event.option_list.id == "slash-hints":
            opt_id = event.option.id
            ta = self.query_one("#chat-input", ChatInput)
            ta.text = opt_id
            event.option_list.remove_class("-visible")
            ta.focus()
            self._submit_chat()

    def on_paste(self, event) -> None:
        text = event.text.strip()
        if not text:
            return
        clean = text.strip("'\"")
        is_url = clean.startswith("http://") or clean.startswith("https://")
        if is_url:
            event.prevent_default()
            self.notify("偵測到網址，開始匯入…")
            self._do_import(clean)
            return
        path = self._clean_dropped_path(text)
        if path:
            event.prevent_default()
            if path.lower().endswith(".pdf"):
                self.notify(f"偵測到 PDF，開始匯入…")
                self._do_import(path)
            else:
                from widgets import ChatInput
                inp = self.query_one("#chat-input", ChatInput)
                inp.text = path
                inp.focus()
                self.notify(f"檔案路徑已填入輸入框")

    def _update_slash_hints(self) -> None:
        from widgets import ChatInput
        try:
            ta = self.query_one("#chat-input", ChatInput)
            hints = self.query_one("#slash-hints", OptionList)
        except NoMatches:
            return
        text = ta.text.strip()
        if text.startswith("/") and "\n" not in text:
            query = text.lower()
            matches = [
                (cmd, desc) for cmd, desc in self.SLASH_HINTS
                if cmd.startswith(query)
            ]
            if matches:
                hints.clear_options()
                for cmd, desc in matches:
                    hints.add_option(Option(f"{cmd}  {desc}", id=cmd))
                hints.add_class("-visible")
                return
        hints.remove_class("-visible")

    def _clean_dropped_path(self, text: str) -> str | None:
        t = text.strip().strip("'\"")
        if not t:
            return None
        if t.startswith("file://"):
            t = unquote(urlparse(t).path)
        if "%20" in t:
            t = unquote(t)
        t = t.replace("\\ ", " ")
        if Path(t).exists() and Path(t).is_file():
            return t
        return None

    def fill_input(self, text: str) -> None:
        from widgets import ChatInput
        try:
            chat_input = self.query_one("#chat-input", ChatInput)
            chat_input.clear()
            chat_input.insert(text)
            chat_input.focus()
        except Exception:
            pass

    def _handle_slash_command(self, text: str) -> bool:
        cmd = text.split()[0].lower()
        if cmd in self.SLASH_COMMANDS:
            action = self.SLASH_COMMANDS[cmd]
            method = getattr(self, f"action_{action}", None)
            if method:
                method()
            return True
        if cmd.startswith("/"):
            self.notify(f"未知指令: {cmd}  輸入 /help 查看")
            return True
        return False

    @work(exclusive=True, group="submit_chat", thread=False)
    async def _submit_chat(self) -> None:
        from widgets import ChatInput, Chatbox, WelcomeBlock
        ta = self.query_one("#chat-input", ChatInput)
        text = ta.text.strip()
        if not text or self.is_loading:
            return

        ta.text = ""

        try:
            self.query_one("#slash-hints", OptionList).remove_class("-visible")
        except NoMatches:
            pass

        if self.state.current_interactive_block:
            self.state.current_interactive_block = None
            self.query_one("#input-frame", Vertical).border_title = "⌨ Message"
            self._send_as_user(text)
            return

        path = self._clean_dropped_path(text)
        if path and path.lower().endswith(".pdf"):
            self.notify("偵測到 PDF，開始匯入…")
            self._do_import(path)
            return
        clean_text = text.strip("'\"")
        if (clean_text.startswith("http://") or clean_text.startswith("https://")) and " " not in clean_text:
            self.notify("偵測到網址，開始匯入…")
            self._do_import(clean_text)
            return

        if text.startswith("/"):
            self._handle_slash_command(text)
            return

        user_content = self._build_user_content(text)
        self.messages.append({"role": "user", "content": user_content})

        if self.state.current_conversation_id is None:
            project_id = self.state.current_project["id"] if self.state.current_project else None
            self.state.current_conversation_id = await self._conv_store.create_conversation(project_id)
            title = text[:30].strip().replace("\n", " ")
            await self._conv_store.set_title(self.state.current_conversation_id, title)
        await self._conv_store.add_message(self.state.current_conversation_id, "user", text)

        container = self.query_one("#messages", Vertical)

        for w in container.query("WelcomeBlock"):
            await w.remove()

        await container.mount(Chatbox("user", text))
        self._scroll_to_bottom()
        self._stream_response()

    def _is_codex_cli_mode(self) -> bool:
        env = load_env()
        return env.get("LLM_MODEL", "").startswith("codex-cli/")

    def _is_direct_api_mode(self) -> bool:
        env = load_env()
        model = env.get("LLM_MODEL", "")
        base = env.get("OPENAI_API_BASE", "")
        if "openrouter.ai" in base:
            return False
        if "deepseek.com" in base:
            return False
        return model.startswith("openai/") and bool(base)

    def _resolve_api_key(self, env: dict) -> str:
        base = env.get("OPENAI_API_BASE", "")
        BASE_TO_KEY = {
            "openrouter.ai":    "OPENROUTER_API_KEY",
            "deepseek.com":     "DEEPSEEK_API_KEY",
            "minimaxi.chat":    "MINIMAX_API_KEY",
            "api.minimax.chat": "MINIMAX_API_KEY",
        }
        for domain, key_env in BASE_TO_KEY.items():
            if domain in base:
                val = env.get(key_env, "")
                if val:
                    return val
        return env.get("OPENAI_API_KEY", "")

    @staticmethod
    def _strip_reasoning(text: str) -> str:
        import re
        m = re.search(r'[\u4e00-\u9fff\u3000-\u303f\uff00-\uffef]', text)
        if not m:
            if re.match(r'\s*\*\*[A-Z]', text):
                return text
            return text
        cn_start = m.start()
        if cn_start == 0:
            return text
        prefix = text[:cn_start]
        if len(prefix) > 50 and prefix.count('.') >= 2:
            return text[cn_start:]
        return text

    @work(exclusive=True, group="stream_response")
    async def _stream_response(self) -> None:
        from widgets import Chatbox, ChatInput
        self.is_loading = True
        self.state.last_rag_sources = []
        container = self.query_one("#messages", Vertical)

        bubble = Chatbox("assistant", streaming=True)
        await container.mount(bubble)
        bubble.set_responding(True)
        self._scroll_to_bottom()

        full_content = ""

        try:
            if self._is_codex_cli_mode():
                # ── Codex CLI 路徑 ──
                if not is_codex_available():
                    raise RuntimeError("codex CLI 未安裝。執行: npm i -g @openai/codex")

                from prompts.foucault import get_system_prompt
                sys_prompt = get_system_prompt(self.mode)
                user_msg = self.messages[-1]["content"]

                # RAG 注入（統一呼叫）
                _, sys_addon = await self._inject_rag_context(
                    [{"role": "user", "content": user_msg}],
                    return_sys_addon=True,
                )
                if sys_addon:
                    sys_prompt += "\n\n" + sys_addon

                context = ""
                for m in self.messages[:-1]:
                    role = "You" if m["role"] == "user" else "De-insight"
                    context += f"{role}: {m['content']}\n\n"

                full_prompt = f"{context}You: {user_msg}" if context else user_msg

                env = load_env()
                codex_model = env.get("LLM_MODEL", "").removeprefix("codex-cli/")

                async for chunk in codex_stream(full_prompt, sys_prompt, model=codex_model):
                    full_content += chunk
                    bubble.stream_update(full_content)
                    self._scroll_to_bottom()
            elif self._is_direct_api_mode():
                # ── 直接 API 路徑（MiniMax 等 OpenAI-compatible）──
                import re as _re
                from prompts.foucault import get_system_prompt as _get_sp
                env = load_env()
                api_base = env.get("OPENAI_API_BASE", "")
                api_key = self._resolve_api_key(env)
                model = env.get("LLM_MODEL", "").removeprefix("openai/")

                sys_prompt = _get_sp(self.mode)

                # RAG 注入（統一呼叫）
                send_messages = await self._inject_rag_context(
                    [{"role": "system", "content": sys_prompt}] + self.messages
                )

                async with httpx.AsyncClient(timeout=120.0) as client:
                    async with client.stream(
                        "POST",
                        f"{api_base}/chat/completions",
                        headers={
                            "Authorization": f"Bearer {api_key}",
                            "Content-Type": "application/json",
                            "HTTP-Referer": "https://github.com/De-insight",
                            "X-Title": "De-insight",
                        },
                        json={"model": model, "messages": send_messages, "stream": True},
                    ) as response:
                        async for line in response.aiter_lines():
                            if not line.startswith("data: "):
                                continue
                            data_str = line[6:]
                            if data_str == "[DONE]":
                                break
                            try:
                                event = json.loads(data_str)
                                choices = event.get("choices", [])
                                if not choices:
                                    continue
                                delta = choices[0].get("delta", {}).get("content", "")
                                if delta:
                                    full_content += delta
                                    display = _re.sub(r"<think>[\s\S]*?</think>\s*", "", full_content)
                                    if "<think>" in full_content and "</think>" not in full_content:
                                        display = ""
                                    bubble.stream_update(display)
                                    self._scroll_to_bottom()
                            except (json.JSONDecodeError, ValueError):
                                pass
                full_content = _re.sub(r"<think>[\s\S]*?</think>\s*", "", full_content)
            else:
                # ── FastAPI 後端路徑 ──
                send_messages = await self._inject_rag_context(self.messages)
                async with httpx.AsyncClient(timeout=120.0) as client:
                    async with client.stream(
                        "POST",
                        f"{self.api_base}/api/chat",
                        json={"messages": send_messages, "mode": self.mode},
                    ) as response:
                        async for line in response.aiter_lines():
                            if line.startswith("0:"):
                                try:
                                    chunk = json.loads(line[2:])
                                    full_content += chunk
                                    bubble.stream_update(full_content)
                                    self._scroll_to_bottom()
                                except (json.JSONDecodeError, ValueError):
                                    pass
                            elif line.startswith("3:"):
                                try:
                                    err = json.loads(line[2:])
                                    raise RuntimeError(err.get("error", "未知錯誤"))
                                except json.JSONDecodeError:
                                    raise RuntimeError(line[2:])
                            elif line.startswith("d:"):
                                break

            full_content = self._strip_reasoning(full_content)

            import re as _re_interactive
            from interaction.prompt_parser import parse_interactive_blocks
            clean_content, interactive_blocks = parse_interactive_blocks(full_content)

            if _re_interactive.search(r'<<\w+', clean_content):
                self.notify("策展人格式未完整解析", severity="warning", timeout=4)

            if interactive_blocks:
                bubble.set_responding(False)
                bubble.stream_update(clean_content)
                await bubble.finalize_stream()
                self.messages.append({"role": "assistant", "content": clean_content})
                if self.state.current_conversation_id:
                    await self._conv_store.add_message(
                        self.state.current_conversation_id, "assistant", clean_content)
                if self.state.interactive_depth < 3:
                    self._handle_interactive_blocks(interactive_blocks)
                else:
                    self.state.interactive_depth = 0
                    self.notify("互動提問深度上限，請直接輸入", timeout=3)
            else:
                bubble.set_responding(False)
                bubble.stream_update(full_content)
                await bubble.finalize_stream()
                self.messages.append({"role": "assistant", "content": full_content})
                if self.state.current_conversation_id:
                    await self._conv_store.add_message(
                        self.state.current_conversation_id, "assistant", full_content)
                self.state.interactive_depth = 0

            raw_content = self.messages[-2]["content"] if len(self.messages) >= 2 else ""
            user_text = self._extract_text(raw_content)
            if user_text and len(user_text.strip()) >= 15:
                self._auto_extract_memories(user_text)

        except httpx.ConnectError:
            bubble.set_responding(False)
            bubble.stream_update(
                "**連線錯誤** — 後端未啟動\n\n"
                "cd backend && .venv/bin/python3 -m uvicorn main:app --reload"
            )
            await bubble.finalize_stream()
        except Exception as e:
            bubble.set_responding(False)
            bubble.stream_update(f"**錯誤** — {escape(str(e))}")
            await bubble.finalize_stream()
        finally:
            self.is_loading = False
            self._update_status()
            self.query_one("#chat-input", ChatInput).focus()

    # ── interactive prompt handling ──

    def _handle_interactive_blocks(self, blocks: list) -> None:
        from widgets import ChatInput
        if not blocks:
            return
        block = blocks[0]
        self.state.current_interactive_block = block

        ta = self.query_one("#chat-input", ChatInput)
        frame = self.query_one("#input-frame", Vertical)
        frame.border_title = f"◇ {block.prompt}"

        if block.type == 'confirm':
            ta.set_choices(["是，繼續", "不，我還想調整"])
        elif block.type == 'select':
            ta.set_choices(block.choices)
        elif block.type == 'multi':
            ta.set_choices(block.choices)
        elif block.type == 'input':
            pass

        ta.focus()

    def _resolve_inline_choice(self) -> None:
        from widgets import ChatInput
        ta = self.query_one("#chat-input", ChatInput)
        if not ta.in_choice_mode:
            return
        idx = ta._choice_idx
        choices = ta._choices
        text = choices[idx] if idx < len(choices) else ""
        ta.clear_choices()
        self.state.current_interactive_block = None
        self.query_one("#input-frame", Vertical).border_title = "⌨ Message"
        if text:
            self._send_as_user(text)

    @work(exclusive=True, group="send_as_user")
    async def _send_as_user(self, text: str) -> None:
        from widgets import Chatbox
        if not text.strip():
            return
        self.state.interactive_depth += 1
        container = self.query_one("#messages", Vertical)
        await container.mount(Chatbox("user", text))
        self._scroll_to_bottom()
        self.messages.append({'role': 'user', 'content': text})
        if self.state.current_conversation_id:
            await self._conv_store.add_message(
                self.state.current_conversation_id, "user", text)
        self._stream_response()

    async def _quick_llm_call(self, prompt: str, max_tokens: int = 500) -> str:
        env = load_env()
        model = env.get("LLM_MODEL", "ollama/llama3.2")

        if model.startswith("codex-cli/"):
            codex_model = model.removeprefix("codex-cli/")
            result = ""
            async for chunk in codex_stream(prompt, model=codex_model):
                result += chunk
            return result

        if model.startswith("codex/"):
            model = "openai/" + model.removeprefix("codex/")

        import litellm
        resp = await litellm.acompletion(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content or ""

    def _build_user_content(self, text: str):
        """組合使用者文字 + pending_images 為 multimodal content。
        若無圖片，回傳純文字字串；有圖片則回傳 OpenAI vision 格式 list。
        """
        pending = getattr(self.state, "pending_images", None) or []
        if not pending:
            return text

        parts = [{"type": "text", "text": text}]
        for img in pending:
            import base64
            img_path = Path(img) if isinstance(img, str) else Path(img.get("path", ""))
            if img_path.exists():
                data = base64.b64encode(img_path.read_bytes()).decode()
                suffix = img_path.suffix.lower().lstrip(".")
                mime = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png",
                        "gif": "image/gif", "webp": "image/webp"}.get(suffix, "image/jpeg")
                parts.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime};base64,{data}"},
                })
        # clear pending
        self.state.pending_images = []
        self._update_menu_bar()
        return parts

    @staticmethod
    def _extract_text(content) -> str:
        """從 multipart content 取出純文字。"""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return " ".join(
                p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text"
            )
        return str(content)

    async def _inject_rag_context(
        self,
        messages: list[dict],
        return_sys_addon: bool = False,
    ) -> list[dict] | tuple[list[dict], str]:
        """
        return_sys_addon=False（預設）：回傳注入後的 messages（Direct API / FastAPI 用）
        return_sys_addon=True：回傳 (messages, sys_addon_str)（Codex CLI 用）
        """
        raw_content = messages[-1]["content"] if messages else ""
        user_msg = self._extract_text(raw_content)
        if not user_msg:
            return (messages, "") if return_sys_addon else messages

        augmented = list(messages)
        insert_idx = 1 if (augmented and augmented[0]["role"] == "system") else 0
        sys_addon_parts = []

        # 1. 記憶向量搜尋
        _lancedb_dir = None
        if self.state.current_project:
            from paths import project_root
            _lancedb_dir = project_root(self.state.current_project["id"]) / "lancedb"
        try:
            from memory.vectorstore import search_similar, has_index
            if has_index(lancedb_dir=_lancedb_dir):
                mem_results = await search_similar(user_msg, limit=3, lancedb_dir=_lancedb_dir)
                if mem_results:
                    mem_lines = "\n".join(
                        f"- [{m['type']}] {m['content']}" + (f" (#{m['topic']})" if m.get('topic') else "")
                        for m in mem_results
                    )
                    context_text = f"使用者過去的想法（語意相關）：\n{mem_lines}"
                    if return_sys_addon:
                        sys_addon_parts.append(context_text)
                    else:
                        augmented.insert(insert_idx, {"role": "system", "content": context_text})
                        insert_idx += 1
        except Exception as e:
            self.log.warning(f"RAG inject memory failed: {e}")

        # 2. 圖片知識庫檢索（EMBED_MODE=local 時）
        try:
            from memory.vectorstore import _is_local_mode
            if _is_local_mode() and self.state.current_project:
                from rag.image_store import search_images
                img_results = await search_images(
                    self.state.current_project["id"], user_msg, limit=3,
                )
                if img_results:
                    img_lines = "\n".join(
                        f"- [圖片] {r['filename']}: {r['caption']}" + (f" (tags: {r['tags']})" if r.get('tags') else "")
                        for r in img_results
                        if r.get("score", 0) > 0.3
                    )
                    if img_lines:
                        context_text = f"相關圖片（可引用路徑或描述）：\n{img_lines}"
                        if return_sys_addon:
                            sys_addon_parts.append(context_text)
                        else:
                            augmented.insert(insert_idx, {"role": "system", "content": context_text})
                            insert_idx += 1
        except Exception as e:
            self.log.warning(f"RAG inject images failed: {e}")

        # 3. 知識庫 RAG
        _pid = self.state.current_project["id"] if self.state.current_project else "default"
        try:
            from rag.knowledge_graph import (
                query_knowledge,
                has_knowledge,
                _clean_rag_chunk,
                _is_no_context_result,
            )
            if has_knowledge(project_id=_pid):
                is_deep = self.rag_mode == "deep"
                result, sources = await query_knowledge(
                    user_msg,
                    mode="hybrid" if is_deep else "naive",
                    context_only=not is_deep,
                    project_id=_pid,
                )
                # Guard: only inject if we got real content (not no-context)
                if result and len(result.strip()) > 10 and not _is_no_context_result(result):
                    cleaned_result = _clean_rag_chunk(result)
                    # Only inject cleaned content — never raw JSON/Document Chunks wrappers
                    if cleaned_result and len(cleaned_result.strip()) > 10:
                        await self._update_research_panel(result)
                        if sources:
                            self.state.last_rag_sources = sources
                        context_text = f"知識庫相關資訊：\n{cleaned_result[:2000]}\n\n（以上為參考資料，請務必用繁體中文回覆使用者。）"
                        if return_sys_addon:
                            sys_addon_parts.append(context_text)
                        else:
                            augmented.insert(insert_idx, {"role": "system", "content": context_text})
        except Exception as e:
            self.log.warning(f"RAG inject knowledge failed: {e}")

        if return_sys_addon:
            addon = "\n\n".join(sys_addon_parts)
            if addon:
                addon += "\n\n（以上為參考資料，請務必用繁體中文回覆使用者。）"
            return augmented, addon

        return augmented
