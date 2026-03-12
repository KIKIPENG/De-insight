"""對話相關方法：送出、串流、互動提問等。"""
from __future__ import annotations

import json
import logging
import inspect
from pathlib import Path

log = logging.getLogger(__name__)
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
from utils.think_filter import ThinkTagFilter
from utils.health_monitor import get_health_monitor


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
        "/caption": "backfill_captions",
        "/reindex-images": "reindex_images",
        "/focus": "focus_evaluate",
        "/health": "show_health",
        "/export": "export_conversation",
    }

    SLASH_HINTS: list[tuple[str, str]] = [
        ("/help", "顯示所有指令說明"),
        ("/new", "開啟新對話"),
        ("/import", "匯入 PDF/TXT/MD 或網頁到知識庫"),
        ("/search", "搜尋知識庫"),
        ("/memory", "管理記憶 / 知識"),
        ("/settings", "開啟設定"),
        ("/mode", "切換感性 / 理性模式"),
        ("/save", "儲存當前對話的洞見"),
        ("/reindex", "重建記憶向量索引"),
        ("/ragmode", "切換模式：討論 / 查詢"),
        ("/project", "切換專案"),
        ("/pending", "記憶待確認"),
        ("/caption", "為圖片庫自動生成描述"),
        ("/reindex-images", "重建圖片向量索引"),
        ("/focus", "對焦評估──比對問題意識與最近記憶"),
        ("/health", "系統健康監測"),
        ("/export", "匯出當前對話為 Markdown 文件"),
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
        """全域不處理貼上/拖入匯入；只在 KnowledgeModal 匯入頁處理。"""
        return

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
        if not Path(t).exists() or not Path(t).is_file():
            return None
        if not t.lower().endswith((".pdf", ".txt", ".md")):
            self.notify("僅支援 PDF、TXT 或 MD 檔案", severity="warning", timeout=3)
            return None
        return t

    def fill_input(self, text: str) -> None:
        from widgets import ChatInput
        try:
            chat_input = self.query_one("#chat-input", ChatInput)
            chat_input.clear()
            chat_input.insert(text)
            chat_input.focus()
        except Exception:
            pass

    async def _handle_slash_command(self, text: str) -> bool:
        cmd = text.split()[0].lower()
        if cmd in self.SLASH_COMMANDS:
            action = self.SLASH_COMMANDS[cmd]
            method = getattr(self, f"action_{action}", None)
            if method:
                result = method()
                if inspect.isawaitable(result):
                    await result
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

        if text.startswith("/"):
            await self._handle_slash_command(text)
            return

        user_content = await self._build_user_content(text)
        self.messages.append({"role": "user", "content": user_content})

        if self.state.current_conversation_id is None:
            project_id = self.state.current_project["id"] if self.state.current_project else None
            self.state.current_conversation_id = await self._conv_store.create_conversation(project_id)
            title = text[:30].strip().replace("\n", " ")
            await self._conv_store.set_title(self.state.current_conversation_id, title)

            # Show guidance if project has no knowledge base
            if self.state.current_project and project_id:
                try:
                    docs = await self._conv_store.list_documents(project_id)
                    if not docs:
                        # Show empty project guidance as a system message
                        guidance = "歡迎！你可以直接開始對話，或用 /import 匯入文獻。"
                        await self.notify(guidance, timeout=4.0)
                except Exception:
                    pass

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
        # Google AI Studio via gemini/ prefix (always direct)
        if model.startswith("gemini/"):
            return True
        # OpenAI-compatible providers with explicit base
        return model.startswith("openai/") and bool(base)

    def _resolve_api_key(self, env: dict) -> str:
        base = env.get("OPENAI_API_BASE", "")
        BASE_TO_KEY = {
            "openrouter.ai":    "OPENROUTER_API_KEY",
            "deepseek.com":     "DEEPSEEK_API_KEY",
            "minimaxi.chat":    "MINIMAX_API_KEY",
            "api.minimax.chat": "MINIMAX_API_KEY",
            "generativelanguage.googleapis.com": "GOOGLE_API_KEY",
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

    def _has_multimodal_content(self) -> bool:
        """Check if current messages contain image/multimodal content.
        Note: 圖片現在走 CLIP 語意檢索，不再送 base64，此方法保留向後相容。
        """
        for m in self.messages:
            if isinstance(m.get("content"), list):
                return True
        return False

    def _trim_long_conversation(self, messages: list) -> list:
        """Trim conversation context if it exceeds 100 messages.

        Keeps system message (index 0 if present) and last 80 messages.
        Returns a trimmed copy without modifying self.messages.
        """
        if len(messages) <= 100:
            return messages

        log.warning("Conversation length %d exceeds 100, trimming to 80 recent messages", len(messages))

        # Determine if first message is system message
        has_system = messages and messages[0].get("role") == "system"
        if has_system:
            # Keep system message + last 80 messages
            trimmed = [messages[0]] + messages[-80:]
        else:
            # Keep only last 80 messages
            trimmed = messages[-80:]

        return trimmed

    @work(exclusive=True, group="stream_response")
    async def _stream_response(self) -> None:
        from widgets import Chatbox, ChatInput
        self.is_loading = True
        self._llm_call_count = 0
        self.state.last_rag_sources = []
        container = self.query_one("#messages", Vertical)

        bubble = Chatbox("assistant", streaming=True)
        await container.mount(bubble)
        bubble.set_responding(True)
        self._scroll_to_bottom()

        full_content = ""
        think_filter = ThinkTagFilter()

        try:
            if self._is_codex_cli_mode():
                # ── Codex CLI 路徑 ──
                if not is_codex_available():
                    raise RuntimeError("codex CLI 未安裝。執行: npm i -g @openai/codex")

                sys_prompt = await self._build_system_prompt()

                # Trim long conversations before sending
                trimmed_messages = self._trim_long_conversation(self.messages)
                user_msg = trimmed_messages[-1]["content"]

                # RAG 注入（統一呼叫）
                _, sys_addon = await self._inject_rag_context(
                    [{"role": "user", "content": user_msg}],
                    return_sys_addon=True,
                )
                if sys_addon:
                    sys_prompt += "\n\n" + sys_addon

                context = ""
                for m in trimmed_messages[:-1]:
                    role = "You" if m["role"] == "user" else "De-insight"
                    context += f"{role}: {m['content']}\n\n"

                full_prompt = f"{context}You: {user_msg}" if context else user_msg

                env = load_env()
                codex_model = env.get("LLM_MODEL", "").removeprefix("codex-cli/")

                async for chunk in codex_stream(full_prompt, sys_prompt, model=codex_model):
                    full_content += chunk
                    display = think_filter.feed(chunk)
                    bubble.stream_update(display)
                    self._scroll_to_bottom()
                think_filter.flush()
                self._llm_call_count += 1
                get_health_monitor().record_api_call()
            elif self._is_direct_api_mode():
                # ── 直接 API 路徑（MiniMax / OpenRouter / multimodal）──
                env = load_env()
                raw_model = env.get("LLM_MODEL", "")
                if raw_model.startswith("gemini/"):
                    api_base = "https://generativelanguage.googleapis.com/v1beta/openai"
                    api_key = env.get("GOOGLE_API_KEY", "")
                    model = raw_model.removeprefix("gemini/")
                else:
                    api_base = env.get("OPENAI_API_BASE", "") or "https://api.openai.com/v1"
                    api_key = self._resolve_api_key(env)
                    model = raw_model.removeprefix("openai/")

                sys_prompt = await self._build_system_prompt()

                # Trim long conversations before sending
                trimmed_messages = self._trim_long_conversation(self.messages)

                # RAG 注入（統一呼叫）
                send_messages = await self._inject_rag_context(
                    [{"role": "system", "content": sys_prompt}] + trimmed_messages
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
                        if response.status_code >= 400:
                            body = await response.aread()
                            err_text = body.decode("utf-8", errors="replace")[:300]
                            if response.status_code in (401, 403):
                                raise RuntimeError("API 金鑰無效或已過期，請到 /settings 更新")
                            if "vision" in err_text.lower() or "image" in err_text.lower() or "multimodal" in err_text.lower():
                                raise RuntimeError(f"此模型不支援圖片輸入。請切換到支援 vision 的模型（如 gpt-4o、gemini）。")
                            raise RuntimeError(f"API 錯誤 {response.status_code}: {err_text}")
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
                                    display = think_filter.feed(delta)
                                    bubble.stream_update(display)
                                    self._scroll_to_bottom()
                            except (json.JSONDecodeError, ValueError):
                                pass
                think_filter.flush()
                self._llm_call_count += 1
                get_health_monitor().record_api_call()
            else:
                # ── FastAPI 後端路徑 ──
                sys_prompt = await self._build_system_prompt()

                # Trim long conversations before sending
                trimmed_messages = self._trim_long_conversation(self.messages)

                send_messages = await self._inject_rag_context(
                    [{"role": "system", "content": sys_prompt}] + trimmed_messages
                )
                async with httpx.AsyncClient(timeout=120.0) as client:
                    async with client.stream(
                        "POST",
                        f"{self.api_base}/api/chat",
                        json={"messages": send_messages, "mode": self.mode},
                    ) as response:
                        if response.status_code in (401, 403):
                            raise RuntimeError("API 金鑰無效或已過期，請到 /settings 更新")
                        if response.status_code >= 400:
                            body = await response.aread()
                            err_text = body.decode("utf-8", errors="replace")[:300]
                            raise RuntimeError(f"後端錯誤 {response.status_code}: {err_text}")
                        async for line in response.aiter_lines():
                            if line.startswith("0:"):
                                try:
                                    chunk = json.loads(line[2:])
                                    full_content += chunk
                                    display = think_filter.feed(chunk)
                                    bubble.stream_update(display)
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
                                think_filter.flush()
                                self._llm_call_count += 1
                                get_health_monitor().record_api_call()
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

                surfaced_bridge = getattr(self, "_last_surfaced_bridge", None)
                if surfaced_bridge:
                    from widgets import Container, Text
                    bridge_container = self.query_one("#messages", Container)
                    bridge_hint = Text(f"💡 {surfaced_bridge}", id="bridge-hint")
                    bridge_hint.styles.color = "silver"
                    bridge_hint.styles.italic = True
                    bridge_hint.styles.margin_top = 1
                    await bridge_container.mount(bridge_hint)
                    self._scroll_to_bottom()

                    import time
                    from widgets import SurfacedBridgeRecord, _normalize_bridge_topic, MAX_RECENT_SURFACED_BRIDGES
                    normalized = _normalize_bridge_topic(surfaced_bridge)
                    self.state.recent_surfaced_bridges.append(
                        SurfacedBridgeRecord(
                            topic=normalized,
                            turn_index=self.state.turn_index,
                            timestamp=time.time()
                        )
                    )
                    if len(self.state.recent_surfaced_bridges) > MAX_RECENT_SURFACED_BRIDGES:
                        self.state.recent_surfaced_bridges = self.state.recent_surfaced_bridges[-MAX_RECENT_SURFACED_BRIDGES:]

            self.state.turn_index += 1

            raw_content = self.messages[-2]["content"] if len(self.messages) >= 2 else ""
            user_text = self._extract_text(raw_content)
            if user_text and len(user_text.strip()) >= 40:
                self._auto_extract_memories(user_text)

            if getattr(self, "_pending_focus_nudge", False):
                self._pending_focus_nudge = False
                await self._inject_focus_nudge()

        except httpx.ConnectError as e:
            get_health_monitor().record_error("ConnectError", str(e))
            bubble.set_responding(False)
            bubble.stream_update(
                "**連線錯誤** — 後端未啟動\n\n"
                "cd backend && .venv/bin/python -m uvicorn main:app --reload"
            )
            await bubble.finalize_stream()
        except Exception as e:
            # Check if it's a rate limit error
            error_msg = str(e)
            if "429" in error_msg or "rate" in error_msg.lower():
                get_health_monitor().record_rate_limit()
            else:
                error_type = type(e).__name__
                get_health_monitor().record_error(error_type, error_msg)
            bubble.set_responding(False)
            bubble.stream_update(f"**錯誤** — {escape(error_msg)}")
            await bubble.finalize_stream()
        finally:
            self.is_loading = False
            self._update_status()
            self._update_menu_bar()
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

        choices = None
        if block.type == 'confirm':
            choices = ["是，繼續", "不，我還想調整"]
        elif block.type == 'select':
            choices = block.choices
        elif block.type == 'multi':
            choices = block.choices
        elif block.type == 'input':
            pass

        # Check if choices are empty for select/multi types
        if block.type in ('select', 'multi') and (not choices or len(choices) == 0):
            self.notify("互動選項為空，已跳過", timeout=3)
            self.state.current_interactive_block = None
            # Continue to next block if any
            if len(blocks) > 1:
                self._handle_interactive_blocks(blocks[1:])
            return

        if choices:
            ta.set_choices(choices)

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

    @staticmethod
    def _extract_retry_after(exc: Exception) -> float | None:
        """從 RateLimitError 訊息中提取 retry-after 秒數。"""
        import re
        text = str(exc)
        m = re.search(r'retry[\s_-]*after[\s:=]*(\d+(?:\.\d+)?)', text, re.IGNORECASE)
        if m:
            return float(m.group(1))
        return None

    async def _quick_llm_call(self, prompt: str, max_tokens: int = 500, max_retries: int = 3) -> str:
        env = load_env()
        model = env.get("RAG_LLM_MODEL", "") or env.get("LLM_MODEL", "ollama/llama3.2")

        if model.startswith("codex-cli/"):
            codex_model = model.removeprefix("codex-cli/")
            result = ""
            async for chunk in codex_stream(prompt, model=codex_model):
                result += chunk
            if hasattr(self, '_llm_call_count'):
                self._llm_call_count += 1
            get_health_monitor().record_api_call()
            return result

        if model.startswith("codex/"):
            model = "openai/" + model.removeprefix("codex/")

        import asyncio as _asyncio
        import os
        import litellm
        from rag.rate_guard import RateLimitError

        # litellm 從 os.environ 讀 API key，load_env() 只回傳 dict
        for key in ("GOOGLE_API_KEY", "GEMINI_API_KEY", "OPENAI_API_KEY",
                     "OPENROUTER_API_KEY", "ANTHROPIC_API_KEY", "OPENAI_API_BASE"):
            val = env.get(key, "")
            if val:
                os.environ[key] = val
        # litellm gemini/ prefix 需要 GEMINI_API_KEY，若只有 GOOGLE_API_KEY 則複製過去
        if not os.environ.get("GEMINI_API_KEY") and os.environ.get("GOOGLE_API_KEY"):
            os.environ["GEMINI_API_KEY"] = os.environ["GOOGLE_API_KEY"]

        # RAG_LLM_MODEL 可能有獨立的 API base/key（例如 Google AI Studio OpenAI 相容端點）
        rag_api_base = env.get("RAG_API_BASE", "")
        rag_api_key = env.get("RAG_API_KEY", "")
        if model == env.get("RAG_LLM_MODEL", "") and (rag_api_base or rag_api_key):
            # RAG 模型用獨立端點，走 openai/ prefix
            if not model.startswith(("openai/", "gemini/", "anthropic/", "ollama/")):
                model = f"openai/{model}"
            if rag_api_base:
                os.environ["OPENAI_API_BASE"] = rag_api_base
            if rag_api_key:
                os.environ["OPENAI_API_KEY"] = rag_api_key

        last_exc = None
        for attempt in range(max_retries):
            try:
                resp = await litellm.acompletion(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,
                    max_tokens=max_tokens,
                )
                if hasattr(self, '_llm_call_count'):
                    self._llm_call_count += 1
                get_health_monitor().record_api_call()
                return resp.choices[0].message.content or ""
            except RateLimitError as exc:
                get_health_monitor().record_rate_limit(self._extract_retry_after(exc))
                last_exc = exc
                retry_after = self._extract_retry_after(exc)
                delay = retry_after if retry_after else 2.0 * (2 ** attempt)
                self.log.warning(
                    f"_quick_llm_call rate limited (attempt {attempt + 1}/{max_retries}), "
                    f"retrying in {delay:.1f}s: {exc}"
                )
                if attempt < max_retries - 1:
                    await _asyncio.sleep(delay)
            except Exception as exc:
                get_health_monitor().record_error("LLMError", str(exc))
                last_exc = exc
                delay = 1.5 * (attempt + 1)
                self.log.warning(
                    f"_quick_llm_call error (attempt {attempt + 1}/{max_retries}), "
                    f"retrying in {delay:.1f}s: {exc}"
                )
                if attempt < max_retries - 1:
                    await _asyncio.sleep(delay)

        raise last_exc

    async def _inject_focus_nudge(self) -> None:
        """在對話裡注入一條策展人的對焦提問。"""
        try:
            if not self.state.current_project:
                return
            from focus import load_focus
            from paths import project_root as get_project_root
            from memory.store import get_memories

            pid = self.state.current_project["id"]
            fields = load_focus(get_project_root(pid))
            focus_question = fields.get("問題意識", "").strip()
            if not focus_question:
                return

            recent = await get_memories(
                type="insight",
                limit=3,
                db_path=get_project_root(pid) / "memories.db",
            )
            recent_str = "\n".join(f"- {m['content']}" for m in recent) if recent else "- （暫無）"

            prompt = (
                f"創作者的問題意識：{focus_question}\n\n"
                f"他最近的洞見：\n{recent_str}\n\n"
                "用一句話，以好奇的語氣，問他最近的討論和他的問題意識是什麼關係。"
                "不要評判，不要說「你偏了」。繁體中文，不超過 40 字。"
            )
            nudge = (await self._quick_llm_call(prompt, max_tokens=80)).strip()
            if nudge:
                await self._add_assistant_bubble(nudge, persist=True)
        except Exception:
            pass

    @work(exclusive=True, group="focus_evaluate", thread=False)
    async def action_focus_evaluate(self) -> None:
        """對焦評估：讀取問題意識與最近記憶，輸出完整評估。"""
        try:
            if not self.state.current_project:
                self.notify("請先選擇一個專案", severity="warning")
                return

            from focus import load_focus, to_prompt_block
            from paths import project_root as get_project_root
            from memory.store import get_memories

            pid = self.state.current_project["id"]
            project_root = get_project_root(pid)
            fields = load_focus(project_root)
            focus_block = to_prompt_block(fields)
            if not focus_block:
                self.notify("請先填寫問題意識（Research 面板 → 問題）", severity="warning")
                return

            recent = await get_memories(limit=10, db_path=project_root / "memories.db")
            if not recent:
                self.notify("尚無記憶可以評估", severity="warning")
                return

            memories_str = "\n".join(f"[{m['type']}] {m['content']}" for m in recent)
            prompt = (
                f"# 這位創作者的問題意識\n{focus_block}\n\n"
                f"# 他最近的記憶條目（最新 10 條）\n{memories_str}\n\n"
                "請做一次對焦評估：\n"
                "1. 哪些記憶和問題意識有直接關聯？\n"
                "2. 哪些記憶偏離了問題意識的方向？\n"
                "3. 有沒有他還沒意識到的張力或矛盾？\n\n"
                "語氣：好奇的觀察者，不是評審。不要說「你做得好/不好」。"
                "繁體中文，直接。"
            )

            await self._add_user_bubble("[對焦評估]", persist=True)
            await self._stream_response_with_prompt(prompt)
        except Exception as e:
            self.notify(f"對焦評估失敗：{e}", severity="error")

    def action_export_conversation(self) -> None:
        """Export current conversation as Markdown file."""
        if not self.messages:
            self.notify("沒有對話可匯出", severity="warning")
            return

        try:
            from pathlib import Path
            from datetime import datetime

            # Get home directory or use project directory
            if self.state.current_project:
                from paths import project_root
                export_dir = project_root(self.state.current_project["id"])
            else:
                export_dir = Path.home()

            export_dir.mkdir(parents=True, exist_ok=True)

            # Generate markdown content
            md_lines = []
            md_lines.append("# 對話記錄")
            md_lines.append(f"\n**匯出時間**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

            if self.state.current_project:
                md_lines.append(f"**專案**: {self.state.current_project.get('name', 'N/A')}\n")

            md_lines.append("---\n")

            # Add messages
            for msg in self.messages:
                role = msg.get("role", "unknown")
                content = msg.get("content", "")

                if role == "user":
                    md_lines.append("### 👤 User\n")
                elif role == "assistant":
                    md_lines.append("### 🤖 Assistant\n")
                elif role == "system":
                    md_lines.append("### ⚙️ System\n")
                else:
                    md_lines.append(f"### {role}\n")

                # Escape content and add to markdown
                md_lines.append(f"{content}\n")
                md_lines.append("")

            md_content = "\n".join(md_lines)

            # Generate filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = export_dir / f"conversation_{timestamp}.md"

            # Write file
            with open(filename, "w", encoding="utf-8") as f:
                f.write(md_content)

            self.notify(f"對話已匯出至: {filename}", timeout=3.0)
        except Exception as e:
            self.notify(f"匯出失敗：{e}", severity="error")

    async def _build_user_content(self, text: str) -> str:
        """組合使用者文字。
        有 pending_images 時，呼叫 Vision LLM 即時生成詳細描述附加到文字中。
        """
        pending = getattr(self.state, "pending_images", None) or []
        if not pending:
            return text

        log.debug("_build_user_content: %d pending images", len(pending))

        from rag.image_store import describe_image_for_chat

        descriptions = []
        for img in pending:
            img_path = Path(img) if isinstance(img, str) else Path(img.get("path", ""))
            if not img_path.exists():
                continue
            # --- 兩源合併：stored caption + live description ---
            # 1) Live Vision 描述
            live_desc = await describe_image_for_chat(img_path, user_question=text)

            # 2) LanceDB stored caption（三段式）
            stored_block = ""
            if self.state.current_project:
                try:
                    from rag.image_store import search_images, _normalize_caption
                    results = await search_images(
                        self.state.current_project["id"],
                        img_path.name, limit=1,
                    )
                    if results and results[0].get("filename") == img_path.name:
                        raw_cap = results[0].get("caption", "")
                        cap = _normalize_caption(raw_cap)
                        # 判斷是否為有效三段式（非 fallback）
                        if isinstance(cap, dict):
                            content = cap.get("content", {})
                            title = content.get("title", "") if isinstance(content, dict) else ""
                            tags = cap.get("style_tags", [])
                            desc_text = cap.get("description", "")
                            stem = Path(img_path.name).stem
                            is_fallback = (desc_text == stem) and not tags
                            if not is_fallback and (title or tags or (desc_text and desc_text != stem)):
                                parts = []
                                if title:
                                    parts.append(f"標題：{title}")
                                if content.get("type"):
                                    parts.append(f"類型：{content['type']}")
                                if content.get("creator"):
                                    parts.append(f"創作者：{content['creator']}")
                                if tags:
                                    parts.append(f"風格標籤：{', '.join(tags[:5])}")
                                if desc_text:
                                    parts.append(f"描述：{desc_text}")
                                stored_block = "\n".join(parts)
                except Exception:
                    pass

            # 3) 組合
            blocks = []
            if stored_block:
                blocks.append(f"[上傳時的理解]\n{stored_block}")
            if live_desc:
                blocks.append(f"[當下觀察]\n{live_desc}")

            if blocks:
                descriptions.append(f"── 圖片分析（{img_path.name}）──\n" + "\n\n".join(blocks))
            elif stored_block or live_desc:
                descriptions.append(f"── 圖片分析（{img_path.name}）──\n{stored_block or live_desc}")
            else:
                descriptions.append(f"（附加圖片：{img_path.name}，描述生成失敗）")

        self.state.pending_images = []
        self._update_menu_bar()

        if descriptions:
            return text + "\n\n" + "\n\n".join(descriptions)
        return text

    @staticmethod
    def _extract_text(content: str | list[dict]) -> str:
        """從 multipart content 取出純文字。"""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return " ".join(
                p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text"
            )
        return str(content)

    async def _add_user_bubble(self, content: str, *, persist: bool = True) -> None:
        from widgets import Chatbox
        container = self.query_one("#messages", Vertical)
        await container.mount(Chatbox("user", content))
        self._scroll_to_bottom()
        self.messages.append({"role": "user", "content": content})
        if persist and self.state.current_conversation_id:
            await self._conv_store.add_message(
                self.state.current_conversation_id, "user", content
            )

    async def _add_assistant_bubble(self, content: str, *, persist: bool = True) -> None:
        from widgets import Chatbox
        container = self.query_one("#messages", Vertical)
        await container.mount(Chatbox("assistant", content))
        self._scroll_to_bottom()
        self.messages.append({"role": "assistant", "content": content})
        if persist and self.state.current_conversation_id:
            await self._conv_store.add_message(
                self.state.current_conversation_id, "assistant", content
            )

    async def _stream_response_with_prompt(self, prompt: str) -> None:
        """以指定 prompt 直接呼叫 LLM 並輸出成 assistant bubble。"""
        try:
            response = (await self._quick_llm_call(prompt, max_tokens=900)).strip()
            if not response:
                response = "目前無法產生評估結果。"
            await self._add_assistant_bubble(response, persist=True)
        except Exception as e:
            self.notify(f"對焦評估失敗：{e}", severity="error")

    async def _get_memory_summary(self) -> str:
        """取得近期記憶摘要（輕量格式，不需 LLM 呼叫）。"""
        try:
            from memory.store import get_memories
            _mem_db = None
            if self.state.current_project:
                from paths import project_root
                _mem_db = project_root(self.state.current_project["id"]) / "memories.db"
            recent = await get_memories(limit=10, db_path=_mem_db)
            if not recent:
                return ""
            lines = []
            for m in recent:
                topic = f" (#{m['topic']})" if m.get('topic') else ""
                lines.append(f"- [{m['type']}] {m['content']}{topic}")
            return "\n".join(lines)
        except Exception:
            return ""

    async def _decompose_query_deep(self, user_msg: str, recent_turns: list[dict] | None = None) -> list[str]:
        """深度模式：用對話脈絡抽取底層敘事結構，轉成跨領域可搜的搜尋查詢。

        核心邏輯：把名詞重要性降低，找出這段對話裡正在形成的論點結構，
        再把它重寫成「不帶特定領域詞彙」的搜尋查詢——這樣才能在知識庫
        裡找到敘事結構相似、但領域完全不同的論述。

        recent_turns: 最近幾輪對話（[{role, content}, ...]），用於理解脈絡。
        """
        # 組合對話摘要：取最近 6 輪（避免過長），只留文字內容
        ctx_lines: list[str] = []
        for m in (recent_turns or [])[-6:]:
            role_label = "創作者" if m.get("role") == "user" else "對話者"
            content = self._extract_text(m.get("content", ""))
            if content and len(content.strip()) > 3:
                ctx_lines.append(f"{role_label}：{content[:200]}")
        context_block = "\n".join(ctx_lines)

        # 即使 user_msg 很短，只要有對話上文就繼續
        if not context_block and len(user_msg.strip()) < 10:
            return [user_msg]

        prompt = (
            "以下是一段創作者正在進行的思考對話：\n\n"
            f"{context_block}\n"
            f"（最新一句）創作者：{user_msg[:300]}\n\n"
            "請完成以下兩個步驟，直接輸出結果，不要輸出步驟說明或標題：\n\n"
            "第一行：把所有具體名詞拿掉，用一句話說出這段對話裡正在形成的底層論點。\n"
            "例：「一種媒介只有不偽裝自身本質，才具有倫理正當性」\n"
            "例：「工藝的真實性不應為外在市場壓力而妥協」\n\n"
            "第二、三行（可選第四行）：把上面那個論點改寫成 2–3 個完整句子的搜尋查詢，\n"
            "用來在知識庫找處理相同邏輯的論述——哪怕那些論述談的是攝影、陶瓷、字體、教育等完全不同的領域。\n"
            "每個查詢都是完整的一句話，不是關鍵字組合。\n\n"
            "直接輸出 3–4 行，每行一個，不要任何格式符號、不要編號、不要解釋。\n"
            "繁體中文，每行不超過30字，不要用「書籍」「建築」等對話裡已有的具體名詞。"
        )
        try:
            result = await self._quick_llm_call(prompt, max_tokens=300)
            lines = [
                l.strip().strip("[]「」\'\"")
                for l in result.strip().splitlines()
                if l.strip() and not l.strip().startswith("#")
            ]
            # 第一行是抽象論點描述，也拿來搜（它比原始 user_msg 更能跨領域命中）
            # 後面是搜尋查詢
            struct_queries = [q for q in lines[:4] if q and len(q) > 4]
            # 組合：原始訊息（保底）+ 結構化查詢
            all_queries = struct_queries + [user_msg]
            # 去重，保留順序
            seen: set[str] = set()
            unique: list[str] = []
            for q in all_queries:
                if q not in seen:
                    seen.add(q)
                    unique.append(q)
            return unique[:4]  # 最多 4 個查詢
        except Exception as e:
            self.log.debug(f"_decompose_query_deep failed: {e}")
            return [user_msg]

    async def _build_system_prompt(self) -> str:
        """統一構建 system prompt：人格 + 模式 + 記憶摘要。\n\n        所有 LLM 路徑都應使用此方法，確保人格注入一致。
        """
        from prompts.foucault import get_system_prompt
        memory_summary = await self._get_memory_summary()
        focus_block = ""
        try:
            from focus import load_focus, to_prompt_block
            from paths import project_root as get_project_root
            if self.state.current_project:
                pid = self.state.current_project["id"]
                fields = load_focus(get_project_root(pid))
                focus_block = to_prompt_block(fields)
        except Exception:
            pass
        return get_system_prompt(
            self.mode,
            memory_summary=memory_summary,
            focus_block=focus_block,
            rag_mode=getattr(self, "rag_mode", "fast"),
        )

    async def _inject_rag_context(
        self,
        messages: list[dict],
        return_sys_addon: bool = False,
    ) -> list[dict] | tuple[list[dict], str]:
        """
        統一 RAG 注入入口。協調各個子方法以處理知識庫檢索。

        Readiness gate 決定要跑哪些增強路徑：
        - empty / building(no chunks)：全部跳過，快速返回
        - building(has chunks) + fast：只跑 pipeline，跳過 memory/image
        - building(has chunks) + deep：完整流程
        - degraded + fast：跳過 memory/image，跑 pipeline + 不完整提示
        - degraded + deep：完整流程
        - ready：完整流程

        return_sys_addon=False（預設）：回傳注入後的 messages（Direct API / FastAPI 用）
        return_sys_addon=True：回傳 (messages, sys_addon_str)（Codex CLI 用）
        """
        raw_content = messages[-1]["content"] if messages else ""
        user_msg = self._extract_text(raw_content)
        if not user_msg:
            return (messages, "") if return_sys_addon else messages

        augmented = list(messages)
        insert_idx = max(len(augmented) - 1, 0)
        sys_addon_parts = []

        # 獲取 readiness 狀態和 RAG 模式
        _pid = self.state.current_project["id"] if self.state.current_project else "default"
        _readiness, _is_fast, _skip_all, _skip_augment, _rag_hint = await self._apply_readiness_gate(_pid)

        # Early return for empty / building-no-chunks
        if _skip_all:
            self.state.last_rag_sources = []
            await self._update_research_panel(status="empty", content="", query=user_msg)
            if _rag_hint and return_sys_addon:
                return augmented, _rag_hint + "\n\n請用繁體中文回覆。"
            return (augmented, "") if return_sys_addon else augmented

        # 計算 insight score（影響記憶注入量）
        _insight_score = await self._compute_insight_score(user_msg, _pid, _skip_augment)

        # 構建記憶上下文
        if not _skip_augment:
            mem_addon = await self._build_memory_context(user_msg, _insight_score, return_sys_addon)
            if mem_addon:
                if return_sys_addon:
                    sys_addon_parts.append(mem_addon)
                else:
                    augmented.insert(insert_idx, {"role": "system", "content": mem_addon})
                    insert_idx += 1

        # 檢測是否應注入圖片上下文
        _user_asks_about_images = self._detect_image_relevance(user_msg)

        # 構建視覺偏好上下文
        if not _skip_augment and _user_asks_about_images:
            pref_addon = await self._build_image_preference_context(return_sys_addon)
            if pref_addon:
                if return_sys_addon:
                    sys_addon_parts.append(pref_addon)
                else:
                    augmented.insert(insert_idx, {"role": "system", "content": pref_addon})
                    insert_idx += 1

        # 構建圖片知識庫上下文
        if not _skip_augment and _user_asks_about_images:
            img_addon = await self._build_image_context(user_msg, return_sys_addon)
            if img_addon:
                if return_sys_addon:
                    sys_addon_parts.append(img_addon)
                else:
                    augmented.insert(insert_idx, {"role": "system", "content": img_addon})
                    insert_idx += 1

        # 清除舊的 RAG sources 並開始管道檢索
        self.state.last_rag_sources = []
        await self._update_research_panel(
            status="loading",
            content="正在檢索知識庫…",
            query=user_msg,
        )

        # 執行 RAG 管道並格式化輸出
        await self._retrieve_and_inject_rag_context(
            user_msg, _pid, _is_fast, _readiness, _skip_augment,
            _insight_score, _rag_hint, augmented, insert_idx,
            sys_addon_parts, return_sys_addon
        )

        if return_sys_addon:
            addon = "\n\n".join(sys_addon_parts)
            if addon:
                addon += "\n\n請用繁體中文回覆。"
            return augmented, addon

        return augmented

    async def _apply_readiness_gate(
        self, project_id: str
    ) -> tuple[object, bool, bool, bool, str]:
        """
        檢查 readiness 狀態並決定要跳過哪些路徑。

        回傳：(readiness_obj, is_fast_mode, skip_all, skip_augment, hint_message)
        """
        try:
            from rag.readiness import get_readiness_service
            _readiness = await get_readiness_service().get_snapshot(project_id)
        except Exception:
            from rag.readiness import ReadinessSnapshot
            _readiness = ReadinessSnapshot()

        _is_fast = (getattr(self, "rag_mode", "fast") != "deep")

        _skip_all = False
        _skip_augment = False
        _rag_hint = ""

        if _readiness.status_label == "empty":
            _skip_all = True
        elif _readiness.status_label == "building" and not _readiness.has_ready_chunks:
            _skip_all = True
            _rag_hint = "知識庫建構中，先不使用知識庫回答。"
        elif _readiness.status_label == "building":
            _rag_hint = "知識庫建構中，以下為部分可用資料。"
            if _is_fast:
                _skip_augment = True
        elif _readiness.status_label == "degraded":
            _rag_hint = "部分文獻匯入失敗，資料可能不完整。"
            if _is_fast:
                _skip_augment = True

        return _readiness, _is_fast, _skip_all, _skip_augment, _rag_hint

    async def _compute_insight_score(
        self, user_msg: str, project_id: str, skip_augment: bool
    ) -> float:
        """計算 insight score，影響記憶注入的數量上限。"""
        if skip_augment:
            return 0.0

        try:
            from rag.pipeline import get_insight_score
            _db_path = None
            if self.state.current_project:
                from paths import project_root as _pr
                _db_path = _pr(self.state.current_project["id"]) / "memories.db"
            return await get_insight_score(user_msg, project_id, db_path=_db_path)
        except Exception:
            return 0.0

    async def _build_memory_context(
        self, user_msg: str, insight_score: float, return_sys_addon: bool
    ) -> str:
        """
        從記憶向量資料庫檢索並格式化相關記憶。
        記憶是 per-project 的（使用者的想法/洞見），不需要合併全局。
        全局文獻庫的合併由 query_knowledge_merged() 負責。
        """
        _mem_limit = 5 if insight_score > 0.4 else 3
        _lancedb_dir = None
        if self.state.current_project:
            from paths import project_root
            _lancedb_dir = project_root(self.state.current_project["id"]) / "lancedb"

        try:
            from memory.vectorstore import search_similar, has_index
            if not has_index(lancedb_dir=_lancedb_dir):
                return ""

            mem_results = await search_similar(user_msg, limit=_mem_limit, lancedb_dir=_lancedb_dir)
            if not mem_results:
                return ""

            mem_lines = "\n".join(
                f"- [{m['type']}] {m['content']}" + (f" (#{m['topic']})" if m.get('topic') else "")
                for m in mem_results
            )
            return f"使用者過去的想法（語意相關）：\n{mem_lines}"
        except Exception as e:
            self.log.warning(f"RAG inject memory failed: {e}")
            return ""

    def _detect_image_relevance(self, user_msg: str) -> bool:
        """
        判斷使用者的提問是否涉及圖片/視覺相關內容。
        """
        _IMAGE_KEYWORDS = (
            "圖", "圖片", "照片", "相片", "影像", "視覺", "畫面",
            "風格", "偏好", "收集", "參考圖", "靈感",
            "image", "photo", "picture", "visual", "style",
        )
        _has_pending_images = bool(getattr(self.state, "pending_images", None))
        return _has_pending_images or any(kw in user_msg for kw in _IMAGE_KEYWORDS)

    async def _build_image_preference_context(self, return_sys_addon: bool) -> str:
        """
        取得使用者的視覺偏好分析並格式化。
        回傳視覺偏好文本，若無則回傳空字串。
        """
        try:
            from memory.store import get_memories
            _mem_db = None
            if self.state.current_project:
                from paths import project_root as _pr
                _mem_db = _pr(self.state.current_project["id"]) / "memories.db"

            prefs = await get_memories(type="preference", limit=1, db_path=_mem_db)
            if not prefs or not prefs[0].get("content"):
                return ""

            return (
                "# 視覺偏好分析（系統根據使用者上傳的參考圖片自動生成）\n\n"
                f"{prefs[0]['content']}\n\n"
                "當使用者問到「我的圖片」「我的風格偏好」「我收集的東西」等問題時，"
                "直接引用以上分析結果回答，不要說你看不到圖片。"
            )
        except Exception:
            return ""

    async def _build_image_context(
        self, user_msg: str, return_sys_addon: bool
    ) -> str:
        """
        檢索圖片知識庫並格式化相關圖片。
        回傳圖片上下文文本，若無相關圖片則回傳空字串。
        """
        try:
            if not self.state.current_project:
                return ""

            from rag.image_store import search_images
            img_results = await search_images(
                self.state.current_project["id"], user_msg, limit=3,
            )
            if not img_results:
                return ""

            def _format_img(r):
                cap = r.get("caption", {})
                if isinstance(cap, dict):
                    parts = []
                    content = cap.get("content", {})
                    if isinstance(content, dict) and content.get("title"):
                        parts.append(content["title"])
                    tags = cap.get("style_tags", [])
                    if tags:
                        parts.append(f"[{', '.join(tags[:4])}]")
                    desc = cap.get("description", "")
                    if desc:
                        parts.append(desc[:80])
                    caption_str = " — ".join(parts) if parts else r["filename"]
                else:
                    caption_str = str(cap) if cap else r["filename"]
                return f"- [圖片] {r['filename']}: {caption_str}"

            img_lines = "\n".join(
                _format_img(r)
                for r in img_results
                if r.get("score", 0) > 0.3 or r.get("filename", "") in user_msg
            )
            if not img_lines:
                return ""

            return f"相關圖片（可引用路徑或描述）：\n{img_lines}"
        except Exception as e:
            self.log.warning(f"RAG inject images failed: {e}")
            return ""

    async def _retrieve_and_inject_rag_context(
        self,
        user_msg: str,
        project_id: str,
        is_fast: bool,
        readiness: object,
        skip_augment: bool,
        insight_score: float,
        rag_hint: str,
        augmented: list[dict],
        insert_idx: int,
        sys_addon_parts: list[str],
        return_sys_addon: bool,
    ) -> None:
        """
        執行 RAG 管道（深度或快速模式）並將結果注入上下文。
        直接修改 augmented 和 sys_addon_parts。
        """
        # Show spinner while searching knowledge base
        try:
            from widgets import MenuBar
            menu = self.query_one("#menu-bar", MenuBar)
            menu.show_progress("正在搜尋知識庫...")
        except Exception:
            pass

        try:
            from rag.pipeline import run_thinking_pipeline
            _db_path = None
            if self.state.current_project:
                from paths import project_root as _pr
                _db_path = _pr(self.state.current_project["id"]) / "memories.db"

            if not is_fast:
                # 深度模式
                pipeline_result = await self._execute_deep_rag_pipeline(
                    user_msg, project_id, _db_path
                )
            else:
                # 快速模式
                pipeline_result = await run_thinking_pipeline(
                    user_input=user_msg,
                    project_id=project_id,
                    mode="fast",
                    db_path=_db_path,
                    recent_surfaced_bridges=self.state.recent_surfaced_bridges,
                )

            # 儲存診斷資訊
            self._last_pipeline_diagnostics = pipeline_result.get("diagnostics", {})
            diag = pipeline_result.get("diagnostics", {})

            context_text = pipeline_result.get("context_text", "")
            raw_result = pipeline_result.get("raw_result", "")
            sources = pipeline_result.get("sources", [])
            self._last_surfaced_bridge = pipeline_result.get("surfaced_bridge")

            self.log.info(
                "RAG pipeline: strategy=%s context_len=%d raw_len=%d sources=%d readiness=%s augment_skipped=%s",
                pipeline_result.get("strategy_used"),
                len(context_text), len(raw_result), len(sources),
                readiness.status_label, skip_augment,
            )

            # 更新研究面板
            if raw_result and len(raw_result.strip()) > 10:
                status = "degraded" if readiness.status_label == "degraded" else "ready"
                await self._update_research_panel(
                    content=raw_result,
                    status=status,
                    sources=sources or [],
                    diagnostics=diag,
                    query=user_msg,
                )
            else:
                await self._update_research_panel(
                    status="empty",
                    content="",
                    sources=sources or [],
                    diagnostics=diag,
                    query=user_msg,
                )
            self.state.last_rag_sources = sources or []

            # 記錄深度模式的 fallback 事件
            if diag.get("deep_error_code"):
                self.log.warning(
                    "RAG deep fallback: %s (strategy=%s)",
                    diag["deep_error_code"], pipeline_result["strategy_used"],
                )

            # 格式化並注入最終的系統提示附加文本
            if context_text and len(context_text.strip()) > 10:
                injection = await self._format_rag_addon(
                    context_text, user_msg, is_fast, insight_score,
                    rag_hint, pipeline_result
                )
                if return_sys_addon:
                    sys_addon_parts.append(injection)
                else:
                    augmented.insert(insert_idx, {"role": "system", "content": injection})

        except Exception as e:
            self.log.warning(f"RAG pipeline failed: {e}")
            self.state.last_rag_sources = []
            await self._update_research_panel(
                status="error",
                content=f"知識庫檢索失敗：{e}",
                query=user_msg,
            )
            # Show MenuBar notification for RAG failure
            try:
                from widgets import MenuBar
                menu = self.query_one("#menu-bar", MenuBar)
                menu.show_message("知識庫查詢失敗，使用純對話模式", severity="warning", timeout=5)
            except Exception:
                pass
        finally:
            # Clear spinner after RAG search completes
            try:
                from widgets import MenuBar
                menu = self.query_one("#menu-bar", MenuBar)
                menu.clear_notification()
            except Exception:
                pass

    async def _execute_deep_rag_pipeline(
        self, user_msg: str, project_id: str, db_path
    ) -> dict:
        """
        執行深度模式的 RAG 管道：拆解查詢、並行檢索、合併結果。
        """
        from rag.pipeline import run_thinking_pipeline

        await self._update_research_panel(
            status="loading",
            content="深度模式：提取對話論點結構…",
            query=user_msg,
        )

        # 拆解查詢
        recent_turns = [
            m for m in self.messages[:-1]
            if m.get("role") in ("user", "assistant")
        ][-10:]
        sub_queries = await self._decompose_query_deep(user_msg, recent_turns=recent_turns)
        self.log.info("Deep decompose: %d queries: %s", len(sub_queries), sub_queries)

        # 初始化思考追蹤
        _thinking_lines: list[str] = ["⟐ 論點拆解"]
        for i, sq in enumerate(sub_queries):
            _thinking_lines.append(f"  {i+1}. {sq[:40]}")

        await self._update_research_panel(
            status="loading",
            content="\n".join(_thinking_lines) + "\n\n檢索中…",
            query=user_msg,
        )

        # 對每個子查詢進行檢索
        merged_context_parts: list[str] = []
        merged_sources: list[dict] = []
        seen_snippets: set[str] = set()
        main_diag: dict = {}

        for idx, q in enumerate(sub_queries):
            try:
                r = await run_thinking_pipeline(
                    user_input=q,
                    project_id=project_id,
                    mode="deep",
                    db_path=db_path,
                    recent_surfaced_bridges=self.state.recent_surfaced_bridges,
                )
                if idx == 0:
                    main_diag = r.get("diagnostics", {})

                ctx = r.get("context_text", "").strip()
                _r_diag = r.get("diagnostics", {})
                _r_strategy = r.get("strategy_used", "?")
                _r_n_src = len(r.get("sources", []))

                self.log.info(
                    "Deep sub-query %d/%d: q='%s' strategy=%s ctx_len=%d sources=%d error=%s",
                    idx + 1, len(sub_queries), q[:50],
                    _r_strategy, len(ctx), _r_n_src,
                    _r_diag.get("deep_error_code"),
                )

                # 更新思考流程
                if ctx:
                    _thinking_lines.append(
                        f"  ✓ [{idx+1}] {q[:30]}… → {len(ctx)}字 ({_r_strategy})"
                    )
                else:
                    _thinking_lines.append(
                        f"  · [{idx+1}] {q[:30]}… → 無結果"
                    )

                _progress = f"檢索中… ({idx+1}/{len(sub_queries)})"
                await self._update_research_panel(
                    status="loading",
                    content="\n".join(_thinking_lines) + f"\n\n{_progress}",
                    query=user_msg,
                )

                # 累積結果
                if ctx:
                    label = f"【{q}】\n" if idx > 0 else ""
                    merged_context_parts.append(f"{label}{ctx}")
                for src in r.get("sources", []):
                    snip = src.get("snippet", "") or src.get("title", "")
                    if snip and snip not in seen_snippets:
                        seen_snippets.add(snip)
                        merged_sources.append(src)

            except Exception as sub_e:
                self.log.warning("Deep sub-query %d failed: %s", idx, sub_e)
                _thinking_lines.append(
                    f"  ✗ [{idx+1}] {q[:30]}… → 失敗"
                )

        # 組合合併結果
        merged_context = "\n\n".join(merged_context_parts)

        # 若深度模式無結果，fallback 到快速模式
        if not merged_context.strip():
            self.log.info(
                "Deep mode returned empty for all %d sub-queries, falling back to fast",
                len(sub_queries),
            )
            _thinking_lines.append("\n⟐ 深度無結果，改用快速模式保底…")
            await self._update_research_panel(
                status="loading",
                content="\n".join(_thinking_lines),
                query=user_msg,
            )
            # Show MenuBar notification for deep search circuit breaker
            try:
                from widgets import MenuBar
                menu = self.query_one("#menu-bar", MenuBar)
                menu.show_message("深度搜尋暫時降級為快速模式", severity="warning", timeout=5)
            except Exception:
                pass

            try:
                fallback_r = await run_thinking_pipeline(
                    user_input=user_msg,
                    project_id=project_id,
                    mode="fast",
                    db_path=db_path,
                    recent_surfaced_bridges=self.state.recent_surfaced_bridges,
                )
                fb_ctx = fallback_r.get("context_text", "").strip()
                if fb_ctx:
                    merged_context = fb_ctx
                    merged_sources = fallback_r.get("sources", [])
                    main_diag = fallback_r.get("diagnostics", {})
                    self.log.info(
                        "Deep→fast fallback found content: %d chars, %d sources",
                        len(fb_ctx), len(merged_sources),
                    )
                    _thinking_lines.append(f"  ✓ 快速模式找到 {len(fb_ctx)}字")
                else:
                    _thinking_lines.append("  · 快速模式也無結果")
            except Exception as fb_e:
                self.log.debug("Deep→fast fallback failed: %s", fb_e)
                _thinking_lines.append("  ✗ 快速模式失敗")

        _thinking_summary = "\n".join(_thinking_lines)

        return {
            "strategy_used": "deep" if merged_context_parts else "deep_fast_fallback",
            "context_text": merged_context,
            "raw_result": merged_context,
            "sources": merged_sources[:8],
            "diagnostics": {
                **main_diag,
                "deep_decomposed_queries": sub_queries,
                "deep_query_count": len(sub_queries),
                "deep_fallback_to_fast": not merged_context_parts,
                "thinking_trace": _thinking_summary,
            },
        }

    async def _format_rag_addon(
        self,
        context_text: str,
        user_msg: str,
        is_fast: bool,
        insight_score: float,
        rag_hint: str,
        pipeline_result: dict,
    ) -> str:
        """
        格式化最終的系統提示附加文本，根據 RAG 模式和 insight score。
        """
        hint_prefix = f"（{rag_hint}）\n\n" if rag_hint else ""
        insight_hint = ""
        if insight_score > 0.4:
            insight_hint = (
                "（這個問題和這位創作者過去的洞見高度相關，回答時可以連結他之前的思考。）\n\n"
            )

        if not is_fast:
            # 深度模式：詳細說明搜尋邏輯
            decomposed = pipeline_result.get("diagnostics", {}).get("deep_decomposed_queries", [])
            abstract_queries = [q for q in (decomposed or []) if q != user_msg]
            search_angle_note = ""
            if abstract_queries:
                search_angle_note = (
                    "（系統從對話脈絡中提取出以下底層論點，用來在知識庫搜尋平行論述——"
                    "這些是搜尋用的角度，不是新的問題：\n"
                    + "\n".join(f"  · {q}" for q in abstract_queries)
                    + "）\n\n"
                )

            return (
                "# 知識庫平行論述（深度模式）\n\n"
                f"{hint_prefix}"
                f"{insight_hint}"
                f"【使用者原始問題／陳述】（你要回應的是這個，不是下面的抽象查詢）：\n"
                f"{user_msg}\n\n"
                f"{search_angle_note}"
                "以下是依上述底層論點角度從知識庫找到的平行論述片段。\n"
                "這些文本不一定用相同的詞，但在處理和使用者論點相同的問題邏輯。\n"
                "帶有【】標籤的段落代表從那個角度搜到的內容，不是新的問題。\n\n"
                f"{context_text}\n\n"
                "你的任務（不要展示給使用者看）：\n"
                "1. 用這些平行論述的視角，回應使用者原始說的那件事\n"
                "2. 說明這些論述強化了、還是動搖了使用者的論點\n"
                "3. 如果相關：從設計實踐角度說，這個原則為什麼難落實\n\n"
                "以你自己的語氣說話，不要整理片段、不要條列。\n"
                "知識庫是你的視角來源，不是你要報告的對象。\n"
                "- 引用知識庫概念時用 [[概念名稱]] 標記\n"
                "- 不補入知識庫未出現的人名、理論、史實\n"
                "請用繁體中文回覆。"
            )
        else:
            # 快速模式：簡潔格式
            return (
                "# 知識庫參考內容\n\n"
                f"{hint_prefix}"
                f"{insight_hint}"
                "以下是從使用者的知識庫中檢索到的相關資料，請優先根據這些內容回答。\n\n"
                f"{context_text}\n\n"
                "使用規則：\n"
                "- 回答應基於上述知識庫內容，不可自行補入未出現的人名、理論或引文\n"
                "- 引用知識庫概念時用 [[概念名稱]] 標記\n"
                "- 若知識庫資訊不足，明確說明後可提供一般性思考方向\n"
                "請用繁體中文回覆。"
            )
