"""對話相關方法：送出、串流、互動提問等。"""
from __future__ import annotations

import json
import logging
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
        ("/caption", "為圖片庫自動生成描述"),
        ("/reindex-images", "重建圖片向量索引"),
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
        """貼上/拖入處理：URL 直接匯入，PDF 路徑填入輸入框。"""
        text = event.text.strip()
        if not text:
            return
        # URL — auto-import
        t = text.strip().strip("'\"")
        if t.startswith("http://") or t.startswith("https://"):
            event.prevent_default()
            self._do_import(t)
            return
        path = self._clean_dropped_path(text)
        if path:
            event.prevent_default()
            self._do_import(path)
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
        if not t.lower().endswith((".pdf", ".txt")):
            self.notify("僅支援 PDF 或 TXT 檔案", severity="warning", timeout=3)
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

        if text.startswith("/"):
            self._handle_slash_command(text)
            return

        user_content = await self._build_user_content(text)
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
                # ── 直接 API 路徑（MiniMax / OpenRouter / multimodal）──
                import re as _re
                from prompts.foucault import get_system_prompt as _get_sp
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
                        if response.status_code >= 400:
                            body = await response.aread()
                            err_text = body.decode("utf-8", errors="replace")[:300]
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
            if user_text and len(user_text.strip()) >= 40:
                self._auto_extract_memories(user_text)

        except httpx.ConnectError:
            bubble.set_responding(False)
            bubble.stream_update(
                "**連線錯誤** — 後端未啟動\n\n"
                "cd backend && .venv/bin/python -m uvicorn main:app --reload"
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
                return resp.choices[0].message.content or ""
            except RateLimitError as exc:
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
                last_exc = exc
                delay = 1.5 * (attempt + 1)
                self.log.warning(
                    f"_quick_llm_call error (attempt {attempt + 1}/{max_retries}), "
                    f"retrying in {delay:.1f}s: {exc}"
                )
                if attempt < max_retries - 1:
                    await _asyncio.sleep(delay)

        raise last_exc

    async def _build_user_content(self, text: str):
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
                        cap = _normalize_caption(raw_cap, img_path.name)
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
        統一 RAG 注入入口。使用 rag.pipeline 統一處理知識庫檢索。

        Readiness gate 放在最前面，決定要跑哪些增強路徑：
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
        # Insert RAG context right before the last user message
        # so the LLM sees it adjacent to the question (not buried at start)
        insert_idx = max(len(augmented) - 1, 0)
        sys_addon_parts = []

        # ── Readiness gate (前移到所有增強路徑之前) ──
        _pid = self.state.current_project["id"] if self.state.current_project else "default"
        try:
            from rag.readiness import get_readiness_service
            _readiness = await get_readiness_service().get_snapshot(_pid)
        except Exception:
            from rag.readiness import ReadinessSnapshot
            _readiness = ReadinessSnapshot()

        _is_fast = (getattr(self, "rag_mode", "fast") != "deep")

        # Determine what to skip based on readiness + mode
        _skip_all = False       # Skip memory, image, and pipeline
        _skip_augment = False   # Skip memory & image only (still run pipeline)
        _rag_hint = ""

        if _readiness.status_label == "empty":
            _skip_all = True
        elif _readiness.status_label == "building" and not _readiness.has_ready_chunks:
            _skip_all = True
            _rag_hint = "知識庫建構中，先不使用知識庫回答。"
        elif _readiness.status_label == "building":
            _rag_hint = "知識庫建構中，以下為部分可用資料。"
            if _is_fast:
                _skip_augment = True   # fast: skip memory/image for speed
        elif _readiness.status_label == "degraded":
            _rag_hint = "部分文獻匯入失敗，資料可能不完整。"
            if _is_fast:
                _skip_augment = True   # fast: skip memory/image for speed

        # Early return for empty / building-no-chunks
        if _skip_all:
            # B3: Clear stale sources
            self.state.last_rag_sources = []
            await self._update_research_panel("")
            if _rag_hint and return_sys_addon:
                return augmented, _rag_hint + "\n\n請用繁體中文回覆。"
            return (augmented, "") if return_sys_addon else augmented

        # ── 0. 計算 insight_score（影響記憶注入量）──
        _insight_score = 0.0
        if not _skip_augment:
            try:
                from rag.pipeline import get_insight_score
                _db_path_for_score = None
                if self.state.current_project:
                    from paths import project_root as _pr_score
                    _db_path_for_score = _pr_score(self.state.current_project["id"]) / "memories.db"
                _insight_score = await get_insight_score(user_msg, _pid, db_path=_db_path_for_score)
            except Exception:
                pass

        # ── 1. 記憶向量搜尋（skip in fast+building/degraded）──
        if not _skip_augment:
            _mem_limit = 5 if _insight_score > 0.4 else 3
            _lancedb_dir = None
            if self.state.current_project:
                from paths import project_root
                _lancedb_dir = project_root(self.state.current_project["id"]) / "lancedb"
            try:
                from memory.vectorstore import search_similar, has_index
                if has_index(lancedb_dir=_lancedb_dir):
                    mem_results = await search_similar(user_msg, limit=_mem_limit, lancedb_dir=_lancedb_dir)
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

        # ── 1.5 視覺偏好注入 ──
        if not _skip_augment:
            try:
                from memory.store import get_memories
                _mem_db = None
                if self.state.current_project:
                    from paths import project_root as _pr2
                    _mem_db = _pr2(self.state.current_project["id"]) / "memories.db"
                prefs = await get_memories(type="preference", limit=1, db_path=_mem_db)
                if prefs and prefs[0].get("content"):
                    pref_text = (
                        "# 視覺偏好分析（系統根據使用者上傳的參考圖片自動生成）\n\n"
                        f"{prefs[0]['content']}\n\n"
                        "當使用者問到「我的圖片」「我的風格偏好」「我收集的東西」等問題時，"
                        "直接引用以上分析結果回答，不要說你看不到圖片。"
                    )
                    if return_sys_addon:
                        sys_addon_parts.append(pref_text)
                    else:
                        augmented.insert(insert_idx, {"role": "system", "content": pref_text})
                        insert_idx += 1
            except Exception:
                pass

        # ── 2. 圖片知識庫檢索（skip in fast+building/degraded）──
        if not _skip_augment:
            try:
                if self.state.current_project:
                    from rag.image_store import search_images
                    img_results = await search_images(
                        self.state.current_project["id"], user_msg, limit=3,
                    )
                    if img_results:
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
                        if img_lines:
                            context_text = f"相關圖片（可引用路徑或描述）：\n{img_lines}"
                            if return_sys_addon:
                                sys_addon_parts.append(context_text)
                            else:
                                augmented.insert(insert_idx, {"role": "system", "content": context_text})
                                insert_idx += 1
            except Exception as e:
                self.log.warning(f"RAG inject images failed: {e}")

        # B3: Clear stale sources before each query to prevent carryover
        self.state.last_rag_sources = []
        await self._update_research_panel("")

        # ── 3. 知識庫 RAG — pipeline ──
        try:
            from rag.pipeline import run_thinking_pipeline
            _db_path = None
            if self.state.current_project:
                from paths import project_root as _pr
                _db_path = _pr(self.state.current_project["id"]) / "memories.db"
            pipeline_result = await run_thinking_pipeline(
                user_input=user_msg,
                project_id=_pid,
                mode="deep" if not _is_fast else "fast",
                db_path=_db_path,
            )
            # Store diagnostics for debugging
            self._last_pipeline_diagnostics = pipeline_result.get("diagnostics", {})

            context_text = pipeline_result.get("context_text", "")
            raw_result = pipeline_result.get("raw_result", "")
            sources = pipeline_result.get("sources", [])

            self.log.info(
                "RAG pipeline: strategy=%s context_len=%d raw_len=%d sources=%d readiness=%s augment_skipped=%s",
                pipeline_result.get("strategy_used"),
                len(context_text), len(raw_result), len(sources),
                _readiness.status_label, _skip_augment,
            )

            # Update research panel with raw result (for display cleaning)
            if raw_result and len(raw_result.strip()) > 10:
                await self._update_research_panel(raw_result)
            else:
                await self._update_research_panel("")
            self.state.last_rag_sources = sources or []

            # Log fallback events
            diag = pipeline_result.get("diagnostics", {})
            if diag.get("deep_error_code"):
                self.log.warning(
                    "RAG deep fallback: %s (strategy=%s)",
                    diag["deep_error_code"], pipeline_result["strategy_used"],
                )

            if context_text and len(context_text.strip()) > 10:
                hint_prefix = f"（{_rag_hint}）\n\n" if _rag_hint else ""
                insight_hint = ""
                if _insight_score > 0.4:
                    insight_hint = "（這個問題和這位創作者過去的洞見高度相關，回答時可以連結他之前的思考。）\n\n"
                injection = (
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
                if return_sys_addon:
                    sys_addon_parts.append(injection)
                else:
                    augmented.insert(insert_idx, {"role": "system", "content": injection})
        except Exception as e:
            self.log.warning(f"RAG pipeline failed: {e}")
            # B3: Also clear panel on failure to prevent stale display
            self.state.last_rag_sources = []
            await self._update_research_panel("")

        if return_sys_addon:
            addon = "\n\n".join(sys_addon_parts)
            if addon:
                addon += "\n\n請用繁體中文回覆。"
            return augmented, addon

        return augmented
