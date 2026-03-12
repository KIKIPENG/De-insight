"""LLMClient — 統一三條對話路徑的抽象層。

將 Codex CLI / 直接 API / FastAPI 後端三種路徑封裝成統一的 async generator 介面。

NOTE: 此檔案目前未被使用。將來的 _stream_response() 遷移可以漸進式改用本 LLMClient。
"""
from __future__ import annotations

import json
import logging
from typing import AsyncIterator

import httpx

from codex_client import codex_stream, is_codex_available
from settings import load_env

log = logging.getLogger("de-insight.llm")


class LLMClient:
    """統一的 LLM 串流客戶端。

    封裝三條路徑，提供統一的 async generator 介面。
    """

    def __init__(self, api_base: str = "http://127.0.0.1:8200"):
        """初始化 LLMClient。

        Args:
            api_base: FastAPI 後端位址（預設 localhost:8200）
        """
        self.api_base = api_base

    def detect_mode(self) -> str:
        """偵測當前應使用的對話路徑。

        Returns:
            'codex' | 'direct' | 'backend'
        """
        env = load_env()
        model = env.get("LLM_MODEL", "")
        base = env.get("OPENAI_API_BASE", "")

        if model.startswith("codex-cli/"):
            return "codex"
        if model.startswith("gemini/"):
            return "direct"
        if model.startswith("openai/") and bool(base):
            return "direct"
        return "backend"

    async def stream(
        self,
        messages: list[dict],
        system_prompt: str = "",
        mode: str = "emotional",
        project_id: str | None = None,
    ) -> AsyncIterator[str]:
        """統一串流介面。依據 detect_mode() 自動選路徑。

        Args:
            messages: 對話歷史（list of {"role": ..., "content": ...}）
            system_prompt: 系統提示詞
            mode: FastAPI 後端用，感性 / 理性模式
            project_id: 專案 ID（預留欄位）

        Yields:
            文字片段

        Raises:
            RuntimeError: 如果 Codex CLI 未安裝或 API 錯誤
        """
        route = self.detect_mode()
        if route == "codex":
            async for chunk in self._stream_codex(messages, system_prompt):
                yield chunk
        elif route == "direct":
            async for chunk in self._stream_direct(messages, system_prompt):
                yield chunk
        else:
            async for chunk in self._stream_backend(messages, system_prompt, mode, project_id):
                yield chunk

    async def _stream_codex(
        self, messages: list[dict], system_prompt: str
    ) -> AsyncIterator[str]:
        """Codex CLI 路徑。

        將對話歷史和系統提示詞組合成單一 prompt，送給 codex exec。
        """
        if not is_codex_available():
            raise RuntimeError("codex CLI 未安裝。執行: npm i -g @openai/codex")

        env = load_env()
        codex_model = env.get("LLM_MODEL", "").removeprefix("codex-cli/")

        # 建構文字形式的對話內容
        user_msg = messages[-1]["content"] if messages else ""
        context = ""
        for m in messages[:-1]:
            role = "You" if m["role"] == "user" else "De-insight"
            context += f"{role}: {m['content']}\n\n"

        full_prompt = f"{context}You: {user_msg}" if context else user_msg

        # 呼叫 codex_stream
        async for chunk in codex_stream(full_prompt, system_prompt, model=codex_model):
            yield chunk

    async def _stream_direct(
        self, messages: list[dict], system_prompt: str
    ) -> AsyncIterator[str]:
        """直接 API 路徑（OpenAI-compatible SSE）。

        支援 OpenRouter / DeepSeek / Google Gemini / MiniMax 等。
        """
        env = load_env()
        model = env.get("LLM_MODEL", "")
        base = env.get("OPENAI_API_BASE", "")

        # 解析模型名稱和 API Base
        if model.startswith("gemini/"):
            base = base or "https://generativelanguage.googleapis.com/v1beta/openai"
            api_key = env.get("GOOGLE_API_KEY", "") or env.get("GEMINI_API_KEY", "")
            model_name = model.removeprefix("gemini/")
        else:
            api_base_resolved = base or "https://api.openai.com/v1"
            api_key = self._resolve_api_key(env)
            model_name = model.removeprefix("openai/")
            base = api_base_resolved

        # 組裝 API messages
        api_messages = []
        if system_prompt:
            api_messages.append({"role": "system", "content": system_prompt})
        api_messages.extend(messages)

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }
        body = {
            "model": model_name,
            "messages": api_messages,
            "stream": True,
        }

        async with httpx.AsyncClient(timeout=httpx.Timeout(120.0, connect=10.0)) as client:
            async with client.stream(
                "POST",
                f"{base}/chat/completions",
                json=body,
                headers=headers,
            ) as resp:
                if resp.status_code >= 400:
                    err_text = (await resp.aread()).decode("utf-8", errors="replace")[:300]
                    raise RuntimeError(f"API 錯誤 {resp.status_code}: {err_text}")

                async for line in resp.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    data = line[6:]
                    if data.strip() == "[DONE]":
                        break
                    try:
                        obj = json.loads(data)
                        delta = obj.get("choices", [{}])[0].get("delta", {})
                        content = delta.get("content", "")
                        if content:
                            yield content
                    except json.JSONDecodeError:
                        continue

    async def _stream_backend(
        self,
        messages: list[dict],
        system_prompt: str,
        mode: str = "emotional",
        project_id: str | None = None,
    ) -> AsyncIterator[str]:
        """FastAPI 後端路徑。

        向本地 FastAPI backend (/api/chat) 發送 SSE 請求。
        """
        payload = {
            "messages": messages,
            "mode": mode,
        }
        if system_prompt:
            payload["system_prompt"] = system_prompt
        if project_id:
            payload["project_id"] = project_id

        async with httpx.AsyncClient(timeout=httpx.Timeout(120.0, connect=10.0)) as client:
            async with client.stream(
                "POST",
                f"{self.api_base}/api/chat",
                json=payload,
            ) as resp:
                if resp.status_code >= 400:
                    err_text = (await resp.aread()).decode("utf-8", errors="replace")[:300]
                    raise RuntimeError(f"API 錯誤 {resp.status_code}: {err_text}")

                async for line in resp.aiter_lines():
                    if not line:
                        continue
                    if line.startswith("0:"):
                        # 文字內容
                        data = line[2:]
                    elif line.startswith("3:"):
                        # 錯誤信號
                        try:
                            err = json.loads(line[2:])
                            raise RuntimeError(err.get("error", "未知錯誤"))
                        except json.JSONDecodeError:
                            raise RuntimeError(line[2:])
                    elif line.startswith("d:"):
                        # 完成信號
                        break
                    else:
                        continue

                    try:
                        yield data
                    except Exception:
                        continue

    @staticmethod
    def _resolve_api_key(env: dict) -> str:
        """從 OPENAI_API_BASE 推斷應該用哪個 API key env var。

        Args:
            env: 環境變數字典

        Returns:
            對應的 API key 值
        """
        base = env.get("OPENAI_API_BASE", "")
        BASE_TO_KEY = {
            "openrouter.ai": "OPENROUTER_API_KEY",
            "deepseek.com": "DEEPSEEK_API_KEY",
            "minimaxi.chat": "MINIMAX_API_KEY",
            "api.minimax.chat": "MINIMAX_API_KEY",
            "generativelanguage.googleapis.com": "GOOGLE_API_KEY",
        }
        for domain, key_env in BASE_TO_KEY.items():
            if domain in base:
                val = env.get(key_env, "")
                if val:
                    return val
        return env.get("OPENAI_API_KEY", "")
