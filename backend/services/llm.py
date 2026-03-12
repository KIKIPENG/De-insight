"""LiteLLM 封裝：統一 LLM 介面（含 Codex Responses API）"""

import json
import os
import re
from collections.abc import AsyncGenerator

import httpx
import litellm
from config.service import get_config_service

def _get_default_model() -> str:
    """每次重新讀取 .env，避免快取舊值。"""
    return get_config_service().get("LLM_MODEL", "ollama/llama3.2")


def _is_codex_model(model: str) -> bool:
    """判斷是否為 Codex 模型（使用 Responses API）。"""
    return model.startswith("codex/")


# Backward compatibility: some endpoints still reference this symbol.
DEFAULT_MODEL = _get_default_model()


async def _codex_stream(
    messages: list[dict],
    model: str,
) -> AsyncGenerator[str, None]:
    """透過 OpenAI Responses API 串流 Codex 回應。"""
    # Codex: prefer CODEX_API_KEY, fallback to OPENAI_API_KEY
    cfg = get_config_service()
    api_key = cfg.get("CODEX_API_KEY", "") or cfg.get("OPENAI_API_KEY", "")
    # Codex 永遠使用 OpenAI 官方 API，不走自訂 base
    base_url = "https://api.openai.com/v1"
    # Strip prefix
    codex_model = model.removeprefix("codex/")

    # Convert chat messages to Responses API input format
    input_items = []
    for msg in messages:
        role = msg["role"]
        if role == "system":
            input_items.append({
                "type": "message",
                "role": "developer",
                "content": msg["content"],
            })
        elif role in ("user", "assistant"):
            input_items.append({
                "type": "message",
                "role": role,
                "content": msg["content"],
            })

    url = f"{base_url.rstrip('/')}/responses"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": codex_model,
        "input": input_items,
        "stream": True,
    }

    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            async with client.stream(
                "POST", url, json=payload, headers=headers,
            ) as response:
                if response.status_code != 200:
                    body = await response.aread()
                    raise RuntimeError(
                        f"Codex API 錯誤 {response.status_code}: {body.decode()}"
                    )
                async for line in response.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    data_str = line[6:]
                    if data_str == "[DONE]":
                        break
                    try:
                        event = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue

                    event_type = event.get("type", "")

                    # Text delta events
                    if event_type == "response.output_text.delta":
                        delta = event.get("delta", "")
                        if delta:
                            yield delta
    except httpx.ConnectError:
        raise RuntimeError("無法連線到 OpenAI API")
    except Exception as e:
        if "Codex API" in str(e):
            raise
        raise RuntimeError(f"Codex 呼叫失敗: {e}") from e


async def chat_completion(
    messages: list[dict],
    model: str | None = None,
) -> AsyncGenerator[str, None]:
    """串流 LLM 回應，自動過濾 <think> 標籤。支援 Codex。"""
    target_model = model or _get_default_model()

    # Codex 走獨立路徑
    if _is_codex_model(target_model):
        async for chunk in _codex_stream(messages, target_model):
            yield chunk
        return

    # 其他模型走 LiteLLM
    buffer = ""
    inside_think = False

    try:
        # 若有自訂 API base，傳給 LiteLLM
        kwargs = {}
        cfg = get_config_service()
        api_base = cfg.get("OPENAI_API_BASE", "")
        
        # Pass API key explicitly to litellm (env vars not inherited by subprocess)
        api_key = cfg.get("OPENAI_API_KEY", "") or cfg.get("OPENROUTER_API_KEY", "")
        if api_key:
            kwargs["api_key"] = api_key
        
        # Apply api_base for OpenRouter or models with "openai/" prefix
        is_openrouter_key = bool(cfg.get("OPENROUTER_API_KEY", ""))
        if api_base:
            kwargs["api_base"] = api_base
        elif is_openrouter_key:
            # OpenRouter key set but no explicit base - use OpenRouter default
            kwargs["api_base"] = "https://openrouter.ai/api/v1"
        response = await litellm.acompletion(
            model=target_model,
            messages=messages,
            stream=True,
            **kwargs,
        )
        async for chunk in response:
            delta = chunk.choices[0].delta
            if not delta.content:
                continue

            buffer += delta.content

            # Still inside a <think> block — check if it closed
            if inside_think:
                close_idx = buffer.find("</think>")
                if close_idx != -1:
                    buffer = buffer[close_idx + len("</think>"):]
                    inside_think = False
                else:
                    continue

            # Check for new <think> opening
            open_idx = buffer.find("<think>")
            if open_idx != -1:
                # Yield text before <think>
                before = buffer[:open_idx]
                if before:
                    yield before
                rest = buffer[open_idx + len("<think>"):]
                close_idx = rest.find("</think>")
                if close_idx != -1:
                    buffer = rest[close_idx + len("</think>"):]
                else:
                    buffer = ""
                    inside_think = True
                continue

            # No think tags — but might have partial "<think" at the end
            safe_boundary = buffer.rfind("<")
            if safe_boundary != -1 and safe_boundary > len(buffer) - 8:
                yield buffer[:safe_boundary]
                buffer = buffer[safe_boundary:]
            else:
                yield buffer
                buffer = ""

        # Flush remaining buffer
        if buffer and not inside_think:
            buffer = re.sub(r"<think>[\s\S]*?</think>", "", buffer)
            if buffer:
                yield buffer
    except Exception as e:
        raise RuntimeError(f"LLM 呼叫失敗: {e}") from e


def get_available_models() -> list[str]:
    """回傳已設定的可用模型列表。"""
    cfg = get_config_service()
    models = [_get_default_model()]
    if cfg.get("OPENAI_API_KEY", ""):
        models.append("gpt-4o")
        models.append("codex/codex-mini-latest")
    if cfg.get("ANTHROPIC_API_KEY", ""):
        models.append("claude-sonnet-4-20250514")
    return models
