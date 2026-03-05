"""LiteLLM 封裝：統一 LLM 介面"""

import os
from collections.abc import AsyncGenerator

import litellm

DEFAULT_MODEL = os.getenv("LLM_MODEL", "ollama/llama3.2")


async def chat_completion(
    messages: list[dict],
    model: str | None = None,
) -> AsyncGenerator[str, None]:
    """串流 LLM 回應。"""
    target_model = model or DEFAULT_MODEL
    try:
        response = await litellm.acompletion(
            model=target_model,
            messages=messages,
            stream=True,
        )
        async for chunk in response:
            delta = chunk.choices[0].delta
            if delta.content:
                yield delta.content
    except Exception as e:
        raise RuntimeError(f"LLM 呼叫失敗: {e}") from e


def get_available_models() -> list[str]:
    """回傳已設定的可用模型列表。"""
    models = [DEFAULT_MODEL]
    if os.getenv("OPENAI_API_KEY"):
        models.append("gpt-4o")
    if os.getenv("ANTHROPIC_API_KEY"):
        models.append("claude-sonnet-4-20250514")
    return models
