"""
測試批評視角注入效果
用同一個中立問題，分別在無視角 / 包浩斯 / 達達主義下取得 LLM 回覆，比較差異。

使用方式：
    cd /path/to/De-insight
    python tests/test_persona_injection.py

如果 OpenRouter 報錯，可改用其他 provider：
    LLM_MODEL=anthropic/claude-sonnet-4-6 ANTHROPIC_API_KEY=sk-xxx python tests/test_persona_injection.py
"""

import sys, os, json, httpx

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

from persona.store import set_active_ids, toggle_persona, build_persona_prompt_block
from backend.prompts.curator import get_system_prompt

# ── 設定 ──
QUESTION = "什麼是好的設計？"

# 偵測 provider
LLM_MODEL = os.getenv("LLM_MODEL", "")
OPENROUTER_KEY = os.getenv("OPENROUTER_API_KEY", "") or os.getenv("OPENAI_API_KEY", "")
ANTHROPIC_KEY = os.getenv("ANTHROPIC_API_KEY", "")
OPENAI_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE", "")

def _call_openrouter(messages: list[dict], model: str) -> str:
    """直接打 OpenRouter REST API。"""
    # model 格式: openai/deepseek/deepseek-chat-v3-0324 → 去掉 openai/ prefix
    api_model = model.removeprefix("openai/")
    resp = httpx.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {OPENROUTER_KEY}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://de-insight.local",
        },
        json={
            "model": api_model,
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": 600,
        },
        timeout=90,
    )
    data = resp.json()
    if "choices" in data:
        return data["choices"][0]["message"]["content"]
    else:
        raise RuntimeError(f"API error: {json.dumps(data, ensure_ascii=False)[:500]}")


def _call_anthropic(messages: list[dict], model: str) -> str:
    """直接打 Anthropic Messages API。"""
    api_model = model.removeprefix("anthropic/")
    # 分離 system 和對話 messages
    system_text = ""
    chat_msgs = []
    for m in messages:
        if m["role"] == "system":
            system_text += m["content"] + "\n"
        else:
            chat_msgs.append({"role": m["role"], "content": m["content"]})

    body = {
        "model": api_model,
        "max_tokens": 600,
        "temperature": 0.7,
        "messages": chat_msgs,
    }
    if system_text.strip():
        body["system"] = system_text.strip()

    resp = httpx.post(
        "https://api.anthropic.com/v1/messages",
        headers={
            "x-api-key": ANTHROPIC_KEY,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        },
        json=body,
        timeout=90,
    )
    data = resp.json()
    if "content" in data:
        return data["content"][0]["text"]
    else:
        raise RuntimeError(f"API error: {json.dumps(data, ensure_ascii=False)[:500]}")


def _call_openai_compat(messages: list[dict], model: str) -> str:
    """OpenAI 相容 API（含自訂 base）。"""
    base = OPENAI_API_BASE.rstrip("/") if OPENAI_API_BASE else "https://api.openai.com/v1"
    resp = httpx.post(
        f"{base}/chat/completions",
        headers={
            "Authorization": f"Bearer {OPENAI_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": model.removeprefix("openai/"),
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": 600,
        },
        timeout=90,
    )
    data = resp.json()
    if "choices" in data:
        return data["choices"][0]["message"]["content"]
    else:
        raise RuntimeError(f"API error: {json.dumps(data, ensure_ascii=False)[:500]}")


def call_llm(messages: list[dict]) -> str:
    """根據 .env 設定自動選擇 provider。"""
    if "openrouter" in OPENAI_API_BASE.lower():
        return _call_openrouter(messages, LLM_MODEL)
    elif LLM_MODEL.startswith("anthropic/"):
        return _call_anthropic(messages, LLM_MODEL)
    else:
        return _call_openai_compat(messages, LLM_MODEL)


CONFIGS = [
    ("A — 無視角（基線）", []),
    ("B — 包浩斯", ["bauhaus"]),
    ("C — 達達主義", ["dadaism"]),
]


def ask(label: str, persona_ids: list[str]) -> str:
    set_active_ids([])
    for pid in persona_ids:
        toggle_persona(pid)

    persona_block = build_persona_prompt_block()
    system_prompt = get_system_prompt("rational", persona_block=persona_block)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": QUESTION},
    ]

    print(f"\n{'=' * 60}")
    print(f"【{label}】")
    print(f"視角: {persona_ids if persona_ids else '無'}")
    print(f"system prompt 長度: {len(system_prompt)} 字")
    print(f"persona block 長度: {len(persona_block)} 字")
    print(f"{'=' * 60}")

    try:
        content = call_llm(messages)
        print(content)
        set_active_ids([])
        return content
    except Exception as e:
        print(f"ERROR: {e}")
        set_active_ids([])
        return ""


def main():
    provider = "OpenRouter" if "openrouter" in OPENAI_API_BASE.lower() else (
        "Anthropic" if LLM_MODEL.startswith("anthropic/") else "OpenAI-compat"
    )
    print(f"Provider: {provider}")
    print(f"模型: {LLM_MODEL}")
    print(f"問題: 「{QUESTION}」")
    print(f"測試組數: {len(CONFIGS)}")

    results = {}
    for label, pids in CONFIGS:
        content = ask(label, pids)
        results[label] = content

    # 比較摘要
    print(f"\n{'=' * 60}")
    print("【比較摘要】")
    print(f"{'=' * 60}")
    for label, content in results.items():
        print(f"\n{label}:")
        print(f"  回覆長度: {len(content)} 字")
        preview = content.replace("\n", " ")[:100]
        print(f"  開頭: {preview}...")


if __name__ == "__main__":
    main()
