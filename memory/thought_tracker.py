"""基礎思維追蹤 — Phase 1: SQLite + LLM prompt."""

import json

from memory.store import search_memories, add_memory


def _clean_json(text: str) -> str:
    """Strip markdown code block wrappers from JSON output."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    return text.strip()

EVOLUTION_PROMPT = """\
你是一個中性觀察者。以下是使用者過去的洞見和一條新洞見。
請判斷新洞見與舊洞見之間是否存在「演變」或「矛盾」。

定義：
- 演變（evolution）：思考自然成長，新觀點深化或擴展了舊觀點
- 矛盾（contradiction）：同時持有互斥的觀點，邏輯上不相容

舊洞見：
{old_insights}

新洞見：
{new_insight}

如果偵測到演變或矛盾，回傳 JSON：
{{"type": "evolution" 或 "contradiction", "summary": "一句話描述偵測到什麼", "old": "相關的舊洞見原文", "new": "新洞見原文"}}

如果沒有顯著變化，回傳：
{{"type": "none"}}

只回傳 JSON，不要加任何解釋。
"""

EXTRACT_PROMPT = """\
以下是使用者在對話中說的話。請判斷是否包含值得記錄的內容。
只抽取使用者說的話，不抽 AI 的回應。
只抽以下三類，有則抽，無則回傳空陣列：

1. insight（洞見）：對藝術、設計、創作的獨到觀察或理解
2. question（問題）：使用者提出的深度問題，不是閒聊
3. reaction（反應）：對作品、藝術家、風格的感性反應或身體感受

每條記錄必須附帶一個 topic（主題分類），從以下選擇最接近的：
藝術史、設計史、批判理論、美學、媒材技法、個人創作、當代藝術、
建築、攝影、數位藝術、身體感知、資本主義、個人主義、後殖民、
女性主義、機構批判、策展、教育

如果都不適合，用最精簡的詞自訂（2-4 字）。

使用者的話：
{user_text}

回傳 JSON 陣列，格式：
[{{"type": "insight"|"question"|"reaction", "content": "精簡摘要", "topic": "主題"}}]

如果沒有值得記錄的內容，回傳：[]
只回傳 JSON，不要加任何解釋。
"""


async def check_for_evolution(
    new_insight: str,
    llm_call: callable,
) -> dict | None:
    """Check if a new insight shows evolution or contradiction with past insights."""
    related = await search_memories(new_insight[:20], limit=5)
    if not related:
        return None

    old_text = "\n".join(
        f"- [{m['type']}] {m['content']}" for m in related
    )

    prompt = EVOLUTION_PROMPT.format(
        old_insights=old_text,
        new_insight=new_insight,
    )

    response = await llm_call(prompt)

    try:
        result = json.loads(_clean_json(response))
        if result.get("type") == "none":
            return None
        return result
    except (json.JSONDecodeError, KeyError):
        return None


async def extract_memories(
    user_text: str,
    llm_call: callable,
) -> list[dict]:
    """Extract memorable items from user's message."""
    if not user_text or len(user_text.strip()) < 10:
        return []

    prompt = EXTRACT_PROMPT.format(user_text=user_text)
    response = await llm_call(prompt)

    try:
        items = json.loads(_clean_json(response))
        if not isinstance(items, list):
            return []
        valid = []
        for item in items:
            if item.get("type") in ("insight", "question", "reaction") and item.get("content"):
                valid.append({
                    "type": item["type"],
                    "content": item["content"],
                    "topic": item.get("topic", ""),
                })
        return valid
    except (json.JSONDecodeError, KeyError):
        return []
