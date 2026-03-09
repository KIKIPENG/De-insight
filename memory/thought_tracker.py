"""基礎思維追蹤 — Phase 1: SQLite + LLM prompt."""

import asyncio
import json

from memory.store import search_memories, add_memory, get_memories


def _clean_json(text: str) -> str:
    """清理 LLM 輸出的 JSON，處理常見的格式漂移。"""
    text = text.strip()

    # 移除 markdown fence（多種變體）
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()

    # 移除前導說明文字（常見於小模型）
    # 例：「以下是記憶條目：\n[...]」
    lines = text.split('\n')
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith('[') or stripped.startswith('{'):
            text = '\n'.join(lines[i:])
            break

    # 移除 trailing comma（JSON 不允許，但常見）
    # 處理 ,] 和 ,\n] 等變體
    import re as _re
    text = _re.sub(r',\s*]', ']', text)
    text = _re.sub(r',\s*}', '}', text)

    # 單引號轉雙引號（JSON 規範要求雙引號）
    # 只在整體看起來是 JSON 且單引號佔多數時才做
    if text.startswith('[') or text.startswith('{'):
        if text.count("'") > text.count('"') * 2:
            text = text.replace("'", '"')

    return text.strip()


def _text_overlap(a: str, b: str) -> float:
    """計算兩段文字的重疊率（字符集合交集）。"""
    a, b = a.lower(), b.lower()
    if not a or not b:
        return 0.0
    a_chars = set(a.replace(' ', ''))
    b_chars = set(b.replace(' ', ''))
    if not a_chars or not b_chars:
        return 0.0
    intersection = len(a_chars & b_chars)
    union = len(a_chars | b_chars)
    return intersection / union if union > 0 else 0.0


EVOLUTION_PROMPT = """\
你是思維演變偵測器。

使用者過去的洞見：
{old_insights}

使用者剛才的新洞見：
{new_insight}

任務：判斷新洞見和舊洞見的關係。

**演變（evolution）**：新洞見是對舊洞見的延伸、深化、質疑、或推翻。兩者在討論同一個主題，但觀點有所發展。即使是微小的觀點轉變也算演變。

**矛盾（contradiction）**：新洞見和舊洞見在邏輯上互斥，無法同時為真。

**無關（null）**：新洞見和舊洞見討論的是不同主題，沒有演變關係。

只回傳 JSON，不要有任何其他文字：

如果有演變或矛盾：
{{"type": "evolution", "summary": "一句話描述演變方向", "old": "舊洞見摘要", "new": "新洞見摘要"}}
或
{{"type": "contradiction", "summary": "一句話描述衝突", "old": "舊洞見摘要", "new": "新洞見摘要"}}

如果無關：
{{"type": null}}

範例：
舊：「包豪斯強調功能決定形式」
新：「包豪斯預設功能是穩定的，但功能常是流動的」
→ {{"type": "evolution", "summary": "從肯定功能決定論到質疑其預設", "old": "功能決定形式", "new": "質疑功能穩定性"}}
"""

EXTRACT_PROMPT = """\
以下是使用者在對話中說的話。請判斷是否包含值得記錄的內容。
只抽取使用者說的話，不抽 AI 的回應。
只抽以下三類，有則抽，無則回傳空陣列：

1. insight（洞見）：對藝術、設計、創作的獨到觀察或理解
2. question（問題）：使用者提出的深度問題，不是閒聊
3. reaction（反應）：對作品、藝術家、風格的感性反應或身體感受

每條記錄必須附帶：
- topic（主題分類），從以下選擇最接近的：
  藝術史、設計史、批判理論、美學、媒材技法、個人創作、當代藝術、
  建築、攝影、數位藝術、身體感知、資本主義、個人主義、後殖民、
  女性主義、機構批判、策展、教育
  如果都不適合，用最精簡的詞自訂（2-4 字）。

- category（理解面向），從以下四個選一個：
  思考方式、美學偏好、創作問題、理論連結

不值得記憶的：閒聊、問候、操作性問題、太模糊的發言。

---

範例 1（值得記憶）：
使用者說：「包豪斯那種功能決定形式的說法，其實預設了功能是穩定的。但很多時候功能本身就是流動的，形式反而在定義功能。」

輸出：
[{{"type": "insight", "content": "包豪斯預設功能穩定，但功能常是流動的，形式反過來定義功能", "topic": "設計史", "category": "思考方式"}}]

---

範例 2（不值得記憶）：
使用者說：「好的我知道了，謝謝你的解釋。」

輸出：
[]

---

範例 3（問題）：
使用者說：「那無媒材的概念藝術到底在說什麼？如果藝術不再需要物件，那它的力量是從哪裡來的？」

輸出：
[{{"type": "question", "content": "無媒材概念藝術的力量來源", "topic": "當代藝術", "category": "思考方式"}}]

---

範例 4（反應）：
使用者說：「看 Donald Judd 的作品時，我感覺到一種安靜但不空洞的狀態，好像物件就是物件，不需要再指向別的東西。」

輸出：
[{{"type": "reaction", "content": "Judd 的作品讓我感到安靜但不空洞，物件不指向他物", "topic": "當代藝術", "category": "美學偏好"}}]

---

現在請處理以下使用者的話：

{user_text}

回傳 JSON 陣列，格式：
[{{"type": "insight"|"question"|"reaction", "content": "精簡摘要", "topic": "主題", "category": "思考方式"|"美學偏好"|"創作問題"|"理論連結"}}]

沒有值得記錄的內容就回傳：[]
只回傳 JSON，不要加任何解釋。
"""


async def check_for_evolution(
    new_insight: str,
    llm_call: callable,
    db_path=None,
) -> dict | None:
    """
    新增洞見後呼叫。
    從歷史取出最相關的洞見，讓 LLM 判斷有無演變或矛盾。

    回傳 None（無顯著變化）
    或 {"type": "evolution"|"contradiction", "summary": str, "old": str, "new": str}
    """
    # 1. 搜尋相關記憶（向量搜索 → LIKE fallback → 直接取 insight）
    query = new_insight[:30] if len(new_insight) > 30 else new_insight
    related = await search_memories(query, limit=5, db_path=db_path)
    if not related:
        related = await get_memories(type="insight", limit=5, db_path=db_path)

    # 2. 只保留 insight，排除新洞見本身
    insights = [
        m for m in (related or [])
        if m.get("type") == "insight" and m.get("content", "").strip() != new_insight.strip()
    ]
    if not insights:
        return None

    # 3. 構建 prompt（最多比較 3 條）
    old_text = "\n".join(
        f"{i+1}. {m['content']}" for i, m in enumerate(insights[:3])
    )
    prompt = EVOLUTION_PROMPT.format(
        old_insights=old_text,
        new_insight=new_insight,
    )

    # 4. 呼叫 LLM
    response = await llm_call(prompt, max_tokens=2000)

    # 5. 解析結果
    try:
        cleaned = _clean_json(response)
        result = json.loads(cleaned)

        if not isinstance(result, dict):
            return None

        result_type = result.get("type")
        if result_type is None or result_type not in ("evolution", "contradiction"):
            return None

        if not result.get("summary"):
            return None

        return result
    except (json.JSONDecodeError, KeyError):
        return None


CROSS_MODAL_PROMPT = """\
你是思維矛盾偵測器。

這位創作者的視覺偏好（從他收集的圖片歸納）：
{visual_preference}

這位創作者過去的文字洞見：
{text_insights}

任務：判斷視覺偏好和文字洞見之間是否存在矛盾。

**矛盾**：他收集的圖片呈現的風格傾向，和他在對話中表達的觀點或偏好，明顯不一致。
例如：文字裡強調極簡和克制，但圖片全是粗糙實驗的東西。

**一致**：視覺偏好和文字洞見指向同一個方向，或者各自談不同面向沒有衝突。

只回傳 JSON：
矛盾時：{{"type": "cross_modal", "summary": "一句話描述矛盾", "visual": "視覺偏好摘要", "textual": "文字觀點摘要"}}
一致時：{{"type": null}}
""".strip()


async def check_cross_modal(
    visual_preference: str,
    llm_call: callable,
    db_path=None,
) -> dict | None:
    """比對視覺偏好和文字洞見，偵測跨模態矛盾。

    回傳 None（一致）
    或 {"type": "cross_modal", "summary": str, "visual": str, "textual": str}
    """
    if not visual_preference or len(visual_preference.strip()) < 10:
        return None

    # 取最近的 insight 記憶
    insights = await get_memories(type="insight", limit=5, db_path=db_path)
    if not insights or len(insights) < 2:
        return None  # 洞見太少，不做比對

    insight_text = "\n".join(
        f"- {m['content']}" for m in insights
    )

    prompt = CROSS_MODAL_PROMPT.format(
        visual_preference=visual_preference,
        text_insights=insight_text,
    )

    response = await llm_call(prompt, max_tokens=300)

    try:
        cleaned = _clean_json(response)
        result = json.loads(cleaned)
        if not isinstance(result, dict):
            return None
        if result.get("type") != "cross_modal":
            return None
        if not result.get("summary"):
            return None
        return result
    except (json.JSONDecodeError, KeyError):
        return None


async def extract_memories(
    user_text: str,
    llm_call: callable,
) -> list[dict]:
    """從使用者發言抽取記憶條目。最多重試 2 次。"""
    if not user_text or len(user_text.strip()) < 10:
        return []

    prompt = EXTRACT_PROMPT.format(user_text=user_text)

    for attempt in range(3):
        try:
            response = await llm_call(prompt)
            clean = _clean_json(response)
            items = json.loads(clean)

            if not isinstance(items, list):
                raise ValueError("回傳格式非陣列")

            valid = []
            for item in items:
                if not isinstance(item, dict):
                    continue
                if "type" not in item or "content" not in item:
                    continue
                if item["type"] not in ("insight", "question", "reaction"):
                    continue
                content = item["content"].strip()
                if not content or len(content) < 5:
                    continue
                if _text_overlap(content, user_text) > 0.8:
                    continue
                valid.append({
                    "type": item["type"],
                    "content": content,
                    "topic": item.get("topic", ""),
                    "category": item.get("category", ""),
                })
            return valid

        except (json.JSONDecodeError, ValueError, KeyError):
            if attempt == 2:
                return []
            await asyncio.sleep(0.5 * (attempt + 1))
            continue
        except Exception:
            return []

    return []
