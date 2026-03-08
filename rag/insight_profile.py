"""De-insight v7.5.0 — 洞見加分引擎。

讀取已儲存的 insight 記憶，建立 profile 快取，
在檢索重排時使用 insight_match_score 加權。

洞見只加分，不覆蓋來源約束。
"""

from __future__ import annotations

import logging
import re
import time
from pathlib import Path

log = logging.getLogger(__name__)

# ── Profile cache ──────────────────────────────────────────────────

_profile_cache: dict[str, dict] = {}  # project_id -> profile
_profile_ts: dict[str, float] = {}    # project_id -> last_updated
_CACHE_TTL = 300  # 5 minutes


def _cache_valid(project_id: str) -> bool:
    return (
        project_id in _profile_cache
        and (time.time() - _profile_ts.get(project_id, 0)) < _CACHE_TTL
    )


def invalidate_cache(project_id: str = "") -> None:
    """Invalidate profile cache. Call after saving new insights."""
    if project_id:
        _profile_cache.pop(project_id, None)
        _profile_ts.pop(project_id, None)
    else:
        _profile_cache.clear()
        _profile_ts.clear()


# ── Profile building ───────────────────────────────────────────────

def _extract_keywords(text: str) -> list[str]:
    """Extract meaningful keywords from insight text."""
    # Remove common filler
    text = re.sub(r'[，。！？、；：「」（）\[\]【】]', ' ', text)
    words = [w.strip() for w in text.split() if len(w.strip()) >= 2]
    # Deduplicate preserving order
    seen = set()
    result = []
    for w in words:
        if w not in seen:
            seen.add(w)
            result.append(w)
    return result[:30]


async def _build_profile(project_id: str, db_path=None) -> dict:
    """Build insight profile from stored memories."""
    from memory.store import get_memories

    insights = await get_memories(type="insight", limit=50, db_path=db_path)
    if not insights:
        return {"keywords": [], "topics": [], "categories": [], "count": 0}

    all_keywords = []
    topics = set()
    categories = set()

    for m in insights:
        all_keywords.extend(_extract_keywords(m.get("content", "")))
        t = m.get("topic", "")
        if t:
            topics.add(t)
        c = m.get("category", "")
        if c:
            categories.add(c)

    # Frequency-rank keywords
    freq = {}
    for kw in all_keywords:
        freq[kw] = freq.get(kw, 0) + 1
    ranked = sorted(freq.items(), key=lambda x: -x[1])
    top_keywords = [kw for kw, _ in ranked[:20]]

    profile = {
        "keywords": top_keywords,
        "topics": sorted(topics),
        "categories": sorted(categories),
        "count": len(insights),
    }

    _profile_cache[project_id] = profile
    _profile_ts[project_id] = time.time()
    return profile


# ── Scoring ────────────────────────────────────────────────────────

async def compute_insight_score(
    user_input: str,
    project_id: str,
    db_path=None,
) -> float:
    """Compute how much the user's query aligns with saved insight profile.

    Returns 0.0-1.0. Higher means the query touches on topics/concepts
    the user has previously saved insights about.
    """
    if _cache_valid(project_id):
        profile = _profile_cache[project_id]
    else:
        profile = await _build_profile(project_id, db_path=db_path)

    if not profile["keywords"] and not profile["topics"]:
        return 0.0

    score = 0.0
    input_lower = user_input.lower()

    # Keyword overlap (max 0.6)
    keyword_hits = sum(1 for kw in profile["keywords"] if kw.lower() in input_lower)
    if profile["keywords"]:
        score += min(0.6, keyword_hits / max(len(profile["keywords"]), 1) * 2.0)

    # Topic overlap (max 0.3)
    topic_hits = sum(1 for t in profile["topics"] if t in input_lower)
    if profile["topics"]:
        score += min(0.3, topic_hits / max(len(profile["topics"]), 1) * 1.5)

    # Category bonus (max 0.1)
    cat_hits = sum(1 for c in profile["categories"] if c in input_lower)
    if cat_hits:
        score += 0.1

    return min(1.0, score)
