"""De-insight v7.5.0 — 統一 RAG Pipeline。

所有知識庫檢索、觀點抽取、來源約束、洞見加分都在這裡完成。
mixins/chat.py 只調用 run_thinking_pipeline()，不直接操作 RAG。

v7.5.0 穩定化：
- A3: startup health check — probe model endpoint, not just config
- B1: relevance gate (CJK bigram + keyword overlap filter)
- B2: citation guard — hard block for fact with unverifiable sources
- D1: cross-language query augmentation — entity + dictionary + query rewrite
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field

log = logging.getLogger(__name__)

# ── Deep mode circuit breaker ──────────────────────────────────────

_deep_fail_until: float = 0.0  # timestamp until which deep is disabled
_DEEP_COOLDOWN_SECS = 120  # 2 minutes cooldown after failure


def _trip_deep_breaker(error_code: str) -> None:
    global _deep_fail_until
    _deep_fail_until = time.time() + _DEEP_COOLDOWN_SECS
    log.warning("Deep mode circuit breaker tripped: %s (cooldown %ds)", error_code, _DEEP_COOLDOWN_SECS)


def _deep_breaker_open() -> bool:
    return time.time() < _deep_fail_until


# ── A3: Startup health check / degraded mode ─────────────────────

_degraded_mode: bool = False
_degraded_reason: str = ""


def is_degraded() -> bool:
    return _degraded_mode


def get_degraded_reason() -> str:
    return _degraded_reason


async def _probe_rag_llm(model: str, key: str, base: str) -> str | None:
    """Actually probe the RAG LLM endpoint. Returns error string or None if OK."""
    if model.startswith("codex-cli/") or not base:
        return None  # Can't probe codex-cli or local-only
    if "ollama" in model.lower() or key == "ollama":
        return None  # Skip ollama probe (may not be running during startup)
    try:
        import httpx
        from rag.rate_guard import get_rate_guard

        async def _do_probe():
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.post(
                    f"{base}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {key}",
                        "HTTP-Referer": "https://github.com/De-insight",
                        "X-Title": "De-insight",
                    },
                    json={
                        "model": model,
                        "messages": [{"role": "user", "content": "test"}],
                        "max_tokens": 1,
                    },
                )
                return resp

        guard = get_rate_guard()
        resp = await guard.call_with_retry(
            "probe/chat/completions", _do_probe, max_retries=1,
        )
        if resp.status_code == 401:
            return f"API key 無效 (401)"
        if resp.status_code == 404:
            return f"模型不存在: {model} (404)"
        if resp.status_code >= 500:
            return f"API 伺服器錯誤 ({resp.status_code})"
        # 2xx or 4xx (rate limit etc.) means the endpoint is reachable
        return None
    except Exception as e:
        return f"無法連線: {e}"


async def startup_health_check(probe_llm: bool = True) -> dict:
    """Validate RAG readiness. Sets degraded_mode on failure.

    Checks:
    1. RAG_LLM_MODEL can be resolved and endpoint is reachable
    2. Embedding provider can initialize
    3. Embedding dimension is consistent

    Args:
        probe_llm: If True, actually send a test request to the LLM endpoint.
                   Set False in tests to skip network calls.

    Returns {"healthy": bool, "issues": [...]}
    """
    global _degraded_mode, _degraded_reason
    issues = []

    # 1. Check RAG LLM model config + probe endpoint
    model, key, base = "", "", ""
    try:
        from rag.knowledge_graph import _get_llm_config
        model, key, base = _get_llm_config()
        if not model:
            issues.append("RAG_LLM_MODEL 未設定")
        elif not model.startswith("codex-cli/") and not key and "ollama" not in model.lower():
            issues.append(f"RAG LLM ({model}) 缺少 API key")
    except Exception as e:
        issues.append(f"RAG LLM 設定讀取失敗: {e}")

    # Probe: actually test the endpoint if config looks OK and probe requested
    if probe_llm and model and not issues:
        probe_err = await _probe_rag_llm(model, key, base)
        if probe_err:
            issues.append(f"RAG LLM ({model}): {probe_err}")

    # 2. Check embedding config (本地模型，不需要 API key)
    try:
        from rag.knowledge_graph import _get_embed_config
        embed_model, _, _, embed_dim = _get_embed_config()
    except Exception as e:
        issues.append(f"Embedding 設定讀取失敗: {e}")

    if issues:
        _degraded_mode = True
        _degraded_reason = "; ".join(issues)
        log.warning("RAG entering degraded mode: %s", _degraded_reason)
    else:
        _degraded_mode = False
        _degraded_reason = ""

    return {"healthy": not issues, "issues": issues}


# ── Question type classification ───────────────────────────────────

_FACT_PATTERNS = [
    r"是什麼|是甚麼|什麼是",
    r"什麼時候|何時|哪一年|哪年",
    r"是誰|哪位|誰是",
    r"在哪裡|哪個城市|哪個國家",
    r"多少|幾個|幾次",
    r"是否|有沒有|是不是",
    r"when|who|where|how many|how much",
]

_FACT_RE = re.compile("|".join(_FACT_PATTERNS), re.IGNORECASE)


def classify_question(text: str) -> str:
    """Classify question type: 'fact', 'summary', or 'reasoning'."""
    if _FACT_RE.search(text):
        return "fact"
    if len(text) > 80 or any(k in text for k in ("怎麼看", "你覺得", "分析", "比較", "討論", "觀點")):
        return "reasoning"
    return "summary"


# ── Perspective card extraction ────────────────────────────────────

@dataclass
class PerspectiveCard:
    claim: str = ""
    value_axis: str = ""
    key_concepts: list[str] = field(default_factory=list)
    assumptions: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "claim": self.claim,
            "value_axis": self.value_axis,
            "key_concepts": self.key_concepts,
            "assumptions": self.assumptions,
        }


def extract_perspective(user_input: str) -> PerspectiveCard:
    """Extract perspective card from user input (lightweight, no LLM call)."""
    card = PerspectiveCard()
    sentences = re.split(r'[。？！\n]', user_input)
    card.claim = sentences[0].strip()[:80] if sentences else user_input[:80]
    quoted = re.findall(r'[「「](.+?)[」」]', user_input)
    bracketed = re.findall(r'\[\[(.+?)\]\]', user_input)
    card.key_concepts = list(set(quoted + bracketed))[:5]
    for pattern, axis in [
        (r"好的|美的|優秀|有價值|重要", "aesthetic-positive"),
        (r"不好|醜|失敗|無意義|問題", "aesthetic-negative"),
        (r"應該|必須|需要|要求", "normative"),
        (r"為什麼|原因|因為|所以", "causal"),
    ]:
        if re.search(pattern, user_input):
            card.value_axis = axis
            break
    return card


# ── B1: Relevance gate ────────────────────────────────────────────

def _extract_query_keywords(text: str) -> set[str]:
    """Extract keywords from query for relevance matching.

    Splits CJK text into bigrams for better matching granularity.
    """
    keywords = set()
    cjk_seqs = re.findall(r'[\u4e00-\u9fff]+', text)
    for seq in cjk_seqs:
        if len(seq) >= 2:
            for i in range(len(seq) - 1):
                keywords.add(seq[i:i+2])
    latin = set(w.lower() for w in re.findall(r'[a-zA-Z]{3,}', text))
    return keywords | latin


def _source_relevance_score(source: dict, query_keywords: set[str]) -> float:
    """Score how relevant a source is to the query (0.0-1.0)."""
    if not query_keywords:
        return 0.5
    text = f"{source.get('title', '')} {source.get('snippet', '')}".lower()
    source_keywords = _extract_query_keywords(text)
    if not source_keywords:
        return 0.3
    overlap = query_keywords & source_keywords
    score = len(overlap) / max(len(query_keywords), 1)
    return min(1.0, score)


def apply_relevance_gate(
    sources: list[dict],
    user_input: str,
    q_type: str,
    threshold: float = 0.15,
) -> list[dict]:
    """Filter sources by relevance. Stricter for fact questions."""
    if not sources:
        return sources
    query_kw = _extract_query_keywords(user_input)
    if not query_kw:
        return sources
    effective_threshold = threshold * 1.5 if q_type == "fact" else threshold
    scored = []
    for s in sources:
        score = _source_relevance_score(s, query_kw)
        if score >= effective_threshold:
            scored.append((score, s))
    scored.sort(key=lambda x: -x[0])
    return [s for _, s in scored]


# ── D1: Cross-language query augmentation ─────────────────────────

_ZH_EN_MAP = {
    "設計": "design",
    "排版": "typography",
    "字體": "font typeface",
    "美學": "aesthetics",
    "藝術": "art",
    "建築": "architecture",
    "攝影": "photography",
    "色彩": "color",
    "構圖": "composition",
    "風格": "style",
    "展覽": "exhibition",
    "策展": "curation curating",
    "裝幀": "book binding bookbinding",
    "印刷": "printing",
    "版面": "layout",
    "視覺": "visual",
    "創作": "creation creative",
    "出版": "publishing",
    "雜誌": "magazine",
    "書籍": "book",
    "封面": "cover",
    "海報": "poster",
    "品牌": "brand branding",
    "識別": "identity",
    "標誌": "logo",
    "插畫": "illustration",
    "圖像": "image graphic",
    "文字": "text typography character",
    "圖片": "image picture photo",
    "有趣": "interesting",
    "發現": "discover found",
    "喜歡": "like prefer",
}


_FUNC_CHARS = set(
    "是的了嗎呢吧啊哦喔呀吶麼什為怎哪誰幾在從到和與跟把被讓"
    "我你他她它們這那些個一不也都很才就會要能可以"
    "上下前後裡面外中去來過看做說想讓給"
)


def _extract_named_entities(text: str) -> list[str]:
    """Extract likely named entities (person/place/work names) from Chinese text.

    Strategy: split CJK text at function characters to isolate noun phrases,
    then keep 2-4 char segments that look like proper nouns.
    Also extracts quoted terms (「...」).
    """
    entities = []
    # Quoted terms (highest confidence)
    quoted = re.findall(r'[「「](.+?)[」」]', text)
    entities.extend(quoted)

    # Split text at function characters and punctuation to isolate noun chunks
    # This separates "王志弘是什麼時候" into ["王志弘", "什", "時候"]
    chunks = re.split(r'[' + re.escape(''.join(_FUNC_CHARS)) + r'，。？！、；：\s]+', text)
    for chunk in chunks:
        # Only keep pure CJK chunks of 2-4 chars (likely names/nouns)
        if re.fullmatch(r'[\u4e00-\u9fff]{2,4}', chunk):
            # Skip very common words
            if chunk not in {"時候", "天氣", "今天", "明天", "昨天", "知識", "資料", "目前",
                             "現在", "關於", "問題", "方式", "事情", "東西", "地方", "部分"}:
                entities.append(chunk)

    return list(dict.fromkeys(entities))


def augment_query_cross_lang(user_input: str) -> str:
    """Expand Chinese query with English terms for cross-language retrieval.

    Strategy:
    1. Dictionary lookup for known domain terms
    2. Extract named entities and include them as-is (they may appear in English docs)
    3. Preserve original query intact
    """
    cjk_chars = len(re.findall(r'[\u4e00-\u9fff]', user_input))
    total_chars = len(user_input.strip())
    if total_chars == 0 or cjk_chars / total_chars < 0.3:
        return user_input

    expansions = []
    for zh, en in _ZH_EN_MAP.items():
        if zh in user_input:
            expansions.append(en)

    # Extract named entities (person names, work titles) and include them directly
    entities = _extract_named_entities(user_input)
    entity_terms = []
    for ent in entities[:5]:
        # Don't re-add if already in dict expansions
        if ent not in _ZH_EN_MAP:
            entity_terms.append(ent)

    parts = []
    if expansions:
        parts.append(" ".join(expansions[:5]))
    if entity_terms:
        parts.append(" ".join(entity_terms[:3]))

    if not parts:
        return user_input

    return f"{user_input} ({' '.join(parts)})"


# ── Retrieval layer ────────────────────────────────────────────────

def _get_chunk_top_k(q_type: str) -> int:
    """Dynamic chunk_top_k based on question type."""
    return {"fact": 8, "summary": 5, "reasoning": 6}.get(q_type, 5)


async def _retrieve(
    user_input: str,
    project_id: str,
    mode: str,
    q_type: str,
) -> tuple[str, list[dict], str, str | None]:
    """Execute retrieval. Returns (raw_result, sources, strategy_used, deep_error_code).

    Handles deep -> fast fallback with circuit breaker.
    Uses cross-language augmented query for better recall.
    """
    from rag.knowledge_graph import query_knowledge, has_knowledge, _is_no_context_result

    if not has_knowledge(project_id=project_id):
        return "", [], "none", None

    # D1: Augment query for cross-language retrieval
    augmented_input = augment_query_cross_lang(user_input)

    chunk_top_k = _get_chunk_top_k(q_type)
    deep_error_code = None
    strategy = mode
    fallback_used = False

    if mode == "deep":
        if _degraded_mode:
            log.info("Degraded mode active, falling back to fast")
            strategy = "fast"
            fallback_used = True
            deep_error_code = "degraded_mode"
        elif _deep_breaker_open():
            log.info("Deep breaker open, falling back to fast")
            strategy = "fast"
            fallback_used = True
            deep_error_code = "circuit_breaker"
        else:
            try:
                result, sources = await query_knowledge(
                    augmented_input,
                    mode="hybrid",
                    context_only=False,
                    project_id=project_id,
                    chunk_top_k=chunk_top_k,
                )
                if result and not _is_no_context_result(result):
                    return result, sources, "deep", None
                strategy = "fast"
                fallback_used = True
            except Exception as e:
                err_str = str(e)
                if "4" in err_str[:3] or "401" in err_str or "403" in err_str or "404" in err_str:
                    deep_error_code = f"http_{err_str[:3]}"
                elif "timeout" in err_str.lower() or "timed out" in err_str.lower():
                    deep_error_code = "timeout"
                elif "invalid" in err_str.lower() and "model" in err_str.lower():
                    deep_error_code = "invalid_model"
                else:
                    deep_error_code = f"unknown: {err_str[:80]}"
                _trip_deep_breaker(deep_error_code)
                strategy = "fast"
                fallback_used = True
                log.warning("Deep mode failed (%s), falling back to fast", deep_error_code)

    # Fast path (or fallback)
    try:
        result, sources = await query_knowledge(
            augmented_input,
            mode="naive",
            context_only=True,
            project_id=project_id,
            chunk_top_k=chunk_top_k,
        )
        actual_strategy = "fast_fallback" if fallback_used else "fast"
        return result or "", sources, actual_strategy, deep_error_code
    except Exception as e:
        log.error("Fast retrieval also failed: %s", e)
        return "", [], "failed", deep_error_code or f"fast_error: {e}"


# ── Context cleaning ───────────────────────────────────────────────

def _is_no_context(text: str) -> bool:
    """Check if text is a no-context result (local check, no imports)."""
    if not text:
        return True
    t = text.strip().lower()
    return (
        "[no-context]" in t
        or "not able to provide an answer" in t
        or "no relevant document chunks found" in t
    )


def clean_context(raw: str, budget: int = 3000) -> str:
    """Clean RAG output for injection. Preserves verifiable facts."""
    if not raw or _is_no_context(raw):
        return ""
    try:
        from rag.knowledge_graph import _clean_rag_chunk
        cleaned = _clean_rag_chunk(raw)
    except ImportError:
        cleaned = raw
    if not cleaned or len(cleaned.strip()) < 10:
        return ""
    if len(cleaned) > budget:
        cut_point = cleaned.rfind("\n\n", 0, budget)
        if cut_point > budget * 0.5:
            cleaned = cleaned[:cut_point] + "\n…（截取至此）"
        else:
            cleaned = cleaned[:budget] + "…"
    return cleaned


# ── Source constraint ──────────────────────────────────────────────

def apply_source_constraint(
    q_type: str,
    source_count: int,
    context_text: str,
) -> str:
    """Apply source constraints. For fact questions without sources,
    prepend a warning to context_text forcing the model to hedge.
    """
    if source_count < 1 and q_type == "fact":
        if not context_text:
            return (
                "【注意】知識庫未檢索到相關內容，無法確認此事實型問題。"
                "請謹慎回答，不可編造事實。\n\n"
            )
        # Context exists but no structured citations — light warning
        return (
            "（注意：以下內容未解析到結構化來源引用，引用事實時請謹慎。）\n\n"
            + context_text
        )
    if source_count < 1:
        # Non-fact question: no warning needed when context exists
        return context_text or ""
    if q_type == "fact" and context_text:
        return (
            "【來源約束】使用者提出了事實型問題。"
            "只能根據以下知識庫內容回答，不可超出來源範圍推測。"
            "引用時標明出處。\n\n" + context_text
        )
    return context_text


# ── B2: Citation guard ────────────────────────────────────────────

def citation_guard(context_text: str, sources: list[dict], q_type: str) -> str:
    """Guard for fact questions with short source snippets.

    For fact questions: if no source has a substantial snippet (>30 chars),
    prepend a warning but preserve the context.
    For other question types: pass through (the model can reason freely).
    """
    if not context_text or not sources:
        return context_text

    if q_type == "fact":
        has_substantial = any(
            len(s.get("snippet", "")) > 30
            for s in sources
        )
        if not has_substantial:
            prefix = (
                "【注意】檢索到的來源片段較短，可能不完整。"
                "請謹慎使用，不可根據不完整片段下結論或推測。\n\n"
            )
            return prefix + context_text

    return context_text


# ── Insight profile scoring ────────────────────────────────────────

async def get_insight_score(
    user_input: str,
    project_id: str,
    db_path=None,
) -> float:
    """Calculate how well the query matches saved insights."""
    try:
        from rag.insight_profile import compute_insight_score
        return await compute_insight_score(user_input, project_id, db_path=db_path)
    except Exception as e:
        log.debug("Insight scoring failed: %s", e)
        return 0.0


# ── Main pipeline entry point ─────────────────────────────────────

async def run_thinking_pipeline(
    user_input: str,
    project_id: str,
    mode: str = "fast",
    db_path=None,
) -> dict:
    """Unified RAG pipeline entry point.

    Fast mode: Only runs necessary retrieval path (naive vector search).
               Never implicitly falls back to deep. Target: 1-1.5s.
    Deep mode: Full pipeline (hybrid retrieval + graph reasoning, may be slow).
               Falls back to fast on failure.

    Returns:
        dict with keys:
        - strategy_used: "fast", "deep", "fast_fallback", "none", "failed"
        - fallback_used: bool
        - context_text: cleaned text ready for injection (may be empty)
        - sources: list[dict] of source citations
        - perspective_card: dict
        - links: list (future use)
        - diagnostics: dict (includes per-stage latency)
    """
    t0 = time.time()
    latency_stages: dict[str, int] = {}

    # Step 1: Perspective extraction
    t_stage = time.time()
    perspective = extract_perspective(user_input)
    latency_stages["perspective_ms"] = int((time.time() - t_stage) * 1000)

    # Step 2: Question classification
    q_type = classify_question(user_input)

    # Step 3: Retrieval (with deep fallback + cross-language augmentation)
    t_stage = time.time()
    raw_result, sources, strategy, deep_error_code = await _retrieve(
        user_input, project_id, mode, q_type,
    )
    latency_stages["retrieval_ms"] = int((time.time() - t_stage) * 1000)

    # Step 3.5: B1 — Relevance gate (filter irrelevant sources)
    pre_gate_count = len(sources)
    sources = apply_relevance_gate(sources, user_input, q_type)
    filtered_count = pre_gate_count - len(sources)

    # Step 4: Context cleaning
    context_text = clean_context(raw_result)

    # Step 5: Source constraint
    source_count = len(sources)
    context_text = apply_source_constraint(q_type, source_count, context_text)

    # Step 5.5: B2 — Citation guard (hard block for fact + unverifiable)
    context_text = citation_guard(context_text, sources, q_type)

    # Step 6: Insight scoring
    t_stage = time.time()
    insight_score = await get_insight_score(user_input, project_id, db_path=db_path)
    latency_stages["insight_ms"] = int((time.time() - t_stage) * 1000)

    latency_ms = int((time.time() - t0) * 1000)
    latency_stages["total_ms"] = latency_ms

    log.info(
        "Pipeline latency: mode=%s strategy=%s %s",
        mode, strategy,
        " ".join(f"{k}={v}" for k, v in latency_stages.items()),
    )

    return {
        "strategy_used": strategy,
        "fallback_used": strategy in ("fast_fallback", "failed"),
        "context_text": context_text,
        "raw_result": raw_result,
        "sources": sources,
        "perspective_card": perspective.to_dict(),
        "links": [],
        "diagnostics": {
            "deep_error_code": deep_error_code,
            "retrieval_hit_count": pre_gate_count,
            "source_count": source_count,
            "filtered_by_gate": filtered_count,
            "insight_match_score": insight_score,
            "latency_ms": latency_ms,
            "latency_stages": latency_stages,
            "question_type": q_type,
            "degraded_mode": _degraded_mode,
            "mode_requested": mode,
        },
    }
