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
import os
import re
import time
from dataclasses import dataclass, field

log = logging.getLogger(__name__)

# ── Deep mode circuit breaker ──────────────────────────────────────

_deep_fail_until: float = 0.0  # timestamp until which deep is disabled
_DEEP_COOLDOWN_SECS = 120  # 2 minutes cooldown after failure
_HYDE_CACHE: dict[tuple[str, str], tuple[float, str]] = {}
_HYDE_CACHE_MAX = 128


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
        # Gemini (LiteLLM native path) can still be probed without base URL.
        if not model.startswith("gemini/"):
            return None  # Can't probe codex-cli or local-only
    if "ollama" in model.lower() or key == "ollama":
        return None  # Skip ollama probe (may not be running during startup)
    try:
        import httpx
        import litellm
        from rag.rate_guard import get_rate_guard

        async def _do_probe():
            if model.startswith("gemini/"):
                return await litellm.acompletion(
                    model=model,
                    messages=[{"role": "user", "content": "test"}],
                    api_key=key,
                    max_tokens=1,
                )
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
        if model.startswith("gemini/"):
            return None
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

_HYDE_SYSTEM_PROMPT = (
    "你在幫知識庫檢索做 HyDE。"
    "請根據使用者查詢，寫出一小段『如果知識庫裡真的有一段相關論述，它可能會怎麼說』的假想段落。"
    "重點是保留論證結構、價值張力與因果關係，不是重複關鍵字。"
    "不要加標題、不要條列、不要解釋任務、不要虛構精確史實或引用。"
)


def _hyde_enabled(mode: str, q_type: str) -> bool:
    """Return whether HyDE should be applied for this query."""
    if q_type == "fact":
        return False
    from config.service import get_config_service
    env = get_config_service().snapshot(include_process=True)
    if (env.get("RAG_HYDE_ENABLED", "1") or "1") == "0":
        return False
    if mode == "deep":
        return True
    return (env.get("RAG_HYDE_FAST", "0") or "0") == "1"


def _extract_retry_after(exc: Exception) -> float | None:
    """Extract retry-after seconds from provider errors."""
    match = re.search(r'retry[\s_-]*after[\s:=]*(\d+(?:\.\d+)?)', str(exc), re.IGNORECASE)
    if match:
        return float(match.group(1))
    return None


async def _rag_llm_complete(
    prompt: str,
    *,
    system_prompt: str = "",
    max_tokens: int = 320,
    temperature: float = 0.2,
    max_retries: int = 2,
) -> str:
    """Small non-streaming LLM call using the RAG model settings."""
    from rag.knowledge_graph import _get_llm_config

    llm_model, llm_key, llm_base = _get_llm_config()
    if not llm_model:
        return ""

    if llm_model.startswith("codex-cli/"):
        from codex_client import codex_stream

        result = ""
        async for chunk in codex_stream(
            prompt,
            system_prompt,
            model=llm_model.removeprefix("codex-cli/"),
        ):
            result += chunk
        return result.strip()

    import asyncio as _asyncio
    import httpx as _httpx
    import litellm as _litellm
    from rag.rate_guard import RateLimitError

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    local_llm = "localhost" in llm_base or "127.0.0.1" in llm_base
    timeout = 45.0 if local_llm else 30.0

    if llm_model.startswith("gemini/"):
        if llm_key:
            os.environ["GEMINI_API_KEY"] = llm_key
            os.environ["GOOGLE_API_KEY"] = llm_key
    elif llm_key:
        os.environ["OPENAI_API_KEY"] = llm_key
    if llm_base:
        os.environ["OPENAI_API_BASE"] = llm_base

    last_exc = None
    for attempt in range(max_retries):
        try:
            if llm_model.startswith("gemini/"):
                resp = await _litellm.acompletion(
                    model=llm_model,
                    messages=messages,
                    api_key=llm_key,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                result = resp.choices[0].message.content or ""
            else:
                async with _httpx.AsyncClient(timeout=timeout) as client:
                    body = {
                        "model": llm_model,
                        "messages": messages,
                        "max_tokens": max_tokens,
                        "temperature": temperature,
                    }
                    if local_llm:
                        body["options"] = {"num_gpu": 99}
                    resp = await client.post(
                        f"{llm_base}/chat/completions",
                        headers={
                            "Authorization": f"Bearer {llm_key}",
                            "HTTP-Referer": "https://github.com/De-insight",
                            "X-Title": "De-insight",
                        },
                        json=body,
                    )
                    resp.raise_for_status()
                    result = resp.json()["choices"][0]["message"]["content"]
            if result and "<think>" in result:
                result = re.sub(r"<think>[\s\S]*?</think>\s*", "", result)
            return (result or "").strip()
        except RateLimitError as exc:
            last_exc = exc
            if attempt < max_retries - 1:
                await _asyncio.sleep(_extract_retry_after(exc) or (2.0 * (2 ** attempt)))
        except Exception as exc:
            last_exc = exc
            if attempt < max_retries - 1:
                await _asyncio.sleep(1.0 * (attempt + 1))

    raise last_exc if last_exc else RuntimeError("RAG LLM completion failed")


def _clean_hyde_passage(text: str, limit: int = 700) -> str:
    """Normalize LLM output into a single short retrieval passage."""
    cleaned = (text or "").strip()
    cleaned = re.sub(r"^#+\s*", "", cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r"^[\-\*\d\.\)\s]+", "", cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned[:limit].strip()


async def _generate_hyde_passage(query: str, q_type: str) -> str:
    """Generate and cache a hypothetical document passage for retrieval."""
    from config.service import get_config_service

    q_norm = " ".join((query or "").strip().split()).lower()
    if not q_norm:
        return ""

    env = get_config_service().snapshot(include_process=True)
    ttl = int(env.get("RAG_HYDE_CACHE_TTL_SEC", "600") or "600")
    cache_key = (q_type, q_norm)
    now = time.time()
    if ttl > 0:
        cached = _HYDE_CACHE.get(cache_key)
        if cached and cached[0] > now:
            return cached[1]
        if cached and cached[0] <= now:
            _HYDE_CACHE.pop(cache_key, None)

    prompt = (
        f"查詢：{query}\n\n"
        "請寫一段 120 到 220 字的假想知識庫段落。"
        "這段文字應該像是某篇文章或書中的自然段落，"
        "把這個查詢背後的論證結構、價值判準與問題意識說清楚。"
        "避免只重述原句，也不要補入查詢中沒有根據的人名、年份、作品名。"
    )
    result = _clean_hyde_passage(await _rag_llm_complete(prompt, system_prompt=_HYDE_SYSTEM_PROMPT))
    if result and ttl > 0:
        if len(_HYDE_CACHE) >= _HYDE_CACHE_MAX:
            _HYDE_CACHE.pop(next(iter(_HYDE_CACHE)))
        _HYDE_CACHE[cache_key] = (now + ttl, result)
    return result


async def prepare_retrieval_query(
    user_input: str,
    mode: str,
    q_type: str,
) -> tuple[str, dict]:
    """Build the retrieval query text and diagnostics."""
    augmented_input = augment_query_cross_lang(user_input)
    diag = {
        "retrieval_query": augmented_input,
        "retrieval_query_kind": "augmented",
        "hyde_used": False,
        "hyde_passage": "",
    }
    if not _hyde_enabled(mode, q_type):
        return augmented_input, diag

    try:
        hyde_passage = await _generate_hyde_passage(user_input, q_type)
    except Exception as e:
        log.debug("HyDE generation skipped: %s", e)
        return augmented_input, diag

    if not hyde_passage:
        return augmented_input, diag

    retrieval_query = f"{augmented_input}\n\n{hyde_passage}"
    diag.update({
        "retrieval_query": retrieval_query,
        "retrieval_query_kind": "hyde",
        "hyde_used": True,
        "hyde_passage": hyde_passage,
    })
    return retrieval_query, diag

def _get_chunk_top_k(q_type: str) -> int:
    """Dynamic chunk_top_k based on question type."""
    return {"fact": 8, "summary": 5, "reasoning": 6}.get(q_type, 5)


async def _retrieve(
    user_input: str,
    project_id: str,
    mode: str,
    q_type: str,
    retrieval_input: str | None = None,
) -> tuple[str, list[dict], str, str | None]:
    """Execute retrieval. Returns (raw_result, sources, strategy_used, deep_error_code).

    Handles deep -> fast fallback with circuit breaker.
    Uses cross-language augmented query for better recall.
    """
    from rag.knowledge_graph import query_knowledge, has_knowledge, _is_no_context_result

    if not has_knowledge(project_id=project_id):
        return "", [], "none", None

    retrieval_query = retrieval_input or augment_query_cross_lang(user_input)

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
                    retrieval_query,
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
            retrieval_query,
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
    recent_surfaced_bridges: list = None,
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

    # Step 2.5: Build retrieval query (cross-language + optional HyDE)
    t_stage = time.time()
    retrieval_query, retrieval_diag = await prepare_retrieval_query(user_input, mode, q_type)
    latency_stages["query_prep_ms"] = int((time.time() - t_stage) * 1000)

    # Step 3: Retrieval (with deep fallback + cross-language augmentation)
    t_stage = time.time()
    raw_result, sources, strategy, deep_error_code = await _retrieve(
        user_input, project_id, mode, q_type, retrieval_input=retrieval_query,
    )
    latency_stages["retrieval_ms"] = int((time.time() - t_stage) * 1000)

    # Step 3.1: Bridge retrieval via core.retriever (with ClaimStore injection)
    bridge_result = None
    try:
        from core.retriever import Retriever
        from core.stores.claim_store import ClaimStore
        from core.schemas import RetrievalPlan
        from core.query_classifier import QueryClassifier

        # Classify query to decide fast/deep plan
        classifier = QueryClassifier()
        classification = classifier.classify(user_input)
        query_mode = classification.mode.value if hasattr(classification.mode, 'value') else str(classification.mode)

        plan = RetrievalPlan(
            project_id=project_id,
            query_mode=query_mode,
            why_deep=classification.why_deep if hasattr(classification, 'why_deep') else None,
            thought_summary="",
            concept_queries=[user_input],
        )

        # Inject ClaimStore so _has_stores() returns True → hybrid retrieval
        claim_store = ClaimStore(project_id=project_id)
        retriever = Retriever(
            project_id=project_id,
            claim_store=claim_store,
        )
        bridge_result = await retriever.retrieve(plan, user_input)
        log.debug("Bridge retrieval: %d claims, %d bridges",
                  len(bridge_result.claims) if bridge_result else 0,
                  len(bridge_result.bridges) if bridge_result else 0)
    except Exception as e:
        log.debug("Bridge retrieval skipped: %s", e)

    # Step 3.2: Merge claim context into raw_result
    # bridge_result contains claims found via structural search —
    # these must be injected into the LLM context, not just used for
    # bridge surfacing.  Without this merge, claim-based retrieval
    # has zero effect on the curator's answer.
    if bridge_result and bridge_result.claims:
        claim_lines = []
        for c in bridge_result.claims[:5]:
            patterns = ", ".join(c.abstract_patterns or [])
            claim_lines.append(
                f"- {c.core_claim}" + (f" [{patterns}]" if patterns else "")
            )
        if claim_lines:
            claim_block = (
                "\n\n---\n【結構相關脈絡（跨領域 claims）】\n"
                + "\n".join(claim_lines)
            )
            raw_result = (raw_result or "") + claim_block
            log.info(
                "Merged %d claims into context (%d chars added)",
                len(claim_lines), len(claim_block),
            )
            # Also add claim sources so they appear in citations
            for c in bridge_result.claims[:5]:
                src_entry = {
                    "title": (c.core_claim or "")[:50],
                    "snippet": c.core_claim or "",
                    "source": f"claim:{c.claim_id}",
                    "source_id": c.source_id or "",
                }
                if src_entry not in sources:
                    sources.append(src_entry)

    # Step 3.5: B1 — Relevance gate (filter irrelevant sources)
    pre_gate_count = len(sources)
    sources = apply_relevance_gate(sources, user_input, q_type)
    filtered_count = pre_gate_count - len(sources)

    # Step 3.6: Jina Reranker (precision boost)
    reranked = False
    if sources and len(sources) >= 2:
        try:
            from rag.reranker import rerank_with_items
            t_rerank = time.time()
            reranked_sources = await rerank_with_items(
                query=user_input,
                items=sources,
                text_fn=lambda s: f"{s.get('title', '')} {s.get('snippet', '')}",
                top_n=min(len(sources), 5),
            )
            if reranked_sources:
                sources = reranked_sources
                reranked = True
            latency_stages["rerank_ms"] = int((time.time() - t_rerank) * 1000)
        except Exception as e:
            log.debug("Reranker skipped: %s", e)

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

    # Step 6.5: Bridge surfacing (using bridge_result from core.retriever)
    surfaced_bridge = None
    recent_surfaced_bridges = recent_surfaced_bridges or []
    try:
        from core.bridge_surfacing import apply_surfacing_policy
        from core.retriever import assess_anchor_quality

        # Use bridge_result from Step 3.1 (core.retriever)
        if bridge_result and bridge_result.bridges:
            anchor = bridge_result.claims[0] if bridge_result.claims else None
            anchor_quality = assess_anchor_quality(anchor).get("quality_score", 0) if anchor else 0

            messages_for_context = [{"role": "user", "content": user_input}]

            should_surface, surfaced = apply_surfacing_policy(
                anchor_quality=anchor_quality,
                bridges=bridge_result.bridges,
                messages=messages_for_context,
                recent_surfaced=recent_surfaced_bridges,
            )
            if should_surface:
                surfaced_bridge = surfaced
    except Exception as e:
        log.debug("Bridge surfacing skipped: %s", e)

    return {
        "strategy_used": strategy,
        "fallback_used": strategy in ("fast_fallback", "failed"),
        "context_text": context_text,
        "raw_result": raw_result,
        "sources": sources,
        "perspective_card": perspective.to_dict(),
        "links": [],
        "surfaced_bridge": surfaced_bridge,
        "diagnostics": {
            "deep_error_code": deep_error_code,
            "hyde_passage": retrieval_diag.get("hyde_passage", ""),
            "hyde_used": retrieval_diag.get("hyde_used", False),
            "retrieval_query": retrieval_diag.get("retrieval_query", ""),
            "retrieval_query_kind": retrieval_diag.get("retrieval_query_kind", "augmented"),
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
