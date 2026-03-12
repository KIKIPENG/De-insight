"""LightRAG 知識圖譜封裝。"""

import asyncio
import json as _json
import logging
import os
import random
import time
from pathlib import Path

from lightrag import LightRAG, QueryParam
from lightrag.utils import EmbeddingFunc

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config.service import get_config_service
from settings import load_env  # backward-compatible export for tests/patch points

from paths import project_root, ensure_project_dirs, DATA_ROOT, GLOBAL_PROJECT_ID

log = logging.getLogger(__name__)
_DEFAULT_LIGHTRAG_DIR = DATA_ROOT / "projects" / "default" / "lightrag"

ART_ENTITY_TYPES = [
    "藝術家", "設計師", "建築師", "理論家", "批評家",
    "藝術運動", "設計風格", "藝術流派",
    "媒材", "技法", "創作方法",
    "哲學概念", "批判理論",
    "藝術機構", "美術館", "畫廊",
    "展覽", "作品", "著作", "批評文本",
    "空間", "場所", "地景",
    "社會現象", "文化現象",
    "策展概念", "展覽形式",
    "身體", "感知", "凝視", "日常生活",
    "都市", "類型學", "生產方式", "資本",
    "主體性", "身份認同", "權力關係", "他者",
    "物質性", "技術", "物件",
    "歷史時期", "集體記憶", "遺產",
    "敘事", "再現", "圖像", "表演性",
]

_rag_instance: LightRAG | None = None
_rag_project_id: str | None = None
_rag_lock = None  # asyncio.Lock, lazy init (event loop not available at import time)


def _get_rag_lock():
    """Lazy-init asyncio.Lock to avoid event loop issues at import time."""
    global _rag_lock
    if _rag_lock is None:
        _rag_lock = asyncio.Lock()
    return _rag_lock


# ── Query cache (process-local) ───────────────────────────────────
_QUERY_CACHE: dict[tuple[str, str, bool, int, str], tuple[float, str, list[dict]]] = {}
_QUERY_CACHE_MAX = 256

# ── Module-level progress counters (worker process only) ──
# llm_func / embed_func inside _ensure_initialized() increment these.
# _ProgressTracker reads them via background task → writes to DB.
_progress_llm_calls: int = 0
_progress_embed_texts: int = 0


def _get_llm_config() -> tuple[str, str, str]:
    """Return (model, api_key, api_base) from current env settings.

    可在 .env 設定 RAG_LLM_MODEL / RAG_API_KEY / RAG_API_BASE
    來獨立控制知識庫用的 LLM，不受主聊天模型影響。
    """
    env = get_config_service().snapshot(include_process=True)

    # 優先使用獨立的 RAG LLM 設定
    rag_model = env.get("RAG_LLM_MODEL", "")
    if rag_model:
        rag_key = env.get("RAG_API_KEY", "") or env.get("OPENAI_API_KEY", "")
        rag_base = env.get("RAG_API_BASE", "") or env.get("OPENAI_API_BASE", "https://api.openai.com/v1")
        if rag_model.startswith("ollama/"):
            return rag_model.removeprefix("ollama/"), "ollama", "http://localhost:11434/v1"
        # Google AI Studio (Gemini): use LiteLLM native Gemini path instead of
        # forcing OpenAI-compatible /chat/completions.
        # Keep OpenRouter models (e.g. google/gemini-*) untouched when base is OpenRouter.
        _base_lower = rag_base.lower()
        _is_openrouter_base = "openrouter.ai" in _base_lower
        _is_google_base = "generativelanguage.googleapis.com" in _base_lower
        _is_gemini_like = rag_model.startswith("gemini-") or rag_model.startswith("gemini/")
        if (_is_gemini_like or _is_google_base) and not _is_openrouter_base:
            if rag_model.startswith("gemini/"):
                _gemini_model = rag_model
            elif rag_model.startswith("google/"):
                _gemini_model = f"gemini/{rag_model.removeprefix('google/')}"
            else:
                _gemini_model = f"gemini/{rag_model.removeprefix('gemini/')}"
            # Guardrail: Google endpoint currently rejects gemini-1.5-flash
            # (404 model not found / unsupported). Auto-upgrade to 2.5-flash.
            if _gemini_model in ("gemini/gemini-1.5-flash", "gemini-1.5-flash"):
                log.warning(
                    "RAG_LLM_MODEL=%s 已不可用，改用 gemini/gemini-2.5-flash",
                    _gemini_model,
                )
                _gemini_model = "gemini/gemini-2.5-flash"
            rag_key = env.get("RAG_API_KEY", "") or env.get("GOOGLE_API_KEY", "") or rag_key
            return _gemini_model, rag_key, ""
        # Strip openai/ prefix — API endpoint expects bare model name
        if rag_model.startswith("openai/"):
            rag_model = rag_model.removeprefix("openai/")
        return rag_model, rag_key, rag_base

    model = env.get("LLM_MODEL", "ollama/llama3.2")

    # Codex CLI: 標記使用 codex_stream
    if model.startswith("codex-cli/"):
        return f"codex-cli/{model.removeprefix('codex-cli/')}", "", ""
    # Codex API: use the underlying OpenAI model
    elif model.startswith("codex/"):
        model = model.removeprefix("codex/")

    # Strip openai/ prefix for the OpenAI client
    if model.startswith("openai/"):
        model = model.removeprefix("openai/")

    api_key = env.get("OPENAI_API_KEY", "") or env.get("CODEX_API_KEY", "")
    api_base = env.get("OPENAI_API_BASE", "")

    # Ollama: use local endpoint
    if "ollama" in (env.get("LLM_MODEL", "")):
        api_base = "http://localhost:11434/v1"
        api_key = "ollama"
        model = model.removeprefix("ollama/")

    return model, api_key, api_base or "https://api.openai.com/v1"


def _get_embed_config() -> tuple[str, str, str, int]:
    """Return (model, api_key, api_base, dim) for embeddings.

    v0.8: 透過 EmbeddingService 取得。
    """
    from embeddings.service import get_embedding_service
    return get_embedding_service().get_embed_config()


def _detect_vdb_dim(working_dir: Path) -> int | None:
    """讀取 vdb_chunks.json 中的 embedding_dim。不存在或無法解析時回傳 None。"""
    vdb_path = working_dir / "vdb_chunks.json"
    if not vdb_path.exists():
        return None
    try:
        data = _json.loads(vdb_path.read_text(encoding="utf-8"))
        dim = data.get("embedding_dim")
        return int(dim) if dim is not None else None
    except Exception:
        return None


def _clear_stale_vdb(working_dir: Path, old_dim: int, new_dim: int) -> None:
    """Embedding 維度變更時，清除舊的向量索引檔，保留文件記錄。"""
    log.warning(
        "Embedding 維度變更: %d → %d，清除舊向量索引 (%s)",
        old_dim, new_dim, working_dir,
    )
    # 清除所有向量索引（含 embedding_dim 斷言的檔案）
    for name in (
        "vdb_chunks.json", "vdb_entities.json",
        "vdb_relationships.json", "vdb_text_chunks.json",
    ):
        p = working_dir / name
        if p.exists():
            p.unlink()
    # 清除 doc_status，讓 LightRAG 視文件為新（可重新處理）
    status_path = working_dir / "kv_store_doc_status.json"
    if status_path.exists():
        status_path.unlink()


def _apply_env() -> None:
    """Set env vars for LightRAG's OpenAI client."""
    env = get_config_service().snapshot(include_process=True)
    for key in (
        "OPENAI_API_KEY", "OPENAI_API_BASE",
        "ANTHROPIC_API_KEY", "CODEX_API_KEY",
        "MINIMAX_API_KEY", "MINIMAX_API_BASE",
        "GOOGLE_API_KEY",
    ):
        val = env.get(key, "")
        if val:
            os.environ[key] = val


def get_rag(project_id: str = "default") -> LightRAG:
    global _rag_instance, _rag_project_id
    if _rag_instance is not None and _rag_project_id == project_id:
        return _rag_instance
    # Different project or first call — create new instance
    _rag_instance = None

    _apply_env()
    llm_model, llm_key, llm_base = _get_llm_config()
    embed_model, embed_key, embed_base, embed_dim = _get_embed_config()

    _is_local_llm = "localhost" in llm_base or "127.0.0.1" in llm_base
    _llm_timeout = 300.0 if _is_local_llm else 120.0

    if llm_model.startswith("codex-cli/"):
        # 用 Codex CLI 的 OAuth 認證，不需要 API key
        codex_model_name = llm_model.removeprefix("codex-cli/")

        async def llm_func(prompt, system_prompt=None, history_messages=None, **kwargs):
            from codex_client import codex_stream
            sys_p = system_prompt or ""
            result = ""
            async for chunk in codex_stream(prompt, sys_p, model=codex_model_name):
                result += chunk
            global _progress_llm_calls
            _progress_llm_calls += 1
            return result
    else:
        async def llm_func(prompt, system_prompt=None, history_messages=None, **kwargs):
            import re as _re
            import httpx as _httpx
            import litellm as _litellm
            from rag.rate_guard import get_rate_guard
            # Drop response_format — reasoning models (MiniMax-M2.5) return
            # <think> tags that break structured output validation.
            kwargs.pop("response_format", None)
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            for m in (history_messages or []):
                messages.append(m)
            messages.append({"role": "user", "content": prompt})

            async def _call_llm():
                # Gemini family: use LiteLLM native provider routing.
                if llm_model.startswith("gemini/"):
                    resp = await _litellm.acompletion(
                        model=llm_model,
                        messages=messages,
                        api_key=llm_key,
                    )
                    return (resp.choices[0].message.content or "")

                async with _httpx.AsyncClient(timeout=_llm_timeout) as client:
                    body = {"model": llm_model, "messages": messages}
                    # Ollama local: use GPU — serialize with embedding via max_async=1
                    if _is_local_llm:
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
                    if resp.status_code >= 400:
                        import logging as _log
                        _log.getLogger("rag.llm").error(
                            f"RAG LLM {resp.status_code} ({llm_model}): {resp.text[:500]}"
                        )
                    resp.raise_for_status()
                    return resp.json()["choices"][0]["message"]["content"]

            guard = get_rate_guard()
            result = await guard.call_with_retry(
                "rag/chat/completions", _call_llm, max_retries=3,
            )
            # Strip <think>...</think> from reasoning models
            if result and "<think>" in result:
                result = _re.sub(r"<think>[\s\S]*?</think>\s*", "", result)
            global _progress_llm_calls
            _progress_llm_calls += 1
            return result

    async def embed_func(texts):
        import numpy as np
        from embeddings.service import get_embedding_service
        svc = get_embedding_service()
        sanitized = [t if t and t.strip() else " " for t in texts]
        vecs = await svc.embed_texts(sanitized)
        global _progress_embed_texts
        _progress_embed_texts += len(texts)
        return np.array(vecs, dtype=np.float32)

    if project_id == "default":
        working_dir = _DEFAULT_LIGHTRAG_DIR
        working_dir.mkdir(parents=True, exist_ok=True)
    else:
        working_dir = ensure_project_dirs(project_id) / "lightrag"

    # Pre-flight: 偵測向量索引的維度是否與目前設定一致
    existing_dim = _detect_vdb_dim(working_dir)
    if existing_dim is not None and existing_dim != embed_dim:
        _clear_stale_vdb(working_dir, existing_dim, embed_dim)

    # Provider 簽章遷移：簽章變更時清除舊索引
    from embeddings.service import get_embedding_service
    _esvc = get_embedding_service()
    if _esvc.check_signature_migration(working_dir):
        _clear_stale_vdb(working_dir, embed_dim, embed_dim)
        log.warning("Provider 簽章變更，已清除舊索引 (%s)", working_dir)

    # Configurable performance parameters (overridable via env)
    # Local models (Ollama): lower LLM concurrency since GPU processes sequentially
    _embed_timeout = int(os.environ.get("LIGHTRAG_EMBEDDING_TIMEOUT", "180"))
    # Jina API 限制同時最多 2 concurrent requests，用 1 確保不超限
    _default_embed_async = "1"
    _embed_max_async = int(os.environ.get("LIGHTRAG_EMBED_MAX_ASYNC", _default_embed_async))
    _default_llm_async = "1" if _is_local_llm else "4"
    _llm_max_async = int(os.environ.get("LIGHTRAG_LLM_MAX_ASYNC", _default_llm_async))
    _chunk_token_size = int(os.environ.get("LIGHTRAG_CHUNK_TOKEN_SIZE", "1000"))
    _summary_trigger = int(os.environ.get("LIGHTRAG_FORCE_LLM_SUMMARY_ON_MERGE", "9999"))
    _summary_context = int(os.environ.get("LIGHTRAG_SUMMARY_CONTEXT_SIZE", "8000"))
    _summary_max_tokens = int(os.environ.get("LIGHTRAG_SUMMARY_MAX_TOKENS", "800"))
    _summary_length = int(os.environ.get("LIGHTRAG_SUMMARY_LENGTH_RECOMMENDED", "400"))

    _rag_project_id = project_id
    _rag_instance = LightRAG(
        working_dir=str(working_dir),
        llm_model_func=llm_func,
        llm_model_name=llm_model,
        embedding_func=EmbeddingFunc(
            embedding_dim=embed_dim,
            max_token_size=8192,
            func=embed_func,
        ),
        entity_extract_max_gleaning=0,
        chunk_token_size=_chunk_token_size,
        default_llm_timeout=600,
        default_embedding_timeout=_embed_timeout,
        llm_model_max_async=_llm_max_async,
        embedding_func_max_async=_embed_max_async,
        force_llm_summary_on_merge=_summary_trigger,
        summary_context_size=_summary_context,
        summary_max_tokens=_summary_max_tokens,
        summary_length_recommended=_summary_length,
        cosine_better_than_threshold=float(os.environ.get("LIGHTRAG_COSINE_THRESHOLD", "0.2")),
        addon_params={
            "entity_types": ART_ENTITY_TYPES,
            "example_number": 3,
            "language": "繁體中文",
        },
    )
    return _rag_instance


async def _ensure_initialized(project_id: str | None = None) -> LightRAG:
    """取得 RAG 並確保 storages 已初始化。"""
    async with _get_rag_lock():
        pid = project_id or _rag_project_id or "default"
        rag = get_rag(project_id=pid)
        await rag.initialize_storages()
        return rag


def reset_rag() -> None:
    """Force re-initialization (e.g. after model change)."""
    global _rag_instance, _rag_project_id
    _rag_instance = None
    _rag_project_id = None


async def _clear_failed(rag: LightRAG) -> None:
    """清除所有 FAILED / 卡住的 PROCESSING 文件記錄，讓使用者可以重新匯入。"""
    try:
        from lightrag.base import DocStatus
        to_delete = []
        for status in (DocStatus.FAILED, DocStatus.PROCESSING):
            docs = await rag.doc_status.get_docs_by_status(status)
            if docs:
                to_delete.extend(docs.keys())
        if to_delete:
            await rag.doc_status.delete(to_delete)
    except Exception:
        pass


async def _count_failed(rag: LightRAG) -> int:
    """回傳目前 FAILED 狀態的文件數。"""
    try:
        from lightrag.base import DocStatus
        failed = await rag.doc_status.get_docs_by_status(DocStatus.FAILED)
        return len(failed)
    except Exception:
        return 0


async def _flush_and_check(rag: LightRAG, prev_failed: int, context: str = "") -> str:
    """確保資料寫入磁碟，回傳警告訊息（空字串表示成功）。"""
    # Force persist all in-memory data to disk
    await rag._insert_done()

    # Check if new failures appeared
    try:
        from lightrag.base import DocStatus
        failed = await rag.doc_status.get_docs_by_status(DocStatus.FAILED)
        new_failures = len(failed) - prev_failed
        if new_failures > 0:
            errors = []
            for doc_id, status in list(failed.items())[-new_failures:]:
                err = getattr(status, "error_msg", "") or "unknown"
                if "HTTPStatusError" in err:
                    import re as _re_err
                    m = _re_err.search(r"(\d{3}\s+\w[\w\s]*?) for url", err)
                    err = m.group(1) if m else err[:120]
                errors.append(err)
            return "; ".join(errors[:3])
    except Exception:
        pass
    return ""


async def _llm_clean_web_content(raw_text: str, source: str, project_id: str = "default", on_progress=None) -> str:
    """用 RAG LLM 清理網頁抓取的原始文字，移除導航、廣告等垃圾，只保留文章主體。

    若 LLM 呼叫失敗則原文回傳（graceful fallback）。
    """
    import logging
    log = logging.getLogger("rag.clean")

    import re as _clean_re
    # 移除 script/style 標籤及內容
    stripped = _clean_re.sub(r'<(script|style)[^>]*>[\s\S]*?</\1>', '', raw_text, flags=_clean_re.IGNORECASE)
    # 移除所有 HTML 標籤，只留純文字（大幅減少 token 數）
    stripped = _clean_re.sub(r'<[^>]+>', ' ', stripped)
    # 解碼常見 HTML entities
    stripped = stripped.replace('&nbsp;', ' ').replace('&amp;', '&').replace('&lt;', '<').replace('&gt;', '>')
    # 壓縮連續空白
    stripped = _clean_re.sub(r'[ \t]+', ' ', stripped)
    stripped = _clean_re.sub(r'\n{3,}', '\n\n', stripped).strip()
    # 截取前 8000 字送 LLM
    truncated = stripped[:8000]

    llm_model, llm_key, llm_base = _get_llm_config()
    if llm_model.startswith("codex-cli/"):
        # Codex CLI 不適合做清理，直接回傳原文
        return raw_text

    prompt = (
        "以下是從網頁抓取的原始文字，包含大量導航選單、頁尾、廣告、cookie 提示等雜訊。\n"
        "請只保留文章的正文內容（標題、段落、引言、訪談對話等有意義的文字）。\n"
        "移除所有網站導航、側邊欄、頁首頁尾、版權聲明、社群連結等非正文內容。\n"
        "直接輸出清理後的純文字，不要加任何說明或標記。\n\n"
        f"來源：{source}\n\n"
        f"--- 原始文字 ---\n{truncated}"
    )

    try:
        if on_progress:
            on_progress("正在用 LLM 清理網頁內容…")
        import httpx
        import re as _re
        from rag.rate_guard import get_rate_guard

        async def _call_clean():
            if llm_model.startswith("gemini/"):
                import litellm
                resp = await litellm.acompletion(
                    model=llm_model,
                    messages=[{"role": "user", "content": prompt}],
                    api_key=llm_key,
                )
                return (resp.choices[0].message.content or "")

            async with httpx.AsyncClient(timeout=120.0) as client:
                resp = await client.post(
                    f"{llm_base}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {llm_key}",
                        "HTTP-Referer": "https://github.com/De-insight",
                        "X-Title": "De-insight",
                    },
                    json={
                        "model": llm_model,
                        "messages": [{"role": "user", "content": prompt}],
                    },
                )
                if resp.status_code >= 400:
                    log.warning(f"LLM clean failed {resp.status_code}: {resp.text[:200]}")
                    return None
                return resp.json()["choices"][0]["message"]["content"]

        guard = get_rate_guard()
        result = await guard.call_with_retry(
            "clean/chat/completions", _call_clean, max_retries=2,
        )
        if result and "<think>" in result:
            result = _re.sub(r"<think>[\s\S]*?</think>\s*", "", result)
        if result and len(result.strip()) > 50:
            log.info(f"LLM cleaned: {len(raw_text)} → {len(result)} chars")
            return result.strip()
    except Exception as e:
        log.warning(f"LLM clean error: {e}")

    return raw_text


async def _post_verify_insert(project_id: str) -> str:
    """C1: Post-insert verification — confirm vdb has chunks AND query can retrieve.

    Two checks:
    1. vdb_chunks.json has data[] > 0
    2. A test query returns non-empty result (actual retrievability)
    """
    import json

    if project_id == "default":
        wd = _DEFAULT_LIGHTRAG_DIR
    else:
        wd = project_root(project_id) / "lightrag"
    vdb_path = wd / "vdb_chunks.json"
    if not vdb_path.exists():
        return "post_verify: vdb_chunks.json 不存在"
    try:
        payload = json.loads(vdb_path.read_text(encoding="utf-8"))
        count = len(payload.get("data", []))
        if count == 0:
            return "post_verify: vdb_chunks 為空（匯入可能失敗）"
    except Exception as e:
        return f"post_verify: 無法讀取 vdb_chunks: {e}"

    # Optional Check 2: query-back — disabled by default to avoid false positives.
    env = get_config_service().snapshot(include_process=True)
    do_query_back = (env.get("POST_VERIFY_QUERY_BACK", "0") or "0") == "1"
    if do_query_back:
        try:
            result, _sources = await query_knowledge(
                "測試", mode="naive", context_only=True,
                project_id=project_id, chunk_top_k=1,
            )
            if _is_no_context_result(result) and not result:
                return "post_verify: vdb 有資料但查詢回傳空（索引可能損壞）"
        except Exception as e:
            import logging as _pv_log
            _pv_log.getLogger("rag.post_verify").debug("query-back failed: %s", e)
            # Don't fail on query-back error (might be LLM issue, not index issue)
            pass

    return ""


# ── Progress Tracker ─────────────────────────────────────────────────

class _ProgressTracker:
    """Reads module-level counters (_progress_llm_calls / _progress_embed_texts)
    via a background asyncio task and writes progress to the job DB every 2 s.

    Primary metric: LLM calls (≈ 1 per chunk during entity extraction).
    The counters are incremented by llm_func / embed_func directly —
    no wrapping of rag.llm_model_func needed.
    """

    def __init__(
        self,
        job_id: str,
        db_path: Path,
        estimated_chunks: int,
        tail_expected_seconds: float | None = None,
    ):
        self._job_id = job_id
        self._db_path = db_path
        self._estimated_chunks = max(estimated_chunks, 1)
        self._started_at = datetime.now()
        self._started_at_str = self._started_at.strftime("%Y-%m-%d %H:%M:%S")
        self._bg_task: asyncio.Task | None = None
        self._stage: str = "分chunk"
        self._stage_base_pct: float = 20.0
        self._stage_span_pct: float = 70.0
        # Snapshot counters at creation so we track delta
        self._llm_base = _progress_llm_calls
        self._embed_base = _progress_embed_texts
        self._last_update_ts: float | None = None
        self._last_progress_pct: float | None = None
        self._eta_estimate: float | None = None
        self._tail_started_ts: float | None = None
        # Build-stage tail (graph merge/write) often has low visible LLM-call deltas.
        # Reserve 90~94% for this period so UI keeps moving.
        _env_tail = float(os.environ.get("RAG_PROGRESS_TAIL_SECONDS", "420") or "420")
        _tail = tail_expected_seconds if tail_expected_seconds is not None else _env_tail
        self._tail_expected_seconds = max(60.0, float(_tail))

    async def start(self):
        """Write initial progress (chunks_total) and start background updater."""
        await self._write_to_db(chunks_done=0, progress_pct=self._stage_base_pct)
        self._bg_task = asyncio.create_task(self._bg_loop())

    async def set_stage(
        self,
        stage: str,
        stage_base_pct: float,
        stage_span_pct: float,
        progress_within_stage: float = 0.0,
    ) -> None:
        self._stage = stage
        self._stage_base_pct = stage_base_pct
        self._stage_span_pct = stage_span_pct
        pct = self._stage_base_pct + self._stage_span_pct * max(0.0, min(progress_within_stage, 1.0))
        await self._write_to_db(chunks_done=self._current_done(), progress_pct=pct)

    async def _bg_loop(self):
        """Periodically write progress to DB."""
        try:
            while True:
                await asyncio.sleep(2.0)
                done = self._current_done()
                ratio = done / self._estimated_chunks if self._estimated_chunks > 0 else 0.0

                if self._stage == "建立圖譜" and ratio >= 0.99:
                    now_ts = datetime.now().timestamp()
                    if self._tail_started_ts is None:
                        self._tail_started_ts = now_ts
                    tail_elapsed = max(0.0, now_ts - self._tail_started_ts)
                    tail_ratio = min(tail_elapsed / self._tail_expected_seconds, 1.0)
                    pct = self._stage_base_pct + self._stage_span_pct * (0.92 + (0.08 * tail_ratio))
                    eta_tail = int(max(0.0, self._tail_expected_seconds - tail_elapsed))
                    await self._write_to_db(
                        chunks_done=self._estimated_chunks,
                        progress_pct=pct,
                        stage_override="圖譜合併",
                        eta_override=eta_tail,
                    )
                else:
                    pct = self._stage_base_pct + self._stage_span_pct * max(0.0, min(ratio, 1.0))
                    await self._write_to_db(chunks_done=done, progress_pct=pct)
        except asyncio.CancelledError:
            pass

    def _current_done(self) -> int:
        """Derive chunks_done from module-level LLM call counter."""
        llm_delta = _progress_llm_calls - self._llm_base
        # Each chunk triggers ≈1 LLM call for entity extraction
        return max(0, min(llm_delta, self._estimated_chunks))

    async def _write_to_db(
        self,
        chunks_done: int,
        progress_pct: float,
        stage_override: str | None = None,
        eta_override: int | None = None,
    ):
        try:
            from rag.job_repository import JobRepository
            repo = JobRepository(self._db_path)
            now_ts = datetime.now().timestamp()
            eta_seconds = eta_override if eta_override is not None else self._compute_eta_seconds(progress_pct, now_ts)
            await repo.update_progress(
                self._job_id,
                chunks_done=chunks_done,
                chunks_total=self._estimated_chunks,
                started_at=self._started_at_str,
                progress_pct=progress_pct,
                progress_stage=(stage_override or self._stage),
                eta_seconds=eta_seconds,
            )
        except Exception as e:
            log.warning("Progress write failed for job %s: %s", self._job_id, e)

    def _compute_eta_seconds(self, progress_pct: float, now_ts: float) -> int | None:
        """ETA estimator that avoids continuous extension during temporary stalls.

        Rule:
        - progress 有前進：重估 ETA（平滑）。
        - progress 停滯：ETA 只倒數，不回升。
        """
        progress = max(0.0, min(progress_pct, 100.0))
        if progress >= 99.9:
            self._last_update_ts = now_ts
            self._last_progress_pct = progress
            self._eta_estimate = 0.0
            return 0

        if self._last_update_ts is None or self._last_progress_pct is None:
            self._last_update_ts = now_ts
            self._last_progress_pct = progress
            return None

        delta_t = max(0.0, now_ts - self._last_update_ts)
        delta_p = progress - self._last_progress_pct

        if delta_p > 0.05 and delta_t > 0.0:
            raw_eta = (100.0 - progress) * (delta_t / delta_p)
            if self._eta_estimate is None:
                self._eta_estimate = raw_eta
            else:
                # EWMA smoothing to reduce jitter
                self._eta_estimate = (0.7 * self._eta_estimate) + (0.3 * raw_eta)
        elif self._eta_estimate is not None and delta_t > 0.0:
            # No visible progress: countdown only, do not increase ETA.
            self._eta_estimate = max(0.0, self._eta_estimate - delta_t)

        self._last_update_ts = now_ts
        self._last_progress_pct = progress

        if self._eta_estimate is None:
            return None
        return int(max(0.0, min(self._eta_estimate, 7200.0)))

    async def finish(self):
        """Stop background task and write 100 %."""
        if self._bg_task:
            self._bg_task.cancel()
            try:
                await self._bg_task
            except asyncio.CancelledError:
                pass
        await self._write_to_db(
            chunks_done=self._estimated_chunks,
            progress_pct=self._stage_base_pct + self._stage_span_pct,
        )


async def _set_job_stage(
    job_id: str | None,
    *,
    stage: str,
    progress_pct: float,
    chunks_done: int = 0,
    chunks_total: int = 0,
    started_at: str | None = None,
) -> None:
    if not job_id:
        return
    try:
        from rag.job_repository import JobRepository
        repo = JobRepository(_get_jobs_db_path())
        eta_seconds = None
        if started_at and chunks_total > 0 and chunks_done > 0 and chunks_done < chunks_total:
            try:
                start = datetime.strptime(started_at, "%Y-%m-%d %H:%M:%S")
                elapsed = max(0.0, (datetime.now() - start).total_seconds())
                ratio = chunks_done / chunks_total
                if ratio > 0 and ratio < 1 and elapsed > 0:
                    total = elapsed / ratio
                    eta_seconds = max(0, int(total - elapsed))
            except Exception:
                eta_seconds = None
        await repo.update_progress(
            job_id,
            chunks_done=chunks_done,
            chunks_total=max(chunks_total, 0),
            started_at=started_at,
            progress_pct=progress_pct,
            progress_stage=stage,
            eta_seconds=eta_seconds,
        )
    except Exception as e:
        log.warning("set_job_stage failed for %s: %s", job_id, e)


def _estimate_chunks(text: str) -> int:
    """Estimate how many chunks LightRAG will create from the text."""
    chunk_token_size = int(os.environ.get("LIGHTRAG_CHUNK_TOKEN_SIZE", "2400"))
    # Rough estimate: 1 token ≈ 2 chars for Chinese, 4 chars for English
    # Use 3 as average
    chars_per_chunk = chunk_token_size * 3
    base = max(1, len(text) // chars_per_chunk)
    # Empirical calibration: LightRAG often creates significantly more chunks than
    # this rough token/char estimate, especially for CJK-heavy docs.
    mult = float(os.environ.get("RAG_PROGRESS_CHUNK_MULTIPLIER", "6.5") or "6.5")
    return max(1, int(base * max(mult, 1.0)))


def _estimate_chunks_from_char_count(char_count: int) -> int:
    """Estimate chunk count without building a huge concatenated string."""
    chunk_token_size = int(os.environ.get("LIGHTRAG_CHUNK_TOKEN_SIZE", "2400"))
    chars_per_chunk = chunk_token_size * 3
    base = max(1, int(char_count) // chars_per_chunk)
    mult = float(os.environ.get("RAG_PROGRESS_CHUNK_MULTIPLIER", "6.5") or "6.5")
    return max(1, int(base * max(mult, 1.0)))


def _build_pdf_batches(
    page_chunks: list[tuple[int, str]],
    *,
    batch_size: int,
) -> list[list[tuple[int, str]]]:
    """Always split PDFs through the batch pipeline.

    A single-shot path defeats checkpointing, retry locality, and ETA quality.
    """
    size = max(1, int(batch_size))
    return [page_chunks[idx: idx + size] for idx in range(0, len(page_chunks), size)]


async def _install_progress_tracker(text: str, job_id: str | None) -> _ProgressTracker | None:
    """Create and start a progress tracker for the given job."""
    if not job_id:
        return None
    estimated = _estimate_chunks(text)
    tracker = _ProgressTracker(job_id, _get_jobs_db_path(), estimated)
    await tracker.start()
    return tracker


async def _install_progress_tracker_with_estimated(
    job_id: str | None,
    *,
    estimated_chunks: int,
    tail_expected_seconds: float | None = None,
) -> _ProgressTracker | None:
    if not job_id:
        return None
    tracker = _ProgressTracker(
        job_id,
        _get_jobs_db_path(),
        max(1, int(estimated_chunks)),
        tail_expected_seconds=tail_expected_seconds,
    )
    await tracker.start()
    return tracker


async def _run_ainsert_with_keepalive(
    rag: LightRAG,
    payload: str,
    *,
    timeout_seconds: float,
    jobs_repo=None,
    job_id: str | None = None,
    heartbeat_interval: float = 10.0,
) -> None:
    keepalive_task = None

    async def _keepalive_loop() -> None:
        if not jobs_repo or not job_id:
            return
        try:
            while True:
                await asyncio.sleep(heartbeat_interval)
                await jobs_repo.touch_heartbeat(job_id)
        except asyncio.CancelledError:
            return

    try:
        if jobs_repo and job_id:
            keepalive_task = asyncio.create_task(_keepalive_loop())
        await asyncio.wait_for(rag.ainsert(payload), timeout=float(timeout_seconds))
    finally:
        if keepalive_task:
            keepalive_task.cancel()
            try:
                await keepalive_task
            except asyncio.CancelledError:
                pass


def _get_jobs_db_path() -> Path:
    from paths import DATA_ROOT
    return DATA_ROOT / "ingest_jobs.db"


from datetime import datetime

async def insert_text(text: str, source: str = "", project_id: str = "default", job_id: str | None = None) -> str:
    """插入文字到知識庫。回傳警告訊息（空字串表示完全成功）。"""
    rag = await _ensure_initialized(project_id=project_id)
    await _clear_failed(rag)
    doc = text
    if source:
        doc = f"[來源: {source}]\n\n{text}"
    started_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    await _set_job_stage(
        job_id,
        stage="分chunk",
        progress_pct=10.0,
        chunks_done=0,
        chunks_total=1,
        started_at=started_at,
    )

    tracker = await _install_progress_tracker(doc, job_id)
    if tracker:
        await tracker.set_stage("建立圖譜", stage_base_pct=20.0, stage_span_pct=70.0)
    try:
        await rag.ainsert(doc)
    finally:
        if tracker:
            await tracker.finish()
    await _set_job_stage(
        job_id,
        stage="寫入圖譜",
        progress_pct=95.0,
        chunks_done=1,
        chunks_total=1,
        started_at=started_at,
    )
    warning = await _flush_and_check(rag, 0, context=source or "text")
    # C1: Post-verify
    pv_warning = await _post_verify_insert(project_id)
    if pv_warning:
        warning = f"{warning}; {pv_warning}" if warning else pv_warning
    return warning


async def insert_pdf(path: str, project_id: str = "default", title: str = "", on_progress=None, job_id: str | None = None) -> dict:
    """匯入 PDF，回傳 {"title": str, "page_count": int, "file_size": int, "saved_path": str}。"""
    import re as _re
    try:
        from pypdf import PdfReader
    except ImportError:
        raise RuntimeError("需要安裝 pypdf: pip install pypdf")

    file_size = Path(path).stat().st_size if Path(path).exists() else 0
    reader = PdfReader(path)
    page_count = len(reader.pages)
    started_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    await _set_job_stage(
        job_id,
        stage="分chunk",
        progress_pct=3.0,
        chunks_done=0,
        chunks_total=max(page_count, 1),
        started_at=started_at,
    )

    # Determine display name: explicit title > PDF metadata > filename
    display_name = title.strip() if title else ""
    if not display_name:
        pdf_meta = reader.metadata
        if pdf_meta and pdf_meta.title and len(pdf_meta.title.strip()) >= 3:
            display_name = pdf_meta.title.strip()
    if not display_name:
        display_name = Path(path).stem

    # Sanitize for filesystem use
    safe_name = _re.sub(r'[/<>:"|?*\\]+', '_', display_name).strip('_. ')
    if not safe_name:
        safe_name = Path(path).stem

    page_chunks: list[tuple[int, str]] = []
    extract_fail_pages: list[int] = []
    for i, page in enumerate(reader.pages):
        try:
            text = page.extract_text()
        except Exception:
            text = ""
            extract_fail_pages.append(i + 1)
        if text and text.strip():
            page_chunks.append((i + 1, f"[{display_name} p.{i+1}]\n{text.strip()}"))
        if page_count > 0:
            chunk_pct = 3.0 + (17.0 * ((i + 1) / page_count))
            await _set_job_stage(
                job_id,
                stage="分chunk",
                progress_pct=chunk_pct,
                chunks_done=i + 1,
                chunks_total=page_count,
                started_at=started_at,
            )

    if on_progress:
        on_progress(f"正在建構知識圖譜（{page_count} 頁）…")
    rag = await _ensure_initialized(project_id=project_id)
    await _clear_failed(rag)
    if not page_chunks:
        raise RuntimeError("PDF 無可匯入文字（可能為掃描檔或抽取失敗）")

    # 大 PDF 以分批寫入 + checkpoint（M2）。
    cfg = get_config_service().snapshot(include_process=True)
    # Keep PDF batches small enough that remote extraction/merge doesn't stall at ~90%.
    batch_size = max(4, int(cfg.get("RAG_PDF_BATCH_PAGES", "6") or "6"))
    total_chunks = max(len(page_chunks), 1)
    processed = 0
    insert_fail_pages: list[int] = []
    duplicate_risk = False
    jobs_repo = None
    if job_id:
        from rag.job_repository import JobRepository
        jobs_repo = JobRepository(_get_jobs_db_path())

    def _graph_deltas() -> tuple[int, int]:
        """Return (entity_count, relation_count) from current lightrag vdb files."""
        wd = ensure_project_dirs(project_id) / "lightrag"
        ent_n = 0
        rel_n = 0
        try:
            ep = wd / "vdb_entities.json"
            if ep.exists():
                data = _json.loads(ep.read_text(encoding="utf-8"))
                ent_n = len(data.get("data", []) or [])
        except Exception:
            ent_n = 0
        try:
            rp = wd / "vdb_relationships.json"
            if rp.exists():
                data = _json.loads(rp.read_text(encoding="utf-8"))
                rel_n = len(data.get("data", []) or [])
        except Exception:
            rel_n = 0
        return ent_n, rel_n

    def _is_rate_limit_error(exc: Exception) -> bool:
        s = str(exc).lower()
        return ("429" in s) or ("rate limit" in s) or ("速率限制" in s)

    def _backoff_seconds(retry_count: int) -> int:
        seq = [30, 60, 120, 240, 300]
        base = seq[min(max(0, int(retry_count) - 1), len(seq) - 1)]
        jitter = base * random.uniform(0, 0.2)
        return int(base + jitter)

    async def _handle_rate_limit_backoff(exc: Exception, phase: str = "extracting") -> None:
        nonlocal jobs_repo
        if not (job_id and jobs_repo):
            raise exc
        from rag.job_executor import JobExecutor

        cur = await jobs_repo.get_job(job_id)
        if not cur:
            raise exc
        rl_count = int(cur.get("rate_limit_retry_count") or 0) + 1
        wall = int(cur.get("backoff_wall_time_seconds") or 0)
        wait_seconds = _backoff_seconds(rl_count)
        new_wall = wall + wait_seconds
        if JobExecutor.is_rate_limit_exhausted(rl_count, new_wall):
            await jobs_repo.update_status(
                job_id,
                "failed",
                last_error=str(exc),
                phase="failed",
                error_code="RATE_LIMIT_EXHAUSTED",
                error_detail=f"retries={rl_count}, wall_time={new_wall}s",
            )
            raise RuntimeError(f"RATE_LIMIT_EXHAUSTED retries={rl_count} wall={new_wall}s")
        await jobs_repo.set_waiting_backoff(
            job_id,
            phase=phase,
            wait_seconds=wait_seconds,
            rate_limit_retry_count=rl_count,
            backoff_wall_time_seconds=new_wall,
        )
        remained = wait_seconds
        while remained > 0:
            step = min(5, remained)
            await asyncio.sleep(step)
            remained -= step
            await jobs_repo.touch_heartbeat(job_id)
        await jobs_repo.clear_waiting_backoff(job_id, phase=phase)

    batches = _build_pdf_batches(page_chunks, batch_size=batch_size)

    if jobs_repo and job_id:
        batch_rows = []
        for i, b in enumerate(batches):
            p_start = b[0][0] if b else 0
            p_end = b[-1][0] if b else 0
            batch_rows.append(
                {
                    "batch_no": i,
                    "page_start": p_start,
                    "page_end": p_end,
                    "actual_chunks": len(b),
                }
            )
        existing_rows = await jobs_repo.list_batches(job_id)
        existing_signature = [
            (
                int(r.get("batch_no") or 0),
                int(r.get("page_start") or 0),
                int(r.get("page_end") or 0),
                int(r.get("actual_chunks") or 0),
            )
            for r in existing_rows
        ]
        target_signature = [
            (
                int(r.get("batch_no") or 0),
                int(r.get("page_start") or 0),
                int(r.get("page_end") or 0),
                int(r.get("actual_chunks") or 0),
            )
            for r in batch_rows
        ]
        # Resume from checkpoint when batch layout matches; otherwise re-seed once.
        if (not existing_rows) or (existing_signature != target_signature):
            await jobs_repo.create_or_replace_batches(job_id, batch_rows)
        await jobs_repo.update_phase(job_id, "extracting", status="running")

    batch_states = await jobs_repo.list_batches(job_id) if (jobs_repo and job_id) else []
    batch_by_no = {int(b.get("batch_no") or 0): b for b in batch_states}
    total_batches = max(1, len(batches))
    for idx, batch in enumerate(batches):
        if not batch:
            continue
        b_row = batch_by_no.get(idx)
        if b_row and (b_row.get("status") or "") == "done":
            processed += len(batch)
            if on_progress:
                on_progress(f"建立圖譜 {processed}/{total_chunks} 頁")
            continue
        payload = "\n\n---\n\n".join(t for _, t in batch)
        if not payload.strip():
            insert_fail_pages.extend([p for p, _ in batch])
            processed += len(batch)
            continue
        if b_row and jobs_repo:
            await jobs_repo.mark_batch_running(int(b_row["id"]))
            await jobs_repo.update_phase(job_id, "extracting", status="running")
        batch_pages = max(1, int(batch[-1][0] - batch[0][0] + 1))
        actual_chunks = max(1, len(batch))
        estimated_batch_chunks = max(1, _estimate_chunks_from_char_count(len(payload)))
        batch_base_pct = 20.0 + (70.0 * (idx / total_batches))
        batch_span_pct = 70.0 / total_batches
        timeout_per_page = float(os.environ.get("RAG_BATCH_TIMEOUT_PER_PAGE", "30") or "30")
        timeout_per_chunk = float(os.environ.get("RAG_BATCH_TIMEOUT_PER_CHUNK", "10") or "10")
        timeout_floor = max(180, int(os.environ.get("RAG_BATCH_TIMEOUT_MIN", "600") or "600"))
        timeout_cap = max(timeout_floor, int(os.environ.get("RAG_BATCH_TIMEOUT_MAX", "7200") or "7200"))
        timeout_seconds = int(timeout_per_page * batch_pages + timeout_per_chunk * actual_chunks)
        timeout_seconds = max(timeout_floor, min(timeout_cap, timeout_seconds))
        tail_per_page = float(os.environ.get("RAG_PROGRESS_TAIL_PER_PAGE_SECONDS", "2.0") or "2.0")
        tail_cap = float(os.environ.get("RAG_PROGRESS_TAIL_MAX_SECONDS", "600") or "600")
        tail_floor = float(os.environ.get("RAG_PROGRESS_TAIL_MIN_SECONDS", "90") or "90")
        batch_tail_seconds = min(
            tail_cap,
            max(tail_floor, float(batch_pages) * tail_per_page),
        )
        tracker = await _install_progress_tracker_with_estimated(
            job_id,
            estimated_chunks=estimated_batch_chunks,
            tail_expected_seconds=batch_tail_seconds,
        )
        if tracker:
            await tracker.set_stage(
                "建立圖譜",
                stage_base_pct=batch_base_pct,
                stage_span_pct=batch_span_pct,
            )
        before_ent, before_rel = _graph_deltas()
        try:
            while True:
                try:
                    await _run_ainsert_with_keepalive(
                        rag,
                        payload,
                        timeout_seconds=float(timeout_seconds),
                        jobs_repo=jobs_repo,
                        job_id=job_id,
                    )
                    break
                except asyncio.TimeoutError:
                    if b_row and jobs_repo:
                        await jobs_repo.mark_batch_failed(int(b_row["id"]), "PHASE_TIMEOUT")
                    raise RuntimeError(f"PHASE_TIMEOUT batch={idx} timeout={timeout_seconds}s")
                except Exception as e:
                    if _is_rate_limit_error(e):
                        await _handle_rate_limit_backoff(e, phase="extracting")
                        continue
                    if b_row and jobs_repo:
                        await jobs_repo.mark_batch_failed(int(b_row["id"]), "EXTRACT_ERROR")
                    raise
        finally:
            if tracker:
                await tracker.finish()
        after_ent, after_rel = _graph_deltas()
        if b_row and jobs_repo:
            await jobs_repo.update_phase(job_id, "merging", status="running")
            ent_delta = max(0, int(after_ent - before_ent))
            rel_delta = max(0, int(after_rel - before_rel))
            risky = await jobs_repo.mark_batch_done(int(b_row["id"]), ent_delta, rel_delta)
            duplicate_risk = duplicate_risk or bool(risky)
        processed += len(batch)
        if on_progress:
            on_progress(f"建立圖譜 {processed}/{total_chunks} 頁")

    await _set_job_stage(
        job_id,
        stage="寫入圖譜",
        progress_pct=95.0,
        chunks_done=total_chunks,
        chunks_total=total_chunks,
        started_at=started_at,
    )
    if jobs_repo and job_id:
        await jobs_repo.update_phase(job_id, "flushing", status="running")
    warning = await _flush_and_check(rag, 0, context=display_name)
    warn_parts: list[str] = []
    if warning:
        warn_parts.append(warning)
    if extract_fail_pages:
        warn_parts.append(
            f"extract_failed_pages={len(extract_fail_pages)}({','.join(map(str, extract_fail_pages[:12]))}{'…' if len(extract_fail_pages) > 12 else ''})"
        )
    if insert_fail_pages:
        uniq_fail = sorted(set(insert_fail_pages))
        warn_parts.append(
            f"insert_failed_pages={len(uniq_fail)}({','.join(map(str, uniq_fail[:12]))}{'…' if len(uniq_fail) > 12 else ''})"
        )
    if duplicate_risk:
        warn_parts.append("DUPLICATE_RISK: batch merge delta exceeded baseline x1.5")

    # C1: Post-verify
    pv_warning = await _post_verify_insert(project_id)
    if pv_warning:
        warn_parts.append(pv_warning)
    warning = "; ".join(warn_parts)

    # 保留原始檔案到專案目錄
    doc_dir = ensure_project_dirs(project_id) / "documents"
    import shutil
    saved = doc_dir / f"{safe_name}.pdf"
    if Path(path).resolve() != saved.resolve():
        shutil.copy2(path, saved)

    result = {"title": display_name, "page_count": page_count, "file_size": file_size, "saved_path": str(saved)}
    if warning:
        result["warning"] = warning
    return result


async def _fetch_with_jina_reader(url: str, timeout: float = 30.0, reader_base: str = "https://r.jina.ai/") -> tuple[str, dict]:
    """用 Jina Reader 抓取網頁正文。

    回傳 (text, meta)。meta 至少含 status_code, latency_ms, reader_url。
    失敗時拋出 RuntimeError，供上層記錄。
    """
    import httpx
    import time

    reader_url = f"{reader_base.rstrip('/')}/{url}"
    meta = {"reader_url": reader_url, "status_code": 0, "latency_ms": 0}

    t0 = time.monotonic()
    try:
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
            resp = await client.get(reader_url, headers={
                "Accept": "text/plain",
                "User-Agent": "De-insight/1.0",
            })
            meta["status_code"] = resp.status_code
            meta["latency_ms"] = int((time.monotonic() - t0) * 1000)

            if resp.status_code >= 400:
                raise RuntimeError(
                    f"Jina Reader HTTP {resp.status_code} for {url}"
                )

            text = resp.text.strip()
            if len(text) < 50:
                raise RuntimeError(
                    f"Jina Reader 回傳內容太短（{len(text)} 字）for {url}"
                )
            return text, meta

    except httpx.HTTPError as e:
        meta["latency_ms"] = int((time.monotonic() - t0) * 1000)
        raise RuntimeError(f"Jina Reader 連線失敗: {e}") from e


async def insert_url(url: str, project_id: str = "default", title: str = "", on_progress=None, job_id: str | None = None) -> dict:
    """Fetch a URL and insert its content. 回傳 {"title": str, "page_count": 0, "file_size": int, "fetch_method": str}。"""
    import httpx
    import re
    import tempfile
    import logging

    log = logging.getLogger("rag.insert_url")

    # 讀取 web fetch 設定
    env = get_config_service().snapshot(include_process=True)
    legacy_env = load_env() or {}

    def _cfg(name: str, default: str) -> str:
        # Backward-compat: explicit .env/load_env values override runtime snapshot.
        v = legacy_env.get(name)
        if v is None or str(v).strip() == "":
            v = env.get(name, default)
        return str(v)

    fetch_provider = _cfg("WEB_FETCH_PROVIDER", "auto").lower()  # auto|reader|legacy
    fetch_timeout = float(_cfg("WEB_FETCH_TIMEOUT_SECS", "30"))
    reader_base = _cfg("WEB_FETCH_READER_BASE", "https://r.jina.ai/")

    # 下載大小上限（預設 5 MB），避免超大頁面撐爆記憶體 / GPU
    _max_download = int(_cfg("WEB_FETCH_MAX_BYTES", str(5 * 1024 * 1024)))
    started_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    await _set_job_stage(
        job_id,
        stage="分chunk",
        progress_pct=5.0,
        chunks_done=0,
        chunks_total=1,
        started_at=started_at,
    )

    # 優化：避免 HTML 路徑雙重抓取。reader/auto 先走 Jina Reader，必要時才回退下載原頁。
    if url.lower().endswith(".pdf"):
        from urllib.parse import urlparse, unquote
        url_path = urlparse(url).path
        url_filename = unquote(Path(url_path).stem) if url_path else ""
        url_filename = re.sub(r'[^\w\-\.\(\)\[\]\u4e00-\u9fff]+', '_', url_filename).strip('_')
        if not url_filename or len(url_filename) < 3:
            url_filename = "downloaded_pdf"

        tmp_dir = Path(tempfile.gettempdir())
        tmp_path = str(tmp_dir / f"{url_filename}.pdf")
        if on_progress:
            on_progress("正在下載 PDF…")
        async with httpx.AsyncClient(timeout=60.0, follow_redirects=True) as client:
            pdf_resp = await client.get(url, headers={
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
            })
            pdf_resp.raise_for_status()
            if len(pdf_resp.content) > _max_download:
                size_mb = len(pdf_resp.content) / (1024 * 1024)
                limit_mb = _max_download / (1024 * 1024)
                raise RuntimeError(
                    f"頁面太大（{size_mb:.1f} MB），超過上限 {limit_mb:.0f} MB。"
                    f" 可透過 WEB_FETCH_MAX_BYTES 調整。"
                )
            Path(tmp_path).write_bytes(pdf_resp.content)
        try:
            result = await insert_pdf(tmp_path, project_id=project_id, title=title, on_progress=on_progress, job_id=job_id)
            result["fetch_method"] = "pdf"
            return result
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    # ── HTML 分支：Jina Reader 優先，legacy 回退 ──
    html = ""
    file_size = 0

    # 解析 title
    if title.strip():
        source = title.strip()
    else:
        title_match = re.search(r"<title[^>]*>(.*?)</title>", html, re.IGNORECASE | re.DOTALL)
        source = title_match.group(1).strip() if title_match else url

    text = None
    fetch_method = None
    fallback_reason = None

    # ── 嘗試 Jina Reader ──
    if fetch_provider in ("auto", "reader"):
        try:
            if on_progress:
                on_progress("正在透過 Jina Reader 抓取正文…")
            reader_text, reader_meta = await _fetch_with_jina_reader(
                url, timeout=fetch_timeout, reader_base=reader_base,
            )
            log.info(
                "Jina Reader 成功: url=%s status=%s latency=%dms len=%d",
                url, reader_meta.get("status_code"), reader_meta.get("latency_ms", 0), len(reader_text),
            )
            text = reader_text
            fetch_method = "jina_reader"
            file_size = len(reader_text.encode("utf-8"))
        except RuntimeError as e:
            fallback_reason = str(e)
            log.warning("Jina Reader 失敗: %s", fallback_reason)
            if fetch_provider == "reader":
                # reader-only 模式：不回退，直接報錯
                raise RuntimeError(f"Jina Reader 失敗且 WEB_FETCH_PROVIDER=reader，不回退: {fallback_reason}")

    # ── Legacy 回退：reader 未命中時才下載原頁 ──
    if text is None and fetch_provider in ("auto", "legacy"):
        if on_progress:
            on_progress("正在下載網頁…")
        async with httpx.AsyncClient(timeout=60.0, follow_redirects=True) as client:
            resp = await client.get(url, headers={
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
            })
            resp.raise_for_status()

        if len(resp.content) > _max_download:
            size_mb = len(resp.content) / (1024 * 1024)
            limit_mb = _max_download / (1024 * 1024)
            raise RuntimeError(
                f"頁面太大（{size_mb:.1f} MB），超過上限 {limit_mb:.0f} MB。"
                f" 可透過 WEB_FETCH_MAX_BYTES 調整。"
            )
        content_type = resp.headers.get("content-type", "")
        file_size = len(resp.content)

        # 若回退抓到 PDF，直接走 PDF 分支。
        if "pdf" in content_type:
            from urllib.parse import urlparse, unquote
            url_path = urlparse(url).path
            url_filename = unquote(Path(url_path).stem) if url_path else ""
            url_filename = re.sub(r'[^\w\-\.\(\)\[\]\u4e00-\u9fff]+', '_', url_filename).strip('_')
            if not url_filename or len(url_filename) < 3:
                url_filename = "downloaded_pdf"
            tmp_dir = Path(tempfile.gettempdir())
            tmp_path = str(tmp_dir / f"{url_filename}.pdf")
            Path(tmp_path).write_bytes(resp.content)
            try:
                result = await insert_pdf(tmp_path, project_id=project_id, title=title, on_progress=on_progress, job_id=job_id)
                result["fetch_method"] = "pdf"
                return result
            finally:
                Path(tmp_path).unlink(missing_ok=True)

        html = resp.text
        if on_progress:
            on_progress("正在用 LLM 清理網頁內容…")
        if fallback_reason:
            log.info("回退 legacy 流程: %s", fallback_reason)
        text = await _llm_clean_web_content(html, source, project_id=project_id, on_progress=on_progress)
        fetch_method = "legacy_html_clean"

    if text is None or len(text.strip()) < 50:
        raise RuntimeError("頁面內容太少，無法匯入")

    if on_progress:
        on_progress("正在建構知識圖譜…")
    await _set_job_stage(
        job_id,
        stage="建立圖譜",
        progress_pct=20.0,
        chunks_done=0,
        chunks_total=1,
        started_at=started_at,
    )
    warning = await insert_text(text, source=source, project_id=project_id, job_id=job_id)
    await _set_job_stage(
        job_id,
        stage="寫入圖譜",
        progress_pct=95.0,
        chunks_done=1,
        chunks_total=1,
        started_at=started_at,
    )
    result = {"title": source, "page_count": 0, "file_size": file_size, "fetch_method": fetch_method}
    if warning:
        result["warning"] = warning
    if fallback_reason:
        result["fallback_reason"] = fallback_reason
    return result


async def resolve_doi(doi: str) -> str:
    """輸入 DOI，回傳 open access PDF URL。找不到時回傳空字串。"""
    import httpx, re
    doi = doi.strip()
    if doi.startswith("http"):
        m = re.search(r'10\.\d{4,}/\S+', doi)
        doi = m.group(0) if m else doi
    url = f"https://api.unpaywall.org/v2/{doi}?email=app@de-insight.local"
    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.get(url)
        if resp.status_code != 200:
            return ""
        data = resp.json()
        best = data.get("best_oa_location") or {}
        return best.get("url_for_pdf", "") or best.get("url", "") or ""


async def insert_doi(doi: str, project_id: str = "default", title: str = "", job_id: str | None = None) -> dict:
    """解析 DOI → 取得 PDF URL → insert_url()。找不到 open access 時 raise。"""
    pdf_url = await resolve_doi(doi)
    if not pdf_url:
        raise RuntimeError(f"找不到 open access 版本：{doi}")
    return await insert_url(pdf_url, project_id=project_id, title=title, job_id=job_id)


async def query_knowledge_merged(
    question: str,
    mode: str = "naive",
    context_only: bool = True,
    project_id: str | None = None,
    chunk_top_k: int = 5,
) -> tuple[str, list[dict], dict]:
    """合併查詢全局 + 專案知識庫。

    同時搜尋全局文獻庫和專案圖譜，合併結果。
    回傳 (merged_text, merged_sources, merge_info)。
    merge_info = {"global_chars": int, "project_chars": int, "global_sources": int, "project_sources": int}
    """
    pid = project_id or "default"
    tasks = {}

    # 全局圖譜查詢（如果有資料且當前不在全局專案）
    if pid != GLOBAL_PROJECT_ID and has_knowledge(project_id=GLOBAL_PROJECT_ID):
        tasks["global"] = query_knowledge(
            question, mode=mode, context_only=context_only,
            project_id=GLOBAL_PROJECT_ID, chunk_top_k=chunk_top_k,
        )

    # 專案圖譜查詢
    if has_knowledge(project_id=pid):
        tasks["project"] = query_knowledge(
            question, mode=mode, context_only=context_only,
            project_id=pid, chunk_top_k=chunk_top_k,
        )

    if not tasks:
        return "", [], {"global_chars": 0, "project_chars": 0, "global_sources": 0, "project_sources": 0}

    # 並行查詢
    results = {}
    for key, coro in tasks.items():
        try:
            results[key] = await coro
        except Exception as e:
            log.warning("Merged query %s failed: %s", key, e)
            results[key] = ("", [])

    global_text, global_sources = results.get("global", ("", []))
    project_text, project_sources = results.get("project", ("", []))

    # 標記來源層級
    for s in global_sources:
        s["tier"] = "global"
    for s in project_sources:
        s["tier"] = "project"

    # 合併：專案結果優先（放前面），全局結果補充
    parts = []
    if project_text:
        parts.append(project_text)
    if global_text:
        if parts:
            parts.append("\n\n─── 全局文獻 ───\n\n")
        parts.append(global_text)

    merged_text = "".join(parts)
    merged_sources = project_sources + global_sources

    merge_info = {
        "global_chars": len(global_text),
        "project_chars": len(project_text),
        "global_sources": len(global_sources),
        "project_sources": len(project_sources),
    }

    return merged_text, merged_sources, merge_info


async def query_knowledge(question: str, mode: str = "naive", context_only: bool = True, project_id: str | None = None, chunk_top_k: int = 5) -> tuple[str, list[dict]]:
    """查詢知識庫。

    預設 naive + context_only：純向量搜尋，不呼叫 LLM，<1秒回應。
    若需要完整圖譜推理，可傳 mode="hybrid", context_only=False。

    回傳 (result_text, sources)。
    sources 格式：[{"title": str, "snippet": str, "file": str}]
    若無來源資訊則回傳空 list。
    """
    rag = await _ensure_initialized(project_id=project_id)
    pid = project_id or "default"
    q_norm = " ".join((question or "").strip().split()).lower()
    env = get_config_service().snapshot(include_process=True)
    cache_ttl = int(env.get("RAG_QUERY_CACHE_TTL_SEC", "120") or "120")
    use_fast_first = (env.get("RAG_FAST_FIRST", "1") or "1") != "0"

    cache_key = (pid, mode, context_only, int(chunk_top_k), q_norm)
    now = time.time()
    if cache_ttl > 0:
        cached = _QUERY_CACHE.get(cache_key)
        if cached and cached[0] > now:
            return cached[1], list(cached[2])
        if cached and cached[0] <= now:
            _QUERY_CACHE.pop(cache_key, None)
    if context_only:
        # Fast-first: graph modes先跑naive，命中就直接回傳，避免每次進入慢速圖譜關鍵詞/圖遍歷。
        if use_fast_first and mode in ("local", "global", "hybrid"):
            naive_param = QueryParam(
                mode="naive",
                only_need_context=True,
                chunk_top_k=chunk_top_k,
                enable_rerank=False,
            )
            naive_result = await rag.aquery(question, param=naive_param)
            if not _is_no_context_result(naive_result):
                naive_sources = _extract_sources(naive_result)
                if cache_ttl > 0:
                    if len(_QUERY_CACHE) >= _QUERY_CACHE_MAX:
                        _QUERY_CACHE.pop(next(iter(_QUERY_CACHE)))
                    _QUERY_CACHE[cache_key] = (now + cache_ttl, naive_result, naive_sources)
                return naive_result, naive_sources

        param = QueryParam(
            mode=mode,
            only_need_context=True,
            chunk_top_k=chunk_top_k,
            enable_rerank=False,
        )
        result = await rag.aquery(question, param=param)
        if _is_no_context_result(result):
            return "", []
        sources = _extract_sources(result)
        if cache_ttl > 0:
            if len(_QUERY_CACHE) >= _QUERY_CACHE_MAX:
                _QUERY_CACHE.pop(next(iter(_QUERY_CACHE)))
            _QUERY_CACHE[cache_key] = (now + cache_ttl, result, sources)
        return result, sources
    param = QueryParam(
        mode=mode,
        chunk_top_k=chunk_top_k,
        user_prompt="請用繁體中文回答。\n\n" + question,
        enable_rerank=False,
    )
    result = await rag.aquery(question, param=param)
    if _is_no_context_result(result):
        return "", []
    sources = _extract_sources(result)
    if cache_ttl > 0:
        if len(_QUERY_CACHE) >= _QUERY_CACHE_MAX:
            _QUERY_CACHE.pop(next(iter(_QUERY_CACHE)))
        _QUERY_CACHE[cache_key] = (now + cache_ttl, result, sources)
    return result, sources


def _clean_rag_chunk(text: str) -> str:
    """清理 LightRAG 原始 chunk，移除 JSON 包裝、轉義字元、導航垃圾。"""
    import re
    # 先從完整文字中提取 JSON content（在移除 header/footer 之前）
    json_contents = re.findall(r'"content"\s*:\s*"((?:[^"\\]|\\.)*)"', text)
    if json_contents:
        text = "\n\n".join(json_contents)
    else:
        # 沒有 JSON 結構時，移除 LightRAG header/footer
        text = re.sub(r'Document Chunks.*?Reference Document List[`\'\s)]*:', '', text, flags=re.DOTALL)
        text = re.sub(r'Reference Document List.*', '', text, flags=re.DOTALL)
    # Remove remaining JSON/code block wrappers
    text = re.sub(r'```json\s*', '', text)
    text = re.sub(r'```\s*', '', text)
    text = re.sub(r'\{"reference_id".*?\}', '', text, flags=re.DOTALL)
    # Unescape JSON escape sequences
    text = text.replace('\\n', '\n').replace('\\t', ' ').replace('\\r', '')
    # 移除來源標記
    text = re.sub(r'\[來源[:：][^\]]*\]\s*', '', text)
    # 壓縮空白
    text = re.sub(r'\r\n?', '\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    # 過濾導航垃圾：移除連續短行區塊（>=3 行每行 <20 字且不含中文句號）
    lines = text.split('\n')
    filtered = []
    buf = []
    for line in lines:
        s = line.strip()
        if len(s) < 20 and not re.search(r'[。！？；]', s):
            buf.append(s)
        else:
            if len(buf) < 3:
                filtered.extend(buf)
            buf = []
            filtered.append(s)
    if len(buf) < 3:
        filtered.extend(buf)
    text = '\n'.join(filtered)
    text = re.sub(r'\n{3,}', '\n\n', text).strip()
    return text


def _is_no_context_result(text: str) -> bool:
    """判斷 LightRAG 是否回傳「無可用上下文」訊息。"""
    if not text:
        return True
    t = text.strip().lower()
    if (
        "[no-context]" in t
        or "not able to provide an answer" in t
        or "no relevant document chunks found" in t
    ):
        return True
    # Chinese "I can't answer" patterns from various LLMs
    if "無法根據提供的上下文" in text or "无法根据提供的" in text:
        return True
    if "我無法回答" in text or "我无法回答" in text:
        return True
    if ("很抱歉" in text or "抱歉" in text) and "無法" in text:
        return True
    return False


def _extract_sources(raw_result: str) -> list[dict]:
    """從 LightRAG 查詢結果解析來源。

    直接解析 JSON 結構取得 reference_id → file_path 對應，
    並從 content 中提取 ~100 字前後文作為 snippet。
    """
    import re
    import json as _json
    sources = []
    if not raw_result or not raw_result.strip() or _is_no_context_result(raw_result):
        return sources

    # 1. 建立 reference_id → file_path 對應表（只從 Reference Document List 區段解析）
    ref_map: dict[str, str] = {}
    ref_section = raw_result
    m_ref = re.search(r'Reference Document List[`\'\s)]*:?(.*)$', raw_result, flags=re.DOTALL | re.IGNORECASE)
    if m_ref:
        ref_section = m_ref.group(1)
    for m in re.finditer(r'^\s*\[(\d+)\]\s*(.+)\s*$', ref_section, flags=re.MULTILINE):
        ref_map[m.group(1)] = m.group(2).strip()

    # 2. 解析每個 JSON chunk
    chunks: list[dict] = []
    for line in raw_result.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            obj = _json.loads(line)
            if "content" in obj:
                chunks.append(obj)
        except (ValueError, _json.JSONDecodeError):
            pass

    # 3. 如果 JSON 解析失敗，fallback 用 regex
    if not chunks:
        for m in re.finditer(
            r'\{\s*"reference_id"\s*:\s*"([^"]*)"\s*,\s*"content"\s*:\s*"((?:[^"\\]|\\.)*)"',
            raw_result,
        ):
            chunks.append({"reference_id": m.group(1), "content": m.group(2)})

    if not chunks:
        return sources

    seen_titles = set()
    for chunk in chunks:
        content = chunk.get("content", "")
        # Unescape JSON string
        content = content.replace("\\n", "\n").replace("\\t", " ").strip()
        if len(content) < 10:
            continue

        ref_id = str(chunk.get("reference_id", ""))
        file_path = ref_map.get(ref_id, "")

        # 從 content 中提取來源標記
        title = ""
        source_match = re.search(r'\[來源[:：]\s*(.+?)\]', content)
        pdf_match = re.search(r'\[([^\[\]]+?)\s+p\.(\d+)\]', content)
        if source_match:
            title = source_match.group(1).strip()
        elif pdf_match:
            title = f"{pdf_match.group(1)} p.{pdf_match.group(2)}"
        elif file_path:
            title = file_path
        else:
            # 無法驗證來源時，不生成 citation，避免「亂引用」。
            continue

        if title in seen_titles:
            continue
        seen_titles.add(title)

        # 移除來源標記後，取前後各 ~100 字作為 snippet
        snippet_text = re.sub(r'\[來源[:：][^\]]*\]\s*', '', content)
        snippet_text = re.sub(r'\[[^\[\]]+?\s+p\.\d+\]\s*', '', snippet_text).strip()
        # 保留約 200 字（前後各 100）
        if len(snippet_text) > 200:
            snippet_text = snippet_text[:200] + "…"

        if snippet_text:
            sources.append({
                "title": title,
                "snippet": snippet_text,
                "file": file_path or title,
            })

    return sources


def has_knowledge(project_id: str = "default") -> bool:
    """Check if the knowledge base has retrievable vector data.

    唯一真值：vdb_chunks.json 的 data 陣列非空。
    不再以檔案大小作為判斷依據。
    """
    if project_id == "default":
        working_dir = _DEFAULT_LIGHTRAG_DIR
    else:
        working_dir = project_root(project_id) / "lightrag"
    if not working_dir.exists():
        return False
    vdb_chunks = working_dir / "vdb_chunks.json"
    if not vdb_chunks.exists():
        return False
    try:
        import json
        payload = json.loads(vdb_chunks.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            return len(payload.get("data", [])) > 0
    except Exception:
        pass
    return False
