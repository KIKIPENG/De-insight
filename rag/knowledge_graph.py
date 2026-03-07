"""LightRAG 知識圖譜封裝。"""

import os
from pathlib import Path

from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import openai_complete_if_cache, openai_embed
from lightrag.utils import EmbeddingFunc

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from settings import load_env

from paths import project_root, ensure_project_dirs, DATA_ROOT
_DEFAULT_LIGHTRAG_DIR = DATA_ROOT / "projects" / "default" / "lightrag"

ART_ENTITY_TYPES = [
    "藝術家", "設計師", "建築師", "理論家", "批評家",
    "藝術運動", "設計風格", "藝術流派",
    "媒材", "技法", "創作方法",
    "哲學概念", "批判理論",
    "藝術機構", "美術館", "畫廊",
    "展覽", "作品", "著作", "批評文本",
]

_rag_instance: LightRAG | None = None
_rag_project_id: str | None = None


def _get_llm_config() -> tuple[str, str, str]:
    """Return (model, api_key, api_base) from current env settings.

    可在 .env 設定 RAG_LLM_MODEL / RAG_API_KEY / RAG_API_BASE
    來獨立控制知識庫用的 LLM，不受主聊天模型影響。
    """
    env = load_env()

    # 優先使用獨立的 RAG LLM 設定
    rag_model = env.get("RAG_LLM_MODEL", "")
    if rag_model:
        rag_key = env.get("RAG_API_KEY", "") or env.get("OPENAI_API_KEY", "")
        rag_base = env.get("RAG_API_BASE", "") or env.get("OPENAI_API_BASE", "https://api.openai.com/v1")
        if rag_model.startswith("ollama/"):
            return rag_model.removeprefix("ollama/"), "ollama", "http://localhost:11434/v1"
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


def _is_local_embed() -> bool:
    """判斷是否使用本地 embedding。

    以 EMBED_PROVIDER 為主判斷來源，EMBED_MODE 做 backward compat。
    缺省（env 未設）時回落 local。
    """
    env = load_env()
    provider = env.get("EMBED_PROVIDER", "").lower()
    mode = env.get("EMBED_MODE", "").lower()

    # 明確設為 local
    if provider == "local" or mode == "local":
        return True
    # 明確設為其他 API provider
    if provider and provider != "local":
        return False
    # 缺省：回落 local
    return True


def _get_embed_config() -> tuple[str, str, str, int]:
    """Return (model, api_key, api_base, dim) for embeddings.

    以 EMBED_PROVIDER 為單一真值：
    - "local" 或缺省 → jina-clip-v1 (512)
    - 其他 → 按 provider 設定走 API
    """
    env = load_env()

    # Local embedding (預設)
    if _is_local_embed():
        return "jina-clip-v1", "", "", 512

    # API embedding — 由 EMBED_PROVIDER 決定
    provider = env.get("EMBED_PROVIDER", "").lower()

    if provider.startswith("ollama"):
        embed_model = env.get("EMBED_MODEL", "nomic-embed-text")
        return embed_model, "ollama", "http://localhost:11434/v1", int(env.get("EMBED_DIM", "768"))

    embed_model = env.get("EMBED_MODEL", "")
    if embed_model:
        embed_key = env.get("EMBED_API_KEY", "") or env.get("JINA_API_KEY", "") or env.get("OPENAI_API_KEY", "")
        embed_base = env.get("EMBED_API_BASE", "https://api.openai.com/v1")
        embed_dim = int(env.get("EMBED_DIM", "1024"))
        return embed_model, embed_key, embed_base, embed_dim

    # Backward compat: bare JINA_API_KEY
    jina_key = env.get("JINA_API_KEY", "")
    if jina_key:
        return "jina-embeddings-v3", jina_key, "https://api.jina.ai/v1", 1024

    # Fallback: local (should not reach here due to _is_local_embed)
    return "jina-clip-v1", "", "", 512


def _apply_env() -> None:
    """Set env vars for LightRAG's OpenAI client."""
    env = load_env()
    for key in (
        "OPENAI_API_KEY", "OPENAI_API_BASE",
        "ANTHROPIC_API_KEY", "CODEX_API_KEY",
        "MINIMAX_API_KEY", "MINIMAX_API_BASE",
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

    if llm_model.startswith("codex-cli/"):
        # 用 Codex CLI 的 OAuth 認證，不需要 API key
        codex_model_name = llm_model.removeprefix("codex-cli/")

        async def llm_func(prompt, system_prompt=None, history_messages=None, **kwargs):
            from codex_client import codex_stream
            sys_p = system_prompt or ""
            result = ""
            async for chunk in codex_stream(prompt, sys_p, model=codex_model_name):
                result += chunk
            return result
    else:
        async def llm_func(prompt, system_prompt=None, history_messages=None, **kwargs):
            import re as _re
            import httpx as _httpx
            # Drop response_format — reasoning models (MiniMax-M2.5) return
            # <think> tags that break structured output validation.
            kwargs.pop("response_format", None)
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            for m in (history_messages or []):
                messages.append(m)
            messages.append({"role": "user", "content": prompt})
            async with _httpx.AsyncClient(timeout=120.0) as client:
                resp = await client.post(
                    f"{llm_base}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {llm_key}",
                        "HTTP-Referer": "https://github.com/De-insight",
                        "X-Title": "De-insight",
                    },
                    json={"model": llm_model, "messages": messages},
                )
                if resp.status_code >= 400:
                    import logging as _log
                    _log.getLogger("rag.llm").error(
                        f"RAG LLM {resp.status_code} ({llm_model}): {resp.text[:500]}"
                    )
                resp.raise_for_status()
                result = resp.json()["choices"][0]["message"]["content"]
            # Strip <think>...</think> from reasoning models
            if result and "<think>" in result:
                result = _re.sub(r"<think>[\s\S]*?</think>\s*", "", result)
            return result

    if _is_local_embed():
        async def embed_func(texts):
            import numpy as np
            from embeddings.local import embed_text
            results = []
            for t in texts:
                t = t if t and t.strip() else " "
                vec = await embed_text(t)
                results.append(vec)
            return np.array(results, dtype=np.float32)
    else:
        async def embed_func(texts):
            import httpx
            import numpy as np
            sanitized = [t if t and t.strip() else " " for t in texts]
            n = len(sanitized)
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.post(
                    f"{embed_base}/embeddings",
                    headers={"Authorization": f"Bearer {embed_key}"},
                    json={"model": embed_model, "input": sanitized},
                )
                resp.raise_for_status()
                data = resp.json()
                sorted_data = sorted(data["data"], key=lambda x: x["index"])
                embeddings = [item["embedding"] for item in sorted_data]
                # Ensure exactly N vectors for N inputs (some APIs return extras via late chunking)
                if len(embeddings) > n:
                    embeddings = embeddings[:n]
                elif len(embeddings) < n:
                    # Pad missing with zeros
                    dim = len(embeddings[0]) if embeddings else embed_dim
                    while len(embeddings) < n:
                        embeddings.append([0.0] * dim)
                return np.array(embeddings, dtype=np.float32)

    if project_id == "default":
        working_dir = _DEFAULT_LIGHTRAG_DIR
        working_dir.mkdir(parents=True, exist_ok=True)
    else:
        working_dir = ensure_project_dirs(project_id) / "lightrag"

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
        default_llm_timeout=600,
        llm_model_max_async=1,
        addon_params={
            "entity_types": ART_ENTITY_TYPES,
            "example_number": 3,
            "language": "繁體中文",
        },
    )
    return _rag_instance


async def _ensure_initialized(project_id: str | None = None) -> LightRAG:
    """取得 RAG 並確保 storages 已初始化。"""
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


async def insert_text(text: str, source: str = "", project_id: str = "default") -> str:
    """插入文字到知識庫。回傳警告訊息（空字串表示完全成功）。"""
    rag = await _ensure_initialized(project_id=project_id)
    await _clear_failed(rag)
    doc = text
    if source:
        doc = f"[來源: {source}]\n\n{text}"
    await rag.ainsert(doc)
    return await _flush_and_check(rag, 0, context=source or "text")


async def insert_pdf(path: str, project_id: str = "default", title: str = "") -> dict:
    """匯入 PDF，回傳 {"title": str, "page_count": int, "file_size": int, "saved_path": str}。"""
    import re as _re
    try:
        from pypdf import PdfReader
    except ImportError:
        raise RuntimeError("需要安裝 pypdf: pip install pypdf")

    file_size = Path(path).stat().st_size if Path(path).exists() else 0
    reader = PdfReader(path)
    page_count = len(reader.pages)

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

    chunks = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text and text.strip():
            chunks.append(f"[{display_name} p.{i+1}]\n{text.strip()}")

    rag = await _ensure_initialized(project_id=project_id)
    await _clear_failed(rag)
    full_text = "\n\n---\n\n".join(chunks)
    await rag.ainsert(full_text)
    warning = await _flush_and_check(rag, 0, context=display_name)

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


async def insert_url(url: str, project_id: str = "default", title: str = "") -> dict:
    """Fetch a URL and insert its content. 回傳 {"title": str, "page_count": 0, "file_size": int}。"""
    import httpx
    import re
    import tempfile
    from html.parser import HTMLParser

    async with httpx.AsyncClient(timeout=60.0, follow_redirects=True) as client:
        resp = await client.get(url, headers={
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
        })
        resp.raise_for_status()
    content_type = resp.headers.get("content-type", "")

    if "pdf" in content_type or url.lower().endswith(".pdf"):
        # Extract readable filename from URL
        from urllib.parse import urlparse, unquote
        url_path = urlparse(url).path
        url_filename = unquote(Path(url_path).stem) if url_path else ""
        # Sanitize: keep only safe chars
        url_filename = re.sub(r'[^\w\-\.\(\)\[\]\u4e00-\u9fff]+', '_', url_filename).strip('_')
        if not url_filename or len(url_filename) < 3:
            url_filename = "downloaded_pdf"

        tmp_dir = Path(tempfile.gettempdir())
        tmp_path = str(tmp_dir / f"{url_filename}.pdf")
        Path(tmp_path).write_bytes(resp.content)
        try:
            return await insert_pdf(tmp_path, project_id=project_id, title=title)
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    # HTML — extract text
    html = resp.text

    class _TextExtractor(HTMLParser):
        def __init__(self):
            super().__init__()
            self.parts: list[str] = []
            self._skip = False

        def handle_starttag(self, tag, attrs):
            self._skip = tag in ("script", "style", "nav", "footer", "header")

        def handle_endtag(self, tag):
            if tag in ("script", "style", "nav", "footer", "header"):
                self._skip = False
            if tag in ("p", "div", "br", "h1", "h2", "h3", "h4", "li", "tr"):
                self.parts.append("\n")

        def handle_data(self, data):
            if not self._skip:
                self.parts.append(data)

    extractor = _TextExtractor()
    extractor.feed(html)
    text = "".join(extractor.parts)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()

    if len(text) < 50:
        raise RuntimeError("頁面內容太少，無法匯入")

    if title.strip():
        source = title.strip()
    else:
        title_match = re.search(r"<title[^>]*>(.*?)</title>", html, re.IGNORECASE | re.DOTALL)
        source = title_match.group(1).strip() if title_match else url

    warning = await insert_text(text, source=source, project_id=project_id)
    result = {"title": source, "page_count": 0, "file_size": len(resp.content)}
    if warning:
        result["warning"] = warning
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


async def insert_doi(doi: str, project_id: str = "default", title: str = "") -> dict:
    """解析 DOI → 取得 PDF URL → insert_url()。找不到 open access 時 raise。"""
    pdf_url = await resolve_doi(doi)
    if not pdf_url:
        raise RuntimeError(f"找不到 open access 版本：{doi}")
    return await insert_url(pdf_url, project_id=project_id, title=title)


async def query_knowledge(question: str, mode: str = "naive", context_only: bool = True, project_id: str | None = None) -> tuple[str, list[dict]]:
    """查詢知識庫。

    預設 naive + context_only：純向量搜尋，不呼叫 LLM，<1秒回應。
    若需要完整圖譜推理，可傳 mode="hybrid", context_only=False。

    回傳 (result_text, sources)。
    sources 格式：[{"title": str, "snippet": str, "file": str}]
    若無來源資訊則回傳空 list。
    """
    rag = await _ensure_initialized(project_id=project_id)
    if context_only:
        param = QueryParam(mode=mode, only_need_context=True)
        result = await rag.aquery(question, param=param)
        if _is_no_context_result(result):
            return "", []
        sources = _extract_sources(result)
        return result, sources
    param = QueryParam(mode=mode, user_prompt="請用繁體中文回答。\n\n" + question)
    result = await rag.aquery(question, param=param)
    if _is_no_context_result(result):
        return "", []
    sources = _extract_sources(result)
    return result, sources


def _clean_rag_chunk(text: str) -> str:
    """清理 LightRAG 原始 chunk，移除 JSON 包裝、轉義字元、metadata。"""
    import re
    # Remove LightRAG header lines
    text = re.sub(r'Document Chunks.*?Reference Document List[`\'\s)]*:', '', text, flags=re.DOTALL)
    text = re.sub(r'Reference Document List.*', '', text, flags=re.DOTALL)
    # Extract content from JSON format: {"reference_id": "", "content": "..."}
    json_contents = re.findall(r'"content"\s*:\s*"((?:[^"\\]|\\.)*)"', text)
    if json_contents:
        text = "\n\n".join(json_contents)
    # Remove remaining JSON/code block wrappers
    text = re.sub(r'```json\s*', '', text)
    text = re.sub(r'```\s*', '', text)
    text = re.sub(r'\{"reference_id".*?\}', '', text, flags=re.DOTALL)
    # Unescape \n to real newlines
    text = text.replace('\\n', '\n')
    # Clean up whitespace
    text = re.sub(r'\n{3,}', '\n\n', text).strip()
    return text


def _is_no_context_result(text: str) -> bool:
    """判斷 LightRAG 是否回傳「無可用上下文」訊息。"""
    if not text:
        return True
    t = text.strip().lower()
    return (
        "[no-context]" in t
        or "not able to provide an answer" in t
        or "no relevant document chunks found" in t
    )


def _extract_sources(raw_result: str) -> list[dict]:
    """Extract source information from LightRAG query result.

    每個 source 的 snippet 至少保留 200 字上下文。
    """
    import re
    sources = []
    if not raw_result or not raw_result.strip() or _is_no_context_result(raw_result):
        return sources

    cleaned = _clean_rag_chunk(raw_result)
    if not cleaned:
        return sources

    # Split by [來源: ...] or [filename p.N] markers into sections
    # Each section is a source
    parts = re.split(r'(?=\[來源[:：]|\[[^\[\]]+?\s+p\.\d+\])', cleaned)
    parts = [p.strip() for p in parts if p.strip() and len(p.strip()) > 10]

    if not parts:
        parts = [cleaned]

    seen_titles = set()
    for part in parts:
        title = ""
        file_name = ""
        source_match = re.search(r'\[來源[:：]\s*(.+?)\]', part)
        pdf_match = re.search(r'\[([^\[\]]+?)\s+p\.(\d+)\]', part)
        if source_match:
            title = source_match.group(1).strip()
            file_name = title
            # Remove the marker from snippet
            part = part.replace(source_match.group(0), '').strip()
        elif pdf_match:
            title = f"{pdf_match.group(1)} p.{pdf_match.group(2)}"
            file_name = pdf_match.group(1)
            part = part.replace(pdf_match.group(0), '').strip()

        if not title:
            first_line = part.split('\n')[0][:40]
            title = first_line if first_line else "知識庫內容"

        if title in seen_titles:
            continue
        seen_titles.add(title)

        # Keep snippet — at least 200 chars, up to 600
        snippet = part[:600] if len(part) > 600 else part
        if snippet:
            sources.append({"title": title, "snippet": snippet, "file": file_name})

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
