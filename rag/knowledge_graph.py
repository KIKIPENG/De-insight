"""LightRAG 知識圖譜封裝。"""

import os
from pathlib import Path

from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import openai_complete_if_cache, openai_embed
from lightrag.utils import EmbeddingFunc

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from settings import load_env

WORKING_DIR = Path(__file__).parent.parent / "data" / "lightrag"
WORKING_DIR.mkdir(parents=True, exist_ok=True)

ART_ENTITY_TYPES = [
    "藝術家", "設計師", "建築師", "理論家", "批評家",
    "藝術運動", "設計風格", "藝術流派",
    "媒材", "技法", "創作方法",
    "哲學概念", "批判理論",
    "藝術機構", "美術館", "畫廊",
    "展覽", "作品", "著作", "批評文本",
]

_rag_instance: LightRAG | None = None


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
        rag_base = env.get("RAG_API_BASE", "https://api.openai.com/v1")
        if rag_model.startswith("ollama/"):
            return rag_model.removeprefix("ollama/"), "ollama", "http://localhost:11434/v1"
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

    讀取順序：
    1. EMBED_MODEL / EMBED_API_KEY / EMBED_API_BASE / EMBED_DIM（Settings 寫入）
    2. JINA_API_KEY（向下相容）
    3. 根據 LLM_MODEL 推斷
    """
    env = load_env()

    # 1. Settings 統一格式
    embed_model = env.get("EMBED_MODEL", "")
    if embed_model:
        embed_key = env.get("EMBED_API_KEY", "") or env.get("JINA_API_KEY", "") or env.get("OPENAI_API_KEY", "")
        embed_base = env.get("EMBED_API_BASE", "https://api.openai.com/v1")
        embed_dim = int(env.get("EMBED_DIM", "1024"))
        # Ollama embed
        if env.get("EMBED_PROVIDER", "").startswith("ollama"):
            embed_key = "ollama"
            embed_base = "http://localhost:11434/v1"
        return embed_model, embed_key, embed_base, embed_dim

    # 2. Jina 向下相容
    jina_key = env.get("JINA_API_KEY", "")
    if jina_key:
        return "jina-embeddings-v3", jina_key, "https://api.jina.ai/v1", 1024

    # 3. 根據 LLM_MODEL 推斷
    llm_model = env.get("LLM_MODEL", "")
    if llm_model.startswith("ollama/"):
        return "nomic-embed-text", "ollama", "http://localhost:11434/v1", 768

    api_key = env.get("OPENAI_API_KEY", "") or env.get("CODEX_API_KEY", "")
    return "text-embedding-3-small", api_key, "https://api.openai.com/v1", 1536


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


def get_rag() -> LightRAG:
    global _rag_instance
    if _rag_instance is not None:
        return _rag_instance

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
                resp.raise_for_status()
                result = resp.json()["choices"][0]["message"]["content"]
            # Strip <think>...</think> from reasoning models
            if result and "<think>" in result:
                result = _re.sub(r"<think>[\s\S]*?</think>\s*", "", result)
            return result

    async def embed_func(texts):
        # Bypass LightRAG's openai_embed (which has decorator validation issues
        # with Jina API). Use httpx directly for reliable 1:1 text→vector mapping.
        import httpx
        import numpy as np
        sanitized = [t if t and t.strip() else " " for t in texts]
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                f"{embed_base}/embeddings",
                headers={"Authorization": f"Bearer {embed_key}"},
                json={"model": embed_model, "input": sanitized},
            )
            resp.raise_for_status()
            data = resp.json()
            # Sort by index to ensure correct order
            sorted_data = sorted(data["data"], key=lambda x: x["index"])
            return np.array([item["embedding"] for item in sorted_data], dtype=np.float32)

    _rag_instance = LightRAG(
        working_dir=str(WORKING_DIR),
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


async def _ensure_initialized() -> LightRAG:
    """取得 RAG 並確保 storages 已初始化。"""
    rag = get_rag()
    await rag.initialize_storages()
    return rag


def reset_rag() -> None:
    """Force re-initialization (e.g. after model change)."""
    global _rag_instance
    _rag_instance = None


async def insert_text(text: str, source: str = "") -> None:
    rag = await _ensure_initialized()
    doc = text
    if source:
        doc = f"[來源: {source}]\n\n{text}"
    await rag.ainsert(doc)


async def insert_pdf(path: str) -> None:
    try:
        from pypdf import PdfReader
    except ImportError:
        raise RuntimeError("需要安裝 pypdf: pip install pypdf")

    reader = PdfReader(path)
    filename = Path(path).stem
    chunks = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text and text.strip():
            chunks.append(f"[{filename} p.{i+1}]\n{text.strip()}")

    rag = await _ensure_initialized()
    full_text = "\n\n---\n\n".join(chunks)
    await rag.ainsert(full_text)


async def insert_url(url: str) -> None:
    """Fetch a URL and insert its content into the knowledge base."""
    import httpx
    import re
    import tempfile
    from html.parser import HTMLParser

    async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
        resp = await client.get(url, headers={
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
        })
        resp.raise_for_status()
    content_type = resp.headers.get("content-type", "")

    if "pdf" in content_type or url.lower().endswith(".pdf"):
        # PDF URL — download to temp file, reuse insert_pdf
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(resp.content)
            tmp_path = tmp.name
        try:
            await insert_pdf(tmp_path)
        finally:
            Path(tmp_path).unlink(missing_ok=True)
        return

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
    # Clean up whitespace
    text = re.sub(r"\n{3,}", "\n\n", text).strip()

    if len(text) < 50:
        raise RuntimeError("頁面內容太少，無法匯入")

    # Extract title from HTML
    title_match = re.search(r"<title[^>]*>(.*?)</title>", html, re.IGNORECASE | re.DOTALL)
    source = title_match.group(1).strip() if title_match else url

    await insert_text(text, source=source)


async def query_knowledge(question: str, mode: str = "naive", context_only: bool = True) -> str:
    """查詢知識庫。

    預設 naive + context_only：純向量搜尋，不呼叫 LLM，<1秒回應。
    若需要完整圖譜推理，可傳 mode="hybrid", context_only=False。
    """
    rag = await _ensure_initialized()
    if context_only:
        param = QueryParam(mode=mode, only_need_context=True)
        result = await rag.aquery(question, param=param)
        return result
    param = QueryParam(mode=mode, user_prompt="請用繁體中文回答。\n\n" + question)
    result = await rag.aquery(question, param=param)
    return result


def has_knowledge() -> bool:
    """Check if the knowledge base has any data."""
    for pattern in ["*.graphml", "kv_store_*.json"]:
        for f in WORKING_DIR.glob(pattern):
            if f.stat().st_size > 200:
                return True
    return False
