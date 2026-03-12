"""Embedding 公開 API — 委派給 EmbeddingService（OpenRouter 後端）。

此模組保留原始 API 簽名以確保向後相容。
所有實際 embedding 邏輯已移至 embeddings.service + embeddings.openrouter_backend。

舊 API 函數仍可用：embed_texts, embed_text, embed_image, get_embed_config, etc.
"""

from __future__ import annotations

import asyncio
import logging
import math
import os
import threading
from pathlib import Path
from typing import Callable, Union

log = logging.getLogger(__name__)

# Hard overwrite env guards (do not use setdefault).
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ── 公開常數（向後相容）────────────────────────────────────────

EMBED_DIM = 1024
EMBED_MODEL = "nvidia/llama-nemotron-embed-vl-1b-v2:free"


# ── 舊版 model 載入介面（供相容測試）───────────────────────────

_model = None
_model_lock = threading.Lock()


def _load_model():
    """Legacy shim: real embedding now lives in EmbeddingService."""
    return object()


def _ensure_model():
    global _model
    if _model is not None:
        return _model
    with _model_lock:
        if _model is None:
            _model = _load_model()
    return _model


def _truncate_and_normalize(vec: list[float], dim: int = EMBED_DIM) -> list[float]:
    """Legacy helper kept for backward compatibility tests."""
    out = list(vec[:dim]) if vec else []
    if len(out) < dim:
        out.extend([0.0] * (dim - len(out)))
    norm = math.sqrt(sum(x * x for x in out))
    if norm > 0:
        out = [x / norm for x in out]
    return out


# ── 公開 API（委派給 EmbeddingService）─────────────────────────


def get_embed_config() -> tuple[str, str, str, int]:
    """回傳 (model, key, base, dim)。"""
    return EMBED_MODEL, "", "https://openrouter.ai/api/v1", EMBED_DIM


async def embed_texts(texts: list[str]) -> list[list[float]]:
    """批次文字 embedding（passage encoding，用於索引文件/記憶）。"""
    from embeddings.service import get_embedding_service
    vecs = await get_embedding_service().embed_texts(texts)
    return [_truncate_and_normalize(v, EMBED_DIM) for v in vecs]


async def embed_text(text: str) -> list[float]:
    """單筆文字 embedding（query encoding，用於搜尋查詢）。"""
    from embeddings.service import get_embedding_service
    vec = await get_embedding_service().embed_text(text)
    return _truncate_and_normalize(vec, EMBED_DIM)


async def embed_image(source: Union[str, Path, bytes]) -> list[float]:
    """將圖片轉為 1024 維 L2-normalized 向量。"""
    from embeddings.service import get_embedding_service
    vec = await get_embedding_service().embed_image(source)
    return _truncate_and_normalize(vec, EMBED_DIM)


def _embed_text_sync(text: str) -> list[float]:
    return asyncio.run(embed_text(text))


def _embed_texts_sync(texts: list[str]) -> list[list[float]]:
    return asyncio.run(embed_texts(texts))


# ── 診斷 / 安裝（向後相容）────────────────────────────────────


def get_device_diagnostics() -> dict[str, object]:
    """回傳 embedding 環境診斷。"""
    from embeddings.service import get_embedding_service
    return get_embedding_service().get_device_diagnostics()


def get_runtime_device() -> str:
    """回傳目前 embedding 後端類型。"""
    return "openrouter"


def ensure_model_downloaded(
    *,
    download_if_missing: bool = True,
    progress_callback: Callable[[str, float], None] | None = None,
) -> None:
    """Legacy shim for call sites that expect a setup step."""
    if progress_callback:
        progress_callback("OpenRouter embedding does not require local installation.", 1.0)
    if not download_if_missing:
        return
    log.info("OpenRouter embedding backend does not require local model download.")
