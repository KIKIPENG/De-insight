"""Jina Embedding API 後端 — 透過 Jina Cloud API 取得文字 embedding。

純文字 embedding，不支援圖片（圖片 embedding 仍需 GGUF 本地模型）。
API 文件：https://jina.ai/embeddings/
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Union

import httpx

from embeddings.backend import EmbeddingBackend

log = logging.getLogger(__name__)

_JINA_API_URL = "https://api.jina.ai/v1/embeddings"
_DEFAULT_MODEL = "jina-embeddings-v3"
_DEFAULT_DIM = 1024
_BATCH_SIZE = 64  # Jina API 最大單次 batch


class JinaAPIBackend(EmbeddingBackend):
    """透過 Jina Embedding API 取得文字向量。

    不支援 embed_image()，會 raise NotImplementedError。
    需要圖片 embedding 時應使用 GGUF 後端。
    """

    def __init__(
        self,
        api_key: str = "",
        model: str = "",
        dim: int = 0,
        timeout: float = 30.0,
    ) -> None:
        self._api_key = api_key or os.environ.get("JINA_API_KEY", "")
        self._model = model or os.environ.get("JINA_EMBED_MODEL", _DEFAULT_MODEL)
        self._dim = dim or int(os.environ.get("GGUF_EMBED_DIM", str(_DEFAULT_DIM)))
        self._timeout = timeout
        self._client: httpx.AsyncClient | None = None

        if not self._api_key:
            raise ValueError(
                "JINA_API_KEY 未設定。請在 .env 中加入 JINA_API_KEY=jina_xxx "
                "或到 https://jina.ai/ 取得免費 API Key。"
            )

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=self._timeout)
        return self._client

    async def close(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()

    # ── 公開 API ──────────────────────────────────────────────

    async def embed_query(self, text: str) -> list[float]:
        """Query encoding（搜尋用）。"""
        results = await self._call_api([text], task="retrieval.query")
        return results[0]

    async def embed_passages(self, texts: list[str]) -> list[list[float]]:
        """Passage encoding（文件索引用）。"""
        if not texts:
            return []
        all_vecs: list[list[float]] = []
        for i in range(0, len(texts), _BATCH_SIZE):
            batch = texts[i:i + _BATCH_SIZE]
            vecs = await self._call_api(batch, task="retrieval.passage")
            all_vecs.extend(vecs)
        return all_vecs

    async def embed_image(self, image: Union[str, Path, bytes]) -> list[float]:
        """Jina API 文字後端不支援圖片 embedding。"""
        raise NotImplementedError(
            "Jina API 後端僅支援文字 embedding。"
            " 圖片 embedding 需要本地 GGUF 模型（EMBED_PROVIDER=gguf）。"
        )

    def dimension(self) -> int:
        return self._dim

    def provider_signature(self) -> str:
        return f"{self._model}-jina-api-f32-{self._dim}"

    # ── 內部 ──────────────────────────────────────────────────

    async def _call_api(
        self,
        texts: list[str],
        task: str = "retrieval.passage",
    ) -> list[list[float]]:
        """呼叫 Jina Embedding API。"""
        client = self._get_client()
        payload = {
            "model": self._model,
            "input": texts,
            "dimensions": self._dim,
            "task": task,
            "late_chunking": False,
        }
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

        resp = await client.post(_JINA_API_URL, json=payload, headers=headers)

        if resp.status_code != 200:
            error_text = resp.text[:300]
            log.error("Jina API error %d: %s", resp.status_code, error_text)
            raise RuntimeError(
                f"Jina Embedding API 錯誤 {resp.status_code}: {error_text}"
            )

        data = resp.json()
        embeddings = data.get("data", [])
        # Sort by index to ensure order
        embeddings.sort(key=lambda x: x.get("index", 0))
        return [item["embedding"] for item in embeddings]
