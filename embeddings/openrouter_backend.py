"""OpenRouter embedding backend for text and image retrieval."""

from __future__ import annotations

import asyncio
import base64
import logging
import mimetypes
import os
import random
import time
from collections import OrderedDict
from pathlib import Path
from typing import Union

import httpx

from embeddings.backend import EmbeddingBackend

log = logging.getLogger(__name__)

_DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
_DEFAULT_MODEL = "nvidia/llama-nemotron-embed-vl-1b-v2:free"
_DEFAULT_DIM = 1024
_DEFAULT_BATCH_SIZE = 16


class OpenRouterEmbeddingBackend(EmbeddingBackend):
    """OpenRouter-hosted multimodal embedding backend."""

    _global_semaphore: asyncio.Semaphore | None = None

    def __init__(
        self,
        api_key: str = "",
        model: str = "",
        dim: int = 0,
        base_url: str = "",
        timeout: float = 45.0,
    ) -> None:
        self._api_key = (
            api_key
            or os.environ.get("EMBED_API_KEY", "")
            or os.environ.get("OPENROUTER_API_KEY", "")
        )
        self._model = model or os.environ.get("EMBED_MODEL", _DEFAULT_MODEL)
        self._dim = dim or int(os.environ.get("EMBED_DIM", str(_DEFAULT_DIM)))
        self._base_url = (
            base_url
            or os.environ.get("EMBED_API_BASE", "")
            or os.environ.get("OPENROUTER_API_BASE", "")
            or _DEFAULT_BASE_URL
        ).rstrip("/")
        self._timeout = timeout
        self._client: httpx.AsyncClient | None = None
        self._request_lock = asyncio.Lock()
        self._next_allowed_ts = 0.0
        self._min_interval_sec = float(
            os.environ.get("OPENROUTER_EMBED_MIN_INTERVAL", "0.35")
        )
        self._max_retries = max(
            0, int(os.environ.get("OPENROUTER_EMBED_MAX_RETRIES", "4"))
        )
        self._batch_size = max(
            1, int(os.environ.get("OPENROUTER_EMBED_BATCH_SIZE", str(_DEFAULT_BATCH_SIZE)))
        )
        self._cache_max_items = max(
            0, int(os.environ.get("OPENROUTER_EMBED_CACHE_MAX_ITEMS", "20000"))
        )
        self._text_cache: OrderedDict[str, list[float]] = OrderedDict()
        self._global_max_concurrency = max(
            1, int(os.environ.get("OPENROUTER_EMBED_MAX_CONCURRENCY", "2"))
        )
        if OpenRouterEmbeddingBackend._global_semaphore is None:
            OpenRouterEmbeddingBackend._global_semaphore = asyncio.Semaphore(
                self._global_max_concurrency
            )

        if not self._api_key:
            raise ValueError(
                "OPENROUTER_API_KEY 未設定。請在 .env 中加入 OPENROUTER_API_KEY。"
            )

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self._base_url,
                timeout=self._timeout,
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": os.environ.get("OPENROUTER_HTTP_REFERER", "https://de-insight.local"),
                    "X-Title": os.environ.get("OPENROUTER_APP_TITLE", "De-insight"),
                },
            )
        return self._client

    async def close(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()

    async def embed_query(self, text: str) -> list[float]:
        results = await self._call_api([text])
        return results[0]

    async def embed_passages(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        results: list[list[float] | None] = [None] * len(texts)
        misses: list[tuple[int, str]] = []
        for idx, text in enumerate(texts):
            cached = self._cache_get(text)
            if cached is not None:
                results[idx] = cached
            else:
                misses.append((idx, text))

        for i in range(0, len(misses), self._batch_size):
            miss_batch = misses[i:i + self._batch_size]
            payload = [text for _, text in miss_batch]
            vecs = await self._call_api(payload)
            for (original_idx, text_value), vec in zip(miss_batch, vecs):
                results[original_idx] = vec
                self._cache_set(text_value, vec)

        if any(v is None for v in results):
            raise RuntimeError("OpenRouter embedding batch result length mismatch")
        return [v for v in results if v is not None]

    async def embed_image(self, image: Union[str, Path, bytes]) -> list[float]:
        image_url = self._image_to_data_url(image)
        results = await self._call_api([[{"type": "input_image", "image_url": image_url}]])
        return results[0]

    def dimension(self) -> int:
        return self._dim

    def provider_signature(self) -> str:
        return f"{self._model}-openrouter-f32-{self._dim}"

    async def _call_api(self, inputs: list[object]) -> list[list[float]]:
        client = self._get_client()
        payload = {
            "model": self._model,
            "input": inputs,
            "encoding_format": "float",
            "dimensions": self._dim,
            "provider": {"allow_fallbacks": False, "data_collection": "deny"},
        }

        for attempt in range(self._max_retries + 1):
            await self._acquire_slot()
            try:
                sem = OpenRouterEmbeddingBackend._global_semaphore
                if sem is None:
                    resp = await client.post("/embeddings", json=payload)
                else:
                    async with sem:
                        resp = await client.post("/embeddings", json=payload)
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                if attempt >= self._max_retries:
                    raise RuntimeError(f"OpenRouter Embedding API 網路錯誤: {exc}") from exc
                await asyncio.sleep(self._compute_backoff(attempt))
                continue

            if resp.status_code == 200:
                data = resp.json()
                items = data.get("data", [])
                items.sort(key=lambda x: x.get("index", 0))
                return [item["embedding"] for item in items]

            body = resp.text[:400]
            if resp.status_code in (429, 500, 502, 503, 504) and attempt < self._max_retries:
                retry_after = resp.headers.get("Retry-After", "").strip()
                wait_sec = float(retry_after) if retry_after.isdigit() else self._compute_backoff(attempt)
                log.warning(
                    "OpenRouter embedding error %d (attempt %d/%d), retry in %.2fs: %s",
                    resp.status_code,
                    attempt + 1,
                    self._max_retries + 1,
                    wait_sec,
                    body,
                )
                await asyncio.sleep(wait_sec)
                continue

            raise RuntimeError(f"OpenRouter Embedding API 錯誤 {resp.status_code}: {body}")

        raise RuntimeError("OpenRouter Embedding API 重試失敗")

    async def _acquire_slot(self) -> None:
        if self._min_interval_sec <= 0:
            return
        async with self._request_lock:
            now = time.monotonic()
            wait_sec = max(0.0, self._next_allowed_ts - now)
            if wait_sec > 0:
                await asyncio.sleep(wait_sec)
                now = time.monotonic()
            self._next_allowed_ts = now + self._min_interval_sec

    @staticmethod
    def _compute_backoff(attempt: int) -> float:
        base = min(8.0, 0.75 * (2 ** attempt))
        return base + random.uniform(0.0, 0.25)

    def _cache_get(self, text: object) -> list[float] | None:
        if self._cache_max_items <= 0 or not isinstance(text, str):
            return None
        value = self._text_cache.get(text)
        if value is None:
            return None
        self._text_cache.move_to_end(text)
        return value

    def _cache_set(self, text: object, vec: list[float]) -> None:
        if self._cache_max_items <= 0 or not isinstance(text, str):
            return
        self._text_cache[text] = vec
        self._text_cache.move_to_end(text)
        while len(self._text_cache) > self._cache_max_items:
            self._text_cache.popitem(last=False)

    @staticmethod
    def _image_to_data_url(source: Union[str, Path, bytes]) -> str:
        if isinstance(source, bytes):
            b64 = base64.b64encode(source).decode("ascii")
            return f"data:image/jpeg;base64,{b64}"

        if isinstance(source, str) and (
            source.startswith("http://")
            or source.startswith("https://")
            or source.startswith("data:image/")
        ):
            return source

        path = Path(source)
        raw = path.read_bytes()
        mime, _ = mimetypes.guess_type(path.name)
        if not mime or not mime.startswith("image/"):
            mime = "image/jpeg"
        return f"data:{mime};base64,{base64.b64encode(raw).decode('ascii')}"
