"""Jina Embedding API 後端 — 透過 Jina Cloud API 取得文字/圖片 embedding。

API 文件：https://jina.ai/embeddings/
"""

from __future__ import annotations

import asyncio
import base64
import logging
import mimetypes
import os
import random
import time
from collections import OrderedDict
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Union

import httpx

from embeddings.backend import EmbeddingBackend

log = logging.getLogger(__name__)

_JINA_API_URL = "https://api.jina.ai/v1/embeddings"
_DEFAULT_MODEL = "jina-embeddings-v4"
_DEFAULT_DIM = 1024
_DEFAULT_BATCH_SIZE = 32


class JinaRateLimitError(RuntimeError):
    """Raised when Jina API rate limit is exhausted after retries."""

    def __init__(self, message: str, retry_after: float | None = None):
        super().__init__(message)
        self.retry_after = retry_after


class JinaAPIBackend(EmbeddingBackend):
    """透過 Jina Embedding API 取得文字/圖片向量。"""
    _global_semaphore: asyncio.Semaphore | None = None


    def __init__(
        self,
        api_key: str = "",
        model: str = "",
        dim: int = 0,
        timeout: float = 30.0,
    ) -> None:
        self._api_key = api_key or os.environ.get("JINA_API_KEY", "")
        self._model = model or os.environ.get("JINA_EMBED_MODEL", _DEFAULT_MODEL)
        self._dim = dim or int(
            os.environ.get("EMBED_DIM")
            or os.environ.get("GGUF_EMBED_DIM")
            or str(_DEFAULT_DIM)
        )
        self._timeout = timeout
        self._client: httpx.AsyncClient | None = None
        self._request_lock = asyncio.Lock()
        self._next_allowed_ts = 0.0
        self._min_interval_sec = float(os.environ.get("JINA_EMBED_MIN_INTERVAL", "1.2"))
        self._max_retries = max(0, int(os.environ.get("JINA_EMBED_MAX_RETRIES", "8")))
        self._rate_limit_max_retries = max(
            0, int(os.environ.get("JINA_EMBED_RATE_LIMIT_MAX_RETRIES", str(self._max_retries)))
        )
        self._rate_limit_cooldown_sec = float(
            os.environ.get("JINA_EMBED_RATE_LIMIT_COOLDOWN", "65")
        )
        self._backoff_base_sec = float(os.environ.get("JINA_EMBED_BACKOFF_BASE", "1.0"))
        self._backoff_multiplier = float(os.environ.get("JINA_EMBED_BACKOFF_MULTIPLIER", "2.0"))
        self._backoff_max_sec = float(os.environ.get("JINA_EMBED_BACKOFF_MAX", "8.0"))
        self._backoff_jitter_sec = float(os.environ.get("JINA_EMBED_BACKOFF_JITTER", "0.3"))
        self._batch_size = max(1, int(os.environ.get("JINA_EMBED_BATCH_SIZE", str(_DEFAULT_BATCH_SIZE))))
        self._cache_max_items = max(0, int(os.environ.get("JINA_EMBED_CACHE_MAX_ITEMS", "20000")))
        self._text_cache: OrderedDict[str, list[float]] = OrderedDict()
        self._global_max_concurrency = max(1, int(os.environ.get("JINA_EMBED_MAX_CONCURRENCY", "1")))
        if JinaAPIBackend._global_semaphore is None:
            JinaAPIBackend._global_semaphore = asyncio.Semaphore(self._global_max_concurrency)

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

        # Cache aggressively during ingest merge stage; repeated entities are common.
        results: list[list[float] | None] = [None] * len(texts)
        misses: list[tuple[int, str]] = []
        for idx, t in enumerate(texts):
            cached = self._cache_get(t)
            if cached is not None:
                results[idx] = cached
            else:
                misses.append((idx, t))

        for i in range(0, len(misses), self._batch_size):
            miss_batch = misses[i:i + self._batch_size]
            batch_inputs = [t for _, t in miss_batch]
            vecs = await self._call_api(batch_inputs, task="retrieval.passage")
            for (original_idx, text_value), vec in zip(miss_batch, vecs):
                results[original_idx] = vec
                self._cache_set(text_value, vec)

        if any(v is None for v in results):
            raise RuntimeError("Jina embedding batch result length mismatch")
        return [v for v in results if v is not None]

    async def embed_image(self, image: Union[str, Path, bytes]) -> list[float]:
        """圖片 embedding（jina-embeddings-v4）。"""
        data_url = self._image_to_data_url(image)
        model = self._model if "v4" in self._model else "jina-embeddings-v4"
        results = await self._call_api(
            [{"image": data_url}],
            task="retrieval.passage",
            model=model,
        )
        return results[0]

    def dimension(self) -> int:
        return self._dim

    def provider_signature(self) -> str:
        return f"{self._model}-jina-api-f32-{self._dim}"

    # ── 內部 ──────────────────────────────────────────────────

    async def _call_api(
        self,
        inputs: list[object],
        task: str = "retrieval.passage",
        model: str | None = None,
    ) -> list[list[float]]:
        """呼叫 Jina Embedding API，含 429 節流與自動重試。"""
        client = self._get_client()
        payload = {
            "model": model or self._model,
            "input": inputs,
            "dimensions": self._dim,
            "task": task,
            "late_chunking": False,
        }
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

        total_attempts = max(self._max_retries, self._rate_limit_max_retries) + 1
        for attempt in range(total_attempts):
            await self._acquire_slot()
            try:
                sem = JinaAPIBackend._global_semaphore
                if sem is None:
                    resp = await client.post(_JINA_API_URL, json=payload, headers=headers)
                else:
                    async with sem:
                        resp = await client.post(_JINA_API_URL, json=payload, headers=headers)
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                if attempt >= self._max_retries:
                    raise RuntimeError(f"Jina Embedding API 網路錯誤: {exc}") from exc
                sleep_sec = self._compute_backoff(None, attempt)
                log.warning(
                    "Jina API network error (attempt %d/%d), retry in %.2fs: %s",
                    attempt + 1,
                    total_attempts,
                    sleep_sec,
                    exc,
                )
                await asyncio.sleep(sleep_sec)
                continue

            if resp.status_code == 200:
                data = resp.json()
                embeddings = data.get("data", [])
                embeddings.sort(key=lambda x: x.get("index", 0))
                return [item["embedding"] for item in embeddings]

            error_text = resp.text[:300]
            is_rate_limit = resp.status_code == 429
            should_retry = is_rate_limit or resp.status_code >= 500
            max_retry_for_status = min(self._rate_limit_max_retries, self._max_retries) if is_rate_limit else self._max_retries
            if should_retry and attempt < max_retry_for_status:
                retry_after = self._parse_retry_after(resp.headers.get("Retry-After", ""))
                if is_rate_limit and retry_after is None:
                    retry_after = self._rate_limit_cooldown_sec
                sleep_sec = self._compute_backoff(retry_after, attempt)
                if is_rate_limit:
                    await self._apply_rate_limit_cooldown(sleep_sec)
                log.warning(
                    "Jina API error %d (attempt %d/%d), retry in %.2fs: %s",
                    resp.status_code,
                    attempt + 1,
                    total_attempts,
                    sleep_sec,
                    error_text,
                )
                await asyncio.sleep(sleep_sec)
                continue

            log.error("Jina API error %d: %s", resp.status_code, error_text)
            if is_rate_limit:
                retry_after = self._parse_retry_after(resp.headers.get("Retry-After", ""))
                if retry_after is None:
                    retry_after = self._rate_limit_cooldown_sec
                await self._apply_rate_limit_cooldown(retry_after)
                raise JinaRateLimitError(
                    "Jina Embedding API 達到速率限制（429）。"
                    " 可稍後重試。"
                    f" 建議等待 {int(max(1.0, retry_after))} 秒後重試。",
                    retry_after=retry_after,
                )
            raise RuntimeError(
                f"Jina Embedding API 錯誤 {resp.status_code}: {error_text}"
            )

        raise RuntimeError("Jina Embedding API 重試失敗")

    async def _acquire_slot(self) -> None:
        """單機節流，避免同時大量請求觸發 Jina free tier 429。"""
        if self._min_interval_sec <= 0:
            return
        async with self._request_lock:
            now = time.monotonic()
            wait_sec = max(0.0, self._next_allowed_ts - now)
            if wait_sec > 0:
                await asyncio.sleep(wait_sec)
                now = time.monotonic()
            self._next_allowed_ts = now + self._min_interval_sec

    async def _apply_rate_limit_cooldown(self, wait_seconds: float) -> None:
        if wait_seconds <= 0:
            return
        async with self._request_lock:
            now = time.monotonic()
            self._next_allowed_ts = max(self._next_allowed_ts, now + wait_seconds)

    def _compute_backoff(self, retry_after: float | None, attempt: int) -> float:
        exp = self._backoff_base_sec * (self._backoff_multiplier ** attempt)
        jitter = random.uniform(0.0, self._backoff_jitter_sec) if self._backoff_jitter_sec > 0 else 0.0
        sleep_sec = min(self._backoff_max_sec, exp + jitter)
        if retry_after is not None:
            sleep_sec = max(sleep_sec, retry_after)
        return max(0.0, sleep_sec)

    @staticmethod
    def _parse_retry_after(raw: str) -> float | None:
        raw = (raw or "").strip()
        if not raw:
            return None
        try:
            return max(0.0, float(raw))
        except ValueError:
            pass
        try:
            dt = parsedate_to_datetime(raw)
            if dt is None:
                return None
            now = time.time()
            return max(0.0, dt.timestamp() - now)
        except Exception:
            return None

    def _cache_get(self, text: object) -> list[float] | None:
        if self._cache_max_items <= 0 or not isinstance(text, str):
            return None
        v = self._text_cache.get(text)
        if v is None:
            return None
        self._text_cache.move_to_end(text)
        return v

    def _cache_set(self, text: object, vec: list[float]) -> None:
        if self._cache_max_items <= 0 or not isinstance(text, str):
            return
        self._text_cache[text] = vec
        self._text_cache.move_to_end(text)
        while len(self._text_cache) > self._cache_max_items:
            self._text_cache.popitem(last=False)

    @staticmethod
    def _image_to_data_url(source: Union[str, Path, bytes]) -> str:
        """將圖片來源轉為 data URL（Jina image input）。"""
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
        b64 = base64.b64encode(raw).decode("ascii")
        return f"data:{mime};base64,{b64}"
