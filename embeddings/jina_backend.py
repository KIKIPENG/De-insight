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
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Union

import httpx

from embeddings.backend import EmbeddingBackend

log = logging.getLogger(__name__)

_JINA_API_URL = "https://api.jina.ai/v1/embeddings"
_DEFAULT_MODEL = "jina-embeddings-v4"
_DEFAULT_DIM = 1024
_BATCH_SIZE = 64  # Jina API 最大單次 batch


class JinaAPIBackend(EmbeddingBackend):
    """透過 Jina Embedding API 取得文字/圖片向量。"""

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
        self._max_retries = max(0, int(os.environ.get("JINA_EMBED_MAX_RETRIES", "5")))
        self._backoff_base_sec = float(os.environ.get("JINA_EMBED_BACKOFF_BASE", "1.0"))
        self._backoff_multiplier = float(os.environ.get("JINA_EMBED_BACKOFF_MULTIPLIER", "2.0"))
        self._backoff_max_sec = float(os.environ.get("JINA_EMBED_BACKOFF_MAX", "20.0"))
        self._backoff_jitter_sec = float(os.environ.get("JINA_EMBED_BACKOFF_JITTER", "0.3"))

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

        for attempt in range(self._max_retries + 1):
            await self._acquire_slot()
            try:
                resp = await client.post(_JINA_API_URL, json=payload, headers=headers)
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                if attempt >= self._max_retries:
                    raise RuntimeError(f"Jina Embedding API 網路錯誤: {exc}") from exc
                sleep_sec = self._compute_backoff(None, attempt)
                log.warning(
                    "Jina API network error (attempt %d/%d), retry in %.2fs: %s",
                    attempt + 1,
                    self._max_retries + 1,
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
            if should_retry and attempt < self._max_retries:
                retry_after = self._parse_retry_after(resp.headers.get("Retry-After", ""))
                sleep_sec = self._compute_backoff(retry_after, attempt)
                log.warning(
                    "Jina API error %d (attempt %d/%d), retry in %.2fs: %s",
                    resp.status_code,
                    attempt + 1,
                    self._max_retries + 1,
                    sleep_sec,
                    error_text,
                )
                await asyncio.sleep(sleep_sec)
                continue

            log.error("Jina API error %d: %s", resp.status_code, error_text)
            if is_rate_limit:
                raise RuntimeError(
                    "Jina Embedding API 達到速率限制（429）。"
                    " 可稍後重試，或將 EMBED_PROVIDER 改為 gguf 使用本地模型。"
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
