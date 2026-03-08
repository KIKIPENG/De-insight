"""GGUF 多模態 Embedding 後端 — 唯一對接 llama-server embeddings API。

- 文字 query: 自動加 "Query: " 前綴
- 文字 passage: 自動加 "Passage: " 前綴
- 圖片: base64 編碼送入 llama-server multimodal endpoint
- 回傳的向量做 Matryoshka 截斷 + L2 正規化
"""

from __future__ import annotations

import asyncio
import base64
import io
import logging
import math
from pathlib import Path
from typing import Union

import httpx

from embeddings.backend import EmbeddingBackend

log = logging.getLogger(__name__)

import os as _os

_EMBED_DIM = 1024
_BATCH_SIZE = int(_os.environ.get("GGUF_EMBED_BATCH_SIZE", "32"))


class GGUFMultimodalBackend(EmbeddingBackend):
    """透過 llama-server HTTP API 取得 embedding 的後端。"""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8999",
        dim: int = _EMBED_DIM,
        timeout: float = 60.0,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._dim = dim
        self._timeout = timeout
        self._client: httpx.AsyncClient | None = None
        # API 格式偵測：None=未偵測, "input"=OpenAI, "prompt"=llama.cpp 原生
        self._api_format: str | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self._base_url,
                timeout=self._timeout,
            )
        return self._client

    async def close(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    # ── EmbeddingBackend 介面 ───────────────────────────────────

    async def embed_query(self, text: str) -> list[float]:
        """Query encoding: 加 'Query: ' 前綴。"""
        prefixed = f"Query: {text}"
        vecs = await self._embed_texts([prefixed])
        return vecs[0]

    async def embed_passages(self, texts: list[str]) -> list[list[float]]:
        """Passage encoding: 加 'Passage: ' 前綴。"""
        if not texts:
            return []
        prefixed = [f"Passage: {t}" if t and t.strip() else "Passage:  " for t in texts]
        return await self._embed_texts(prefixed)

    async def embed_image(self, image: Union[str, Path, bytes]) -> list[float]:
        """圖片 embedding: 透過 llama-server multimodal endpoint。"""
        b64 = self._image_to_base64(image)
        return await self._embed_image_b64(b64)

    def dimension(self) -> int:
        return self._dim

    def provider_signature(self) -> str:
        import os
        model_file = os.environ.get(
            "GGUF_MODEL_FILE",
            "jina-embeddings-v4-text-retrieval-Q4_K_M.gguf",
        )
        return f"jina-v4-gguf-{model_file}-{self._dim}"

    # ── 內部實作 ────────────────────────────────────────────────

    async def _embed_texts(self, texts: list[str]) -> list[list[float]]:
        """批次 embedding，自動分批。"""
        all_vecs: list[list[float]] = []

        for i in range(0, len(texts), _BATCH_SIZE):
            batch = texts[i:i + _BATCH_SIZE]
            vecs = await self._call_embedding(batch)
            all_vecs.extend(vecs)

        return all_vecs

    async def _call_embedding(self, texts: list[str]) -> list[list[float]]:
        """呼叫 llama-server /v1/embeddings endpoint。

        自動偵測 API 格式（首次呼叫），之後記住格式避免重複 500。
        - OpenAI 格式: {"input": ...} → {"data": [{"embedding": [...]}]}
        - llama.cpp 原生: {"prompt": ...} → {"embedding": [...]}
        """
        client = await self._get_client()

        # ── 已偵測為 prompt 格式：直接走快速路徑 ──
        if self._api_format == "prompt":
            return await self._call_prompt_format(client, texts)

        # ── 首次或已知為 input 格式：嘗試 OpenAI 格式 ──
        payload = {"input": texts if len(texts) > 1 else texts[0]}

        try:
            resp = await client.post("/v1/embeddings", json=payload)
            resp.raise_for_status()
            data = resp.json()
        except httpx.HTTPStatusError as exc:
            body = (exc.response.text or "")[:500]
            if exc.response.status_code >= 500 and "key 'prompt' not found" in body:
                # 切換到 prompt 格式，之後不再嘗試 input
                self._api_format = "prompt"
                log.info("llama-server 不支援 OpenAI input 格式，切換到 prompt 格式")
                return await self._call_prompt_format(client, texts)
            if "too large" in body:
                raise RuntimeError(
                    "llama-server: 文字太長無法 embedding。請重啟 llama-server（會自動使用更大的 batch size）。"
                ) from exc
            log.error("llama-server embedding failed (%d): %s", exc.response.status_code, body)
            raise RuntimeError(f"llama-server embedding 失敗: HTTP {exc.response.status_code}") from exc
        except httpx.ConnectError as exc:
            raise RuntimeError(
                "無法連接 llama-server。請確認 server 已啟動。"
            ) from exc

        self._api_format = "input"
        embeddings = sorted(data["data"], key=lambda d: d["index"])
        return [self._truncate_and_normalize(e["embedding"]) for e in embeddings]

    async def _call_prompt_format(
        self, client: httpx.AsyncClient, texts: list[str],
    ) -> list[list[float]]:
        """llama.cpp 原生 prompt 格式：並行送 {"prompt": text}，上限 4 併發。"""
        sem = asyncio.Semaphore(4)

        async def _single(t: str) -> list[float]:
            async with sem:
                try:
                    resp = await client.post("/v1/embeddings", json={"prompt": t})
                    resp.raise_for_status()
                except httpx.HTTPStatusError as exc:
                    body = (exc.response.text or "")[:500]
                    if "too large" in body:
                        raise RuntimeError(
                            "llama-server: 文字太長無法 embedding。"
                            " 請重啟 llama-server 或增大 -ub 參數。"
                        ) from exc
                    log.error("llama-server prompt embedding failed (%d): %s", exc.response.status_code, body)
                    raise RuntimeError(f"llama-server embedding 失敗: HTTP {exc.response.status_code}") from exc
                data = resp.json()
                if "embedding" in data and not isinstance(data.get("data"), list):
                    return self._truncate_and_normalize(data["embedding"])
                emb2 = sorted(data["data"], key=lambda d: d.get("index", 0))
                return self._truncate_and_normalize(emb2[0]["embedding"])

        results = await asyncio.gather(*[_single(t) for t in texts])
        return list(results)

    async def _embed_image_b64(self, b64_data: str) -> list[float]:
        """透過 llama-server 的 multimodal endpoint 取得圖片 embedding。"""
        client = await self._get_client()

        # llama.cpp multimodal embedding 格式
        # content 不可為空字串，否則 server 回傳文字生成格式而非 embedding
        payload = {
            "content": "<image>",
            "image_data": [{"data": b64_data, "id": 0}],
        }

        try:
            resp = await client.post("/embedding", json=payload)
            resp.raise_for_status()
            data = resp.json()
        except httpx.HTTPStatusError as exc:
            log.error("llama-server image embedding failed (%d): %s", exc.response.status_code, exc.response.text[:500])
            raise RuntimeError(f"llama-server 圖片 embedding 失敗: HTTP {exc.response.status_code}") from exc
        except httpx.ConnectError as exc:
            raise RuntimeError(
                "無法連接 llama-server。請確認 server 已啟動。"
            ) from exc

        # 回傳格式: {"embedding": [...]} 或 {"data": [{"embedding": [...]}]}
        if "embedding" in data:
            raw = data["embedding"]
        elif "data" in data and data["data"]:
            raw = data["data"][0]["embedding"]
        else:
            raise RuntimeError(f"llama-server 回傳格式異常: {list(data.keys())}")

        # 多模態可能回傳巢狀 list（多個 patch 向量），取平均
        if raw and isinstance(raw[0], list):
            n = len(raw)
            dim = len(raw[0])
            raw = [sum(raw[j][i] for j in range(n)) / n for i in range(dim)]

        return self._truncate_and_normalize(raw)

    def _truncate_and_normalize(self, vec: list[float]) -> list[float]:
        """Matryoshka 截斷到 self._dim 後做 L2 normalize。"""
        truncated = vec[:self._dim]
        # L2 normalize
        norm = math.sqrt(sum(x * x for x in truncated)) or 1.0
        return [x / norm for x in truncated]

    @staticmethod
    def _image_to_base64(source: Union[str, Path, bytes]) -> str:
        """將圖片來源轉為 base64 字串。"""
        if isinstance(source, bytes):
            return base64.b64encode(source).decode("ascii")

        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"圖片檔案不存在: {path}")

        return base64.b64encode(path.read_bytes()).decode("ascii")
