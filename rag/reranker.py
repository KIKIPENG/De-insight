"""Jina Reranker — 用 Jina Reranker API 對搜尋結果重新排序。

使用 jina-reranker-v2-base-multilingual（免費），支援中英文。
API 文件：https://jina.ai/reranker/
"""

from __future__ import annotations

import logging
import os
from typing import Any

import httpx

log = logging.getLogger(__name__)

_JINA_RERANK_URL = "https://api.jina.ai/v1/rerank"
_DEFAULT_MODEL = "jina-reranker-v2-base-multilingual"


def _get_api_key() -> str:
    """從 .env 或 os.environ 取得 JINA_API_KEY。"""
    key = os.environ.get("JINA_API_KEY", "")
    if not key:
        try:
            from settings import load_env
            key = load_env().get("JINA_API_KEY", "")
        except Exception:
            pass
    return key


async def rerank(
    query: str,
    documents: list[str],
    top_n: int = 5,
    model: str = "",
    return_documents: bool = True,
) -> list[dict[str, Any]]:
    """用 Jina Reranker 對文件重新排序。

    Args:
        query: 搜尋查詢
        documents: 待排序的文件列表
        top_n: 回傳前 N 筆
        model: 使用的模型（預設 jina-reranker-v2-base-multilingual）
        return_documents: 是否在回傳中包含文件內容

    Returns:
        按相關性排序的結果列表，每項包含：
        - index: 原始位置
        - relevance_score: 相關性分數 (0-1)
        - document: 文件內容（若 return_documents=True）
    """
    api_key = _get_api_key()
    if not api_key:
        log.debug("JINA_API_KEY not set, skipping rerank")
        return []

    if not documents:
        return []

    if len(documents) <= 1:
        # 只有一筆不需要 rerank
        return [{"index": 0, "relevance_score": 1.0, "document": {"text": documents[0]}}]

    model = model or _DEFAULT_MODEL
    top_n = min(top_n, len(documents))

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(
                _JINA_RERANK_URL,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": model,
                    "query": query,
                    "documents": documents,
                    "top_n": top_n,
                    "return_documents": return_documents,
                },
            )

            if resp.status_code != 200:
                log.warning("Jina Reranker error %d: %s", resp.status_code, resp.text[:200])
                return []

            data = resp.json()
            results = data.get("results", [])
            log.info(
                "Reranked %d docs → top %d (scores: %s)",
                len(documents),
                top_n,
                ", ".join(f"{r.get('relevance_score', 0):.3f}" for r in results[:3]),
            )
            return results

    except Exception as e:
        log.warning("Jina Reranker failed: %s", e)
        return []


async def rerank_with_items(
    query: str,
    items: list[Any],
    text_key: str | None = None,
    text_fn=None,
    top_n: int = 5,
) -> list[Any]:
    """對任意物件列表做 rerank，回傳重排後的物件。

    Args:
        query: 搜尋查詢
        items: 原始物件列表
        text_key: 從物件中取出文字的 key（若物件是 dict）
        text_fn: 自訂函式，從物件中取出文字
        top_n: 回傳前 N 筆

    Returns:
        重排後的物件列表（保留原始物件，不修改）
    """
    if not items or len(items) <= 1:
        return items[:top_n]

    # 取出文字
    documents = []
    for item in items:
        if text_fn:
            documents.append(str(text_fn(item)))
        elif text_key and isinstance(item, dict):
            documents.append(str(item.get(text_key, "")))
        else:
            documents.append(str(item))

    results = await rerank(query, documents, top_n=top_n, return_documents=False)
    if not results:
        # Rerank 失敗時回傳原始排序
        return items[:top_n]

    # 按 rerank 結果重排
    reranked = []
    for r in results:
        idx = r.get("index", 0)
        if 0 <= idx < len(items):
            reranked.append(items[idx])
    return reranked
