"""Dynamic model registry helpers (incremental rollout)."""

from __future__ import annotations

import asyncio
import os
import time
from typing import Final

import httpx
import logging

log = logging.getLogger(__name__)

_OPENROUTER_MODELS_URL: Final[str] = "https://openrouter.ai/api/v1/models"
_CACHE_TTL_SEC: Final[int] = 300

_cache: dict[str, tuple[float, list[str]]] = {}


def _from_cache(key: str) -> list[str] | None:
    item = _cache.get(key)
    if not item:
        return None
    ts, models = item
    if (time.time() - ts) > _CACHE_TTL_SEC:
        return None
    return models


def _save_cache(key: str, models: list[str]) -> None:
    _cache[key] = (time.time(), models)


async def fetch_openrouter_models(api_key: str, *, force_refresh: bool = False) -> list[str]:
    """Fetch all available OpenRouter model ids for this account/key."""
    ck = f"openrouter:{hash(api_key)}"
    if not force_refresh:
        cached = _from_cache(ck)
        if cached is not None:
            return cached

    if not api_key:
        return []

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    last_exc: Exception | None = None
    payload: dict = {}
    for attempt in range(2):
        try:
            async with httpx.AsyncClient(timeout=20.0) as client:
                resp = await client.get(_OPENROUTER_MODELS_URL, headers=headers)
                resp.raise_for_status()
                payload = resp.json()
                break
        except Exception as exc:
            last_exc = exc
            if attempt == 0:
                await asyncio.sleep(0.5)
            continue
    else:
        if last_exc:
            raise last_exc

    data = payload.get("data", [])
    models = sorted(
        {
            str(item.get("id", "")).strip()
            for item in data
            if item.get("id")
        }
    )
    _save_cache(ck, models)
    return models


async def resolve_dynamic_models(
    *,
    provider_id: str,
    service: str,
    fallback: list[str],
    env: dict[str, str] | None = None,
) -> list[str]:
    """Resolve dynamic models for selected provider; fallback on any failure."""
    env = env or {}
    # Incremental rollout: currently OpenRouter dynamic list.
    if provider_id in {"openrouter", "openrouter-rag", "openrouter-vision"}:
        key = env.get("OPENROUTER_API_KEY", "") or os.environ.get("OPENROUTER_API_KEY", "")
        try:
            models = await fetch_openrouter_models(key)
            if models:
                return models
        except Exception as exc:
            log.warning("Dynamic OpenRouter models unavailable, fallback to static list: %s", exc)
            return fallback
    return fallback
