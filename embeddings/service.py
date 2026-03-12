"""EmbeddingService — 統一的 embedding facade。"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
from pathlib import Path
from typing import Union

from config.service import get_config_service
from embeddings.backend import EmbeddingBackend

log = logging.getLogger(__name__)

# ── 全域單例 ────────────────────────────────────────────────────

_service: EmbeddingService | None = None
_service_lock = threading.Lock()

EMBED_DIM = 1024
EMBED_MODEL = "nvidia/llama-nemotron-embed-vl-1b-v2:free"

_SIGNATURE_FILE = "embed_provider_signature.json"


def _default_provider() -> str:
    return "openrouter"


def _resolve_model(raw_model: str) -> str:
    model = (raw_model or "").strip()
    if not model or "/" not in model:
        return EMBED_MODEL
    return model


def get_embedding_service() -> EmbeddingService:
    """取得全域 EmbeddingService 單例。"""
    global _service
    if _service is not None:
        return _service
    with _service_lock:
        if _service is not None:
            return _service
        _service = EmbeddingService()
        return _service


def reset_embedding_service() -> None:
    """重置 service（設定變更後使用）。"""
    global _service
    with _service_lock:
        if _service is not None:
            _service._backend = None
        _service = None


class EmbeddingService:
    """Embedding 外觀層，封裝 backend 生命週期。"""

    def __init__(self) -> None:
        self._backend: EmbeddingBackend | None = None
        self._init_lock = threading.Lock()

    @property
    def backend(self) -> EmbeddingBackend:
        if self._backend is None:
            self._init_backend()
        return self._backend

    def _init_backend(self) -> None:
        """初始化 embedding 後端（lazy, thread-safe）。"""
        with self._init_lock:
            if self._backend is not None:
                return

            cfg = get_config_service()
            env = cfg.snapshot(include_process=True)

            embed_dim = int(
                env.get("EMBED_DIM", "")
                or os.environ.get("EMBED_DIM")
                or str(EMBED_DIM)
            )

            api_key = (
                env.get("EMBED_API_KEY", "")
                or env.get("OPENROUTER_API_KEY", "")
                or os.environ.get("EMBED_API_KEY", "")
                or os.environ.get("OPENROUTER_API_KEY", "")
            )
            raw_model = env.get("EMBED_MODEL", "") or os.environ.get("EMBED_MODEL", "")
            model = _resolve_model(raw_model)
            base_url = (
                env.get("EMBED_API_BASE", "")
                or env.get("OPENROUTER_API_BASE", "")
                or os.environ.get("EMBED_API_BASE", "")
                or os.environ.get("OPENROUTER_API_BASE", "")
                or "https://openrouter.ai/api/v1"
            )
            if api_key:
                os.environ["OPENROUTER_API_KEY"] = api_key
                os.environ["EMBED_API_KEY"] = api_key
            os.environ["EMBED_MODEL"] = model
            os.environ["EMBED_API_BASE"] = base_url

            from embeddings.openrouter_backend import OpenRouterEmbeddingBackend

            self._backend = OpenRouterEmbeddingBackend(
                api_key=api_key,
                model=model,
                dim=embed_dim,
                base_url=base_url,
            )
            self._provider_type = "openrouter"
            log.info("Using OpenRouter embedding backend")

    def ensure_server_running(self) -> None:
        """確保 embedding 後端就緒。"""
        if self._backend is None:
            self._init_backend()

    # ── 公開 API（與舊 embeddings.local 相容的介面）───────────

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """批次 passage embedding。"""
        if not texts:
            return []
        await asyncio.to_thread(self.ensure_server_running)
        return await self.backend.embed_passages(texts)

    async def embed_text(self, text: str) -> list[float]:
        """單筆 query embedding。"""
        await asyncio.to_thread(self.ensure_server_running)
        return await self.backend.embed_query(text)

    async def embed_image(self, image: Union[str, Path, bytes]) -> list[float]:
        """圖片 embedding。"""
        await asyncio.to_thread(self.ensure_server_running)
        return await self.backend.embed_image(image)

    def dimension(self) -> int:
        return self.backend.dimension()

    def provider_signature(self) -> str:
        return self.backend.provider_signature()

    def get_embed_config(self) -> tuple[str, str, str, int]:
        """回傳 (model, key, base, dim)。相容舊 API。"""
        cfg = get_config_service().snapshot(include_process=True)
        model = _resolve_model(cfg.get("EMBED_MODEL", ""))
        key = cfg.get("EMBED_API_KEY", "") or cfg.get("OPENROUTER_API_KEY", "")
        base = cfg.get("EMBED_API_BASE", "") or cfg.get("OPENROUTER_API_BASE", "") or "https://openrouter.ai/api/v1"
        return model, key, base, self.dimension()

    # ── Provider 簽章遷移 ───────────────────────────────────────

    def check_signature_migration(self, data_dir: Path) -> bool:
        """檢查簽章是否變更。回傳 True 表示需要重建索引。"""
        sig_file = data_dir / _SIGNATURE_FILE
        current_sig = self.provider_signature()

        if not sig_file.exists():
            # 首次：寫入簽章，不觸發重建
            self._write_signature(sig_file, current_sig)
            return False

        try:
            stored = json.loads(sig_file.read_text(encoding="utf-8"))
            old_sig = stored.get("signature", "")
        except Exception:
            old_sig = ""

        if old_sig == current_sig:
            return False

        log.warning(
            "Embedding provider 簽章變更: '%s' → '%s'，將觸發索引重建",
            old_sig, current_sig,
        )
        self._write_signature(sig_file, current_sig)
        return True

    @staticmethod
    def _write_signature(sig_file: Path, signature: str) -> None:
        sig_file.parent.mkdir(parents=True, exist_ok=True)
        sig_file.write_text(
            json.dumps({"signature": signature}, indent=2),
            encoding="utf-8",
        )

    # ── 診斷 ────────────────────────────────────────────────────

    def get_diagnostics(self) -> dict:
        """回傳 embedding 環境診斷資訊。"""
        model, _, base, dim = self.get_embed_config()

        return {
            "backend": "openrouter",
            "model": model,
            "dimension": dim,
            "provider_signature": self.provider_signature(),
            "server_running": True,
            "server_healthy": True,
            "server_url": base,
            "installation": {"remote_api": True},
        }

    def get_device_diagnostics(self) -> dict:
        """回傳相容舊 API 的診斷（不載入模型）。"""
        return {
            "runtime_device": "remote-api",
            "reason": "openrouter_backend",
            "installed": True,
            "cmake": False,
            "model_ready": True,
        }
