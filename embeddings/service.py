"""EmbeddingService — 統一的 embedding facade。

所有 embedding 使用方（rag/knowledge_graph, memory/vectorstore, rag/image_store）
都透過此 service 取得 embedding，不再直接依賴 embeddings.local。

功能：
- 管理 GGUFMultimodalBackend 生命週期
- 負責 llama-server 自動啟動
- Provider 簽章比對與索引重建觸發
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
from pathlib import Path
from typing import Union

from embeddings.backend import EmbeddingBackend

log = logging.getLogger(__name__)

# ── 全域單例 ────────────────────────────────────────────────────

_service: EmbeddingService | None = None
_service_lock = threading.Lock()

EMBED_DIM = 1024
EMBED_MODEL = "jina-embeddings-v4-gguf"

_SIGNATURE_FILE = "embed_provider_signature.json"


def _enforce_local_only() -> None:
    """Phase A: 強制 GGUF 本地 embedding，禁止任何雲端 fallback。

    偵測 remote embedding 設定即啟動失敗。
    """
    if os.environ.get("EMBEDDING_LOCAL_ONLY", "1") != "1":
        return
    provider = os.environ.get("EMBED_PROVIDER", "gguf").lower()
    if provider not in ("gguf", "local", ""):
        raise RuntimeError(
            f"EMBEDDING_LOCAL_ONLY=1 但偵測到 remote embedding provider: '{provider}'。"
            " Phase A 只允許 GGUF 本地 embedding。"
        )


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
            _service._server_started = False
        _service = None


class EmbeddingService:
    """Embedding 外觀層，封裝 backend 生命週期。"""

    def __init__(self) -> None:
        self._backend: EmbeddingBackend | None = None
        self._server_started = False
        self._init_lock = threading.Lock()

    @property
    def backend(self) -> EmbeddingBackend:
        if self._backend is None:
            self._init_backend()
        return self._backend

    def _init_backend(self) -> None:
        """初始化 GGUF 後端（lazy, thread-safe）。"""
        _enforce_local_only()
        with self._init_lock:
            if self._backend is not None:
                return
            from embeddings.gguf_backend import GGUFMultimodalBackend
            from embeddings.llama_server import LlamaServerManager

            mgr = LlamaServerManager()
            self._backend = GGUFMultimodalBackend(
                base_url=mgr.base_url,
                dim=int(os.environ.get("GGUF_EMBED_DIM", str(EMBED_DIM))),
            )

    def ensure_server_running(self) -> None:
        """確保 llama-server 在跑。Fail-fast，不回退。"""
        if self._server_started:
            return

        with self._init_lock:
            if self._server_started:
                return

            from embeddings.llama_server import LlamaServerManager
            from embeddings.gguf_installer import GGUFInstaller

            mgr = LlamaServerManager()

            # 如果已經在跑（可能是上次的），檢查健康
            if mgr.is_running and mgr.health_check():
                self._server_started = True
                log.info("llama-server already healthy")
                return

            # 確保已安裝
            installer = GGUFInstaller()
            if not installer.is_fully_installed():
                auto_install = os.environ.get("GGUF_AUTO_INSTALL", "1") == "1"
                if not auto_install:
                    raise RuntimeError(
                        "GGUF 環境未安裝且 GGUF_AUTO_INSTALL=0。"
                        " 請執行 GGUFInstaller().install() 或設定 GGUF_AUTO_INSTALL=1。"
                    )
                log.info("Auto-installing GGUF environment...")
                installer.install()

            # 啟動 server
            mgr.start(
                model_path=installer.model_path,
                mmproj_path=installer.mmproj_path,
            )
            self._server_started = True

    # ── 公開 API（與舊 embeddings.local 相容的介面）───────────

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """批次 passage embedding。"""
        if not texts:
            return []
        self.ensure_server_running()
        return await self.backend.embed_passages(texts)

    async def embed_text(self, text: str) -> list[float]:
        """單筆 query embedding。"""
        self.ensure_server_running()
        return await self.backend.embed_query(text)

    async def embed_image(self, image: Union[str, Path, bytes]) -> list[float]:
        """圖片 embedding。"""
        self.ensure_server_running()
        return await self.backend.embed_image(image)

    def dimension(self) -> int:
        return self.backend.dimension()

    def provider_signature(self) -> str:
        return self.backend.provider_signature()

    def get_embed_config(self) -> tuple[str, str, str, int]:
        """回傳 (model, key, base, dim)。相容舊 API。"""
        return EMBED_MODEL, "local", "", self.dimension()

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
        from embeddings.llama_server import LlamaServerManager
        from embeddings.gguf_installer import GGUFInstaller

        mgr = LlamaServerManager()
        installer = GGUFInstaller()

        return {
            "backend": "gguf",
            "model": EMBED_MODEL,
            "dimension": self.dimension(),
            "provider_signature": self.provider_signature(),
            "server_running": mgr.is_running,
            "server_healthy": mgr.health_check() if mgr.is_running else False,
            "server_url": mgr.base_url,
            "installation": installer.installation_status(),
        }

    def get_device_diagnostics(self) -> dict:
        """回傳相容舊 API 的診斷（不載入模型）。"""
        from embeddings.gguf_installer import GGUFInstaller
        installer = GGUFInstaller()
        status = installer.installation_status()
        return {
            "runtime_device": "gguf-server",
            "reason": "gguf_backend",
            "installed": status["fully_installed"],
            "cmake": status["cmake"],
            "model_ready": status["model_downloaded"],
        }
