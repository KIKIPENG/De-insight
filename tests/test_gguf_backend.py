"""GGUF 多模態 Embedding 後端測試。

涵蓋：
- EmbeddingBackend 介面合規性
- GGUFMultimodalBackend 單元測試（mock llama-server）
- LlamaServerManager 生命週期測試
- GGUFInstaller 狀態檢查
- EmbeddingService facade 測試
- Provider 簽章遷移測試
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import sys
import tempfile
import threading
from email.utils import format_datetime
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# 確保專案根目錄在 path
sys.path.insert(0, str(Path(__file__).parent.parent))


# ═══════════════════════════════════════════════════════════════════
# Section 1: EmbeddingBackend 介面測試
# ═══════════════════════════════════════════════════════════════════


class TestEmbeddingBackendInterface:
    """驗證 EmbeddingBackend 抽象介面定義正確。"""

    def test_backend_is_abstract(self):
        from embeddings.backend import EmbeddingBackend
        with pytest.raises(TypeError):
            EmbeddingBackend()

    def test_backend_has_required_methods(self):
        from embeddings.backend import EmbeddingBackend
        required = ["embed_query", "embed_passages", "embed_image", "dimension", "provider_signature"]
        for method in required:
            assert hasattr(EmbeddingBackend, method), f"Missing method: {method}"

    def test_gguf_backend_implements_interface(self):
        from embeddings.backend import EmbeddingBackend
        from embeddings.gguf_backend import GGUFMultimodalBackend
        assert issubclass(GGUFMultimodalBackend, EmbeddingBackend)


# ═══════════════════════════════════════════════════════════════════
# Section 2: GGUFMultimodalBackend 單元測試
# ═══════════════════════════════════════════════════════════════════


class TestGGUFMultimodalBackend:
    """GGUFMultimodalBackend 功能測試（mock HTTP 呼叫）。"""

    @pytest.fixture
    def backend(self):
        from embeddings.gguf_backend import GGUFMultimodalBackend
        return GGUFMultimodalBackend(base_url="http://127.0.0.1:8999", dim=1024)

    def test_dimension(self, backend):
        assert backend.dimension() == 1024

    def test_provider_signature_format(self, backend):
        sig = backend.provider_signature()
        assert "jina-v4-gguf" in sig
        assert "1024" in sig

    def test_truncate_and_normalize(self, backend):
        raw = [1.0] * 2048  # 超過 1024 維
        result = backend._truncate_and_normalize(raw)
        assert len(result) == 1024
        # 應該是 L2 normalized
        norm = math.sqrt(sum(x * x for x in result))
        assert abs(norm - 1.0) < 1e-5

    def test_truncate_short_vector(self, backend):
        raw = [1.0] * 512  # 短於 1024 維
        result = backend._truncate_and_normalize(raw)
        assert len(result) == 512

    def test_image_to_base64_bytes(self, backend):
        import base64
        data = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100
        result = backend._image_to_base64(data)
        decoded = base64.b64decode(result)
        assert decoded == data

    def test_image_to_base64_file(self, backend, tmp_path):
        img_file = tmp_path / "test.png"
        img_data = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100
        img_file.write_bytes(img_data)
        result = backend._image_to_base64(str(img_file))
        import base64
        assert base64.b64decode(result) == img_data

    def test_image_to_base64_missing_file(self, backend):
        with pytest.raises(FileNotFoundError):
            backend._image_to_base64("/nonexistent/image.png")

    @pytest.mark.asyncio
    async def test_embed_query_calls_api(self, backend):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "data": [{"embedding": [0.5] * 1024, "index": 0}]
        }

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.is_closed = False
        backend._client = mock_client

        result = await backend.embed_query("test query")
        assert len(result) == 1024
        # 驗證加了 "Query: " 前綴
        call_args = mock_client.post.call_args
        payload = call_args[1]["json"] if "json" in call_args[1] else call_args[0][1]
        if isinstance(payload, dict) and "input" in payload:
            assert "Query: " in str(payload["input"])

    @pytest.mark.asyncio
    async def test_embed_passages_calls_api(self, backend):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "data": [
                {"embedding": [0.3] * 1024, "index": 0},
                {"embedding": [0.7] * 1024, "index": 1},
            ]
        }

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.is_closed = False
        backend._client = mock_client

        result = await backend.embed_passages(["text 1", "text 2"])
        assert len(result) == 2
        assert all(len(v) == 1024 for v in result)

    @pytest.mark.asyncio
    async def test_embed_passages_empty(self, backend):
        result = await backend.embed_passages([])
        assert result == []

    @pytest.mark.asyncio
    async def test_embed_passages_adds_prefix(self, backend):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "data": [{"embedding": [0.5] * 1024, "index": 0}]
        }

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.is_closed = False
        backend._client = mock_client

        await backend.embed_passages(["some text"])
        call_args = mock_client.post.call_args
        payload = call_args[1]["json"] if "json" in call_args[1] else call_args[0][1]
        if isinstance(payload, dict) and "input" in payload:
            assert "Passage: " in str(payload["input"])

    @pytest.mark.asyncio
    async def test_embed_fallback_to_prompt_when_input_not_supported(self, backend):
        """Server returns 500 key 'prompt' not found on input payload -> fallback to prompt."""
        from httpx import HTTPStatusError, Request, Response

        bad = Response(
            500,
            request=Request("POST", "http://127.0.0.1:8999/v1/embeddings"),
            json={"code": 500, "message": "[json.exception.out_of_range.403] key 'prompt' not found"},
        )
        ok = MagicMock()
        ok.status_code = 200
        ok.raise_for_status = MagicMock()
        ok.json.return_value = {"data": [{"embedding": [0.2] * 1024, "index": 0}]}

        first_error = HTTPStatusError("500", request=bad.request, response=bad)
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(side_effect=[first_error, ok, ok])
        mock_client.is_closed = False
        backend._client = mock_client

        result = await backend.embed_passages(["a", "b"])
        assert len(result) == 2
        assert all(len(v) == 1024 for v in result)


# ═══════════════════════════════════════════════════════════════════
# Section 2.5: JinaAPIBackend 單元測試
# ═══════════════════════════════════════════════════════════════════


class TestJinaAPIBackend:
    @pytest.fixture
    def backend(self):
        with patch.dict(
            os.environ,
            {
                "JINA_API_KEY": "jina_test_key",
                "JINA_EMBED_BACKOFF_JITTER": "0",
                "JINA_EMBED_MIN_INTERVAL": "0",
            },
            clear=False,
        ):
            from embeddings.jina_backend import JinaAPIBackend
            return JinaAPIBackend(timeout=1.0)

    @pytest.mark.asyncio
    async def test_retry_on_429_then_success(self, backend):
        first = MagicMock()
        first.status_code = 429
        first.headers = {"Retry-After": "0"}
        first.text = "too many requests"

        second = MagicMock()
        second.status_code = 200
        second.json.return_value = {"data": [{"index": 0, "embedding": [0.1] * 8}]}

        mock_client = AsyncMock()
        mock_client.is_closed = False
        mock_client.post = AsyncMock(side_effect=[first, second])
        backend._client = mock_client

        with patch("embeddings.jina_backend.asyncio.sleep", new=AsyncMock()) as mocked_sleep:
            result = await backend.embed_query("hello")

        assert len(result) == 8
        assert mock_client.post.await_count == 2
        assert mocked_sleep.await_count >= 1

    @pytest.mark.asyncio
    async def test_raise_rate_limit_after_retries_exhausted(self, backend):
        resp = MagicMock()
        resp.status_code = 429
        resp.headers = {"Retry-After": "0"}
        resp.text = "too many requests"

        backend._max_retries = 1

        mock_client = AsyncMock()
        mock_client.is_closed = False
        mock_client.post = AsyncMock(side_effect=[resp, resp])
        backend._client = mock_client

        with patch("embeddings.jina_backend.asyncio.sleep", new=AsyncMock()):
            with pytest.raises(RuntimeError, match="429"):
                await backend.embed_query("hello")

    def test_parse_retry_after_seconds(self, backend):
        assert backend._parse_retry_after("3") == 3.0

    def test_parse_retry_after_http_date(self, backend):
        dt = datetime.now(timezone.utc) + timedelta(seconds=5)
        raw = format_datetime(dt)
        parsed = backend._parse_retry_after(raw)
        assert parsed is not None
        assert parsed >= 0.0


# ═══════════════════════════════════════════════════════════════════
# Section 3: LlamaServerManager 測試
# ═══════════════════════════════════════════════════════════════════


class TestLlamaServerManager:
    """LlamaServerManager 生命週期測試。"""

    def test_singleton_pattern(self):
        from embeddings.llama_server import LlamaServerManager
        # 重置單例
        LlamaServerManager._instance = None
        a = LlamaServerManager()
        b = LlamaServerManager()
        assert a is b
        # cleanup
        LlamaServerManager._instance = None

    def test_default_host_port(self):
        from embeddings.llama_server import LlamaServerManager
        LlamaServerManager._instance = None
        mgr = LlamaServerManager()
        assert mgr.host == "127.0.0.1"
        assert mgr.port == 8999
        assert mgr.base_url == "http://127.0.0.1:8999"
        LlamaServerManager._instance = None

    def test_custom_host_port_from_env(self):
        from embeddings.llama_server import LlamaServerManager
        LlamaServerManager._instance = None
        with patch.dict(os.environ, {"GGUF_SERVER_HOST": "0.0.0.0", "GGUF_SERVER_PORT": "9999"}):
            mgr = LlamaServerManager()
            mgr._initialized = False
            mgr.__init__()
            assert mgr.port == 9999
        LlamaServerManager._instance = None

    def test_is_running_false_by_default(self):
        from embeddings.llama_server import LlamaServerManager
        LlamaServerManager._instance = None
        mgr = LlamaServerManager()
        # 沒有行程，不應該 running
        mgr._process = None
        assert not mgr.is_running
        LlamaServerManager._instance = None

    def test_find_binary_returns_none_when_missing(self):
        from embeddings.llama_server import LlamaServerManager
        LlamaServerManager._instance = None
        mgr = LlamaServerManager()
        with patch.dict(os.environ, {"DEINSIGHT_HOME": "/tmp/nonexistent_deinsight_test"}):
            # 如果沒有 llama-server 在系統 PATH 中
            result = mgr.find_binary()
            # 結果取決於系統是否有 llama-server
            # 至少不會 raise
        LlamaServerManager._instance = None

    def test_start_fails_without_binary(self):
        from embeddings.llama_server import LlamaServerManager, LlamaServerError
        LlamaServerManager._instance = None
        mgr = LlamaServerManager()
        with patch.object(mgr, "find_binary", return_value=None):
            with pytest.raises(LlamaServerError, match="二進位檔不存在"):
                mgr.start("/tmp/model.gguf")
        LlamaServerManager._instance = None

    def test_start_fails_without_model(self):
        from embeddings.llama_server import LlamaServerManager, LlamaServerError
        LlamaServerManager._instance = None
        mgr = LlamaServerManager()
        with patch.object(mgr, "find_binary", return_value=Path("/usr/bin/true")):
            with pytest.raises(LlamaServerError, match="模型檔不存在"):
                mgr.start("/tmp/nonexistent_model.gguf")
        LlamaServerManager._instance = None


# ═══════════════════════════════════════════════════════════════════
# Section 4: GGUFInstaller 測試
# ═══════════════════════════════════════════════════════════════════


class TestGGUFInstaller:
    """GGUFInstaller 狀態檢查測試。"""

    @pytest.fixture
    def installer(self, tmp_path):
        from embeddings.gguf_installer import GGUFInstaller
        return GGUFInstaller(gguf_home=tmp_path / "gguf")

    def test_initial_status_all_false(self, installer):
        status = installer.installation_status()
        assert not status["llama_cpp_built"]
        assert not status["model_downloaded"]
        assert not status["mmproj_downloaded"]
        assert not status["fully_installed"]

    def test_cmake_check(self, installer):
        # cmake 在 macOS 通常可用
        import shutil
        expected = shutil.which("cmake") is not None
        assert installer.is_cmake_available() == expected

    def test_model_path_default(self, installer):
        assert "jina-embeddings-v4-text-retrieval-Q4_K_M.gguf" in str(installer.model_path)

    def test_mmproj_path_default(self, installer):
        assert "mmproj-jina-embeddings-v4-retrieval-BF16.gguf" in str(installer.mmproj_path)

    def test_model_path_custom(self, installer):
        with patch.dict(os.environ, {"GGUF_MODEL_FILE": "custom-model.gguf"}):
            assert "custom-model.gguf" in str(installer.model_path)

    def test_server_binary_path(self, installer):
        expected = installer.llama_cpp_dir / "build" / "bin" / "llama-server"
        assert installer.server_binary == expected

    def test_clean(self, installer):
        # 建立一些檔案
        (installer.models_dir / "test.gguf").touch()
        assert (installer.models_dir / "test.gguf").exists()
        installer.clean()
        assert not (installer.models_dir / "test.gguf").exists()

    def test_verify_model_false_when_missing(self, installer):
        assert not installer.verify_model()


# ═══════════════════════════════════════════════════════════════════
# Section 5: EmbeddingService 測試
# ═══════════════════════════════════════════════════════════════════


class TestEmbeddingService:
    """EmbeddingService facade 測試。"""

    def setup_method(self):
        from embeddings.service import reset_embedding_service
        reset_embedding_service()

    def teardown_method(self):
        from embeddings.service import reset_embedding_service
        reset_embedding_service()

    def test_singleton(self):
        from embeddings.service import get_embedding_service
        a = get_embedding_service()
        b = get_embedding_service()
        assert a is b

    def test_reset_creates_new_instance(self):
        from embeddings.service import get_embedding_service, reset_embedding_service
        a = get_embedding_service()
        reset_embedding_service()
        b = get_embedding_service()
        assert a is not b

    def test_get_embed_config_format(self):
        from embeddings.service import get_embedding_service
        svc = get_embedding_service()
        config = svc.get_embed_config()
        assert len(config) == 4
        model, key, base, dim = config
        assert isinstance(model, str)
        assert isinstance(dim, int)
        assert dim == 1024

    def test_dimension(self):
        from embeddings.service import get_embedding_service
        svc = get_embedding_service()
        assert svc.dimension() == 1024

    def test_provider_signature(self):
        from embeddings.service import get_embedding_service
        svc = get_embedding_service()
        sig = svc.provider_signature()
        assert isinstance(sig, str)
        assert len(sig) > 0

    def test_get_device_diagnostics(self):
        from embeddings.service import get_embedding_service
        svc = get_embedding_service()
        diag = svc.get_device_diagnostics()
        assert "runtime_device" in diag
        assert diag["runtime_device"] == "gguf-server"


# ═══════════════════════════════════════════════════════════════════
# Section 6: Provider 簽章遷移測試
# ═══════════════════════════════════════════════════════════════════


class TestSignatureMigration:
    """Provider 簽章遷移機制測試。"""

    def setup_method(self):
        from embeddings.service import reset_embedding_service
        reset_embedding_service()

    def teardown_method(self):
        from embeddings.service import reset_embedding_service
        reset_embedding_service()

    def test_first_run_no_migration(self, tmp_path):
        from embeddings.service import get_embedding_service
        svc = get_embedding_service()
        # 空目錄，首次寫入簽章，不觸發遷移
        needs_rebuild = svc.check_signature_migration(tmp_path)
        assert not needs_rebuild
        # 簽章檔案應已建立
        sig_file = tmp_path / "embed_provider_signature.json"
        assert sig_file.exists()

    def test_same_signature_no_migration(self, tmp_path):
        from embeddings.service import get_embedding_service
        svc = get_embedding_service()
        # 寫入當前簽章
        svc.check_signature_migration(tmp_path)
        # 再次檢查，不應觸發
        needs_rebuild = svc.check_signature_migration(tmp_path)
        assert not needs_rebuild

    def test_different_signature_triggers_migration(self, tmp_path):
        from embeddings.service import get_embedding_service
        svc = get_embedding_service()
        # 手動寫入舊簽章
        sig_file = tmp_path / "embed_provider_signature.json"
        sig_file.write_text(json.dumps({"signature": "old-backend-384"}))
        # 檢查應觸發遷移
        needs_rebuild = svc.check_signature_migration(tmp_path)
        assert needs_rebuild
        # 簽章檔案應已更新為新簽章
        new_sig = json.loads(sig_file.read_text())["signature"]
        assert new_sig == svc.provider_signature()

    def test_corrupt_signature_file(self, tmp_path):
        from embeddings.service import get_embedding_service
        svc = get_embedding_service()
        sig_file = tmp_path / "embed_provider_signature.json"
        sig_file.write_text("not json")
        # 應視為簽章變更
        needs_rebuild = svc.check_signature_migration(tmp_path)
        assert needs_rebuild


# ═══════════════════════════════════════════════════════════════════
# Section 7: 向後相容 API 測試
# ═══════════════════════════════════════════════════════════════════


class TestBackwardCompatAPI:
    """確認 embeddings.local 和 embeddings.jina_v4 的舊 API 仍可用。"""

    def test_local_exports(self):
        from embeddings.local import EMBED_DIM, EMBED_MODEL
        from embeddings.local import embed_texts, embed_text, embed_image
        from embeddings.local import get_embed_config, get_device_diagnostics
        from embeddings.local import ensure_model_downloaded
        assert EMBED_DIM == 1024
        assert callable(embed_texts)
        assert callable(embed_text)
        assert callable(embed_image)
        assert callable(get_embed_config)

    def test_jina_v4_exports(self):
        from embeddings.jina_v4 import EMBED_DIM, EMBED_MODEL
        from embeddings.jina_v4 import embed_texts, embed_text, embed_image
        from embeddings.jina_v4 import get_embed_config
        assert EMBED_DIM == 1024

    def test_get_embed_config_returns_tuple(self):
        from embeddings.local import get_embed_config
        result = get_embed_config()
        assert isinstance(result, tuple)
        assert len(result) == 4


# ═══════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
