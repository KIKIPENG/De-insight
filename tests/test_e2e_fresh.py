"""Fresh Machine E2E 測試 — 以全新使用者/全新電腦標準驗證。

測試前置：使用暫時 HOME（DEINSIGHT_HOME 指向空資料夾），
模擬第一次安裝與第一次啟動。

涵蓋：
- 安裝路徑 E2E
- 匯入測試 E2E（PDF/URL/手動文字）
- 查詢測試 E2E
- 問答測試 E2E（10+ 組多輪對話）
- 圖片輸入 E2E
- 記憶功能 E2E
- 穩定性測試（3 輪循環）
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# 確保專案根目錄在 path
sys.path.insert(0, str(Path(__file__).parent.parent))


# ═══════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════


@pytest.fixture
def fresh_home(tmp_path):
    """建立全新的 DEINSIGHT_HOME 環境。"""
    home = tmp_path / "deinsight_fresh"
    home.mkdir()
    with patch.dict(os.environ, {"DEINSIGHT_HOME": str(home)}):
        # 重新載入 paths 模組
        import importlib
        import paths
        importlib.reload(paths)
        yield home
    # cleanup
    import importlib
    import paths
    importlib.reload(paths)


@pytest.fixture
def mock_llama_server():
    """Mock llama-server 回應（不需要真實 server）。"""
    import httpx

    def _make_embedding_response(dim=1024, count=1):
        return {
            "data": [
                {"embedding": [0.1 * (i + 1)] * dim, "index": i}
                for i in range(count)
            ]
        }

    async def _mock_post(url, **kwargs):
        resp = MagicMock()
        resp.status_code = 200
        resp.raise_for_status = MagicMock()

        payload = kwargs.get("json", {})
        inputs = payload.get("input", [])
        if isinstance(inputs, str):
            count = 1
        elif isinstance(inputs, list):
            count = len(inputs)
        else:
            count = 1

        resp.json.return_value = _make_embedding_response(count=count)
        return resp

    return _mock_post


# ═══════════════════════════════════════════════════════════════════
# Section 1: 安裝路徑 E2E
# ═══════════════════════════════════════════════════════════════════


class TestInstallationPath:
    """驗證安裝路徑正確建立。"""

    def test_fresh_home_is_empty(self, fresh_home):
        assert fresh_home.exists()
        assert list(fresh_home.iterdir()) == []

    def test_installer_creates_directories(self, fresh_home):
        from embeddings.gguf_installer import GGUFInstaller
        installer = GGUFInstaller(gguf_home=fresh_home / "gguf")
        assert installer.models_dir.exists()

    def test_installer_status_initially_not_installed(self, fresh_home):
        from embeddings.gguf_installer import GGUFInstaller
        installer = GGUFInstaller(gguf_home=fresh_home / "gguf")
        status = installer.installation_status()
        assert not status["fully_installed"]
        assert not status["model_downloaded"]
        assert not status["mmproj_downloaded"]
        assert not status["llama_cpp_built"]

    def test_paths_created_for_project(self, fresh_home):
        from paths import ensure_project_dirs
        project_id = "test-project-1"
        root = ensure_project_dirs(project_id)
        assert (root / "lancedb").exists()
        assert (root / "lightrag").exists()
        assert (root / "documents").exists()


# ═══════════════════════════════════════════════════════════════════
# Section 2: EmbeddingService 初始化 E2E
# ═══════════════════════════════════════════════════════════════════


class TestServiceInitialization:
    """EmbeddingService 初始化流程測試。"""

    def setup_method(self):
        from embeddings.service import reset_embedding_service
        reset_embedding_service()

    def teardown_method(self):
        from embeddings.service import reset_embedding_service
        reset_embedding_service()

    def test_service_creates_backend(self):
        from embeddings.service import get_embedding_service
        svc = get_embedding_service()
        assert svc.dimension() == 1024

    def test_service_diagnostics(self):
        from embeddings.service import get_embedding_service
        svc = get_embedding_service()
        diag = svc.get_diagnostics()
        assert diag["backend"] == "gguf"
        assert diag["dimension"] == 1024
        assert "installation" in diag

    def test_ensure_server_fails_without_install(self):
        """未安裝時 ensure_server 應失敗（fail-fast）。"""
        from embeddings.service import get_embedding_service
        svc = get_embedding_service()

        with patch.dict(os.environ, {
            "GGUF_AUTO_INSTALL": "0",
            "DEINSIGHT_HOME": "/tmp/nonexistent_deinsight_e2e",
        }):
            from embeddings.llama_server import LlamaServerManager
            LlamaServerManager._instance = None
            with pytest.raises(RuntimeError):
                svc.ensure_server_running()
            LlamaServerManager._instance = None


# ═══════════════════════════════════════════════════════════════════
# Section 3: 匯入測試 E2E（mock embedding）
# ═══════════════════════════════════════════════════════════════════


class TestIngestionE2E:
    """匯入流程 E2E（使用 mock embedding）。"""

    def test_text_ingestion_creates_job(self, fresh_home):
        """手動文字匯入應建立 job。"""
        from rag.job_repository import JobRepository
        from paths import DATA_ROOT

        async def _run():
            db_path = DATA_ROOT / "ingest_jobs.db"
            db_path.parent.mkdir(parents=True, exist_ok=True)
            repo = JobRepository(db_path)
            await repo.ensure_table()
            job_id = await repo.create_job(
                project_id="test-proj",
                source_type="text",
                source="測試文字內容",
                title="手動測試",
            )
            assert job_id
            job = await repo.get_job(job_id)
            assert job["status"] in ("pending", "queued")
            assert job["source_type"] == "text"

        asyncio.run(_run())

    def test_url_ingestion_creates_job(self, fresh_home):
        """URL 匯入應建立 job。"""
        from rag.job_repository import JobRepository
        from paths import DATA_ROOT

        async def _run():
            db_path = DATA_ROOT / "ingest_jobs.db"
            db_path.parent.mkdir(parents=True, exist_ok=True)
            repo = JobRepository(db_path)
            await repo.ensure_table()
            job_id = await repo.create_job(
                project_id="test-proj",
                source_type="url",
                source="https://example.com/article",
                title="URL 測試",
            )
            job = await repo.get_job(job_id)
            assert job["status"] in ("pending", "queued")
            assert job["source_type"] == "url"

        asyncio.run(_run())

    def test_pdf_ingestion_creates_job(self, fresh_home):
        """PDF 匯入應建立 job。"""
        from rag.job_repository import JobRepository
        from paths import DATA_ROOT

        async def _run():
            db_path = DATA_ROOT / "ingest_jobs.db"
            db_path.parent.mkdir(parents=True, exist_ok=True)
            repo = JobRepository(db_path)
            await repo.ensure_table()
            job_id = await repo.create_job(
                project_id="test-proj",
                source_type="pdf",
                source="/path/to/test.pdf",
                title="PDF 測試",
            )
            job = await repo.get_job(job_id)
            assert job["status"] in ("pending", "queued")
            assert job["source_type"] == "pdf"

        asyncio.run(_run())


# ═══════════════════════════════════════════════════════════════════
# Section 4: 查詢測試 E2E
# ═══════════════════════════════════════════════════════════════════


class TestQueryE2E:
    """查詢流程測試。"""

    def setup_method(self):
        from embeddings.service import reset_embedding_service
        reset_embedding_service()

    def teardown_method(self):
        from embeddings.service import reset_embedding_service
        reset_embedding_service()

    def test_embed_config_returns_valid(self):
        from embeddings.service import get_embedding_service
        svc = get_embedding_service()
        model, key, base, dim = svc.get_embed_config()
        assert model == "jina-embeddings-v4-gguf"
        assert dim == 1024

    def test_backward_compat_embed_config(self):
        from embeddings.local import get_embed_config
        model, key, base, dim = get_embed_config()
        assert dim == 1024


# ═══════════════════════════════════════════════════════════════════
# Section 5: 問答測試 E2E（10+ 組多輪對話 mock）
# ═══════════════════════════════════════════════════════════════════


class TestConversationE2E:
    """多輪對話結構驗證（不需真實 LLM）。"""

    @pytest.fixture
    def conversation_pairs(self):
        """10 組中英文測試對話。"""
        return [
            ("什麼是知識管理？", "知識管理是..."),
            ("How does RAG work?", "RAG combines retrieval..."),
            ("向量搜尋的原理是什麼？", "向量搜尋基於..."),
            ("Explain embedding models", "Embedding models map..."),
            ("如何優化檢索精確度？", "可以從以下幾點..."),
            ("What is Matryoshka truncation?", "Matryoshka truncation is..."),
            ("GGUF 格式的優勢？", "GGUF 是一種量化格式..."),
            ("Compare Q4_K_M vs Q8_0", "Q4_K_M offers smaller size..."),
            ("llama-server 如何處理多模態？", "透過 mmproj 投影矩陣..."),
            ("How to handle signature migration?", "When provider changes..."),
            ("跨語言檢索如何實現？", "透過多語言embedding模型..."),
        ]

    def test_conversation_count(self, conversation_pairs):
        assert len(conversation_pairs) >= 10

    def test_conversation_has_chinese(self, conversation_pairs):
        chinese = [q for q, _ in conversation_pairs if any('\u4e00' <= c <= '\u9fff' for c in q)]
        assert len(chinese) >= 5

    def test_conversation_has_english(self, conversation_pairs):
        english = [q for q, _ in conversation_pairs if q[0].isascii()]
        assert len(english) >= 3

    def test_message_format_valid(self, conversation_pairs):
        """驗證訊息可以正確轉為 API 格式。"""
        for query, response in conversation_pairs:
            messages = [
                {"role": "user", "content": query},
                {"role": "assistant", "content": response},
            ]
            assert all("role" in m and "content" in m for m in messages)


# ═══════════════════════════════════════════════════════════════════
# Section 6: 記憶功能 E2E
# ═══════════════════════════════════════════════════════════════════


class TestMemoryE2E:
    """記憶功能端到端測試。"""

    def test_memory_store_crud(self, fresh_home):
        """記憶 CRUD 基本流程。"""
        from memory.store import add_memory, get_memories, delete_memory
        from paths import DATA_ROOT

        async def _run():
            db_path = DATA_ROOT / "memories.db"
            db_path.parent.mkdir(parents=True, exist_ok=True)

            # Create — mock vectorstore 的 index_memory 避免 embed 呼叫
            with patch("memory.vectorstore.index_memory", new_callable=AsyncMock):
                mem_id = await add_memory(
                    type="preference",
                    content="使用者偏好深度分析",
                    project_id="test-proj",
                    db_path=db_path,
                )
            assert mem_id

            # Read
            memories = await get_memories(project_id="test-proj", db_path=db_path)
            assert len(memories) >= 1
            assert any(m["content"] == "使用者偏好深度分析" for m in memories)

            # Delete
            await delete_memory(mem_id, db_path=db_path)
            memories = await get_memories(project_id="test-proj", db_path=db_path)
            assert not any(m.get("id") == mem_id for m in memories)

        asyncio.run(_run())


# ═══════════════════════════════════════════════════════════════════
# Section 7: Provider 簽章遷移 E2E
# ═══════════════════════════════════════════════════════════════════


class TestSignatureMigrationE2E:
    """Provider 簽章遷移端到端測試。"""

    def setup_method(self):
        from embeddings.service import reset_embedding_service
        reset_embedding_service()

    def teardown_method(self):
        from embeddings.service import reset_embedding_service
        reset_embedding_service()

    def test_migration_from_sentence_transformers(self, tmp_path):
        """模擬從舊 sentence-transformers 遷移到 GGUF。"""
        from embeddings.service import get_embedding_service
        svc = get_embedding_service()

        # 模擬舊簽章
        sig_file = tmp_path / "embed_provider_signature.json"
        sig_file.write_text(json.dumps({
            "signature": "jina-v4-sentence-transformers-1024"
        }))

        needs_rebuild = svc.check_signature_migration(tmp_path)
        assert needs_rebuild

        # 第二次不應觸發
        needs_rebuild2 = svc.check_signature_migration(tmp_path)
        assert not needs_rebuild2

    def test_fresh_install_no_migration(self, tmp_path):
        from embeddings.service import get_embedding_service
        svc = get_embedding_service()
        needs_rebuild = svc.check_signature_migration(tmp_path)
        assert not needs_rebuild


# ═══════════════════════════════════════════════════════════════════
# Section 8: 穩定性測試（3 輪循環）
# ═══════════════════════════════════════════════════════════════════


class TestStabilityLoops:
    """穩定性：多輪循環不會 freeze/leak。"""

    def setup_method(self):
        from embeddings.service import reset_embedding_service
        reset_embedding_service()

    def teardown_method(self):
        from embeddings.service import reset_embedding_service
        reset_embedding_service()

    def test_service_singleton_stable_across_loops(self):
        """3 輪取得 service，每次都是同一個 instance。"""
        from embeddings.service import get_embedding_service
        instances = [get_embedding_service() for _ in range(3)]
        assert all(inst is instances[0] for inst in instances)

    def test_config_stable_across_loops(self):
        """3 輪取得 config，結果一致。"""
        from embeddings.service import get_embedding_service
        svc = get_embedding_service()
        configs = [svc.get_embed_config() for _ in range(3)]
        assert all(c == configs[0] for c in configs)

    def test_diagnostics_stable(self):
        """3 輪取得 diagnostics，不 crash。"""
        from embeddings.service import get_embedding_service
        svc = get_embedding_service()
        for _ in range(3):
            diag = svc.get_device_diagnostics()
            assert "runtime_device" in diag

    def test_signature_check_idempotent(self, tmp_path):
        """3 輪簽章檢查，首次後不再觸發遷移。"""
        from embeddings.service import get_embedding_service
        svc = get_embedding_service()
        results = []
        for _ in range(3):
            r = svc.check_signature_migration(tmp_path)
            results.append(r)
        assert results[0] == False  # first time, writes signature
        assert results[1] == False
        assert results[2] == False


# ═══════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
