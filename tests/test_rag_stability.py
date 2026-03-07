"""RAG 檢索穩定性測試 — 覆蓋 v0.5a→v0.6 回歸修復。"""

import asyncio
import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

# Project root
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

os.environ.setdefault("DEINSIGHT_HOME", tempfile.mkdtemp())
os.environ.setdefault("DEINSIGHT_DATA_VERSION", "test_v0.6")


def test_embed_default_is_local():
    """缺省（env 未設）時 embedding 回落 local/512。"""
    with patch("settings.load_env", return_value={}):
        from rag.knowledge_graph import _is_local_embed, _get_embed_config

        assert _is_local_embed() is True, "缺省應為 local"
        model, key, base, dim = _get_embed_config()
        assert dim == 512, f"預設維度應為 512，實際 {dim}"
        assert model == "jina-clip-v1"


def test_embed_explicit_local():
    """EMBED_PROVIDER=local 時應為 local。"""
    env = {"EMBED_PROVIDER": "local", "EMBED_MODE": "local"}
    with patch("settings.load_env", return_value=env):
        from rag.knowledge_graph import _is_local_embed, _get_embed_config

        assert _is_local_embed() is True
        _, _, _, dim = _get_embed_config()
        assert dim == 512


def test_embed_api_provider():
    """EMBED_PROVIDER=jina 時應走 API。"""
    env = {
        "EMBED_PROVIDER": "jina",
        "EMBED_MODEL": "jina-embeddings-v3",
        "EMBED_DIM": "1024",
        "EMBED_API_KEY": "test-key",
        "EMBED_API_BASE": "https://api.jina.ai/v1",
    }
    import rag.knowledge_graph as kg
    orig = kg.load_env
    kg.load_env = lambda: env
    try:
        assert kg._is_local_embed() is False
        model, key, base, dim = kg._get_embed_config()
        assert dim == 1024
        assert model == "jina-embeddings-v3"
    finally:
        kg.load_env = orig


def test_embed_backward_compat_mode():
    """EMBED_MODE=local (without PROVIDER) 仍應為 local。"""
    env = {"EMBED_MODE": "local"}
    with patch("settings.load_env", return_value=env):
        from rag.knowledge_graph import _is_local_embed

        assert _is_local_embed() is True


def test_no_context_detection():
    """_is_no_context_result 應正確偵測 no-context 訊息。"""
    from rag.knowledge_graph import _is_no_context_result

    assert _is_no_context_result("") is True
    assert _is_no_context_result(None) is True
    assert _is_no_context_result(
        "Sorry, I'm not able to provide an answer to that question.[no-context]"
    ) is True
    assert _is_no_context_result(
        "No relevant document chunks found"
    ) is True
    assert _is_no_context_result(
        "[來源: 王志宏]\n這是一段有效內容"
    ) is False


def test_has_knowledge_empty():
    """vdb_chunks 空或不存在時 has_knowledge 應回傳 False。"""
    with tempfile.TemporaryDirectory() as td:
        wd = Path(td) / "lightrag"
        wd.mkdir()

        # No vdb file
        from rag.knowledge_graph import has_knowledge
        with patch("rag.knowledge_graph.project_root", return_value=Path(td)):
            assert has_knowledge(project_id="test") is False

        # Empty vdb
        vdb = wd / "vdb_chunks.json"
        vdb.write_text(json.dumps({"data": [], "embedding_dim": 512}))
        with patch("rag.knowledge_graph.project_root", return_value=Path(td)):
            assert has_knowledge(project_id="test") is False

        # Non-empty vdb
        vdb.write_text(json.dumps({"data": [{"__vector__": [0.1] * 512}], "embedding_dim": 512}))
        with patch("rag.knowledge_graph.project_root", return_value=Path(td)):
            assert has_knowledge(project_id="test") is True


def test_clean_rag_chunk():
    """_clean_rag_chunk 應移除 JSON 包裝。"""
    from rag.knowledge_graph import _clean_rag_chunk

    raw = '```json\n{"reference_id": "1", "content": "王志宏是一位策展人"}\n```'
    cleaned = _clean_rag_chunk(raw)
    assert "reference_id" not in cleaned
    assert "王志宏" in cleaned


def test_extract_sources_no_context():
    """no-context 結果不應產生任何 source。"""
    from rag.knowledge_graph import _extract_sources

    sources = _extract_sources(
        "Sorry, I'm not able to provide an answer.[no-context]"
    )
    assert sources == []

    sources = _extract_sources("")
    assert sources == []


def test_diagnose_healthy():
    """健康的知識庫應回傳 healthy=True。"""
    with tempfile.TemporaryDirectory() as td:
        wd = Path(td) / "lightrag"
        wd.mkdir(parents=True)

        # No docs
        (wd / "kv_store_full_docs.json").write_text("{}")
        (wd / "vdb_chunks.json").write_text(json.dumps({"data": []}))

        with patch("rag.repair._lightrag_dir", return_value=wd):
            from rag.repair import diagnose
            result = diagnose("test")
            assert result["healthy"] is True


def test_diagnose_unhealthy():
    """docs > 0 但 vdb 空應回傳 unhealthy。"""
    with tempfile.TemporaryDirectory() as td:
        wd = Path(td) / "lightrag"
        wd.mkdir(parents=True)

        (wd / "kv_store_full_docs.json").write_text(json.dumps({"doc1": {"content": "test"}}))
        (wd / "vdb_chunks.json").write_text(json.dumps({"data": []}))
        (wd / "kv_store_doc_status.json").write_text(
            json.dumps({"doc1": {"status": "processing"}})
        )

        with patch("rag.repair._lightrag_dir", return_value=wd):
            from rag.repair import diagnose
            result = diagnose("test")
            assert result["healthy"] is False
            assert len(result["issues"]) >= 1


def test_image_guard():
    """有圖片資料時不允許從 local 切到 API。"""
    from rag.repair import any_project_has_images
    # When no projects exist
    with tempfile.TemporaryDirectory() as td:
        with patch("paths.PROJECTS_DIR", Path(td) / "projects"):
            assert any_project_has_images() is False


def test_vectorstore_local_mode_default():
    """vectorstore._is_local_mode 缺省應為 True。"""
    with patch("settings.load_env", return_value={}):
        from memory.vectorstore import _is_local_mode
        assert _is_local_mode() is True


if __name__ == "__main__":
    passed = 0
    failed = 0
    for name, func in list(globals().items()):
        if name.startswith("test_") and callable(func):
            try:
                func()
                print(f"  PASS  {name}")
                passed += 1
            except Exception as e:
                print(f"  FAIL  {name}: {e}")
                failed += 1

    print(f"\n{'='*50}")
    print(f"Results: {passed} passed, {failed} failed")
    if failed > 0:
        sys.exit(1)
