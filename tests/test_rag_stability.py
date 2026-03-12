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


def test_embed_config_always_v4():
    """v0.8: _get_embed_config 統一回傳 OpenRouter embedding / 1024。"""
    from rag.knowledge_graph import _get_embed_config

    model, key, base, dim = _get_embed_config()
    assert dim == 1024, f"預設維度應為 1024，實際 {dim}"
    assert model == "nvidia/llama-nemotron-embed-vl-1b-v2:free"
    assert base == "https://openrouter.ai/api/v1"


def test_embed_dim_fixed_1024():
    """v0.8: 無論 env 如何設定，embed dim 始終為 1024。"""
    from embeddings.local import EMBED_DIM
    assert EMBED_DIM == 1024


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
        vdb.write_text(json.dumps({"data": [], "embedding_dim": 1024}))
        with patch("rag.knowledge_graph.project_root", return_value=Path(td)):
            assert has_knowledge(project_id="test") is False

        # Non-empty vdb
        vdb.write_text(json.dumps({"data": [{"__vector__": [0.1] * 1024}], "embedding_dim": 1024}))
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


def test_extract_sources_requires_verifiable_reference():
    """沒有來源標記與 reference list 時，不應產生 citation。"""
    from rag.knowledge_graph import _extract_sources

    raw = (
        '-----Document Chunks-----\n'
        '{"reference_id":"1","content":"這段文字只有內容，沒有來源標記。"}\n'
        "Reference Document List:\n"
    )
    assert _extract_sources(raw) == []


def test_extract_sources_uses_reference_section_only():
    """reference id 應從 Reference Document List 區段解析，避免誤抓正文。"""
    from rag.knowledge_graph import _extract_sources

    raw = (
        '-----Document Chunks-----\n'
        '{"reference_id":"1","content":"[來源: 測試文件A] 內容A"}\n'
        "正文中可能也有 [1] 但不是來源列表\n"
        "Reference Document List:\n"
        "[1] /tmp/test_a.pdf\n"
    )
    sources = _extract_sources(raw)
    assert len(sources) == 1
    assert sources[0]["file"] == "/tmp/test_a.pdf"


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


def test_vectorstore_uses_jina_v4():
    """v0.8: vectorstore embedding 統一使用 embeddings.local。"""
    import memory.vectorstore as vs
    # Reset cached fn
    vs._embed_fn = None
    vs._embed_dim = None

    import asyncio
    from unittest.mock import AsyncMock
    with patch("embeddings.local.embed_texts", new_callable=AsyncMock) as mock_embed:
        mock_embed.return_value = [[0.1] * 1024]
        fn, dim = asyncio.run(vs._get_embedding_fn())
        assert dim == 1024
    # Reset again for other tests
    vs._embed_fn = None
    vs._embed_dim = None


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
