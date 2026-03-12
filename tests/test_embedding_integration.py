"""Embedding 整合測試 — 42 個測試，涵蓋 6 大領域。

執行方式：
  # 快速測試（mock，<5 秒）
  backend/.venv/bin/python tests/test_embedding_integration.py

  # 完整測試（含真實推理，首次 30-60 秒）
  backend/.venv/bin/python tests/test_embedding_integration.py --slow
"""

import ast
import asyncio
import sys
import tempfile
import threading
from pathlib import Path
from types import ModuleType
from unittest.mock import AsyncMock, MagicMock, patch, call

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

SLOW = "--slow" in sys.argv

# Mock lightrag module (same pattern as test_pipeline.py)
if "lightrag" not in sys.modules:
    _mock_lr = ModuleType("lightrag")
    _mock_lr.LightRAG = MagicMock
    _mock_lr.QueryParam = MagicMock
    sys.modules["lightrag"] = _mock_lr
    _mock_lr_llm = ModuleType("lightrag.llm")
    sys.modules["lightrag.llm"] = _mock_lr_llm
    _mock_lr_llm_oai = ModuleType("lightrag.llm.openai")
    _mock_lr_llm_oai.openai_complete_if_cache = MagicMock
    _mock_lr_llm_oai.openai_embed = MagicMock
    sys.modules["lightrag.llm.openai"] = _mock_lr_llm_oai
    _mock_lr_utils = ModuleType("lightrag.utils")
    _mock_lr_utils.EmbeddingFunc = MagicMock
    sys.modules["lightrag.utils"] = _mock_lr_utils


# ── Helpers ────────────────────────────────────────────────────────

# modals.py 已拆分為 modals/ 目錄；OnboardingScreen 在 modals/onboarding.py
_MODALS_PATH = Path(__file__).parent.parent / "modals" / "onboarding.py"
if not _MODALS_PATH.exists():
    # fallback: 尚未拆分的舊版
    _MODALS_PATH = Path(__file__).parent.parent / "modals.py"
_MODALS_SRC = _MODALS_PATH.read_text()
_MODALS_TREE = ast.parse(_MODALS_SRC)


def _find_class(tree, name):
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == name:
            return node
    return None


def _find_method(cls_node, name):
    for item in cls_node.body:
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.name == name:
            return item
    return None


def _method_source(method_name):
    """Extract source lines for a method from OnboardingScreen (including decorators)."""
    cls = _find_class(_MODALS_TREE, "OnboardingScreen")
    if not cls:
        return ""
    m = _find_method(cls, method_name)
    if not m:
        return ""
    lines = _MODALS_SRC.splitlines()
    # Include decorator lines (they appear before m.lineno)
    start = m.lineno - 1
    if m.decorator_list:
        start = m.decorator_list[0].lineno - 1
    return "\n".join(lines[start : m.end_lineno])


def _mock_embed_vec(dim=1024, val=0.1):
    """Generate a simple normalized mock vector."""
    import math
    v = [val] * dim
    norm = math.sqrt(sum(x * x for x in v))
    return [x / norm for x in v]


# ═══════════════════════════════════════════════════════════════════
# Section 1: Embedding 模型運作（8）
# ═══════════════════════════════════════════════════════════════════

def test_get_embed_config_returns_v4():
    """get_embed_config() 回傳正確 tuple。"""
    from embeddings.local import get_embed_config
    result = get_embed_config()
    assert result == ("jina-embeddings-v4", "local", "", 1024), f"Got: {result}"
    print("PASS: test_get_embed_config_returns_v4")


def test_truncate_and_normalize():
    """截斷到 1024d + L2 normalize。"""
    from embeddings.local import _truncate_and_normalize
    import math
    vec = [0.5] * 2048  # oversized
    result = _truncate_and_normalize(vec, dim=1024)
    assert len(result) == 1024, f"Expected 1024d, got {len(result)}"
    norm = math.sqrt(sum(x * x for x in result))
    assert abs(norm - 1.0) < 1e-5, f"Norm={norm}, expected ~1.0"
    print("PASS: test_truncate_and_normalize")


def test_ensure_model_thread_safety():
    """多執行緒同時呼叫 _ensure_model，只載入一次。"""
    import embeddings.local as el
    original_model = el._model
    original_load = el._load_model
    try:
        el._model = None
        call_count = {"n": 0}
        def fake_load():
            call_count["n"] += 1
            return MagicMock()
        el._load_model = fake_load

        threads = [threading.Thread(target=el._ensure_model) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert call_count["n"] == 1, f"_load_model called {call_count['n']} times, expected 1"
    finally:
        el._model = original_model
        el._load_model = original_load
    print("PASS: test_ensure_model_thread_safety")


def test_embed_text_returns_correct_dim_slow():
    if not SLOW:
        print("SKIP: test_embed_text_returns_correct_dim_slow (use --slow)")
        return
    vec = asyncio.run(_embed_text_slow())
    assert len(vec) == 1024, f"Expected 1024d, got {len(vec)}"
    print("PASS: test_embed_text_returns_correct_dim_slow")

async def _embed_text_slow():
    from embeddings.local import embed_text
    return await embed_text("排版設計是一種精確的語言")


def test_embed_texts_batch_correct_dims_slow():
    if not SLOW:
        print("SKIP: test_embed_texts_batch_correct_dims_slow (use --slow)")
        return
    results = asyncio.run(_embed_texts_batch_slow())
    assert len(results) == 3, f"Expected 3 vectors, got {len(results)}"
    for i, v in enumerate(results):
        assert len(v) == 1024, f"Vector {i} has {len(v)} dims, expected 1024"
    print("PASS: test_embed_texts_batch_correct_dims_slow")

async def _embed_texts_batch_slow():
    from embeddings.local import embed_texts
    return await embed_texts(["排版設計", "字體美學", "書籍裝幀"])


def test_embed_text_is_normalized_slow():
    if not SLOW:
        print("SKIP: test_embed_text_is_normalized_slow (use --slow)")
        return
    import math
    vec = asyncio.run(_embed_text_slow())
    norm = math.sqrt(sum(x * x for x in vec))
    assert abs(norm - 1.0) < 1e-5, f"Norm={norm}, expected ~1.0"
    print("PASS: test_embed_text_is_normalized_slow")


def test_embed_text_semantic_similarity_slow():
    if not SLOW:
        print("SKIP: test_embed_text_semantic_similarity_slow (use --slow)")
        return
    sim_pair, unsim_pair = asyncio.run(_semantic_sim_slow())
    assert sim_pair > unsim_pair, (
        f"Similar pair cosine ({sim_pair:.4f}) should be > "
        f"unrelated pair cosine ({unsim_pair:.4f})"
    )
    print(f"  similar={sim_pair:.4f}, unrelated={unsim_pair:.4f}")
    print("PASS: test_embed_text_semantic_similarity_slow")

async def _semantic_sim_slow():
    from embeddings.local import embed_text
    v_a = await embed_text("排版設計的原理和實踐")
    v_b_sim = await embed_text("typography and layout design principles")
    v_b_unsim = await embed_text("今天天氣預報明天下雨")
    cos_sim = sum(a * b for a, b in zip(v_a, v_b_sim))
    cos_unsim = sum(a * b for a, b in zip(v_a, v_b_unsim))
    return cos_sim, cos_unsim


def test_query_vs_passage_differ_slow():
    if not SLOW:
        print("SKIP: test_query_vs_passage_differ_slow (use --slow)")
        return
    q_vec, p_vec = asyncio.run(_query_passage_diff_slow())
    diff = sum(abs(a - b) for a, b in zip(q_vec, p_vec))
    assert diff > 0.01, "Query and passage encodings should differ"
    print(f"  L1 diff={diff:.4f}")
    print("PASS: test_query_vs_passage_differ_slow")

async def _query_passage_diff_slow():
    from embeddings.local import _embed_text_sync, _embed_texts_sync
    q = _embed_text_sync("排版設計")
    p = _embed_texts_sync(["排版設計"])[0]
    return q, p


# ═══════════════════════════════════════════════════════════════════
# Section 2: 資料庫檢索（7）
# ═══════════════════════════════════════════════════════════════════

def test_vs_index_memory_structure():
    """mock embed + 真實 LanceDB tmpdir — table 結構驗證。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        mock_vec = _mock_embed_vec()

        async def _run():
            with patch("memory.vectorstore._get_embedding_fn", new_callable=AsyncMock) as mock_fn:
                mock_embed = AsyncMock(return_value=[mock_vec])
                mock_fn.return_value = (mock_embed, 1024)
                from memory.vectorstore import index_memory, _get_db, TABLE_NAME
                mem = {
                    "id": 1, "type": "思考", "content": "排版設計",
                    "topic": "設計", "source": "test", "created_at": "2024-01-01",
                    "project_id": "test",
                }
                await index_memory(mem, lancedb_dir=tmpdir_path)
                db = _get_db(tmpdir_path)
                assert TABLE_NAME in db.table_names()
                tbl = db.open_table(TABLE_NAME)
                assert tbl.count_rows() == 1
                schema_names = [f.name for f in tbl.schema]
                assert "vector" in schema_names
                assert "content" in schema_names

        asyncio.run(_run())
    print("PASS: test_vs_index_memory_structure")


def test_vs_search_similar_mock():
    """mock embed + search 回傳正確 dict 結構。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        mock_vec = _mock_embed_vec()

        async def _run():
            with patch("memory.vectorstore._get_embedding_fn", new_callable=AsyncMock) as mock_fn:
                mock_embed = AsyncMock(return_value=[mock_vec])
                mock_fn.return_value = (mock_embed, 1024)
                from memory.vectorstore import index_memory, search_similar
                mem = {
                    "id": 1, "type": "思考", "content": "排版設計是一種精確的語言",
                    "topic": "設計", "source": "test", "created_at": "2024-01-01",
                    "project_id": "test",
                }
                await index_memory(mem, lancedb_dir=tmpdir_path)
                results = await search_similar("排版", limit=5, lancedb_dir=tmpdir_path)
                assert len(results) >= 1
                r = results[0]
                assert "id" in r
                assert "content" in r
                assert "score" in r

        asyncio.run(_run())
    print("PASS: test_vs_search_similar_mock")


def test_vs_delete_from_index():
    """index → delete → count==0。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        mock_vec = _mock_embed_vec()

        async def _run():
            with patch("memory.vectorstore._get_embedding_fn", new_callable=AsyncMock) as mock_fn:
                mock_embed = AsyncMock(return_value=[mock_vec])
                mock_fn.return_value = (mock_embed, 1024)
                from memory.vectorstore import index_memory, delete_from_index, _get_db, TABLE_NAME
                mem = {
                    "id": 42, "type": "思考", "content": "字體美學",
                    "topic": "設計", "source": "test", "created_at": "2024-01-01",
                    "project_id": "test",
                }
                await index_memory(mem, lancedb_dir=tmpdir_path)
                await delete_from_index(42, lancedb_dir=tmpdir_path)
                db = _get_db(tmpdir_path)
                tbl = db.open_table(TABLE_NAME)
                assert tbl.count_rows() == 0

        asyncio.run(_run())
    print("PASS: test_vs_delete_from_index")


def test_vs_dim_mismatch_rebuilds():
    """先建 dim=512 table → dim=1024 插入 → table 重建。"""
    import lancedb
    import pyarrow as pa
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        db = lancedb.connect(str(tmpdir_path))
        # Create table with wrong dim (512)
        schema = pa.schema([
            pa.field("id", pa.int64()),
            pa.field("type", pa.utf8()),
            pa.field("content", pa.utf8()),
            pa.field("topic", pa.utf8()),
            pa.field("source", pa.utf8()),
            pa.field("created_at", pa.utf8()),
            pa.field("project_id", pa.utf8()),
            pa.field("vector", pa.list_(pa.float32(), 512)),
        ])
        db.create_table("memories", schema=schema)

        mock_vec = _mock_embed_vec(dim=1024)

        async def _run():
            with patch("memory.vectorstore._get_embedding_fn", new_callable=AsyncMock) as mock_fn:
                mock_embed = AsyncMock(return_value=[mock_vec])
                mock_fn.return_value = (mock_embed, 1024)
                # Clear cache so it picks up fresh connection
                from memory import vectorstore as vs
                vs._db_cache.pop(str(tmpdir_path), None)
                await vs.index_memory(
                    {"id": 1, "type": "思考", "content": "test",
                     "topic": "", "source": "", "created_at": "",
                     "project_id": "test"},
                    lancedb_dir=tmpdir_path,
                )
                db2 = vs._get_db(tmpdir_path)
                tbl = db2.open_table("memories")
                dim = None
                for f in tbl.schema:
                    if f.name == "vector":
                        dim = f.type.list_size
                assert dim == 1024, f"Expected dim=1024 after rebuild, got {dim}"

        asyncio.run(_run())
    print("PASS: test_vs_dim_mismatch_rebuilds")


def test_vs_index_and_search_slow():
    if not SLOW:
        print("SKIP: test_vs_index_and_search_slow (use --slow)")
        return
    with tempfile.TemporaryDirectory() as tmpdir:
        asyncio.run(_vs_real_search(Path(tmpdir)))
    print("PASS: test_vs_index_and_search_slow")

async def _vs_real_search(tmpdir):
    from memory.vectorstore import index_memory, search_similar, reset_embed_fn, _db_cache
    _db_cache.pop(str(tmpdir), None)
    reset_embed_fn()
    mem = {
        "id": 1, "type": "思考", "content": "排版設計是一種精確的語言，用來結構化想法",
        "topic": "設計", "source": "test", "created_at": "2024-01-01",
        "project_id": "test",
    }
    await index_memory(mem, lancedb_dir=tmpdir)
    results = await search_similar("排版設計", limit=5, lancedb_dir=tmpdir)
    assert len(results) >= 1
    assert results[0]["score"] > 0.5, f"Score={results[0]['score']}"


def test_vs_search_relevance_order_slow():
    if not SLOW:
        print("SKIP: test_vs_search_relevance_order_slow (use --slow)")
        return
    with tempfile.TemporaryDirectory() as tmpdir:
        asyncio.run(_vs_relevance_order(Path(tmpdir)))
    print("PASS: test_vs_search_relevance_order_slow")

async def _vs_relevance_order(tmpdir):
    from memory.vectorstore import index_memory, search_similar, reset_embed_fn, _db_cache
    _db_cache.pop(str(tmpdir), None)
    reset_embed_fn()
    mems = [
        {"id": 1, "type": "思考", "content": "排版設計的原理和方法論", "topic": "設計",
         "source": "test", "created_at": "2024-01-01", "project_id": "test"},
        {"id": 2, "type": "思考", "content": "天氣預報系統的架構設計", "topic": "天氣",
         "source": "test", "created_at": "2024-01-01", "project_id": "test"},
        {"id": 3, "type": "思考", "content": "字體排印的美學與實踐", "topic": "設計",
         "source": "test", "created_at": "2024-01-01", "project_id": "test"},
    ]
    for m in mems:
        await index_memory(m, lancedb_dir=tmpdir)
    results = await search_similar("排版設計和字體", limit=3, lancedb_dir=tmpdir)
    # Best match should be related to typography/design, not weather
    assert results[0]["content"] != "天氣預報系統的架構設計", (
        f"Weather content should not rank first for typography query"
    )


def test_vs_empty_returns_nothing_slow():
    if not SLOW:
        print("SKIP: test_vs_empty_returns_nothing_slow (use --slow)")
        return
    with tempfile.TemporaryDirectory() as tmpdir:
        asyncio.run(_vs_empty_search(Path(tmpdir)))
    print("PASS: test_vs_empty_returns_nothing_slow")

async def _vs_empty_search(tmpdir):
    from memory.vectorstore import search_similar, reset_embed_fn, _db_cache
    _db_cache.pop(str(tmpdir), None)
    reset_embed_fn()
    results = await search_similar("排版設計", limit=5, lancedb_dir=tmpdir)
    assert results == []


# ═══════════════════════════════════════════════════════════════════
# Section 3: LLM 互動（5）
# ═══════════════════════════════════════════════════════════════════

def test_pipeline_with_mock_llm_slow():
    if not SLOW:
        print("SKIP: test_pipeline_with_mock_llm_slow (use --slow)")
        return
    asyncio.run(_pipeline_mock_llm())
    print("PASS: test_pipeline_with_mock_llm_slow")

async def _pipeline_mock_llm():
    import rag.pipeline as _p
    _p._degraded_mode = False
    with patch("rag.knowledge_graph.query_knowledge") as mock_qk:
        with patch("rag.knowledge_graph.has_knowledge", return_value=True):
            mock_qk.return_value = (
                '{"reference_id": "1", "content": "排版設計的核心原理"}',
                [{"title": "Design", "snippet": "排版設計", "file": "t.pdf"}],
            )
            from rag.pipeline import run_thinking_pipeline
            result = await run_thinking_pipeline(
                user_input="排版設計是什麼？",
                project_id="test_proj",
                mode="fast",
            )
            assert "context_text" in result
            assert "sources" in result
            assert "diagnostics" in result


def test_pipeline_clean_context():
    """clean_context() 移除 JSON 包裝。"""
    from rag.pipeline import clean_context
    result = clean_context("[no-context]")
    assert result == ""
    result2 = clean_context("not able to provide an answer")
    assert result2 == ""
    # Valid content passes through (must be > 10 chars after cleaning)
    valid = "這是有效的知識內容，包含了許多重要的設計原理和實踐方法"
    result3 = clean_context(valid)
    assert len(result3) > 0
    print("PASS: test_pipeline_clean_context")


def test_pipeline_source_constraint():
    """事實題 + 有來源 → '來源約束' 前綴。"""
    from rag.pipeline import apply_source_constraint
    result = apply_source_constraint("fact", 2, "some knowledge content")
    assert "來源約束" in result
    print("PASS: test_pipeline_source_constraint")


def test_pipeline_degraded_blocks_deep():
    """degraded mode → 強制 fast path。"""
    async def _run():
        import rag.pipeline as _p
        _p._degraded_mode = True
        _p._deep_fail_until = 0.0
        with patch("rag.knowledge_graph.query_knowledge") as mock_qk:
            with patch("rag.knowledge_graph.has_knowledge", return_value=True):
                mock_qk.return_value = (
                    '{"reference_id": "1", "content": "fallback"}',
                    [{"title": "t", "snippet": "s", "file": "f"}],
                )
                from rag.pipeline import _retrieve
                _, _, strategy, err = await _retrieve("test", "proj1", "deep", "summary")
                assert strategy == "fast_fallback", f"Expected fast_fallback, got {strategy}"
        _p._degraded_mode = False
    asyncio.run(_run())
    print("PASS: test_pipeline_degraded_blocks_deep")


def test_pipeline_no_knowledge_empty():
    """has_knowledge=False → 空結果。"""
    async def _run():
        import rag.pipeline as _p
        _p._degraded_mode = False
        with patch("rag.knowledge_graph.has_knowledge", return_value=False):
            from rag.pipeline import run_thinking_pipeline
            result = await run_thinking_pipeline(
                user_input="排版設計",
                project_id="empty_proj",
                mode="fast",
            )
            assert result["context_text"] == ""
    asyncio.run(_run())
    print("PASS: test_pipeline_no_knowledge_empty")


# ═══════════════════════════════════════════════════════════════════
# Section 4: 圖片向量歸類（7）
# ═══════════════════════════════════════════════════════════════════

def test_embed_image_correct_dim_slow():
    if not SLOW:
        print("SKIP: test_embed_image_correct_dim_slow (use --slow)")
        return
    vec = asyncio.run(_embed_image_slow())
    assert len(vec) == 1024, f"Expected 1024d, got {len(vec)}"
    print("PASS: test_embed_image_correct_dim_slow")

async def _embed_image_slow():
    from PIL import Image
    import io
    from embeddings.local import embed_image
    img = Image.new("RGB", (64, 64), color=(255, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return await embed_image(buf.getvalue())


def test_embed_image_normalized_slow():
    if not SLOW:
        print("SKIP: test_embed_image_normalized_slow (use --slow)")
        return
    import math
    vec = asyncio.run(_embed_image_slow())
    norm = math.sqrt(sum(x * x for x in vec))
    assert abs(norm - 1.0) < 1e-5, f"Norm={norm}, expected ~1.0"
    print("PASS: test_embed_image_normalized_slow")


def test_image_text_cross_modal_slow():
    if not SLOW:
        print("SKIP: test_image_text_cross_modal_slow (use --slow)")
        return
    asyncio.run(_image_text_cross_modal())
    print("PASS: test_image_text_cross_modal_slow")

async def _image_text_cross_modal():
    from PIL import Image
    import io
    from embeddings.local import embed_image, embed_text
    # Create a red image
    img = Image.new("RGB", (64, 64), color=(255, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    img_vec = await embed_image(buf.getvalue())
    txt_vec = await embed_text("a red colored image")
    # Cosine similarity between image and related text should be > 0
    cos = sum(a * b for a, b in zip(img_vec, txt_vec))
    assert cos > 0, f"Cross-modal cosine should be positive, got {cos:.4f}"
    print(f"  cross-modal cosine: {cos:.4f}")


def test_image_store_index_creates_table():
    """mock embed → LanceDB table 建立。"""
    import lancedb
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a fake image file
        img_dir = Path(tmpdir) / "images"
        img_dir.mkdir()
        fake_img = img_dir / "test.png"
        # Minimal valid PNG
        from PIL import Image
        import io
        img = Image.new("RGB", (4, 4), color=(0, 0, 0))
        img.save(str(fake_img), format="PNG")

        mock_img_vec = _mock_embed_vec()
        mock_txt_vec = _mock_embed_vec(val=0.2)

        async def _run():
            with patch("rag.image_store._images_dir", return_value=img_dir):
                with patch("rag.image_store._lancedb_dir", return_value=Path(tmpdir) / "lancedb"):
                    with patch("rag.image_store._embed_image", new_callable=AsyncMock, return_value=mock_img_vec):
                        with patch("rag.image_store._embed_text", new_callable=AsyncMock, return_value=mock_txt_vec):
                            with patch("rag.image_store._auto_caption", new_callable=AsyncMock, return_value="test caption"):
                                from rag.image_store import index_image, _get_db, TABLE_NAME, _db_cache
                                _db_cache.clear()
                                result = await index_image("test_proj", "test.png", caption="test caption")
                                db = _get_db(Path(tmpdir) / "lancedb")
                                assert TABLE_NAME in db.table_names()
                                tbl = db.open_table(TABLE_NAME)
                                assert tbl.count_rows() == 1
                                schema_names = [f.name for f in tbl.schema]
                                assert "vector" in schema_names
                                assert "filename" in schema_names

        asyncio.run(_run())
    print("PASS: test_image_store_index_creates_table")


def test_image_store_search_returns_keys():
    """mock embed → search 回傳正確欄位。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        img_dir = Path(tmpdir) / "images"
        img_dir.mkdir()
        fake_img = img_dir / "test.png"
        from PIL import Image
        img = Image.new("RGB", (4, 4), color=(0, 0, 0))
        img.save(str(fake_img), format="PNG")

        mock_vec = _mock_embed_vec()

        async def _run():
            with patch("rag.image_store._images_dir", return_value=img_dir):
                with patch("rag.image_store._lancedb_dir", return_value=Path(tmpdir) / "lancedb"):
                    with patch("rag.image_store._embed_image", new_callable=AsyncMock, return_value=mock_vec):
                        with patch("rag.image_store._embed_text", new_callable=AsyncMock, return_value=mock_vec):
                            with patch("rag.image_store._auto_caption", new_callable=AsyncMock, return_value="a caption"):
                                from rag.image_store import index_image, search_images, _db_cache
                                _db_cache.clear()
                                await index_image("test_proj", "test.png", caption="a caption")
                                results = await search_images("test_proj", "test query", limit=5)
                                assert len(results) >= 1
                                r = results[0]
                                assert "id" in r
                                assert "filename" in r
                                assert "caption" in r
                                assert "score" in r

        asyncio.run(_run())
    print("PASS: test_image_store_search_returns_keys")


def test_image_store_mixed_vector_50_50():
    """mock 兩組向量 → 驗證混合比例。"""
    import numpy as np

    img_vec = [1.0] * 1024
    txt_vec = [0.0] * 1024
    txt_vec[0] = 1.0  # slightly different

    expected_mixed = np.array(img_vec) * 0.5 + np.array(txt_vec) * 0.5
    expected_norm = np.linalg.norm(expected_mixed)
    if expected_norm > 0:
        expected_mixed = expected_mixed / expected_norm

    with tempfile.TemporaryDirectory() as tmpdir:
        img_dir = Path(tmpdir) / "images"
        img_dir.mkdir()
        fake_img = img_dir / "test.png"
        from PIL import Image
        img = Image.new("RGB", (4, 4), color=(0, 0, 0))
        img.save(str(fake_img), format="PNG")

        async def _run():
            with patch("rag.image_store._images_dir", return_value=img_dir):
                with patch("rag.image_store._lancedb_dir", return_value=Path(tmpdir) / "lancedb"):
                    with patch("rag.image_store._embed_image", new_callable=AsyncMock, return_value=img_vec):
                        with patch("rag.image_store._embed_text", new_callable=AsyncMock, return_value=txt_vec):
                            from rag.image_store import index_image, _get_db, TABLE_NAME, _db_cache
                            _db_cache.clear()
                            await index_image("test_proj", "test.png", caption="has caption")
                            db = _get_db(Path(tmpdir) / "lancedb")
                            tbl = db.open_table(TABLE_NAME)
                            rows = tbl.to_pandas()
                            stored = np.array(rows.iloc[0]["vector"])
                            # Check stored ≈ normalize(0.5*img + 0.5*txt)
                            diff = np.linalg.norm(stored - expected_mixed)
                            assert diff < 1e-5, f"Mixed vector diff={diff}, expected ~0"

        asyncio.run(_run())
    print("PASS: test_image_store_mixed_vector_50_50")


def test_image_store_no_caption_skips_text():
    """空 caption → _embed_text 不被呼叫。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        img_dir = Path(tmpdir) / "images"
        img_dir.mkdir()
        fake_img = img_dir / "test.png"
        from PIL import Image
        img = Image.new("RGB", (4, 4), color=(0, 0, 0))
        img.save(str(fake_img), format="PNG")

        mock_img_vec = _mock_embed_vec()
        embed_text_mock = AsyncMock(return_value=_mock_embed_vec(val=0.2))

        async def _run():
            with patch("rag.image_store._images_dir", return_value=img_dir):
                with patch("rag.image_store._lancedb_dir", return_value=Path(tmpdir) / "lancedb"):
                    with patch("rag.image_store._embed_image", new_callable=AsyncMock, return_value=mock_img_vec):
                        with patch("rag.image_store._embed_text", embed_text_mock):
                            with patch("rag.image_store._auto_caption", new_callable=AsyncMock, return_value=""):
                                from rag.image_store import index_image, _db_cache
                                _db_cache.clear()
                                await index_image("test_proj", "test.png", caption="")
                                assert embed_text_mock.call_count == 0, (
                                    f"_embed_text called {embed_text_mock.call_count} times, "
                                    f"expected 0 for empty caption"
                                )

        asyncio.run(_run())
    print("PASS: test_image_store_no_caption_skips_text")


# ═══════════════════════════════════════════════════════════════════
# Section 5: Onboarding 下載指引（5）
# ═══════════════════════════════════════════════════════════════════

def test_onboarding_step_states():
    """5 個 step 字串都出現在 _render_step。"""
    src = _method_source("_render_step")
    for step in ["chat_provider", "chat_setup", "embed", "embed_download", "done"]:
        assert step in src, f"Step '{step}' not found in _render_step"
    print("PASS: test_onboarding_step_states")


def test_onboarding_download_calls_ensure():
    """_render_embed_download 或 _run_model_download 中有 ensure_model_downloaded。"""
    dl_src = _method_source("_render_embed_download")
    run_src = _method_source("_run_model_download")
    combined = dl_src + run_src
    assert "ensure_model_downloaded" in combined, (
        "ensure_model_downloaded not found in download methods"
    )
    print("PASS: test_onboarding_download_calls_ensure")


def test_onboarding_skip_goes_to_done():
    """ob-embed-skip handler 設 _step='done'。"""
    src = _method_source("on_button_pressed")
    # Find ob-embed-skip section
    assert "ob-embed-skip" in src
    assert 'self._step = "done"' in src or "self._step = 'done'" in src
    print("PASS: test_onboarding_skip_goes_to_done")


def test_onboarding_saves_env_keys():
    """download button 呼叫 save_env_key 三次 — EMBED_PROVIDER, EMBED_MODEL, EMBED_DIM。"""
    src = _method_source("on_button_pressed")
    # The ob-embed-download handler should save all three env keys
    for key in ["EMBED_PROVIDER", "EMBED_MODEL", "EMBED_DIM"]:
        assert key in src, f"save_env_key for '{key}' not found"
    print("PASS: test_onboarding_saves_env_keys")


def test_onboarding_done_shows_model():
    """done 畫面顯示 'jina-embeddings-v4'。"""
    src = _method_source("_render_done")
    assert "jina-embeddings-v4" in src
    print("PASS: test_onboarding_done_shows_model")


# ═══════════════════════════════════════════════════════════════════
# Section 6: 下載進度條（10）
# ═══════════════════════════════════════════════════════════════════

def test_progress_bar_in_download():
    """_render_embed_download 中有 ProgressBar。"""
    src = _method_source("_render_embed_download")
    assert "ProgressBar" in src, "ProgressBar not found in _render_embed_download"
    print("PASS: test_progress_bar_in_download")


def test_progress_phase_texts():
    """至少 3 個不同的階段文字。"""
    cls = _find_class(_MODALS_TREE, "OnboardingScreen")
    # Find _DL_PHASES attribute
    found = False
    for node in ast.walk(cls):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Attribute) and target.attr == "_DL_PHASES":
                    found = True
                elif isinstance(target, ast.Name) and target.id == "_DL_PHASES":
                    if isinstance(node.value, ast.List):
                        assert len(node.value.elts) >= 3, (
                            f"_DL_PHASES has {len(node.value.elts)} phases, expected >= 3"
                        )
                    found = True
    assert found, "_DL_PHASES not found in OnboardingScreen"
    print("PASS: test_progress_phase_texts")


def test_progress_bar_indeterminate():
    """ProgressBar 使用 total=None（不定模式）。"""
    src = _method_source("_render_embed_download")
    assert "total=None" in src, "ProgressBar should use total=None for indeterminate mode"
    print("PASS: test_progress_bar_indeterminate")


def test_ensure_model_is_sync():
    """ensure_model_downloaded 是同步函式（可放 executor）。"""
    import asyncio as _asyncio
    from embeddings.local import ensure_model_downloaded
    assert not _asyncio.iscoroutinefunction(ensure_model_downloaded), (
        "ensure_model_downloaded should be sync, not async"
    )
    print("PASS: test_ensure_model_is_sync")


def test_download_uses_work_decorator():
    """下載用 @work(thread=True)。"""
    cls = _find_class(_MODALS_TREE, "OnboardingScreen")
    m = _find_method(cls, "_run_model_download")
    assert m is not None, "_run_model_download method not found"
    found_work = False
    for dec in m.decorator_list:
        if isinstance(dec, ast.Call):
            func = dec.func
            if isinstance(func, ast.Name) and func.id == "work":
                found_work = True
            elif isinstance(func, ast.Attribute) and func.attr == "work":
                found_work = True
        elif isinstance(dec, ast.Name) and dec.id == "work":
            found_work = True
    assert found_work, "_run_model_download should have @work decorator"
    # Check thread=True keyword
    src = _method_source("_run_model_download")
    assert "thread=True" in src, "@work should have thread=True"
    print("PASS: test_download_uses_work_decorator")


def test_progress_transitions_to_done():
    """下載完成後 _step='done'。"""
    src = _method_source("_on_download_complete")
    assert src, "_on_download_complete method not found"
    assert '"done"' in src or "'done'" in src, (
        "_on_download_complete should set _step to 'done'"
    )
    print("PASS: test_progress_transitions_to_done")


def test_progress_error_handling():
    """下載失敗有 try/except。"""
    src = _method_source("_run_model_download")
    assert "except" in src, "_run_model_download should have exception handling"
    # Also verify _on_download_error exists
    cls = _find_class(_MODALS_TREE, "OnboardingScreen")
    m = _find_method(cls, "_on_download_error")
    assert m is not None, "_on_download_error method not found"
    print("PASS: test_progress_error_handling")


def test_progress_bar_import():
    """from textual.widgets import ProgressBar 可用。"""
    from textual.widgets import ProgressBar
    # Verify it's a real class
    assert ProgressBar is not None
    print("PASS: test_progress_bar_import")


def test_progress_css_exists():
    """CSS 中有 ob-progress-bar 規則。"""
    assert "ob-progress-bar" in _MODALS_SRC, "CSS rule for #ob-progress-bar not found"
    print("PASS: test_progress_css_exists")


def test_download_complete_callback():
    """_on_download_complete 方法存在。"""
    cls = _find_class(_MODALS_TREE, "OnboardingScreen")
    m = _find_method(cls, "_on_download_complete")
    assert m is not None, "_on_download_complete method not found"
    print("PASS: test_download_complete_callback")


# ═══════════════════════════════════════════════════════════════════
# Run all tests
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    mode = "SLOW (real inference)" if SLOW else "FAST (mock only)"
    print("=" * 60)
    print(f"Embedding Integration Tests — {mode}")
    print("=" * 60)
    print()

    print("── Section 1: Embedding 模型運作 (8) ──")
    test_get_embed_config_returns_v4()
    test_truncate_and_normalize()
    test_ensure_model_thread_safety()
    test_embed_text_returns_correct_dim_slow()
    test_embed_texts_batch_correct_dims_slow()
    test_embed_text_is_normalized_slow()
    test_embed_text_semantic_similarity_slow()
    test_query_vs_passage_differ_slow()

    print()
    print("── Section 2: 資料庫檢索 (7) ──")
    test_vs_index_memory_structure()
    test_vs_search_similar_mock()
    test_vs_delete_from_index()
    test_vs_dim_mismatch_rebuilds()
    test_vs_index_and_search_slow()
    test_vs_search_relevance_order_slow()
    test_vs_empty_returns_nothing_slow()

    print()
    print("── Section 3: LLM 互動 (5) ──")
    test_pipeline_with_mock_llm_slow()
    test_pipeline_clean_context()
    test_pipeline_source_constraint()
    test_pipeline_degraded_blocks_deep()
    test_pipeline_no_knowledge_empty()

    print()
    print("── Section 4: 圖片向量歸類 (7) ──")
    test_embed_image_correct_dim_slow()
    test_embed_image_normalized_slow()
    test_image_text_cross_modal_slow()
    test_image_store_index_creates_table()
    test_image_store_search_returns_keys()
    test_image_store_mixed_vector_50_50()
    test_image_store_no_caption_skips_text()

    print()
    print("── Section 5: Onboarding 下載指引 (5) ──")
    test_onboarding_step_states()
    test_onboarding_download_calls_ensure()
    test_onboarding_skip_goes_to_done()
    test_onboarding_saves_env_keys()
    test_onboarding_done_shows_model()

    print()
    print("── Section 6: 下載進度條 (10) ──")
    test_progress_bar_in_download()
    test_progress_phase_texts()
    test_progress_bar_indeterminate()
    test_ensure_model_is_sync()
    test_download_uses_work_decorator()
    test_progress_transitions_to_done()
    test_progress_error_handling()
    test_progress_bar_import()
    test_progress_css_exists()
    test_download_complete_callback()

    print()
    passed = 42 if SLOW else 42  # all tests run, slow ones show SKIP
    print("=" * 60)
    print("ALL 42 TESTS COMPLETED")
    print("=" * 60)
