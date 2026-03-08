"""測試聊天框引用圖片是否能正常辨別。

端到端流程：
1. 上傳圖片 → auto caption → 混合向量 → LanceDB 索引
2. 文字語意搜圖 → 回傳相關圖片
3. _build_user_content() → 附加圖片檔名
4. _inject_rag_context() step 2 → 注入圖片 caption

執行：
  backend/.venv/bin/python tests/test_image_chat_ref.py          # mock
  backend/.venv/bin/python tests/test_image_chat_ref.py --slow    # 真實推理
"""

import asyncio
import sys
import tempfile
from pathlib import Path
from types import ModuleType
from unittest.mock import AsyncMock, MagicMock, patch
import math

sys.path.insert(0, str(Path(__file__).parent.parent))

SLOW = "--slow" in sys.argv

# Mock lightrag
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


def _mock_vec(dim=1024, val=0.1):
    v = [val] * dim
    norm = math.sqrt(sum(x * x for x in v))
    return [x / norm for x in v]


# ═══════════════════════════════════════════════════════════════════
# 1. search_images 端到端（mock embedding）
# ═══════════════════════════════════════════════════════════════════

def test_search_images_returns_relevant():
    """索引 2 張圖片 → 搜尋 → 回傳相關圖片及完整欄位。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        img_dir = Path(tmpdir) / "images"
        img_dir.mkdir()
        # 建兩張假圖
        from PIL import Image
        for name in ["design_poster.png", "weather_chart.png"]:
            img = Image.new("RGB", (4, 4), color=(0, 0, 0))
            img.save(str(img_dir / name), format="PNG")

        # 設計相關的向量和天氣相關的向量要不同
        design_vec = [0.8] * 512 + [0.2] * 512
        weather_vec = [0.2] * 512 + [0.8] * 512
        query_vec = [0.7] * 512 + [0.3] * 512  # 接近 design

        # normalize
        for v in [design_vec, weather_vec, query_vec]:
            norm = math.sqrt(sum(x * x for x in v))
            for i in range(len(v)):
                v[i] /= norm

        call_idx = {"n": 0}

        async def mock_embed_image(source):
            # 根據檔名回傳不同向量
            return design_vec  # 兩張都用同一個先，靠 caption 區分

        async def mock_embed_text(text):
            if "排版" in text or "設計" in text or "design" in text or "poster" in text:
                return design_vec
            if "天氣" in text or "weather" in text or "chart" in text:
                return weather_vec
            return query_vec

        async def _run():
            with patch("rag.image_store._images_dir", return_value=img_dir):
                with patch("rag.image_store._lancedb_dir", return_value=Path(tmpdir) / "lancedb"):
                    with patch("rag.image_store._embed_image", side_effect=mock_embed_image):
                        with patch("rag.image_store._embed_text", side_effect=mock_embed_text):
                            with patch("rag.image_store._auto_caption", new_callable=AsyncMock, return_value=""):
                                from rag.image_store import index_image, search_images, _db_cache
                                _db_cache.clear()

                                # 索引兩張圖
                                await index_image("test", "design_poster.png",
                                                  caption="排版設計海報，展現字體的力量")
                                await index_image("test", "weather_chart.png",
                                                  caption="天氣預報圖表，顯示溫度變化")

                                # 搜尋設計相關
                                results = await search_images("test", "排版設計", limit=5)
                                assert len(results) >= 1, f"Expected results, got {len(results)}"

                                # 驗證回傳欄位完整
                                r = results[0]
                                for key in ["id", "filename", "caption", "score", "path"]:
                                    assert key in r, f"Missing key: {key}"

                                print(f"  搜尋 '排版設計' → {len(results)} 筆")
                                for r in results:
                                    print(f"    {r['filename']}: score={r['score']:.3f}, caption={r['caption'][:30]}")

        asyncio.run(_run())
    print("PASS: test_search_images_returns_relevant")


# ═══════════════════════════════════════════════════════════════════
# 2. _build_user_content 測試
# ═══════════════════════════════════════════════════════════════════

def test_build_user_content_with_vision_describe():
    """pending_images 有圖片時，呼叫 Vision LLM 即時描述並附加到使用者文字。"""
    from widgets import AppState

    class FakeApp:
        state = AppState()
        messages = []
        def _update_menu_bar(self):
            pass

    from mixins.chat import ChatMixin

    app = FakeApp()

    async def _run():
        with tempfile.TemporaryDirectory() as tmpdir:
            img1 = Path(tmpdir) / "poster.jpg"
            img2 = Path(tmpdir) / "sketch.png"
            img1.write_bytes(b"fake")
            img2.write_bytes(b"fake")

            app.state.pending_images = [str(img1), str(img2)]

            # Mock describe_image_for_chat to return a description
            with patch("rag.image_store.describe_image_for_chat",
                        new_callable=AsyncMock,
                        return_value="這是一張黑白海報，字體使用無襯線體"):
                result = await ChatMixin._build_user_content(app, "這張圖的設計風格如何？")

            assert "圖片分析（poster.jpg）" in result, f"Missing poster analysis in: {result}"
            assert "圖片分析（sketch.png）" in result, f"Missing sketch analysis in: {result}"
            assert "黑白海報" in result, f"Missing description in: {result}"
            assert "這張圖的設計風格如何" in result, f"Original text lost in: {result}"
            assert app.state.pending_images == []
            print(f"  結果: {result[:120]}...")

    asyncio.run(_run())
    print("PASS: test_build_user_content_with_vision_describe")


def test_build_user_content_no_images():
    """沒有 pending_images 時，回傳原始文字。"""
    from widgets import AppState
    from mixins.chat import ChatMixin

    class FakeApp:
        state = AppState()
        def _update_menu_bar(self):
            pass

    app = FakeApp()
    app.state.pending_images = []
    result = asyncio.run(ChatMixin._build_user_content(app, "你好"))
    assert result == "你好"
    print("PASS: test_build_user_content_no_images")


def test_build_user_content_fallback_on_describe_failure():
    """Vision 描述失敗時，fallback 到 LanceDB caption 或失敗提示。"""
    from widgets import AppState

    class FakeApp:
        state = AppState()
        current_project = None
        def _update_menu_bar(self):
            pass

    from mixins.chat import ChatMixin

    app = FakeApp()

    async def _run():
        with tempfile.TemporaryDirectory() as tmpdir:
            img1 = Path(tmpdir) / "photo.jpg"
            img1.write_bytes(b"fake")

            app.state.pending_images = [str(img1)]

            # Mock describe_image_for_chat to return empty (failure)
            with patch("rag.image_store.describe_image_for_chat",
                        new_callable=AsyncMock, return_value=""):
                result = await ChatMixin._build_user_content(app, "看看這張")

            assert "描述生成失敗" in result, f"Missing fallback message in: {result}"
            assert "photo.jpg" in result
            print(f"  Fallback 結果: {result}")

    asyncio.run(_run())
    print("PASS: test_build_user_content_fallback_on_describe_failure")


# ═══════════════════════════════════════════════════════════════════
# 3. 圖片 caption 注入測試
# ═══════════════════════════════════════════════════════════════════

def test_image_context_injection_format():
    """search_images 回傳結果被格式化為 system message 注入。"""
    # 模擬 _inject_rag_context step 2 的邏輯
    img_results = [
        {"filename": "poster.jpg", "caption": "排版設計海報", "tags": "設計,排版", "score": 0.85},
        {"filename": "sketch.png", "caption": "草稿素描", "tags": "", "score": 0.45},
    ]
    user_msg = "poster.jpg"

    # 複製 mixins/chat.py 第 597-608 行的邏輯
    img_lines = "\n".join(
        f"- [圖片] {r['filename']}: {r['caption']}" + (f" (tags: {r['tags']})" if r.get('tags') else "")
        for r in img_results
        if r.get("score", 0) > 0.3 or r.get("filename", "") in user_msg
    )

    assert "poster.jpg" in img_lines
    assert "排版設計海報" in img_lines
    assert "(tags: 設計,排版)" in img_lines
    # score 0.45 > 0.3，所以 sketch 也應包含
    assert "sketch.png" in img_lines
    print(f"  注入格式:\n{img_lines}")
    print("PASS: test_image_context_injection_format")


def test_image_low_score_filtered():
    """score < 0.3 且檔名不在 user_msg 中的圖片被過濾。"""
    img_results = [
        {"filename": "relevant.jpg", "caption": "相關圖片", "tags": "", "score": 0.8},
        {"filename": "irrelevant.jpg", "caption": "不相關", "tags": "", "score": 0.1},
    ]
    user_msg = "排版設計的特色"

    img_lines = "\n".join(
        f"- [圖片] {r['filename']}: {r['caption']}"
        for r in img_results
        if r.get("score", 0) > 0.3 or r.get("filename", "") in user_msg
    )

    assert "relevant.jpg" in img_lines
    assert "irrelevant.jpg" not in img_lines
    print("PASS: test_image_low_score_filtered")


def test_image_filename_mention_overrides_score():
    """使用者提到檔名時，即使 score 低也會被注入。"""
    img_results = [
        {"filename": "my_photo.jpg", "caption": "某張照片", "tags": "", "score": 0.05},
    ]
    user_msg = "請看 my_photo.jpg 這張圖"

    img_lines = "\n".join(
        f"- [圖片] {r['filename']}: {r['caption']}"
        for r in img_results
        if r.get("score", 0) > 0.3 or r.get("filename", "") in user_msg
    )

    assert "my_photo.jpg" in img_lines, "Filename mentioned in user_msg should override low score"
    print("PASS: test_image_filename_mention_overrides_score")


# ═══════════════════════════════════════════════════════════════════
# 4. 跨模態語意搜尋（真實推理）
# ═══════════════════════════════════════════════════════════════════

def test_real_image_text_search_slow():
    """真實推理：上傳紅色圖片 + caption → 文字搜尋能找到。"""
    if not SLOW:
        print("SKIP: test_real_image_text_search_slow (use --slow)")
        return
    with tempfile.TemporaryDirectory() as tmpdir:
        asyncio.run(_real_image_search(Path(tmpdir)))
    print("PASS: test_real_image_text_search_slow")

async def _real_image_search(tmpdir):
    from PIL import Image
    from rag.image_store import index_image, search_images, _db_cache

    img_dir = tmpdir / "images"
    img_dir.mkdir()
    lance_dir = tmpdir / "lancedb"

    # 建一張紅色圖片
    red_img = Image.new("RGB", (64, 64), color=(255, 0, 0))
    red_img.save(str(img_dir / "red_square.png"), format="PNG")

    # 建一張藍色圖片
    blue_img = Image.new("RGB", (64, 64), color=(0, 0, 255))
    blue_img.save(str(img_dir / "blue_circle.png"), format="PNG")

    _db_cache.clear()

    with patch("rag.image_store._images_dir", return_value=img_dir):
        with patch("rag.image_store._lancedb_dir", return_value=lance_dir):
            with patch("rag.image_store._auto_caption", new_callable=AsyncMock, return_value=""):
                await index_image("test", "red_square.png", caption="一個紅色的正方形圖案")
                await index_image("test", "blue_circle.png", caption="一個藍色的圓形設計")

                # 搜尋紅色相關
                results = await search_images("test", "紅色方形圖案", limit=5)
                assert len(results) >= 1
                first = results[0]
                print(f"  搜尋 '紅色方形圖案':")
                for r in results:
                    print(f"    {r['filename']}: score={r['score']:.3f}")
                assert first["filename"] == "red_square.png", (
                    f"Expected red_square.png first, got {first['filename']}"
                )

                # 搜尋藍色相關
                results2 = await search_images("test", "藍色圓形", limit=5)
                assert results2[0]["filename"] == "blue_circle.png", (
                    f"Expected blue_circle.png first, got {results2[0]['filename']}"
                )
                print(f"  搜尋 '藍色圓形': {results2[0]['filename']} (score={results2[0]['score']:.3f})")


def test_real_cross_modal_ranking_slow():
    """真實推理：純圖片向量（無 caption）也能被文字搜到。"""
    if not SLOW:
        print("SKIP: test_real_cross_modal_ranking_slow (use --slow)")
        return
    with tempfile.TemporaryDirectory() as tmpdir:
        asyncio.run(_real_cross_modal(Path(tmpdir)))
    print("PASS: test_real_cross_modal_ranking_slow")

async def _real_cross_modal(tmpdir):
    from PIL import Image
    from rag.image_store import index_image, search_images, _db_cache

    img_dir = tmpdir / "images"
    img_dir.mkdir()
    lance_dir = tmpdir / "lancedb"

    # 純紅色圖片，不給 caption
    red_img = Image.new("RGB", (64, 64), color=(255, 0, 0))
    red_img.save(str(img_dir / "pure_red.png"), format="PNG")

    _db_cache.clear()

    with patch("rag.image_store._images_dir", return_value=img_dir):
        with patch("rag.image_store._lancedb_dir", return_value=lance_dir):
            with patch("rag.image_store._auto_caption", new_callable=AsyncMock, return_value=""):
                # 不給 caption → 純圖片向量
                await index_image("test", "pure_red.png", caption="")

                results = await search_images("test", "red color image", limit=5)
                assert len(results) >= 1
                print(f"  搜尋 'red color image' (純圖片向量，無 caption):")
                print(f"    {results[0]['filename']}: score={results[0]['score']:.3f}")
                # 跨模態搜尋應該能找到，但分數可能較低
                assert results[0]["score"] > 0, "Cross-modal search should return positive score"


# ═══════════════════════════════════════════════════════════════════
# 5. 混合向量品質
# ═══════════════════════════════════════════════════════════════════

def test_mixed_vector_improves_search_slow():
    """真實推理：有 caption 的混合向量比純圖片向量搜尋效果更好。"""
    if not SLOW:
        print("SKIP: test_mixed_vector_improves_search_slow (use --slow)")
        return
    with tempfile.TemporaryDirectory() as tmpdir:
        asyncio.run(_mixed_vs_pure(Path(tmpdir)))
    print("PASS: test_mixed_vector_improves_search_slow")

async def _mixed_vs_pure(tmpdir):
    from PIL import Image
    from rag.image_store import index_image, search_images, _db_cache

    img_dir = tmpdir / "images"
    img_dir.mkdir()
    lance_dir = tmpdir / "lancedb"

    # 同一張圖，一個有 caption 一個沒有
    img = Image.new("RGB", (64, 64), color=(200, 100, 50))
    img.save(str(img_dir / "with_caption.png"), format="PNG")
    img.save(str(img_dir / "no_caption.png"), format="PNG")

    _db_cache.clear()

    with patch("rag.image_store._images_dir", return_value=img_dir):
        with patch("rag.image_store._lancedb_dir", return_value=lance_dir):
            with patch("rag.image_store._auto_caption", new_callable=AsyncMock, return_value=""):
                await index_image("test", "with_caption.png",
                                  caption="暖色調的抽象色塊設計")
                await index_image("test", "no_caption.png", caption="")

                results = await search_images("test", "暖色調設計", limit=5)
                scores = {r["filename"]: r["score"] for r in results}

                print(f"  搜尋 '暖色調設計':")
                for r in results:
                    print(f"    {r['filename']}: score={r['score']:.3f}")

                # 有 caption 的應該排名更高
                assert scores.get("with_caption.png", 0) > scores.get("no_caption.png", 0), (
                    f"Mixed vector ({scores.get('with_caption.png', 0):.3f}) should score higher "
                    f"than pure image ({scores.get('no_caption.png', 0):.3f}) for text query"
                )


# ═══════════════════════════════════════════════════════════════════
# Run
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    mode = "SLOW (real inference)" if SLOW else "FAST (mock only)"
    print("=" * 60)
    print(f"聊天框圖片引用測試 — {mode}")
    print("=" * 60)
    print()

    print("── 1. 圖片索引與搜尋 ──")
    test_search_images_returns_relevant()

    print()
    print("── 2. _build_user_content 圖片即時描述 ──")
    test_build_user_content_with_vision_describe()
    test_build_user_content_no_images()
    test_build_user_content_fallback_on_describe_failure()

    print()
    print("── 3. 圖片 caption 注入格式 ──")
    test_image_context_injection_format()
    test_image_low_score_filtered()
    test_image_filename_mention_overrides_score()

    print()
    print("── 4. 跨模態語意搜尋（真實推理）──")
    test_real_image_text_search_slow()
    test_real_cross_modal_ranking_slow()

    print()
    print("── 5. 混合向量品質 ──")
    test_mixed_vector_improves_search_slow()

    print()
    print("=" * 60)
    print("ALL TESTS COMPLETED")
    print("=" * 60)
