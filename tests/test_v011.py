"""v0.11 測試：雙向量、偏好萃取、狀態透明化、搜尋注入格式。

測試策略：
- 所有 LLM 呼叫用 mock，不需要真的 API
- 所有 LanceDB 操作用臨時目錄，不碰真實資料
- 測試貼近使用場景：模擬使用者上傳圖片 → 搜尋 → 萃取偏好的完整流程
"""

import asyncio
import json
import math
import os
import shutil
import sys
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

import pytest


# ── Fixtures ─────────────────────────────────────────────────────

@pytest.fixture
def temp_project(tmp_path):
    """建立臨時專案目錄，monkey-patch image_store 的路徑函式。"""
    import rag.image_store as store

    images_dir = tmp_path / "images"
    images_dir.mkdir()
    lancedb_dir = tmp_path / "lancedb"

    orig_images = store._images_dir
    orig_lancedb = store._lancedb_dir
    store._images_dir = lambda pid: images_dir
    store._lancedb_dir = lambda pid: lancedb_dir
    # 清 db cache 避免跨測試污染
    store._db_cache.clear()

    yield {"root": tmp_path, "images": images_dir, "lancedb": lancedb_dir}

    store._images_dir = orig_images
    store._lancedb_dir = orig_lancedb
    store._db_cache.clear()


@pytest.fixture
def fake_image(temp_project):
    """在臨時目錄放一張 1x1 的測試 PNG。"""
    from PIL import Image
    img = Image.new("RGB", (64, 64), color=(128, 64, 32))
    path = temp_project["images"] / "test_book.png"
    img.save(path)
    return path


def _make_caption_dict(
    title="Test Book",
    creator="Author",
    style_tags=None,
    description="一本書的測試描述",
    content_type="書籍",
):
    """建立測試用的三段式 caption dict。"""
    return {
        "content": {
            "type": content_type,
            "title": title,
            "creator": creator,
            "content_design_link": "測試連結",
        },
        "style_tags": style_tags or ["粗糙", "實驗", "手工", "高密度", "單色", "當代"],
        "description": description,
    }


# ── A. Schema 與雙向量 ──────────────────────────────────────────

class TestDualVectorSchema:
    """Schema 正確包含雙向量欄位。"""

    def test_schema_fields(self):
        from rag.image_store import _make_schema, IMAGE_DIM
        schema = _make_schema()
        names = [f.name for f in schema]
        assert "vector" in names
        assert "vector_image" in names

    def test_schema_dimensions_match(self):
        from rag.image_store import _make_schema, IMAGE_DIM
        schema = _make_schema()
        for field in schema:
            if field.name in ("vector", "vector_image"):
                assert field.type.list_size == IMAGE_DIM


class TestIndexImageDualVector:
    """index_image 生成雙向量。"""

    @pytest.mark.asyncio
    async def test_index_produces_both_vectors(self, temp_project, fake_image):
        """上傳一張圖，確認 LanceDB 裡同時有文字向量和圖片向量。"""
        from rag.image_store import IMAGE_DIM

        # Mock embedding + vision — use different patterns so normalization doesn't collapse them
        fake_text_vec = [float(i % 7) / 7.0 for i in range(IMAGE_DIM)]
        fake_img_vec = [float(i % 11) / 11.0 for i in range(IMAGE_DIM)]
        caption = _make_caption_dict()

        with patch("rag.image_store._auto_caption", new_callable=AsyncMock, return_value=caption), \
             patch("rag.image_store._embed_text", new_callable=AsyncMock, return_value=fake_text_vec), \
             patch("rag.image_store._embed_image", new_callable=AsyncMock, return_value=fake_img_vec):

            from rag.image_store import index_image
            result = await index_image("test", fake_image.name)

        # 驗證回傳
        assert result["caption"]["style_tags"] == caption["style_tags"]

        # 直接讀 LanceDB 驗證雙向量
        import lancedb
        db = lancedb.connect(str(temp_project["lancedb"]))
        table = db.open_table("images")
        rows = table.to_pandas()
        assert len(rows) == 1

        row = rows.iloc[0]
        vec_text = list(row["vector"])
        vec_image = list(row["vector_image"])

        # 文字向量非零
        assert sum(abs(x) for x in vec_text) > 0.01, "文字向量不應該全零"
        # 圖片向量非零
        assert sum(abs(x) for x in vec_image) > 0.01, "圖片向量不應該全零"
        # 兩個向量不同
        assert vec_text[:5] != vec_image[:5], "文字向量和圖片向量不應相同"

    @pytest.mark.asyncio
    async def test_index_image_embed_failure_graceful(self, temp_project, fake_image):
        """圖片 embedding 失敗時，文字向量仍存入，圖片向量填零。"""
        from rag.image_store import IMAGE_DIM

        fake_text_vec = [0.1] * IMAGE_DIM
        caption = _make_caption_dict()

        with patch("rag.image_store._auto_caption", new_callable=AsyncMock, return_value=caption), \
             patch("rag.image_store._embed_text", new_callable=AsyncMock, return_value=fake_text_vec), \
             patch("rag.image_store._embed_image", new_callable=AsyncMock, side_effect=RuntimeError("GPU OOM")):

            from rag.image_store import index_image
            result = await index_image("test", fake_image.name)

        # 仍然成功
        assert result["filename"] == fake_image.name

        # 文字向量正常，圖片向量全零
        import lancedb
        db = lancedb.connect(str(temp_project["lancedb"]))
        row = db.open_table("images").to_pandas().iloc[0]
        assert sum(abs(x) for x in list(row["vector"])) > 0.01
        assert sum(abs(x) for x in list(row["vector_image"])) < 0.01


class TestSearchUsesTextVector:
    """搜尋只用文字向量（不受圖片向量影響）。"""

    @pytest.mark.asyncio
    async def test_search_uses_vector_not_vector_image(self, temp_project, fake_image):
        """搜尋用的是 vector 欄位（文字向量），不是 vector_image。"""
        from rag.image_store import IMAGE_DIM

        fake_text_vec = [0.5] * IMAGE_DIM
        fake_img_vec = [0.9] * IMAGE_DIM
        caption = _make_caption_dict(style_tags=["粗糙", "實驗"])

        with patch("rag.image_store._auto_caption", new_callable=AsyncMock, return_value=caption), \
             patch("rag.image_store._embed_text", new_callable=AsyncMock, return_value=fake_text_vec), \
             patch("rag.image_store._embed_image", new_callable=AsyncMock, return_value=fake_img_vec):
            from rag.image_store import index_image
            await index_image("test", fake_image.name)

        # 搜尋
        with patch("rag.image_store._embed_text", new_callable=AsyncMock, return_value=fake_text_vec):
            from rag.image_store import search_images
            results = await search_images("test", "粗糙 實驗", limit=3)

        assert len(results) >= 1
        assert results[0]["filename"] == fake_image.name
        # score 接近 1（因為查詢向量和文字向量完全相同）
        assert results[0]["score"] > 0.9


# ── B. Schema Migration（舊資料相容）──────────────────────────

class TestSchemaMigration:
    """舊的 images table（沒有 vector_image）可以自動 migration。"""

    def test_migration_adds_vector_image(self, temp_project):
        """模擬一個只有 vector 沒有 vector_image 的舊 table，確認 migration 成功。"""
        import lancedb
        import pyarrow as pa
        from rag.image_store import IMAGE_DIM

        # 建舊 schema（沒有 vector_image）
        old_schema = pa.schema([
            pa.field("id", pa.utf8()),
            pa.field("filename", pa.utf8()),
            pa.field("caption", pa.utf8()),
            pa.field("tags", pa.utf8()),
            pa.field("created_at", pa.utf8()),
            pa.field("vector", pa.list_(pa.float32(), IMAGE_DIM)),
        ])

        db = lancedb.connect(str(temp_project["lancedb"]))
        table = db.create_table("images", schema=old_schema)
        table.add([{
            "id": "old-001",
            "filename": "old_pic.jpg",
            "caption": "舊的純文字 caption",
            "tags": "[]",
            "created_at": "2024-01-01",
            "vector": [0.3] * IMAGE_DIM,
        }])

        # 關閉並重新用 _get_or_create_table 打開（觸發 migration）
        from rag.image_store import _get_db, _get_or_create_table, _db_cache
        _db_cache.clear()
        db2 = _get_db(temp_project["lancedb"])
        migrated_table = _get_or_create_table(db2)

        # 驗證 schema 更新
        field_names = [f.name for f in migrated_table.schema]
        assert "vector_image" in field_names, f"Migration failed, fields: {field_names}"

        # 驗證舊資料保留
        rows = migrated_table.to_pandas()
        assert len(rows) == 1
        assert rows.iloc[0]["filename"] == "old_pic.jpg"
        # 舊資料的 vector 保留
        assert sum(abs(x) for x in list(rows.iloc[0]["vector"])) > 0.01
        # vector_image 是零向量
        assert sum(abs(x) for x in list(rows.iloc[0]["vector_image"])) < 0.01


# ── C. Reindex ───────────────────────────────────────────────

class TestReindexAllImages:
    """reindex_all_images 為舊圖片補圖片向量。"""

    @pytest.mark.asyncio
    async def test_reindex_fills_zero_vectors(self, temp_project, fake_image):
        """已有文字向量但圖片向量全零的圖片，reindex 後應該有非零圖片向量。"""
        from rag.image_store import IMAGE_DIM

        # 先用零圖片向量索引
        fake_text_vec = [0.5] * IMAGE_DIM
        caption = _make_caption_dict()

        with patch("rag.image_store._auto_caption", new_callable=AsyncMock, return_value=caption), \
             patch("rag.image_store._embed_text", new_callable=AsyncMock, return_value=fake_text_vec), \
             patch("rag.image_store._embed_image", new_callable=AsyncMock, side_effect=RuntimeError("fail")):
            from rag.image_store import index_image
            await index_image("test", fake_image.name)

        # 確認圖片向量全零
        import lancedb
        db = lancedb.connect(str(temp_project["lancedb"]))
        before = db.open_table("images").to_pandas().iloc[0]
        assert sum(abs(x) for x in list(before["vector_image"])) < 0.01

        # Reindex
        fake_img_vec = [0.7] * IMAGE_DIM
        with patch("rag.image_store._embed_image", new_callable=AsyncMock, return_value=fake_img_vec):
            from rag.image_store import reindex_all_images, _db_cache
            _db_cache.clear()
            result = await reindex_all_images("test")

        assert result["updated"] == 1
        assert result["skipped"] == 0

        # 驗證圖片向量更新
        _db_cache.clear()
        db2 = lancedb.connect(str(temp_project["lancedb"]))
        after = db2.open_table("images").to_pandas().iloc[0]
        assert sum(abs(x) for x in list(after["vector_image"])) > 0.01

    @pytest.mark.asyncio
    async def test_reindex_skips_already_indexed(self, temp_project, fake_image):
        """已有非零圖片向量的圖片，reindex 應該跳過。"""
        from rag.image_store import IMAGE_DIM

        fake_text_vec = [0.5] * IMAGE_DIM
        fake_img_vec = [0.7] * IMAGE_DIM
        caption = _make_caption_dict()

        with patch("rag.image_store._auto_caption", new_callable=AsyncMock, return_value=caption), \
             patch("rag.image_store._embed_text", new_callable=AsyncMock, return_value=fake_text_vec), \
             patch("rag.image_store._embed_image", new_callable=AsyncMock, return_value=fake_img_vec):
            from rag.image_store import index_image
            await index_image("test", fake_image.name)

        # Reindex — 不應觸發 _embed_image
        mock_embed = AsyncMock(side_effect=RuntimeError("should not be called"))
        with patch("rag.image_store._embed_image", mock_embed):
            from rag.image_store import reindex_all_images, _db_cache
            _db_cache.clear()
            result = await reindex_all_images("test")

        assert result["skipped"] == 1
        assert result["updated"] == 0
        mock_embed.assert_not_called()


# ── D. 偏好萃取 ─────────────────────────────────────────────

class TestExtractVisualPreference:
    """模擬使用者上傳了多張圖片後，萃取視覺偏好。"""

    @pytest.mark.asyncio
    async def test_preference_needs_minimum_images(self, temp_project):
        """圖片不足 5 張時回傳 None。"""
        from rag.image_store import extract_visual_preference
        result = await extract_visual_preference("test")
        assert result is None

    @pytest.mark.asyncio
    async def test_preference_extracts_from_style_tags(self, temp_project, fake_image):
        """上傳 6 張有 style_tags 的圖片，萃取出詞頻和摘要。"""
        from rag.image_store import IMAGE_DIM, _db_cache

        fake_text_vec = [0.5] * IMAGE_DIM
        fake_img_vec = [0.7] * IMAGE_DIM

        # 製造 6 張圖片，style_tags 有重複
        tag_sets = [
            ["粗糙", "實驗", "手工", "高密度", "單色", "當代"],
            ["粗糙", "手工", "DIY", "不規則", "低飽和", "激進"],
            ["粗糙", "偶然", "手工", "稀疏", "冷色", "實驗"],
            ["精緻", "控制", "工業", "均勻", "高飽和", "商業"],
            ["粗糙", "實驗", "手工", "張力", "消色", "嚴肅"],
            ["手工", "DIY", "不規則", "低飽和", "在地", "激進"],
        ]

        from PIL import Image as PILImage
        for i, tags in enumerate(tag_sets):
            img = PILImage.new("RGB", (64, 64), color=(i * 30, i * 20, i * 10))
            path = temp_project["images"] / f"test_{i}.png"
            img.save(path)
            caption = _make_caption_dict(
                title=f"Test {i}",
                style_tags=tags,
                description=f"測試圖片 {i}",
            )
            with patch("rag.image_store._auto_caption", new_callable=AsyncMock, return_value=caption), \
                 patch("rag.image_store._embed_text", new_callable=AsyncMock, return_value=fake_text_vec), \
                 patch("rag.image_store._embed_image", new_callable=AsyncMock, return_value=fake_img_vec):
                from rag.image_store import index_image
                _db_cache.clear()
                await index_image("test", f"test_{i}.png")

        # 萃取偏好
        mock_llm = AsyncMock(return_value="他的收藏偏向粗糙手工的質感，大量使用非工業生產的材料和方法，呈現出一種對主流精緻設計的抵抗姿態。")

        _db_cache.clear()
        from rag.image_store import extract_visual_preference
        result = await extract_visual_preference("test", llm_call=mock_llm)

        # 驗證
        assert result is not None
        assert result["image_count"] == 6

        # 詞頻：「粗糙」和「手工」應該是最高頻的
        freq = result["tags_freq"]
        assert "粗糙" in freq, f"Expected '粗糙' in freq, got: {freq}"
        assert "手工" in freq, f"Expected '手工' in freq, got: {freq}"
        assert freq["粗糙"] >= 4, f"'粗糙' should appear >= 4 times, got {freq['粗糙']}"
        assert freq["手工"] >= 5, f"'手工' should appear >= 5 times, got {freq['手工']}"

        # 摘要非空
        assert len(result["summary"]) > 10

        # LLM 被呼叫了，且 prompt 包含詞頻資訊
        mock_llm.assert_called_once()
        prompt_arg = mock_llm.call_args[0][0]
        assert "粗糙" in prompt_arg
        assert "手工" in prompt_arg

    @pytest.mark.asyncio
    async def test_preference_works_without_llm(self, temp_project, fake_image):
        """不傳 llm_call 時只回傳詞頻，不生成摘要。"""
        from rag.image_store import IMAGE_DIM, _db_cache

        fake_vec = [0.5] * IMAGE_DIM

        from PIL import Image as PILImage
        for i in range(5):
            img = PILImage.new("RGB", (64, 64), color=(100, 100, 100))
            path = temp_project["images"] / f"nollm_{i}.png"
            img.save(path)
            caption = _make_caption_dict(style_tags=["極簡", "控制"])
            with patch("rag.image_store._auto_caption", new_callable=AsyncMock, return_value=caption), \
                 patch("rag.image_store._embed_text", new_callable=AsyncMock, return_value=fake_vec), \
                 patch("rag.image_store._embed_image", new_callable=AsyncMock, return_value=fake_vec):
                from rag.image_store import index_image
                _db_cache.clear()
                await index_image("test", f"nollm_{i}.png")

        _db_cache.clear()
        from rag.image_store import extract_visual_preference
        result = await extract_visual_preference("test", llm_call=None)

        assert result is not None
        assert result["summary"] == ""
        assert "極簡" in result["tags_freq"]


# ── E. 搜尋注入格式 ─────────────────────────────────────────

class TestSearchInjectionFormat:
    """搜尋結果注入 context 時，caption 是 dict 不應 crash。"""

    @pytest.mark.asyncio
    async def test_search_result_caption_is_dict(self, temp_project, fake_image):
        """search_images 回傳的 caption 是正規化的 dict，不是 raw JSON string。"""
        from rag.image_store import IMAGE_DIM, _db_cache

        fake_vec = [0.5] * IMAGE_DIM
        caption = _make_caption_dict()

        with patch("rag.image_store._auto_caption", new_callable=AsyncMock, return_value=caption), \
             patch("rag.image_store._embed_text", new_callable=AsyncMock, return_value=fake_vec), \
             patch("rag.image_store._embed_image", new_callable=AsyncMock, return_value=fake_vec):
            from rag.image_store import index_image
            await index_image("test", fake_image.name)

        with patch("rag.image_store._embed_text", new_callable=AsyncMock, return_value=fake_vec):
            _db_cache.clear()
            from rag.image_store import search_images
            results = await search_images("test", "粗糙")

        assert len(results) >= 1
        cap = results[0]["caption"]
        assert isinstance(cap, dict), f"Expected dict, got {type(cap)}"
        assert "style_tags" in cap
        assert "description" in cap

    def test_format_caption_for_injection(self):
        """模擬 _inject_rag_context 裡的圖片注入格式化，確認不 crash。"""
        from rag.image_store import _normalize_caption
        cap = _normalize_caption({
            "content": {"type": "書籍", "title": "Test", "creator": "A"},
            "style_tags": ["粗糙", "實驗"],
            "description": "一本書的描述",
        })
        # 模擬 chat.py 裡的格式化邏輯
        desc = cap.get("description", "")
        tags = cap.get("style_tags", [])
        content = cap.get("content", {})
        title = content.get("title", "")

        line_parts = ["[圖片] test.jpg"]
        if title:
            line_parts.append(f"（{title}）")
        if tags:
            line_parts.append(f"風格：{', '.join(tags[:5])}")
        if desc:
            line_parts.append(desc[:120])

        result = " | ".join(line_parts)
        assert "Test" in result
        assert "粗糙" in result
        assert "一本書的描述" in result


# ── F. 狀態透明化 ───────────────────────────────────────────

class TestMenuBarStatus:
    """MenuBar 接受並渲染新的狀態參數。"""

    def test_set_system_status_new_params(self):
        """set_system_status 接受 rag_llm_ok 和 vision_ok 參數。"""
        from widgets import MenuBar
        bar = MenuBar(id="test-bar")
        # Patch _rebuild to avoid Textual NoActiveAppError in unit test
        with patch.object(bar, "_rebuild"):
            bar.set_system_status(
                llm_model="openai/deepseek",
                llm_ok=True,
                embed_label="jina-v4",
                embed_ok=True,
                rag_llm_ok=True,
                vision_ok=True,
            )
        assert bar._rag_llm_ok is True
        assert bar._vision_ok is True

    def test_status_degraded(self):
        """RAG LLM 和 Vision 不可用時，狀態被正確記錄。"""
        from widgets import MenuBar
        bar = MenuBar(id="test-bar2")
        with patch.object(bar, "_rebuild"):
            bar.set_system_status(
                llm_model="openai/deepseek",
                llm_ok=True,
                embed_ok=True,
                rag_llm_ok=False,
                vision_ok=False,
            )
        assert bar._rag_llm_ok is False
        assert bar._vision_ok is False


# ── G. update_image 重算雙向量 ──────────────────────────────

class TestUpdateImageRecalcDualVector:
    """編輯 caption 時重算雙向量。"""

    @pytest.mark.asyncio
    async def test_recalc_updates_both_vectors(self, temp_project, fake_image):
        """recalc_vector=True 時，文字向量和圖片向量都重算。"""
        from rag.image_store import IMAGE_DIM, _db_cache

        vec_a = [0.3] * IMAGE_DIM
        vec_b = [0.6] * IMAGE_DIM
        vec_img_a = [0.2] * IMAGE_DIM
        vec_img_b = [0.8] * IMAGE_DIM
        caption = _make_caption_dict()

        # 初次索引
        with patch("rag.image_store._auto_caption", new_callable=AsyncMock, return_value=caption), \
             patch("rag.image_store._embed_text", new_callable=AsyncMock, return_value=vec_a), \
             patch("rag.image_store._embed_image", new_callable=AsyncMock, return_value=vec_img_a):
            from rag.image_store import index_image
            result = await index_image("test", fake_image.name)

        image_id = result["id"]

        # 更新 caption + recalc
        new_caption = _make_caption_dict(style_tags=["精緻", "控制", "工業"])
        with patch("rag.image_store._embed_text", new_callable=AsyncMock, return_value=vec_b), \
             patch("rag.image_store._embed_image", new_callable=AsyncMock, return_value=vec_img_b):
            _db_cache.clear()
            from rag.image_store import update_image
            ok = await update_image("test", image_id, caption=new_caption, recalc_vector=True)

        assert ok is True

        # 驗證向量都更新了
        import lancedb
        _db_cache.clear()
        db = lancedb.connect(str(temp_project["lancedb"]))
        row = db.open_table("images").to_pandas().iloc[0]
        # 向量已改變（不再是 vec_a / vec_img_a）
        new_text = list(row["vector"])
        new_img = list(row["vector_image"])
        assert abs(new_text[0] - vec_a[0]) > 0.01, "文字向量應該已更新"
        assert abs(new_img[0] - vec_img_a[0]) > 0.01, "圖片向量應該已更新"


# ── H. 完整使用流程（整合測試）─────────────────────────────

class TestEndToEndFlow:
    """模擬完整使用場景：上傳 → 搜尋 → 偏好萃取。"""

    @pytest.mark.asyncio
    async def test_upload_search_preference_flow(self, temp_project):
        """
        場景：使用者上傳 6 張設計書籍照片，用文字搜尋，然後系統萃取偏好。
        """
        from rag.image_store import IMAGE_DIM, _db_cache
        from PIL import Image as PILImage

        fake_text_vec = [0.5] * IMAGE_DIM
        fake_img_vec = [0.7] * IMAGE_DIM

        # 使用者上傳 6 張圖，模擬不同風格
        uploads = [
            ("bauhaus_poster.jpg", ["控制", "均勻", "單色", "現代主義", "歐洲", "嚴肅"]),
            ("punk_zine.jpg", ["粗糙", "不規則", "單色", "後現代", "美國", "激進"]),
            ("muji_catalog.jpg", ["精緻", "留白", "消色", "當代", "日本", "中性"]),
            ("soviet_poster.jpg", ["控制", "填滿", "高飽和", "現代主義", "歐洲", "嚴肅"]),
            ("riso_print.jpg", ["手工", "不規則", "高飽和", "當代", "在地", "實驗"]),
            ("swiss_grid.jpg", ["控制", "均勻", "單色", "現代主義", "歐洲", "嚴肅"]),
        ]

        for filename, tags in uploads:
            img = PILImage.new("RGB", (64, 64))
            img.save(temp_project["images"] / filename)

            caption = _make_caption_dict(title=filename.split(".")[0], style_tags=tags)
            with patch("rag.image_store._auto_caption", new_callable=AsyncMock, return_value=caption), \
                 patch("rag.image_store._embed_text", new_callable=AsyncMock, return_value=fake_text_vec), \
                 patch("rag.image_store._embed_image", new_callable=AsyncMock, return_value=fake_img_vec):
                from rag.image_store import index_image
                _db_cache.clear()
                await index_image("test", filename)

        # 搜尋
        with patch("rag.image_store._embed_text", new_callable=AsyncMock, return_value=fake_text_vec):
            _db_cache.clear()
            from rag.image_store import search_images
            results = await search_images("test", "現代主義 控制 網格", limit=3)

        assert len(results) >= 1
        for r in results:
            assert isinstance(r["caption"], dict)

        # 偏好萃取
        mock_llm = AsyncMock(return_value="他偏好結構性強、帶有現代主義傳統的設計。")
        _db_cache.clear()
        from rag.image_store import extract_visual_preference
        pref = await extract_visual_preference("test", llm_call=mock_llm)

        assert pref is not None
        assert pref["image_count"] == 6
        # 「控制」出現 3 次（bauhaus, soviet, swiss），「嚴肅」也 3 次
        assert pref["tags_freq"].get("控制", 0) >= 3
        assert pref["tags_freq"].get("嚴肅", 0) >= 3
        assert len(pref["summary"]) > 10


# ── I. 現有測試不退化 ───────────────────────────────────────

class TestBackwardCompatibility:
    """確認 v0.10.3 的功能沒被破壞。"""

    def test_normalize_caption_old_str(self):
        from rag.image_store import _normalize_caption
        cap = _normalize_caption("舊的純文字 caption")
        assert isinstance(cap, dict)
        assert cap["description"] == "舊的純文字 caption"
        assert cap["style_tags"] == []

    def test_normalize_caption_json_str(self):
        from rag.image_store import _normalize_caption
        d = {"content": {"type": "書籍", "title": "X"}, "style_tags": ["粗糙"], "description": "desc"}
        cap = _normalize_caption(json.dumps(d, ensure_ascii=False))
        assert cap["style_tags"] == ["粗糙"]

    def test_build_embed_text_empty(self):
        from rag.image_store import _build_embed_text
        cap = {"content": {}, "style_tags": [], "description": "fallback"}
        assert _build_embed_text(cap) == "fallback"

    def test_parse_caption_valid(self):
        from rag.image_store import _parse_caption
        raw = json.dumps({
            "content": {"type": "海報", "title": "X", "creator": "Y", "content_design_link": "Z"},
            "style_tags": ["精緻"],
            "description": "desc",
        })
        result = _parse_caption(raw)
        assert result["content"]["type"] == "海報"
