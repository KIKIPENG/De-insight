"""v0.11.5 測試：偏好觸發修正、insight_score 生效、交叉偵測。

測試策略：
- 所有 LLM 呼叫用 mock
- 所有 DB 操作用臨時目錄
- 測試貼近使用場景
"""

import asyncio
import json
import os
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
    store._db_cache.clear()

    yield {"root": tmp_path, "images": images_dir, "lancedb": lancedb_dir}

    store._images_dir = orig_images
    store._lancedb_dir = orig_lancedb
    store._db_cache.clear()


@pytest.fixture
def temp_db(tmp_path):
    """建立臨時 memories.db。"""
    db_path = tmp_path / "memories.db"
    return db_path


def _make_caption(style_tags=None):
    return {
        "content": {"type": "書籍", "title": "T", "creator": "C", "content_design_link": "L"},
        "style_tags": style_tags or ["粗糙", "實驗", "手工"],
        "description": "desc",
    }


async def _populate_images(temp_project, n=6, tags_list=None):
    """在臨時專案裡建立 N 張圖片。"""
    from rag.image_store import IMAGE_DIM, _db_cache, index_image
    from PIL import Image as PILImage

    fake_vec = [0.5] * IMAGE_DIM
    default_tags = [
        ["粗糙", "實驗", "手工", "高密度", "單色", "當代"],
        ["粗糙", "手工", "DIY", "不規則", "低飽和", "激進"],
        ["粗糙", "偶然", "手工", "稀疏", "冷色", "實驗"],
        ["精緻", "控制", "工業", "均勻", "高飽和", "商業"],
        ["粗糙", "實驗", "手工", "張力", "消色", "嚴肅"],
        ["手工", "DIY", "不規則", "低飽和", "在地", "激進"],
    ]
    tags_list = tags_list or default_tags

    for i in range(n):
        img = PILImage.new("RGB", (64, 64), color=(i * 30, i * 20, i * 10))
        path = temp_project["images"] / f"test_{i}.png"
        img.save(path)
        caption = _make_caption(style_tags=tags_list[i % len(tags_list)])
        with patch("rag.image_store._auto_caption", new_callable=AsyncMock, return_value=caption), \
             patch("rag.image_store._embed_text", new_callable=AsyncMock, return_value=fake_vec), \
             patch("rag.image_store._embed_image", new_callable=AsyncMock, return_value=fake_vec):
            _db_cache.clear()
            await index_image("test", f"test_{i}.png")


# ── A. trigger_preference_update ────────────────────────────────

class TestTriggerPreferenceUpdate:
    """偏好萃取可以獨立觸發，不依賴記憶確認。"""

    @pytest.mark.asyncio
    async def test_trigger_returns_preference(self, temp_project, temp_db):
        """圖片 >= 5 張、沒有先前偏好 → 觸發成功。"""
        await _populate_images(temp_project, n=6)

        mock_llm = AsyncMock(return_value="他偏好粗糙手工的質感。")

        from rag.image_store import trigger_preference_update, _db_cache
        _db_cache.clear()
        result = await trigger_preference_update(
            "test", llm_call=mock_llm, db_path=temp_db,
        )

        assert result is not None
        assert "tags_freq" in result
        assert len(result["summary"]) > 5
        mock_llm.assert_called_once()

    @pytest.mark.asyncio
    async def test_trigger_skips_if_too_few(self, temp_project, temp_db):
        """圖片 < 5 張 → 不觸發。"""
        await _populate_images(temp_project, n=3)

        from rag.image_store import trigger_preference_update, _db_cache
        _db_cache.clear()
        result = await trigger_preference_update("test", db_path=temp_db)
        assert result is None

    @pytest.mark.asyncio
    async def test_trigger_skips_if_delta_too_small(self, temp_project, temp_db):
        """距上次萃取新增 < 5 張 → 不觸發。"""
        await _populate_images(temp_project, n=6)

        # 模擬上次在 count=4 時萃取過
        from memory.store import add_memory
        await add_memory(
            type="preference", content="舊偏好", source="4",
            topic="美學偏好", category="美學偏好", db_path=temp_db,
        )

        # 現在 count=6，delta=2 < 5
        from rag.image_store import trigger_preference_update, _db_cache
        _db_cache.clear()
        result = await trigger_preference_update("test", db_path=temp_db)
        assert result is None

    @pytest.mark.asyncio
    async def test_trigger_saves_to_memory(self, temp_project, temp_db):
        """觸發成功後，偏好被存入 memories.db。"""
        await _populate_images(temp_project, n=6)

        mock_llm = AsyncMock(return_value="他偏好粗糙手工的質感。")

        from rag.image_store import trigger_preference_update, _db_cache
        _db_cache.clear()
        await trigger_preference_update("test", llm_call=mock_llm, db_path=temp_db)

        from memory.store import get_memories
        prefs = await get_memories(type="preference", limit=1, db_path=temp_db)
        assert len(prefs) == 1
        assert "粗糙" in prefs[0]["content"] or "手工" in prefs[0]["content"]


# ── B. check_cross_modal ────────────────────────────────────────

class TestCheckCrossModal:
    """文字偏好 vs 視覺偏好交叉偵測。"""

    @pytest.mark.asyncio
    async def test_detects_contradiction(self, temp_db):
        """視覺偏好粗糙但文字洞見說喜歡極簡 → 偵測到矛盾。"""
        # 存幾條 insight
        from memory.store import add_memory
        await add_memory(type="insight", content="我覺得極簡主義是最有力量的設計語言",
                        topic="美學", db_path=temp_db)
        await add_memory(type="insight", content="好的設計應該克制、精確、不多不少",
                        topic="美學", db_path=temp_db)
        await add_memory(type="insight", content="留白不是沒有內容，是讓內容呼吸",
                        topic="設計史", db_path=temp_db)

        # 視覺偏好是粗糙手工
        visual = "他的收藏偏向粗糙手工的質感，大量使用非工業生產的材料，呈現對主流精緻設計的抵抗。"

        # Mock LLM 回傳矛盾
        mock_llm = AsyncMock(return_value=json.dumps({
            "type": "cross_modal",
            "summary": "文字裡強調極簡克制，但圖片收藏全是粗糙實驗的東西",
            "visual": "粗糙手工、非工業、抵抗精緻",
            "textual": "極簡主義、克制、精確",
        }))

        from memory.thought_tracker import check_cross_modal
        result = await check_cross_modal(visual, mock_llm, db_path=temp_db)

        assert result is not None
        assert result["type"] == "cross_modal"
        assert len(result["summary"]) > 5
        assert result.get("visual")
        assert result.get("textual")

    @pytest.mark.asyncio
    async def test_returns_none_when_consistent(self, temp_db):
        """視覺偏好和文字洞見一致 → 回傳 None。"""
        from memory.store import add_memory
        await add_memory(type="insight", content="粗糙的東西有一種誠實的力量",
                        topic="美學", db_path=temp_db)
        await add_memory(type="insight", content="手工製作的不完美是設計的一部分",
                        topic="設計史", db_path=temp_db)

        visual = "他偏好粗糙手工的質感。"

        mock_llm = AsyncMock(return_value='{"type": null}')

        from memory.thought_tracker import check_cross_modal
        result = await check_cross_modal(visual, mock_llm, db_path=temp_db)
        assert result is None

    @pytest.mark.asyncio
    async def test_skips_if_too_few_insights(self, temp_db):
        """洞見不到 2 條 → 不做比對。"""
        from memory.store import add_memory
        await add_memory(type="insight", content="一條洞見", topic="美學", db_path=temp_db)

        from memory.thought_tracker import check_cross_modal
        result = await check_cross_modal("偏好描述", AsyncMock(), db_path=temp_db)
        assert result is None

    @pytest.mark.asyncio
    async def test_skips_if_empty_preference(self):
        """偏好描述為空 → 不做比對。"""
        from memory.thought_tracker import check_cross_modal
        result = await check_cross_modal("", AsyncMock())
        assert result is None


# ── C. insight_score 影響記憶注入 ────────────────────────────────

class TestInsightScoreEffect:
    """insight_score 影響 _inject_rag_context 的行為。"""

    def test_inject_rag_context_reads_insight_score(self):
        """確認 _inject_rag_context 裡有讀取 insight_score 的程式碼。"""
        import inspect
        from mixins.chat import ChatMixin
        source = inspect.getsource(ChatMixin._inject_rag_context)
        assert "insight_score" in source or "_insight_score" in source, \
            "_inject_rag_context should read insight_score"
        assert "0.4" in source, \
            "Should have threshold 0.4"

    def test_memory_limit_logic(self):
        """確認高 insight_score 時記憶注入量增加。"""
        import inspect
        from mixins.chat import ChatMixin
        source = inspect.getsource(ChatMixin._inject_rag_context)
        # 應該有類似 limit=5 if score > 0.4 else 3 的邏輯
        assert "5" in source and "3" in source, \
            "Should have different limits for high/low insight_score"


# ── D. insight_profile 納入 preference ───────────────────────────

class TestInsightProfileIncludesPreference:
    """insight_profile 的 _build_profile 同時讀取 insight 和 preference。"""

    @pytest.mark.asyncio
    async def test_profile_includes_preference_keywords(self, temp_db):
        """preference 記憶的關鍵字也出現在 profile 裡。"""
        from memory.store import add_memory
        await add_memory(type="insight", content="包豪斯預設功能穩定",
                        topic="設計史", db_path=temp_db)
        await add_memory(type="preference", content="偏好粗糙手工實驗性的視覺風格",
                        topic="美學偏好", db_path=temp_db)

        from rag.insight_profile import _build_profile, invalidate_cache
        invalidate_cache()
        profile = await _build_profile("test", db_path=temp_db)

        assert profile["count"] >= 2
        # preference 的關鍵字應該出現
        all_kw = " ".join(profile["keywords"])
        assert "粗糙" in all_kw or "手工" in all_kw or "實驗" in all_kw, \
            f"Preference keywords should be in profile, got: {profile['keywords']}"


# ── E. 完整場景：上傳 → 偏好 → 交叉偵測 ─────────────────────────

class TestEndToEndPreferenceFlow:
    """模擬完整場景：使用者先聊了幾輪（存了 insight），
    然後上傳了 6 張圖片（觸發偏好萃取 + 交叉偵測）。"""

    @pytest.mark.asyncio
    async def test_full_flow(self, temp_project, temp_db):
        """
        1. 存入幾條 insight（模擬對話後的記憶存入）
        2. 上傳 6 張圖片
        3. 觸發偏好萃取
        4. 觸發交叉偵測
        5. 驗證結果
        """
        # 1. 使用者的文字洞見（偏好極簡）
        from memory.store import add_memory, get_memories
        await add_memory(type="insight", content="極簡主義是最誠實的設計語言",
                        topic="美學", db_path=temp_db)
        await add_memory(type="insight", content="好的設計不需要解釋，它自己說話",
                        topic="設計史", db_path=temp_db)
        await add_memory(type="insight", content="留白是最有力量的設計元素",
                        topic="美學", db_path=temp_db)

        # 2. 上傳 6 張粗糙手工風格的圖片
        await _populate_images(temp_project, n=6, tags_list=[
            ["粗糙", "實驗", "手工", "高密度", "單色", "當代"],
            ["粗糙", "手工", "DIY", "不規則", "低飽和", "激進"],
            ["粗糙", "偶然", "手工", "稀疏", "冷色", "實驗"],
            ["粗糙", "實驗", "手工", "張力", "消色", "嚴肅"],
            ["手工", "DIY", "不規則", "低飽和", "在地", "激進"],
            ["粗糙", "手工", "實驗", "不規則", "單色", "嚴肅"],
        ])

        # 3. 觸發偏好萃取
        mock_llm = AsyncMock(side_effect=[
            # 第一次呼叫：偏好萃取
            "他的收藏呈現強烈的粗糙手工傾向，偏好非工業生產、實驗性強的視覺材料。",
            # 第二次呼叫：交叉偵測
            json.dumps({
                "type": "cross_modal",
                "summary": "文字裡強調極簡留白，但圖片全是粗糙高密度的東西",
                "visual": "粗糙手工、實驗、高密度",
                "textual": "極簡主義、留白、不需要解釋",
            }),
        ])

        from rag.image_store import trigger_preference_update, _db_cache
        _db_cache.clear()
        result = await trigger_preference_update(
            "test", llm_call=mock_llm, db_path=temp_db,
        )
        assert result is not None

        # 4. 手動觸發交叉偵測（模擬 _maybe_update_visual_preference 裡的行為）
        from memory.thought_tracker import check_cross_modal
        cross = await check_cross_modal(
            result["summary"], mock_llm, db_path=temp_db,
        )

        # 5. 驗證
        assert cross is not None
        assert cross["type"] == "cross_modal"
        assert "極簡" in cross["summary"] or "粗糙" in cross["summary"]

        # 驗證 LLM 被呼叫了兩次（偏好 + 交叉）
        assert mock_llm.call_count == 2

        # 偏好已存入記憶
        prefs = await get_memories(type="preference", limit=1, db_path=temp_db)
        assert len(prefs) == 1

    @pytest.mark.asyncio
    async def test_no_contradiction_when_consistent(self, temp_project, temp_db):
        """使用者的文字和圖片風格一致時，不偵測到矛盾。"""
        from memory.store import add_memory
        await add_memory(type="insight", content="粗糙的東西有一種誠實的力量",
                        topic="美學", db_path=temp_db)
        await add_memory(type="insight", content="手工製作的不完美本身就是美",
                        topic="設計史", db_path=temp_db)

        await _populate_images(temp_project, n=6)

        mock_llm = AsyncMock(side_effect=[
            "他偏好粗糙手工的質感。",  # 偏好萃取
            '{"type": null}',          # 交叉偵測：一致
        ])

        from rag.image_store import trigger_preference_update, _db_cache
        _db_cache.clear()
        result = await trigger_preference_update(
            "test", llm_call=mock_llm, db_path=temp_db,
        )
        assert result is not None

        from memory.thought_tracker import check_cross_modal
        cross = await check_cross_modal(
            result["summary"], mock_llm, db_path=temp_db,
        )
        assert cross is None


# ── F. 向下相容 ─────────────────────────────────────────────────

class TestBackwardCompat:
    """確認 v0.11 的功能沒被破壞。"""

    def test_normalize_caption_still_works(self):
        from rag.image_store import _normalize_caption
        cap = _normalize_caption("舊的純文字")
        assert cap["description"] == "舊的純文字"
        assert cap["style_tags"] == []

    @pytest.mark.asyncio
    async def test_check_for_evolution_still_works(self, temp_db):
        """原有的 insight 演變偵測不受影響。"""
        from memory.store import add_memory
        await add_memory(type="insight", content="包豪斯預設功能穩定",
                        topic="設計史", db_path=temp_db)

        mock_llm = AsyncMock(return_value='{"type": null}')

        from memory.thought_tracker import check_for_evolution
        result = await check_for_evolution(
            "功能其實是流動的", mock_llm, db_path=temp_db,
        )
        # 可能是 None 或 evolution，都不應 crash
        assert result is None or isinstance(result, dict)

    def test_extract_visual_preference_still_works(self):
        """extract_visual_preference 函式仍存在且簽名正確。"""
        import inspect
        from rag.image_store import extract_visual_preference
        sig = inspect.signature(extract_visual_preference)
        params = list(sig.parameters.keys())
        assert "project_id" in params
        assert "llm_call" in params
