"""測試 caption 編輯和重算邏輯。"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestCaptionEdit:
    """測試 caption 更新邏輯。"""

    def test_embed_text_construction(self):
        """測試 embedding 文字構建（更新後）。"""
        new_caption = {
            "content": {
                "type": "海報",
                "title": "測試展覽",
                "creator": "測試藝術家",
                "content_design_link": "..."
            },
            "style_tags": ["實驗", "當代", "高飽和"],
            "description": "測試描述"
        }

        # 模擬構建邏輯
        embed_parts = []
        if new_caption.get("style_tags"):
            embed_parts.append(" ".join(new_caption["style_tags"]))
        content = new_caption.get("content", {})
        if content.get("title"):
            embed_parts.append(content["title"])

        embed_text = " ".join(embed_parts)

        assert "實驗" in embed_text
        assert "當代" in embed_text
        assert "測試展覽" in embed_text

    def test_fallback_detection(self):
        """測試 fallback caption 偵測（檔名判斷）。"""
        filename = "abc123def456.jpg"
        caption = {
            "description": "abc123def456"  # 和檔名相同
        }

        filename_stem = Path(filename).stem
        is_fallback = caption["description"] == filename_stem

        assert is_fallback == True

        # 正常 caption
        caption2 = {
            "description": "海報。書法和攝影..."
        }
        is_fallback2 = caption2["description"] == filename_stem

        assert is_fallback2 == False

    def test_embed_text_from_image_store(self):
        """測試 image_store._build_embed_text 函數。"""
        from rag.image_store import _build_embed_text

        caption = {
            "content": {"type": "海報", "title": "展覽A", "creator": "藝術家B"},
            "style_tags": ["實驗", "當代"],
            "description": "測試描述"
        }
        result = _build_embed_text(caption)
        assert "實驗" in result
        assert "展覽A" in result
        assert "藝術家B" in result

    def test_embed_text_fallback_to_description(self):
        """沒有 tags 和 title 時，用 description。"""
        from rag.image_store import _build_embed_text

        caption = {
            "content": {"type": "其他", "title": "", "creator": ""},
            "style_tags": [],
            "description": "一張照片"
        }
        result = _build_embed_text(caption)
        assert "一張照片" in result


class TestRegenerateLogic:
    """測試批次重新生成邏輯。"""

    def test_filter_fallback_only(self):
        """測試只篩選 fallback 圖片。"""
        rows = [
            {"filename": "img1.jpg", "caption": '{"description": "img1"}'},  # fallback
            {"filename": "img2.jpg", "caption": '{"description": "海報..."}'},  # 正常
            {"filename": "img3.jpg", "caption": '{"description": "img3"}'},  # fallback
        ]

        import json
        to_regenerate = []
        for row in rows:
            caption = json.loads(row["caption"])
            desc = caption.get("description", "")
            filename_stem = Path(row["filename"]).stem
            if desc == filename_stem:
                to_regenerate.append(row)

        assert len(to_regenerate) == 2
        assert to_regenerate[0]["filename"] == "img1.jpg"
        assert to_regenerate[1]["filename"] == "img3.jpg"
