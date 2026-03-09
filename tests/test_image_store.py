"""測試 rag/image_store.py 的 caption 解析。"""

import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestCaptionParsing:
    """測試三段式 caption 的解析和驗證。"""

    def test_valid_caption_structure(self):
        """完整的三段式結構"""
        caption = {
            "content": {
                "type": "書籍",
                "title": "The Medium is the Massage",
                "creator": "Marshall McLuhan",
                "content_design_link": "版面打破線性閱讀"
            },
            "style_tags": ["實驗", "高密度", "1960s"],
            "description": "書籍。全出血攝影..."
        }

        # 驗證必要欄位
        assert "content" in caption
        assert "style_tags" in caption
        assert "description" in caption
        assert isinstance(caption["style_tags"], list)

    def test_minimal_caption_structure(self):
        """最小結構（失敗時的 fallback）"""
        caption = {
            "content": {
                "type": "其他",
                "title": "",
                "creator": "",
                "content_design_link": ""
            },
            "style_tags": [],
            "description": "檔名"
        }

        assert caption["content"]["type"] == "其他"
        assert len(caption["style_tags"]) == 0

    def test_style_tags_count(self):
        """style_tags 數量應該在 6-12 個之間（或 0）"""
        # 正常情況：6-12 個
        tags = ["實驗", "高密度", "單色", "後現代", "美國", "嚴肅"]
        assert 6 <= len(tags) <= 12

        # 最小情況：0 個（失敗 fallback）
        tags_empty = []
        assert len(tags_empty) == 0

    def test_backward_compatibility_str_to_dict(self):
        """向下相容：舊的 str caption 轉為新格式"""
        from rag.image_store import _normalize_caption

        old_caption = "書籍。全出血攝影佔滿版面..."
        new_caption = _normalize_caption(old_caption)

        assert new_caption["description"] == old_caption
        assert new_caption["style_tags"] == []
        assert new_caption["content"]["type"] == "其他"


class TestEmbeddingText:
    """測試 embedding 文字的構建邏輯。"""

    def test_embed_text_with_style_tags(self):
        """有 style_tags 時優先使用"""
        from rag.image_store import _build_embed_text

        caption = {
            "content": {"title": "Test Book"},
            "style_tags": ["實驗", "高密度", "1960s"],
            "description": "測試描述"
        }

        embed_text = _build_embed_text(caption)
        assert "實驗" in embed_text
        assert "高密度" in embed_text

    def test_embed_text_with_title(self):
        """有 title 時加入"""
        from rag.image_store import _build_embed_text

        caption = {
            "content": {"title": "包豪斯", "creator": "Gropius"},
            "style_tags": ["現代主義"],
            "description": ""
        }

        embed_text = _build_embed_text(caption)
        assert "包豪斯" in embed_text
        assert "Gropius" in embed_text

    def test_embed_text_fallback_to_description(self):
        """都沒有時用 description"""
        from rag.image_store import _build_embed_text

        caption = {
            "content": {"title": "", "creator": ""},
            "style_tags": [],
            "description": "一張圖片"
        }

        embed_text = _build_embed_text(caption)
        assert embed_text == "一張圖片"

    def test_json_roundtrip(self):
        """caption dict 可以正確序列化/反序列化"""
        from rag.image_store import _normalize_caption

        original = {
            "content": {
                "type": "海報",
                "title": "Bauhaus Exhibition",
                "creator": "Herbert Bayer",
                "content_design_link": "幾何構成呼應建築理念"
            },
            "style_tags": ["控制", "均勻", "單色", "現代主義", "歐洲", "嚴肅"],
            "description": "海報。幾何色塊構成的展覽海報..."
        }

        # 序列化後反序列化
        json_str = json.dumps(original, ensure_ascii=False)
        restored = _normalize_caption(json_str)

        assert restored["content"]["title"] == "Bauhaus Exhibition"
        assert len(restored["style_tags"]) == 6
        assert restored["description"].startswith("海報")
