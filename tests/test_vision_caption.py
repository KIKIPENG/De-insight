"""測試真實 Vision LLM 生成三段式 caption。"""

import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


async def test_real_vision_caption():
    """用真實圖片測試 Vision LLM 生成。"""
    # 需要準備一張測試圖片
    test_image = Path(__file__).parent / "fixtures" / "test_book.jpg"

    if not test_image.exists():
        print(f"⚠ 測試圖片不存在: {test_image}")
        print("請手動準備一張圖片並重新測試")
        return

    # 設定環境變數
    if not os.getenv("VISION_MODEL"):
        print("⚠ VISION_MODEL 未設定，使用 LLM_MODEL")

    from rag.image_store import _auto_caption

    print("=" * 80)
    print("測試圖片:", test_image)
    print("=" * 80)

    image_bytes = test_image.read_bytes()
    caption = await _auto_caption(image_bytes, test_image.name)

    print("\n生成的 caption:")
    print("-" * 80)
    import json
    print(json.dumps(caption, ensure_ascii=False, indent=2))
    print("-" * 80)

    # 驗證結構
    assert isinstance(caption, dict), "caption 應該是 dict"
    assert "content" in caption, "缺少 content"
    assert "style_tags" in caption, "缺少 style_tags"
    assert "description" in caption, "缺少 description"

    # 驗證 content
    content = caption["content"]
    assert "type" in content, "content 缺少 type"

    # 驗證 style_tags
    tags = caption["style_tags"]
    assert isinstance(tags, list), "style_tags 應該是 list"
    if len(tags) > 0:
        assert 1 <= len(tags) <= 15, f"style_tags 數量異常: {len(tags)}"
        print(f"✓ style_tags 數量: {len(tags)}")

    # 驗證 description
    desc = caption["description"]
    assert isinstance(desc, str), "description 應該是 str"
    assert len(desc) > 0, "description 不能為空"
    print(f"✓ description 長度: {len(desc)} 字")

    print("\n✓ 所有驗證通過")


if __name__ == "__main__":
    asyncio.run(test_real_vision_caption())
