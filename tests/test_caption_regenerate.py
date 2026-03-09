"""測試批次重新生成 caption 功能（只測 fallback 篩選邏輯）。"""
import asyncio
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))


async def test_regenerate_import():
    """測試 regenerate_all_captions 可正常匯入。"""
    from rag.image_store import regenerate_all_captions, _normalize_caption
    print("✓ regenerate_all_captions 匯入成功")

    # 測試 _normalize_caption 邏輯
    fallback = _normalize_caption('{"description": "abc123"}')
    assert fallback["description"] == "abc123"
    assert isinstance(fallback["style_tags"], list)
    print("✓ _normalize_caption 正常")

    # 測試空專案不會 crash
    total, updated = await regenerate_all_captions(
        project_id="nonexistent_test_project_xyz",
        only_fallback=True,
    )
    assert total == 0
    assert updated == 0
    print("✓ 空專案回傳 (0, 0)")

    print("\n✓ 所有測試通過")


if __name__ == "__main__":
    asyncio.run(test_regenerate_import())
