"""端到端測試：上傳圖片 → 生成 caption → 向量索引 → 搜尋。"""

import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


async def test_full_pipeline():
    """完整流程測試。"""
    test_image = Path(__file__).parent / "fixtures" / "test_book.jpg"
    if not test_image.exists():
        print("⚠ 測試圖片不存在，跳過")
        return

    from rag.image_store import add_image, search_images, list_images

    # 設定測試用的 lancedb_dir（使用 temp project）
    import tempfile
    import shutil
    test_dir = Path(tempfile.mkdtemp(prefix="test_deinsight_"))

    # Monkey-patch project_root and _images_dir for test
    import rag.image_store as store
    original_images_dir = store._images_dir
    original_lancedb_dir = store._lancedb_dir

    test_images = test_dir / "images"
    test_images.mkdir(parents=True, exist_ok=True)
    test_lancedb = test_dir / "lancedb"

    store._images_dir = lambda pid: test_images
    store._lancedb_dir = lambda pid: test_lancedb

    try:
        # 複製測試圖片到測試目錄
        import shutil as _shutil
        _shutil.copy2(test_image, test_images / test_image.name)

        print("=" * 80)
        print("步驟 1：索引圖片並生成 caption")
        print("=" * 80)

        result = await store.index_image("test", test_image.name)
        caption = result["caption"]

        print("✓ caption 生成完成")
        print(f"  type: {caption['content']['type']}")
        print(f"  style_tags: {caption['style_tags'][:5]}")
        desc = caption['description']
        print(f"  description: {desc[:50]}..." if len(desc) > 50 else f"  description: {desc}")

        print("\n" + "=" * 80)
        print("步驟 2：列出所有圖片")
        print("=" * 80)

        all_images = await list_images("test")
        print(f"✓ 找到 {len(all_images)} 張圖片")

        assert len(all_images) > 0, "應該至少有一張圖片"
        # 驗證 caption 是 dict
        assert isinstance(all_images[0]["caption"], dict), "list_images 的 caption 應該是 dict"

        print("\n" + "=" * 80)
        print("步驟 3：向量搜尋")
        print("=" * 80)

        # 搜尋 style_tags 裡的關鍵字
        if caption["style_tags"]:
            query = caption["style_tags"][0]
            print(f"搜尋關鍵字: {query}")

            results = await search_images("test", query, limit=3)
            print(f"✓ 找到 {len(results)} 個結果")

            if results:
                print(f"  第一個結果 score: {results[0]['score']:.3f}")
                # 驗證 caption 是 dict
                assert isinstance(results[0]["caption"], dict), "search 結果的 caption 應該是 dict"

        print("\n✓ 所有步驟通過")

    finally:
        # 還原
        store._images_dir = original_images_dir
        store._lancedb_dir = original_lancedb_dir
        # 清理
        shutil.rmtree(test_dir, ignore_errors=True)


if __name__ == "__main__":
    asyncio.run(test_full_pipeline())
