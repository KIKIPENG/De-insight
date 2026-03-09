"""測試編輯 caption API。"""
import asyncio
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import httpx


async def test_edit_caption():
    """測試編輯 caption API。"""
    async with httpx.AsyncClient() as client:
        # 1. 取得圖片列表
        resp = await client.get("http://localhost:8000/api/images")
        if resp.status_code != 200:
            print(f"⚠ API 無法連線 ({resp.status_code})，跳過")
            return
        data = resp.json()
        imgs = data.get("images", [])

        if not imgs:
            print("⚠ 沒有圖片，跳過測試")
            return

        img = imgs[0]
        img_id = img["id"]
        print(f"測試圖片：{img['filename']}, id={img_id}")

        # 2. 編輯 caption
        old_caption = img.get("caption", {})
        if not isinstance(old_caption, dict):
            old_caption = {"content": {}, "style_tags": [], "description": str(old_caption)}

        new_caption = {
            "content": old_caption.get("content", {}),
            "style_tags": old_caption.get("style_tags", []) + ["測試標籤"],
            "description": old_caption.get("description", ""),
        }

        old_tags = old_caption.get("style_tags", [])
        print(f"舊標籤：{old_tags}")
        print(f"新標籤：{new_caption['style_tags']}")

        # 3. 送出
        resp = await client.put(
            f"http://localhost:8000/api/images/{img_id}/caption",
            json=new_caption,
        )
        print(f"API 回應：{resp.status_code}")
        assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"

        # 4. 驗證
        resp = await client.get("http://localhost:8000/api/images")
        updated_img = next(
            (i for i in resp.json()["images"] if i["id"] == img_id), None
        )
        assert updated_img is not None, "Image not found after update"

        updated_tags = updated_img["caption"].get("style_tags", [])
        print(f"更新後標籤：{updated_tags}")
        assert "測試標籤" in updated_tags, f"測試標籤 not found in {updated_tags}"

        # 5. 還原（移除測試標籤）
        restore_caption = {
            "content": old_caption.get("content", {}),
            "style_tags": old_tags,
            "description": old_caption.get("description", ""),
        }
        await client.put(
            f"http://localhost:8000/api/images/{img_id}/caption",
            json=restore_caption,
        )
        print("✓ Caption 編輯成功（已還原）")


if __name__ == "__main__":
    asyncio.run(test_edit_caption())
