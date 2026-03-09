import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import lancedb
import json
from paths import DATA_ROOT

# 找 LanceDB
projects_dir = DATA_ROOT / "projects"
lancedb_dirs = list(projects_dir.glob("*/lancedb"))

if not lancedb_dirs:
    print("找不到任何 LanceDB")
    sys.exit(1)

for lancedb_dir in lancedb_dirs:
    print(f"LanceDB: {lancedb_dir}\n")

    db = lancedb.connect(str(lancedb_dir))

    if "images" not in db.table_names():
        print("❌ images table 不存在\n")
        continue

    tbl = db.open_table("images")
    count = tbl.count_rows()
    print(f"總共 {count} 張圖片\n")

    if count == 0:
        print("❌ table 是空的\n")
        continue

    df = tbl.to_pandas()

    print("=" * 80)
    print("檢查每張圖片的 caption：")
    print("=" * 80)

    for idx, row in df.iterrows():
        filename = row['filename']
        caption_raw = row['caption']

        print(f"\n[{idx + 1}] {filename}")
        print(f"    caption type: {type(caption_raw).__name__}")
        print(f"    caption length: {len(caption_raw) if isinstance(caption_raw, str) else 'N/A'}")

        try:
            if isinstance(caption_raw, str):
                caption = json.loads(caption_raw)
            else:
                caption = caption_raw

            has_content = "content" in caption
            has_tags = "style_tags" in caption
            has_desc = "description" in caption

            print(f"    三段式結構: content={has_content}, style_tags={has_tags}, description={has_desc}")

            if has_desc:
                desc = caption["description"]
                stem = Path(filename).stem
                is_fallback = desc == stem

                if is_fallback:
                    print(f"    ❌ description 是檔名（fallback）")
                else:
                    print(f"    ✓ description: {desc[:60]}...")

            if has_tags:
                tags = caption.get("style_tags", [])
                print(f"    ✓ style_tags: {len(tags)} 個 - {tags[:3]}")

            if has_content:
                content = caption.get("content", {})
                title = content.get("title", "")
                if title:
                    print(f"    ✓ title: {title}")

        except json.JSONDecodeError:
            # 舊的純文字 caption
            print(f"    ⚠ 舊格式（純文字）: {caption_raw[:80]}...")
        except Exception as e:
            print(f"    ❌ 其他錯誤: {e}")

    print("\n" + "=" * 80)
    print()
