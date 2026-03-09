# check_style_tags.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import lancedb
import json
from collections import Counter
from paths import DATA_ROOT

# 找 LanceDB
projects_dir = DATA_ROOT / "projects"
lancedb_dirs = list(projects_dir.glob("*/lancedb"))

if not lancedb_dirs:
    print("找不到 LanceDB")
    sys.exit(1)

lancedb_dir = lancedb_dirs[0]
db = lancedb.connect(str(lancedb_dir))

if "images" not in db.table_names():
    print("沒有 images table")
    sys.exit(1)

tbl = db.open_table("images")
df = tbl.to_pandas()

print(f"總共 {len(df)} 張圖片\n")

# 收集所有 style_tags
all_tags = []
for idx, row in df.iterrows():
    try:
        caption = json.loads(row['caption'])
        tags = caption.get('style_tags', [])
        all_tags.extend(tags)
    except:
        pass

# 統計詞頻
tag_counts = Counter(all_tags)

print("你的圖片庫 style_tags 詞頻：")
print("=" * 60)
for tag, count in tag_counts.most_common(20):
    print(f"{tag}: {count} 次")

print("\n" + "=" * 60)
print("AI 說的標籤：")
ai_tags = ["極簡構圖", "高飽和色調", "硬邊幾何", "負空間運用", "材質對比"]
for tag in ai_tags:
    if tag in tag_counts:
        print(f"✓ {tag}: {tag_counts[tag]} 次")
    else:
        print(f"✗ {tag}: 不存在")