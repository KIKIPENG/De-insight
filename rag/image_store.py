"""圖片知識庫 — LanceDB 圖片語意索引（dim=1024）。

每個專案有獨立的 images table，存放圖片 metadata + embedding。
支援文字語意搜圖（text-to-image retrieval）。
"""

from __future__ import annotations

import base64
import json
import logging
import math
import os
import time
from pathlib import Path

import lancedb
import pyarrow as pa

from paths import DATA_ROOT, project_root
from embeddings.service import get_embedding_service as _get_svc


async def _embed_image(source):
    return await _get_svc().embed_image(source)


async def _embed_text(text):
    return await _get_svc().embed_text(text)


def _truncate(vec: list[float], dim: int | None = None) -> list[float]:
    if dim is None:
        dim = IMAGE_DIM
    out = list(vec[:dim]) if vec else []
    if len(out) < dim:
        out.extend([0.0] * (dim - len(out)))
    return out


def _truncate_and_normalize(vec: list[float], dim: int | None = None) -> list[float]:
    out = _truncate(vec, dim)
    norm = math.sqrt(sum(x * x for x in out))
    if norm > 0:
        out = [x / norm for x in out]
    return out

log = logging.getLogger(__name__)


TABLE_NAME = "images"
IMAGE_DIM = 1024

_db_cache: dict[str, "lancedb.DBConnection"] = {}


def _make_schema(dim: int = IMAGE_DIM) -> pa.Schema:
    return pa.schema([
        pa.field("id", pa.utf8()),
        pa.field("filename", pa.utf8()),
        pa.field("caption", pa.utf8()),
        pa.field("tags", pa.utf8()),
        pa.field("created_at", pa.utf8()),
        pa.field("vector", pa.list_(pa.float32(), dim)),
        pa.field("vector_image", pa.list_(pa.float32(), dim)),
    ])


def _get_db(lancedb_dir: Path) -> "lancedb.DBConnection":
    lancedb_dir.mkdir(parents=True, exist_ok=True)
    key = str(lancedb_dir)
    if key not in _db_cache:
        _db_cache[key] = lancedb.connect(str(lancedb_dir))
    return _db_cache[key]


def _detect_table_dim(table) -> int | None:
    """偵測既有 LanceDB table 的向量維度。"""
    try:
        schema = table.schema
        for field in schema:
            if field.name == "vector":
                list_type = field.type
                if hasattr(list_type, "list_size"):
                    return list_type.list_size
    except Exception:
        pass
    return None


def _get_or_create_table(db):
    if TABLE_NAME in db.table_names():
        tbl = db.open_table(TABLE_NAME)
        existing_dim = _detect_table_dim(tbl)
        if existing_dim and existing_dim != IMAGE_DIM:
            log.warning(
                "Image table dim=%d != expected dim=%d, rebuilding table",
                existing_dim, IMAGE_DIM,
            )
            db.drop_table(TABLE_NAME)
            return db.create_table(TABLE_NAME, schema=_make_schema())
        # Migration: add vector_image column if missing
        field_names = [f.name for f in tbl.schema]
        if "vector_image" not in field_names:
            log.info("Migrating image table: adding vector_image column")
            df = tbl.to_pandas()
            zero_vec = [[0.0] * IMAGE_DIM] * len(df)
            df["vector_image"] = zero_vec
            db.drop_table(TABLE_NAME)
            new_tbl = db.create_table(TABLE_NAME, schema=_make_schema())
            if len(df) > 0:
                new_tbl.add(df.to_dict("records"))
            return new_tbl
        return tbl
    return db.create_table(TABLE_NAME, schema=_make_schema())


def _images_dir(project_id: str) -> Path:
    """專案圖片存放目錄。"""
    d = project_root(project_id) / "images"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _lancedb_dir(project_id: str) -> Path:
    return project_root(project_id) / "lancedb"


def _dedup_filename(img_dir: Path, filename: str) -> str:
    """若檔名已存在，自動加上遞增尾碼避免覆蓋。"""
    base = Path(filename).stem
    ext = Path(filename).suffix
    candidate = filename
    idx = 1
    while (img_dir / candidate).exists():
        candidate = f"{base}_{idx}{ext}"
        idx += 1
    return candidate


CAPTION_PROMPT = """分析這張圖片，提取三個面向的資訊。

**第一部分：CONTENT（內容辨識）**
辨識：
- type: 書籍 | 海報 | 展覽現場 | 作品照片 | 網頁截圖 | 其他
- title: 如果可辨識書名、展覽名稱、作品名稱，提取出來（沒有就留空）
- creator: 如果可辨識作者、藝術家、設計師，提取出來（沒有就留空）
- content_design_link: 一句話，說明設計語言和內容的關係（如何透過形式傳遞訊息）

**第二部分：STYLE_TAGS（風格座標）**
從以下六個維度各選 0-2 個最突出的標籤，總共 6-12 個：

1. 製作態度：精緻/粗糙/控制/偶然/手工/工業/DIY/量產
2. 密度節奏：密集/稀疏/均勻/不規則/留白/填滿/張力/平靜
3. 色彩傾向：單色/多色/高飽和/低飽和/冷色/暖色/消色/螢光
4. 時代感：古典/現代主義/後現代/當代/復古/未來感/無時間
5. 文化軸：歐洲/美國/日本/東亞/全球南方/在地/國際主義
6. 態度：嚴肅/玩味/諷刺/中性/激進/商業/實驗/學院

**第三部分：DESCRIPTION（給人看的描述）**
用策展語彙，2-3 句話，綜合描述這張圖片的視覺特質和設計手法。

只回傳 JSON，不要有任何解釋，格式：
{
  "content": {
    "type": "...",
    "title": "...",
    "creator": "...",
    "content_design_link": "..."
  },
  "style_tags": ["標籤1", "標籤2"],
  "description": "..."
}
"""


def _minimal_caption(filename: str = "") -> dict:
    """失敗時的最小 caption 結構。"""
    return {
        "content": {
            "type": "其他",
            "title": "",
            "creator": "",
            "content_design_link": "",
        },
        "style_tags": [],
        "description": filename or "",
    }


def _parse_caption(text: str) -> dict:
    """解析 LLM 回傳的 JSON caption，失敗時 raise。"""
    # 清理 markdown fence
    text = text.strip()
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()

    # 嘗試直接解析
    try:
        result = json.loads(text)
    except json.JSONDecodeError:
        # 常見 LLM 問題：單引號、尾隨逗號、不加引號的 key
        import re
        # 移除尾隨逗號（] 或 } 前面的逗號）
        cleaned = re.sub(r',\s*([}\]])', r'\1', text)
        # 如果仍失敗，嘗試用 ast.literal_eval 再轉 json
        try:
            result = json.loads(cleaned)
        except json.JSONDecodeError:
            import ast
            result = ast.literal_eval(cleaned)

    if not isinstance(result, dict):
        raise ValueError("Not a dict")
    if "content" not in result or "style_tags" not in result or "description" not in result:
        raise ValueError("Missing required fields")
    return result


def _normalize_caption(caption) -> dict:
    """將 caption 正規化為三段式 dict。支援舊的 str 格式向下相容。"""
    if isinstance(caption, dict):
        # 已經是新格式
        if "content" in caption and "style_tags" in caption and "description" in caption:
            return caption
        # 部分結構
        return {
            "content": caption.get("content", _minimal_caption()["content"]),
            "style_tags": caption.get("style_tags", []),
            "description": caption.get("description", ""),
        }
    if isinstance(caption, str):
        # 嘗試 JSON 解析
        try:
            parsed = json.loads(caption)
            if isinstance(parsed, dict) and "description" in parsed:
                return _normalize_caption(parsed)
        except (json.JSONDecodeError, ValueError):
            pass
        # 舊的純文字 caption
        return {
            "content": {
                "type": "其他",
                "title": "",
                "creator": "",
                "content_design_link": "",
            },
            "style_tags": [],
            "description": caption,
        }
    return _minimal_caption()


def _build_embed_text(caption: dict) -> str:
    """從三段式 caption 構建用於 embedding 的文字。"""
    embed_parts = []
    # style_tags（最重要，用於風格定位）
    if caption.get("style_tags"):
        embed_parts.append(" ".join(caption["style_tags"]))
    # content.title + creator（如果有）
    content = caption.get("content", {})
    if content.get("title"):
        embed_parts.append(content["title"])
    if content.get("creator"):
        embed_parts.append(content["creator"])
    # 如果都沒有，用 description
    if not embed_parts:
        embed_parts.append(caption.get("description", ""))
    return " ".join(embed_parts)


def _resolve_vision_config() -> tuple[str, str, str]:
    """解析 vision model 設定。回傳 (model, api_key, api_base)。

    優先順序：
    1. VISION_MODEL / VISION_API_KEY / VISION_API_BASE（獨立 vision 設定）
    2. 非本地的 RAG_LLM_MODEL（跳過 ollama/ 等無 vision 的本地模型）
    3. LLM_MODEL（主聊天模型）
    4. 預設 gemini-2.5-flash（免費且支援 vision）
    """
    from settings import load_env
    env = load_env()

    # 1. 獨立 vision 設定
    model = env.get("VISION_MODEL", "")
    if model:
        api_key = env.get("VISION_API_KEY", "") or env.get("GOOGLE_API_KEY", "")
        api_base = env.get("VISION_API_BASE", "")
    else:
        # 2. RAG_LLM_MODEL（跳過本地模型，無 vision）
        rag_model = env.get("RAG_LLM_MODEL", "")
        if rag_model and not rag_model.startswith("ollama/"):
            model = rag_model
        else:
            # 3. 主聊天模型
            model = env.get("LLM_MODEL", "")
        api_key = ""
        api_base = ""

    if not model:
        return "", "", ""

    # If VISION_API_BASE is explicitly set (e.g. OpenRouter), the user chose a
    # specific endpoint — keep the model ID as-is (OpenRouter needs the full
    # "google/gemini-2.5-flash" identifier) and just fill in missing key.
    if api_base:
        if not api_key:
            api_key = (
                env.get("OPENROUTER_API_KEY", "")
                or env.get("OPENAI_API_KEY", "")
                or env.get("GOOGLE_API_KEY", "")
            )
        return model, api_key, api_base

    # No explicit base — infer from model prefix
    if model.startswith("gemini/"):
        model = model.removeprefix("gemini/")
        api_base = "https://generativelanguage.googleapis.com/v1beta/openai"
        api_key = api_key or env.get("GOOGLE_API_KEY", "")
    elif model.startswith("google/"):
        # google/ prefix without explicit base → Google AI Studio
        model = model.removeprefix("google/")
        api_base = "https://generativelanguage.googleapis.com/v1beta/openai"
        api_key = api_key or env.get("GOOGLE_API_KEY", "")
    elif model.startswith("openai/"):
        model = model.removeprefix("openai/")
        api_base = env.get("OPENAI_API_BASE", "") or "https://api.openai.com/v1"
        api_key = api_key or env.get("OPENAI_API_KEY", "") or env.get("OPENROUTER_API_KEY", "")
    else:
        api_base = (
            env.get("RAG_API_BASE", "")
            or env.get("OPENAI_API_BASE", "")
            or "https://api.openai.com/v1"
        )
        api_key = (
            api_key
            or env.get("RAG_API_KEY", "")
            or env.get("OPENAI_API_KEY", "")
            or env.get("OPENROUTER_API_KEY", "")
        )

    return model, api_key, api_base


async def _auto_caption(image_bytes: bytes, filename: str = "") -> dict:
    """用 vision LLM 自動生成三段式 caption。失敗時回傳最小結構。"""
    raw_text = ""
    try:
        import httpx

        model, api_key, api_base = _resolve_vision_config()
        if not model:
            log.warning("Auto-caption skipped: no vision model configured")
            return _minimal_caption(Path(filename).stem if filename else "")

        b64 = base64.b64encode(image_bytes).decode()
        suffix = Path(filename).suffix.lower().lstrip(".")
        mime = {
            "jpg": "image/jpeg", "jpeg": "image/jpeg",
            "png": "image/png", "gif": "image/gif",
            "webp": "image/webp",
        }.get(suffix, "image/jpeg")

        from rag.rate_guard import get_rate_guard

        async def _call_caption():
            async with httpx.AsyncClient(timeout=60.0) as client:
                resp = await client.post(
                    f"{api_base}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": model,
                        "messages": [{
                            "role": "user",
                            "content": [
                                {"type": "text", "text": CAPTION_PROMPT},
                                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
                            ],
                        }],
                        "temperature": 0.3,
                        "max_tokens": 4096,
                    },
                )
                if resp.status_code != 200:
                    log.warning("Auto-caption API error %d: %s", resp.status_code, resp.text[:200])
                    return ""
                data = resp.json()
                text = (data["choices"][0]["message"]["content"] or "").strip()
                finish = data["choices"][0].get("finish_reason", "")
                if finish == "length":
                    log.warning("Auto-caption truncated (finish_reason=length), response: %s", text[:200])
                return text

        guard = get_rate_guard()
        raw_text = await guard.call_with_retry(
            "image/auto_caption", _call_caption, max_retries=2,
        )

        if not raw_text:
            return _minimal_caption(Path(filename).stem if filename else "")

        result = _parse_caption(raw_text)
        log.info("Auto-caption generated: %s", json.dumps(result, ensure_ascii=False)[:120])
        return result

    except Exception as e:
        log.warning("Auto-caption failed for %s: %s", filename, e)
        return _minimal_caption(Path(filename).stem if filename else "")


CHAT_DESCRIBE_PROMPT = """你是一位策展人的視覺助手。使用者正在對話中引用這張圖片，請提供詳細的視覺描述。

請描述：
1. 這張圖是什麼（物件類型、媒材）
2. 視覺特徵：色彩、構圖、排版、材質、字體等
3. 設計語言：風格、時代感、情緒、態度
4. 值得注意的細節

繁體中文，200-300 字。重點放在使用者的問題可能關心的面向。"""


async def describe_image_for_chat(
    image_path: str | Path,
    user_question: str = "",
) -> str:
    """聊天時即時看圖描述。用 vision model。失敗時回傳空字串。"""
    try:
        import httpx

        model, api_key, api_base = _resolve_vision_config()
        if not model:
            return ""

        img_path = Path(image_path)
        image_bytes = img_path.read_bytes()
        b64 = base64.b64encode(image_bytes).decode()
        suffix = img_path.suffix.lower().lstrip(".")
        mime = {
            "jpg": "image/jpeg", "jpeg": "image/jpeg",
            "png": "image/png", "gif": "image/gif",
            "webp": "image/webp",
        }.get(suffix, "image/jpeg")

        prompt = CHAT_DESCRIBE_PROMPT
        if user_question:
            prompt += f"\n\n使用者的問題：{user_question}"

        from rag.rate_guard import get_rate_guard

        async def _call_describe():
            async with httpx.AsyncClient(timeout=60.0) as client:
                resp = await client.post(
                    f"{api_base}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": model,
                        "messages": [{
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
                            ],
                        }],
                        "temperature": 0.3,
                        "max_tokens": 500,
                    },
                )
                if resp.status_code != 200:
                    log.warning("Chat describe API error %d: %s", resp.status_code, resp.text[:200])
                    return ""
                data = resp.json()
                return (data["choices"][0]["message"]["content"] or "").strip()

        guard = get_rate_guard()
        desc = await guard.call_with_retry(
            "image/chat_describe", _call_describe, max_retries=2,
        )
        log.info("Chat image description generated: %s", desc[:80])
        return desc
    except Exception as e:
        log.warning("Chat image description failed for %s: %s", image_path, e)
        return ""


async def save_image_file(
    project_id: str,
    filename: str,
    image_bytes: bytes,
) -> dict:
    """只存檔，不做 API 呼叫（毫秒級完成）。"""
    img_dir = _images_dir(project_id)
    filename = _dedup_filename(img_dir, filename)
    img_path = img_dir / filename
    img_path.write_bytes(image_bytes)
    return {"filename": filename, "path": str(img_path)}


async def index_image(
    project_id: str,
    filename: str,
    caption: str = "",
    tags: str = "",
) -> dict:
    """為已存檔的圖片建立 auto_caption + embedding + LanceDB 索引。

    caption 為空時呼叫 LLM 自動生成（三段式）。
    回傳包含完整 caption dict 的 metadata。
    """
    import uuid
    img_path = _images_dir(project_id) / filename
    image_bytes = img_path.read_bytes()

    # 生成或正規化 caption
    if not caption.strip():
        caption_dict = await _auto_caption(image_bytes, filename)
    else:
        caption_dict = _normalize_caption(caption)

    # 用 style_tags + title/creator 構建 embedding 文字（文字向量，用於搜尋）
    embed_text_str = _build_embed_text(caption_dict)
    try:
        vec = _truncate(await _embed_text(embed_text_str))
        final_vec = _truncate_and_normalize(vec)
    except Exception as e:
        log.warning("Text embedding failed for %s, using zero vector: %s", filename, e)
        final_vec = [0.0] * IMAGE_DIM

    # 圖片向量（CLIP image embedding，用於偏好聚類）
    try:
        vec_image = await _embed_image(image_bytes)
        final_vec_image = _truncate_and_normalize(vec_image)
    except Exception as e:
        log.warning("Image embedding failed for %s, using zero vector: %s", filename, e)
        final_vec_image = [0.0] * IMAGE_DIM

    image_id = str(uuid.uuid4())[:8]
    created_at = time.strftime("%Y-%m-%d %H:%M:%S")
    caption_json = json.dumps(caption_dict, ensure_ascii=False)

    db = _get_db(_lancedb_dir(project_id))
    table = _get_or_create_table(db)
    row = {
        "id": image_id,
        "filename": filename,
        "caption": caption_json,
        "tags": tags,
        "created_at": created_at,
        "vector": final_vec,
        "vector_image": final_vec_image,
    }
    table.add([row])

    log.info("Indexed image %s: %s", filename, caption_json[:80])
    return {
        "id": image_id,
        "filename": filename,
        "caption": caption_dict,
        "tags": tags,
        "created_at": created_at,
        "path": str(img_path),
    }


async def add_image(
    project_id: str,
    filename: str,
    image_bytes: bytes,
    caption: str = "",
    tags: str = "",
) -> dict:
    """儲存圖片檔案並建立向量索引（向後相容）。"""
    saved = await save_image_file(project_id, filename, image_bytes)
    return await index_image(project_id, saved["filename"], caption=caption, tags=tags)


async def search_images(
    project_id: str,
    query: str,
    limit: int = 5,
) -> list[dict]:
    """用文字語意搜尋圖片（用 style_tags 搜尋）。回傳最相關的圖片 metadata。"""
    db = _get_db(_lancedb_dir(project_id))
    if TABLE_NAME not in db.table_names():
        return []

    table = db.open_table(TABLE_NAME)
    if table.count_rows() == 0:
        return []

    # 粗篩：向量搜尋取 top 2x，留給 reranker 精排
    fetch_limit = min(limit * 2, table.count_rows())
    query_vec = await _embed_text(query)
    results = (
        table.search(query_vec, vector_column_name="vector")
        .metric("cosine")
        .limit(fetch_limit)
        .to_list()
    )

    img_dir = _images_dir(project_id)
    parsed = []
    for r in results:
        caption = _normalize_caption(r.get("caption", ""))
        parsed.append({
            "id": r.get("id", ""),
            "filename": r.get("filename", ""),
            "caption": caption,
            "tags": r.get("tags", ""),
            "created_at": r.get("created_at", ""),
            "path": str(img_dir / r.get("filename", "")),
            "score": 1.0 - r.get("_distance", 0),
        })

    # Jina Reranker 精排
    if len(parsed) >= 2:
        try:
            from rag.reranker import rerank_with_items
            reranked = await rerank_with_items(
                query=query,
                items=parsed,
                text_fn=lambda item: (
                    f"{' '.join(item['caption'].get('style_tags', []))} "
                    f"{item['caption'].get('content', {}).get('title', '')} "
                    f"{item['caption'].get('description', '')}"
                ),
                top_n=limit,
            )
            if reranked:
                return reranked
        except Exception as e:
            log.debug("Image reranker skipped: %s", e)

    return parsed[:limit]


async def list_images(project_id: str) -> list[dict]:
    """列出專案所有圖片 metadata。caption 為三段式 dict。"""
    db = _get_db(_lancedb_dir(project_id))
    if TABLE_NAME not in db.table_names():
        return []

    table = db.open_table(TABLE_NAME)
    if table.count_rows() == 0:
        return []

    rows = table.to_pandas().drop(columns=["vector", "vector_image"], errors="ignore")
    img_dir = _images_dir(project_id)
    result = []
    for _, r in rows.iterrows():
        caption = _normalize_caption(r.get("caption", ""))
        result.append({
            "id": r.get("id", ""),
            "filename": r.get("filename", ""),
            "caption": caption,
            "tags": r.get("tags", ""),
            "created_at": r.get("created_at", ""),
            "path": str(img_dir / r.get("filename", "")),
        })
    return result


async def delete_image(project_id: str, image_id: str) -> bool:
    """刪除圖片（索引 + 檔案）。"""
    db = _get_db(_lancedb_dir(project_id))
    if TABLE_NAME not in db.table_names():
        return False

    table = db.open_table(TABLE_NAME)
    # 先取 filename 以便刪檔
    try:
        df = table.to_pandas()
        match = df[df["id"] == image_id]
        if not match.empty:
            filename = match.iloc[0].get("filename", "")
            if filename:
                img_path = _images_dir(project_id) / filename
                img_path.unlink(missing_ok=True)
    except Exception:
        pass

    safe_id = image_id.replace("'", "''")
    table.delete(f"id = '{safe_id}'")
    return True


async def update_image(
    project_id: str,
    image_id: str,
    caption: str | dict = "",
    tags: str = "",
    recalc_vector: bool = False,
) -> bool:
    """更新圖片的 caption/tags。

    caption 可以是 str（向下相容）或 dict（三段式）。
    recalc_vector=True 時重算向量。
    """
    db = _get_db(_lancedb_dir(project_id))
    if TABLE_NAME not in db.table_names():
        return False

    table = db.open_table(TABLE_NAME)
    df = table.to_pandas()
    match = df[df["id"] == image_id]
    if match.empty:
        return False

    row = match.iloc[0].to_dict()
    safe_id = image_id.replace("'", "''")
    table.delete(f"id = '{safe_id}'")

    # 正規化 caption 為 JSON 字串
    if isinstance(caption, dict):
        caption_dict = _normalize_caption(caption)
        row["caption"] = json.dumps(caption_dict, ensure_ascii=False)
    else:
        row["caption"] = caption
    row["tags"] = tags

    # 重算向量
    if recalc_vector:
        caption_dict = _normalize_caption(row["caption"])
        embed_text_str = _build_embed_text(caption_dict)
        try:
            vec = _truncate(await _embed_text(embed_text_str))
            row["vector"] = _truncate_and_normalize(vec)
        except Exception as e:
            log.warning("Text vector recalc failed for %s: %s", image_id, e)
        # 重算圖片向量
        try:
            img_path = _images_dir(project_id) / row.get("filename", "")
            if img_path.exists():
                vec_image = await _embed_image(img_path.read_bytes())
                row["vector_image"] = _truncate_and_normalize(vec_image)
        except Exception as e:
            log.warning("Image vector recalc failed for %s: %s", image_id, e)

    # Ensure vector_image exists (migration compat)
    if "vector_image" not in row:
        row["vector_image"] = [0.0] * IMAGE_DIM

    table.add([row])
    return True


async def backfill_captions(project_id: str, notify=None) -> dict:
    """批次為沒有 caption 的圖片自動生成描述並更新向量。

    回傳 {"updated": int, "failed": int, "total": int}
    """
    db = _get_db(_lancedb_dir(project_id))
    if TABLE_NAME not in db.table_names():
        return {"updated": 0, "failed": 0, "total": 0}

    table = db.open_table(TABLE_NAME)
    if table.count_rows() == 0:
        return {"updated": 0, "failed": 0, "total": 0}

    df = table.to_pandas()
    no_caption = df[df["caption"].fillna("").str.strip() == ""]
    total = len(no_caption)
    if total == 0:
        return {"updated": 0, "failed": 0, "total": 0}

    img_dir = _images_dir(project_id)
    updated = 0
    failed = 0

    for _, row in no_caption.iterrows():
        filename = row.get("filename", "")
        image_id = row.get("id", "")
        img_path = img_dir / filename

        if notify:
            notify(f"生成描述中：{filename}（{updated + failed + 1}/{total}）")

        if not img_path.exists():
            log.warning("Image file not found for backfill: %s", img_path)
            failed += 1
            continue

        caption_dict = await _auto_caption(img_path.read_bytes(), filename)
        if not caption_dict.get("description"):
            failed += 1
            continue

        caption = json.dumps(caption_dict, ensure_ascii=False)
        ok = await update_image(project_id, image_id, caption=caption, tags=row.get("tags", ""))
        if ok:
            updated += 1
            log.info("Backfilled caption for %s: %s", filename, caption[:60])
        else:
            failed += 1

    return {"updated": updated, "failed": failed, "total": total}


async def update_caption_and_reindex(
    project_id: str,
    image_id: str,
    new_caption: dict,
) -> None:
    """更新 caption 並重算向量（基於新的 style_tags）。"""
    caption_dict = _normalize_caption(new_caption)
    ok = await update_image(
        project_id, image_id,
        caption=caption_dict, tags="",
        recalc_vector=True,
    )
    if not ok:
        raise ValueError(f"Image {image_id} not found")


async def regenerate_all_captions(
    project_id: str,
    only_fallback: bool = False,
    progress_callback=None,
) -> tuple[int, int]:
    """為所有（或只有 fallback）圖片重新生成 caption。

    回傳 (total, updated)
    """
    import asyncio

    db = _get_db(_lancedb_dir(project_id))
    if TABLE_NAME not in db.table_names():
        return (0, 0)

    table = db.open_table(TABLE_NAME)
    if table.count_rows() == 0:
        return (0, 0)

    df = table.to_pandas().drop(columns=["vector", "vector_image"], errors="ignore")
    rows = df.to_dict("records")

    # 篩選需要重新生成的
    to_regenerate = []
    for row in rows:
        if only_fallback:
            cap = _normalize_caption(row.get("caption", ""))
            desc = cap.get("description", "")
            filename_stem = Path(row["filename"]).stem
            if desc == filename_stem or not desc:
                to_regenerate.append(row)
        else:
            to_regenerate.append(row)

    total = len(to_regenerate)
    updated = 0
    img_dir = _images_dir(project_id)

    for idx, row in enumerate(to_regenerate):
        if progress_callback:
            progress_callback(idx + 1, total)

        img_path = img_dir / row["filename"]
        if not img_path.exists():
            continue

        try:
            # 延遲避免 rate limit（15 RPM）
            if idx > 0:
                await asyncio.sleep(4.5)

            new_caption = await _auto_caption(img_path.read_bytes(), row["filename"])
            await update_caption_and_reindex(
                project_id=project_id,
                image_id=row["id"],
                new_caption=new_caption,
            )
            updated += 1
        except Exception as e:
            log.warning("Regenerate caption failed for %s: %s", row["filename"], e)
            continue

    return (total, updated)


async def reindex_all_images(
    project_id: str,
    progress_callback=None,
) -> dict:
    """為所有圖片生成圖片向量（舊圖片可能只有文字向量）。

    跳過已有非零 vector_image 的圖片。
    Returns: {"total": int, "updated": int, "skipped": int, "failed": int}
    """
    images_dir = _images_dir(project_id)
    if not images_dir.exists():
        return {"total": 0, "updated": 0, "skipped": 0, "failed": 0}

    db = _get_db(_lancedb_dir(project_id))
    table = _get_or_create_table(db)
    if table.count_rows() == 0:
        return {"total": 0, "updated": 0, "skipped": 0, "failed": 0}

    df = table.to_pandas()
    total = len(df)
    updated = 0
    skipped = 0
    failed = 0

    for idx, (_, row) in enumerate(df.iterrows()):
        if progress_callback:
            progress_callback(idx + 1, total)

        # 檢查是否已有圖片向量
        vec_img = row.get("vector_image")
        if vec_img is not None and hasattr(vec_img, '__len__') and sum(abs(x) for x in vec_img) > 0.01:
            skipped += 1
            continue

        filename = row.get("filename", "")
        img_path = images_dir / filename
        if not img_path.exists():
            failed += 1
            continue

        try:
            new_vec = await _embed_image(img_path.read_bytes())
            new_vec = _truncate_and_normalize(_truncate(new_vec))
            # delete + add（LanceDB 不支援 in-place update）
            row_dict = row.to_dict()
            safe_id = str(row_dict["id"]).replace("'", "''")
            table.delete(f"id = '{safe_id}'")
            row_dict["vector_image"] = new_vec
            table.add([row_dict])
            updated += 1
        except Exception as e:
            log.warning("Reindex failed for %s: %s", filename, e)
            failed += 1

    return {"total": total, "updated": updated, "skipped": skipped, "failed": failed}


async def get_image_count(project_id: str) -> int:
    """取得專案圖片數量。"""
    db = _get_db(_lancedb_dir(project_id))
    if TABLE_NAME not in db.table_names():
        return 0
    table = db.open_table(TABLE_NAME)
    return table.count_rows()


async def extract_visual_preference(
    project_id: str,
    llm_call=None,
) -> dict | None:
    """從所有圖片的 style_tags 萃取視覺偏好。

    Returns: {"tags_freq": dict, "summary": str, "image_count": int} 或 None（圖片不足）。
    llm_call: async callable(prompt, max_tokens) -> str，用於生成自然語言摘要。
    """
    images = await list_images(project_id)
    if len(images) < 5:
        return None

    # 第一階段：style_tags 詞頻統計
    from collections import Counter
    all_tags = []
    for img in images:
        cap = img.get("caption", {})
        if isinstance(cap, str):
            cap = _normalize_caption(cap)
        tags = cap.get("style_tags", [])
        if isinstance(tags, list):
            all_tags.extend(tags)

    if not all_tags:
        return None

    freq = Counter(all_tags)
    top_tags = freq.most_common(15)

    # 第二階段：LLM 自然語言描述（可選）
    summary = ""
    if llm_call and len(top_tags) >= 3:
        freq_text = "\n".join(f"- {tag}：{count} 次" for tag, count in top_tags)
        # 取前 5 張最有代表性的 description
        sample_descs = []
        for img in images[:5]:
            cap = img.get("caption", {})
            if isinstance(cap, str):
                cap = _normalize_caption(cap)
            desc = cap.get("description", "")
            if desc and len(desc) > 10:
                sample_descs.append(desc[:100])

        prompt = f"""以下是一位創作者收集的參考圖片的風格標籤統計：

{freq_text}

代表圖片描述：
{chr(10).join(f"- {d}" for d in sample_descs)}

用 2-3 句話描述這位創作者的視覺偏好傾向。
語氣像策展人在介紹一個人的眼光，不要用「使用者」，用「他」。
不要條列，寫成自然段落，150 字以內。繁體中文。"""

        try:
            summary = await llm_call(prompt, max_tokens=300)
        except Exception:
            summary = ""

    return {
        "tags_freq": dict(top_tags),
        "summary": summary,
        "image_count": len(images),
    }


async def trigger_preference_update(
    project_id: str,
    llm_call=None,
    db_path=None,
    min_images: int = 5,
    min_delta: int = 5,
) -> dict | None:
    """獨立的偏好萃取入口。圖片上傳完成後呼叫。

    檢查是否需要更新（圖片數 >= min_images 且距上次萃取 >= min_delta 張）。
    回傳偏好結果 dict 或 None（不需更新）。
    """
    count = await get_image_count(project_id)
    if count < min_images:
        return None

    # 檢查上次萃取時的圖片數
    if db_path:
        from memory.store import get_memories
        prefs = await get_memories(type="preference", limit=1, db_path=db_path)
        if prefs:
            last_count = 0
            try:
                last_count = int(prefs[0].get("source", "0"))
            except (ValueError, TypeError):
                pass
            if count - last_count < min_delta:
                return None

    result = await extract_visual_preference(project_id, llm_call=llm_call)
    if not result or not result.get("summary"):
        return None

    # 存入記憶
    if db_path:
        from memory.store import add_memory
        await add_memory(
            type="preference",
            content=result["summary"],
            source=str(count),
            topic="美學偏好",
            category="美學偏好",
            db_path=db_path,
        )

    return result


def get_selected_path(project_id: str) -> Path:
    """selected.json 路徑（統一放 DATA_ROOT）。"""
    return DATA_ROOT / "selected.json"


async def set_selected(project_id: str, image_ids: list[str]) -> list[dict]:
    """設定選取的圖片 ID，寫入 selected.json，回傳完整 metadata。"""
    import json as _json
    path = get_selected_path(project_id)
    path.write_text(_json.dumps(image_ids))

    if not image_ids:
        return []

    db = _get_db(_lancedb_dir(project_id))
    if TABLE_NAME not in db.table_names():
        return []

    table = db.open_table(TABLE_NAME)
    df = table.to_pandas().drop(columns=["vector", "vector_image"], errors="ignore")
    selected = df[df["id"].isin(image_ids)]
    img_dir = _images_dir(project_id)
    result = []
    for _, r in selected.iterrows():
        caption = _normalize_caption(r.get("caption", ""))
        result.append({
            "id": r.get("id", ""),
            "filename": r.get("filename", ""),
            "caption": caption,
            "tags": r.get("tags", ""),
            "path": str(img_dir / r.get("filename", "")),
        })
    return result


async def get_selected(project_id: str) -> list[dict]:
    """取得目前選取的圖片 metadata。"""
    import json as _json
    path = get_selected_path(project_id)
    if not path.exists():
        return []
    try:
        ids = _json.loads(path.read_text())
    except Exception:
        return []
    if not ids:
        return []
    return await set_selected(project_id, ids)
