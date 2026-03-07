"""本地 jina-clip-v1 embedding（文字 + 圖片語意向量）。

- dim = 512（Matryoshka 截斷 + L2 normalize）
- torch / transformers 懶載入，首次呼叫時才 import
- 支援 embed_text(str) 和 embed_image(path|bytes)
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Union

log = logging.getLogger(__name__)

EMBED_DIM = 512
_MODEL_NAME = "jinaai/jina-clip-v1"

# ── lazy singleton ───────────────────────────────────────────────────

_model = None
_processor = None
_tokenizer = None


def _has_meta_tensors(model) -> bool:
    """檢查模型參數或 buffer 是否包含 meta tensor。"""
    try:
        for p in model.parameters():
            if getattr(p, "is_meta", False):
                return True
        for b in model.buffers():
            if getattr(b, "is_meta", False):
                return True
    except Exception:
        pass
    return False


def _load_model():
    """載入模型，強制 CPU 實體權重。"""
    import torch
    from transformers import AutoModel

    model = AutoModel.from_pretrained(
        _MODEL_NAME,
        trust_remote_code=True,
        low_cpu_mem_usage=False,
        device_map="cpu",
        torch_dtype=torch.float32,
    )
    model.eval()

    if _has_meta_tensors(model):
        raise RuntimeError(
            f"jina-clip-v1 loaded with meta tensors despite device_map='cpu'. "
            f"Check transformers/accelerate versions."
        )
    return model


def _reset_and_reload():
    """清掉 singleton 並重新載入模型（只應被 retry 呼叫一次）。"""
    global _model
    log.warning("Meta tensor detected at inference, reloading model...")
    _model = _load_model()


def _ensure_model():
    """懶載入 jina-clip-v1 模型（首次呼叫約 5-15 秒）。"""
    global _model, _processor, _tokenizer
    if _model is not None:
        return

    from transformers import AutoProcessor, AutoTokenizer

    log.info("Loading jina-clip-v1 model (first call, may take a few seconds)...")
    _model = _load_model()
    _processor = AutoProcessor.from_pretrained(_MODEL_NAME, trust_remote_code=True)
    _tokenizer = AutoTokenizer.from_pretrained(_MODEL_NAME, trust_remote_code=True)
    log.info("jina-clip-v1 model loaded successfully.")


def ensure_model_downloaded() -> None:
    """確認模型已下載（Onboarding 時呼叫，同步阻塞）。"""
    _ensure_model()


def _truncate_and_normalize(vec, dim: int = EMBED_DIM) -> list[float]:
    """Matryoshka 截斷到指定維度後做 L2 normalize。"""
    import torch
    t = torch.tensor(vec[:dim], dtype=torch.float32)
    t = torch.nn.functional.normalize(t, p=2, dim=0)
    return t.tolist()


# ── public API ───────────────────────────────────────────────────────


async def embed_text(text: str) -> list[float]:
    """將文字轉為 512 維 L2-normalized 向量。"""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _embed_text_sync, text)


async def embed_image(source: Union[str, Path, bytes]) -> list[float]:
    """將圖片轉為 512 維 L2-normalized 向量。source 可為檔案路徑或 bytes。"""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _embed_image_sync, source)


# ── sync internals ───────────────────────────────────────────────────


def _embed_text_sync(text: str) -> list[float]:
    import torch

    _ensure_model()
    inputs = _tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    try:
        with torch.no_grad():
            text_embed = _model.get_text_features(**inputs)
    except RuntimeError as e:
        if "meta tensor" not in str(e).lower():
            raise
        _reset_and_reload()
        with torch.no_grad():
            text_embed = _model.get_text_features(**inputs)
    return _truncate_and_normalize(text_embed[0].cpu().numpy())


def _embed_image_sync(source: Union[str, Path, bytes]) -> list[float]:
    import torch
    from PIL import Image
    import io

    _ensure_model()

    if isinstance(source, bytes):
        img = Image.open(io.BytesIO(source)).convert("RGB")
    else:
        img = Image.open(str(source)).convert("RGB")

    inputs = _processor(images=img, return_tensors="pt")
    try:
        with torch.no_grad():
            img_embed = _model.get_image_features(**inputs)
    except RuntimeError as e:
        if "meta tensor" not in str(e).lower():
            raise
        _reset_and_reload()
        with torch.no_grad():
            img_embed = _model.get_image_features(**inputs)
    return _truncate_and_normalize(img_embed[0].cpu().numpy())
