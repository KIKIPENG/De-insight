"""jina-embeddings-v4 entry — 委派給 EmbeddingService。

保留此 shim 以確保所有 consumer 的 import 路徑不變。
"""

from embeddings.local import (
    EMBED_DIM,
    EMBED_MODEL,
    embed_texts,
    embed_text,
    embed_image,
    get_embed_config,
)

__all__ = [
    "EMBED_DIM", "EMBED_MODEL",
    "embed_texts", "embed_text", "embed_image",
    "get_embed_config",
]
