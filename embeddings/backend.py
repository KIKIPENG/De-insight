"""抽象 Embedding 後端介面。

所有 embedding 實作（GGUF、SentenceTransformers 等）都必須實作此介面。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Union


class EmbeddingBackend(ABC):
    """Embedding 後端抽象介面。"""

    @abstractmethod
    async def embed_query(self, text: str) -> list[float]:
        """將單筆查詢文字轉為向量（query encoding）。"""

    @abstractmethod
    async def embed_passages(self, texts: list[str]) -> list[list[float]]:
        """將多筆段落文字批次轉為向量（passage encoding）。"""

    @abstractmethod
    async def embed_image(self, image: Union[str, Path, bytes]) -> list[float]:
        """將圖片轉為向量。image 可為檔案路徑或 bytes。"""

    @abstractmethod
    def dimension(self) -> int:
        """回傳 embedding 維度。"""

    @abstractmethod
    def provider_signature(self) -> str:
        """回傳此後端的唯一簽章字串。

        簽章變更時觸發所有向量索引自動重建。
        格式："{model}-{backend}-{quantization}-{dim}"
        """
