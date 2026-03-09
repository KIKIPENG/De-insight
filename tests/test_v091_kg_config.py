"""v0.9.1: knowledge_graph 本地模型 GPU + 序列化設定。"""

import os
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock
from types import ModuleType

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

# Mock lightrag
if "lightrag" not in sys.modules:
    _mock_lr = ModuleType("lightrag")
    _mock_lr.LightRAG = MagicMock
    _mock_lr.QueryParam = MagicMock
    sys.modules["lightrag"] = _mock_lr
    _mock_lr_utils = ModuleType("lightrag.utils")
    _mock_lr_utils.EmbeddingFunc = MagicMock
    sys.modules["lightrag.utils"] = _mock_lr_utils


def test_local_llm_uses_gpu():
    """本地 LLM 應使用 GPU（num_gpu=99）。"""
    source = Path("rag/knowledge_graph.py").read_text()
    # 確認 num_gpu 設為 99
    assert 'body["options"] = {"num_gpu": 99}' in source or \
           "body[\"options\"] = {\"num_gpu\": 99}" in source, \
        "num_gpu should be 99 for local LLM"


def test_local_llm_async_is_1():
    """本地 LLM 的 llm_max_async 預設應為 1。"""
    source = Path("rag/knowledge_graph.py").read_text()
    assert '"1" if _is_local_llm else "8"' in source, \
        "Local LLM default llm_max_async should be 1"


def test_local_embed_async_is_1():
    """本地模型的 embed_max_async 預設應為 1。"""
    source = Path("rag/knowledge_graph.py").read_text()
    assert '"1" if _is_local_llm else "4"' in source, \
        "Local embed default max_async should be 1"


def test_no_num_gpu_0_remains():
    """確認沒有殘留的 num_gpu: 0。"""
    source = Path("rag/knowledge_graph.py").read_text()
    assert '"num_gpu": 0' not in source and "'num_gpu': 0" not in source, \
        "num_gpu: 0 should not remain in codebase"
