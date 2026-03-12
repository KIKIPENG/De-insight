"""Pytest configuration — exclude standalone integration scripts."""

import pytest

collect_ignore = [
    "test_rag_switch.py",
    "test_stability.py",
]

# Configure pytest-asyncio for all async tests
pytest_plugins = ('pytest_asyncio',)
