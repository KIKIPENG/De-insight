"""Pytest configuration — exclude standalone integration scripts."""

collect_ignore = [
    "test_rag_switch.py",
    "test_stability.py",
]
