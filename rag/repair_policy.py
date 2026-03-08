"""Repair policies — 決定是否觸發 auto_repair。"""

from __future__ import annotations

from abc import ABC, abstractmethod


class RepairPolicy(ABC):
    @abstractmethod
    def should_repair(self, diagnosis: dict, error: Exception | None = None) -> bool:
        ...


class TransientErrorPolicy(RepairPolicy):
    """Transient errors (fds_to_keep, timeout, rate limit) should NEVER trigger destructive repair."""

    TRANSIENT_KEYWORDS = ("fds_to_keep", "timeout", "rate limit", "connection")

    def should_repair(self, diagnosis: dict, error: Exception | None = None) -> bool:
        if error and any(
            kw in str(error).lower() for kw in self.TRANSIENT_KEYWORDS
        ):
            return False
        return False  # This policy never triggers repair


class CorruptionPolicy(RepairPolicy):
    """Corruption (vdb empty + docs exist, or dim mismatch) triggers repair."""

    def should_repair(self, diagnosis: dict, error: Exception | None = None) -> bool:
        if diagnosis.get("healthy", True):
            return False
        issues = diagnosis.get("issues", [])
        has_empty_vdb = any("向量索引為空" in i for i in issues)
        has_dim_mismatch = bool(diagnosis.get("dim_mismatch"))
        return has_empty_vdb or has_dim_mismatch

    async def repair(self, project_id: str, diagnosis: dict, notify=None):
        from rag.repair import auto_repair as _original

        return await _original(
            project_id, notify=notify, _skip_policy_check=True
        )
