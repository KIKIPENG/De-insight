"""Job executor — 執行單一 ingestion job，分類錯誤，管理重試。

Phase A: RateGuard 感知 — 429/rate-limit 錯誤由 RateGuard 熔斷器管理，
job 狀態會記錄 throttle 原因。
"""

from __future__ import annotations

import json
import logging
from enum import Enum
from pathlib import Path

log = logging.getLogger(__name__)


class ErrorCategory(Enum):
    TRANSIENT = "transient"
    PERMANENT = "permanent"
    CORRUPTION = "corruption"


def classify_error(error: Exception) -> ErrorCategory:
    """Classify an ingestion error for retry decisions."""
    from rag.rate_guard import RateLimitError
    if isinstance(error, RateLimitError):
        return ErrorCategory.TRANSIENT
    err = str(error).lower()
    if "fds_to_keep" in err:
        return ErrorCategory.TRANSIENT
    if "429" in err or "rate limit" in err:
        return ErrorCategory.TRANSIENT
    if "timeout" in err:
        return ErrorCategory.TRANSIENT
    if "connection" in err:
        return ErrorCategory.TRANSIENT
    if "502" in err or "503" in err or "504" in err:
        return ErrorCategory.TRANSIENT
    if "not found" in err:
        return ErrorCategory.PERMANENT
    if "401" in err or "403" in err:
        return ErrorCategory.PERMANENT
    if "invalid" in err and "format" in err:
        return ErrorCategory.PERMANENT
    # Default: conservative — retry
    return ErrorCategory.TRANSIENT


class JobExecutor:
    """Execute a single ingestion job with error classification and retry scheduling."""

    MAX_ATTEMPTS = 3

    def __init__(self, db_path: Path) -> None:
        from rag.job_repository import JobRepository

        self._repo = JobRepository(db_path)

    async def execute(self, job: dict) -> None:
        """Execute a job: dispatch to handler, update status based on outcome."""
        from rag.source_handlers import get_handler

        job_id = job["id"]
        source_type = job.get("source_type", "pdf")
        attempts = job.get("attempts", 1)

        log.info(
            "Executing job %s (type=%s, attempt=%d): %s",
            job_id, source_type, attempts, job.get("source", "")[:80],
        )

        try:
            handler = get_handler(source_type)
            result = await handler.ingest(job)

            # "warning" can contain hard failure signals (e.g. post_verify vdb empty).
            # These cases must not be treated as successful ingestion.
            if self._warning_is_hard_failure(result.warning):
                raise RuntimeError(result.warning)

            # Register document and get doc_id
            doc_id = await self._register_document(job, result)

            # Store result in job row
            result_data = {
                "title": result.title,
                "doc_id": doc_id or "",
                "file_size": result.file_size,
                "page_count": result.page_count,
                "warning": result.warning,
            }
            await self._repo.update_result(job_id, json.dumps(result_data, ensure_ascii=False))

            if result.warning:
                await self._repo.update_status(job_id, "done_with_warning", last_error=result.warning)
                log.warning("Job %s done with warning: %s", job_id, result.warning[:100])
            else:
                await self._repo.update_status(job_id, "done")
                log.info("Job %s done: %s", job_id, result.title)

        except Exception as e:
            category = classify_error(e)
            log.error(
                "Job %s failed (attempt=%d, category=%s): %s",
                job_id, attempts, category.value, e,
            )

            if category == ErrorCategory.TRANSIENT and attempts < self.MAX_ATTEMPTS:
                next_retry = self._repo.compute_next_retry_at(attempts)
                await self._repo.update_status(
                    job_id, "failed", last_error=str(e), next_retry_at=next_retry,
                )
            else:
                # Terminal failure — no retry
                await self._repo.update_status(
                    job_id, "failed", last_error=str(e), next_retry_at=None,
                )

    @staticmethod
    def _warning_is_hard_failure(warning: str) -> bool:
        if not warning:
            return False
        w = warning.lower()
        hard_markers = (
            "post_verify: vdb_chunks 為空",
            "vdb_chunks 為空",
            "llama-server embedding 失敗",
            "embedding 失敗: http 500",
            "http 500",
            "key 'prompt' not found",
        )
        return any(m in w for m in hard_markers)

    async def _register_document(self, job: dict, result) -> str | None:
        """Register the ingested document in the project's conversation store.

        Returns doc_id on success, None on failure.
        """
        try:
            from conversation.store import ConversationStore
            from paths import project_root

            pid = job["project_id"]
            store = ConversationStore(
                db_path=project_root(pid) / "conversations.db"
            )
            source_type = result.source_type or job.get("source_type", "pdf")
            source_path = job["source"]
            if source_type == "text":
                source_path = "(manual_paste)"

            doc_id = await store.add_document(
                title=result.title,
                source_path=source_path,
                source_type=source_type,
                file_size=result.file_size,
                page_count=result.page_count,
                project_id=pid,
            )
            return doc_id
        except Exception as e:
            log.warning("Failed to register document for job %s: %s", job["id"], e)
            return None
