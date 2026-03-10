"""Job executor — 執行單一 ingestion job，分類錯誤，管理重試。

Phase A: RateGuard 感知 — 429/rate-limit 錯誤由 RateGuard 熔斷器管理，
job 狀態會記錄 throttle 原因。
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from datetime import datetime
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
    if "rate_limit_exhausted" in err:
        return ErrorCategory.PERMANENT
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

    MAX_ATTEMPTS = max(1, int(os.environ.get("RAG_INGEST_MAX_ATTEMPTS", "20")))

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
            await self._repo.update_phase(job_id, "chunking", status="running")
            handler = get_handler(source_type)
            stall_triggered = False
            stall_task: asyncio.Task | None = None
            ingest_task: asyncio.Task | None = None
            try:
                import asyncio

                ingest_task = asyncio.create_task(handler.ingest(job))
                stall_task = asyncio.create_task(
                    self._stall_watchdog(job_id, ingest_task)
                )
                result = await ingest_task
            except asyncio.CancelledError:
                if stall_task and not stall_task.done():
                    stall_task.cancel()
                stall_triggered = True
                raise RuntimeError("STALL_DETECTED: no progress within threshold")
            finally:
                if stall_task:
                    stall_task.cancel()
                    try:
                        await stall_task
                    except asyncio.CancelledError:
                        pass
                    except Exception:
                        pass

            # "warning" can contain hard failure signals (e.g. post_verify vdb empty).
            # These cases must not be treated as successful ingestion.
            if self._warning_is_hard_failure(result.warning):
                raise RuntimeError(result.warning)

            # Register document and get doc_id
            doc_id, register_warning = await self._register_document(job, result)
            if register_warning:
                if result.warning:
                    result.warning = f"{result.warning}; {register_warning}"
                else:
                    result.warning = register_warning

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
                await self._repo.update_status(
                    job_id,
                    "done_with_warning",
                    last_error=result.warning,
                    phase="flushing",
                )
                log.warning("Job %s done with warning: %s", job_id, result.warning[:100])
            else:
                await self._repo.update_status(job_id, "done", phase="flushing")
                log.info("Job %s done: %s", job_id, result.title)

        except Exception as e:
            category = classify_error(e)
            log.error(
                "Job %s failed (attempt=%d, category=%s): %s",
                job_id, attempts, category.value, e,
            )

            if category == ErrorCategory.TRANSIENT and attempts < self.MAX_ATTEMPTS:
                next_retry = self._compute_transient_retry_at(attempts, e)
                await self._repo.update_status(
                    job_id,
                    "retrying",
                    last_error=str(e),
                    next_retry_at=next_retry,
                    phase="retrying",
                    error_code=self._error_code_from_exception(e, category),
                    error_detail=str(e)[:300],
                )
            else:
                # Terminal failure — no retry
                await self._repo.update_status(
                    job_id,
                    "failed",
                    last_error=str(e),
                    next_retry_at=None,
                    phase="failed",
                    error_code=self._error_code_from_exception(e, category),
                    error_detail=str(e)[:300],
                )

    async def _stall_watchdog(self, job_id: str, ingest_task) -> None:
        import asyncio

        thresholds = {
            "extracting": 180,
            "merging": 600,
            "chunking": 300,
            "flushing": 300,
        }
        while not ingest_task.done():
            await asyncio.sleep(5)
            job = await self._repo.get_job(job_id)
            if not job:
                continue
            status = (job.get("status") or "").strip()
            if "waiting_backoff" in status:
                continue
            phase = (job.get("phase") or "").strip() or "chunking"
            threshold = thresholds.get(phase)
            if threshold is None:
                continue
            try:
                prog = float(job.get("progress_pct") or 0.0)
            except Exception:
                prog = 0.0
            # Late extraction usually includes heavy merge/write in provider internals.
            # Give a longer safety window to avoid false-positive stall kills.
            if phase in ("extracting", "merging") and prog >= 85.0:
                threshold = max(threshold, 900)
            last = (job.get("last_progress_at") or job.get("phase_started_at") or job.get("updated_at") or "").strip()
            if not last:
                continue
            try:
                dt = datetime.strptime(last, "%Y-%m-%d %H:%M:%S")
            except Exception:
                continue
            elapsed = (datetime.now() - dt).total_seconds()
            if elapsed > threshold:
                log.error(
                    "Stall detected for job %s (phase=%s, elapsed=%.1fs > %ss)",
                    job_id,
                    phase,
                    elapsed,
                    threshold,
                )
                ingest_task.cancel()
                return

    @staticmethod
    def _error_code_from_exception(error: Exception, category: ErrorCategory) -> str:
        msg = str(error).lower()
        if "stall_detected" in msg:
            return "STALL_DETECTED"
        if "429" in msg or "rate limit" in msg:
            return "RATE_LIMIT"
        if "timeout" in msg:
            return "PHASE_TIMEOUT"
        if category == ErrorCategory.PERMANENT:
            return "PERMANENT_ERROR"
        if category == ErrorCategory.CORRUPTION:
            return "CORRUPTION"
        return "TRANSIENT_ERROR"

    @staticmethod
    def is_rate_limit_exhausted(
        rate_limit_retry_count: int,
        backoff_wall_time_seconds: int,
        *,
        max_rate_limit_retries: int = 5,
        max_backoff_wall_time: int = 1800,
    ) -> bool:
        return (
            int(rate_limit_retry_count) >= int(max_rate_limit_retries)
            or int(backoff_wall_time_seconds) >= int(max_backoff_wall_time)
        )

    def _compute_transient_retry_at(self, attempts: int, error: Exception) -> str | None:
        msg = str(error).lower()
        if "429" in msg or "rate limit" in msg:
            # For API rate-limit failures, delay longer than generic transient retry
            # to avoid immediate re-hit and "running" stalls.
            default_sec = max(30, int(os.environ.get("RAG_RATE_LIMIT_RETRY_SECONDS", "180") or "180"))
            parsed = self._extract_retry_seconds(str(error))
            return self._repo.compute_next_retry_after_seconds(max(default_sec, parsed or 0))
        nxt = self._repo.compute_next_retry_at(attempts)
        if nxt is None:
            # Keep retry cadence alive for attempts beyond BACKOFF_SECONDS table.
            fallback_sec = max(60, int(os.environ.get("RAG_TRANSIENT_RETRY_SECONDS", "300") or "300"))
            return self._repo.compute_next_retry_after_seconds(fallback_sec)
        return nxt

    @staticmethod
    def _extract_retry_seconds(message: str) -> int | None:
        m = re.search(r"等待\s*(\d+)\s*秒", message)
        if m:
            return int(m.group(1))
        m = re.search(r"retry in\s*(\d+)", message.lower())
        if m:
            return int(m.group(1))
        return None

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

    async def _register_document(self, job: dict, result) -> tuple[str | None, str]:
        """Register the ingested document in the project's conversation store.

        Returns (doc_id, warning). warning is non-empty if registration failed.
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
            if result.warning:
                try:
                    await store.update_document_meta(
                        doc_id,
                        note=f"warning: {result.warning[:1000]}",
                    )
                except Exception:
                    log.debug("update_document_meta warning note failed", exc_info=True)
            return doc_id, ""
        except Exception as e:
            log.warning("Failed to register document for job %s: %s", job["id"], e)
            return None, f"register_document_failed: {e}"
