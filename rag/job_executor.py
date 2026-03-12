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

    async def _prepare_snapshot(self, job: dict) -> dict:
        from rag.rollback import prepare_job_snapshot, restore_job_snapshot

        current = await self._repo.get_job(job["id"]) or job
        snapshot_dir = str(current.get("rollback_snapshot_dir") or "").strip()
        if not snapshot_dir:
            snap = await asyncio.to_thread(prepare_job_snapshot, current)
            await self._repo.set_rollback_snapshot(current["id"], str(snap))
            current = await self._repo.get_job(current["id"]) or current
        if int(current.get("rollback_pending") or 0):
            await asyncio.to_thread(restore_job_snapshot, current)
            await self._repo.set_rollback_pending(current["id"], False)
            current = await self._repo.get_job(current["id"]) or current
        return current

    async def _restore_if_pending(self, job: dict, *, failure_detail: str | None = None) -> tuple[dict, str]:
        from rag.rollback import restore_job_snapshot

        current = await self._repo.get_job(job["id"]) or job
        if not int(current.get("rollback_pending") or 0):
            return current, failure_detail or ""
        try:
            await asyncio.to_thread(restore_job_snapshot, current)
            await self._repo.set_rollback_pending(current["id"], False)
            return (await self._repo.get_job(current["id"]) or current), failure_detail or ""
        except Exception as restore_error:
            extra = f"rollback_restore_failed: {restore_error}"
            if failure_detail:
                return current, f"{failure_detail}; {extra}"
            return current, extra

    async def _finalize_snapshot(self, job: dict) -> None:
        from rag.rollback import cleanup_job_snapshot

        current = await self._repo.get_job(job["id"]) or job
        snapshot_dir = str(current.get("rollback_snapshot_dir") or "").strip()
        try:
            if snapshot_dir:
                await asyncio.to_thread(cleanup_job_snapshot, current)
        except Exception:
            log.exception("cleanup_job_snapshot failed for job %s", current["id"])
        try:
            await self._repo.clear_rollback(current["id"])
        except Exception:
            log.exception("clear_rollback failed for job %s", current["id"])

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
            job = await self._prepare_snapshot(job)
            await self._repo.update_phase(job_id, "chunking", status="running")
            await self._repo.set_rollback_pending(job_id, True)
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

            # Post-ingestion: extract structural claims from chunks
            await self._extract_claims_from_chunks(job, doc_id or "")

            await self._finalize_snapshot(job)

        except Exception as e:
            category = classify_error(e)
            log.error(
                "Job %s failed (attempt=%d, category=%s): %s",
                job_id, attempts, category.value, e,
            )
            restored_job, restore_warning = await self._restore_if_pending(job, failure_detail=str(e)[:300])
            error_detail = restore_warning or str(e)[:300]

            if category == ErrorCategory.TRANSIENT and attempts < self.MAX_ATTEMPTS:
                next_retry = self._compute_transient_retry_at(attempts, e)
                await self._repo.update_status(
                    job_id,
                    "retrying",
                    last_error=str(e),
                    next_retry_at=next_retry,
                    phase="retrying",
                    error_code=self._error_code_from_exception(e, category),
                    error_detail=error_detail,
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
                    error_detail=error_detail,
                )
                await self._finalize_snapshot(restored_job)

    async def _stall_watchdog(self, job_id: str, ingest_task) -> None:
        import asyncio

        thresholds = {
            "extracting": 180,
            "merging": 1200,
            "chunking": 300,
            "flushing": 900,
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
            last = (
                job.get("last_progress_at")
                or job.get("heartbeat_at")
                or job.get("phase_started_at")
                or job.get("updated_at")
                or ""
            ).strip()
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
            "embedding api 錯誤",
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

    async def _extract_claims_from_chunks(self, job: dict, doc_id: str) -> None:
        """Extract structural claims from ingested chunks and store them.

        Reads vdb_chunks.json, samples representative chunks, runs
        ThoughtExtractor.extract_from_passage() on each, and persists
        resulting Claims to ClaimStore.

        This is a best-effort step — failures are logged but do not
        affect the job's success status.
        """
        try:
            from pathlib import Path
            from core.thought_extractor import ThoughtExtractor, LLMCallable
            from core.stores.claim_store import ClaimStore
            from rag.vdb_utils import find_vdb_chunks_file

            pid = job["project_id"]

            # 1. Read chunks from vdb_chunks.json
            vdb_path = find_vdb_chunks_file(pid)
            if not vdb_path or not vdb_path.exists():
                log.debug("_extract_claims: no vdb_chunks file for project %s", pid)
                return

            chunks_data = json.loads(vdb_path.read_text(encoding="utf-8"))
            all_chunks = chunks_data.get("data", [])
            if not all_chunks:
                log.debug("_extract_claims: empty chunks for project %s", pid)
                return

            # 2. Sample chunks — avoid excessive LLM calls
            # Take up to 10 representative chunks, evenly spaced
            max_samples = min(10, len(all_chunks))
            if len(all_chunks) <= max_samples:
                sampled = all_chunks
            else:
                step = len(all_chunks) / max_samples
                sampled = [all_chunks[int(i * step)] for i in range(max_samples)]

            # 3. Create LLM callable (reuse RAG LLM config)
            from config_service import get_config_service
            env = get_config_service().snapshot(include_process=True)
            llm_model = env.get("LLM_MODEL", "")
            llm_key = env.get("OPENAI_API_KEY", "") or env.get("ANTHROPIC_API_KEY", "")
            llm_base = env.get("OPENAI_API_BASE", "https://openrouter.ai/api/v1")

            async def _llm_for_claims(prompt: str) -> str:
                import httpx
                import litellm
                from rag.rate_guard import get_rate_guard

                messages = [{"role": "user", "content": prompt}]

                async def _call():
                    if llm_model.startswith("gemini/"):
                        resp = await litellm.acompletion(
                            model=llm_model, messages=messages, api_key=llm_key,
                        )
                        return resp.choices[0].message.content or ""
                    async with httpx.AsyncClient(timeout=120.0) as client:
                        body = {"model": llm_model, "messages": messages}
                        resp = await client.post(
                            f"{llm_base}/chat/completions",
                            headers={
                                "Authorization": f"Bearer {llm_key}",
                                "HTTP-Referer": "https://github.com/De-insight",
                                "X-Title": "De-insight",
                            },
                            json=body,
                        )
                        resp.raise_for_status()
                        return resp.json()["choices"][0]["message"]["content"]

                guard = get_rate_guard()
                result = await guard.call_with_retry(
                    "claim_extraction/chat", _call, max_retries=2,
                )
                # Strip <think>...</think> from reasoning models
                if result and "<think>" in result:
                    result = re.sub(r"<think>[\s\S]*?</think>\s*", "", result)
                return result

            # 4. Run extraction on sampled chunks
            extractor = ThoughtExtractor(
                llm_callable=LLMCallable(func=_llm_for_claims),
                project_id=pid,
            )
            claim_store = ClaimStore(project_id=pid)
            total_claims = 0
            chunk_errors = 0

            for idx, chunk in enumerate(sampled):
                # Extract text content from chunk
                chunk_text = ""
                if isinstance(chunk, dict):
                    chunk_text = chunk.get("content", "") or chunk.get("text", "")
                elif isinstance(chunk, str):
                    chunk_text = chunk

                if not chunk_text or len(chunk_text.strip()) < 30:
                    continue

                try:
                    result = await extractor.extract_from_passage(
                        passage_text=chunk_text,
                        source_id=doc_id,
                    )
                    if result.was_extracted and result.claims:
                        for claim in result.claims:
                            await claim_store.add(claim)
                            total_claims += 1
                except Exception as chunk_err:
                    chunk_errors += 1
                    log.warning(
                        "_extract_claims: chunk %d/%d extraction failed: %s: %s",
                        idx + 1, len(sampled),
                        type(chunk_err).__name__,
                        str(chunk_err)[:200],
                    )
                    continue

            if total_claims > 0:
                log.info(
                    "Extracted %d structural claims from %d chunks for job %s (doc_id=%s, errors=%d)",
                    total_claims, len(sampled), job["id"], doc_id, chunk_errors,
                )
            elif chunk_errors > 0:
                log.error(
                    "Claim extraction produced 0 claims with %d errors from %d chunks "
                    "for job %s (doc_id=%s). Run: python tools/reextract_claims.py %s",
                    chunk_errors, len(sampled), job["id"], doc_id, pid,
                )
            else:
                log.warning(
                    "Claim extraction produced 0 claims (no errors) from %d chunks "
                    "for job %s — LLM may have returned empty results",
                    len(sampled), job["id"],
                )

        except Exception as e:
            log.error(
                "_extract_claims_from_chunks failed for job %s: %s: %s",
                job["id"], type(e).__name__, str(e)[:300],
            )
