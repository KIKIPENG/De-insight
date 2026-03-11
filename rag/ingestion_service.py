"""IngestionService — Facade for submitting ingestion jobs.

全域 singleton，管理單一 worker process。
TUI 透過這個類別 submit job + 查詢狀態。
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

log = logging.getLogger(__name__)

# Module-level singleton
_instance: IngestionService | None = None


class IngestionService:
    """Global singleton managing a single worker subprocess."""

    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        self._worker: subprocess.Popen | None = None
        self._last_poll_ts: str | None = None
        self._seen_completed_ids: set[str] = set()

        from rag.job_repository import JobRepository
        self._repo = JobRepository(db_path)

    async def ensure_table(self) -> None:
        await self._repo.ensure_table()

    async def submit(
        self,
        project_id: str,
        source: str,
        source_type: str,
        title: str = "",
        payload: dict | None = None,
    ) -> str:
        """Submit a new ingestion job and ensure worker is running."""
        await self._repo.ensure_table()
        self.ensure_worker_running()
        idempotency_key, config_signature = self._build_idempotency_key(
            project_id=project_id,
            source=source,
            source_type=source_type,
            payload=payload or {},
        )
        existing = await self._repo.find_by_idempotency(project_id, idempotency_key)
        if existing:
            log.info(
                "Dedup submit -> existing job %s (status=%s)",
                existing.get("id"),
                existing.get("status"),
            )
            return str(existing["id"])
        max_retries = max(1, int(os.environ.get("RAG_INGEST_MAX_RETRIES", "3") or "3"))
        job_id = await self._repo.create_job(
            project_id=project_id,
            source=source,
            source_type=source_type,
            title=title,
            payload_json=json.dumps(payload or {}, ensure_ascii=False),
            idempotency_key=idempotency_key,
            config_signature=config_signature,
            max_retries=max_retries,
        )
        log.info("Submitted job %s: type=%s source=%s", job_id, source_type, source[:80])
        return job_id

    async def submit_meta(
        self,
        project_id: str,
        source: str,
        source_type: str,
        title: str = "",
        payload: dict | None = None,
    ) -> dict:
        await self._repo.ensure_table()
        self.ensure_worker_running()
        idempotency_key, config_signature = self._build_idempotency_key(
            project_id=project_id,
            source=source,
            source_type=source_type,
            payload=payload or {},
        )
        existing = await self._repo.find_by_idempotency(project_id, idempotency_key)
        if existing:
            return {"job_id": str(existing["id"]), "accepted": False, "deduped": True}
        max_retries = max(1, int(os.environ.get("RAG_INGEST_MAX_RETRIES", "3") or "3"))
        job_id = await self._repo.create_job(
            project_id=project_id,
            source=source,
            source_type=source_type,
            title=title,
            payload_json=json.dumps(payload or {}, ensure_ascii=False),
            idempotency_key=idempotency_key,
            config_signature=config_signature,
            max_retries=max_retries,
        )
        return {"job_id": str(job_id), "accepted": True, "deduped": False}

    @staticmethod
    def _build_idempotency_key(
        *,
        project_id: str,
        source: str,
        source_type: str,
        payload: dict,
    ) -> tuple[str, str]:
        """Compute idempotency key based on content + ingestion-affecting config."""
        from config.service import get_config_service

        content_fp = IngestionService._content_fingerprint(source, source_type, payload)
        cfg = get_config_service().snapshot(include_process=True)
        signature_payload = {
            "chunk_size": cfg.get("LIGHTRAG_CHUNK_TOKEN_SIZE", "1000"),
            "chunk_overlap": cfg.get("LIGHTRAG_CHUNK_OVERLAP", "0"),
            "embedding_provider": (cfg.get("EMBED_PROVIDER", "") or "openrouter").lower(),
            "embedding_model": cfg.get("EMBED_MODEL", ""),
            "embedding_dim": cfg.get("EMBED_DIM", "") or "1024",
            "extraction_model": cfg.get("RAG_LLM_MODEL", "") or cfg.get("LLM_MODEL", ""),
            "extraction_prompt_version": cfg.get("RAG_EXTRACTION_PROMPT_VERSION", "v1"),
            "entity_schema_version": cfg.get("RAG_ENTITY_SCHEMA_VERSION", "v1"),
        }
        config_signature = hashlib.sha256(
            json.dumps(signature_payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest()[:16]
        idem = f"{project_id}:{content_fp}:{config_signature}"
        return idem, config_signature

    @staticmethod
    def _content_fingerprint(source: str, source_type: str, payload: dict) -> str:
        if source_type in ("pdf", "txt", "md"):
            p = Path(source)
            if p.exists() and p.is_file():
                h = hashlib.sha256()
                with p.open("rb") as f:
                    for chunk in iter(lambda: f.read(1024 * 1024), b""):
                        h.update(chunk)
                return h.hexdigest()[:16]
        if source_type == "text":
            content = (payload.get("content") or "").encode("utf-8")
            return hashlib.sha256(content).hexdigest()[:16]
        return hashlib.sha256(source.encode("utf-8")).hexdigest()[:16]

    def ensure_worker_running(self) -> None:
        """Start worker subprocess if not already running."""
        if self._worker is not None and self._worker.poll() is None:
            return  # still alive

        # Kill orphan workers from previous sessions
        self._kill_orphan_workers()

        project_root = Path(__file__).resolve().parent.parent
        env = {**os.environ, "PYTHONPATH": str(project_root)}

        self._worker = subprocess.Popen(
            [sys.executable, "-m", "rag.ingestion_worker", str(self._db_path)],
            env=env,
            cwd=str(project_root),
        )
        log.info("Started ingestion worker (pid=%d)", self._worker.pid)

    def _kill_orphan_workers(self) -> None:
        """Kill any leftover ingestion_worker processes from previous sessions."""
        import signal
        try:
            result = subprocess.run(
                ["pgrep", "-f", "rag.ingestion_worker"],
                capture_output=True, text=True, timeout=5,
            )
            for line in result.stdout.strip().splitlines():
                pid = int(line.strip())
                try:
                    os.kill(pid, signal.SIGTERM)
                    log.info("Killed orphan ingestion worker (pid=%d)", pid)
                except ProcessLookupError:
                    pass
        except Exception:
            pass

    def stop_worker(self) -> None:
        """Terminate worker subprocess."""
        if self._worker is None:
            self._kill_orphan_workers()
            return
        if self._worker.poll() is not None:
            self._worker = None
            self._kill_orphan_workers()
            return
        try:
            self._worker.terminate()
            self._worker.wait(timeout=5)
            log.info("Stopped ingestion worker")
        except Exception as e:
            log.warning("Failed to stop worker: %s", e)
            try:
                self._worker.kill()
            except Exception:
                pass
        self._worker = None
        self._kill_orphan_workers()

    async def submit_and_wait(
        self,
        project_id: str,
        source: str,
        source_type: str,
        title: str = "",
        payload: dict | None = None,
        poll_interval: float = 1.0,
        timeout: float = 3600.0,
    ) -> dict:
        """Submit a job and poll until it completes. Returns job dict with result_json.

        Used by bulk import modals that need per-item feedback.
        Raises RuntimeError on terminal failure or timeout.
        """
        job_id = await self.submit(project_id, source, source_type, title, payload)
        elapsed = 0.0
        while elapsed < timeout:
            await asyncio.sleep(poll_interval)
            elapsed += poll_interval
            job = await self._repo.get_job(job_id)
            if job is None:
                raise RuntimeError(f"Job {job_id} disappeared")
            status = job["status"]
            if status in ("done", "done_with_warning"):
                # Parse result_json for caller
                try:
                    job["_result"] = json.loads(job.get("result_json", "{}"))
                except (json.JSONDecodeError, TypeError):
                    job["_result"] = {}
                self._seen_completed_ids.add(job_id)
                return job
            if status == "failed" and job.get("next_retry_at") is None:
                raise RuntimeError(job.get("last_error", "unknown error"))
            # If failed with retry scheduled, keep waiting
        raise RuntimeError(f"Job {job_id} timed out after {timeout}s")

    async def poll_completed(self) -> list[dict]:
        """Return jobs completed since last poll (dedup by job id)."""
        completed = await self._repo.list_completed_since(self._last_poll_ts)
        new = [j for j in completed if j["id"] not in self._seen_completed_ids]
        for j in new:
            self._seen_completed_ids.add(j["id"])
        if completed:
            self._last_poll_ts = completed[-1]["updated_at"]
        return new

    async def poll_failed_terminal(self) -> list[dict]:
        """Return permanently failed jobs."""
        return await self._repo.list_failed_terminal()

    async def count_active(self) -> dict:
        """Return {'queued': N, 'running': N} for active jobs."""
        return await self._repo.count_active()

    async def get_active_progress(self, project_id: str | None = None) -> list[dict]:
        """Return all active jobs with progress info for UI display."""
        return await self._repo.get_active_jobs(project_id)

    async def get_deferred_retries(self, project_id: str | None = None) -> list[dict]:
        """Return failed jobs waiting for retry, with countdown seconds."""
        jobs = await self._repo.list_failed_retrying()
        if project_id:
            jobs = [j for j in jobs if str(j.get("project_id") or "") == str(project_id)]
        now = datetime.now()
        enriched: list[dict] = []
        for j in jobs:
            retry_at = (j.get("next_retry_at") or "").strip()
            retry_in = None
            if retry_at:
                try:
                    dt = datetime.strptime(retry_at, "%Y-%m-%d %H:%M:%S")
                    retry_in = max(0, int((dt - now).total_seconds()))
                except Exception:
                    retry_in = None
            jj = dict(j)
            jj["retry_in_seconds"] = retry_in
            enriched.append(jj)
        return enriched

    async def get_job(self, job_id: str) -> dict | None:
        return await self._repo.get_job(job_id)

    @staticmethod
    def _parse_db_datetime(value: str | None) -> datetime | None:
        if not value:
            return None
        s = str(value).strip()
        if not s:
            return None
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"):
            try:
                return datetime.strptime(s[:19], fmt)
            except Exception:
                continue
        return None

    @classmethod
    def _estimate_eta_from_batches(cls, batches: list[dict]) -> int | None:
        done_durations: list[float] = []
        for b in batches:
            if str(b.get("status") or "") != "done":
                continue
            st = cls._parse_db_datetime(b.get("started_at"))
            ft = cls._parse_db_datetime(b.get("finished_at"))
            if not st or not ft:
                continue
            delta = (ft - st).total_seconds()
            if delta > 0:
                done_durations.append(delta)
        if len(done_durations) < 2:
            return None
        recent = done_durations[-5:]
        avg = sum(recent) / len(recent)
        remaining = len([b for b in batches if str(b.get("status") or "") != "done"])
        return max(0, int(avg * max(remaining, 0)))

    async def get_job_detail(self, job_id: str) -> dict | None:
        job = await self._repo.get_job(job_id)
        if not job:
            return None
        batches = await self._repo.list_batches(job_id)
        effective_phase = self._repo._phase_from_stage(job.get("progress_stage", "") or "") or str(
            job.get("phase", "") or ""
        )
        running = [b for b in batches if (b.get("status") or "") == "running"]
        queued = [b for b in batches if (b.get("status") or "") == "queued"]
        done = [b for b in batches if (b.get("status") or "") == "done"]
        batch_current = None
        page_range = None
        if running:
            b = running[0]
            batch_current = int(b.get("batch_no") or 0) + 1
            page_range = f"{b.get('page_start')}-{b.get('page_end')}"
        elif queued:
            b = queued[0]
            batch_current = int(b.get("batch_no") or 0) + 1
            page_range = f"{b.get('page_start')}-{b.get('page_end')}"
        elif done:
            b = done[-1]
            batch_current = int(b.get("batch_no") or 0) + 1
            page_range = f"{b.get('page_start')}-{b.get('page_end')}"
        eta_seconds = self._estimate_eta_from_batches(batches)
        if eta_seconds is None:
            raw_eta = job.get("eta_seconds")
            if isinstance(raw_eta, (int, float)):
                eta_seconds = max(0, int(raw_eta))
        detail = {
            "job_id": str(job.get("id", "")),
            "status": job.get("status", ""),
            "phase": effective_phase,
            "overall_progress": int(float(job.get("progress_pct", 0) or 0)),
            "phase_detail": {
                "phase": effective_phase,
                "batch_current": batch_current,
                "batch_total": len(batches) if batches else None,
                "page_range": page_range,
            },
            "last_progress_at": job.get("last_progress_at"),
            "next_retry_at": job.get("next_retry_at"),
            "error_code": job.get("error_code"),
            "error_detail": job.get("error_detail"),
            "eta_seconds": eta_seconds,
        }
        return detail

    async def _restore_job_if_needed(self, job: dict, *, cleanup: bool) -> str:
        from rag.rollback import cleanup_job_snapshot, restore_job_snapshot

        restore_warning = ""
        current = await self._repo.get_job(job["id"]) or job
        if int(current.get("rollback_pending") or 0):
            try:
                await asyncio.to_thread(restore_job_snapshot, current)
                await self._repo.set_rollback_pending(current["id"], False)
                current = await self._repo.get_job(current["id"]) or current
            except Exception as e:
                restore_warning = f"rollback_restore_failed: {e}"
        snapshot_dir = str(current.get("rollback_snapshot_dir") or "").strip()
        if cleanup and snapshot_dir:
            try:
                await asyncio.to_thread(cleanup_job_snapshot, current)
            finally:
                await self._repo.clear_rollback(current["id"])
        return restore_warning

    async def cancel_job(self, job_id: str) -> bool:
        job = await self._repo.get_job(job_id)
        if not job:
            return False
        status = str(job.get("status") or "")
        if not (status == "queued" or status == "retrying" or status.startswith("running")):
            return False
        if status.startswith("running"):
            self.stop_worker()
            job = await self._repo.get_job(job_id) or job
        restore_warning = await self._restore_job_if_needed(job, cleanup=True)
        ok = await self._repo.cancel_job(job_id)
        if ok and restore_warning:
            await self._repo.update_status(
                job_id,
                "failed",
                last_error=restore_warning,
                next_retry_at=None,
                phase="failed",
                error_code="CANCELLED_BY_USER",
                error_detail=restore_warning[:300],
            )
        counts = await self.count_active()
        if (counts.get("queued") or 0) > 0 or (counts.get("retrying") or 0) > 0:
            self.ensure_worker_running()
        return ok

    async def retry_job(self, job_id: str) -> bool:
        ok = await self._repo.retry_job(job_id)
        if ok:
            self.ensure_worker_running()
        return ok

    async def abort_incomplete(self) -> list[dict]:
        """Abort all queued/running jobs from a previous session. Returns aborted jobs."""
        return await self._repo.abort_incomplete()

    async def abort_and_rollback_incomplete(self) -> list[dict]:
        await self._repo.ensure_table()
        self.stop_worker()
        jobs = await self._repo.list_incomplete_jobs()
        aborted: list[dict] = []
        for job in jobs:
            restore_warning = await self._restore_job_if_needed(job, cleanup=True)
            await self._repo.update_status(
                str(job["id"]),
                "failed",
                last_error=restore_warning or "TUI closed before ingestion completed",
                next_retry_at=None,
                phase="failed",
                error_code="CANCELLED_ON_TUI_EXIT",
                error_detail=(restore_warning or "cancelled on TUI exit")[:300],
            )
            latest = await self._repo.get_job(str(job["id"]))
            if latest:
                aborted.append(latest)
        return aborted

    async def restore_pending_rollbacks(self) -> list[dict]:
        await self._repo.ensure_table()
        jobs = await self._repo.list_jobs_requiring_restore()
        restored: list[dict] = []
        for job in jobs:
            warning = await self._restore_job_if_needed(job, cleanup=False)
            latest = await self._repo.get_job(str(job["id"])) or job
            if warning:
                await self._repo.update_status(
                    str(job["id"]),
                    str(latest.get("status") or "retrying"),
                    last_error=str(latest.get("last_error") or ""),
                    next_retry_at=latest.get("next_retry_at"),
                    phase=str(latest.get("phase") or "retrying"),
                    error_code=str(latest.get("error_code") or "ROLLBACK_RESTORE_FAILED"),
                    error_detail=warning[:300],
                )
                latest = await self._repo.get_job(str(job["id"])) or latest
            restored.append(latest)
        return restored

    @property
    def worker_alive(self) -> bool:
        return self._worker is not None and self._worker.poll() is None


def get_ingestion_service() -> IngestionService:
    """Get or create global IngestionService singleton."""
    global _instance
    if _instance is None:
        from paths import DATA_ROOT
        db_path = DATA_ROOT / "ingest_jobs.db"
        _instance = IngestionService(db_path)
    return _instance
