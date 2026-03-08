"""IngestionService — Facade for submitting ingestion jobs.

全域 singleton，管理單一 worker process。
TUI 透過這個類別 submit job + 查詢狀態。
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import subprocess
import sys
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
        job_id = await self._repo.create_job(
            project_id=project_id,
            source=source,
            source_type=source_type,
            title=title,
            payload_json=json.dumps(payload or {}, ensure_ascii=False),
        )
        log.info("Submitted job %s: type=%s source=%s", job_id, source_type, source[:80])
        return job_id

    def ensure_worker_running(self) -> None:
        """Start worker subprocess if not already running."""
        if self._worker is not None and self._worker.poll() is None:
            return  # still alive

        project_root = Path(__file__).resolve().parent.parent
        env = {**os.environ, "PYTHONPATH": str(project_root)}

        self._worker = subprocess.Popen(
            [sys.executable, "-m", "rag.ingestion_worker", str(self._db_path)],
            env=env,
            cwd=str(project_root),
        )
        log.info("Started ingestion worker (pid=%d)", self._worker.pid)

    def stop_worker(self) -> None:
        """Terminate worker subprocess."""
        if self._worker is None:
            return
        if self._worker.poll() is not None:
            self._worker = None
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

    async def submit_and_wait(
        self,
        project_id: str,
        source: str,
        source_type: str,
        title: str = "",
        payload: dict | None = None,
        poll_interval: float = 1.0,
        timeout: float = 600.0,
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

    async def get_job(self, job_id: str) -> dict | None:
        return await self._repo.get_job(job_id)

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
