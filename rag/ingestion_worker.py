"""Ingestion worker process — 獨立進程入口。

用法：python -m rag.ingestion_worker <db_path>

在任何 heavy import 之前設定 env var，避免 fds_to_keep 等子進程錯誤。
"""

from __future__ import annotations

# ── env var 硬覆蓋（不用 setdefault，避免被外部環境值污染觸發 fds_to_keep）──
import os
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import asyncio
import logging
import signal
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [worker] %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("rag.ingestion_worker")


class IngestionWorkerProcess:
    """Polls ingest_jobs DB for queued jobs and executes them."""

    POLL_INTERVAL = 3.0
    LEASE_NAME = "ingestion_worker"
    LEASE_TTL_SECONDS = 20
    LEASE_HEARTBEAT_SECONDS = 5
    LEASE_ACQUIRE_MAX_WAIT_SECONDS = 15

    def __init__(self, db_path: str) -> None:
        self._db_path = Path(db_path)
        self._running = True
        self._loop: asyncio.AbstractEventLoop | None = None
        self._current_task: asyncio.Task | None = None
        self._current_job_id: str | None = None
        self._lease_stop = asyncio.Event()
        self._lease_task: asyncio.Task | None = None

        from rag.job_repository import JobRepository
        from rag.job_executor import JobExecutor

        self._repo = JobRepository(self._db_path)
        self._executor = JobExecutor(self._db_path)
        self._owner_id = self._repo.build_owner_id()

    def _handle_signal(self, signum, frame):
        log.info("Received signal %d, shutting down...", signum)
        self._running = False
        if self._current_task and not self._current_task.done():
            self._current_task.cancel()
        if self._loop and self._loop.is_running():
            try:
                self._loop.create_task(self._safe_land_current_job("WORKER_TERMINATED"))
            except Exception:
                pass

    async def _safe_land_current_job(self, reason: str) -> None:
        if not self._current_job_id:
            return
        try:
            await self._repo.update_status(
                self._current_job_id,
                "retrying",
                phase="retrying",
                error_code=reason,
                error_detail="worker terminated during running task",
            )
        except Exception:
            log.exception("safe_land_current_job failed")

    async def _acquire_lease_loop(self) -> bool:
        waited = 0.0
        while self._running:
            ok = await self._repo.acquire_lease(
                self.LEASE_NAME,
                self._owner_id,
                ttl_seconds=self.LEASE_TTL_SECONDS,
            )
            if ok:
                return True
            log.info("Lease busy; worker recovering (<20s)")
            await asyncio.sleep(2.0)
            waited += 2.0
            if waited >= self.LEASE_ACQUIRE_MAX_WAIT_SECONDS:
                log.info("Lease still busy after %.0fs; exiting idle worker", waited)
                return False
        return False

    async def _lease_heartbeat_loop(self) -> None:
        try:
            while not self._lease_stop.is_set():
                await asyncio.sleep(self.LEASE_HEARTBEAT_SECONDS)
                ok = await self._repo.heartbeat_lease(
                    self.LEASE_NAME,
                    self._owner_id,
                    ttl_seconds=self.LEASE_TTL_SECONDS,
                )
                if not ok:
                    log.warning("Lost worker lease; stopping worker loop")
                    self._running = False
                    if self._current_task and not self._current_task.done():
                        self._current_task.cancel()
                    return
        except asyncio.CancelledError:
            return
        except Exception:
            log.exception("lease heartbeat loop failed")
            self._running = False

    async def run(self) -> None:
        self._loop = asyncio.get_running_loop()
        signal.signal(signal.SIGTERM, self._handle_signal)
        signal.signal(signal.SIGINT, self._handle_signal)

        await self._repo.ensure_table()
        leased = await self._acquire_lease_loop()
        if not leased:
            return
        self._lease_task = asyncio.create_task(self._lease_heartbeat_loop())
        reconciled = await self._repo.reconcile_stale_running_jobs(stale_seconds=30)
        if reconciled:
            log.info("Reconciled %d stale running jobs", reconciled)
        try:
            from rag.ingestion_service import IngestionService

            svc = IngestionService(self._db_path)
            restored = await svc.restore_pending_rollbacks()
            if restored:
                log.info("Restored %d pending rollback snapshot(s)", len(restored))
        except Exception:
            log.exception("Failed to restore pending rollback snapshots")

        log.info("Worker started, polling %s", self._db_path)

        try:
            while self._running:
                try:
                    job = await self._repo.claim_next_job()
                    if not job:
                        retryable = await self._repo.get_retryable_jobs()
                        if retryable:
                            job = await self._repo.claim_next_job()

                    if job:
                        self._current_job_id = str(job.get("id", ""))
                        self._current_task = asyncio.create_task(self._executor.execute(job))
                        await self._current_task
                        self._current_task = None
                        self._current_job_id = None
                    else:
                        await asyncio.sleep(self.POLL_INTERVAL)
                except asyncio.CancelledError:
                    raise
                except Exception:
                    log.exception("Worker loop error")
                    await asyncio.sleep(self.POLL_INTERVAL)
        finally:
            await self._safe_land_current_job("WORKER_TERMINATED")
            self._lease_stop.set()
            if self._lease_task:
                self._lease_task.cancel()
                try:
                    await self._lease_task
                except asyncio.CancelledError:
                    pass
            try:
                await self._repo.release_lease(self.LEASE_NAME, self._owner_id)
            except Exception:
                log.exception("Failed to release worker lease")


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python -m rag.ingestion_worker <db_path>", file=sys.stderr)
        sys.exit(1)

    db_path = sys.argv[1]
    worker = IngestionWorkerProcess(db_path)
    asyncio.run(worker.run())


if __name__ == "__main__":
    main()
