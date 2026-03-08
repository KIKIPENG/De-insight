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

    def __init__(self, db_path: str) -> None:
        self._db_path = Path(db_path)
        self._running = True

        from rag.job_repository import JobRepository
        from rag.job_executor import JobExecutor

        self._repo = JobRepository(self._db_path)
        self._executor = JobExecutor(self._db_path)

    def _handle_signal(self, signum, frame):
        log.info("Received signal %d, shutting down...", signum)
        self._running = False

    async def run(self) -> None:
        signal.signal(signal.SIGTERM, self._handle_signal)
        signal.signal(signal.SIGINT, self._handle_signal)

        await self._repo.ensure_table()
        reset_count = await self._repo.reset_stale_running(stale_seconds=600)
        if reset_count:
            log.info("Reset %d stale running jobs to queued", reset_count)

        log.info("Worker started, polling %s", self._db_path)

        while self._running:
            try:
                job = await self._repo.claim_next_job()
                if not job:
                    # Check for retryable jobs
                    retryable = await self._repo.get_retryable_jobs()
                    if retryable:
                        job = await self._repo.claim_next_job()

                if job:
                    await self._executor.execute(job)
                else:
                    await asyncio.sleep(self.POLL_INTERVAL)
            except Exception:
                log.exception("Worker loop error")
                await asyncio.sleep(self.POLL_INTERVAL)


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python -m rag.ingestion_worker <db_path>", file=sys.stderr)
        sys.exit(1)

    db_path = sys.argv[1]
    worker = IngestionWorkerProcess(db_path)
    asyncio.run(worker.run())


if __name__ == "__main__":
    main()
