"""Ingestion job SQLite CRUD。

ingest_jobs 表放在全域 DATA_ROOT/ingest_jobs.db，
靠 project_id 欄位路由到正確專案。
"""

from __future__ import annotations

import hashlib
import json
import uuid
from datetime import datetime, timedelta
from pathlib import Path

import aiosqlite


class JobRepository:
    """SQLite CRUD for ingest_jobs table."""

    BACKOFF_SECONDS = [30, 120, 300]  # 3 retries

    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path

    async def ensure_table(self) -> None:
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        async with aiosqlite.connect(self._db_path, timeout=15) as db:
            await db.execute("PRAGMA journal_mode=WAL")
            await db.execute("""
                CREATE TABLE IF NOT EXISTS ingest_jobs (
                    id            TEXT PRIMARY KEY,
                    project_id    TEXT NOT NULL,
                    source        TEXT NOT NULL,
                    source_type   TEXT NOT NULL DEFAULT 'pdf',
                    source_hash   TEXT DEFAULT '',
                    title         TEXT DEFAULT '',
                    payload_json  TEXT DEFAULT '{}',
                    status        TEXT NOT NULL DEFAULT 'queued',
                    attempts      INTEGER DEFAULT 0,
                    last_error    TEXT DEFAULT '',
                    result_json   TEXT DEFAULT '{}',
                    next_retry_at TEXT DEFAULT NULL,
                    created_at    TEXT DEFAULT (datetime('now','localtime')),
                    updated_at    TEXT DEFAULT (datetime('now','localtime'))
                )
            """)

            # Migrate existing DBs with older ingest_jobs schemas.
            async with db.execute("PRAGMA table_info(ingest_jobs)") as cur:
                columns = {row[1] for row in await cur.fetchall()}
            add_columns = [
                ("source_type", "TEXT NOT NULL DEFAULT 'pdf'"),
                ("source_hash", "TEXT DEFAULT ''"),
                ("title", "TEXT DEFAULT ''"),
                ("payload_json", "TEXT DEFAULT '{}'"),
                ("result_json", "TEXT DEFAULT '{}'"),
                ("progress_pct", "REAL DEFAULT 0"),
                ("chunks_total", "INTEGER DEFAULT 0"),
                ("chunks_done", "INTEGER DEFAULT 0"),
                ("started_at", "TEXT DEFAULT NULL"),
            ]
            for col_name, col_sql in add_columns:
                if col_name in columns:
                    continue
                await db.execute(f"ALTER TABLE ingest_jobs ADD COLUMN {col_name} {col_sql}")
                columns.add(col_name)

            if "source_hash" in columns:
                await db.execute("""
                    CREATE INDEX IF NOT EXISTS idx_source_hash
                        ON ingest_jobs(project_id, source_hash)
                """)
            # Drop old unique dedup index if migrating (replaced by app-level check)
            try:
                await db.execute("DROP INDEX IF EXISTS idx_dedup_source")
            except Exception:
                pass
            await db.commit()

    @staticmethod
    def _compute_source_hash(project_id: str, source: str) -> str:
        """Compute dedup key from project_id + source."""
        raw = f"{project_id}::{source}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]

    async def create_job(
        self,
        project_id: str,
        source: str,
        source_type: str,
        title: str = "",
        payload_json: str = "{}",
        force: bool = False,
    ) -> str:
        job_id = str(uuid.uuid4())
        source_hash = self._compute_source_hash(project_id, source)
        async with aiosqlite.connect(self._db_path, timeout=15) as db:
            # Dedup: only block if same source is currently queued or running.
            # Done/done_with_warning are allowed (re-import / update).
            # force=True bypasses dedup entirely.
            if not force:
                async with db.execute(
                    """SELECT id, status FROM ingest_jobs
                       WHERE project_id = ? AND source_hash = ?
                         AND status IN ('queued', 'running')
                       LIMIT 1""",
                    (project_id, source_hash),
                ) as cur:
                    existing = await cur.fetchone()
                if existing:
                    raise DuplicateJobError(
                        f"Source already submitted (job {existing[0]}, status={existing[1]})"
                    )
            await db.execute(
                """INSERT INTO ingest_jobs
                   (id, project_id, source, source_type, source_hash, title, payload_json)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (job_id, project_id, source, source_type, source_hash, title, payload_json),
            )
            await db.commit()
        return job_id

    async def claim_next_job(self) -> dict | None:
        """Atomically claim the oldest queued job (queued → running).

        Uses BEGIN IMMEDIATE + single UPDATE..RETURNING-style pattern to
        prevent two workers from claiming the same job. The SELECT and
        UPDATE run inside the same IMMEDIATE transaction, which acquires
        a RESERVED lock before any read.
        """
        async with aiosqlite.connect(self._db_path, timeout=15) as db:
            db.row_factory = aiosqlite.Row
            # BEGIN IMMEDIATE acquires a reserved lock immediately,
            # preventing concurrent writers from interleaving.
            await db.execute("BEGIN IMMEDIATE")
            try:
                async with db.execute(
                    """SELECT * FROM ingest_jobs
                       WHERE status = 'queued'
                       ORDER BY created_at ASC LIMIT 1"""
                ) as cur:
                    row = await cur.fetchone()
                if not row:
                    await db.execute("ROLLBACK")
                    return None
                job = dict(row)
                await db.execute(
                    """UPDATE ingest_jobs
                       SET status = 'running', attempts = attempts + 1,
                           updated_at = datetime('now','localtime')
                       WHERE id = ? AND status = 'queued'""",
                    (job["id"],),
                )
                await db.commit()
            except Exception:
                await db.execute("ROLLBACK")
                raise
            job["status"] = "running"
            job["attempts"] = (job.get("attempts") or 0) + 1
            return job

    async def update_status(
        self,
        job_id: str,
        status: str,
        last_error: str = "",
        next_retry_at: str | None = None,
    ) -> None:
        async with aiosqlite.connect(self._db_path, timeout=15) as db:
            await db.execute(
                """UPDATE ingest_jobs
                   SET status = ?, last_error = ?, next_retry_at = ?,
                       updated_at = datetime('now','localtime')
                   WHERE id = ?""",
                (status, last_error, next_retry_at, job_id),
            )
            await db.commit()

    async def update_progress(
        self,
        job_id: str,
        chunks_done: int,
        chunks_total: int,
        started_at: str | None = None,
    ) -> None:
        """Update real-time progress for a running job."""
        pct = (chunks_done / chunks_total * 100) if chunks_total > 0 else 0
        async with aiosqlite.connect(self._db_path, timeout=15) as db:
            if started_at:
                await db.execute(
                    """UPDATE ingest_jobs
                       SET progress_pct = ?, chunks_done = ?, chunks_total = ?,
                           started_at = ?,
                           updated_at = datetime('now','localtime')
                       WHERE id = ?""",
                    (pct, chunks_done, chunks_total, started_at, job_id),
                )
            else:
                await db.execute(
                    """UPDATE ingest_jobs
                       SET progress_pct = ?, chunks_done = ?, chunks_total = ?,
                           updated_at = datetime('now','localtime')
                       WHERE id = ?""",
                    (pct, chunks_done, chunks_total, job_id),
                )
            await db.commit()

    async def get_active_jobs(self) -> list[dict]:
        """Return all running jobs with progress info."""
        async with aiosqlite.connect(self._db_path, timeout=15) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                """SELECT * FROM ingest_jobs
                   WHERE status IN ('queued', 'running')
                   ORDER BY created_at ASC"""
            ) as cur:
                return [dict(r) for r in await cur.fetchall()]

    async def update_result(self, job_id: str, result_json: str) -> None:
        """Store structured result data on the job row."""
        async with aiosqlite.connect(self._db_path, timeout=15) as db:
            await db.execute(
                """UPDATE ingest_jobs
                   SET result_json = ?, updated_at = datetime('now','localtime')
                   WHERE id = ?""",
                (result_json, job_id),
            )
            await db.commit()

    async def get_retryable_jobs(self) -> list[dict]:
        """Return failed jobs whose next_retry_at has passed."""
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        async with aiosqlite.connect(self._db_path, timeout=15) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                """SELECT * FROM ingest_jobs
                   WHERE status = 'failed'
                     AND next_retry_at IS NOT NULL
                     AND next_retry_at <= ?
                   ORDER BY next_retry_at ASC""",
                (now,),
            ) as cur:
                rows = await cur.fetchall()
        # Re-queue them atomically
        jobs = []
        for row in rows:
            job = dict(row)
            async with aiosqlite.connect(self._db_path, timeout=15) as db:
                await db.execute(
                    """UPDATE ingest_jobs
                       SET status = 'queued',
                           updated_at = datetime('now','localtime')
                       WHERE id = ? AND status = 'failed'""",
                    (job["id"],),
                )
                await db.commit()
            job["status"] = "queued"
            jobs.append(job)
        return jobs

    async def reset_stale_running(self, stale_seconds: int = 600) -> int:
        """Reset jobs stuck in 'running' for > stale_seconds back to 'queued'."""
        cutoff = (datetime.now() - timedelta(seconds=stale_seconds)).strftime(
            "%Y-%m-%d %H:%M:%S"
        )
        async with aiosqlite.connect(self._db_path, timeout=15) as db:
            cursor = await db.execute(
                """UPDATE ingest_jobs
                   SET status = 'queued',
                       updated_at = datetime('now','localtime')
                   WHERE status = 'running' AND updated_at < ?""",
                (cutoff,),
            )
            await db.commit()
            return cursor.rowcount

    async def list_completed_since(self, since_ts: str | None = None) -> list[dict]:
        """Return jobs completed (done/done_with_warning) at or after since_ts.

        Uses >= to avoid missing same-second completions.
        Caller is responsible for dedup by job id.
        """
        async with aiosqlite.connect(self._db_path, timeout=15) as db:
            db.row_factory = aiosqlite.Row
            if since_ts:
                async with db.execute(
                    """SELECT * FROM ingest_jobs
                       WHERE status IN ('done', 'done_with_warning')
                         AND updated_at >= ?
                       ORDER BY updated_at ASC, id ASC""",
                    (since_ts,),
                ) as cur:
                    return [dict(r) for r in await cur.fetchall()]
            else:
                async with db.execute(
                    """SELECT * FROM ingest_jobs
                       WHERE status IN ('done', 'done_with_warning')
                       ORDER BY updated_at ASC, id ASC"""
                ) as cur:
                    return [dict(r) for r in await cur.fetchall()]

    async def count_active(self) -> dict:
        """Return counts of queued and running jobs."""
        async with aiosqlite.connect(self._db_path, timeout=15) as db:
            async with db.execute(
                """SELECT status, COUNT(*) FROM ingest_jobs
                   WHERE status IN ('queued', 'running')
                   GROUP BY status"""
            ) as cur:
                rows = await cur.fetchall()
        counts = {"queued": 0, "running": 0}
        for status, cnt in rows:
            counts[status] = cnt
        return counts

    async def list_failed_terminal(self) -> list[dict]:
        """Return permanently failed jobs (no next_retry_at)."""
        async with aiosqlite.connect(self._db_path, timeout=15) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                """SELECT * FROM ingest_jobs
                   WHERE status = 'failed' AND next_retry_at IS NULL
                   ORDER BY updated_at DESC"""
            ) as cur:
                return [dict(r) for r in await cur.fetchall()]

    async def get_job(self, job_id: str) -> dict | None:
        async with aiosqlite.connect(self._db_path, timeout=15) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT * FROM ingest_jobs WHERE id = ?", (job_id,)
            ) as cur:
                row = await cur.fetchone()
                return dict(row) if row else None

    def compute_next_retry_at(self, attempts: int) -> str | None:
        """Compute next retry timestamp based on attempt count."""
        if attempts >= len(self.BACKOFF_SECONDS):
            return None  # max retries exceeded
        backoff = self.BACKOFF_SECONDS[attempts]
        return (datetime.now() + timedelta(seconds=backoff)).strftime(
            "%Y-%m-%d %H:%M:%S"
        )


class DuplicateJobError(Exception):
    """Raised when a job with the same source + project already exists."""
