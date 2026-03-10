"""Ingestion job SQLite CRUD。

ingest_jobs 表放在全域 DATA_ROOT/ingest_jobs.db，
靠 project_id 欄位路由到正確專案。
"""

from __future__ import annotations

import hashlib
import os
import socket
import uuid
from datetime import datetime, timedelta
from pathlib import Path

import aiosqlite


class JobRepository:
    """SQLite CRUD for ingest_jobs table."""

    BACKOFF_SECONDS = [30, 120, 300]  # default transient retries

    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path

    async def ensure_table(self) -> None:
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        async with aiosqlite.connect(self._db_path, timeout=15) as db:
            await db.execute("PRAGMA journal_mode=WAL")
            await db.execute(
                """
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
                    progress_pct  REAL DEFAULT 0,
                    chunks_total  INTEGER DEFAULT 0,
                    chunks_done   INTEGER DEFAULT 0,
                    started_at    TEXT DEFAULT NULL,
                    progress_stage TEXT DEFAULT '',
                    eta_seconds   INTEGER DEFAULT NULL,
                    phase         TEXT DEFAULT 'unknown',
                    phase_started_at TEXT DEFAULT NULL,
                    last_progress_at TEXT DEFAULT NULL,
                    progress_counter INTEGER DEFAULT 0,
                    heartbeat_at  TEXT DEFAULT NULL,
                    error_code    TEXT DEFAULT NULL,
                    error_detail  TEXT DEFAULT NULL,
                    idempotency_key TEXT DEFAULT NULL,
                    config_signature TEXT DEFAULT NULL,
                    retry_count   INTEGER DEFAULT 0,
                    max_retries   INTEGER DEFAULT 3,
                    rate_limit_retry_count INTEGER DEFAULT 0,
                    backoff_wall_time_seconds INTEGER DEFAULT 0,
                    created_at    TEXT DEFAULT (datetime('now','localtime')),
                    updated_at    TEXT DEFAULT (datetime('now','localtime'))
                )
                """
            )

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
                ("progress_stage", "TEXT DEFAULT ''"),
                ("eta_seconds", "INTEGER DEFAULT NULL"),
                ("phase", "TEXT DEFAULT 'unknown'"),
                ("phase_started_at", "TEXT DEFAULT NULL"),
                ("last_progress_at", "TEXT DEFAULT NULL"),
                ("progress_counter", "INTEGER DEFAULT 0"),
                ("heartbeat_at", "TEXT DEFAULT NULL"),
                ("error_code", "TEXT DEFAULT NULL"),
                ("error_detail", "TEXT DEFAULT NULL"),
                ("idempotency_key", "TEXT DEFAULT NULL"),
                ("config_signature", "TEXT DEFAULT NULL"),
                ("retry_count", "INTEGER DEFAULT 0"),
                ("max_retries", "INTEGER DEFAULT 3"),
                ("rate_limit_retry_count", "INTEGER DEFAULT 0"),
                ("backoff_wall_time_seconds", "INTEGER DEFAULT 0"),
            ]
            for col_name, col_sql in add_columns:
                if col_name in columns:
                    continue
                await db.execute(f"ALTER TABLE ingest_jobs ADD COLUMN {col_name} {col_sql}")
                columns.add(col_name)

            await db.execute(
                """
                CREATE TABLE IF NOT EXISTS ingest_batches (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    job_id TEXT NOT NULL,
                    batch_no INTEGER NOT NULL,
                    page_start INTEGER NOT NULL,
                    page_end INTEGER NOT NULL,
                    chunk_start INTEGER DEFAULT NULL,
                    chunk_end INTEGER DEFAULT NULL,
                    actual_chunks INTEGER DEFAULT NULL,
                    status TEXT DEFAULT 'queued',
                    attempts INTEGER DEFAULT 0,
                    started_at TEXT DEFAULT NULL,
                    finished_at TEXT DEFAULT NULL,
                    error_code TEXT DEFAULT NULL,
                    baseline_entity_delta INTEGER DEFAULT NULL,
                    baseline_relation_delta INTEGER DEFAULT NULL,
                    baseline_locked INTEGER DEFAULT 0,
                    UNIQUE(job_id, batch_no)
                )
                """
            )
            await db.execute(
                """
                CREATE TABLE IF NOT EXISTS worker_lease (
                    lease_name   TEXT PRIMARY KEY,
                    owner_id     TEXT NOT NULL,
                    heartbeat_at TEXT NOT NULL,
                    expires_at   TEXT NOT NULL
                )
                """
            )

            if "source_hash" in columns:
                await db.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_source_hash
                        ON ingest_jobs(project_id, source_hash)
                    """
                )
            await db.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_status_created
                    ON ingest_jobs(status, created_at)
                """
            )
            await db.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_jobs_phase
                    ON ingest_jobs(status, phase, updated_at)
                """
            )
            await db.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_jobs_idempotency
                    ON ingest_jobs(project_id, idempotency_key, created_at)
                """
            )
            await db.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_batches_job_status
                    ON ingest_batches(job_id, status, batch_no)
                """
            )

            # Drop old unique dedup index if migrating (replaced by app-level check)
            try:
                await db.execute("DROP INDEX IF EXISTS idx_dedup_source")
            except Exception:
                pass

            # Backfill new phase columns for old rows
            await db.execute(
                """
                UPDATE ingest_jobs SET phase = CASE
                    WHEN status IN ('queued', 'retrying') THEN 'queued'
                    WHEN status IN ('done', 'done_with_warning') THEN 'flushing'
                    WHEN status IN ('failed', 'failed_recoverable') THEN 'failed'
                    ELSE 'unknown'
                END
                WHERE phase IS NULL OR phase = ''
                """
            )
            await db.execute(
                """
                UPDATE ingest_jobs SET last_progress_at = updated_at
                WHERE last_progress_at IS NULL
                """
            )
            await db.execute(
                """
                UPDATE ingest_jobs SET retry_count = attempts
                WHERE retry_count = 0 AND attempts > 0
                """
            )
            await db.commit()

    @staticmethod
    def _compute_source_hash(project_id: str, source: str) -> str:
        """Compute dedup key from project_id + source."""
        raw = f"{project_id}::{source}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]

    @staticmethod
    def _phase_from_stage(stage: str) -> str | None:
        s = (stage or "").strip()
        if not s:
            return None
        if s.startswith("分chunk"):
            return "chunking"
        if s.startswith("建立圖譜"):
            return "extracting"
        if s.startswith("圖譜合併"):
            return "merging"
        if s.startswith("寫入圖譜"):
            return "flushing"
        return None

    async def create_job(
        self,
        project_id: str,
        source: str,
        source_type: str,
        title: str = "",
        payload_json: str = "{}",
        force: bool = False,
        idempotency_key: str = "",
        config_signature: str = "",
        max_retries: int = 3,
    ) -> str:
        job_id = str(uuid.uuid4())
        source_hash = self._compute_source_hash(project_id, source)
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        async with aiosqlite.connect(self._db_path, timeout=15) as db:
            if not force:
                async with db.execute(
                    """SELECT id, status FROM ingest_jobs
                       WHERE project_id = ? AND source_hash = ?
                         AND status IN ('queued', 'running', 'retrying')
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
                   (id, project_id, source, source_type, source_hash, title, payload_json,
                    status, phase, phase_started_at, last_progress_at, heartbeat_at,
                    idempotency_key, config_signature, max_retries)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    job_id,
                    project_id,
                    source,
                    source_type,
                    source_hash,
                    title,
                    payload_json,
                    "queued",
                    "queued",
                    now,
                    now,
                    now,
                    idempotency_key,
                    config_signature,
                    max(1, int(max_retries)),
                ),
            )
            await db.commit()
        return job_id

    async def find_by_idempotency(self, project_id: str, idempotency_key: str) -> dict | None:
        if not idempotency_key:
            return None
        async with aiosqlite.connect(self._db_path, timeout=15) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                """SELECT * FROM ingest_jobs
                   WHERE project_id = ? AND idempotency_key = ?
                     AND status IN ('queued', 'running', 'retrying', 'done', 'done_with_warning')
                   ORDER BY created_at DESC LIMIT 1""",
                (project_id, idempotency_key),
            ) as cur:
                row = await cur.fetchone()
                return dict(row) if row else None

    async def claim_next_job(self) -> dict | None:
        """Atomically claim the oldest queued job (queued -> running)."""
        async with aiosqlite.connect(self._db_path, timeout=15) as db:
            db.row_factory = aiosqlite.Row
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
                       SET status='running', attempts=attempts+1,
                           retry_count=CASE
                               WHEN retry_count < attempts + 1 THEN attempts + 1
                               ELSE retry_count
                           END,
                           phase=CASE
                               WHEN phase IN ('queued', 'retrying', 'unknown', '') THEN 'chunking'
                               ELSE phase
                           END,
                           phase_started_at=COALESCE(phase_started_at, datetime('now','localtime')),
                           heartbeat_at=datetime('now','localtime'),
                           last_progress_at=datetime('now','localtime'),
                           updated_at=datetime('now','localtime')
                       WHERE id=? AND status='queued'""",
                    (job["id"],),
                )
                await db.commit()
            except Exception:
                await db.execute("ROLLBACK")
                raise
            job["status"] = "running"
            job["attempts"] = (job.get("attempts") or 0) + 1
            if (job.get("phase") or "") in ("", "queued", "retrying", "unknown"):
                job["phase"] = "chunking"
            return job

    async def update_status(
        self,
        job_id: str,
        status: str,
        last_error: str = "",
        next_retry_at: str | None = None,
        phase: str | None = None,
        error_code: str | None = None,
        error_detail: str | None = None,
    ) -> None:
        clear_error_state = status in ("done", "done_with_warning")
        async with aiosqlite.connect(self._db_path, timeout=15) as db:
            await db.execute(
                """UPDATE ingest_jobs
                   SET status = ?, last_error = ?, next_retry_at = ?,
                       phase = COALESCE(?, phase),
                       error_code = CASE
                           WHEN ? = 1 THEN NULL
                           WHEN ? IS NOT NULL THEN ?
                           ELSE error_code
                       END,
                       error_detail = CASE
                           WHEN ? = 1 THEN NULL
                           WHEN ? IS NOT NULL THEN ?
                           ELSE error_detail
                       END,
                       heartbeat_at = datetime('now','localtime'),
                       updated_at = datetime('now','localtime')
                   WHERE id = ?""",
                (
                    status,
                    last_error,
                    next_retry_at,
                    phase,
                    1 if clear_error_state else 0,
                    error_code,
                    error_code,
                    1 if clear_error_state else 0,
                    error_detail,
                    error_detail,
                    job_id,
                ),
            )
            await db.commit()

    async def update_phase(
        self,
        job_id: str,
        phase: str,
        status: str = "running",
        *,
        error_code: str | None = None,
        error_detail: str | None = None,
    ) -> None:
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        async with aiosqlite.connect(self._db_path, timeout=15) as db:
            await db.execute(
                """UPDATE ingest_jobs
                   SET status=?, phase=?, phase_started_at=?,
                       heartbeat_at=?, updated_at=?,
                       error_code=CASE WHEN ? IS NOT NULL THEN ? ELSE error_code END,
                       error_detail=CASE WHEN ? IS NOT NULL THEN ? ELSE error_detail END
                   WHERE id=?""",
                (
                    status,
                    phase,
                    now,
                    now,
                    now,
                    error_code,
                    error_code,
                    error_detail,
                    error_detail,
                    job_id,
                ),
            )
            await db.commit()

    async def touch_heartbeat(self, job_id: str) -> None:
        async with aiosqlite.connect(self._db_path, timeout=15) as db:
            await db.execute(
                """UPDATE ingest_jobs
                   SET heartbeat_at=datetime('now','localtime'),
                       updated_at=datetime('now','localtime')
                   WHERE id=?""",
                (job_id,),
            )
            await db.commit()

    async def update_progress(
        self,
        job_id: str,
        chunks_done: int,
        chunks_total: int,
        started_at: str | None = None,
        progress_pct: float | None = None,
        progress_stage: str | None = None,
        eta_seconds: int | None = None,
    ) -> None:
        """Update real-time progress for a running job."""
        if progress_pct is None:
            pct = min(chunks_done / chunks_total * 100, 100.0) if chunks_total > 0 else 0
        else:
            pct = max(0.0, min(float(progress_pct), 100.0))
        stage = progress_stage or ""
        phase = self._phase_from_stage(stage)
        async with aiosqlite.connect(self._db_path, timeout=15) as db:
            prev_done = 0
            prev_pct = 0.0
            prev_stage = ""
            async with db.execute(
                "SELECT chunks_done, progress_pct, progress_stage FROM ingest_jobs WHERE id = ?",
                (job_id,),
            ) as cur:
                row = await cur.fetchone()
                if row:
                    prev_done = int(row[0] or 0)
                    prev_pct = float(row[1] or 0.0)
                    prev_stage = str(row[2] or "")
            bump = 1 if (
                (chunks_done > prev_done)
                or (pct > (prev_pct + 0.01))
                or (stage != prev_stage)
            ) else 0

            if started_at:
                await db.execute(
                    """UPDATE ingest_jobs
                       SET progress_pct=?, chunks_done=?, chunks_total=?,
                           progress_stage=?, eta_seconds=?, started_at=?,
                           phase=COALESCE(?, phase),
                           last_progress_at=CASE WHEN ? > 0 THEN datetime('now','localtime') ELSE last_progress_at END,
                           progress_counter=progress_counter + ?,
                           heartbeat_at=datetime('now','localtime'),
                           updated_at=datetime('now','localtime')
                       WHERE id=?""",
                    (pct, chunks_done, chunks_total, stage, eta_seconds, started_at, phase, bump, bump, job_id),
                )
            else:
                await db.execute(
                    """UPDATE ingest_jobs
                       SET progress_pct=?, chunks_done=?, chunks_total=?,
                           progress_stage=?, eta_seconds=?,
                           phase=COALESCE(?, phase),
                           last_progress_at=CASE WHEN ? > 0 THEN datetime('now','localtime') ELSE last_progress_at END,
                           progress_counter=progress_counter + ?,
                           heartbeat_at=datetime('now','localtime'),
                           updated_at=datetime('now','localtime')
                       WHERE id=?""",
                    (pct, chunks_done, chunks_total, stage, eta_seconds, phase, bump, bump, job_id),
                )
            await db.commit()

    async def get_active_jobs(self) -> list[dict]:
        """Return all active jobs with progress info."""
        async with aiosqlite.connect(self._db_path, timeout=15) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                """SELECT * FROM ingest_jobs
                   WHERE status = 'queued'
                      OR status = 'retrying'
                      OR status LIKE 'running%'
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
        """Return retrying jobs whose next_retry_at has passed."""
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        async with aiosqlite.connect(self._db_path, timeout=15) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                """SELECT * FROM ingest_jobs
                   WHERE status = 'retrying'
                     AND next_retry_at IS NOT NULL
                     AND next_retry_at <= ?
                   ORDER BY next_retry_at ASC""",
                (now,),
            ) as cur:
                rows = await cur.fetchall()
        jobs = []
        for row in rows:
            job = dict(row)
            async with aiosqlite.connect(self._db_path, timeout=15) as db:
                await db.execute(
                    """UPDATE ingest_jobs
                       SET status='queued', phase='queued',
                           updated_at=datetime('now','localtime')
                       WHERE id=? AND status='retrying'""",
                    (job["id"],),
                )
                await db.commit()
            job["status"] = "queued"
            job["phase"] = "queued"
            jobs.append(job)
        return jobs

    async def reset_stale_running(self, stale_seconds: int = 600) -> int:
        """Reset jobs stuck in running for > stale_seconds back to queued."""
        cutoff = (datetime.now() - timedelta(seconds=stale_seconds)).strftime(
            "%Y-%m-%d %H:%M:%S"
        )
        async with aiosqlite.connect(self._db_path, timeout=15) as db:
            cursor = await db.execute(
                """UPDATE ingest_jobs
                   SET status='queued', phase='queued',
                       updated_at=datetime('now','localtime')
                   WHERE status LIKE 'running%'
                     AND (updated_at < ? OR COALESCE(heartbeat_at, updated_at) < ?)""",
                (cutoff, cutoff),
            )
            await db.commit()
            return cursor.rowcount

    async def reconcile_unknown_job(self, job_id: str) -> None:
        async with aiosqlite.connect(self._db_path, timeout=15) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT * FROM ingest_batches WHERE job_id=? ORDER BY batch_no ASC",
                (job_id,),
            ) as cur:
                batches = [dict(r) for r in await cur.fetchall()]
            if not batches:
                await db.execute(
                    "UPDATE ingest_jobs SET phase='queued', status='queued', updated_at=datetime('now','localtime') WHERE id=?",
                    (job_id,),
                )
                await db.commit()
                return
            all_done = all((b.get("status") or "") == "done" for b in batches)
            if all_done:
                await db.execute(
                    "UPDATE ingest_jobs SET phase='flushing', status='running', updated_at=datetime('now','localtime') WHERE id=?",
                    (job_id,),
                )
                await db.commit()
                return
            in_progress = [
                b for b in batches if (b.get("status") or "") in ("running", "extracting", "merging")
            ]
            if in_progress:
                for b in in_progress:
                    await db.execute("UPDATE ingest_batches SET status='queued' WHERE id=?", (b["id"],))
                await db.execute(
                    "UPDATE ingest_jobs SET phase='extracting', status='running', updated_at=datetime('now','localtime') WHERE id=?",
                    (job_id,),
                )
                await db.commit()
                return
            queued = [b for b in batches if (b.get("status") or "") == "queued"]
            if queued:
                await db.execute(
                    "UPDATE ingest_jobs SET phase='extracting', status='running', updated_at=datetime('now','localtime') WHERE id=?",
                    (job_id,),
                )
                await db.commit()
                return
            await db.execute(
                """UPDATE ingest_jobs
                   SET phase='failed', status='failed',
                       error_code='RECONCILE_UNCERTAIN',
                       updated_at=datetime('now','localtime')
                   WHERE id=?""",
                (job_id,),
            )
            await db.commit()

    async def reconcile_stale_running_jobs(self, stale_seconds: int = 30) -> int:
        cutoff = (datetime.now() - timedelta(seconds=stale_seconds)).strftime("%Y-%m-%d %H:%M:%S")
        async with aiosqlite.connect(self._db_path, timeout=15) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                """SELECT * FROM ingest_jobs
                   WHERE status LIKE 'running%'
                     AND COALESCE(heartbeat_at, updated_at) < ?""",
                (cutoff,),
            ) as cur:
                stale = [dict(r) for r in await cur.fetchall()]

        for job in stale:
            jid = job["id"]
            phase = (job.get("phase") or "").strip() or "unknown"
            retry_count = int(job.get("retry_count") or 0)
            max_retries = int(job.get("max_retries") or 3)
            if phase == "unknown":
                await self.reconcile_unknown_job(jid)
                continue
            if retry_count < max_retries:
                await self.update_status(
                    jid,
                    "retrying",
                    last_error=job.get("last_error", ""),
                    phase="retrying",
                    error_code="WORKER_CRASHED",
                    error_detail="stale running reconciled to retrying",
                )
            else:
                await self.update_status(
                    jid,
                    "failed_recoverable",
                    last_error=job.get("last_error", ""),
                    phase="failed",
                    error_code="WORKER_CRASHED",
                    error_detail="stale running reconciled to failed_recoverable",
                )
        return len(stale)

    async def abort_incomplete(self) -> list[dict]:
        """Mark all queued/running jobs as failed and return them."""
        async with aiosqlite.connect(self._db_path, timeout=15) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                """SELECT * FROM ingest_jobs
                   WHERE status IN ('queued', 'running')
                   ORDER BY created_at ASC"""
            ) as cur:
                jobs = [dict(r) for r in await cur.fetchall()]
            if jobs:
                await db.execute(
                    """UPDATE ingest_jobs
                       SET status='failed', phase='failed',
                           last_error='程式關閉前未完成匯入',
                           next_retry_at=NULL,
                           error_code='ABORTED_ON_STARTUP',
                           updated_at=datetime('now','localtime')
                       WHERE status IN ('queued', 'running')"""
                )
                await db.commit()
            return jobs

    async def list_completed_since(self, since_ts: str | None = None) -> list[dict]:
        """Return jobs completed (done/done_with_warning) at or after since_ts."""
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
            async with db.execute(
                """SELECT * FROM ingest_jobs
                   WHERE status IN ('done', 'done_with_warning')
                   ORDER BY updated_at ASC, id ASC"""
            ) as cur:
                return [dict(r) for r in await cur.fetchall()]

    async def count_active(self) -> dict:
        """Return counts of queued/running/retrying jobs."""
        async with aiosqlite.connect(self._db_path, timeout=15) as db:
            async with db.execute(
                """SELECT
                       CASE
                         WHEN status LIKE 'running%' THEN 'running'
                         ELSE status
                       END AS norm_status,
                       COUNT(*)
                   FROM ingest_jobs
                   WHERE status = 'queued'
                      OR status = 'retrying'
                      OR status LIKE 'running%'
                   GROUP BY norm_status"""
            ) as cur:
                rows = await cur.fetchall()
        counts = {"queued": 0, "running": 0, "retrying": 0}
        for status, cnt in rows:
            counts[status] = cnt
        return counts

    async def list_failed_terminal(self) -> list[dict]:
        """Return terminal failed jobs."""
        async with aiosqlite.connect(self._db_path, timeout=15) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                """SELECT * FROM ingest_jobs
                   WHERE status IN ('failed', 'failed_recoverable')
                     AND next_retry_at IS NULL
                   ORDER BY updated_at DESC"""
            ) as cur:
                return [dict(r) for r in await cur.fetchall()]

    async def list_failed_retrying(self) -> list[dict]:
        """Return jobs waiting for retry countdown (compat + new statuses)."""
        async with aiosqlite.connect(self._db_path, timeout=15) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                """SELECT * FROM ingest_jobs
                   WHERE (
                     status = 'retrying'
                     OR (status = 'failed' AND next_retry_at IS NOT NULL)
                   )
                   AND next_retry_at IS NOT NULL
                   ORDER BY next_retry_at ASC"""
            ) as cur:
                return [dict(r) for r in await cur.fetchall()]

    async def get_job(self, job_id: str) -> dict | None:
        async with aiosqlite.connect(self._db_path, timeout=15) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute("SELECT * FROM ingest_jobs WHERE id = ?", (job_id,)) as cur:
                row = await cur.fetchone()
                return dict(row) if row else None

    async def cancel_job(self, job_id: str) -> bool:
        async with aiosqlite.connect(self._db_path, timeout=15) as db:
            cur = await db.execute(
                """UPDATE ingest_jobs
                   SET status='failed', phase='failed',
                       next_retry_at=NULL,
                       error_code='CANCELLED_BY_USER',
                       error_detail='cancel requested by user',
                       updated_at=datetime('now','localtime')
                   WHERE id=?
                     AND (status='queued' OR status='retrying' OR status LIKE 'running%')""",
                (job_id,),
            )
            await db.commit()
            return (cur.rowcount or 0) > 0

    async def retry_job(self, job_id: str) -> bool:
        async with aiosqlite.connect(self._db_path, timeout=15) as db:
            cur = await db.execute(
                """UPDATE ingest_jobs
                   SET status='queued', phase='queued',
                       next_retry_at=NULL,
                       error_code=NULL, error_detail=NULL,
                       updated_at=datetime('now','localtime')
                   WHERE id=?
                     AND status IN ('failed', 'failed_recoverable')""",
                (job_id,),
            )
            await db.commit()
            return (cur.rowcount or 0) > 0

    def compute_next_retry_at(self, attempts: int) -> str | None:
        """Compute next retry timestamp based on attempt count."""
        if attempts >= len(self.BACKOFF_SECONDS):
            return None
        backoff = self.BACKOFF_SECONDS[attempts]
        return (datetime.now() + timedelta(seconds=backoff)).strftime("%Y-%m-%d %H:%M:%S")

    def compute_next_retry_after_seconds(self, delay_seconds: int) -> str:
        """Compute retry timestamp with explicit delay seconds."""
        sec = max(1, int(delay_seconds))
        return (datetime.now() + timedelta(seconds=sec)).strftime("%Y-%m-%d %H:%M:%S")

    # ── Worker lease ──────────────────────────────────────────────

    @staticmethod
    def build_owner_id(pid: int | None = None) -> str:
        p = int(pid if pid is not None else os.getpid())
        return f"{socket.gethostname()}:{uuid.uuid4().hex[:8]}:{p}:{int(datetime.now().timestamp())}"

    @staticmethod
    def _is_same_host_and_pid_dead(owner_id: str) -> bool:
        try:
            parts = owner_id.split(":")
            if len(parts) < 3:
                return False
            host = parts[0]
            pid = int(parts[2])
            if host != socket.gethostname():
                return False
            try:
                os.kill(pid, 0)
                return False
            except ProcessLookupError:
                return True
            except PermissionError:
                return False
        except Exception:
            return False

    async def acquire_lease(self, lease_name: str, owner_id: str, ttl_seconds: int = 20) -> bool:
        now = datetime.now()
        now_s = now.strftime("%Y-%m-%d %H:%M:%S")
        exp_s = (now + timedelta(seconds=max(1, int(ttl_seconds)))).strftime("%Y-%m-%d %H:%M:%S")
        async with aiosqlite.connect(self._db_path, timeout=15) as db:
            db.row_factory = aiosqlite.Row
            await db.execute("BEGIN IMMEDIATE")
            try:
                async with db.execute(
                    "SELECT owner_id, expires_at FROM worker_lease WHERE lease_name=?",
                    (lease_name,),
                ) as cur:
                    row = await cur.fetchone()
                if row is None:
                    await db.execute(
                        "INSERT INTO worker_lease (lease_name, owner_id, heartbeat_at, expires_at) VALUES (?, ?, ?, ?)",
                        (lease_name, owner_id, now_s, exp_s),
                    )
                    await db.commit()
                    return True
                old_owner = row["owner_id"]
                old_exp = row["expires_at"] or ""
                can_takeover = self._is_same_host_and_pid_dead(old_owner) or (old_exp < now_s)
                if can_takeover:
                    await db.execute(
                        "UPDATE worker_lease SET owner_id=?, heartbeat_at=?, expires_at=? WHERE lease_name=?",
                        (owner_id, now_s, exp_s, lease_name),
                    )
                    await db.commit()
                    return True
                await db.execute("ROLLBACK")
                return False
            except Exception:
                await db.execute("ROLLBACK")
                raise

    async def heartbeat_lease(self, lease_name: str, owner_id: str, ttl_seconds: int = 20) -> bool:
        now = datetime.now()
        now_s = now.strftime("%Y-%m-%d %H:%M:%S")
        exp_s = (now + timedelta(seconds=max(1, int(ttl_seconds)))).strftime("%Y-%m-%d %H:%M:%S")
        async with aiosqlite.connect(self._db_path, timeout=15) as db:
            cur = await db.execute(
                """UPDATE worker_lease
                   SET heartbeat_at=?, expires_at=?
                   WHERE lease_name=? AND owner_id=?""",
                (now_s, exp_s, lease_name, owner_id),
            )
            await db.commit()
            return (cur.rowcount or 0) > 0

    async def release_lease(self, lease_name: str, owner_id: str) -> None:
        async with aiosqlite.connect(self._db_path, timeout=15) as db:
            await db.execute(
                "DELETE FROM worker_lease WHERE lease_name=? AND owner_id=?",
                (lease_name, owner_id),
            )
            await db.commit()

    # ── Batch checkpoint APIs (M2) ────────────────────────────────

    async def create_or_replace_batches(
        self,
        job_id: str,
        batches: list[dict],
    ) -> None:
        """Create ingest_batches rows for a job.

        batches item schema:
          {
            "batch_no": int,
            "page_start": int,
            "page_end": int,
            "actual_chunks": int,
          }
        """
        async with aiosqlite.connect(self._db_path, timeout=15) as db:
            await db.execute("DELETE FROM ingest_batches WHERE job_id = ?", (job_id,))
            for b in batches:
                await db.execute(
                    """INSERT INTO ingest_batches
                       (job_id, batch_no, page_start, page_end, actual_chunks, status)
                       VALUES (?, ?, ?, ?, ?, 'queued')""",
                    (
                        job_id,
                        int(b.get("batch_no", 0)),
                        int(b.get("page_start", 0)),
                        int(b.get("page_end", 0)),
                        int(b.get("actual_chunks", 0)),
                    ),
                )
            await db.commit()

    async def list_batches(self, job_id: str) -> list[dict]:
        async with aiosqlite.connect(self._db_path, timeout=15) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT * FROM ingest_batches WHERE job_id = ? ORDER BY batch_no ASC",
                (job_id,),
            ) as cur:
                return [dict(r) for r in await cur.fetchall()]

    async def mark_batch_running(self, batch_id: int) -> None:
        async with aiosqlite.connect(self._db_path, timeout=15) as db:
            await db.execute(
                """UPDATE ingest_batches
                   SET status='running',
                       attempts=attempts+1,
                       started_at=datetime('now','localtime')
                   WHERE id=?""",
                (batch_id,),
            )
            await db.commit()

    async def mark_batch_failed(self, batch_id: int, error_code: str) -> None:
        async with aiosqlite.connect(self._db_path, timeout=15) as db:
            await db.execute(
                """UPDATE ingest_batches
                   SET status='failed',
                       error_code=?,
                       finished_at=datetime('now','localtime')
                   WHERE id=?""",
                (error_code, batch_id),
            )
            await db.commit()

    async def mark_batch_done(
        self,
        batch_id: int,
        entity_delta: int,
        relation_delta: int,
    ) -> bool:
        """Mark batch done and return duplicate risk flag."""
        async with aiosqlite.connect(self._db_path, timeout=15) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT attempts, baseline_locked, baseline_entity_delta, baseline_relation_delta FROM ingest_batches WHERE id=?",
                (batch_id,),
            ) as cur:
                row = await cur.fetchone()
            if row is None:
                return False

            attempts = int(row["attempts"] or 0)
            baseline_locked = int(row["baseline_locked"] or 0)
            baseline_entity = row["baseline_entity_delta"]
            baseline_relation = row["baseline_relation_delta"]

            if attempts == 1 and baseline_locked == 0:
                await db.execute(
                    """UPDATE ingest_batches
                       SET baseline_entity_delta=?, baseline_relation_delta=?,
                           baseline_locked=1
                       WHERE id=?""",
                    (int(entity_delta), int(relation_delta), batch_id),
                )
                baseline_entity = int(entity_delta)
                baseline_relation = int(relation_delta)
                baseline_locked = 1

            await db.execute(
                """UPDATE ingest_batches
                   SET status='done',
                       finished_at=datetime('now','localtime'),
                       error_code=NULL
                   WHERE id=?""",
                (batch_id,),
            )
            await db.commit()

            if not baseline_locked:
                return False
            be = max(0, int(baseline_entity or 0))
            br = max(0, int(baseline_relation or 0))
            entity_ok = int(entity_delta) <= int(be * 1.5)
            relation_ok = int(relation_delta) <= int(br * 1.5)
            return not (entity_ok and relation_ok)

    async def set_waiting_backoff(
        self,
        job_id: str,
        *,
        phase: str,
        wait_seconds: int,
        rate_limit_retry_count: int,
        backoff_wall_time_seconds: int,
    ) -> None:
        next_retry = (datetime.now() + timedelta(seconds=max(1, int(wait_seconds)))).strftime(
            "%Y-%m-%d %H:%M:%S"
        )
        status = f"running:{phase}:waiting_backoff"
        async with aiosqlite.connect(self._db_path, timeout=15) as db:
            await db.execute(
                """UPDATE ingest_jobs
                   SET status=?, phase=?, error_code='RATE_LIMIT',
                       rate_limit_retry_count=?, backoff_wall_time_seconds=?,
                       next_retry_at=?,
                       heartbeat_at=datetime('now','localtime'),
                       updated_at=datetime('now','localtime')
                   WHERE id=?""",
                (
                    status,
                    phase,
                    int(rate_limit_retry_count),
                    int(backoff_wall_time_seconds),
                    next_retry,
                    job_id,
                ),
            )
            await db.commit()

    async def clear_waiting_backoff(self, job_id: str, *, phase: str) -> None:
        async with aiosqlite.connect(self._db_path, timeout=15) as db:
            await db.execute(
                """UPDATE ingest_jobs
                   SET status='running', phase=?,
                       error_code=NULL, next_retry_at=NULL,
                       heartbeat_at=datetime('now','localtime'),
                       updated_at=datetime('now','localtime')
                   WHERE id=?""",
                (phase, job_id),
            )
            await db.commit()


class DuplicateJobError(Exception):
    """Raised when a job with the same source + project already exists."""
