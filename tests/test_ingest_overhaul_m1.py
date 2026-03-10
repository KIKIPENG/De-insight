from __future__ import annotations

import asyncio
import os
from pathlib import Path

import aiosqlite
import pytest

from rag.ingestion_service import IngestionService
from rag.job_executor import JobExecutor
from rag.job_repository import JobRepository
from rag.knowledge_graph import _build_pdf_batches


@pytest.mark.asyncio
async def test_1_idempotency_same_content_returns_existing_job(tmp_path: Path):
    db_path = tmp_path / "ingest_jobs.db"
    pdf_path = tmp_path / "a.pdf"
    pdf_path.write_bytes(b"same-content")

    svc = IngestionService(db_path)
    svc.ensure_worker_running = lambda: None  # type: ignore[method-assign]

    j1 = await svc.submit("p1", str(pdf_path), "pdf", title="doc")
    j2 = await svc.submit("p1", str(pdf_path), "pdf", title="doc")

    assert j1 == j2


@pytest.mark.asyncio
async def test_2_timeout_stall_reconcile_to_retrying(tmp_path: Path):
    db_path = tmp_path / "ingest_jobs.db"
    repo = JobRepository(db_path)
    await repo.ensure_table()
    job_id = await repo.create_job("p1", "/tmp/a.pdf", "pdf")

    async with aiosqlite.connect(db_path, timeout=15) as db:
        await db.execute(
            """
            UPDATE ingest_jobs
            SET status='running', phase='extracting', retry_count=0, max_retries=3,
                heartbeat_at=datetime('now','-120 seconds'),
                last_progress_at=datetime('now','-120 seconds'),
                updated_at=datetime('now','-120 seconds')
            WHERE id=?
            """,
            (job_id,),
        )
        await db.commit()

    changed = await repo.reconcile_stale_running_jobs(stale_seconds=30)
    assert changed == 1
    job = await repo.get_job(job_id)
    assert job is not None
    assert job["status"] == "retrying"
    assert job["error_code"] == "WORKER_CRASHED"


@pytest.mark.asyncio
async def test_3_retry_policy_exceeds_max_goes_failed(tmp_path: Path, monkeypatch):
    db_path = tmp_path / "ingest_jobs.db"
    repo = JobRepository(db_path)
    await repo.ensure_table()
    job_id = await repo.create_job("p1", "/tmp/a.pdf", "pdf")

    async with aiosqlite.connect(db_path, timeout=15) as db:
        await db.execute(
            """
            UPDATE ingest_jobs
            SET status='running', attempts=?, retry_count=?, phase='extracting', max_retries=3
            WHERE id=?
            """,
            (JobExecutor.MAX_ATTEMPTS, JobExecutor.MAX_ATTEMPTS, job_id),
        )
        await db.commit()

    class _BadHandler:
        async def ingest(self, job):
            await asyncio.sleep(0)
            raise RuntimeError("timeout while calling provider")

    import rag.source_handlers as sh

    monkeypatch.setattr(sh, "get_handler", lambda _st: _BadHandler())

    executor = JobExecutor(db_path)
    job = await repo.get_job(job_id)
    assert job is not None
    await executor.execute(job)

    row = await repo.get_job(job_id)
    assert row is not None
    assert row["status"] == "failed"


@pytest.mark.asyncio
async def test_4_lease_competition_only_one_owner(tmp_path: Path):
    db_path = tmp_path / "ingest_jobs.db"
    repo = JobRepository(db_path)
    await repo.ensure_table()

    owner1 = repo.build_owner_id(os.getpid())
    owner2 = repo.build_owner_id(1002)

    ok1 = await repo.acquire_lease("ingestion_worker", owner1, ttl_seconds=20)
    ok2 = await repo.acquire_lease("ingestion_worker", owner2, ttl_seconds=20)

    assert ok1 is True
    assert ok2 is False


def test_5_rate_limit_convergence_exhausted_on_retry_count():
    assert JobExecutor.is_rate_limit_exhausted(5, 0) is True
    assert JobExecutor.is_rate_limit_exhausted(4, 0) is False


def test_6_backoff_wall_time_exhausted_on_1800s():
    assert JobExecutor.is_rate_limit_exhausted(0, 1800) is True
    assert JobExecutor.is_rate_limit_exhausted(0, 1799) is False


def test_7_classify_rate_limit_exhausted_is_permanent():
    from rag.job_executor import classify_error, ErrorCategory

    cat = classify_error(RuntimeError("RATE_LIMIT_EXHAUSTED retries=5 wall=1900s"))
    assert cat == ErrorCategory.PERMANENT


@pytest.mark.asyncio
async def test_8_eta_uses_batch_moving_average(tmp_path: Path):
    db_path = tmp_path / "ingest_jobs.db"
    repo = JobRepository(db_path)
    await repo.ensure_table()
    job_id = await repo.create_job("p1", "/tmp/a.pdf", "pdf")
    await repo.update_status(job_id, "running", phase="extracting")
    await repo.create_or_replace_batches(
        job_id,
        [
            {"batch_no": 0, "page_start": 1, "page_end": 40, "actual_chunks": 20},
            {"batch_no": 1, "page_start": 41, "page_end": 80, "actual_chunks": 20},
            {"batch_no": 2, "page_start": 81, "page_end": 120, "actual_chunks": 20},
        ],
    )
    async with aiosqlite.connect(db_path, timeout=15) as db:
        await db.execute(
            """UPDATE ingest_batches
               SET status='done', started_at='2026-03-10 10:00:00', finished_at='2026-03-10 10:01:00'
               WHERE job_id=? AND batch_no=0""",
            (job_id,),
        )
        await db.execute(
            """UPDATE ingest_batches
               SET status='done', started_at='2026-03-10 10:01:00', finished_at='2026-03-10 10:02:00'
               WHERE job_id=? AND batch_no=1""",
            (job_id,),
        )
        await db.commit()

    svc = IngestionService(db_path)
    detail = await svc.get_job_detail(job_id)
    assert detail is not None
    # avg(60, 60) * remaining(1) = 60
    assert detail["eta_seconds"] == 60


@pytest.mark.asyncio
async def test_9_progress_counter_bumps_when_pct_increases(tmp_path: Path):
    db_path = tmp_path / "ingest_jobs.db"
    repo = JobRepository(db_path)
    await repo.ensure_table()
    job_id = await repo.create_job("p1", "/tmp/a.pdf", "pdf")

    await repo.update_progress(
        job_id,
        chunks_done=0,
        chunks_total=10,
        progress_pct=10.0,
        progress_stage="建立圖譜",
        eta_seconds=120,
    )
    j1 = await repo.get_job(job_id)
    assert j1 is not None
    c1 = int(j1.get("progress_counter") or 0)

    await repo.update_progress(
        job_id,
        chunks_done=0,
        chunks_total=10,
        progress_pct=20.0,
        progress_stage="建立圖譜",
        eta_seconds=100,
    )
    j2 = await repo.get_job(job_id)
    assert j2 is not None
    c2 = int(j2.get("progress_counter") or 0)
    assert c2 > c1
    assert (j2.get("last_progress_at") or "").strip() != ""


def test_10_pdf_batch_builder_always_splits_pipeline():
    page_chunks = [(i, f"p{i}") for i in range(1, 44)]
    batches = _build_pdf_batches(page_chunks, batch_size=20)
    assert len(batches) == 3
    assert len(batches[0]) == 20
    assert len(batches[1]) == 20
    assert len(batches[2]) == 3


@pytest.mark.asyncio
async def test_11_job_detail_phase_prefers_progress_stage(tmp_path: Path):
    db_path = tmp_path / "ingest_jobs.db"
    repo = JobRepository(db_path)
    await repo.ensure_table()
    job_id = await repo.create_job("p1", "/tmp/a.pdf", "pdf")
    await repo.update_status(job_id, "running", phase="extracting")
    await repo.update_progress(
        job_id,
        chunks_done=9,
        chunks_total=10,
        progress_pct=91.0,
        progress_stage="圖譜合併",
        eta_seconds=50,
    )
    svc = IngestionService(db_path)
    detail = await svc.get_job_detail(job_id)
    assert detail is not None
    assert detail["phase"] == "merging"


@pytest.mark.asyncio
async def test_12_done_status_clears_stale_error_fields(tmp_path: Path):
    db_path = tmp_path / "ingest_jobs.db"
    repo = JobRepository(db_path)
    await repo.ensure_table()
    job_id = await repo.create_job("p1", "/tmp/a.pdf", "pdf")
    await repo.update_status(
        job_id,
        "retrying",
        last_error="timeout",
        next_retry_at="2026-03-11 00:00:00",
        phase="retrying",
        error_code="PHASE_TIMEOUT",
        error_detail="batch=1 timeout=520s",
    )

    await repo.update_status(job_id, "done", phase="flushing")
    job = await repo.get_job(job_id)
    assert job is not None
    assert job["status"] == "done"
    assert job["error_code"] is None
    assert job["error_detail"] is None
