from __future__ import annotations

import asyncio
import os
import shutil
from pathlib import Path

import aiosqlite
import pytest

from rag.ingestion_service import IngestionService
from rag.job_executor import JobExecutor
from rag.job_repository import JobRepository
from rag.knowledge_graph import _build_pdf_batches
from rag.rollback import cleanup_job_snapshot, prepare_job_snapshot, restore_job_snapshot


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


@pytest.mark.asyncio
async def test_13_rollback_snapshot_restores_project_files(tmp_path: Path, monkeypatch):
    import paths
    import rag.rollback as rollback_mod

    data_root = tmp_path / "data"
    projects_dir = data_root / "projects"
    monkeypatch.setattr(paths, "DATA_ROOT", data_root)
    monkeypatch.setattr(paths, "PROJECTS_DIR", projects_dir)
    monkeypatch.setattr(rollback_mod, "DATA_ROOT", data_root)

    project_id = "p1"
    root = paths.ensure_project_dirs(project_id)
    (root / "lightrag" / "state.txt").write_text("clean", encoding="utf-8")
    (root / "documents" / "doc.txt").write_text("doc-clean", encoding="utf-8")
    (root / "conversations.db").write_text("conv-clean", encoding="utf-8")
    job = {"id": "j1", "project_id": project_id}

    snap = prepare_job_snapshot(job)
    (root / "lightrag" / "state.txt").write_text("dirty", encoding="utf-8")
    (root / "documents" / "extra.txt").write_text("extra", encoding="utf-8")
    (root / "conversations.db").write_text("conv-dirty", encoding="utf-8")

    restore_job_snapshot({**job, "rollback_snapshot_dir": str(snap)})

    assert (root / "lightrag" / "state.txt").read_text(encoding="utf-8") == "clean"
    assert (root / "documents" / "doc.txt").read_text(encoding="utf-8") == "doc-clean"
    assert not (root / "documents" / "extra.txt").exists()
    assert (root / "conversations.db").read_text(encoding="utf-8") == "conv-clean"

    cleanup_job_snapshot({**job, "rollback_snapshot_dir": str(snap)})
    assert not snap.exists()


@pytest.mark.asyncio
async def test_14_transient_failure_restores_snapshot_and_keeps_retry(tmp_path: Path, monkeypatch):
    import paths
    import rag.rollback as rollback_mod
    import rag.source_handlers as sh

    data_root = tmp_path / "data"
    projects_dir = data_root / "projects"
    monkeypatch.setattr(paths, "DATA_ROOT", data_root)
    monkeypatch.setattr(paths, "PROJECTS_DIR", projects_dir)
    monkeypatch.setattr(rollback_mod, "DATA_ROOT", data_root)

    project_id = "p1"
    root = paths.ensure_project_dirs(project_id)
    (root / "lightrag" / "state.txt").write_text("clean", encoding="utf-8")

    db_path = tmp_path / "ingest_jobs.db"
    repo = JobRepository(db_path)
    await repo.ensure_table()
    job_id = await repo.create_job(project_id, "/tmp/a.txt", "txt", title="doc")
    job = await repo.get_job(job_id)
    assert job is not None

    class _TransientHandler:
        async def ingest(self, _job):
            (root / "lightrag" / "state.txt").write_text("dirty", encoding="utf-8")
            raise RuntimeError("timeout while calling provider")

    monkeypatch.setattr(sh, "get_handler", lambda _st: _TransientHandler())

    executor = JobExecutor(db_path)
    await executor.execute(job)

    row = await repo.get_job(job_id)
    assert row is not None
    assert row["status"] == "retrying"
    assert int(row.get("rollback_pending") or 0) == 0
    assert str(row.get("rollback_snapshot_dir") or "").strip() != ""
    assert (root / "lightrag" / "state.txt").read_text(encoding="utf-8") == "clean"


@pytest.mark.asyncio
async def test_15_abort_and_rollback_incomplete_restores_only_pending_jobs(tmp_path: Path, monkeypatch):
    import paths
    import rag.rollback as rollback_mod

    data_root = tmp_path / "data"
    projects_dir = data_root / "projects"
    monkeypatch.setattr(paths, "DATA_ROOT", data_root)
    monkeypatch.setattr(paths, "PROJECTS_DIR", projects_dir)
    monkeypatch.setattr(rollback_mod, "DATA_ROOT", data_root)

    db_path = tmp_path / "ingest_jobs.db"
    repo = JobRepository(db_path)
    await repo.ensure_table()

    root_running = paths.ensure_project_dirs("p-running")
    (root_running / "lightrag" / "state.txt").write_text("clean-running", encoding="utf-8")
    running_id = await repo.create_job("p-running", "/tmp/running.txt", "txt", title="running")
    running_job = await repo.get_job(running_id)
    assert running_job is not None
    running_snap = prepare_job_snapshot(running_job)
    (root_running / "lightrag" / "state.txt").write_text("dirty-running", encoding="utf-8")
    await repo.set_rollback_snapshot(running_id, str(running_snap))
    await repo.set_rollback_pending(running_id, True)
    await repo.update_status(
        running_id,
        "running",
        phase="extracting",
        error_code="STALL_DETECTED",
        error_detail="still running",
    )

    root_retry = paths.ensure_project_dirs("p-retrying")
    (root_retry / "lightrag" / "state.txt").write_text("stable-retrying", encoding="utf-8")
    retry_id = await repo.create_job("p-retrying", "/tmp/retrying.txt", "txt", title="retrying")
    retry_job = await repo.get_job(retry_id)
    assert retry_job is not None
    retry_snap = prepare_job_snapshot(retry_job)
    await repo.set_rollback_snapshot(retry_id, str(retry_snap))
    await repo.update_status(
        retry_id,
        "retrying",
        last_error="timeout",
        next_retry_at="2099-01-01 00:00:00",
        phase="retrying",
        error_code="PHASE_TIMEOUT",
        error_detail="already restored",
    )

    queued_id = await repo.create_job("p-queued", "/tmp/queued.txt", "txt", title="queued")

    svc = IngestionService(db_path)
    monkeypatch.setattr(svc, "stop_worker", lambda: None)
    aborted = await svc.abort_and_rollback_incomplete()

    assert {row["id"] for row in aborted} == {running_id, retry_id, queued_id}

    running_row = await repo.get_job(running_id)
    retry_row = await repo.get_job(retry_id)
    queued_row = await repo.get_job(queued_id)
    assert running_row is not None and retry_row is not None and queued_row is not None
    assert running_row["status"] == "failed"
    assert retry_row["status"] == "failed"
    assert queued_row["status"] == "failed"
    assert running_row["error_code"] == "CANCELLED_ON_TUI_EXIT"
    assert retry_row["error_code"] == "CANCELLED_ON_TUI_EXIT"
    assert queued_row["error_code"] == "CANCELLED_ON_TUI_EXIT"
    assert (root_running / "lightrag" / "state.txt").read_text(encoding="utf-8") == "clean-running"
    assert (root_retry / "lightrag" / "state.txt").read_text(encoding="utf-8") == "stable-retrying"
    assert str((await repo.get_job(running_id))["rollback_snapshot_dir"] or "") == ""
    assert str((await repo.get_job(retry_id))["rollback_snapshot_dir"] or "") == ""
    assert not running_snap.exists()
    assert not retry_snap.exists()


@pytest.mark.asyncio
async def test_16_get_active_progress_filters_by_project(tmp_path: Path):
    db_path = tmp_path / "ingest_jobs.db"
    repo = JobRepository(db_path)
    await repo.ensure_table()
    j1 = await repo.create_job("p1", "/tmp/a.pdf", "pdf")
    j2 = await repo.create_job("p2", "/tmp/b.pdf", "pdf")
    await repo.update_status(j1, "running", phase="extracting")
    await repo.update_status(j2, "queued", phase="queued")

    svc = IngestionService(db_path)
    only_p1 = await svc.get_active_progress("p1")

    assert [job["id"] for job in only_p1] == [j1]
