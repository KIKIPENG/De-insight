from __future__ import annotations

import sys
from pathlib import Path

import aiosqlite
import pytest
from fastapi.testclient import TestClient

from rag.ingest_metrics import compute_ingest_metrics
from rag.job_repository import JobRepository

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
from main import app  # noqa: E402


@pytest.mark.asyncio
async def test_compute_ingest_metrics_basic(tmp_path: Path):
    db_path = tmp_path / "ingest_jobs.db"
    repo = JobRepository(db_path)
    await repo.ensure_table()

    done = await repo.create_job("p1", "/tmp/a.pdf", "pdf")
    failed = await repo.create_job("p1", "/tmp/b.pdf", "pdf")
    running = await repo.create_job("p1", "/tmp/c.pdf", "pdf")

    async with aiosqlite.connect(db_path, timeout=15) as db:
        await db.execute(
            """UPDATE ingest_jobs
               SET status='done', started_at='2026-03-10 10:00:00', updated_at='2026-03-10 10:01:40'
               WHERE id=?""",
            (done,),
        )
        await db.execute(
            """UPDATE ingest_jobs
               SET status='failed', started_at='2026-03-10 10:00:00', updated_at='2026-03-10 10:03:20'
               WHERE id=?""",
            (failed,),
        )
        await db.execute(
            """UPDATE ingest_jobs
               SET status='running:extracting', heartbeat_at=datetime('now','-600 seconds')
               WHERE id=?""",
            (running,),
        )
        await db.commit()

    m = await compute_ingest_metrics(db_path, project_id="p1", since_hours=0, stale_seconds=300)
    assert m.total_jobs == 3
    assert m.terminal_jobs == 2
    assert m.completed_jobs == 1
    assert m.failed_jobs == 1
    assert m.active_jobs == 1
    assert m.stuck_jobs == 1
    assert m.completion_rate == 0.5
    assert m.p50_seconds == 150  # midpoint of 100 and 200
    assert m.p95_seconds is not None


def test_ingest_metrics_api(monkeypatch):
    class _FakeMetrics:
        def to_dict(self):
            return {
                "total_jobs": 20,
                "terminal_jobs": 20,
                "completed_jobs": 19,
                "failed_jobs": 1,
                "active_jobs": 0,
                "stuck_jobs": 0,
                "completion_rate": 0.95,
                "stuck_rate": 0.0,
                "p50_seconds": 120,
                "p95_seconds": 320,
                "sampled_durations": 20,
            }

    async def _fake_compute(*args, **kwargs):
        return _FakeMetrics()

    import rag.ingest_metrics as mod

    monkeypatch.setattr(mod, "compute_ingest_metrics", _fake_compute)
    client = TestClient(app)
    r = client.get("/api/ingest-metrics?project_id=p1&since_hours=48")
    assert r.status_code == 200
    body = r.json()
    assert body["completion_rate"] == 0.95
    assert body["p95_seconds"] == 320
