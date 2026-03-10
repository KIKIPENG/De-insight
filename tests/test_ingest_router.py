from __future__ import annotations

import sys
from pathlib import Path

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
from main import app


class _FakeSvc:
    async def submit_meta(self, **kwargs):
        return {"job_id": "j1", "accepted": True, "deduped": False}

    async def get_job_detail(self, job_id: str):
        if job_id == "missing":
            return None
        return {
            "job_id": job_id,
            "status": "running:extracting",
            "phase": "extracting",
            "phase_detail": {"batch_current": 1, "batch_total": 2, "page_range": "1-40"},
            "error_code": None,
            "error_detail": None,
            "next_retry_at": None,
            "eta_seconds": 30,
        }

    async def cancel_job(self, job_id: str):
        return job_id == "j1"

    async def retry_job(self, job_id: str):
        return job_id == "j1"


def test_ingest_router_submit_and_status(monkeypatch):
    import rag.ingestion_service as mod

    monkeypatch.setattr(mod, "get_ingestion_service", lambda: _FakeSvc())
    client = TestClient(app)

    r = client.post(
        "/api/ingest",
        json={"project_id": "p1", "source_path": "/tmp/a.pdf", "source_type": "pdf", "title": "t"},
    )
    assert r.status_code == 200
    assert r.json()["job_id"] == "j1"

    r = client.get("/api/ingest/j1")
    assert r.status_code == 200
    assert r.json()["phase"] == "extracting"


def test_ingest_router_cancel_retry_and_404(monkeypatch):
    import rag.ingestion_service as mod

    monkeypatch.setattr(mod, "get_ingestion_service", lambda: _FakeSvc())
    client = TestClient(app)

    r = client.get("/api/ingest/missing")
    assert r.status_code == 404

    r = client.post("/api/ingest/j1/cancel")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"

    r = client.post("/api/ingest/j1/retry")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
