from __future__ import annotations

import asyncio
import json
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

router = APIRouter()


class IngestSubmitRequest(BaseModel):
    project_id: str
    source_path: str
    source_type: str
    title: str = ""
    payload: dict[str, Any] | None = None


class IngestActionResponse(BaseModel):
    status: str
    job_id: str | None = None


@router.post("/ingest")
async def submit_ingest(req: IngestSubmitRequest):
    from rag.ingestion_service import get_ingestion_service

    svc = get_ingestion_service()
    result = await svc.submit_meta(
        project_id=req.project_id,
        source=req.source_path,
        source_type=req.source_type,
        title=req.title,
        payload=req.payload or {},
    )
    return result


@router.get("/ingest/{job_id}")
async def get_ingest(job_id: str):
    from rag.ingestion_service import get_ingestion_service

    svc = get_ingestion_service()
    detail = await svc.get_job_detail(job_id)
    if not detail:
        raise HTTPException(status_code=404, detail="job not found")
    return detail


@router.post("/ingest/{job_id}/cancel", response_model=IngestActionResponse)
async def cancel_ingest(job_id: str):
    from rag.ingestion_service import get_ingestion_service

    svc = get_ingestion_service()
    ok = await svc.cancel_job(job_id)
    if not ok:
        raise HTTPException(status_code=404, detail="cancel failed or job not active")
    return IngestActionResponse(status="ok", job_id=job_id)


@router.post("/ingest/{job_id}/retry", response_model=IngestActionResponse)
async def retry_ingest(job_id: str):
    from rag.ingestion_service import get_ingestion_service

    svc = get_ingestion_service()
    ok = await svc.retry_job(job_id)
    if not ok:
        raise HTTPException(status_code=400, detail="job is not retryable")
    return IngestActionResponse(status="ok", job_id=job_id)


@router.get("/ingest/{job_id}/events")
async def ingest_events(job_id: str):
    from rag.ingestion_service import get_ingestion_service

    svc = get_ingestion_service()

    async def event_gen():
        last_payload = None
        while True:
            detail = await svc.get_job_detail(job_id)
            if not detail:
                yield "data: {\"event\":\"not_found\"}\n\n"
                return

            payload = {
                "event": "phase_change",
                "job_id": job_id,
                "status": detail.get("status"),
                "phase": detail.get("phase"),
                "phase_detail": detail.get("phase_detail"),
                "error_code": detail.get("error_code"),
                "error_detail": detail.get("error_detail"),
                "next_retry_at": detail.get("next_retry_at"),
            }
            encoded = json.dumps(payload, ensure_ascii=False)
            if encoded != last_payload:
                yield f"data: {encoded}\n\n"
                last_payload = encoded

            status = str(detail.get("status") or "")
            if status in ("done", "done_with_warning", "failed", "failed_recoverable"):
                done_payload = {
                    "event": "done" if status in ("done", "done_with_warning") else "failed",
                    "status": status,
                    "error_code": detail.get("error_code"),
                }
                yield f"data: {json.dumps(done_payload, ensure_ascii=False)}\n\n"
                return

            await asyncio.sleep(1.5)

    return StreamingResponse(event_gen(), media_type="text/event-stream")


@router.get("/ingest-metrics")
async def ingest_metrics(
    project_id: str | None = None,
    since_hours: int = 24,
    stale_seconds: int = 300,
):
    from paths import DATA_ROOT
    from rag.ingest_metrics import compute_ingest_metrics

    metrics = await compute_ingest_metrics(
        DATA_ROOT / "ingest_jobs.db",
        project_id=project_id,
        since_hours=since_hours,
        stale_seconds=stale_seconds,
    )
    return metrics.to_dict()
