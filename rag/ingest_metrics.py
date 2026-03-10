from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path

import aiosqlite


@dataclass(frozen=True)
class IngestMetrics:
    total_jobs: int
    terminal_jobs: int
    completed_jobs: int
    failed_jobs: int
    active_jobs: int
    stuck_jobs: int
    completion_rate: float
    stuck_rate: float
    p50_seconds: int | None
    p95_seconds: int | None
    sampled_durations: int

    def to_dict(self) -> dict:
        return asdict(self)


def _parse_db_time(value: str | None) -> datetime | None:
    if not value:
        return None
    s = str(value).strip()
    if not s:
        return None
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(s[:19], fmt)
        except Exception:
            continue
    return None


def _percentile(values: list[float], p: float) -> int | None:
    if not values:
        return None
    if len(values) == 1:
        return int(round(values[0]))
    sorted_vals = sorted(values)
    pos = max(0.0, min(1.0, p)) * (len(sorted_vals) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = pos - lo
    v = sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * frac
    return int(round(v))


async def compute_ingest_metrics(
    db_path: Path,
    *,
    project_id: str | None = None,
    since_hours: int | None = None,
    stale_seconds: int = 300,
) -> IngestMetrics:
    where = []
    params: list[str] = []
    if project_id:
        where.append("project_id = ?")
        params.append(project_id)
    if since_hours and since_hours > 0:
        cutoff = (datetime.now() - timedelta(hours=int(since_hours))).strftime("%Y-%m-%d %H:%M:%S")
        where.append("created_at >= ?")
        params.append(cutoff)
    where_sql = f"WHERE {' AND '.join(where)}" if where else ""

    rows: list[dict] = []
    if db_path.exists():
        try:
            async with aiosqlite.connect(db_path, timeout=15) as db:
                db.row_factory = aiosqlite.Row
                async with db.execute(
                    f"SELECT * FROM ingest_jobs {where_sql} ORDER BY created_at ASC",
                    tuple(params),
                ) as cur:
                    rows = [dict(r) for r in await cur.fetchall()]
        except aiosqlite.OperationalError:
            rows = []

    terminal_statuses = {"done", "done_with_warning", "failed", "failed_recoverable"}
    completed_statuses = {"done", "done_with_warning"}
    failed_statuses = {"failed", "failed_recoverable"}

    total_jobs = len(rows)
    terminal_jobs = len([r for r in rows if str(r.get("status") or "") in terminal_statuses])
    completed_jobs = len([r for r in rows if str(r.get("status") or "") in completed_statuses])
    failed_jobs = len([r for r in rows if str(r.get("status") or "") in failed_statuses])
    active_rows = [
        r
        for r in rows
        if (str(r.get("status") or "") == "queued")
        or (str(r.get("status") or "") == "retrying")
        or str(r.get("status") or "").startswith("running")
    ]
    active_jobs = len(active_rows)

    now = datetime.now()
    stuck_jobs = 0
    for r in active_rows:
        hb = _parse_db_time(r.get("heartbeat_at"))
        lp = _parse_db_time(r.get("last_progress_at"))
        ref = hb or lp or _parse_db_time(r.get("updated_at")) or _parse_db_time(r.get("created_at"))
        if not ref:
            continue
        if (now - ref).total_seconds() > float(max(1, int(stale_seconds))):
            stuck_jobs += 1

    durations: list[float] = []
    for r in rows:
        status = str(r.get("status") or "")
        if status not in terminal_statuses:
            continue
        st = _parse_db_time(r.get("started_at")) or _parse_db_time(r.get("created_at"))
        ed = _parse_db_time(r.get("updated_at"))
        if not st or not ed:
            continue
        delta = (ed - st).total_seconds()
        if delta >= 0:
            durations.append(delta)

    p50 = _percentile(durations, 0.5)
    p95 = _percentile(durations, 0.95)
    completion_rate = 0.0 if terminal_jobs == 0 else completed_jobs / terminal_jobs
    stuck_rate = 0.0 if total_jobs == 0 else stuck_jobs / total_jobs
    return IngestMetrics(
        total_jobs=total_jobs,
        terminal_jobs=terminal_jobs,
        completed_jobs=completed_jobs,
        failed_jobs=failed_jobs,
        active_jobs=active_jobs,
        stuck_jobs=stuck_jobs,
        completion_rate=completion_rate,
        stuck_rate=stuck_rate,
        p50_seconds=p50,
        p95_seconds=p95,
        sampled_durations=len(durations),
    )
