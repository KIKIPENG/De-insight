#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rag.ingest_metrics import compute_ingest_metrics  # noqa: E402
from paths import DATA_ROOT  # noqa: E402


async def _main() -> int:
    parser = argparse.ArgumentParser(
        description="Compute ingestion benchmark metrics from ingest_jobs.db"
    )
    parser.add_argument(
        "--db-path",
        default=str(DATA_ROOT / "ingest_jobs.db"),
        help="Path to ingest_jobs.db",
    )
    parser.add_argument("--project-id", default=None, help="Filter by project id")
    parser.add_argument(
        "--since-hours",
        type=int,
        default=24,
        help="Only include jobs created in last N hours (0=all)",
    )
    parser.add_argument(
        "--stale-seconds",
        type=int,
        default=300,
        help="Active job older than this is counted as stuck",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output JSON only",
    )
    args = parser.parse_args()

    metrics = await compute_ingest_metrics(
        Path(args.db_path),
        project_id=args.project_id,
        since_hours=None if args.since_hours <= 0 else args.since_hours,
        stale_seconds=args.stale_seconds,
    )
    payload = metrics.to_dict()

    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    print("Ingestion Benchmark")
    print(f"- db_path: {args.db_path}")
    if args.project_id:
        print(f"- project_id: {args.project_id}")
    print(f"- since_hours: {args.since_hours}")
    print(f"- total_jobs: {payload['total_jobs']}")
    print(f"- terminal_jobs: {payload['terminal_jobs']}")
    print(f"- completed_jobs: {payload['completed_jobs']}")
    print(f"- failed_jobs: {payload['failed_jobs']}")
    print(f"- active_jobs: {payload['active_jobs']}")
    print(f"- stuck_jobs: {payload['stuck_jobs']}")
    print(f"- completion_rate: {payload['completion_rate']:.3f}")
    print(f"- stuck_rate: {payload['stuck_rate']:.3f}")
    print(f"- p50_seconds: {payload['p50_seconds']}")
    print(f"- p95_seconds: {payload['p95_seconds']}")
    print(f"- sampled_durations: {payload['sampled_durations']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_main()))
