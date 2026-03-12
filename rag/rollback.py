"""Rollback snapshots for ingestion jobs."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from paths import DATA_ROOT, project_root

SNAPSHOT_ITEMS = ("lightrag", "documents", "conversations.db")


def rollback_root() -> Path:
    root = DATA_ROOT / "rollback_snapshots"
    root.mkdir(parents=True, exist_ok=True)
    return root


def snapshot_dir_for_job(job_id: str) -> Path:
    return rollback_root() / str(job_id)


def _manifest_path(snapshot_dir: Path) -> Path:
    return snapshot_dir / "manifest.json"


def _write_manifest(snapshot_dir: Path, payload: dict) -> None:
    _manifest_path(snapshot_dir).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _read_manifest(snapshot_dir: Path) -> dict:
    path = _manifest_path(snapshot_dir)
    if not path.exists():
        return {"items": []}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"items": []}


def _copy_item(src: Path, dst: Path) -> None:
    if src.is_dir():
        shutil.copytree(src, dst)
    else:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def prepare_job_snapshot(job: dict) -> Path:
    job_id = str(job["id"])
    pid = str(job["project_id"])
    root = project_root(pid)
    snap = snapshot_dir_for_job(job_id)
    if snap.exists():
        shutil.rmtree(snap)
    snap.mkdir(parents=True, exist_ok=True)

    items: list[dict] = []
    for name in SNAPSHOT_ITEMS:
        src = root / name
        item = {
            "name": name,
            "exists": src.exists(),
            "kind": "dir" if src.is_dir() else "file",
        }
        if src.exists():
            _copy_item(src, snap / name)
        items.append(item)
    _write_manifest(
        snap,
        {
            "job_id": job_id,
            "project_id": pid,
            "items": items,
        },
    )
    return snap


def restore_job_snapshot(job: dict) -> None:
    snapshot_dir = Path(str(job.get("rollback_snapshot_dir") or ""))
    if not snapshot_dir.exists():
        raise FileNotFoundError(f"rollback snapshot missing: {snapshot_dir}")
    manifest = _read_manifest(snapshot_dir)
    root = project_root(str(job["project_id"]))
    root.mkdir(parents=True, exist_ok=True)

    for item in manifest.get("items", []):
        name = str(item.get("name") or "")
        if not name:
            continue
        dst = root / name
        if dst.exists():
            if dst.is_dir():
                shutil.rmtree(dst)
            else:
                dst.unlink()
        if bool(item.get("exists")):
            src = snapshot_dir / name
            if src.exists():
                _copy_item(src, dst)


def cleanup_job_snapshot(job: dict) -> None:
    snapshot_dir = Path(str(job.get("rollback_snapshot_dir") or ""))
    if snapshot_dir.exists():
        shutil.rmtree(snapshot_dir)
