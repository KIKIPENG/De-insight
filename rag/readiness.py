"""IngestionReadinessService — 單一真相來源，判定 project 級別的可檢索狀態。

UI / search / chat 統一透過此 service 取得 readiness，
不再自己拼湊 has_knowledge() + job status + repair 條件。
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class ReadinessSnapshot:
    """Project-level readiness state.

    status_label 語意：
    - "ready":    有可檢索的 chunks，沒有進行中的 job → 正常檢索
    - "building": 有進行中或排隊中的 job（可能已有部分 chunks）→ 提示建圖中
    - "degraded": 有 chunks 但有終端失敗或維度異常 → 提示資料可能不完整
    - "empty":    完全沒有 chunks，也沒有待處理 job → 提示先匯入
    """

    has_ready_chunks: bool = False
    has_pending_jobs: bool = False
    has_running_jobs: bool = False
    has_terminal_failures: bool = False
    has_warning_jobs: bool = False
    last_error: str = ""
    status_label: str = "empty"


class IngestionReadinessService:
    """Read-only service that computes ReadinessSnapshot from ingest_jobs.db + vdb_chunks.json."""

    def __init__(self, jobs_db_path: Path | None = None) -> None:
        self._jobs_db_path = jobs_db_path

    def _resolve_jobs_db(self) -> Path:
        if self._jobs_db_path:
            return self._jobs_db_path
        from paths import DATA_ROOT
        return DATA_ROOT / "ingest_jobs.db"

    async def get_snapshot(self, project_id: str) -> ReadinessSnapshot:
        """Compute a fresh ReadinessSnapshot for the given project."""
        has_ready_chunks = self._check_vdb_chunks(project_id)
        pending, running, terminal_failures, warning_jobs, last_error = await self._check_jobs(project_id)

        status_label = self._compute_label(
            has_ready_chunks=has_ready_chunks,
            has_pending=pending,
            has_running=running,
            has_terminal=terminal_failures,
            has_warning=warning_jobs,
        )

        return ReadinessSnapshot(
            has_ready_chunks=has_ready_chunks,
            has_pending_jobs=pending,
            has_running_jobs=running,
            has_terminal_failures=terminal_failures,
            has_warning_jobs=warning_jobs,
            last_error=last_error,
            status_label=status_label,
        )

    def get_snapshot_sync(self, project_id: str) -> ReadinessSnapshot:
        """Synchronous variant for use in non-async contexts (e.g. panel refresh).

        Only checks vdb_chunks (no DB query for jobs).
        """
        has_ready_chunks = self._check_vdb_chunks(project_id)
        status = "ready" if has_ready_chunks else "empty"
        return ReadinessSnapshot(
            has_ready_chunks=has_ready_chunks,
            status_label=status,
        )

    @staticmethod
    def _check_vdb_chunks(project_id: str) -> bool:
        """Check if vdb_chunks.json has non-empty data array."""
        try:
            from paths import project_root
            if project_id == "default":
                from paths import DATA_ROOT
                wd = DATA_ROOT / "projects" / "default" / "lightrag"
            else:
                wd = project_root(project_id) / "lightrag"
            vdb_path = wd / "vdb_chunks.json"
            if not vdb_path.exists():
                return False
            payload = json.loads(vdb_path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                return len(payload.get("data", [])) > 0
        except Exception:
            pass
        return False

    async def _check_jobs(self, project_id: str) -> tuple[bool, bool, bool, bool, str]:
        """Query ingest_jobs.db for job states of this project.

        Returns (has_pending, has_running, has_terminal_failures, last_error).
        """
        db_path = self._resolve_jobs_db()
        if not db_path.exists():
            return False, False, False, False, ""

        try:
            import aiosqlite
            async with aiosqlite.connect(db_path, timeout=10) as db:
                db.row_factory = aiosqlite.Row

                # Pending (queued)
                async with db.execute(
                    "SELECT COUNT(*) as c FROM ingest_jobs WHERE project_id = ? AND status = 'queued'",
                    (project_id,),
                ) as cur:
                    row = await cur.fetchone()
                    has_pending = (row["c"] if row else 0) > 0

                # Running
                async with db.execute(
                    """SELECT COUNT(*) as c FROM ingest_jobs
                       WHERE project_id = ?
                         AND status LIKE 'running%'""",
                    (project_id,),
                ) as cur:
                    row = await cur.fetchone()
                    has_running = (row["c"] if row else 0) > 0

                # Terminal failures (failed with no next_retry_at)
                async with db.execute(
                    """SELECT COUNT(*) as c FROM ingest_jobs
                       WHERE project_id = ? AND status = 'failed' AND next_retry_at IS NULL""",
                    (project_id,),
                ) as cur:
                    row = await cur.fetchone()
                    has_terminal = (row["c"] if row else 0) > 0

                # Warnings completed (done_with_warning)
                async with db.execute(
                    "SELECT COUNT(*) as c FROM ingest_jobs WHERE project_id = ? AND status = 'done_with_warning'",
                    (project_id,),
                ) as cur:
                    row = await cur.fetchone()
                    has_warning = (row["c"] if row else 0) > 0

                # Last error
                last_error = ""
                if has_terminal or has_warning:
                    async with db.execute(
                        """SELECT last_error FROM ingest_jobs
                           WHERE project_id = ?
                             AND status IN ('failed', 'done_with_warning')
                           ORDER BY updated_at DESC LIMIT 1""",
                        (project_id,),
                    ) as cur:
                        row = await cur.fetchone()
                        if row:
                            last_error = row["last_error"] or ""

                return has_pending, has_running, has_terminal, has_warning, last_error

        except Exception as e:
            log.debug("Failed to check jobs for readiness: %s", e)
            return False, False, False, False, ""

    @staticmethod
    def _compute_label(
        *,
        has_ready_chunks: bool,
        has_pending: bool,
        has_running: bool,
        has_terminal: bool,
        has_warning: bool = False,
    ) -> str:
        if has_pending or has_running:
            return "building"
        if has_ready_chunks and has_terminal:
            return "degraded"
        if has_warning and not has_ready_chunks:
            return "degraded"
        if has_ready_chunks:
            return "ready"
        if has_terminal:
            return "degraded"
        return "empty"


# ── Module-level singleton ──────────────────────────────────────────

_readiness_service: IngestionReadinessService | None = None


def get_readiness_service() -> IngestionReadinessService:
    """Get or create the global IngestionReadinessService singleton."""
    global _readiness_service
    if _readiness_service is None:
        _readiness_service = IngestionReadinessService()
    return _readiness_service
