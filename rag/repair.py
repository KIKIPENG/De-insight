"""知識庫健康檢查與自動修復。

偵測條件：
- documents > 0 但 vdb_chunks 無資料
- doc_status 長期 processing/failed 導致不可檢索

修復步驟：
- 備份該 project 的 lightrag 到 timestamp 目錄
- 清理壞索引與壞狀態
- 依 documents 清單重建索引
- 驗證查詢可命中
"""

from __future__ import annotations

import json
import logging
import shutil
import time
from pathlib import Path

log = logging.getLogger(__name__)


def _lightrag_dir(project_id: str) -> Path:
    from paths import project_root
    return project_root(project_id) / "lightrag"


def diagnose(project_id: str) -> dict:
    """診斷知識庫健康狀態。回傳 {"healthy": bool, "issues": [...], ...}。"""
    result = {"healthy": True, "issues": [], "doc_count": 0, "vdb_count": 0}
    wd = _lightrag_dir(project_id)
    if not wd.exists():
        return result

    # Count documents in full_docs
    full_docs_path = wd / "kv_store_full_docs.json"
    if full_docs_path.exists():
        try:
            data = json.loads(full_docs_path.read_text(encoding="utf-8"))
            result["doc_count"] = len(data)
        except Exception:
            pass

    # Count vdb_chunks
    vdb_path = wd / "vdb_chunks.json"
    if vdb_path.exists():
        try:
            data = json.loads(vdb_path.read_text(encoding="utf-8"))
            result["vdb_count"] = len(data.get("data", []))
        except Exception:
            pass

    # Check doc_status for stuck entries
    status_path = wd / "kv_store_doc_status.json"
    stuck_count = 0
    if status_path.exists():
        try:
            statuses = json.loads(status_path.read_text(encoding="utf-8"))
            for doc_id, info in statuses.items():
                s = info.get("status", "") if isinstance(info, dict) else ""
                if s in ("processing", "failed"):
                    stuck_count += 1
        except Exception:
            pass

    # Issue: documents exist but vdb is empty
    if result["doc_count"] > 0 and result["vdb_count"] == 0:
        result["healthy"] = False
        result["issues"].append(
            f"知識庫有 {result['doc_count']} 份文件但向量索引為空（vdb_chunks=0）"
        )

    # Issue: stuck processing/failed
    if stuck_count > 0:
        result["healthy"] = False
        result["issues"].append(
            f"有 {stuck_count} 份文件卡在 processing/failed 狀態"
        )

    return result


def backup_lightrag(project_id: str) -> Path | None:
    """備份 lightrag 目錄到 timestamp 子目錄。回傳備份路徑。"""
    wd = _lightrag_dir(project_id)
    if not wd.exists():
        return None
    ts = time.strftime("%Y%m%d_%H%M%S")
    backup_dir = wd.parent / f"lightrag_backup_{ts}"
    try:
        shutil.copytree(wd, backup_dir)
        log.info("Backed up lightrag to %s", backup_dir)
        return backup_dir
    except Exception as e:
        log.error("Backup failed: %s", e)
        return None


def clear_lightrag_index(project_id: str) -> None:
    """清除 lightrag 目錄中的所有索引檔（保留目錄結構）。"""
    wd = _lightrag_dir(project_id)
    if not wd.exists():
        return
    for f in wd.iterdir():
        if f.is_file():
            f.unlink()
    log.info("Cleared lightrag index for project %s", project_id)


async def rebuild_from_documents(project_id: str, notify=None) -> dict:
    """從 conversations.db 的 documents 表重建知識庫索引。

    回傳 {"rebuilt": int, "failed": int, "errors": [...]}
    """
    from conversation.store import ConversationStore
    from paths import project_root

    store = ConversationStore(
        db_path=project_root(project_id) / "conversations.db"
    )
    docs = await store.list_documents(project_id)

    result = {"rebuilt": 0, "failed": 0, "errors": []}
    if not docs:
        return result

    from rag.knowledge_graph import (
        _ensure_initialized,
        _clear_failed,
        _flush_and_check,
        reset_rag,
    )

    # Force new RAG instance with clean state
    reset_rag()

    for doc in docs:
        source = doc.get("source_path", "")
        source_type = doc.get("source_type", "pdf")
        title = doc.get("title", "")
        if notify:
            notify(f"重建中：{title[:30]}…")

        try:
            if source_type == "pdf" and Path(source).exists():
                from rag.knowledge_graph import insert_pdf
                await insert_pdf(source, project_id=project_id, title=title)
            elif source_type in ("url", "doi") and source.startswith("http"):
                from rag.knowledge_graph import insert_url
                await insert_url(source, project_id=project_id, title=title)
            else:
                # Try saved document path
                from paths import ensure_project_dirs
                doc_dir = ensure_project_dirs(project_id) / "documents"
                candidates = list(doc_dir.glob("*.pdf"))
                matched = [c for c in candidates if title and title[:10] in c.stem]
                if matched:
                    from rag.knowledge_graph import insert_pdf
                    await insert_pdf(str(matched[0]), project_id=project_id, title=title)
                else:
                    result["errors"].append(f"找不到檔案: {title}")
                    result["failed"] += 1
                    continue
            result["rebuilt"] += 1
        except Exception as e:
            result["errors"].append(f"{title}: {e}")
            result["failed"] += 1

    return result


async def auto_repair(project_id: str, notify=None) -> dict:
    """自動修復流程：診斷 → 備份 → 清理 → 重建 → 驗證。

    回傳 {"status": "ok"|"repaired"|"failed", "diagnosis": {...}, "repair": {...}, ...}
    """
    diag = diagnose(project_id)
    report = {"status": "ok", "diagnosis": diag}

    if diag["healthy"]:
        return report

    log.info("Knowledge base unhealthy for project %s: %s", project_id, diag["issues"])
    if notify:
        notify(f"偵測到知識庫問題：{'; '.join(diag['issues'][:2])}")

    # Backup
    backup_path = backup_lightrag(project_id)
    report["backup_path"] = str(backup_path) if backup_path else None

    # Clear
    clear_lightrag_index(project_id)
    if notify:
        notify("已清理壞索引，開始重建…")

    # Rebuild
    rebuild_result = await rebuild_from_documents(project_id, notify=notify)
    report["repair"] = rebuild_result

    # Verify
    from rag.knowledge_graph import has_knowledge
    if has_knowledge(project_id=project_id):
        report["status"] = "repaired"
        log.info("Repair succeeded for project %s: %s", project_id, rebuild_result)
        if notify:
            notify(f"知識庫修復完成（重建 {rebuild_result['rebuilt']} 份）")
    else:
        report["status"] = "failed"
        log.error("Repair failed for project %s: %s", project_id, rebuild_result)
        if notify:
            notify(f"知識庫修復失敗（{rebuild_result['failed']} 份失敗）")

    return report


def any_project_has_images() -> bool:
    """檢查全域所有專案是否有圖片資料（images table 有 row）。"""
    from paths import PROJECTS_DIR
    if not PROJECTS_DIR.exists():
        return False
    for proj_dir in PROJECTS_DIR.iterdir():
        if not proj_dir.is_dir():
            continue
        lance_dir = proj_dir / "lancedb"
        if not lance_dir.exists():
            continue
        try:
            import lancedb as _ldb
            db = _ldb.connect(str(lance_dir))
            if "images" in db.table_names():
                tbl = db.open_table("images")
                if tbl.count_rows() > 0:
                    return True
        except Exception:
            continue
    return False
