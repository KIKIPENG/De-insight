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

# Prevent repair loops: track projects currently being repaired
_repairing: set[str] = set()
# Track projects that already attempted repair (prevent re-trigger on failure)
_repair_attempted: set[str] = set()


def _lightrag_dir(project_id: str) -> Path:
    from paths import project_root
    return project_root(project_id) / "lightrag"


def _detect_vdb_dim(working_dir: Path) -> int | None:
    """讀取 vdb_chunks.json 中的 embedding_dim。"""
    vdb_path = working_dir / "vdb_chunks.json"
    if not vdb_path.exists():
        return None
    try:
        data = json.loads(vdb_path.read_text(encoding="utf-8"))
        dim = data.get("embedding_dim")
        return int(dim) if dim is not None else None
    except Exception:
        return None


def _get_expected_embed_dim() -> int:
    """取得目前設定的 embedding 維度 — v0.7 固定 1024。"""
    return 1024


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

    # Check embedding dimension mismatch
    existing_dim = _detect_vdb_dim(wd)
    expected_dim = _get_expected_embed_dim()
    if existing_dim is not None and existing_dim != expected_dim:
        result["healthy"] = False
        result["dim_mismatch"] = {"existing": existing_dim, "expected": expected_dim}
        result["issues"].append(
            f"向量維度不符：索引為 {existing_dim} 維，目前設定為 {expected_dim} 維"
        )

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

    # Issue: stuck processing/failed（僅記錄，不觸發完整 repair）
    if stuck_count > 0:
        result["stuck_count"] = stuck_count
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
    """清除 lightrag 目錄中的索引檔，保留原始文件內容供重建用。"""
    wd = _lightrag_dir(project_id)
    if not wd.exists():
        return
    # 保留原始文件內容（rebuild 需要），只刪除索引和狀態檔
    _PRESERVE = {"kv_store_full_docs.json", "kv_store_llm_response_cache.json"}
    for f in wd.iterdir():
        if f.is_file() and f.name not in _PRESERVE:
            f.unlink()
    log.info("Cleared lightrag index for project %s (preserved full_docs)", project_id)


async def rebuild_from_documents(project_id: str, notify=None) -> dict:
    """重建知識庫索引。

    優先從 conversations.db 讀取文件清單；若為空則回退到
    kv_store_full_docs.json 直接重新 insert 原始文字。

    回傳 {"rebuilt": int, "failed": int, "errors": [...]}
    """
    from conversation.store import ConversationStore
    from paths import project_root

    store = ConversationStore(
        db_path=project_root(project_id) / "conversations.db"
    )
    docs = await store.list_documents(project_id)

    result = {"rebuilt": 0, "failed": 0, "errors": []}

    from rag.knowledge_graph import (
        _ensure_initialized,
        _clear_failed,
        _flush_and_check,
        reset_rag,
    )

    # ── 路徑 A：從 conversations.db 重建（有完整來源資訊）──
    if docs:
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

    # ── 路徑 B：從 kv_store_full_docs.json 回退重建 ──
    wd = _lightrag_dir(project_id)
    full_docs_path = wd / "kv_store_full_docs.json"
    if not full_docs_path.exists():
        return result

    try:
        raw_docs = json.loads(full_docs_path.read_text(encoding="utf-8"))
    except Exception:
        return result

    if not raw_docs:
        return result

    log.info("Rebuilding from kv_store_full_docs.json (%d docs)", len(raw_docs))
    reset_rag()

    for doc_id, content in raw_docs.items():
        text = content.get("content", "") if isinstance(content, dict) else str(content)
        if not text or len(text.strip()) < 20:
            continue
        title = doc_id[:30]
        if notify:
            notify(f"重建中：{title}…")
        try:
            from rag.knowledge_graph import insert_text
            await insert_text(text, project_id=project_id)
            result["rebuilt"] += 1
        except Exception as e:
            result["errors"].append(f"{title}: {e}")
            result["failed"] += 1

    return result


async def auto_repair(project_id: str, notify=None, triggering_error=None, _skip_policy_check=False) -> dict:
    """自動修復流程：診斷 → 備份 → 清理 → 重建 → 驗證。

    回傳 {"status": "ok"|"repaired"|"failed", "diagnosis": {...}, "repair": {...}, ...}

    triggering_error: 觸發修復的原始錯誤（若為 transient，跳過 destructive repair）
    _skip_policy_check: 由 CorruptionPolicy 呼叫時跳過 policy 檢查
    """
    # Prevent concurrent/looping repairs
    if project_id in _repairing:
        return {"status": "skip", "reason": "already repairing"}
    if project_id in _repair_attempted:
        return {"status": "skip", "reason": "already attempted"}

    # Policy check: transient errors should not trigger destructive repair
    if not _skip_policy_check and triggering_error is not None:
        from rag.repair_policy import TransientErrorPolicy
        diag_for_policy = diagnose(project_id)
        if TransientErrorPolicy().should_repair(diag_for_policy, triggering_error) is False:
            # TransientErrorPolicy always returns False — if error is transient, skip repair
            transient_kws = ("fds_to_keep", "timeout", "rate limit", "connection")
            if any(kw in str(triggering_error).lower() for kw in transient_kws):
                return {"status": "skip", "reason": "transient_error"}

    diag = diagnose(project_id)
    report = {"status": "ok", "diagnosis": diag}

    if diag["healthy"]:
        # 如果之前修復失敗但現在健康了（手動修復），清除記錄
        _repair_attempted.discard(project_id)
        return report

    _repairing.add(project_id)
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
            errors = rebuild_result.get("errors", [])
            hint = errors[0][:60] if errors else "未知原因"
            notify(f"知識庫修復失敗：{hint}。請手動重新匯入文件。")

    # C1: Post-verify after repair
    if report["status"] == "repaired":
        try:
            from rag.knowledge_graph import _post_verify_insert
            pv = await _post_verify_insert(project_id)
            if pv:
                report["post_verify_warning"] = pv
                log.warning("Post-repair verify warning: %s", pv)
        except Exception:
            pass

    _repairing.discard(project_id)
    _repair_attempted.add(project_id)
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
