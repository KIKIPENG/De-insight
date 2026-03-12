#!/usr/bin/env python3
"""De-insight 檢索管線全鏈診斷工具。

一次檢查所有環節：
1. 知識庫向量索引（vdb_chunks）
2. LightRAG naive 搜尋
3. ClaimStore claims 數量 + 內容
4. core.retriever 結構搜尋
5. run_thinking_pipeline 完整結果

用法：
  python3 tools/diagnose_pipeline.py <project_id> "你的測試問題"
  python3 tools/diagnose_pipeline.py   # 列出所有專案
"""

import asyncio
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")


def _sep(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def list_projects():
    from paths import PROJECTS_DIR
    if not PROJECTS_DIR.exists():
        print("  (no projects directory)")
        return
    for d in sorted(PROJECTS_DIR.iterdir()):
        if not d.is_dir():
            continue
        print(f"  {d.name}")


async def diagnose(project_id: str, query: str):
    # ── 1. VDB Chunks ──
    _sep("1. VDB Chunks（向量索引）")
    from rag.vdb_utils import find_vdb_chunks_file
    vdb_path = find_vdb_chunks_file(project_id)
    if not vdb_path or not vdb_path.exists():
        print("  ❌ vdb_chunks.json 不存在！知識庫為空。")
        return
    chunks_data = json.loads(vdb_path.read_text(encoding="utf-8"))
    all_chunks = chunks_data.get("data", [])
    print(f"  ✓ {len(all_chunks)} 個 chunks")
    for i, c in enumerate(all_chunks[:5]):
        text = ""
        if isinstance(c, dict):
            text = c.get("content", "") or c.get("text", "")
        elif isinstance(c, str):
            text = c
        if text:
            print(f"    [{i}] {text[:80]}...")
    if len(all_chunks) > 5:
        print(f"    ... 還有 {len(all_chunks) - 5} 個")

    # ── 2. has_knowledge check ──
    _sep("2. has_knowledge 檢查")
    from rag.knowledge_graph import has_knowledge
    hk = has_knowledge(project_id=project_id)
    print(f"  has_knowledge: {hk}")

    # ── 3. LightRAG naive 搜尋 ──
    _sep("3. LightRAG naive 搜尋（純向量，不需 LLM）")
    try:
        from rag.knowledge_graph import query_knowledge, _is_no_context_result
        result, sources = await query_knowledge(
            query, mode="naive", context_only=True,
            project_id=project_id, chunk_top_k=5,
        )
        is_empty = _is_no_context_result(result)
        print(f"  結果長度: {len(result)} chars")
        print(f"  _is_no_context_result: {is_empty}")
        print(f"  sources: {len(sources)}")
        if result and not is_empty:
            print(f"  前200字: {result[:200]}...")
        else:
            print("  ⚠️  naive 搜尋無結果！向量相似度太低或索引有問題。")
        for s in sources[:3]:
            print(f"    source: {s.get('title', '')[:50]}")
    except Exception as e:
        print(f"  ❌ naive 搜尋失敗: {type(e).__name__}: {e}")

    # ── 4. ClaimStore ──
    _sep("4. ClaimStore（結構化 claims）")
    try:
        from core.stores.claim_store import ClaimStore
        store = ClaimStore(project_id=project_id)
        claims = await store.list_by_project(project_id, limit=100)
        print(f"  Claims: {len(claims)}")
        for c in claims[:5]:
            patterns = ", ".join(c.abstract_patterns or [])
            values = ", ".join(c.value_axes or [])
            print(f"    claim: {c.core_claim[:60]}")
            print(f"      patterns: [{patterns}]")
            print(f"      values: [{values}]")
        if len(claims) > 5:
            print(f"    ... 還有 {len(claims) - 5} 個")
    except Exception as e:
        print(f"  ❌ ClaimStore 讀取失敗: {type(e).__name__}: {e}")

    # ── 5. ClaimStore structural search ──
    _sep("5. ClaimStore 結構搜尋")
    try:
        from core.stores.claim_store import ClaimStore
        store = ClaimStore(project_id=project_id)
        # Extract some keywords from query for structural search
        test_terms = [query[:15]] if len(query) > 15 else [query]
        # Also try bigrams for Chinese
        if any('\u4e00' <= ch <= '\u9fff' for ch in query):
            chars = [ch for ch in query if '\u4e00' <= ch <= '\u9fff']
            bigrams = [chars[i] + chars[i+1] for i in range(min(8, len(chars)-1))]
            test_terms.extend(bigrams[:6])
        print(f"  搜尋 terms: {test_terms}")
        results = await store.search_by_structure(
            value_axes=test_terms,
            abstract_patterns=test_terms,
            limit=5,
        )
        print(f"  結果: {len(results)} claims")
        for c in results[:3]:
            print(f"    → {c.core_claim[:60]}")
    except Exception as e:
        print(f"  ❌ 結構搜尋失敗: {type(e).__name__}: {e}")

    # ── 6. core.retriever bridge search ──
    _sep("6. core.retriever（bridge 檢索）")
    try:
        from core.retriever import Retriever
        from core.stores.claim_store import ClaimStore
        from core.schemas import RetrievalPlan

        plan = RetrievalPlan(
            project_id=project_id,
            query_mode="deep",
            thought_summary="",
            concept_queries=[query],
        )
        claim_store = ClaimStore(project_id=project_id)
        retriever = Retriever(project_id=project_id, claim_store=claim_store)
        bridge_result = await retriever.retrieve(plan, query)
        n_claims = len(bridge_result.claims) if bridge_result else 0
        n_bridges = len(bridge_result.bridges) if bridge_result else 0
        print(f"  claims: {n_claims}, bridges: {n_bridges}")
        if bridge_result and bridge_result.claims:
            for c in bridge_result.claims[:3]:
                print(f"    claim: {c.core_claim[:60]}")
        else:
            print("  ⚠️  bridge 檢索無結果")
    except Exception as e:
        print(f"  ❌ bridge 檢索失敗: {type(e).__name__}: {e}")

    # ── 7. Readiness ──
    _sep("7. Readiness 狀態")
    try:
        from rag.readiness import get_readiness_service
        svc = get_readiness_service()
        snap = await svc.get_snapshot(project_id)
        print(f"  status_label: {snap.status_label}")
        print(f"  has_ready_chunks: {snap.has_ready_chunks}")
        if snap.last_error:
            print(f"  last_error: {snap.last_error}")
    except Exception as e:
        print(f"  ❌ readiness 檢查失敗: {type(e).__name__}: {e}")

    # ── 8. Full pipeline ──
    _sep("8. run_thinking_pipeline（完整管線）")
    try:
        from rag.pipeline import run_thinking_pipeline
        from paths import project_root
        db_path = project_root(project_id) / "memories.db"
        r = await run_thinking_pipeline(
            user_input=query,
            project_id=project_id,
            mode="deep",
            db_path=db_path,
        )
        ctx = r.get("context_text", "")
        raw = r.get("raw_result", "")
        strategy = r.get("strategy_used", "")
        sources = r.get("sources", [])
        diag = r.get("diagnostics", {})
        print(f"  strategy: {strategy}")
        print(f"  context_text: {len(ctx)} chars")
        print(f"  raw_result: {len(raw)} chars")
        print(f"  sources: {len(sources)}")
        print(f"  deep_error_code: {diag.get('deep_error_code', 'None')}")
        if ctx:
            print(f"  前200字: {ctx[:200]}...")
        else:
            print("  ⚠️  pipeline 無結果！")
        for s in sources[:3]:
            print(f"    source: {s.get('title', '')[:50]}")
    except Exception as e:
        print(f"  ❌ pipeline 失敗: {type(e).__name__}: {e}")

    _sep("診斷完成")


def main():
    if len(sys.argv) < 2:
        print("De-insight Pipeline 診斷工具")
        print("=" * 40)
        print("\n可用專案：")
        list_projects()
        print(f"\n用法：python3 {sys.argv[0]} <project_id> \"測試問題\"")
        return

    project_id = sys.argv[1]
    query = sys.argv[2] if len(sys.argv) > 2 else "資源稀缺反而激發創新"

    print(f"🔍 診斷專案: {project_id}")
    print(f"🔍 測試問題: {query}")
    asyncio.run(diagnose(project_id, query))


if __name__ == "__main__":
    main()
