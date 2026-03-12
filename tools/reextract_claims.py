#!/usr/bin/env python3
"""手動重新提取 structural claims。

當匯入時 claim 提取靜默失敗（core_claims.db 為空），
用此工具從已存在的 vdb_chunks 重新跑一次。

用法：
  python tools/reextract_claims.py                    # 列出所有專案
  python tools/reextract_claims.py <project_id>       # 對指定專案重新提取
  python tools/reextract_claims.py <project_id> --dry  # 只顯示會做什麼，不實際呼叫 LLM
"""

import asyncio
import json
import os
import re
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")


def list_projects():
    """列出所有有 vdb_chunks 的專案。"""
    from rag.vdb_utils import get_lightrag_dir, VDB_CHUNKS_FILE_NAMES
    from paths import PROJECTS_DIR

    if not PROJECTS_DIR.exists():
        print("  (no projects directory)")
        return

    for d in sorted(PROJECTS_DIR.iterdir()):
        if not d.is_dir():
            continue
        lightrag_dir = d / "lightrag"
        claims_db = d / "core_claims.db"

        # Check for vdb_chunks
        chunk_count = 0
        for name in VDB_CHUNKS_FILE_NAMES:
            vdb_path = lightrag_dir / name
            if vdb_path.exists():
                try:
                    data = json.loads(vdb_path.read_text(encoding="utf-8"))
                    chunk_count = len(data.get("data", []))
                except Exception:
                    pass
                break

        # Check claim count
        claim_count = 0
        if claims_db.exists():
            import sqlite3
            try:
                db = sqlite3.connect(str(claims_db))
                claim_count = db.execute("SELECT COUNT(*) FROM claims").fetchone()[0]
                db.close()
            except Exception:
                pass

        status = ""
        if chunk_count > 0 and claim_count == 0:
            status = " ⚠️  有 chunks 但沒有 claims — 需要重新提取！"
        elif chunk_count > 0 and claim_count > 0:
            status = " ✓"
        elif chunk_count == 0:
            status = " (空)"

        print(f"  {d.name}  chunks={chunk_count}  claims={claim_count}{status}")


async def reextract(project_id: str, dry_run: bool = False):
    """對指定專案重新提取 claims。"""
    from rag.vdb_utils import find_vdb_chunks_file
    from core.thought_extractor import ThoughtExtractor, LLMCallable
    from core.stores.claim_store import ClaimStore
    from config_service import get_config_service

    # 1. 找 vdb_chunks
    vdb_path = find_vdb_chunks_file(project_id)
    if not vdb_path or not vdb_path.exists():
        print(f"❌ 找不到 vdb_chunks.json (project: {project_id})")
        return

    chunks_data = json.loads(vdb_path.read_text(encoding="utf-8"))
    all_chunks = chunks_data.get("data", [])
    print(f"📦 找到 {len(all_chunks)} 個 chunks (from {vdb_path})")

    if not all_chunks:
        print("❌ chunks 為空，無法提取")
        return

    # 2. Sample chunks
    max_samples = min(10, len(all_chunks))
    if len(all_chunks) <= max_samples:
        sampled = all_chunks
    else:
        step = len(all_chunks) / max_samples
        sampled = [all_chunks[int(i * step)] for i in range(max_samples)]

    print(f"🔍 取樣 {len(sampled)} 個 chunks 進行提取")

    # Show sample text
    for i, chunk in enumerate(sampled):
        text = ""
        if isinstance(chunk, dict):
            text = chunk.get("content", "") or chunk.get("text", "")
        elif isinstance(chunk, str):
            text = chunk
        text = text.strip()
        if text:
            print(f"  chunk {i+1}: {text[:80]}...")

    if dry_run:
        print("\n🏁 [DRY RUN] 不實際呼叫 LLM，到此為止。")
        return

    # 3. Setup LLM callable
    env = get_config_service().snapshot(include_process=True)
    llm_model = env.get("LLM_MODEL", "")
    llm_key = env.get("OPENAI_API_KEY", "") or env.get("ANTHROPIC_API_KEY", "")
    llm_base = env.get("OPENAI_API_BASE", "https://openrouter.ai/api/v1")

    if not llm_model:
        print("❌ LLM_MODEL 未設定。請確認 .env")
        return
    if not llm_key:
        print("❌ 沒有 API key (OPENAI_API_KEY 或 ANTHROPIC_API_KEY)")
        return

    print(f"🤖 使用模型: {llm_model}")
    print(f"🔗 API base: {llm_base}")

    async def _llm_for_claims(prompt: str) -> str:
        import httpx
        import litellm

        messages = [{"role": "user", "content": prompt}]

        if llm_model.startswith("gemini/"):
            resp = await litellm.acompletion(
                model=llm_model, messages=messages, api_key=llm_key,
            )
            return resp.choices[0].message.content or ""

        async with httpx.AsyncClient(timeout=120.0) as client:
            body = {"model": llm_model, "messages": messages}
            resp = await client.post(
                f"{llm_base}/chat/completions",
                headers={
                    "Authorization": f"Bearer {llm_key}",
                    "HTTP-Referer": "https://github.com/De-insight",
                    "X-Title": "De-insight",
                },
                json=body,
            )
            resp.raise_for_status()
            result = resp.json()["choices"][0]["message"]["content"]

        # Strip <think>...</think>
        if result and "<think>" in result:
            result = re.sub(r"<think>[\s\S]*?</think>\s*", "", result)
        return result

    # 4. Run extraction
    extractor = ThoughtExtractor(
        llm_callable=LLMCallable(func=_llm_for_claims),
        project_id=project_id,
    )
    claim_store = ClaimStore(project_id=project_id)

    # Check existing claims
    existing = await claim_store.list_by_project(project_id, limit=1000)
    if existing:
        print(f"⚠️  已有 {len(existing)} 個 claims，將跳過重複")

    total_claims = 0
    errors = 0

    for i, chunk in enumerate(sampled):
        chunk_text = ""
        if isinstance(chunk, dict):
            chunk_text = chunk.get("content", "") or chunk.get("text", "")
        elif isinstance(chunk, str):
            chunk_text = chunk

        if not chunk_text or len(chunk_text.strip()) < 30:
            print(f"  chunk {i+1}: 太短，跳過")
            continue

        try:
            print(f"  chunk {i+1}/{len(sampled)}: 提取中...", end="", flush=True)
            result = await extractor.extract_from_passage(
                passage_text=chunk_text,
                source_id=f"reextract_{project_id}",
            )
            if result.was_extracted and result.claims:
                for claim in result.claims:
                    await claim_store.add(claim)
                    total_claims += 1
                print(f" ✓ {len(result.claims)} claims")
                for c in result.claims:
                    patterns = ", ".join(c.abstract_patterns or [])
                    print(f"    → {c.core_claim[:60]}  [{patterns}]")
            else:
                print(f" (no claims extracted)")
        except Exception as e:
            errors += 1
            print(f" ❌ {type(e).__name__}: {str(e)[:100]}")

    print(f"\n{'='*60}")
    print(f"✅ 完成：提取了 {total_claims} 個 claims（{errors} 個錯誤）")

    # Verify
    all_claims = await claim_store.list_by_project(project_id, limit=1000)
    print(f"📊 ClaimStore 現在共有 {len(all_claims)} 個 claims")


def main():
    if len(sys.argv) < 2:
        print("De-insight Claim Re-extraction Tool")
        print("=" * 40)
        print("\n可用專案：")
        list_projects()
        print(f"\n用法：python {sys.argv[0]} <project_id> [--dry]")
        return

    project_id = sys.argv[1]
    dry_run = "--dry" in sys.argv

    asyncio.run(reextract(project_id, dry_run=dry_run))


if __name__ == "__main__":
    main()
