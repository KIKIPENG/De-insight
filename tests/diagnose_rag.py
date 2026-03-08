"""RAG 診斷工具 — 追蹤知識庫內容從檢索到注入的每一步。

用法：python3 tests/diagnose_rag.py [project_id] [query]
預設：project_id=default, query="設計"
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


async def diagnose(project_id: str = "default", query: str = "設計"):
    from settings import load_env
    env = load_env()

    print("=" * 60)
    print("RAG 診斷報告")
    print("=" * 60)

    # ── 1. Embedding 設定 ──
    print("\n── 1. Embedding 設定 ──")
    from rag.knowledge_graph import _get_embed_config, _get_llm_config
    embed_model, embed_key, embed_base, embed_dim = _get_embed_config()
    print(f"  模型:            {embed_model}")
    print(f"  維度:            {embed_dim}")
    print(f"  API base:        {embed_base}")
    print(f"  API key 存在:    {'是' if embed_key and len(embed_key) > 3 else '否'}")
    print("  ℹ jina-embeddings-v4（1024 維，文字+圖片共用向量空間）")

    # ── 2. LLM 設定 ──
    print("\n── 2. LLM 設定 ──")
    llm_model, llm_key, llm_base = _get_llm_config()
    print(f"  LLM_MODEL:       {env.get('LLM_MODEL', '(未設)')}")
    print(f"  RAG_LLM_MODEL:   {env.get('RAG_LLM_MODEL', '(未設)')}")
    print(f"  解析後模型:       {llm_model}")
    print(f"  API key 存在:    {'是' if llm_key and len(llm_key) > 3 else '否'}")
    print(f"  API base:        {llm_base}")

    # ── 3. 知識庫狀態 ──
    print("\n── 3. 知識庫狀態 ──")
    from rag.knowledge_graph import has_knowledge
    has_kg = has_knowledge(project_id=project_id)
    print(f"  project_id:      {project_id}")
    print(f"  has_knowledge:   {has_kg}")

    if not has_kg:
        print("  ✗ 知識庫為空！請先匯入文件。")
        return

    # ── 4. 原始檢索 ──
    print(f"\n── 4. 原始檢索 (query={query!r}) ──")
    from rag.knowledge_graph import query_knowledge, _is_no_context_result
    raw_result, sources = await query_knowledge(
        query, mode="naive", context_only=True,
        project_id=project_id, chunk_top_k=5,
    )
    print(f"  raw_result 長度:  {len(raw_result)} 字元")
    print(f"  is_no_context:   {_is_no_context_result(raw_result)}")
    print(f"  sources 數量:    {len(sources)}")
    if sources:
        for i, s in enumerate(sources[:3]):
            print(f"    [{i+1}] title={s.get('title', '?')}")
            print(f"        snippet={s.get('snippet', '?')[:60]}...")
    if raw_result:
        print(f"\n  raw_result 前 500 字：")
        print(f"  ---")
        for line in raw_result[:500].split('\n'):
            print(f"  | {line}")
        if len(raw_result) > 500:
            print(f"  | ... (截取，共 {len(raw_result)} 字)")
        print(f"  ---")
    else:
        print("  ✗ raw_result 為空！知識庫檢索沒回傳任何內容。")
        print("    可能原因：")
        print("    a) embedding 模型品質問題")
        print("    b) cosine_better_than_threshold 太高（目前 0.4）")
        print("    c) 知識庫索引損壞")
        return

    # ── 5. clean_context 處理 ──
    print("\n── 5. clean_context 處理 ──")
    from rag.pipeline import clean_context
    context_text = clean_context(raw_result)
    print(f"  context_text 長度: {len(context_text)} 字元")
    if context_text:
        print(f"\n  context_text 前 400 字：")
        print(f"  ---")
        for line in context_text[:400].split('\n'):
            print(f"  | {line}")
        if len(context_text) > 400:
            print(f"  | ... (截取，共 {len(context_text)} 字)")
        print(f"  ---")
    else:
        print("  ✗ clean_context 回傳空！")
        print("    raw_result 有 {0} 字，但 _clean_rag_chunk 處理後 < 10 字".format(len(raw_result)))
        print("    問題出在 _clean_rag_chunk 過度清理。")

        # 手動 debug _clean_rag_chunk
        import re
        json_contents = re.findall(r'"content"\s*:\s*"((?:[^"\\]|\\.)*)"', raw_result)
        print(f"    JSON content 擷取數: {len(json_contents)}")
        if json_contents:
            for i, c in enumerate(json_contents[:3]):
                print(f"    content[{i}] = {c[:80]}...")
        else:
            print("    JSON regex 沒匹配到任何 content！")
            print("    raw_result 格式可能不是 JSON，查看前 200 字：")
            print(f"    {repr(raw_result[:200])}")
        return

    # ── 6. Pipeline 完整流程 ──
    print("\n── 6. Pipeline 完整流程 ──")
    from rag.pipeline import run_thinking_pipeline
    result = await run_thinking_pipeline(
        user_input=query,
        project_id=project_id,
        mode="fast",
    )
    diag = result["diagnostics"]
    final_context = result["context_text"]
    print(f"  strategy:        {result['strategy_used']}")
    print(f"  question_type:   {diag['question_type']}")
    print(f"  source_count:    {diag['source_count']}")
    print(f"  filtered_by_gate: {diag['filtered_by_gate']}")
    print(f"  final context 長度: {len(final_context)} 字元")

    if final_context and len(final_context.strip()) > 10:
        print(f"\n  最終注入 context 前 400 字：")
        print(f"  ---")
        for line in final_context[:400].split('\n'):
            print(f"  | {line}")
        print(f"  ---")
        print("\n  ✓ 知識庫內容會被注入到 LLM messages 中")
    else:
        print("  ✗ 最終 context 為空或過短，不會注入！")
        print(f"    clean_context 有 {len(context_text)} 字，但 pipeline 後剩 {len(final_context)} 字")

    # ── 7. 注入位置模擬 ──
    print("\n── 7. 注入位置模擬 ──")
    fake_messages = [
        {"role": "user", "content": "之前的問題"},
        {"role": "assistant", "content": "之前的回答"},
        {"role": "user", "content": query},
    ]
    insert_idx = max(len(fake_messages) - 1, 0)
    print(f"  messages 數量:   {len(fake_messages)}")
    print(f"  insert_idx:      {insert_idx} (在最新 user 訊息之前)")
    print(f"  注入後順序:      [..., {{system: 知識庫}}, {{user: {query}}}]")

    print("\n" + "=" * 60)
    if final_context and len(final_context.strip()) > 10:
        print("結論：知識庫內容有成功檢索並清理，應該會被注入到 LLM。")
        print("如果 LLM 仍然不使用知識庫內容，問題可能在：")
        print("  a) 目前使用的 LLM 模型不善於遵循 system message 中的知識")
        print("  b) 對話歷史太長，知識庫被稀釋")
        print("  c) curator system prompt 的「資料不足就拒答」指令太強")
    else:
        print("結論：知識庫內容在 pipeline 中被丟失，需要修復上述標記 ✗ 的步驟。")
    print("=" * 60)


if __name__ == "__main__":
    pid = sys.argv[1] if len(sys.argv) > 1 else "default"
    q = sys.argv[2] if len(sys.argv) > 2 else "設計"
    asyncio.run(diagnose(pid, q))
