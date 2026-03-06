import asyncio, gc, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rag.knowledge_graph import get_rag

async def test():
    rag_a = get_rag(project_id="test_proj_a")
    await rag_a.initialize_storages()
    await rag_a.ainsert("包豪斯預設了功能的穩定性")
    del rag_a; gc.collect()

    # B 是空的，不應該找到 A 的資料
    rag_b = get_rag(project_id="test_proj_b")
    await rag_b.initialize_storages()
    result_b = await rag_b.aquery("包豪斯")
    assert "包豪斯" not in result_b, "隔離失敗：B 讀到了 A 的資料"
    del rag_b; gc.collect()

    # A 應該還在
    rag_a2 = get_rag(project_id="test_proj_a")
    await rag_a2.initialize_storages()
    result_a = await rag_a2.aquery("包豪斯")
    assert "包豪斯" in result_a, "持久化失敗：A 的資料消失了"

    print("LightRAG 熱切換測試通過")

asyncio.run(test())
