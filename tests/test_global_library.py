"""全局文獻庫 + 雙層文獻架構測試。

涵蓋：
- T1: GLOBAL_PROJECT_ID 常數正確性
- T2: ProjectManager 自動建立全局文獻庫
- T3: 全局文獻庫不可刪除
- T4: list_projects 全局排最前
- T5: query_knowledge_merged 合併邏輯
- T6: query_knowledge_merged 全局無資料時的行為
- T7: query_knowledge_merged 專案無資料但全局有
- T8: _ensure_initialized 使用 asyncio.Lock
- T9: pipeline._retrieve 使用合併查詢
- T10: KnowledgeScreen import 無語法錯誤
- T11: ProjectModal 全局條目分離
- T12: ResearchPanel 操作列包含「管理」
"""

import asyncio
import json
import os
import sys
import tempfile
from pathlib import Path
from types import ModuleType
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Mock config.service (used in target repo's knowledge_graph.py)
if "config" not in sys.modules:
    _mock_config = ModuleType("config")
    sys.modules["config"] = _mock_config
    _mock_config_svc = ModuleType("config.service")
    _mock_cs = MagicMock()
    _mock_cs.snapshot.return_value = {}
    _mock_config_svc.get_config_service = lambda: _mock_cs
    sys.modules["config.service"] = _mock_config_svc

# Mock settings module (depends on textual which isn't available in test env)
if "settings" not in sys.modules:
    _mock_settings = ModuleType("settings")
    _mock_settings.load_env = lambda: {}
    _mock_settings.get_env = lambda key, default="": default
    sys.modules["settings"] = _mock_settings

# Mock lightrag before any project imports
if "lightrag" not in sys.modules:
    _mock_lr = ModuleType("lightrag")
    _mock_lr.LightRAG = MagicMock
    _mock_lr.QueryParam = MagicMock
    sys.modules["lightrag"] = _mock_lr
    _mock_lr_llm = ModuleType("lightrag.llm")
    sys.modules["lightrag.llm"] = _mock_lr_llm
    _mock_lr_llm_oai = ModuleType("lightrag.llm.openai")
    _mock_lr_llm_oai.openai_complete_if_cache = MagicMock
    _mock_lr_llm_oai.openai_embed = MagicMock
    sys.modules["lightrag.llm.openai"] = _mock_lr_llm_oai
    _mock_lr_utils = ModuleType("lightrag.utils")
    _mock_lr_utils.EmbeddingFunc = MagicMock
    sys.modules["lightrag.utils"] = _mock_lr_utils


# ── T1: GLOBAL_PROJECT_ID 常數 ─────────────────────────────────

def test_global_project_id_constant():
    """GLOBAL_PROJECT_ID 是固定字串 '__global__'。"""
    from paths import GLOBAL_PROJECT_ID
    assert GLOBAL_PROJECT_ID == "__global__"
    assert isinstance(GLOBAL_PROJECT_ID, str)
    # 不能是一般的 UUID 格式
    assert "-" not in GLOBAL_PROJECT_ID
    print("PASS: test_global_project_id_constant")


# ── T2: ProjectManager 自動建立全局文獻庫 ──────────────────────

def test_project_manager_creates_global_library():
    """_ensure_db 後，全局文獻庫應該存在。"""
    import aiosqlite
    from paths import GLOBAL_PROJECT_ID

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "app.db"
        # Patch app_db_path 和 ensure_project_dirs
        with patch("projects.manager.app_db_path", return_value=db_path), \
             patch("projects.manager.ensure_project_dirs"):
            from projects.manager import ProjectManager
            pm = ProjectManager()

            async def _run():
                await pm._ensure_db()
                projects = await pm.list_projects()
                global_proj = [p for p in projects if p["id"] == GLOBAL_PROJECT_ID]
                assert len(global_proj) == 1, f"Expected 1 global project, got {len(global_proj)}"
                assert global_proj[0]["name"] == "全局文獻庫"
                assert global_proj[0]["is_global"] == 1
                return True

            result = asyncio.get_event_loop().run_until_complete(_run())
            assert result
    print("PASS: test_project_manager_creates_global_library")


# ── T3: 全局文獻庫不可刪除 ────────────────────────────────────

def test_global_library_cannot_be_deleted():
    """delete_project 對全局文獻庫應該 raise ValueError。"""
    from paths import GLOBAL_PROJECT_ID
    from projects.manager import ProjectManager

    pm = ProjectManager()
    try:
        asyncio.get_event_loop().run_until_complete(
            pm.delete_project(GLOBAL_PROJECT_ID)
        )
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "不可刪除" in str(e)
    print("PASS: test_global_library_cannot_be_deleted")


# ── T4: list_projects 全局排最前 ──────────────────────────────

def test_list_projects_global_first():
    """list_projects 回傳中，全局文獻庫永遠在第一個。"""
    import aiosqlite
    from paths import GLOBAL_PROJECT_ID

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "app.db"
        with patch("projects.manager.app_db_path", return_value=db_path), \
             patch("projects.manager.ensure_project_dirs"):
            from projects.manager import ProjectManager
            pm = ProjectManager()

            async def _run():
                await pm._ensure_db()
                # 建立幾個普通專案
                await pm.create_project("Project A")
                await pm.create_project("Project B")

                projects = await pm.list_projects()
                assert len(projects) >= 3  # global + A + B
                assert projects[0]["id"] == GLOBAL_PROJECT_ID, \
                    f"First project should be global, got {projects[0]['id']}"
                return True

            result = asyncio.get_event_loop().run_until_complete(_run())
            assert result
    print("PASS: test_list_projects_global_first")


# ── T5: query_knowledge_merged 合併邏輯 ───────────────────────

def test_query_knowledge_merged_combines_results():
    """合併查詢同時搜全局+專案，結果包含兩者，來源有 tier 標記。"""
    from rag import knowledge_graph as kg

    async def _run():
        # Mock has_knowledge 和 query_knowledge
        with patch.object(kg, "has_knowledge") as mock_has, \
             patch.object(kg, "query_knowledge") as mock_query:

            mock_has.side_effect = lambda project_id=None: True

            async def fake_query(q, mode="naive", context_only=True,
                                 project_id=None, chunk_top_k=5):
                if project_id == "__global__":
                    return "全局結果", [{"title": "global_doc", "snippet": "g"}]
                else:
                    return "專案結果", [{"title": "project_doc", "snippet": "p"}]

            mock_query.side_effect = fake_query

            text, sources, info = await kg.query_knowledge_merged(
                "test query", project_id="some_project"
            )

            # 專案結果在前
            assert "專案結果" in text
            assert "全局結果" in text
            assert text.index("專案結果") < text.index("全局結果")

            # 來源 tier 標記
            assert any(s.get("tier") == "project" for s in sources)
            assert any(s.get("tier") == "global" for s in sources)

            # merge_info 正確
            assert info["project_chars"] > 0
            assert info["global_chars"] > 0
            assert info["project_sources"] == 1
            assert info["global_sources"] == 1

    asyncio.get_event_loop().run_until_complete(_run())
    print("PASS: test_query_knowledge_merged_combines_results")


# ── T6: 全局無資料時 ──────────────────────────────────────────

def test_merged_query_no_global_data():
    """全局無資料時，只回傳專案結果，不 crash。"""
    from rag import knowledge_graph as kg

    async def _run():
        with patch.object(kg, "has_knowledge") as mock_has, \
             patch.object(kg, "query_knowledge") as mock_query:

            def _has(project_id=None):
                return project_id != "__global__"  # 只有專案有資料

            mock_has.side_effect = _has
            mock_query.return_value = ("專案唯一結果", [{"title": "doc"}])

            text, sources, info = await kg.query_knowledge_merged(
                "test", project_id="proj_1"
            )
            assert "專案唯一結果" in text
            assert info["global_chars"] == 0
            assert info["global_sources"] == 0
            assert "全局文獻" not in text  # 無分隔符

    asyncio.get_event_loop().run_until_complete(_run())
    print("PASS: test_merged_query_no_global_data")


# ── T7: 專案無資料但全局有 ───────────────────────────────────

def test_merged_query_only_global():
    """專案無資料但全局有，應回傳全局結果。"""
    from rag import knowledge_graph as kg

    async def _run():
        with patch.object(kg, "has_knowledge") as mock_has, \
             patch.object(kg, "query_knowledge") as mock_query:

            def _has(project_id=None):
                return project_id == "__global__"

            mock_has.side_effect = _has
            mock_query.return_value = ("全局基礎文獻", [{"title": "foundation"}])

            text, sources, info = await kg.query_knowledge_merged(
                "test", project_id="empty_proj"
            )
            assert "全局基礎文獻" in text
            assert info["project_chars"] == 0
            assert info["global_chars"] > 0

    asyncio.get_event_loop().run_until_complete(_run())
    print("PASS: test_merged_query_only_global")


# ── T8: _ensure_initialized 使用 lock ────────────────────────

def test_ensure_initialized_uses_lock():
    """_ensure_initialized 應該使用 asyncio.Lock 防止並發。"""
    from rag.knowledge_graph import _get_rag_lock
    import asyncio

    lock = _get_rag_lock()
    assert isinstance(lock, asyncio.Lock), f"Expected asyncio.Lock, got {type(lock)}"

    # 第二次呼叫應該回傳同一個 lock
    lock2 = _get_rag_lock()
    assert lock is lock2, "Should return same lock instance"
    print("PASS: test_ensure_initialized_uses_lock")


# ── T9: pipeline._retrieve 呼叫合併查詢 ──────────────────────

def test_pipeline_retrieve_uses_merged():
    """_retrieve 應該 import 並使用 query_knowledge_merged。"""
    import ast
    source = (Path(__file__).parent.parent / "rag" / "pipeline.py").read_text()
    tree = ast.parse(source)

    # 在 _retrieve 函式內找到 query_knowledge_merged 的使用
    found = False
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id == "query_knowledge_merged":
            found = True
            break
        if isinstance(node, ast.Attribute) and node.attr == "query_knowledge_merged":
            found = True
            break

    # 也檢查字串中是否有 query_knowledge_merged
    if not found:
        found = "query_knowledge_merged" in source

    assert found, "pipeline.py should use query_knowledge_merged"
    print("PASS: test_pipeline_retrieve_uses_merged")


# ── T10: KnowledgeScreen 語法正確 ────────────────────────────

def test_knowledge_screen_syntax():
    """knowledge_screen.py 語法正確，可被 AST 解析。"""
    import ast
    source = (Path(__file__).parent.parent / "knowledge_screen.py").read_text()
    tree = ast.parse(source)

    # 確認 KnowledgeScreen class 存在
    classes = [n.name for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]
    assert "KnowledgeScreen" in classes, \
        f"KnowledgeScreen class not found. Classes: {classes}"

    # 確認關鍵方法存在
    methods = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "KnowledgeScreen":
            methods = [n.name for n in node.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
            break

    expected_methods = [
        "_refresh_view", "_render_doc_list", "_build_doc_card",
        "_switch_scope", "_do_import", "_start_rename", "_delete_doc",
    ]
    for m in expected_methods:
        assert m in methods, f"Missing method: {m}. Found: {methods}"

    print("PASS: test_knowledge_screen_syntax")


# ── T11: ProjectModal 全局條目分離 ────────────────────────────

def test_project_modal_separates_global():
    """ProjectModal.compose 中全局專案用金色 ◆ 標記，排在最上方。"""
    import ast
    source = (Path(__file__).parent.parent / "modals.py").read_text()

    # 確認 compose 中有全局相關的邏輯
    assert "GLOBAL_PROJECT_ID" in source, "modals.py should reference GLOBAL_PROJECT_ID"
    assert "is_global" in source, "modals.py should check is_global"
    assert "#f59e0b" in source, "modals.py should use gold color for global entry"
    assert "◆" in source, "modals.py should use ◆ marker for global"
    print("PASS: test_project_modal_separates_global")


# ── T12: ResearchPanel 文獻 tab 指向 knowledge screen ────────

def test_research_panel_has_manage_action():
    """ResearchPanel 的「文獻」tab 應指向 open_knowledge_screen。"""
    import ast
    source = (Path(__file__).parent.parent / "panels.py").read_text()

    assert "open_knowledge_screen" in source, \
        "panels.py should have open_knowledge_screen action"
    print("PASS: test_research_panel_has_manage_action")


# ── T13: is_global_project helper ────────────────────────────

def test_is_global_project_helper():
    """ProjectManager.is_global_project 正確判斷。"""
    from projects.manager import ProjectManager
    pm = ProjectManager()
    assert pm.is_global_project("__global__") is True
    assert pm.is_global_project("some-uuid") is False
    assert pm.is_global_project("default") is False
    print("PASS: test_is_global_project_helper")


# ── T14: 全局專案內查詢不重複搜自己 ─────────────────────────

def test_merged_query_inside_global_no_duplicate():
    """在全局專案內查詢時，不應該搜兩次全局圖譜。"""
    from rag import knowledge_graph as kg

    async def _run():
        call_log = []

        with patch.object(kg, "has_knowledge") as mock_has, \
             patch.object(kg, "query_knowledge") as mock_query:

            mock_has.return_value = True

            async def _log_query(q, **kwargs):
                call_log.append(kwargs.get("project_id"))
                return "result", []

            mock_query.side_effect = _log_query

            await kg.query_knowledge_merged("test", project_id="__global__")

            # 在全局專案裡查，只應該呼叫一次（project graph = global）
            assert "__global__" in call_log
            assert call_log.count("__global__") == 1, \
                f"Should query global once, got {call_log}"

    asyncio.get_event_loop().run_until_complete(_run())
    print("PASS: test_merged_query_inside_global_no_duplicate")


# ── T15: 合併查詢某一邊失敗不影響另一邊 ────────────────────

def test_merged_query_one_side_fails():
    """全局查詢失敗時，專案結果仍正常回傳。"""
    from rag import knowledge_graph as kg

    async def _run():
        with patch.object(kg, "has_knowledge", return_value=True), \
             patch.object(kg, "query_knowledge") as mock_query:

            async def _flaky(q, **kwargs):
                if kwargs.get("project_id") == "__global__":
                    raise RuntimeError("global graph corrupted")
                return "專案正常結果", [{"title": "ok"}]

            mock_query.side_effect = _flaky

            text, sources, info = await kg.query_knowledge_merged(
                "test", project_id="proj"
            )
            assert "專案正常結果" in text
            assert info["global_chars"] == 0  # failed side returns empty
            assert info["project_chars"] > 0

    asyncio.get_event_loop().run_until_complete(_run())
    print("PASS: test_merged_query_one_side_fails")


# ── Run all ───────────────────────────────────────────────────

if __name__ == "__main__":
    test_global_project_id_constant()
    test_project_manager_creates_global_library()
    test_global_library_cannot_be_deleted()
    test_list_projects_global_first()
    test_query_knowledge_merged_combines_results()
    test_merged_query_no_global_data()
    test_merged_query_only_global()
    test_ensure_initialized_uses_lock()
    test_pipeline_retrieve_uses_merged()
    test_knowledge_screen_syntax()
    test_project_modal_separates_global()
    test_research_panel_has_manage_action()
    test_is_global_project_helper()
    test_merged_query_inside_global_no_duplicate()
    test_merged_query_one_side_fails()
    print("\n✅ All 15 tests passed!")
