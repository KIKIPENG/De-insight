"""Comprehensive test suite for all data layer features.

Tests real user interaction scenarios using aiosqlite directly (not mocked).
Uses temp directories via DEINSIGHT_HOME environment variable.

All async tests are run using asyncio.run() to avoid pytest-asyncio issues.

Run with: pytest tests/test_comprehensive_data.py -v
"""

import sys
import os
import asyncio
import tempfile
import shutil
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

import pytest
import aiosqlite


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def temp_deinsight_home(tmp_path):
    """Provide isolated temp directory for each test.

    Reloads paths module so module-level constants pick up env changes.
    Clears db_pool cache to prevent stale cross-loop connections.
    """
    deinsight_home = tmp_path / "deinsight"
    deinsight_home.mkdir()
    old_home = os.environ.get("DEINSIGHT_HOME")
    old_version = os.environ.get("DEINSIGHT_DATA_VERSION")

    os.environ["DEINSIGHT_HOME"] = str(deinsight_home)
    os.environ["DEINSIGHT_DATA_VERSION"] = "v0.7"

    # Reload paths so DATA_ROOT etc. pick up new env
    import importlib
    import paths
    importlib.reload(paths)

    # Reload memory.store so _DEFAULT_DB picks up new DATA_ROOT
    from memory import store as _ms
    importlib.reload(_ms)

    # Clear the db_pool so stale connections don't leak between tests
    from utils import db_pool
    db_pool._pool.clear()

    yield deinsight_home

    # Cleanup: restore original env
    if old_home:
        os.environ["DEINSIGHT_HOME"] = old_home
    else:
        os.environ.pop("DEINSIGHT_HOME", None)
    if old_version:
        os.environ["DEINSIGHT_DATA_VERSION"] = old_version
    else:
        os.environ.pop("DEINSIGHT_DATA_VERSION", None)

    # Re-reload to restore
    importlib.reload(paths)
    importlib.reload(_ms)
    db_pool._pool.clear()


# ============================================================================
# Helper functions for async operations
# ============================================================================

def run_async(coro):
    """Run async coroutine in sync test context."""
    return asyncio.run(coro)


async def _init_memory_tables():
    """Initialize memory store tables with all migrations."""
    from memory import store as memory_store
    from utils.db_pool import get_connection
    import paths

    db_path = paths.DATA_ROOT / "projects" / "default" / "memories.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)

    async with get_connection(db_path) as db:
        await db.execute(memory_store._CREATE_TABLE)
        await db.execute(memory_store._CREATE_PENDING_TABLE)
        for migration_sql in [
            memory_store._MIGRATE_PROJECT_ID,
            memory_store._MIGRATE_CATEGORY,
            memory_store._MIGRATE_PENDING_INDEX,
        ]:
            try:
                await db.execute(migration_sql)
            except Exception:
                pass
        await db.commit()


async def _init_conversation_tables():
    """Initialize conversation store tables."""
    from conversation.store import ConversationStore
    store = ConversationStore()
    await store._ensure_db()


async def _init_project_tables():
    """Initialize project manager tables."""
    from projects.manager import ProjectManager
    manager = ProjectManager()
    await manager._ensure_db()


# ============================================================================
# A. Memory Store Tests (15+ tests)
# ============================================================================

class TestMemoryStoreBasicOperations:
    """Test basic add, get, delete memory operations."""

    def test_add_memory_basic_insert(self, temp_deinsight_home):
        """add_memory inserts record and returns ID."""
        async def _test():
            await _init_memory_tables()
            from memory import store as memory_store
            mem_id = await memory_store.add_memory(
                type="insight",
                content="This is a test insight",
                source="test"
            )
            assert isinstance(mem_id, int)
            assert mem_id > 0

        run_async(_test())

    def test_add_memory_with_tags(self, temp_deinsight_home):
        """add_memory with tags stores JSON array."""
        async def _test():
            await _init_memory_tables()
            from memory import store as memory_store
            tags = ["concept", "learning"]
            mem_id = await memory_store.add_memory(
                type="question",
                content="How does learning work?",
                tags=tags
            )
            memories = await memory_store.get_memories(type="question")
            assert len(memories) == 1
            assert memories[0]["tags"] == tags

        run_async(_test())

    def test_add_memory_with_topic(self, temp_deinsight_home):
        """add_memory stores topic field."""
        async def _test():
            await _init_memory_tables()
            from memory import store as memory_store
            mem_id = await memory_store.add_memory(
                type="fact",
                content="Water boils at 100°C",
                topic="Physics"
            )
            memories = await memory_store.get_memories()
            assert len(memories) == 1
            assert memories[0]["topic"] == "Physics"

        run_async(_test())

    def test_add_memory_with_category(self, temp_deinsight_home):
        """add_memory stores category field."""
        async def _test():
            await _init_memory_tables()
            from memory import store as memory_store
            mem_id = await memory_store.add_memory(
                type="definition",
                content="Entropy is disorder",
                category="Science"
            )
            memories = await memory_store.get_memories(category="Science")
            assert len(memories) == 1
            assert memories[0]["category"] == "Science"

        run_async(_test())

    def test_add_memory_with_project_id(self, temp_deinsight_home):
        """add_memory stores project_id for filtering."""
        async def _test():
            await _init_memory_tables()
            from memory import store as memory_store
            mem_id = await memory_store.add_memory(
                type="note",
                content="Project-specific note",
                project_id="proj-1"
            )
            memories = await memory_store.get_memories()
            assert len(memories) == 1
            assert memories[0]["project_id"] == "proj-1"

        run_async(_test())

    def test_get_memories_default_limit(self, temp_deinsight_home):
        """get_memories returns max 20 by default."""
        async def _test():
            await _init_memory_tables()
            from memory import store as memory_store
            for i in range(25):
                await memory_store.add_memory(
                    type="test",
                    content=f"Memory {i}"
                )
            memories = await memory_store.get_memories()
            assert len(memories) == 20

        run_async(_test())

    def test_get_memories_custom_limit(self, temp_deinsight_home):
        """get_memories respects limit parameter."""
        async def _test():
            await _init_memory_tables()
            from memory import store as memory_store
            for i in range(10):
                await memory_store.add_memory(
                    type="test",
                    content=f"Memory {i}"
                )
            memories = await memory_store.get_memories(limit=5)
            assert len(memories) == 5

        run_async(_test())

    def test_get_memories_filter_by_type(self, temp_deinsight_home):
        """get_memories filters by type."""
        async def _test():
            await _init_memory_tables()
            from memory import store as memory_store
            await memory_store.add_memory(type="insight", content="insight1")
            await memory_store.add_memory(type="question", content="q1")
            await memory_store.add_memory(type="insight", content="insight2")

            insights = await memory_store.get_memories(type="insight")
            assert len(insights) == 2
            assert all(m["type"] == "insight" for m in insights)

        run_async(_test())

    def test_get_memories_filter_by_category(self, temp_deinsight_home):
        """get_memories filters by category."""
        async def _test():
            await _init_memory_tables()
            from memory import store as memory_store
            await memory_store.add_memory(type="a", content="a1", category="Math")
            await memory_store.add_memory(type="b", content="b1", category="Science")
            await memory_store.add_memory(type="c", content="c1", category="Math")

            math_mems = await memory_store.get_memories(category="Math")
            assert len(math_mems) == 2
            assert all(m["category"] == "Math" for m in math_mems)

        run_async(_test())

    def test_delete_memory(self, temp_deinsight_home):
        """delete_memory removes record."""
        async def _test():
            await _init_memory_tables()
            from memory import store as memory_store
            mem_id = await memory_store.add_memory(type="test", content="test")
            await memory_store.delete_memory(mem_id)

            memories = await memory_store.get_memories()
            assert len(memories) == 0

        run_async(_test())

    def test_search_memories_like_fallback(self, temp_deinsight_home):
        """search_memories uses LIKE fallback when no vector index."""
        async def _test():
            await _init_memory_tables()
            from memory import store as memory_store
            await memory_store.add_memory(type="note", content="Python programming language")
            await memory_store.add_memory(type="note", content="JavaScript basics")
            await memory_store.add_memory(type="note", content="Python data structures")

            results = await memory_store.search_memories("Python", limit=10)
            assert len(results) == 2
            assert all("Python" in m["content"] for m in results)

        run_async(_test())

    def test_pending_memories_save_load_clear(self, temp_deinsight_home):
        """Test pending memories workflow: save, load, clear."""
        async def _test():
            await _init_memory_tables()
            from memory import store as memory_store
            candidates = [
                {"type": "insight", "content": "thought 1", "source": "user"},
                {"type": "question", "content": "thought 2", "source": "user"},
            ]

            await memory_store.save_pending_memories(candidates)
            loaded = await memory_store.load_pending_memories()
            assert len(loaded) == 2
            assert loaded[0]["type"] == "insight"

            await memory_store.clear_pending_memories()
            loaded_again = await memory_store.load_pending_memories()
            assert len(loaded_again) == 0

        run_async(_test())

    def test_empty_table_queries_return_empty_list(self, temp_deinsight_home):
        """Queries on empty memory table return []."""
        async def _test():
            await _init_memory_tables()
            from memory import store as memory_store
            memories = await memory_store.get_memories()
            assert memories == []

            topics = await memory_store.get_topics()
            assert topics == []

            results = await memory_store.search_memories("anything")
            assert results == []

        run_async(_test())

    def test_schema_migration_columns_auto_added(self, temp_deinsight_home):
        """Schema migration adds topic, project_id, category columns on init."""
        async def _test():
            await _init_memory_tables()
            from memory import store as memory_store
            mem_id = await memory_store.add_memory(
                type="test",
                content="test",
                topic="TestTopic",
                category="TestCat",
                project_id="test-proj"
            )

            memories = await memory_store.get_memories()
            assert len(memories) == 1
            assert memories[0]["topic"] == "TestTopic"
            assert memories[0]["category"] == "TestCat"
            assert memories[0]["project_id"] == "test-proj"

        run_async(_test())


class TestMemoryStoreTopic:
    """Test topic-related memory operations."""

    def test_get_topics_returns_distinct(self, temp_deinsight_home):
        """get_topics returns distinct topic strings."""
        async def _test():
            await _init_memory_tables()
            from memory import store as memory_store
            await memory_store.add_memory(type="a", content="a1", topic="Philosophy")
            await memory_store.add_memory(type="b", content="b1", topic="Science")
            await memory_store.add_memory(type="c", content="c1", topic="Philosophy")

            topics = await memory_store.get_topics()
            assert len(topics) == 2
            assert set(topics) == {"Philosophy", "Science"}

        run_async(_test())

    def test_get_memories_by_topic(self, temp_deinsight_home):
        """get_memories_by_topic filters by exact topic."""
        async def _test():
            await _init_memory_tables()
            from memory import store as memory_store
            await memory_store.add_memory(type="a", content="a1", topic="Philosophy")
            await memory_store.add_memory(type="b", content="b1", topic="Science")
            await memory_store.add_memory(type="c", content="c1", topic="Philosophy")

            philo = await memory_store.get_memories_by_topic("Philosophy")
            assert len(philo) == 2

        run_async(_test())

    def test_update_memory_topic(self, temp_deinsight_home):
        """update_memory_topic changes topic."""
        async def _test():
            await _init_memory_tables()
            from memory import store as memory_store
            mem_id = await memory_store.add_memory(
                type="note",
                content="note",
                topic="Old"
            )
            await memory_store.update_memory_topic(mem_id, "New")

            memories = await memory_store.get_memories()
            assert memories[0]["topic"] == "New"

        run_async(_test())


class TestMemoryStoreStats:
    """Test memory statistics."""

    def test_get_memory_stats_total_count(self, temp_deinsight_home):
        """get_memory_stats returns total count."""
        async def _test():
            await _init_memory_tables()
            from memory import store as memory_store
            for i in range(5):
                await memory_store.add_memory(type="test", content=f"m{i}")

            stats = await memory_store.get_memory_stats()
            assert stats["total"] == 5

        run_async(_test())

    def test_get_memory_stats_by_type(self, temp_deinsight_home):
        """get_memory_stats breaks down by type."""
        async def _test():
            await _init_memory_tables()
            from memory import store as memory_store
            await memory_store.add_memory(type="insight", content="i1")
            await memory_store.add_memory(type="insight", content="i2")
            await memory_store.add_memory(type="question", content="q1")

            stats = await memory_store.get_memory_stats()
            assert stats["by_type"]["insight"] == 2
            assert stats["by_type"]["question"] == 1

        run_async(_test())

    def test_get_memory_stats_by_topic(self, temp_deinsight_home):
        """get_memory_stats breaks down by topic."""
        async def _test():
            await _init_memory_tables()
            from memory import store as memory_store
            await memory_store.add_memory(type="a", content="a1", topic="Math")
            await memory_store.add_memory(type="b", content="b1", topic="Math")
            await memory_store.add_memory(type="c", content="c1", topic="Science")

            stats = await memory_store.get_memory_stats()
            assert stats["by_topic"]["Math"] == 2
            assert stats["by_topic"]["Science"] == 1

        run_async(_test())


# ============================================================================
# B. Conversation Store Tests (10+ tests)
# ============================================================================

class TestConversationStoreBasic:
    """Test basic conversation operations."""

    def test_create_conversation_returns_uuid(self, temp_deinsight_home):
        """create_conversation returns non-empty UUID string."""
        async def _test():
            await _init_conversation_tables()
            from conversation.store import ConversationStore
            store = ConversationStore()
            cid = await store.create_conversation()
            assert isinstance(cid, str)
            assert len(cid) == 36  # UUID4 format

        run_async(_test())

    def test_add_message_and_retrieve(self, temp_deinsight_home):
        """add_message inserts and get_messages retrieves."""
        async def _test():
            await _init_conversation_tables()
            from conversation.store import ConversationStore
            store = ConversationStore()
            cid = await store.create_conversation()

            await store.add_message(cid, "user", "Hello")
            await store.add_message(cid, "assistant", "Hi there")

            messages = await store.get_messages(cid)
            assert len(messages) == 2
            assert messages[0]["role"] == "user"
            assert messages[0]["content"] == "Hello"

        run_async(_test())

    def test_set_title(self, temp_deinsight_home):
        """set_title updates conversation title."""
        async def _test():
            await _init_conversation_tables()
            from conversation.store import ConversationStore
            store = ConversationStore()
            cid = await store.create_conversation()

            await store.set_title(cid, "My Great Conversation")
            conv = await store.get_conversation(cid)
            assert conv["title"] == "My Great Conversation"

        run_async(_test())

    def test_list_conversations_ordered_by_updated_at(self, temp_deinsight_home):
        """list_conversations returns newest first."""
        import time as _time
        async def _test():
            await _init_conversation_tables()
            from conversation.store import ConversationStore
            store = ConversationStore()
            cid1 = await store.create_conversation()
            await store.add_message(cid1, "user", "msg1")

            # Ensure different updated_at timestamp (SQLite datetime has second precision)
            _time.sleep(1.1)

            cid2 = await store.create_conversation()
            await store.add_message(cid2, "user", "msg2")

            convs = await store.list_conversations()
            assert len(convs) == 2
            assert convs[0]["id"] == cid2

        run_async(_test())

    def test_get_conversation_by_id(self, temp_deinsight_home):
        """get_conversation returns single conversation."""
        async def _test():
            await _init_conversation_tables()
            from conversation.store import ConversationStore
            store = ConversationStore()
            cid = await store.create_conversation()

            await store.set_title(cid, "Test Conv")
            conv = await store.get_conversation(cid)

            assert conv is not None
            assert conv["id"] == cid
            assert conv["title"] == "Test Conv"

        run_async(_test())

    def test_get_conversation_not_found(self, temp_deinsight_home):
        """get_conversation returns None for missing ID."""
        async def _test():
            await _init_conversation_tables()
            from conversation.store import ConversationStore
            store = ConversationStore()
            conv = await store.get_conversation("nonexistent-uuid")
            assert conv is None

        run_async(_test())

    def test_delete_conversation_cascading(self, temp_deinsight_home):
        """delete_conversation removes messages too."""
        async def _test():
            await _init_conversation_tables()
            from conversation.store import ConversationStore
            store = ConversationStore()
            cid = await store.create_conversation()

            await store.add_message(cid, "user", "msg")
            await store.delete_conversation(cid)

            messages = await store.get_messages(cid)
            assert messages == []

        run_async(_test())

    def test_multiple_conversations_per_project(self, temp_deinsight_home):
        """Can create multiple conversations for same project."""
        async def _test():
            await _init_conversation_tables()
            from conversation.store import ConversationStore
            store = ConversationStore()
            cid1 = await store.create_conversation(project_id="proj1")
            cid2 = await store.create_conversation(project_id="proj1")

            convs = await store.list_conversations(project_id="proj1")
            assert len(convs) == 2

        run_async(_test())


class TestConversationStoreDocuments:
    """Test document CRUD operations."""

    def test_add_document(self, temp_deinsight_home):
        """add_document inserts and returns doc_id."""
        async def _test():
            await _init_conversation_tables()
            from conversation.store import ConversationStore
            store = ConversationStore()
            doc_id = await store.add_document(
                title="Research Paper",
                source_path="/path/to/paper.pdf",
                source_type="pdf",
                file_size=1024000,
                page_count=42,
                project_id="proj1"
            )
            assert isinstance(doc_id, str)
            assert len(doc_id) == 36

        run_async(_test())

    def test_list_documents(self, temp_deinsight_home):
        """list_documents returns all docs for project."""
        async def _test():
            await _init_conversation_tables()
            from conversation.store import ConversationStore
            store = ConversationStore()
            doc_id1 = await store.add_document(
                title="Doc1",
                project_id="proj1"
            )
            doc_id2 = await store.add_document(
                title="Doc2",
                project_id="proj1"
            )

            docs = await store.list_documents(project_id="proj1")
            assert len(docs) == 2
            assert any(d["title"] == "Doc1" for d in docs)

        run_async(_test())

    def test_delete_document(self, temp_deinsight_home):
        """delete_document removes document."""
        async def _test():
            await _init_conversation_tables()
            from conversation.store import ConversationStore
            store = ConversationStore()
            doc_id = await store.add_document(title="Doc")

            await store.delete_document(doc_id)

            docs = await store.list_documents()
            assert len(docs) == 0

        run_async(_test())

    def test_update_document_meta(self, temp_deinsight_home):
        """update_document_meta updates title, tags, note."""
        async def _test():
            await _init_conversation_tables()
            from conversation.store import ConversationStore
            store = ConversationStore()
            doc_id = await store.add_document(title="Old Title")

            await store.update_document_meta(
                doc_id,
                title="New Title",
                tags=["tag1", "tag2"],
                note="This is important"
            )

            docs = await store.list_documents()
            assert docs[0]["title"] == "New Title"
            assert docs[0]["note"] == "This is important"

        run_async(_test())


# ============================================================================
# C. Project Manager Tests (10+ tests)
# ============================================================================

class TestProjectManagerBasic:
    """Test basic project operations."""

    def test_ensure_db_creates_table(self, temp_deinsight_home):
        """_ensure_db creates projects table."""
        async def _test():
            from projects.manager import ProjectManager
            import paths
            manager = ProjectManager()
            await manager._ensure_db()

            db_path = paths.app_db_path()
            async with aiosqlite.connect(db_path) as db:
                async with db.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name='projects'"
                ) as cur:
                    result = await cur.fetchone()
                    assert result is not None

        run_async(_test())

    def test_ensure_global_library_created(self, temp_deinsight_home):
        """_ensure_global_library creates __global__ project."""
        async def _test():
            from projects.manager import ProjectManager
            import paths
            manager = ProjectManager()
            await manager._ensure_db()

            global_proj = await manager.get_project(paths.GLOBAL_PROJECT_ID)
            assert global_proj is not None
            assert global_proj["name"] == "全局文獻庫"

        run_async(_test())

    def test_create_project_returns_dict(self, temp_deinsight_home):
        """create_project returns project dict with id, name."""
        async def _test():
            from projects.manager import ProjectManager
            manager = ProjectManager()
            proj = await manager.create_project("Test Project", "Test Description")

            assert proj is not None
            assert "id" in proj
            assert proj["name"] == "Test Project"
            assert proj["description"] == "Test Description"

        run_async(_test())

    def test_list_projects_global_first(self, temp_deinsight_home):
        """list_projects returns global project first."""
        async def _test():
            from projects.manager import ProjectManager
            import paths
            manager = ProjectManager()
            await manager.create_project("Project A")
            await manager.create_project("Project B")

            projects = await manager.list_projects()
            assert len(projects) >= 3
            assert projects[0]["id"] == paths.GLOBAL_PROJECT_ID

        run_async(_test())

    def test_get_project_by_id(self, temp_deinsight_home):
        """get_project returns single project."""
        async def _test():
            from projects.manager import ProjectManager
            manager = ProjectManager()
            created = await manager.create_project("My Project")
            proj_id = created["id"]

            retrieved = await manager.get_project(proj_id)
            assert retrieved is not None
            assert retrieved["name"] == "My Project"

        run_async(_test())

    def test_delete_project(self, temp_deinsight_home):
        """delete_project removes project and its directory."""
        async def _test():
            from projects.manager import ProjectManager
            import paths
            manager = ProjectManager()
            created = await manager.create_project("To Delete")
            proj_id = created["id"]

            proj_root = paths.project_root(proj_id)
            assert proj_root.exists()

            await manager.delete_project(proj_id)

            assert not proj_root.exists()

        run_async(_test())

    def test_delete_global_project_raises_error(self, temp_deinsight_home):
        """delete_project raises ValueError for __global__."""
        async def _test():
            from projects.manager import ProjectManager
            import paths
            manager = ProjectManager()

            with pytest.raises(ValueError, match="全局文獻庫不可刪除"):
                await manager.delete_project(paths.GLOBAL_PROJECT_ID)

        run_async(_test())

    def test_is_global_project(self, temp_deinsight_home):
        """is_global_project identifies global project."""
        async def _test():
            from projects.manager import ProjectManager
            import paths
            manager = ProjectManager()
            assert manager.is_global_project(paths.GLOBAL_PROJECT_ID) is True

            proj = await manager.create_project("Regular")
            assert manager.is_global_project(proj["id"]) is False

        run_async(_test())

    def test_get_project_data_dir(self, temp_deinsight_home):
        """get_project_data_dir returns project root."""
        async def _test():
            from projects.manager import ProjectManager
            import paths
            manager = ProjectManager()
            proj = await manager.create_project("Data Dir Test")
            proj_id = proj["id"]

            data_dir = manager.get_project_data_dir(proj_id)
            assert data_dir == paths.project_root(proj_id)

        run_async(_test())


# ============================================================================
# D. Path Management Tests (8+ tests)
# ============================================================================

class TestPathManagement:
    """Test paths module functions."""

    def test_version_is_correct(self):
        """__version__ is '1.0.0-pre'."""
        import paths
        assert paths.__version__ == "1.0.0-pre"

    def test_project_root_returns_path(self, temp_deinsight_home):
        """project_root returns correct path."""
        import paths
        root = paths.project_root("test-proj")
        assert isinstance(root, Path)
        assert "test-proj" in str(root)

    def test_validate_project_id_valid_ids_pass(self):
        """_validate_project_id accepts valid IDs."""
        import paths
        paths._validate_project_id("simple-id")
        paths._validate_project_id("proj_123")
        paths._validate_project_id("a")

    def test_validate_project_id_blocks_path_traversal(self):
        """_validate_project_id rejects '..' in ID."""
        import paths
        with pytest.raises(ValueError):
            paths._validate_project_id("../etc/passwd")

    def test_validate_project_id_blocks_slashes(self):
        """_validate_project_id rejects '/' and '\\' in ID."""
        import paths
        with pytest.raises(ValueError):
            paths._validate_project_id("foo/bar")
        with pytest.raises(ValueError):
            paths._validate_project_id("foo\\bar")

    def test_validate_project_id_global_passes(self):
        """_validate_project_id accepts GLOBAL_PROJECT_ID."""
        import paths
        paths._validate_project_id(paths.GLOBAL_PROJECT_ID)

    def test_ensure_project_dirs_creates_subdirs(self, temp_deinsight_home):
        """ensure_project_dirs creates lancedb, lightrag, documents."""
        import paths
        proj_id = "test-proj-dirs"
        root = paths.ensure_project_dirs(proj_id)

        assert (root / "lancedb").exists()
        assert (root / "lightrag").exists()
        assert (root / "documents").exists()

    def test_app_db_path_creates_parent_dirs(self, temp_deinsight_home):
        """app_db_path creates parent directories."""
        import paths
        db_path = paths.app_db_path()
        assert db_path.parent.exists()

    def test_data_root_uses_env_var(self):
        """DATA_ROOT respects DEINSIGHT_DATA_VERSION env var."""
        import paths
        assert "v0.7" in str(paths.DATA_ROOT)


# ============================================================================
# E. DB Connection Pool Tests (5+ tests)
# ============================================================================

class TestDBConnectionPool:
    """Test utils.db_pool connection pooling."""

    def test_get_connection_returns_working_connection(self, temp_deinsight_home):
        """get_connection returns functional connection."""
        async def _test():
            from utils.db_pool import get_connection
            db_path = temp_deinsight_home / "test.db"
            async with get_connection(db_path) as db:
                await db.execute("CREATE TABLE IF NOT EXISTS test (id INTEGER)")
                await db.commit()

            async with get_connection(db_path) as db:
                async with db.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name='test'"
                ) as cur:
                    result = await cur.fetchone()
                    assert result is not None

        run_async(_test())

    def test_same_path_returns_same_connection(self, temp_deinsight_home):
        """Same db_path returns pooled connection."""
        async def _test():
            from utils.db_pool import get_connection
            db_path = temp_deinsight_home / "pooled.db"

            async with get_connection(db_path) as db1:
                id1 = id(db1)
            async with get_connection(db_path) as db2:
                id2 = id(db2)

            assert id1 == id2

        run_async(_test())

    def test_wal_mode_is_set(self, temp_deinsight_home):
        """Connection has WAL mode enabled."""
        async def _test():
            from utils.db_pool import get_connection
            db_path = temp_deinsight_home / "wal_test.db"

            async with get_connection(db_path) as db:
                async with db.execute("PRAGMA journal_mode") as cur:
                    result = await cur.fetchone()
                    assert result[0].upper() in ("WAL", "wal")

        run_async(_test())

    def test_stale_connection_auto_reconnects(self, temp_deinsight_home):
        """Pool detects and reconnects stale connections."""
        async def _test():
            from utils.db_pool import get_connection, _pool
            db_path = temp_deinsight_home / "stale_test.db"

            async with get_connection(db_path) as db:
                await db.execute("CREATE TABLE stale_test (id INTEGER)")

            if str(db_path) in _pool:
                try:
                    await _pool[str(db_path)].close()
                except Exception:
                    pass

            async with get_connection(db_path) as db:
                async with db.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name='stale_test'"
                ) as cur:
                    result = await cur.fetchone()
                    assert result is not None

        run_async(_test())
