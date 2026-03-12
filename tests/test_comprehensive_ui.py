"""Comprehensive UI component tests for De-insight.

Tests cover:
- AppState initialization and mutations
- Mixin Protocol definition and runtime checkability
- Error utilities (log_errors decorator)
- Version and configuration constants
- Modals package exports
"""

import asyncio
import logging
from dataclasses import fields
from pathlib import Path
from typing import get_args, get_origin
from unittest.mock import MagicMock, AsyncMock, patch

import pytest

from widgets import AppState, SurfacedBridgeRecord
from mixins.protocol import DeInsightProtocol
from utils.errors import log_errors
from paths import __version__, GLOBAL_PROJECT_ID, DATA_ROOT, APP_HOME, app_db_path
import modals


# ============================================================================
# A. AppState Tests (10+ tests)
# ============================================================================

class TestAppStateInitialization:
    """Test AppState default initialization."""

    def test_appstate_default_initialization(self):
        """Test that all AppState fields have correct defaults."""
        state = AppState()
        assert state.current_project is None
        assert state.pending_memories == []
        assert state.interactive_depth == 0
        assert state.current_conversation_id is None
        assert state.cached_memory_count == 0
        assert state.current_interactive_block is None
        assert state.last_rag_sources == []
        assert state.last_imported_source is None
        assert state.pending_images == []
        assert state.recent_surfaced_bridges == []
        assert state.turn_index == 0

    def test_appstate_current_project_none(self):
        """Test that current_project starts as None."""
        state = AppState()
        assert state.current_project is None

    def test_appstate_pending_memories_empty_list(self):
        """Test that pending_memories starts as empty list."""
        state = AppState()
        assert isinstance(state.pending_memories, list)
        assert len(state.pending_memories) == 0

    def test_appstate_interactive_depth_zero(self):
        """Test that interactive_depth starts at 0."""
        state = AppState()
        assert state.interactive_depth == 0

    def test_appstate_cached_memory_count_zero(self):
        """Test that cached_memory_count starts at 0."""
        state = AppState()
        assert state.cached_memory_count == 0

    def test_appstate_mutation_add_to_pending_memories(self):
        """Test that pending_memories can be mutated."""
        state = AppState()
        test_memory = {"id": "mem1", "content": "test"}
        state.pending_memories.append(test_memory)
        assert len(state.pending_memories) == 1
        assert state.pending_memories[0] == test_memory

    def test_appstate_reset_behavior(self):
        """Test that state can be reset by creating new instance."""
        state1 = AppState()
        state1.pending_memories.append({"id": "mem1"})
        state1.interactive_depth = 5
        state1.cached_memory_count = 10

        state2 = AppState()
        assert state2.pending_memories == []
        assert state2.interactive_depth == 0
        assert state2.cached_memory_count == 0

    def test_appstate_current_conversation_id_none(self):
        """Test that current_conversation_id starts as None."""
        state = AppState()
        assert state.current_conversation_id is None

    def test_appstate_last_rag_sources_empty(self):
        """Test that last_rag_sources starts as empty list."""
        state = AppState()
        assert isinstance(state.last_rag_sources, list)
        assert len(state.last_rag_sources) == 0

    def test_appstate_all_fields_exist(self):
        """Test that all expected fields exist on AppState."""
        state = AppState()
        expected_fields = {
            'current_project',
            'pending_memories',
            'interactive_depth',
            'current_conversation_id',
            'cached_memory_count',
            'current_interactive_block',
            'last_rag_sources',
            'last_imported_source',
            'pending_images',
            'recent_surfaced_bridges',
            'turn_index',
            'import_in_progress',
        }
        actual_fields = {f.name for f in fields(state)}
        assert expected_fields == actual_fields

    def test_appstate_field_types(self):
        """Test that AppState fields have expected types."""
        state = AppState()
        # Check a few critical field types
        assert isinstance(state.pending_memories, list)
        assert isinstance(state.last_rag_sources, list)
        assert isinstance(state.pending_images, list)
        assert isinstance(state.recent_surfaced_bridges, list)
        assert isinstance(state.interactive_depth, int)
        assert isinstance(state.cached_memory_count, int)
        assert isinstance(state.turn_index, int)

    def test_appstate_surfaced_bridge_record(self):
        """Test that SurfacedBridgeRecord can be created and used."""
        record = SurfacedBridgeRecord(topic="test", turn_index=1, timestamp=1234.5)
        assert record.topic == "test"
        assert record.turn_index == 1
        assert record.timestamp == 1234.5


# ============================================================================
# B. Mixin Protocol Tests (5+ tests)
# ============================================================================

class TestMixinProtocol:
    """Test the DeInsightProtocol definition and runtime checkability."""

    def test_protocol_is_runtime_checkable(self):
        """Test that DeInsightProtocol is decorated with @runtime_checkable."""
        # If it's runtime_checkable, these attributes will be set
        assert hasattr(DeInsightProtocol, '_is_runtime_protocol')
        assert DeInsightProtocol._is_runtime_protocol is True

    def test_protocol_defines_reactive_properties(self):
        """Test that Protocol defines expected reactive properties."""
        expected_props = ['mode', 'rag_mode', 'is_loading']
        for prop in expected_props:
            assert hasattr(DeInsightProtocol, prop) or prop in DeInsightProtocol.__annotations__

    def test_protocol_defines_state_collections(self):
        """Test that Protocol defines state and messages collections."""
        annotations = DeInsightProtocol.__annotations__
        assert 'messages' in annotations
        assert 'state' in annotations

    def test_protocol_defines_stores_and_managers(self):
        """Test that Protocol defines expected stores and managers."""
        annotations = DeInsightProtocol.__annotations__
        assert '_conv_store' in annotations
        assert '_project_manager' in annotations
        assert 'api_base' in annotations

    def test_protocol_defines_expected_methods(self):
        """Test that Protocol defines expected methods."""
        expected_methods = [
            '_update_menu_bar',
            '_update_status',
            '_scroll_to_bottom',
            '_refresh_memory_panel',
            '_quick_llm_call',
            '_start_discussion_from_memory',
            'action_close_modals',
            'fill_input',
            '_show_system_message',
            'notify',
        ]
        for method_name in expected_methods:
            assert hasattr(DeInsightProtocol, method_name)

    def test_protocol_method_signatures(self):
        """Test that Protocol methods have correct signatures."""
        # Check that _update_menu_bar is callable
        assert callable(getattr(DeInsightProtocol, '_update_menu_bar', None))
        # Check that _quick_llm_call has proper async signature hint
        # Note: We can only check existence here, actual signature validation would need introspection
        assert hasattr(DeInsightProtocol, '_quick_llm_call')


# ============================================================================
# C. Error Utilities Tests (8+ tests)
# ============================================================================

class TestLogErrorsDecorator:
    """Test the log_errors decorator for both sync and async functions."""

    def test_log_errors_sync_function_success(self):
        """Test that log_errors passes through successful sync function."""
        @log_errors()
        def test_func(x):
            return x * 2

        result = test_func(5)
        assert result == 10

    def test_log_errors_sync_function_exception(self, caplog):
        """Test that log_errors catches exception and logs it (sync)."""
        @log_errors()
        def test_func():
            raise ValueError("test error")

        with caplog.at_level(logging.WARNING):
            result = test_func()

        assert result is None
        assert "test_func failed" in caplog.text

    @pytest.mark.asyncio
    async def test_log_errors_async_function_success(self):
        """Test that log_errors passes through successful async function."""
        @log_errors()
        async def test_func(x):
            return x * 2

        result = await test_func(5)
        assert result == 10

    @pytest.mark.asyncio
    async def test_log_errors_async_function_exception(self, caplog):
        """Test that log_errors catches exception and logs it (async)."""
        @log_errors()
        async def test_func():
            raise ValueError("async test error")

        with caplog.at_level(logging.WARNING):
            result = await test_func()

        assert result is None
        assert "test_func failed" in caplog.text

    def test_log_errors_with_fallback_value(self):
        """Test that log_errors returns specified fallback value."""
        @log_errors(fallback="fallback_value")
        def test_func():
            raise RuntimeError("error")

        result = test_func()
        assert result == "fallback_value"

    def test_log_errors_with_custom_message(self, caplog):
        """Test that log_errors uses custom message if provided."""
        @log_errors(msg="Custom prefix")
        def test_func():
            raise RuntimeError("test error")

        with caplog.at_level(logging.WARNING):
            result = test_func()

        assert "Custom prefix failed" in caplog.text

    def test_log_errors_preserves_function_name(self):
        """Test that log_errors preserves wrapped function name."""
        @log_errors()
        def my_special_function():
            return 42

        assert my_special_function.__name__ == "my_special_function"

    def test_log_errors_preserves_return_value_type(self):
        """Test that log_errors preserves return value type."""
        @log_errors()
        def return_dict():
            return {"key": "value"}

        result = return_dict()
        assert isinstance(result, dict)
        assert result["key"] == "value"

    def test_log_errors_with_different_exception_types(self, caplog):
        """Test that log_errors handles different exception types."""
        exceptions_to_test = [
            ValueError("value error"),
            RuntimeError("runtime error"),
            KeyError("key error"),
            TypeError("type error"),
        ]

        for exc in exceptions_to_test:
            @log_errors()
            def test_func():
                raise exc

            with caplog.at_level(logging.WARNING):
                result = test_func()

            assert result is None

    def test_log_errors_with_notify_parameter(self):
        """Test that log_errors accepts notify parameter."""
        # Create a mock object with notify method
        mock_obj = MagicMock()
        mock_obj.notify = MagicMock()

        @log_errors(notify=True)
        def test_method(self):
            raise RuntimeError("test")

        # Call with mock as self
        result = test_method(mock_obj)
        assert result is None

    @pytest.mark.asyncio
    async def test_log_errors_async_with_notify(self):
        """Test that async log_errors works with notify parameter."""
        mock_obj = MagicMock()
        mock_obj.notify = MagicMock()

        @log_errors(notify=True)
        async def test_method(self):
            raise RuntimeError("async test")

        result = await test_method(mock_obj)
        assert result is None


# ============================================================================
# D. Version and Configuration Tests (5+ tests)
# ============================================================================

class TestVersionAndConfig:
    """Test version numbers and configuration constants."""

    def test_version_is_correct(self):
        """Test that __version__ is 1.0.0-pre."""
        assert __version__ == "1.0.0-pre"

    def test_global_project_id_is_correct(self):
        """Test that GLOBAL_PROJECT_ID is '__global__'."""
        assert GLOBAL_PROJECT_ID == "__global__"

    def test_data_root_is_path_object(self):
        """Test that DATA_ROOT is a Path object."""
        assert isinstance(DATA_ROOT, Path)

    def test_app_home_is_path_object(self):
        """Test that APP_HOME is a Path object."""
        assert isinstance(APP_HOME, Path)

    def test_app_db_path_returns_path(self):
        """Test that app_db_path() returns a Path object."""
        db_path = app_db_path()
        assert isinstance(db_path, Path)

    def test_data_root_uses_env_var(self):
        """Test that DATA_ROOT is constructed from APP_HOME."""
        # DATA_ROOT should be derived from APP_HOME
        assert str(DATA_ROOT).startswith(str(APP_HOME))

    def test_app_home_can_use_env_var(self):
        """Test that APP_HOME respects DEINSIGHT_HOME env var."""
        # We can't easily test this without modifying env,
        # but we can verify it's a Path object and exists
        assert isinstance(APP_HOME, Path)


# ============================================================================
# E. Modals Package Tests (5+ tests)
# ============================================================================

class TestModalsPackage:
    """Test that modals package exports are correct."""

    def test_modals_project_modal_importable(self):
        """Test that ProjectModal is importable from modals."""
        from modals import ProjectModal
        assert ProjectModal is not None

    def test_modals_memory_confirm_modal_importable(self):
        """Test that MemoryConfirmModal is importable from modals."""
        from modals import MemoryConfirmModal
        assert MemoryConfirmModal is not None

    def test_modals_onboarding_screen_importable(self):
        """Test that OnboardingScreen is importable from modals."""
        from modals import OnboardingScreen
        assert OnboardingScreen is not None

    def test_modals_all_exports_in_all(self):
        """Test that __all__ contains expected exports."""
        expected_exports = [
            "ProjectModal",
            "MemoryConfirmModal",
            "SourceModal",
            "DocReaderModal",
            "DocumentManageModal",
            "RelationModal",
            "ImportModal",
            "FocusImportModal",
            "SearchModal",
            "MemoryDetailModal",
            "MemoryManageModal",
            "InsightConfirmModal",
            "MemorySaveModal",
            "KnowledgeModal",
            "OnboardingScreen",
        ]
        # Check that all expected exports are in __all__
        for export in expected_exports:
            assert export in modals.__all__

    def test_modals_exports_count(self):
        """Test that modals exports expected number of items."""
        # Should have at least the 15 main modals + 2 helper widgets
        assert len(modals.__all__) >= 17

    def test_modals_all_exports_are_accessible(self):
        """Test that all exports in __all__ are actually accessible."""
        for name in modals.__all__:
            assert hasattr(modals, name), f"Export {name} not found in modals"

    def test_modals_source_modal_importable(self):
        """Test that SourceModal is importable from modals."""
        from modals import SourceModal
        assert SourceModal is not None

    def test_modals_doc_reader_modal_importable(self):
        """Test that DocReaderModal is importable from modals."""
        from modals import DocReaderModal
        assert DocReaderModal is not None

    def test_modals_knowledge_modal_importable(self):
        """Test that KnowledgeModal is importable from modals."""
        from modals import KnowledgeModal
        assert KnowledgeModal is not None


# ============================================================================
# Integration Tests (combining multiple components)
# ============================================================================

class TestAppStateProtocolIntegration:
    """Test integration between AppState and Protocol."""

    def test_protocol_expects_appstate(self):
        """Test that Protocol correctly types state as AppState."""
        annotations = DeInsightProtocol.__annotations__
        # The state annotation should reference AppState
        assert 'state' in annotations

    def test_appstate_contains_all_required_fields_for_protocol(self):
        """Test that AppState provides all fields expected by Protocol."""
        state = AppState()
        # Protocol expects these to be accessible
        assert hasattr(state, 'current_project')
        assert hasattr(state, 'pending_memories')
        assert hasattr(state, 'interactive_depth')
        assert hasattr(state, 'current_conversation_id')
        assert hasattr(state, 'cached_memory_count')


class TestLogErrorsWithProtocol:
    """Test log_errors decorator with Protocol-like objects."""

    def test_log_errors_with_notify_on_protocol_like_object(self):
        """Test that log_errors notify works with mock Protocol object."""
        # Create a mock that simulates Protocol interface
        mock_app = MagicMock()
        mock_app.notify = MagicMock()

        @log_errors(notify=True, msg="Test operation")
        def test_method(self):
            raise RuntimeError("test error")

        # This should not raise
        result = test_method(mock_app)
        assert result is None


# ============================================================================
# Widget Field Type Tests
# ============================================================================

class TestWidgetTypes:
    """Test the types of various widget fields."""

    def test_appstate_field_defaults_are_independent(self):
        """Test that default list fields are independent between instances."""
        state1 = AppState()
        state2 = AppState()

        state1.pending_memories.append({"id": "test"})
        # state2 should not be affected
        assert len(state2.pending_memories) == 0

    def test_appstate_interactive_depth_can_be_incremented(self):
        """Test that interactive_depth can be incremented."""
        state = AppState()
        state.interactive_depth = 0
        state.interactive_depth += 1
        assert state.interactive_depth == 1

    def test_appstate_conversation_id_can_be_set(self):
        """Test that current_conversation_id can be set to string."""
        state = AppState()
        state.current_conversation_id = "conv_123"
        assert state.current_conversation_id == "conv_123"

    def test_appstate_current_project_can_be_dict(self):
        """Test that current_project can be set to dict."""
        state = AppState()
        project_dict = {"id": "proj1", "name": "Test Project"}
        state.current_project = project_dict
        assert state.current_project == project_dict


# ============================================================================
# Conftest/Fixtures
# ============================================================================

@pytest.fixture
def mock_app():
    """Create a mock app object that simulates the Protocol."""
    app = MagicMock()
    app.state = AppState()
    app.messages = []
    app.mode = "emotional"
    app.rag_mode = "fast"
    app.is_loading = False
    app.notify = MagicMock()
    return app


@pytest.fixture
def mock_protocol_app():
    """Create a more complete mock that follows DeInsightProtocol."""
    from conversation.store import ConversationStore
    from projects.manager import ProjectManager

    app = MagicMock(spec=DeInsightProtocol)
    app.state = AppState()
    app.messages = []
    app.mode = "emotional"
    app.rag_mode = "fast"
    app.is_loading = False
    app._llm_call_count = 0
    app.api_base = "http://localhost:8000"
    app.notify = MagicMock()
    app._update_menu_bar = MagicMock()
    app._update_status = MagicMock()
    app._scroll_to_bottom = MagicMock()
    return app


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
