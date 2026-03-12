"""Modals package — 從原 modals.py 拆分而來。

所有 ModalScreen 子類別和輔助函式向後相容匯出，
確保 `from modals import XModal` 繼續運作。
"""

from modals.project import ProjectModal
from modals.memory_confirm import MemoryConfirmModal
from modals.source import SourceModal
from modals.doc_reader import _Paragraph, DocReaderModal
from modals.document_manage import DocumentManageModal
from modals.relation import RelationModal
from modals.import_modal import ImportModal
from modals.focus_import import FocusImportModal
from modals.search import SearchModal
from modals.memory_detail import MemoryDetailModal
from modals.memory_manage import MemoryManageModal, _MemEntry
from modals.insight_confirm import InsightConfirmModal
from modals.memory_save import MemorySaveModal
from modals.knowledge import KnowledgeModal
from modals.onboarding import OnboardingScreen
from modals.health_dashboard import HealthDashboardModal

__all__ = [
    # Main modals (公開 API)
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
    "HealthDashboardModal",
    # Helper widgets (內部使用)
    "_Paragraph",
    "_MemEntry",
]
