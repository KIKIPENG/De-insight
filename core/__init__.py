"""De-insight v2 Core Package.

This package provides the thought-structure-centric architecture for De-insight.
It replaces the chunk-centric RAG approach with structured thought extraction,
concept mapping, and bridge-based retrieval.

Phase 1 Components:
- schemas: Pydantic models for Claim, ThoughtUnit, ConceptMapping, Bridge, RetrievalPlan
- stores: SQLite-backed persistence for v2 entities
- thought_extractor: Extract structured thought from user utterances
- query_classifier: Determine fast vs deep retrieval mode
- retrieval_planner: Plan retrieval strategies
- retriever: Execute retrieval based on plans
- compat: Compatibility layer for integration with existing system

Usage:
    # Enable core pipeline (disabled by default for Phase 1)
    from core.compat import enable_core_pipeline
    enable_core_pipeline()

    # Or use individual components
    from core.schemas import Claim, ThoughtUnit
    from core.query_classifier import classify_query
    from core.retrieval_planner import create_retrieval_plan

References:
- Tech spec: deinsight_v2_codex_technical_spec.md
"""

# Schema models
from core.schemas import (
    Bridge,
    BridgeType,
    Claim,
    ConceptMapping,
    ExtractionResult,
    OwnerKind,
    QueryMode,
    RetrievalPlan,
    RetrievalResult,
    SourceKind,
    ThoughtStatus,
    ThoughtUnit,
    VocabSource,
)

# Stores
from core.stores import BridgeStore, ClaimStore, ConceptStore, ThoughtStore

# Core modules
from core.compat import (
    CorePipelineInput,
    CorePipelineOutput,
    classify_query_mode,
    disable_core_pipeline,
    enable_core_pipeline,
    extract_thought,
    is_core_enabled,
    plan_retrieval,
    run_core_pipeline,
    run_legacy_pipeline,
)

from core.query_classifier import QueryClassifier, classify_query
from core.retrieval_planner import RetrievalPlanner, create_retrieval_plan
from core.retriever import Retriever, retrieve_with_plan
from core.thought_extractor import LLMCallable, ThoughtExtractor, quick_extract
from core.concept_mapper import ConceptMapper, map_text_async

__all__ = [
    # Schemas
    "Bridge",
    "BridgeType",
    "Claim",
    "ConceptMapping",
    "ExtractionResult",
    "OwnerKind",
    "QueryMode",
    "RetrievalPlan",
    "RetrievalResult",
    "SourceKind",
    "ThoughtStatus",
    "ThoughtUnit",
    "VocabSource",
    # Stores
    "BridgeStore",
    "ClaimStore",
    "ConceptStore",
    "ThoughtStore",
    # Modules
    "ConceptMapper",
    "QueryClassifier",
    "RetrievalPlanner",
    "Retriever",
    "ThoughtExtractor",
    "LLMCallable",
    "map_text_async",
    # Compat
    "CorePipelineInput",
    "CorePipelineOutput",
    "classify_query",
    "classify_query_mode",
    "create_retrieval_plan",
    "disable_core_pipeline",
    "enable_core_pipeline",
    "extract_thought",
    "is_core_enabled",
    "plan_retrieval",
    "quick_extract",
    "retrieve_with_plan",
    "run_core_pipeline",
    "run_legacy_pipeline",
]
