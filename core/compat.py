"""Compatibility layer for De-insight v2 Core.

This module provides integration points between the new core pipeline
and the existing system. It allows existing chat paths to call the new
core without breaking current functionality.

References:
- Tech spec: deinsight_v2_codex_technical_spec.md
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Callable

from core.query_classifier import QueryClassifier, classify_query
from core.retrieval_planner import RetrievalPlanner, create_retrieval_plan
from core.retriever import Retriever, retrieve_with_plan
from core.schemas import (
    Claim,
    ConceptMapping,
    ExtractionResult,
    QueryMode,
    RetrievalPlan,
    RetrievalResult,
    ThoughtUnit,
)
from core.thought_extractor import ThoughtExtractor, quick_extract


# Flag to enable/disable core pipeline
_CORE_ENABLED = False  # Disabled by default for Phase 1


def enable_core_pipeline() -> None:
    """Enable the v2 core pipeline.

    This must be called explicitly to switch to the new pipeline.
    """
    global _CORE_ENABLED
    _CORE_ENABLED = True


def disable_core_pipeline() -> None:
    """Disable the v2 core pipeline (fallback to legacy)."""
    global _CORE_ENABLED
    _CORE_ENABLED = False


def is_core_enabled() -> bool:
    """Check if core pipeline is enabled."""
    return _CORE_ENABLED


@dataclass
class CorePipelineInput:
    """Input for the core pipeline.

    Attributes:
        user_message: User's input text
        project_id: Project identifier
        conversation_context: Previous messages in the conversation
        llm_callable: Callable for LLM requests (async function)
    """

    user_message: str
    project_id: str = "default"
    conversation_context: list[dict] | None = None
    llm_callable: Callable | None = None


@dataclass
class CorePipelineOutput:
    """Output from the core pipeline.

    Attributes:
        extraction: Extracted thought structure (claims, concepts, thought unit)
        retrieval_plan: Generated retrieval plan
        retrieval_result: Retrieved content
        context_text: Combined context for LLM response
        sources: Source metadata for citations
    """

    extraction: ExtractionResult | None = None
    retrieval_plan: RetrievalPlan | None = None
    retrieval_result: RetrievalResult | None = None
    context_text: str = ""
    sources: list[dict] = field(default_factory=list)


async def run_core_pipeline(
    input_data: CorePipelineInput,
) -> CorePipelineOutput:
    """Run the v2 core pipeline.

    This is the main entry point for the new core architecture.
    It performs:
    1. Thought extraction (structured claims from user input)
    2. Query classification (fast vs deep mode)
    3. Retrieval planning (concept queries, paths)
    4. Retrieval execution (multi-route if deep)

    Args:
        input_data: Pipeline input with user message and context

    Returns:
        CorePipelineOutput with extracted thought, plan, and retrieval results
    """
    if not _CORE_ENABLED:
        return CorePipelineOutput()

    output = CorePipelineOutput()

    # Step 1: Thought extraction
    if input_data.llm_callable:
        from core.thought_extractor import LLMCallable

        llm_wrapper = LLMCallable(func=input_data.llm_callable)
        extractor = ThoughtExtractor(llm_wrapper, input_data.project_id)
        output.extraction = await extractor.extract(input_data.user_message)

    # Step 2: Query classification
    classifier = QueryClassifier()
    classification = classifier.classify(
        input_data.user_message,
        input_data.conversation_context,
    )

    # Step 3: Retrieval planning
    planner = RetrievalPlanner()
    output.retrieval_plan = planner.create_plan(
        query=input_data.user_message,
        context=input_data.conversation_context,
        project_id=input_data.project_id,
    )

    # Step 4: Retrieval execution
    retriever = Retriever(input_data.project_id)
    output.retrieval_result = await retriever.retrieve(
        output.retrieval_plan,
        input_data.user_message,
    )

    # Step 5: Build context for LLM
    if output.retrieval_result:
        output.context_text = _build_context_text(
            output.extraction,
            output.retrieval_result,
        )
        output.sources = output.retrieval_result.sources

    return output


def _build_context_text(
    extraction: ExtractionResult | None,
    result: RetrievalResult,
) -> str:
    """Build context text for LLM from extraction and retrieval results.

    Args:
        extraction: Extraction result
        result: Retrieval result

    Returns:
        Context text string
    """
    parts = []

    # Add extraction context if available
    if extraction and extraction.was_extracted:
        if extraction.thought_unit:
            parts.append(
                f"# 使用者當前思考\n{extraction.thought_unit.summary}"
            )
        if extraction.claims:
            claims_text = "\n".join(
                f"- {c.core_claim}" for c in extraction.claims
            )
            parts.append(f"# 提取的主張\n{claims_text}")

    # Add retrieval context
    if result.passages:
        passages_text = "\n\n".join(
            str(p) for p in result.passages[:5]
        )
        parts.append(f"# 知識庫檢索結果\n{passages_text}")

    return "\n\n".join(parts)


async def run_legacy_pipeline(
    user_message: str,
    project_id: str = "default",
    mode: str = "fast",
) -> dict:
    """Run the legacy pipeline (fallback).

    This delegates to the existing rag.pipeline for backward compatibility.

    Args:
        user_message: User input
        project_id: Project identifier
        mode: Retrieval mode (fast or deep)

    Returns:
        Legacy pipeline result dict
    """
    try:
        from rag.pipeline import run_thinking_pipeline

        result = await run_thinking_pipeline(
            user_input=user_message,
            project_id=project_id,
            mode=mode,
        )
        return result
    except Exception:
        return {
            "context_text": "",
            "sources": [],
            "diagnostics": {},
        }


# Convenience functions for external callers

async def extract_thought(
    user_text: str,
    llm_callable: Any,
    project_id: str = "default",
) -> ExtractionResult:
    """Extract structured thought from user input.

    Args:
        user_text: User input
        llm_callable: LLM callable
        project_id: Project identifier

    Returns:
        ExtractionResult
    """
    return await quick_extract(user_text, llm_callable, project_id)


def classify_query_mode(
    query: str,
    context: list[dict] | None = None,
) -> tuple[QueryMode, str | None]:
    """Classify query mode.

    Args:
        query: User query
        context: Conversation context

    Returns:
        Tuple of (mode, why_deep)
    """
    result = classify_query(query, context)
    return result.mode, result.why_deep


def plan_retrieval(
    query: str,
    context: list[dict] | None = None,
    project_id: str = "default",
) -> RetrievalPlan:
    """Plan retrieval strategy.

    Args:
        query: User query
        context: Conversation context
        project_id: Project identifier

    Returns:
        RetrievalPlan
    """
    return create_retrieval_plan(query, context, project_id)

