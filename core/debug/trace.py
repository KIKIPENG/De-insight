"""Debug and observability module for De-insight v2 Core.

This module provides structured trace objects and telemetry for debugging
semantic failures in the core pipeline.

References:
- Tech spec: deinsight_phase2_test_debug_spec.md
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class ExtractorTelemetry:
    """Telemetry data from the thought extractor.

    Attributes:
        raw_output_length: Length of raw LLM output
        parse_success: Whether initial parse succeeded
        repair_attempted: Whether repair was attempted
        repair_success: Whether repair succeeded
        normalized_fields: Fields that were normalized/coerced
        dropped_fields: Fields that were dropped
        validation_errors: Schema validation errors encountered
    """

    raw_output_length: int = 0
    parse_success: bool = False
    repair_attempted: bool = False
    repair_success: bool = False
    normalized_fields: list[str] = field(default_factory=list)
    dropped_fields: list[str] = field(default_factory=list)
    validation_errors: list[str] = field(default_factory=list)


@dataclass
class PlannerDecision:
    """Decision log from the retrieval planner.

    Attributes:
        query_mode: fast or deep
        why_fast: Explanation if fast mode
        why_deep: Explanation if deep mode
        patterns_triggered: Patterns that triggered the decision
        routes_generated: Retrieval routes created
        routes_omitted: Routes that were omitted and why
    """

    query_mode: str = "fast"
    why_fast: str | None = None
    why_deep: str | None = None
    patterns_triggered: list[str] = field(default_factory=list)
    routes_generated: list[str] = field(default_factory=list)
    routes_omitted: list[str] = field(default_factory=list)


@dataclass
class StoreWriteRecord:
    """Audit log for store writes.

    Attributes:
        store_name: Name of the store (claim, thought, concept, bridge)
        record_type: Type of record written
        project_id: Project identifier
        record_id: ID of the written record
        success: Whether write succeeded
        timing_ms: Time taken for write in milliseconds
        skipped: Whether write was skipped
    """

    store_name: str = ""
    record_type: str = ""
    project_id: str = ""
    record_id: str = ""
    success: bool = False
    timing_ms: float = 0.0
    skipped: bool = False


@dataclass
class PipelineTrace:
    """Structured trace object for core pipeline execution.

    This is the main debug artifact for understanding pipeline behavior.

    Attributes:
        trace_id: Unique identifier for this trace
        project_id: Project identifier
        user_message_preview: First 100 chars of user message
        core_enabled: Whether core pipeline was enabled
        classification_mode: fast or deep
        extractor_telemetry: Extractor-specific telemetry
        extractor_success: Whether extraction succeeded
        extractor_repair_attempted: Whether repair was attempted
        extractor_warnings: Warnings from extraction
        core_claim_summary: Summary of extracted core claim
        value_axes: Extracted value axes
        abstract_patterns: Extracted abstract patterns
        theory_hints: Extracted theory hints
        planner_decision: Planner decision log
        retrieval_plan_summary: Summary of retrieval plan
        legacy_delegate_used: Whether legacy pipeline was used
        store_writes: List of store write records
        fallback_triggered: Whether fallback was triggered
        fallback_reason: Reason for fallback
        errors: List of errors encountered
        timings: Timing information for each stage
        created_at: Timestamp when trace was created
    """

    trace_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    project_id: str = "default"
    user_message_preview: str = ""
    core_enabled: bool = False
    classification_mode: str = "fast"
    extractor_telemetry: ExtractorTelemetry | None = None
    extractor_success: bool = False
    extractor_repair_attempted: bool = False
    extractor_warnings: list[str] = field(default_factory=list)
    core_claim_summary: str = ""
    value_axes: list[str] = field(default_factory=list)
    abstract_patterns: list[str] = field(default_factory=list)
    theory_hints: list[str] = field(default_factory=list)
    planner_decision: PlannerDecision | None = None
    retrieval_plan_summary: str = ""
    legacy_delegate_used: bool = False
    store_writes: list[StoreWriteRecord] = field(default_factory=list)
    fallback_triggered: bool = False
    fallback_reason: str | None = None
    errors: list[str] = field(default_factory=list)
    timings: dict[str, float] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        """Convert trace to JSON-serializable dict."""
        return {
            "trace_id": self.trace_id,
            "project_id": self.project_id,
            "user_message_preview": self.user_message_preview,
            "core_enabled": self.core_enabled,
            "classification_mode": self.classification_mode,
            "extractor_telemetry": {
                "raw_output_length": self.extractor_telemetry.raw_output_length if self.extractor_telemetry else 0,
                "parse_success": self.extractor_telemetry.parse_success if self.extractor_telemetry else False,
                "repair_attempted": self.extractor_telemetry.repair_attempted if self.extractor_telemetry else False,
                "repair_success": self.extractor_telemetry.repair_success if self.extractor_telemetry else False,
                "normalized_fields": self.extractor_telemetry.normalized_fields if self.extractor_telemetry else [],
                "dropped_fields": self.extractor_telemetry.dropped_fields if self.extractor_telemetry else [],
                "validation_errors": self.extractor_telemetry.validation_errors if self.extractor_telemetry else [],
            } if self.extractor_telemetry else None,
            "extractor_success": self.extractor_success,
            "extractor_repair_attempted": self.extractor_repair_attempted,
            "extractor_warnings": self.extractor_warnings,
            "core_claim_summary": self.core_claim_summary,
            "value_axes": self.value_axes,
            "abstract_patterns": self.abstract_patterns,
            "theory_hints": self.theory_hints,
            "planner_decision": {
                "query_mode": self.planner_decision.query_mode if self.planner_decision else "unknown",
                "why_fast": self.planner_decision.why_fast if self.planner_decision else None,
                "why_deep": self.planner_decision.why_deep if self.planner_decision else None,
                "patterns_triggered": self.planner_decision.patterns_triggered if self.planner_decision else [],
                "routes_generated": self.planner_decision.routes_generated if self.planner_decision else [],
                "routes_omitted": self.planner_decision.routes_omitted if self.planner_decision else [],
            } if self.planner_decision else None,
            "retrieval_plan_summary": self.retrieval_plan_summary,
            "legacy_delegate_used": self.legacy_delegate_used,
            "store_writes": [
                {
                    "store_name": w.store_name,
                    "record_type": w.record_type,
                    "project_id": w.project_id,
                    "record_id": w.record_id,
                    "success": w.success,
                    "timing_ms": w.timing_ms,
                    "skipped": w.skipped,
                }
                for w in self.store_writes
            ],
            "fallback_triggered": self.fallback_triggered,
            "fallback_reason": self.fallback_reason,
            "errors": self.errors,
            "timings": self.timings,
            "created_at": self.created_at.isoformat(),
        }


class TraceContext:
    """Context manager for pipeline traces.

    Usage:
        async with TraceContext("my-project") as trace:
            trace.user_message_preview = user_message[:100]
            # ... do pipeline work ...
            trace.extractor_success = True
            trace.timings["extraction"] = elapsed
    """

    def __init__(self, project_id: str = "default", user_message: str = ""):
        """Initialize trace context.

        Args:
            project_id: Project identifier
            user_message: User message for preview
        """
        self.trace = PipelineTrace(
            project_id=project_id,
            user_message_preview=user_message[:100] if user_message else "",
        )
        self._start_time = time.time()

    async def __aenter__(self) -> PipelineTrace:
        """Enter context."""
        return self.trace

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit context."""
        elapsed = time.time() - self._start_time
        self.trace.timings["total"] = elapsed
        if exc_type is not None:
            self.trace.errors.append(f"{exc_type.__name__}: {exc_val}")

    def add_store_write(
        self,
        store_name: str,
        record_type: str,
        record_id: str,
        success: bool = True,
        skipped: bool = False,
    ) -> None:
        """Record a store write.

        Args:
            store_name: Name of store
            record_type: Type of record
            record_id: ID of record
            success: Whether write succeeded
            skipped: Whether write was skipped
        """
        self.trace.store_writes.append(StoreWriteRecord(
            store_name=store_name,
            record_type=record_type,
            project_id=self.trace.project_id,
            record_id=record_id,
            success=success,
            skipped=skipped,
        ))


# Global trace collector for debugging
_traces: list[PipelineTrace] = []


def get_traces() -> list[PipelineTrace]:
    """Get all collected traces."""
    return list(_traces)


def clear_traces() -> None:
    """Clear all collected traces."""
    _traces.clear()


def collect_trace(trace: PipelineTrace) -> None:
    """Add a trace to the global collector.

    Args:
        trace: Pipeline trace to collect
    """
    _traces.append(trace)
