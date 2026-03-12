"""De-insight v2 Core Schemas.

These Pydantic models define the core entities for the thought-structure-centric
architecture. They replace the chunk-centric RAG approach with structured
thought extraction, concept mapping, and bridge-based retrieval.

References:
- Tech spec: deinsight_v2_codex_technical_spec.md
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


class SourceKind(str, Enum):
    """The origin of a claim."""

    DOCUMENT_PASSAGE = "document_passage"
    USER_UTTERANCE = "user_utterance"
    IMAGE_SUMMARY = "image_summary"


class ThoughtStatus(str, Enum):
    """Status of a thought unit in the user's thinking evolution."""

    EMERGING = "emerging"
    STABLE = "stable"
    CONTESTED = "contested"


class OwnerKind(str, Enum):
    """The type of owner for a concept mapping."""

    CLAIM = "claim"
    PASSAGE = "passage"
    THOUGHT_UNIT = "thought_unit"


class VocabSource(str, Enum):
    """The vocabulary source for concept mapping."""

    AAT = "aat"  # AAT (Art & Architecture Thesaurus)
    INTERNAL = "internal"


class BridgeType(str, Enum):
    """Type of cross-domain or structural relation between claims."""

    ANALOGY = "analogy"
    TRADITION_LINK = "tradition_link"
    VALUE_STRUCTURE_MATCH = "value_structure_match"


class QueryMode(str, Enum):
    """Retrieval mode selection."""

    FAST = "fast"
    DEEP = "deep"


class Claim(BaseModel):
    """Represents a structured proposition extracted from a source.

    A claim is a declarative statement that can be evaluated, challenged,
    or mapped to theoretical frameworks. It is the atomic unit of
    thought structure in the v2 architecture.

    Fields:
        claim_id: Unique identifier (UUID format)
        project_id: Project isolation boundary
        source_kind: Origin of the claim
        source_id: Reference to source document/utterance
        core_claim: The core proposition (1-2 sentences)
        critique_target: What this claim critiques or challenges
        value_axes: Value dimensions this claim engages with
        materiality_axes: Material/production considerations
        labor_time_axes: Labor/time-related dimensions
        abstract_patterns: Structural patterns this claim relates to
        theory_hints: Theoretical frameworks this might connect to
        confidence: Extraction confidence (0.0-1.0)
        created_at: Creation timestamp
    """

    claim_id: str = Field(default_factory=lambda: __import__("uuid").uuid4().hex)
    project_id: str = "default"
    source_kind: SourceKind = SourceKind.USER_UTTERANCE
    source_id: str = ""
    core_claim: str = ""
    critique_target: list[str] = Field(default_factory=list)
    value_axes: list[str] = Field(default_factory=list)
    materiality_axes: list[str] = Field(default_factory=list)
    labor_time_axes: list[str] = Field(default_factory=list)
    abstract_patterns: list[str] = Field(default_factory=list)
    theory_hints: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    created_at: datetime = Field(default_factory=datetime.now)

    class Config:
        use_enum_values = True


class ThoughtUnit(BaseModel):
    """Represents a stable or emerging user-level thought tracked across conversations.

    A thought unit aggregates one or more claims into a coherent thinking unit.
    It tracks the evolution of the user's ideas over time.

    Fields:
        thought_id: Unique identifier
        project_id: Project isolation boundary
        title: Short descriptive title
        summary: Concise summary of the thought
        core_claim_ids: Claims that form this thought's foundation
        value_axes: Key value dimensions this thought engages
        recurring_patterns: Patterns that repeat across this thought
        supporting_claim_ids: Claims that support this thought
        status: Current status in thinking evolution
        last_updated_at: Last update timestamp
    """

    thought_id: str = Field(default_factory=lambda: __import__("uuid").uuid4().hex)
    project_id: str = "default"
    title: str = ""
    summary: str = ""
    core_claim_ids: list[str] = Field(default_factory=list)
    value_axes: list[str] = Field(default_factory=list)
    recurring_patterns: list[str] = Field(default_factory=list)
    supporting_claim_ids: list[str] = Field(default_factory=list)
    status: ThoughtStatus = ThoughtStatus.EMERGING
    last_updated_at: datetime = Field(default_factory=datetime.now)

    class Config:
        use_enum_values = True


class ConceptMapping(BaseModel):
    """Maps a claim/passage/thought to controlled concepts.

    Concept mappings provide structured categorization using controlled
    vocabularies (AAT or internal).

    Fields:
        mapping_id: Unique identifier
        project_id: Project isolation boundary
        owner_kind: Type of owner entity
        owner_id: Reference to owner
        vocab_source: Vocabulary source (AAT or internal)
        concept_id: Concept identifier
        preferred_label: Human-readable label
        alt_labels: Alternative labels
        broader_terms: Broader concept terms
        related_terms: Related concept terms
        confidence: Mapping confidence (0.0-1.0)
    """

    mapping_id: str = Field(default_factory=lambda: __import__("uuid").uuid4().hex)
    project_id: str = "default"
    owner_kind: OwnerKind = OwnerKind.CLAIM
    owner_id: str = ""
    vocab_source: VocabSource = VocabSource.INTERNAL
    concept_id: str = ""
    preferred_label: str = ""
    alt_labels: list[str] = Field(default_factory=list)
    broader_terms: list[str] = Field(default_factory=list)
    related_terms: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)

    class Config:
        use_enum_values = True


class Bridge(BaseModel):
    """Represents a cross-domain or structural relation between claims.

    Bridges connect claims from potentially different domains based on
    structural or analogical similarity.

    Fields:
        bridge_id: Unique identifier
        project_id: Project isolation boundary
        source_claim_id: Source claim for the bridge
        target_claim_id: Target claim for the bridge
        bridge_type: Type of bridge connection
        reason_summary: Human-readable explanation of the connection
        shared_patterns: Structural patterns shared between claims
        confidence: Bridge confidence (0.0-1.0)
        score: Bridge ranking score from BridgeRanker (0.0-1.0)
        score_breakdown: Detailed score breakdown by signal type
        created_at: Creation timestamp
    """

    bridge_id: str = Field(default_factory=lambda: __import__("uuid").uuid4().hex)
    project_id: str = "default"
    source_claim_id: str = ""
    target_claim_id: str = ""
    bridge_type: BridgeType = BridgeType.ANALOGY
    reason_summary: str = ""
    shared_patterns: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    score: float | None = None
    score_breakdown: dict[str, float] | None = None
    created_at: datetime = Field(default_factory=datetime.now)

    class Config:
        use_enum_values = True


class RetrievalPlan(BaseModel):
    """Represents a plan for fast/deep retrieval.

    The retrieval plan defines how to query the knowledge base,
    including concept queries, supporting paths, and analogy paths.

    Fields:
        plan_id: Unique identifier
        project_id: Project isolation boundary
        query_mode: fast or deep
        why_deep: Explanation of why deep mode was selected (None for fast)
        thought_summary: Summary of the user's thought
        concept_queries: List of concept-oriented queries
        supporting_paths: Paths for supporting evidence
        analogy_paths: Paths for analogical connections
        max_passages_per_path: Maximum passages to retrieve per path
        created_at: Creation timestamp
    """

    plan_id: str = Field(default_factory=lambda: __import__("uuid").uuid4().hex)
    project_id: str = "default"
    query_mode: QueryMode = QueryMode.FAST
    why_deep: str | None = None
    thought_summary: str = ""
    concept_queries: list[str] = Field(default_factory=list)
    supporting_paths: list[str] = Field(default_factory=list)
    analogy_paths: list[str] = Field(default_factory=list)
    max_passages_per_path: int = 3
    created_at: datetime = Field(default_factory=datetime.now)

    class Config:
        use_enum_values = True


class ExtractionResult(BaseModel):
    """Result of thought extraction from user utterance.

    This is the output format from ThoughtExtractor, containing
    structured typed data rather than freeform text.

    Fields:
        claims: Extracted claims from the utterance
        thought_unit: Aggregated thought unit (if applicable)
        concept_mappings: Concept mappings for the extracted content
        raw_utterance: Original user input
        was_extracted: Whether meaningful content was found
    """

    claims: list[Claim] = Field(default_factory=list)
    thought_unit: ThoughtUnit | None = None
    concept_mappings: list[ConceptMapping] = Field(default_factory=list)
    raw_utterance: str = ""
    was_extracted: bool = False


class RetrievalResult(BaseModel):
    """Result of retrieval execution.

    Fields:
        plan: The retrieval plan that was executed
        passages: Retrieved passages
        claims: Retrieved claims
        bridges: Relevant bridges
        sources: Source metadata for citations
    """

    plan: RetrievalPlan
    passages: list[dict] = Field(default_factory=list)
    claims: list[Claim] = Field(default_factory=list)
    bridges: list[Bridge] = Field(default_factory=list)
    sources: list[dict] = Field(default_factory=list)
