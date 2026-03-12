# De-insight v2 Core Rewrite — Technical Spec for Codex

## 1. Goal

Rewrite the De-insight core around **thought structure**, not document chunks.

The new core must support:
- extracting structured thought from user dialogue
- mapping thought to concepts and theory directions
- planning multi-route retrieval for structural / analogical matches
- storing long-term thought units, not just memory snippets
- enriching ingested corpus with claims, concepts, and theory metadata

This is **not** a full rewrite of the product shell.
Keep the existing app shell, project isolation, API entrypoints, ingestion jobs, and image gallery where possible.

## 2. Current public repo structure to respect

The current repo publicly shows these top-level modules and files:
- `app.py`
- `backend/`
- `conversation/`
- `embeddings/`
- `frontend/`
- `interaction/`
- `memory/`
- `projects/`
- `rag/`
- `tui.py`
- supporting config / settings files

The README describes De-insight as:
- a TUI-first app with conversation modes
- per-project isolation for conversations, memories, knowledge base, and images
- a background-built knowledge graph
- fast/deep retrieval modes
- candidate memory extraction and thought evolution detection
- image analysis and cross-modal preference checking

Reference: GitHub repo structure and README. citeturn461559view0

## 3. Rewrite boundary

### Preserve
Preserve these responsibilities and adapt them rather than rewriting everything:
- project isolation in `projects/`
- backend entrypoint and router structure in `backend/`
- ingestion job orchestration and readiness tracking in `rag/`
- image upload / gallery shell in `frontend/` and image router layer
- top-level TUI / app shell

### Rewrite
Rewrite these core responsibilities:
- thought extraction
- memory data model
- retrieval planning
- structural / analogical ranking
- corpus annotation outputs
- cross-domain bridge generation

## 4. New architecture principle

The semantic center of the system must change from:
- `chunk -> embedding -> retrieve`

to:
- `utterance -> thought structure -> concept mapping -> retrieval planning -> multi-route retrieval -> bridge ranking -> answer -> thought update`

## 5. New core entities

Implement these as first-class schema models and persistent stores.

### 5.1 Claim
Represents a structured proposition extracted from a user utterance, document passage, or image summary.

Suggested fields:
- `claim_id: str`
- `project_id: str`
- `source_kind: Literal["document_passage", "user_utterance", "image_summary"]`
- `source_id: str`
- `core_claim: str`
- `critique_target: list[str]`
- `value_axes: list[str]`
- `materiality_axes: list[str]`
- `labor_time_axes: list[str]`
- `abstract_patterns: list[str]`
- `theory_hints: list[str]`
- `confidence: float`
- `created_at: datetime`
- optional embedding reference / vector id

### 5.2 ThoughtUnit
Represents a stable or emerging user-level thought tracked across conversations.

Suggested fields:
- `thought_id: str`
- `project_id: str`
- `title: str`
- `summary: str`
- `core_claim_ids: list[str]`
- `value_axes: list[str]`
- `recurring_patterns: list[str]`
- `supporting_claim_ids: list[str]`
- `status: Literal["emerging", "stable", "contested"]`
- `last_updated_at: datetime`

### 5.3 ConceptMapping
Maps a claim/passage/thought to controlled concepts.

Suggested fields:
- `mapping_id: str`
- `project_id: str`
- `owner_kind: Literal["claim", "passage", "thought_unit"]`
- `owner_id: str`
- `vocab_source: Literal["aat", "internal"]`
- `concept_id: str`
- `preferred_label: str`
- `alt_labels: list[str]`
- `broader_terms: list[str]`
- `related_terms: list[str]`
- `confidence: float`

### 5.4 Bridge
Represents a cross-domain or structural relation between claims.

Suggested fields:
- `bridge_id: str`
- `project_id: str`
- `source_claim_id: str`
- `target_claim_id: str`
- `bridge_type: Literal["analogy", "tradition_link", "value_structure_match"]`
- `reason_summary: str`
- `shared_patterns: list[str]`
- `confidence: float`
- `created_at: datetime`

### 5.5 RetrievalPlan
Represents a plan for fast/deep retrieval.

Suggested fields:
- `plan_id: str`
- `project_id: str`
- `query_mode: Literal["fast", "deep"]`
- `why_deep: str | None`
- `thought_summary: str`
- `concept_queries: list[str]`
- `supporting_paths: list[str]`
- `analogy_paths: list[str]`
- `max_passages_per_path: int`
- `created_at: datetime`

## 6. New module layout

Create a new `core/` package and gradually route the system through it.

Suggested new files:
- `core/schemas.py`
- `core/thought_extractor.py`
- `core/concept_mapper.py`
- `core/query_classifier.py`
- `core/retrieval_planner.py`
- `core/retriever.py`
- `core/bridge_ranker.py`
- `core/thought_evolution.py`
- `core/stores/document_store.py`
- `core/stores/claim_store.py`
- `core/stores/thought_store.py`
- `core/stores/concept_store.py`
- `core/stores/bridge_store.py`

Legacy modules may remain temporarily, but should be treated as compatibility layers.

## 7. Mapping from current modules to v2 responsibilities

### Freeze / replace gradually
- `memory/store.py` -> replace with `core/stores/thought_store.py` and `claim_store.py`
- `memory/thought_tracker.py` -> split into `thought_extractor.py` and `thought_evolution.py`
- `memory/vectorstore.py` -> keep only if useful as a retrieval index abstraction
- `rag/pipeline.py` -> replace with `query_classifier.py`, `retrieval_planner.py`, `retriever.py`, `bridge_ranker.py`
- `rag/reranker.py` -> upgrade into multi-signal ranking logic
- `rag/knowledge_graph.py` and `rag/insight_profile.py` -> adapt into concept/bridge/profile layer or deprecate carefully

### Keep and adapt
- `projects/manager.py`
- `backend/main.py`
- ingestion job orchestration in `rag/`
- image router / gallery shell

## 8. Runtime paths

### 8.1 Fast path
For ordinary conversation:
1. user utterance
2. lightweight thought extraction
3. concept mapping
4. single-pass retrieval
5. lightweight rerank
6. response
7. optional background thought merge

### 8.2 Deep path
For theory mapping / analogical retrieval:
1. user utterance
2. full thought extraction
3. concept normalization
4. retrieval planner
5. multi-route retrieval
6. bridge ranking
7. answer synthesis
8. thought evolution update

### 8.3 Offline path
For ingestion/background work:
1. source ingestion
2. passage chunking
3. claim extraction
4. concept tagging
5. theory tagging
6. bridge candidate generation
7. index build

## 9. Ingestion requirements

When a document is ingested, do not only produce chunks and embeddings.
The new pipeline must also produce:
- passages
- claims per passage where applicable
- concept mappings
- theory tags / hints
- optional bridge candidates

Create annotator modules under something like:
- `rag/annotators/claim_annotator.py`
- `rag/annotators/concept_annotator.py`
- `rag/annotators/theory_annotator.py`
- `rag/annotators/bridge_candidate_annotator.py`

## 10. Retrieval requirements

Retrieval must combine multiple signals, not raw semantic similarity alone.

Target ranking signals:
- semantic similarity
- claim similarity
- concept overlap
- value-structure similarity
- cross-domain bridge fit

The system should support retrieval across:
- passages
- claims
- thought units
- existing bridges

## 11. Persistence strategy

Do not directly reuse old memory records as v2 truth.
Preserve raw source data, but regenerate interpretation layers.

Preserve if available:
- documents
- conversations
- images

Regenerate:
- memory records
- vector indices where necessary
- claim / concept / bridge layers

## 12. Implementation constraints

- Use Python 3.11+
- Prefer Pydantic models for schemas
- Keep changes modular and incremental
- Avoid a giant all-in-one prompt architecture
- Design for debuggability: each stage should have inspectable inputs/outputs
- Keep legacy interfaces working where practical

## 13. Deliverables expected from Codex

### Phase 1 deliverables
1. Create `core/` package and schema models
2. Implement stores for `Claim`, `ThoughtUnit`, `ConceptMapping`, `Bridge`
3. Add a basic `ThoughtExtractor` interface with deterministic typed output
4. Add a basic `RetrievalPlanner` interface and plan schema
5. Refactor one existing chat path to call the new core behind a compatibility layer

### Phase 2 deliverables
1. Add ingestion annotators
2. Add bridge ranker
3. Add migration/bootstrap scripts for rebuilding v2 interpretation data from raw project data

## 14. Important design intent

De-insight v2 should not merely find similar text.
It should translate creator language into thought structure and map that structure onto theory-relevant and cross-domain contexts.
