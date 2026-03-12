# De-insight v2 Phase 2 Test and Debug Specification

## 1. Purpose

This document defines the minimum testing and debugging requirements that must be completed before moving De-insight v2 from Phase 1 core scaffolding into Phase 2 semantic retrieval work.

Phase 1 proved that the new core architecture can run. Phase 2 will start changing how the product actually interprets creative language. Because of that, the main risk is no longer import errors or schema wiring. The main risk is semantic drift, unstable extraction, silent fallback, and retrieval behavior that looks valid structurally but is wrong conceptually.

This spec focuses on four areas:

- semantic golden-set evaluation
- extractor robustness and malformed output handling
- end-to-end pipeline integration
- debug traces and observability

## 2. Current assessment

### What is already covered
The current test suite appears to cover:
- schema validation
- module imports
- fast vs deep query classification
- retrieval planning
- compatibility-layer behavior

This is a good Phase 1 baseline.

### What is not yet strong enough
The current setup is not yet strong enough in these areas:
- extractor stability under real LLM variability
- semantic correctness of extracted claims and value axes
- end-to-end behavior across feature flags
- traceability of decisions made by the core pipeline
- ability to compare legacy outputs vs core outputs on the same input

## 3. Required test categories before major Phase 2 expansion

### 3.1 Golden-set semantic tests

#### Goal
Ensure that the new core pipeline extracts the right thought structure from real creator-language examples.

#### Requirement
Create a manually curated golden set with at least 20 cases, ideally 30 to 40.

Each case must contain:
- `case_id`
- `user_message`
- `expected_mode` (`fast` or `deep`)
- `expected_core_claim`
- `expected_critique_targets`
- `expected_value_axes`
- `expected_abstract_patterns`
- `expected_theory_hints`
- `notes`

#### Coverage requirements
The golden set must include examples from:
- materiality
- craft and labor
- temporality / time investment
- design ethics
- form vs function
- exhibition narrative
- image-text relationship
- tacit preference statements
- ambiguous artistic intuition
- cross-domain theory mapping

#### Example golden case
```json
{
  "case_id": "golden_001",
  "user_message": "我覺得書籍設計師在設計書籍時，應該要考量到書籍的材質，不應該為了特定目的去把書籍本該有的材質性忽略，或是在設計時沒有用心，用時間去編排。",
  "expected_mode": "deep",
  "expected_core_claim": "book design should respect the material nature of the book instead of subordinating it to external purposes",
  "expected_critique_targets": [
    "instrumentalized design",
    "neglect of materiality",
    "careless design labor"
  ],
  "expected_value_axes": [
    "material honesty",
    "craftsmanship",
    "temporal labor"
  ],
  "expected_abstract_patterns": [
    "truth to materials",
    "anti-instrumentalism",
    "ethics of making"
  ],
  "expected_theory_hints": [
    "Ruskin - Lamp of Truth",
    "Arts and Crafts"
  ],
  "notes": "Core cross-domain analogy case"
}
```

#### Assertions
For each golden case, tests should verify:
- mode classification is correct
- extracted core claim is non-empty and semantically aligned
- critique targets overlap expected targets
- value axes overlap expected values
- abstract patterns overlap expected patterns
- theory hints contain at least one relevant path for deep cases

These do not need exact string equality. Use normalized overlap scoring.

### 3.2 Extractor robustness tests

#### Goal
Ensure the thought extractor does not become the single point of failure.

#### Requirement
Add at least 10 malformed-output tests simulating bad LLM responses.

#### Must-cover malformed cases
- non-JSON text
- JSON missing required keys
- wrong field types (`string` instead of `list`)
- `null` values where lists are expected
- extra unknown fields
- mixed-language list items
- truncated JSON
- empty response
- invalid enum values
- nested object shape drift

#### Expected behavior
The extractor must:
- attempt parse repair when possible
- normalize or coerce minor type mismatches
- fill safe defaults for missing optional fields
- emit warnings in trace logs
- fail gracefully when repair is impossible
- never crash the full pipeline because of malformed extraction output

#### Assertions
For each malformed case, verify:
- whether repair was attempted
- whether the returned object still validates
- whether fallback behavior is triggered
- whether warning/error trace fields are populated

### 3.3 End-to-end pipeline tests

#### Goal
Verify that the new core pipeline behaves correctly from input to storage.

#### Requirement
Add at least 6 E2E tests.

#### Minimum required E2E scenarios

##### E2E 1: fast path
- input clearly belongs to fast path
- classifier returns `fast`
- planner creates a lightweight retrieval plan
- retriever returns a typed result
- pipeline writes expected claim/thought data

##### E2E 2: deep path
- input clearly belongs to deep path
- classifier returns `deep`
- planner creates multiple retrieval routes
- retriever executes delegated or hybrid retrieval
- pipeline stores outputs correctly

##### E2E 3: feature flag off
- core disabled
- request flows through legacy path only
- no core-only writes occur unexpectedly

##### E2E 4: feature flag on
- core enabled
- request flows through core path
- compat output shape remains valid for caller

##### E2E 5: extractor repair path
- malformed LLM output
- repair succeeds
- pipeline still returns valid result

##### E2E 6: extractor hard failure path
- malformed output cannot be repaired
- pipeline falls back safely
- trace records reason for fallback

### 3.4 Store integration tests

#### Goal
Ensure the new stores are not only writable, but usable as future retrieval substrate.

#### Requirement
Add tests for:
- claim insert and fetch
- thought unit insert and fetch
- concept mapping insert and fetch
- bridge insert and fetch
- project isolation
- duplicate-safe behavior where relevant

#### Additional required checks
- records written under one project do not appear in another
- timestamps and IDs are generated correctly
- store read APIs return typed objects
- invalid writes are rejected safely

### 3.5 Legacy/core comparison tests

#### Goal
Make semantic drift visible while the retriever still delegates to legacy systems.

#### Requirement
Add at least 5 comparison fixtures where the same user input is run through:
- legacy pipeline
- core pipeline with feature flag enabled

#### Record for each fixture
- classifier decision
- extracted thought summary
- retrieval plan
- legacy result summary
- core result summary
- notes on divergence

This can be stored as snapshot-like diagnostic data rather than strict pass/fail initially.

## 4. Required debug and observability features

### 4.1 Pipeline trace object

Every core pipeline execution should produce a structured trace object.

#### Required fields
- `trace_id`
- `project_id`
- `user_message_preview`
- `core_enabled`
- `classification_mode`
- `extractor_success`
- `extractor_repair_attempted`
- `extractor_warnings`
- `core_claim_summary`
- `value_axes`
- `abstract_patterns`
- `theory_hints`
- `retrieval_plan_summary`
- `legacy_delegate_used`
- `store_writes`
- `fallback_triggered`
- `fallback_reason`
- `errors`
- `timings`

#### Format
Use a JSON-serializable dict or Pydantic model.

#### Purpose
This trace is mandatory for debugging semantic failures.

### 4.2 Extractor-specific telemetry

The extractor should report:
- raw output length
- parse success or failure
- repair success or failure
- normalized fields
- dropped fields
- schema validation errors

This data can be included in pipeline trace or a nested extractor trace object.

### 4.3 Planner decision logs

The retrieval planner should log:
- why the query was classified as fast or deep
- which patterns triggered the decision
- which retrieval routes were generated
- why certain routes were omitted

This is required because deep-path errors will often be planner errors, not retriever errors.

### 4.4 Store write audit log

Whenever the pipeline writes to:
- claim store
- thought store
- concept store
- bridge store

it should record:
- store name
- record type
- project_id
- record ID
- success/failure
- timing
- write skipped or not

This can stay lightweight, but it must exist.

## 5. Minimum implementation targets before Phase 2 retrieval deepening

Before building concept-driven retrieval or bridge ranking in earnest, the codebase should satisfy the following minimum bar:

### Tests
- 20+ golden semantic cases
- 10+ malformed extractor cases
- 6+ end-to-end pipeline tests
- 5+ legacy/core comparison fixtures
- store integration coverage for all four stores

### Debug
- pipeline trace object implemented
- extractor repair telemetry implemented
- planner decision logging implemented
- store write audit logging implemented

### CI expectation
All of the above should run in automated test flow where possible. Golden semantic comparisons may use overlap thresholds instead of exact match.

## 6. Suggested file layout

```text
core/
  tests/
    test_core.py
    test_extractor_robustness.py
    test_pipeline_e2e.py
    test_store_integration.py
    fixtures/
      golden_cases.json
      malformed_outputs.json
      legacy_core_comparison_cases.json

  debug/
    trace.py
    telemetry.py
```

## 7. Implementation notes

- Do not block progress waiting for perfect evaluation infrastructure.
- Prefer overlap-based semantic assertions over brittle exact strings.
- Keep feature-flag behavior explicit and observable.
- Preserve backward compatibility, but do not hide fallback behavior.
- All failures should be inspectable through trace output.

## 8. Definition of done for test/debug hardening

Test/debug hardening is considered complete when:
1. the new core pipeline can be run repeatedly on curated creator-language examples,
2. malformed extractor outputs no longer break the pipeline silently,
3. fast/deep decisions are explainable,
4. legacy/core divergence can be inspected,
5. store writes and fallback behavior are visible in traces.

Until then, the system is safe for continued development but not yet safe for aggressive semantic refactors.
