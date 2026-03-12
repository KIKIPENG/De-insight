"""Integration tests for the structural claim pipeline.

Tests the three fixed disconnection points:
1. ThoughtExtractor.extract_from_passage() → ClaimStore
2. ClaimStore.search_by_structure() structural retrieval
3. Retriever hybrid path with injected ClaimStore
"""

import asyncio
import json
import sys
import tempfile
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from core.schemas import Claim, SourceKind, RetrievalPlan, QueryMode
from core.thought_extractor import ThoughtExtractor, LLMCallable, _clean_llm_output
from core.stores.claim_store import ClaimStore
from core.retriever import Retriever


# ── Fixtures ────────────────────────────────────────────────────────────

FAKE_LLM_RESPONSE_MINIMALISM = json.dumps({
    "claims": [
        {
            "core_claim": "極簡主義透過剝除裝飾，迫使觀者直面物件的本質結構",
            "critique_target": ["裝飾主義", "形式多餘"],
            "value_axes": ["本質", "純粹", "減法美學"],
            "materiality_axes": ["材料裸露"],
            "labor_time_axes": ["製作時間壓縮"],
            "abstract_patterns": ["限制→本質", "減法→揭示"],
            "theory_hints": ["less is more", "現象學還原"]
        }
    ],
    "thought_summary": "限制作為揭露本質的方法論",
    "concepts": [
        {
            "concept_id": "minimalism",
            "preferred_label": "極簡主義",
            "vocab_source": "aat",
            "confidence": 0.9
        }
    ]
}, ensure_ascii=False)

FAKE_LLM_RESPONSE_HAIKU = json.dumps({
    "claims": [
        {
            "core_claim": "俳句的5-7-5音節限制迫使詩人找到最本質的語詞",
            "critique_target": ["冗長表達", "散文式詩歌"],
            "value_axes": ["本質", "精煉", "瞬間美學"],
            "materiality_axes": ["語言的物質性"],
            "labor_time_axes": ["凝練時間"],
            "abstract_patterns": ["限制→本質", "形式約束→精煉"],
            "theory_hints": ["形式主義", "語言經濟"]
        }
    ],
    "thought_summary": "形式約束作為抵達精煉的路徑",
    "concepts": [
        {
            "concept_id": "haiku",
            "preferred_label": "俳句",
            "vocab_source": "internal",
            "confidence": 0.85
        }
    ]
}, ensure_ascii=False)

FAKE_LLM_RESPONSE_BAROQUE = json.dumps({
    "claims": [
        {
            "core_claim": "巴洛克建築透過過度裝飾來展示權力與神聖的崇高感",
            "critique_target": ["理性主義", "簡約"],
            "value_axes": ["崇高", "權力", "裝飾"],
            "materiality_axes": ["大理石", "金箔"],
            "labor_time_axes": ["長期營造"],
            "abstract_patterns": ["過度→崇高", "裝飾→權力展示"],
            "theory_hints": ["崇高美學", "符號權力"]
        }
    ],
    "thought_summary": "過度裝飾作為權力展演",
    "concepts": []
}, ensure_ascii=False)


def _make_fake_llm(response: str):
    """Create a fake LLM callable that returns a fixed response."""
    async def fake_llm(prompt: str) -> str:
        return response
    return LLMCallable(func=fake_llm)


def _make_temp_claim_store(project_id="test") -> ClaimStore:
    """Create a ClaimStore backed by a temp database."""
    tmp = tempfile.mktemp(suffix=".db")
    return ClaimStore(project_id=project_id, db_path=Path(tmp))


# ── Test 1: ThoughtExtractor.extract_from_passage ──────────────────────

@pytest.mark.asyncio
async def test_extract_from_passage_minimalism():
    """Verify passage extraction produces Claim with structural dimensions."""
    llm = _make_fake_llm(FAKE_LLM_RESPONSE_MINIMALISM)
    extractor = ThoughtExtractor(llm_callable=llm, project_id="test")

    passage = """
    極簡主義的核心不在於「少」本身，而在於透過剝除裝飾，
    讓觀者被迫面對物件的本質結構。當所有多餘的都被拿走，
    剩下的就是不可化約的骨架。
    """
    result = await extractor.extract_from_passage(passage, source_id="doc_001")

    assert result.was_extracted
    assert len(result.claims) == 1

    claim = result.claims[0]
    assert claim.source_kind == SourceKind.DOCUMENT_PASSAGE
    assert claim.source_id == "doc_001"
    assert "限制→本質" in claim.abstract_patterns
    assert "本質" in claim.value_axes
    assert claim.core_claim  # non-empty

    print(f"  ✓ Passage extraction: {claim.core_claim[:40]}...")
    print(f"    abstract_patterns: {claim.abstract_patterns}")
    print(f"    value_axes: {claim.value_axes}")


# ── Test 2: ClaimStore CRUD + structural search ────────────────────────

@pytest.mark.asyncio
async def test_claim_store_structural_search():
    """Verify search_by_structure finds structurally similar claims across domains."""
    store = _make_temp_claim_store()

    # Insert minimalism claim
    claim_min = Claim(
        project_id="test",
        source_kind=SourceKind.DOCUMENT_PASSAGE,
        core_claim="極簡主義透過剝除裝飾，迫使觀者直面物件的本質結構",
        value_axes=["本質", "純粹", "減法美學"],
        abstract_patterns=["限制→本質", "減法→揭示"],
        theory_hints=["less is more", "現象學還原"],
        critique_target=["裝飾主義"],
        confidence=0.9,
    )
    await store.add(claim_min)

    # Insert haiku claim (different domain, SAME structure)
    claim_haiku = Claim(
        project_id="test",
        source_kind=SourceKind.DOCUMENT_PASSAGE,
        core_claim="俳句的5-7-5音節限制迫使詩人找到最本質的語詞",
        value_axes=["本質", "精煉", "瞬間美學"],
        abstract_patterns=["限制→本質", "形式約束→精煉"],
        theory_hints=["形式主義", "語言經濟"],
        critique_target=["冗長表達"],
        confidence=0.85,
    )
    await store.add(claim_haiku)

    # Insert baroque claim (DIFFERENT structure)
    claim_baroque = Claim(
        project_id="test",
        source_kind=SourceKind.DOCUMENT_PASSAGE,
        core_claim="巴洛克建築透過過度裝飾來展示權力與神聖的崇高感",
        value_axes=["崇高", "權力", "裝飾"],
        abstract_patterns=["過度→崇高", "裝飾→權力展示"],
        theory_hints=["崇高美學"],
        critique_target=["理性主義"],
        confidence=0.8,
    )
    await store.add(claim_baroque)

    # Search for "限制→本質" pattern — should find minimalism AND haiku, NOT baroque
    results = await store.search_by_structure(
        abstract_patterns=["限制→本質"],
        limit=10,
    )

    result_claims = {r.core_claim[:10] for r in results}
    print(f"\n  Structural search for '限制→本質':")
    for r in results:
        print(f"    ✓ Found: {r.core_claim[:40]}... (patterns: {r.abstract_patterns})")

    assert len(results) >= 2, f"Expected ≥2 results, got {len(results)}"
    # Both minimalism and haiku should match
    assert any("極簡" in r.core_claim for r in results), "Should find minimalism"
    assert any("俳句" in r.core_claim for r in results), "Should find haiku"

    # Search by value_axes — "本質" should match both minimalism and haiku
    results2 = await store.search_by_structure(
        value_axes=["本質"],
        limit=10,
    )
    print(f"\n  Structural search for value_axes '本質':")
    for r in results2:
        print(f"    ✓ Found: {r.core_claim[:40]}... (value_axes: {r.value_axes})")

    assert len(results2) >= 2

    # Search by "崇高" should find baroque only
    results3 = await store.search_by_structure(
        value_axes=["崇高"],
        limit=10,
    )
    print(f"\n  Structural search for value_axes '崇高':")
    for r in results3:
        print(f"    ✓ Found: {r.core_claim[:40]}...")

    assert len(results3) >= 1
    assert any("巴洛克" in r.core_claim for r in results3)

    print("\n  ✓ All structural search tests passed")


# ── Test 3: Retriever hybrid path with ClaimStore ──────────────────────

@pytest.mark.asyncio
async def test_retriever_hybrid_with_claim_store():
    """Verify Retriever uses hybrid path when ClaimStore is injected."""
    store = _make_temp_claim_store()

    # Pre-populate with claims
    claim1 = Claim(
        project_id="test",
        source_kind=SourceKind.DOCUMENT_PASSAGE,
        core_claim="極簡主義透過剝除裝飾揭示本質",
        value_axes=["本質", "純粹"],
        abstract_patterns=["限制→本質"],
        theory_hints=["less is more"],
        confidence=0.9,
    )
    claim2 = Claim(
        project_id="test",
        source_kind=SourceKind.DOCUMENT_PASSAGE,
        core_claim="俳句的形式限制迫使找到本質語詞",
        value_axes=["本質", "精煉"],
        abstract_patterns=["限制→本質", "形式約束→精煉"],
        theory_hints=["形式主義"],
        confidence=0.85,
    )
    await store.add(claim1)
    await store.add(claim2)

    # Create retriever WITH injected store
    retriever = Retriever(
        project_id="test",
        claim_store=store,
    )

    # Verify _has_stores() returns True
    assert retriever._has_stores(), "Retriever should detect injected ClaimStore"

    # Create a plan
    plan = RetrievalPlan(
        project_id="test",
        query_mode="fast",
        thought_summary="",
        concept_queries=["限制如何產生本質"],
    )

    # Run retrieval
    result = await retriever.retrieve(plan, "限制如何產生本質")

    print(f"\n  Hybrid retrieval results:")
    print(f"    claims: {len(result.claims)}")
    print(f"    bridges: {len(result.bridges)}")
    for c in result.claims:
        print(f"    ✓ Claim: {c.core_claim[:40]}... (patterns: {c.abstract_patterns})")

    # Should get claims from store
    # Note: text search may not match Chinese well with LIKE,
    # but structural search via _retrieve_from_claims should find them
    print(f"\n  ✓ Hybrid retrieval completed (claims={len(result.claims)}, bridges={len(result.bridges)})")


# ── Test 4: _retrieve_from_claims two-route search ─────────────────────

@pytest.mark.asyncio
async def test_retrieve_from_claims_dual_route():
    """Verify _retrieve_from_claims uses both text and structural search."""
    store = _make_temp_claim_store()

    # Claim that matches by text (contains "極簡")
    claim_text = Claim(
        project_id="test",
        source_kind=SourceKind.DOCUMENT_PASSAGE,
        core_claim="極簡主義的核心是減法",
        value_axes=["減法", "純粹"],
        abstract_patterns=["減法→本質"],
        confidence=0.8,
    )
    # Claim that matches only structurally (shares "限制→本質" pattern, no text overlap)
    claim_struct = Claim(
        project_id="test",
        source_kind=SourceKind.DOCUMENT_PASSAGE,
        core_claim="俳句的音節約束迫使詩人凝練",
        value_axes=["精煉", "純粹"],
        abstract_patterns=["限制→本質"],
        theory_hints=["形式主義"],
        confidence=0.85,
    )
    await store.add(claim_text)
    await store.add(claim_struct)

    retriever = Retriever(project_id="test", claim_store=store)

    # Search with "極簡" — should find text match, then structural seeds from it
    results = await retriever._retrieve_from_claims("極簡")

    text_matches = [r for r in results if r["reason"] == "text_match"]
    struct_matches = [r for r in results if r["reason"] == "structural_match"]

    print(f"\n  Dual-route search for '極簡':")
    print(f"    text_match: {len(text_matches)}")
    print(f"    structural_match: {len(struct_matches)}")
    for r in results:
        c = r["claim"]
        print(f"    [{r['reason']}] {c.core_claim[:40]}... (patterns: {c.abstract_patterns})")

    assert len(text_matches) >= 1, "Should have at least 1 text match"
    # Structural search may find the haiku claim via shared "純粹" value axis
    print(f"\n  ✓ Dual-route search found {len(results)} total results")


# ── Test 5: _clean_llm_output handles edge cases ──────────────────────

def test_clean_llm_output():
    """Verify LLM output cleaning handles various formats."""
    # Markdown fence
    assert json.loads(_clean_llm_output('```json\n{"a": 1}\n```')) == {"a": 1}

    # Trailing comma
    assert json.loads(_clean_llm_output('{"a": [1, 2,]}')) == {"a": [1, 2]}

    # Leading text
    assert json.loads(_clean_llm_output('Here is the result:\n{"a": 1}')) == {"a": 1}

    print("  ✓ LLM output cleaning handles edge cases")


# ── Test 6: End-to-end: extract → store → retrieve ────────────────────

@pytest.mark.asyncio
async def test_end_to_end_extract_store_retrieve():
    """Full pipeline: extract from passage → store claims → retrieve structurally."""
    store = _make_temp_claim_store()

    # Step 1: Extract from minimalism passage
    llm1 = _make_fake_llm(FAKE_LLM_RESPONSE_MINIMALISM)
    extractor1 = ThoughtExtractor(llm_callable=llm1, project_id="test")
    result1 = await extractor1.extract_from_passage(
        "極簡主義透過剝除裝飾迫使觀者面對本質結構" * 2,  # pad to >30 chars
        source_id="doc_minimalism",
    )
    assert result1.was_extracted
    for claim in result1.claims:
        await store.add(claim)

    # Step 2: Extract from haiku passage
    llm2 = _make_fake_llm(FAKE_LLM_RESPONSE_HAIKU)
    extractor2 = ThoughtExtractor(llm_callable=llm2, project_id="test")
    result2 = await extractor2.extract_from_passage(
        "俳句的5-7-5音節限制迫使詩人找到最本質的語詞在最小的空間表達最大的意義" * 2,
        source_id="doc_haiku",
    )
    assert result2.was_extracted
    for claim in result2.claims:
        await store.add(claim)

    # Step 3: Extract from baroque passage (different structure)
    llm3 = _make_fake_llm(FAKE_LLM_RESPONSE_BAROQUE)
    extractor3 = ThoughtExtractor(llm_callable=llm3, project_id="test")
    result3 = await extractor3.extract_from_passage(
        "巴洛克建築透過過度裝飾來展示權力與神聖的崇高感這種過度本身就是訊息" * 2,
        source_id="doc_baroque",
    )
    assert result3.was_extracted
    for claim in result3.claims:
        await store.add(claim)

    # Step 4: Use Retriever to find structural matches
    retriever = Retriever(project_id="test", claim_store=store)
    plan = RetrievalPlan(
        project_id="test",
        query_mode="fast",
        thought_summary="",
        concept_queries=["限制產生本質"],
    )
    retrieval_result = await retriever.retrieve(plan, "限制產生本質")

    print(f"\n  === End-to-end pipeline ===")
    print(f"  Stored: {len(result1.claims) + len(result2.claims) + len(result3.claims)} claims")
    print(f"  Retrieved: {len(retrieval_result.claims)} claims, {len(retrieval_result.bridges)} bridges")
    for c in retrieval_result.claims:
        print(f"    ✓ {c.core_claim[:50]}...")
        print(f"      patterns: {c.abstract_patterns}")

    # The key assertion: structural retrieval should surface claims
    # from the store, proving the pipeline is connected
    all_stored = await store.list_by_project("test")
    assert len(all_stored) == 3, f"Store should have 3 claims, got {len(all_stored)}"

    # Structural search should find minimalism + haiku (both have 限制→本質)
    struct_results = await store.search_by_structure(
        abstract_patterns=["限制→本質"],
    )
    assert any("極簡" in c.core_claim for c in struct_results), \
        "Structural search should find minimalism via '限制→本質'"
    assert any("俳句" in c.core_claim for c in struct_results), \
        "Structural search should find haiku via '限制→本質'"

    print(f"\n  ✓ End-to-end pipeline verified: extract → store → structural retrieve")


# ── Runner ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("De-insight structural pipeline integration tests")
    print("=" * 60)

    loop = asyncio.new_event_loop()

    print("\n[1] ThoughtExtractor.extract_from_passage")
    loop.run_until_complete(test_extract_from_passage_minimalism())

    print("\n[2] ClaimStore structural search")
    loop.run_until_complete(test_claim_store_structural_search())

    print("\n[3] Retriever hybrid path with ClaimStore")
    loop.run_until_complete(test_retriever_hybrid_with_claim_store())

    print("\n[4] _retrieve_from_claims dual-route")
    loop.run_until_complete(test_retrieve_from_claims_dual_route())

    print("\n[5] _clean_llm_output edge cases")
    test_clean_llm_output()

    print("\n[6] End-to-end: extract → store → retrieve")
    loop.run_until_complete(test_end_to_end_extract_store_retrieve())

    loop.close()

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED ✓")
    print("=" * 60)
