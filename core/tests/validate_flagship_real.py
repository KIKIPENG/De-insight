"""Real-semantic validation runner for Milestone F4.

This script tests production anchor quality checks in the pipeline.
"""

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.schemas import Claim, QueryMode, RetrievalPlan
from core.stores import ClaimStore, ThoughtStore, ConceptStore
from core.thought_extractor import ThoughtExtractor, LLMCallable
from core.retriever import Retriever, assess_anchor_quality, enrich_thin_anchor
from core.bridge_ranker import BridgeRanker


def load_flagship_cases():
    fixture_path = Path(__file__).parent / "fixtures" / "flagship_validation_cases.json"
    with open(fixture_path) as f:
        data = json.load(f)
    return data["flagship_cases"]


def load_curated_candidates():
    fixture_path = Path(__file__).parent / "fixtures" / "curated_validation_candidates.json"
    with open(fixture_path) as f:
        data = json.load(f)
    return data["candidates"]


async def baseline_llm(prompt: str) -> str:
    """Baseline LLM - thin output."""
    if "結構化思維抽取" in prompt or "主張" in prompt:
        user_text = ""
        if "使用者發言" in prompt:
            start = prompt.find("使用者發言：") + 6
            user_text = prompt[start:80]
        
        if "書籍" in user_text or "材質" in user_text:
            return json.dumps({
                "claims": [{"core_claim": "book design material", "value_axes": ["material"], "theory_hints": ["Ruskin"]}],
                "thought_summary": "book design"
            })
        elif "展覽" in user_text or "動線" in user_text:
            return json.dumps({
                "claims": [{"core_claim": "exhibition design", "value_axes": ["design"], "theory_hints": []}],
                "thought_summary": "exhibition"
            })
        elif "痕跡" in user_text or "修補" in user_text:
            return json.dumps({
                "claims": [{"core_claim": "traces value", "value_axes": [], "theory_hints": ["wabi"]}],
                "thought_summary": "traces"
            })
    return json.dumps({"claims": [], "thought_summary": ""})


async def run_production_test(case: dict, tmp_path: Path):
    """Test with production anchor quality checks."""
    case_id = case["case_id"]
    user_message = case["user_message"]
    
    # Setup stores
    claim_store = ClaimStore(project_id=case_id, db_path=tmp_path / "c.db")
    
    # Add candidates
    for cand in load_curated_candidates():
        claim = Claim(
            project_id=case_id,
            core_claim=cand["core_claim"],
            value_axes=cand.get("value_axes", []),
            abstract_patterns=cand.get("abstract_patterns", []),
            critique_target=cand.get("critique_targets", []),
            theory_hints=cand.get("theory_hints", []),
        )
        await claim_store.add(claim)
    
    # Extract with baseline LLM (thin output)
    extractor = ThoughtExtractor(LLMCallable(func=baseline_llm), project_id=case_id)
    result = await extractor.extract(user_message)
    
    # Add extracted claims
    for c in result.claims:
        await claim_store.add(c)
    
    # Create plan with concept queries
    plan = RetrievalPlan(
        project_id=case_id,
        query_mode=QueryMode.DEEP,
        concept_queries=case["expected_thought_structure"]["theory_hints"][:2]
    )
    
    # Test retrieval with production path
    retriever = Retriever(project_id=case_id, claim_store=claim_store)
    retrieval = await retriever.retrieve(plan, "")
    
    return {
        "extracted": result.claims,
        "bridges": retrieval.bridges,
    }


async def run_direct_test(case: dict):
    """Direct test with production quality checks."""
    case_id = case["case_id"]
    
    # Get candidates
    candidates = []
    for cand in load_curated_candidates():
        claim = Claim(
            project_id=case_id,
            core_claim=cand["core_claim"],
            value_axes=cand.get("value_axes", []),
            abstract_patterns=cand.get("abstract_patterns", []),
            critique_target=cand.get("critique_targets", []),
            theory_hints=cand.get("theory_hints", []),
        )
        candidates.append(claim)
    
    # Baseline thin anchor (simulating poor LLM output)
    if "書籍" in case["user_message"] or "材質" in case["user_message"]:
        baseline_anchor = Claim(
            project_id=case_id,
            core_claim="book",
            value_axes=["material"],
            theory_hints=["Ruskin"],
        )
    elif "展覽" in case["user_message"] or "動線" in case["user_message"]:
        baseline_anchor = Claim(
            project_id=case_id,
            core_claim="exhibition",
            value_axes=["design"],
            theory_hints=[],
        )
    else:
        baseline_anchor = Claim(
            project_id=case_id,
            core_claim="traces",
            value_axes=[],
            theory_hints=["wabi"],
        )
    
    # Test quality assessment
    baseline_quality = assess_anchor_quality(baseline_anchor)
    
    # Create plan for enrichment
    plan = RetrievalPlan(
        project_id=case_id,
        query_mode=QueryMode.DEEP,
        concept_queries=case["expected_thought_structure"]["theory_hints"][:2]
    )
    
    # Test enrichment
    enriched_anchor = enrich_thin_anchor(baseline_anchor, plan)
    enriched_quality = assess_anchor_quality(enriched_anchor)
    
    # Test ranking
    ranker = BridgeRanker()
    baseline_results = ranker.rank_candidates(baseline_anchor, candidates)
    enriched_results = ranker.rank_candidates(enriched_anchor, candidates)
    
    return {
        "baseline": {
            "anchor": baseline_anchor,
            "quality": baseline_quality,
            "top_bridge": baseline_results[0].score if baseline_results else 0,
        },
        "enriched": {
            "anchor": enriched_anchor,
            "quality": enriched_quality,
            "top_bridge": enriched_results[0].score if enriched_results else 0,
        },
    }


async def main():
    print("=" * 70)
    print("Milestone F4: Production Anchor Quality Checks")
    print("=" * 70)
    
    cases = load_flagship_cases()
    print(f"Flagship cases: {len(cases)}")
    print()
    
    for case in cases:
        print(f"\n{'='*60}")
        print(f"Case: {case['case_id']} - {case['name']}")
        print(f"{'='*60}")
        
        result = await run_direct_test(case)
        
        # Baseline
        b = result["baseline"]
        print(f"\n--- BASELINE (Thin) ---")
        print(f"Anchor: {b['anchor'].core_claim}")
        print(f"Value axes: {b['anchor'].value_axes}")
        print(f"Theory hints: {b['anchor'].theory_hints}")
        print(f"Quality: {b['quality']['quality_score']}/25 ({b['quality']['percentage']:.1f}%)")
        print(f"Top bridge: {b['top_bridge']:.4f}")
        
        # Enriched
        e = result["enriched"]
        print(f"\n--- WITH PRODUCTION FALLBACK ---")
        print(f"Anchor: {e['anchor'].core_claim}")
        print(f"Value axes: {e['anchor'].value_axes}")
        print(f"Theory hints: {e['anchor'].theory_hints}")
        print(f"Quality: {e['quality']['quality_score']}/25 ({e['quality']['percentage']:.1f}%)")
        print(f"Top bridge: {e['top_bridge']:.4f}")
        
        # Improvement
        quality_imp = e['quality']['quality_score'] - b['quality']['quality_score']
        bridge_imp = e['top_bridge'] - b['top_bridge']
        print(f"\n--- IMPROVEMENT ---")
        print(f"Quality: +{quality_imp}")
        print(f"Bridge: +{bridge_imp:.4f}")
        
        # Show bridges
        print(f"\n--- TOP BRIDGES (Enriched) ---")
        ranker = BridgeRanker()
        results = ranker.rank_candidates(e['anchor'], [])
        # Need actual candidates for ranking
        candidates = []
        for cand in load_curated_candidates():
            c = Claim(
                project_id='test',
                core_claim=cand["core_claim"],
                value_axes=cand.get("value_axes", []),
                theory_hints=cand.get("theory_hints", []),
            )
            candidates.append(c)
        results = ranker.rank_candidates(e['anchor'], candidates)
        for i, r in enumerate(results[:3]):
            print(f"  {i+1}. {r.score:.4f}: {r.reason}")
    
    print("\n" + "=" * 70)
    print("PRODUCTION QUALITY CHECKS ACTIVE")
    print("=" * 70)
    print("""
- Thin anchor detection: quality < 10 or < 2 value_axes or < 1 theory_hints
- Fallback enrichment: uses concept_queries from plan
- Quality assessment: 5 dimensions, 0-5 each, 25 max
""")


if __name__ == "__main__":
    asyncio.run(main())
