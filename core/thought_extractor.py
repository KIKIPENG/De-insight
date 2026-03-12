"""Thought extraction module for De-insight v2 Core.

This module extracts structured thought from user utterances and outputs
typed Claim, ThoughtUnit, and ConceptMapping objects rather than freeform text.

The extractor uses an LLM to analyze user input and produce structured JSON
that is then parsed into Pydantic models.

References:
- Tech spec: deinsight_v2_codex_technical_spec.md
- Existing code: memory/thought_tracker.py (for LLM prompt patterns)
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from core.schemas import (
    Claim,
    ConceptMapping,
    ExtractionResult,
    OwnerKind,
    SourceKind,
    ThoughtStatus,
    ThoughtUnit,
    VocabSource,
)


# Prompt for extracting structured claims from document passages
EXTRACT_PASSAGE_CLAIMS_PROMPT = """\
你是結構化論述抽取器。

任務：從文件段落中抽取結構化的「主張」(claims)。
注意：這不是使用者發言，而是匯入的文件內容（論文、文章、書籍段落等）。

以下是你要判斷的維度：

## 主張維度

1. **critique_target**: 這段文字批判或挑戰的對象/預設/論述
2. **value_axes**: 這段文字涉及的美學/倫理/社會價值維度
3. **materiality_axes**: 這段文字涉及的物質/技術/材料面向
4. **labor_time_axes**: 這段文字涉及的勞動/時間/過程面向
5. **abstract_patterns**: 這段文字的論述結構模式（例如：限制→本質、對比→張力、類比→跨域、因果→機制、辯證→超越）
6. **theory_hints**: 這段文字可能連接的理論方向

## 抽取規則

- 一段文字可能包含 0-3 個主張
- 每個主張必須有 core_claim（核心命題，1-2 句話）
- abstract_patterns 欄位特別重要——用短語描述論證的骨架結構，而不是內容本身
  例如：「限制產生本質」「對比揭示盲點」「跨域類比建立新框架」
- 如果沒有值得抽取的主張，回傳空陣列

## 輸出格式

只回傳 JSON，不要加任何說明：

{{
  "claims": [
    {{
      "core_claim": "主張的核心命題（1-2句話）",
      "critique_target": ["批判的對象1"],
      "value_axes": ["價值維度1"],
      "materiality_axes": ["物質維度1"],
      "labor_time_axes": ["勞動/時間維度1"],
      "abstract_patterns": ["論述結構模式1"],
      "theory_hints": ["理論方向1"]
    }}
  ],
  "thought_summary": "這些主張形成的整體論述方向",
  "concepts": [
    {{
      "concept_id": "概念ID",
      "preferred_label": "人類可讀的標籤",
      "vocab_source": "aat 或 internal",
      "confidence": 0.0-1.0
    }}
  ]
}}

---

文件段落：
{passage_text}

只回傳 JSON 格式的結果。
"""


# Prompt for extracting structured claims from user utterances
EXTRACT_CLAIMS_PROMPT = """\
你是結構化思維抽取器。

任務：從使用者的發言中抽取結構化的「主張」(claims)，而不是一般的記憶條目。

以下是你要判斷的維度：

## 主張維度

1. **critique_target**: 這個主張批判或挑戰的對象/預設/論述
2. **value_axes**: 這個主張涉及的美學/倫理/社會價值維度
3. **materiality_axes**: 這個主張涉及的物質/技術/材料面向
4. **labor_time_axes**: 這個主張涉及的勞動/時間/過程面向
5. **abstract_patterns**: 這個主張涉及的結構性模式（形式/功能/系統/符號等）
6. **theory_hints**: 這個主張可能連接的理論方向

## 抽取規則

- 只抽取使用者說的話，不抽 AI 回應
- 一段發言可能包含 0-3 個主張
- 每個主張必須有 core_claim（核心命題，1-2 句話）
- 其他維度欄位可以為空陣列，但必須回傳結構
- 如果沒有值得抽取的主張，回傳空陣列

## 輸出格式

只回傳 JSON，不要加任何說明：

{{
  "claims": [
    {{
      "core_claim": "主張的核心命題（1-2句話）",
      "critique_target": ["批判的對象1", "批判的對象2"],
      "value_axes": ["價值維度1", "價值維度2"],
      "materiality_axes": ["物質維度1"],
      "labor_time_axes": ["勞動/時間維度1"],
      "abstract_patterns": ["結構模式1"],
      "theory_hints": ["理論方向1"]
    }}
  ],
  "thought_summary": "如果這些主張形成一個整體思考，用一句話概括這個思考方向",
  "concepts": [
    {{
      "concept_id": "概念ID",
      "preferred_label": "人類可讀的標籤",
      "vocab_source": "aat 或 internal",
      "confidence": 0.0-1.0
    }}
  ]
}}

---

使用者發言：
{user_text}

只回傳 JSON 格式的結果。
"""


def _clean_llm_output(text: str) -> str:
    """Clean LLM output to extract JSON.

    Handles common formatting issues:
    - Markdown code fences
    - Trailing commas
    - Leading/trailing whitespace
    - Explanatory text before JSON
    """
    text = text.strip()

    # Remove markdown fences
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()

    # Find first [ or { and last } or ]
    start_idx = -1
    for i, c in enumerate(text):
        if c in "[{":
            start_idx = i
            break
    if start_idx >= 0:
        text = text[start_idx:]

    # Remove trailing comma
    text = re.sub(r',\s*([\]}])', r'\1', text)

    return text.strip()


@dataclass
class LLMCallable:
    """Interface for LLM calls.

    This is a simple interface that wraps the actual LLM call.
    The caller provides a callable that takes a prompt and returns text.
    """

    func: Any  # Callable[[str], Awaitable[str]]


class ThoughtExtractor:
    """Extracts structured thought from user utterances.

    This is the v2 replacement for memory/thought_tracker.extract_memories().
    Instead of returning simple memory records, it produces typed Claim,
    ThoughtUnit, and ConceptMapping objects.

    Usage:
        extractor = ThoughtExtractor(llm_callable)
        result = await extractor.extract("使用者說的話...")
        # result is ExtractionResult with typed objects
    """

    def __init__(
        self,
        llm_callable: LLMCallable | None = None,
        project_id: str = "default",
    ):
        """Initialize the thought extractor.

        Args:
            llm_callable: Callable that takes a prompt and returns text
            project_id: Project identifier for isolation
        """
        self._llm = llm_callable
        self.project_id = project_id

    async def extract(self, user_text: str) -> ExtractionResult:
        """Extract structured thought from user utterance.

        Args:
            user_text: The user's input text

        Returns:
            ExtractionResult containing typed Claim, ThoughtUnit, and ConceptMapping objects

        Raises:
            ValueError: If no LLM callable is provided
        """
        if not self._llm:
            raise ValueError("LLM callable is required for extraction")

        if not user_text or len(user_text.strip()) < 10:
            return ExtractionResult(
                raw_utterance=user_text,
                was_extracted=False,
            )

        prompt = EXTRACT_CLAIMS_PROMPT.format(user_text=user_text[:2000])

        # Try extraction with retries
        for attempt in range(3):
            try:
                response = await self._llm.func(prompt)
                cleaned = _clean_llm_output(response)
                data = json.loads(cleaned)

                if not isinstance(data, dict):
                    continue

                claims = self._parse_claims(data.get("claims", []))
                concepts = self._parse_concepts(
                    data.get("concepts", []),
                    [c.claim_id for c in claims],
                )

                thought_summary = data.get("thought_summary", "")
                thought_unit = None
                if thought_summary and claims:
                    thought_unit = ThoughtUnit(
                        project_id=self.project_id,
                        title=thought_summary[:50],
                        summary=thought_summary,
                        core_claim_ids=[c.claim_id for c in claims],
                        value_axes=list(set(
                            ax for c in claims for ax in c.value_axes
                        )),
                        status=ThoughtStatus.EMERGING,
                    )

                return ExtractionResult(
                    claims=claims,
                    thought_unit=thought_unit,
                    concept_mappings=concepts,
                    raw_utterance=user_text,
                    was_extracted=len(claims) > 0,
                )

            except (json.JSONDecodeError, KeyError, TypeError) as e:
                if attempt == 2:
                    break
                continue

        # Extraction failed - return empty result
        return ExtractionResult(
            raw_utterance=user_text,
            was_extracted=False,
        )

    def _parse_claims(self, claims_data: list) -> list[Claim]:
        """Parse claim data from LLM output into Claim objects."""
        claims = []
        for item in claims_data:
            if not isinstance(item, dict):
                continue
            if not item.get("core_claim"):
                continue

            claim = Claim(
                project_id=self.project_id,
                source_kind=SourceKind.USER_UTTERANCE,
                core_claim=item["core_claim"],
                critique_target=item.get("critique_target", []),
                value_axes=item.get("value_axes", []),
                materiality_axes=item.get("materiality_axes", []),
                labor_time_axes=item.get("labor_time_axes", []),
                abstract_patterns=item.get("abstract_patterns", []),
                theory_hints=item.get("theory_hints", []),
                confidence=item.get("confidence", 0.5),
            )
            claims.append(claim)

        return claims

    def _parse_concepts(
        self,
        concepts_data: list,
        owner_ids: list[str],
    ) -> list[ConceptMapping]:
        """Parse concept data from LLM output into ConceptMapping objects."""
        concepts = []
        for idx, item in enumerate(concepts_data):
            if not isinstance(item, dict):
                continue
            if not item.get("concept_id"):
                continue

            # Determine owner (use first claim if available, otherwise use utterance)
            owner_kind = OwnerKind.CLAIM
            owner_id = owner_ids[idx] if idx < len(owner_ids) else ""

            mapping = ConceptMapping(
                project_id=self.project_id,
                owner_kind=owner_kind,
                owner_id=owner_id,
                vocab_source=VocabSource(item.get("vocab_source", "internal")),
                concept_id=item["concept_id"],
                preferred_label=item.get("preferred_label", item["concept_id"]),
                confidence=item.get("confidence", 0.5),
            )
            concepts.append(mapping)

        return concepts


    async def extract_from_passage(
        self, passage_text: str, source_id: str = "",
    ) -> ExtractionResult:
        """Extract structured claims from a document passage.

        Unlike extract(), this uses a passage-specific prompt that focuses on
        argumentative structure (abstract_patterns) rather than user intent.

        Args:
            passage_text: Document passage text
            source_id: Reference to the source document

        Returns:
            ExtractionResult with claims sourced as DOCUMENT_PASSAGE
        """
        if not self._llm:
            raise ValueError("LLM callable is required for extraction")

        if not passage_text or len(passage_text.strip()) < 30:
            return ExtractionResult(
                raw_utterance=passage_text,
                was_extracted=False,
            )

        prompt = EXTRACT_PASSAGE_CLAIMS_PROMPT.format(
            passage_text=passage_text[:3000],
        )

        for attempt in range(3):
            try:
                response = await self._llm.func(prompt)
                cleaned = _clean_llm_output(response)
                data = json.loads(cleaned)

                if not isinstance(data, dict):
                    continue

                claims = self._parse_claims(data.get("claims", []))
                # Override source_kind for document passages
                for claim in claims:
                    claim.source_kind = SourceKind.DOCUMENT_PASSAGE
                    claim.source_id = source_id

                concepts = self._parse_concepts(
                    data.get("concepts", []),
                    [c.claim_id for c in claims],
                )

                thought_summary = data.get("thought_summary", "")
                thought_unit = None
                if thought_summary and claims:
                    thought_unit = ThoughtUnit(
                        project_id=self.project_id,
                        title=thought_summary[:50],
                        summary=thought_summary,
                        core_claim_ids=[c.claim_id for c in claims],
                        value_axes=list(set(
                            ax for c in claims for ax in c.value_axes
                        )),
                        status=ThoughtStatus.EMERGING,
                    )

                return ExtractionResult(
                    claims=claims,
                    thought_unit=thought_unit,
                    concept_mappings=concepts,
                    raw_utterance=passage_text,
                    was_extracted=len(claims) > 0,
                )

            except (json.JSONDecodeError, KeyError, TypeError):
                if attempt == 2:
                    break
                continue

        return ExtractionResult(
            raw_utterance=passage_text,
            was_extracted=False,
        )


async def quick_extract(
    user_text: str,
    llm_callable: LLMCallable,
    project_id: str = "default",
) -> ExtractionResult:
    """Convenience function for quick thought extraction.

    Args:
        user_text: User input text
        llm_callable: LLM callable
        project_id: Project identifier

    Returns:
        ExtractionResult with extracted thought structure
    """
    extractor = ThoughtExtractor(llm_callable, project_id)
    return await extractor.extract(user_text)
