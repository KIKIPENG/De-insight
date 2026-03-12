"""Query classification module for De-insight v2 Core.

This module determines whether a user query should use fast or deep retrieval.
Deep mode is selected when users ask about theory mapping, structural similarity,
cross-domain analogy, or conceptual interpretation.

References:
- Tech spec: deinsight_v2_codex_technical_spec.md
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from core.schemas import QueryMode


# Patterns that indicate deep mode is needed
_DEEP_PATTERNS = [
    # Theory mapping
    r"理論|理論上|理論框架|理論依據",
    r"有.*什麼.*理論|什麼.*理論",
    r"觀點.*理論|理論.*觀點",
    r"和.*什麼關係|有.*關係",  # Added for relationship questions
    
    # Structural similarity
    r"結構|架構|模式|pattern",
    r"類似.*結構|結構.*類似",
    r"邏輯|邏輯結構",
    
    # Cross-domain analogy
    r"像.*這樣|類似.*這樣|如同.*",
    r"跨界|跨領域|跨學科",
    r"相通|相通的|相同.*邏輯",
    
    # Conceptual interpretation
    r"意思是|意義是|在說什麼",
    r"概念|概念化|概念上",
    r"如何理解|怎麼理解|理解為",
    
    # Analysis and comparison
    r"分析|比較|對比",
    r"你.*怎麼.*看|你.*覺得",
    r"有.*張力|有.*矛盾",
    
    # Reflection questions
    r"為什麼|為甚麼",
    r"這.*代表.*什麼",
    r"這.*反映",
]

# Compile patterns for performance
_DEEP_RE = re.compile("|".join(_DEEP_PATTERNS), re.IGNORECASE)

# Simple fact patterns that should use fast mode
_FACT_PATTERNS = [
    r"是什麼|是甚麼|什麼是",
    r"什麼時候|何時|哪一年|哪年",
    r"是誰|哪位|誰是",
    r"在哪裡|哪個城市|哪個國家",
    r"多少|幾個|幾次",
    r"是否|有沒有|是不是",
    r"when|who|where|how many|how much",
]
_FACT_RE = re.compile("|".join(_FACT_PATTERNS), re.IGNORECASE)


@dataclass
class ClassificationResult:
    """Result of query classification.

    Attributes:
        mode: Selected query mode (fast or deep)
        why_deep: Explanation of why deep mode was selected (None for fast)
        confidence: Classification confidence (0.0-1.0)
        signals: List of detected signals that triggered the classification
    """

    mode: QueryMode
    why_deep: str | None
    confidence: float
    signals: list[str]


class QueryClassifier:
    """Classifies user queries to determine retrieval mode.

    The classifier uses heuristic patterns to determine whether a query
    requires deep retrieval (theory mapping, structural analysis, etc.)
    or can be handled with fast retrieval.

    Deep mode indicators:
    - Theory mapping: Questions about theoretical frameworks
    - Structural similarity: Questions about patterns and structures
    - Cross-domain analogy: Questions comparing across domains
    - Conceptual interpretation: Questions about meaning and concepts

    Fast mode indicators:
    - Factual questions: Who, what, when, where questions
    - Simple lookups: Basic information retrieval
    """

    def __init__(self, min_deep_confidence: float = 0.6):
        """Initialize the query classifier.

        Args:
            min_deep_confidence: Minimum confidence threshold for deep mode
        """
        self._min_deep_confidence = min_deep_confidence

    def classify(self, query: str, context: list[dict] | None = None) -> ClassificationResult:
        """Classify a query to determine retrieval mode.

        Args:
            query: User query text
            context: Optional conversation context (previous messages)

        Returns:
            ClassificationResult with mode, confidence, and signals
        """
        query = query.strip()
        if not query:
            return ClassificationResult(
                mode=QueryMode.FAST,
                why_deep=None,
                confidence=1.0,
                signals=[],
            )

        # Check for deep patterns FIRST (they take priority over fact patterns)
        deep_matches = _DEEP_RE.findall(query)
        if deep_matches:
            # Use context to boost confidence
            context_boost = 0.0
            if context:
                context_text = " ".join(
                    m.get("content", "") for m in context[-3:]
                )
                if _DEEP_RE.search(context_text):
                    context_boost = 0.15

            confidence = min(0.95, 0.6 + (len(deep_matches) * 0.1) + context_boost)

            if confidence >= self._min_deep_confidence:
                # Determine specific reason
                reasons = self._explain_deep_reason(query)
                return ClassificationResult(
                    mode=QueryMode.DEEP,
                    why_deep=reasons,
                    confidence=confidence,
                    signals=list(set(deep_matches)),
                )

        # Only check fact patterns if deep didn't match
        # Check for fact patterns (fast mode)
        if _FACT_RE.search(query):
            return ClassificationResult(
                mode=QueryMode.FAST,
                why_deep=None,
                confidence=0.9,
                signals=["fact_pattern"],
            )

        # Default to fast mode
        return ClassificationResult(
            mode=QueryMode.FAST,
            why_deep=None,
            confidence=0.7,
            signals=[],
        )

    def _explain_deep_reason(self, query: str) -> str:
        """Explain why deep mode was selected.

        Args:
            query: User query

        Returns:
            Human-readable explanation
        """
        reasons = []

        if any(kw in query.lower() for kw in ["理論", "framework", "理論上"]):
            reasons.append("理論框架相關")
        if any(kw in query.lower() for kw in ["結構", "模式", "pattern", "架構"]):
            reasons.append("結構分析相關")
        if any(kw in query.lower() for kw in ["像", "類似", "相通", "analog"]):
            reasons.append("類比聯想相關")
        if any(kw in query.lower() for kw in ["意思", "概念", "理解", "meaning"]):
            reasons.append("概念詮釋相關")
        if any(kw in query.lower() for kw in ["為什麼", "為甚麼", "why"]):
            reasons.append("反思性問題")

        if reasons:
            return "; ".join(reasons)
        return "複雜查詢需要深度檢索"


def classify_query(query: str, context: list[dict] | None = None) -> ClassificationResult:
    """Convenience function for query classification.

    Args:
        query: User query text
        context: Optional conversation context

    Returns:
        ClassificationResult
    """
    classifier = QueryClassifier()
    return classifier.classify(query, context)
