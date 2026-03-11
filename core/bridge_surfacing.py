"""Bridge Surfacing Module.

Implements the first-pass surfacing policy from Milestone G.

Decision flow:
1. Assess conversation state (simple conservative inference)
2. Check trigger conditions (anchor quality, bridge score)
3. Check suppression conditions (recent similar, weak signal)
4. Select surfacing style based on state
5. Format bridge text following policy constraints
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any

from core.schemas import Bridge


class SurfacingStyle(Enum):
    GENTLE_HINT = "gentle_hint"
    CROSS_DOMAIN = "cross_domain"
    REFLECTIVE_COMPANION = "reflective_companion"


class ConversationState(Enum):
    FORMING = "forming"  # Default - suppress unless exceptional
    STABILIZING = "stabilizing"
    DEEP_ENGAGED = "deep_engaged"
    SEEKING_CONNECTION = "seeking_connection"


@dataclass
class SurfacingDecision:
    should_surface: bool
    style: SurfacingStyle | None
    reason: str
    confidence: float = 0.0


MIN_ANCHOR_QUALITY = 10
MIN_BRIDGE_SCORE = 0.05
MIN_BRIDGE_SCORE_STRONG = 0.10
EXCEPTIONAL_BRIDGE_SCORE = 0.20
MAX_WORDS = 50
RECENT_SUPPRESSION_TURNS = 3


def infer_conversation_state(messages: list[dict]) -> ConversationState:
    """Infer conversation state from recent messages.
    
    Conservative first-pass approach using simple heuristics.
    """
    if not messages:
        return ConversationState.FORMING
    
    recent = messages[-5:]  # Look at last 5 messages
    user_msgs = [m for m in recent if m.get("role") == "user"]
    
    if not user_msgs:
        return ConversationState.FORMING
    
    last_user = user_msgs[-1].get("content", "")
    
    # Check for explicit connection seeking
    seeking_markers = ["有沒有", "相關", "理論", "連接", "connection", "related"]
    if any(marker in last_user.lower() for marker in seeking_markers):
        return ConversationState.SEEKING_CONNECTION
    
    # Check for deep engagement: longer messages with reflective content
    if len(last_user) > 50:
        deep_markers = ["我覺得", "我認為", "我喜歡", "我特別", "一直", "總是"]
        if any(marker in last_user for marker in deep_markers):
            # Check for repetition across messages
            if len(user_msgs) >= 2:
                # Simple repetition check: any word > 5 chars appears in 2+ msgs
                words = set(w for w in last_user if len(w) > 5)
                for w in words:
                    if sum(1 for m in user_msgs if w in m.get("content", "")) >= 2:
                        return ConversationState.DEEP_ENGAGED
    
    # Check for stabilizing: multiple messages with sustained topic
    if len(user_msgs) >= 2:
        # Check if last message continues same topic as previous
        return ConversationState.STABILIZING
    
    return ConversationState.FORMING


def should_surface_bridge(
    anchor_quality: int,
    bridges: list[Bridge],
    conversation_state: ConversationState,
    recent_surfaced: list[str] | None = None,
) -> SurfacingDecision:
    """Decide whether to surface a bridge.
    
    Args:
        anchor_quality: Quality score 0-25 from assess_anchor_quality()
        bridges: List of ranked bridge candidates
        conversation_state: Current inferred conversation state
        recent_surfaced: List of recently surfaced bridge topics for suppression
    
    Returns:
        SurfacingDecision with should_surface, style, reason, confidence
    """
    recent_surfaced = recent_surfaced or []
    
    # Check anchor quality threshold
    if anchor_quality < MIN_ANCHOR_QUALITY:
        return SurfacingDecision(
            should_surface=False,
            style=None,
            reason="anchor_quality_low",
            confidence=0.0,
        )
    
    # Check if we have bridges
    if not bridges:
        return SurfacingDecision(
            should_surface=False,
            style=None,
            reason="no_bridges",
            confidence=0.0,
        )
    
    # Get best bridge score
    best_bridge = max(bridges, key=lambda b: b.score)
    best_score = best_bridge.score if best_bridge else 0.0
    
    # Check bridge score thresholds
    if best_score < MIN_BRIDGE_SCORE:
        return SurfacingDecision(
            should_surface=False,
            style=None,
            reason="bridge_score_low",
            confidence=float(best_score),
        )
    
    # Determine required threshold based on conversation state
    required_threshold = MIN_BRIDGE_SCORE
    if conversation_state == ConversationState.FORMING:
        required_threshold = EXCEPTIONAL_BRIDGE_SCORE
    elif conversation_state == ConversationState.STABILIZING:
        required_threshold = MIN_BRIDGE_SCORE_STRONG
    
    if best_score < required_threshold:
        return SurfacingDecision(
            should_surface=False,
            style=None,
            reason="insufficient_confidence_for_state",
            confidence=float(best_score),
        )
    
    # Check recent suppression
    bridge_topic = _extract_bridge_topic(best_bridge)
    normalized_topic = _normalize_for_comparison(bridge_topic)
    if recent_surfaced:
        recent_topics = []
        for r in recent_surfaced[-RECENT_SUPPRESSION_TURNS:]:
            if isinstance(r, str):
                recent_topics.append(_normalize_for_comparison(r))
            elif hasattr(r, 'topic'):
                recent_topics.append(r.topic)
        
        for recent_norm in recent_topics:
            if recent_norm and normalized_topic:
                if recent_norm in normalized_topic or normalized_topic in recent_norm:
                    return SurfacingDecision(
                        should_surface=False,
                        style=None,
                        reason="suppressed_recent_similar",
                        confidence=float(best_score),
                    )
    
    # Select style based on state and bridge characteristics
    style = select_bridge_style(conversation_state, best_score, anchor_quality)
    
    return SurfacingDecision(
        should_surface=True,
        style=style,
        reason="passed_all_checks",
        confidence=float(best_score),
    )


def select_bridge_style(
    conversation_state: ConversationState,
    bridge_score: float,
    anchor_quality: int,
) -> SurfacingStyle:
    """Select appropriate surfacing style based on context.
    
    Style selection rules:
    - FORMING: Gentle Hint only (low intervention)
    - STABILIZING: Gentle Hint or Cross-Domain
    - SEEKING_CONNECTION: Cross-Domain or Reflective
    - DEEP_ENGAGED: Cross-Domain or Reflective (higher intervention allowed)
    """
    # Higher scores and quality allow more intervention
    intervention_level = 1
    if bridge_score >= MIN_BRIDGE_SCORE_STRONG:
        intervention_level = 2
    if bridge_score >= 0.15 and anchor_quality >= 15:
        intervention_level = 3
    
    # Override based on state
    if conversation_state == ConversationState.FORMING:
        return SurfacingStyle.GENTLE_HINT
    
    if conversation_state == ConversationState.SEEKING_CONNECTION:
        return SurfacingStyle.CROSS_DOMAIN if intervention_level < 4 else SurfacingStyle.REFLECTIVE_COMPANION
    
    if conversation_state == ConversationState.DEEP_ENGAGED:
        return SurfacingStyle.CROSS_DOMAIN if intervention_level < 3 else SurfacingStyle.REFLECTIVE_COMPANION
    
    # Default: STABILIZING
    if intervention_level >= 2:
        return SurfacingStyle.CROSS_DOMAIN
    return SurfacingStyle.GENTLE_HINT


def _extract_bridge_topic(bridge: Bridge) -> str:
    """Extract topic from bridge for suppression checking."""
    topic = bridge.reason_summary or ""
    if not topic and hasattr(bridge, "target_claim_id"):
        topic = bridge.target_claim_id
    return topic


def _normalize_for_comparison(text: str) -> str:
    """Normalize text for comparison (lowercase, remove punctuation)."""
    import re
    if not text:
        return ""
    normalized = text.lower()
    normalized = re.sub(r'[^\w\s\u4e00-\u9fff]', '', normalized)
    normalized = re.sub(r'\s+', ' ', normalized).strip()
    return normalized


def format_bridge(
    bridge: Bridge,
    style: SurfacingStyle,
    user_context: str = "",
) -> str:
    """Format a bridge for surfacing following policy constraints.
    
    Constraints:
    - Max 50 words (~2-3 sentences)
    - Use hedging language (可能, 說不定, 好像)
    - Non-dominating tone
    - Easy to ignore or continue
    - Avoid: lecture tone, over-certainty, theory-dumping
    """
    if not bridge:
        return ""
    
    # Extract bridge content
    reason = bridge.reason_summary or "這個想法有深挖的空間"
    target = getattr(bridge, "target_claim_id", "") or ""
    
    # Style-specific formatting
    if style == SurfacingStyle.GENTLE_HINT:
        return _format_gentle_hint(reason, user_context)
    elif style == SurfacingStyle.CROSS_DOMAIN:
        return _format_cross_domain(reason, target, user_context)
    else:  # REFLECTIVE_COMPANION
        return _format_reflective(reason, target, user_context)


def _format_gentle_hint(reason: str, user_context: str) -> str:
    """Format as gentle hint - minimal intervention."""
    # Keep it very short, soft suggestion
    hint = reason[:80] if len(reason) > 80 else reason
    
    # Add hedging if not present
    if not any(w in hint for w in ["可能", "說不定", "好像", "或許"]):
        hint = f"這條思路 {hint}"
    
    return hint


def _format_cross_domain(reason: str, target: str, user_context: str) -> str:
    """Format as cross-domain suggestion - medium intervention."""
    parts = []
    
    # Acknowledge user's direction briefly
    if user_context:
        # Extract first key term from user context
        key_term = _extract_key_term(user_context)
        if key_term:
            parts.append(f"你說的「{key_term}」")
    
    # Add theory connection with hedging
    if reason:
        parts.append(f"可能跟 {reason} 有關")
    
    # Keep it to 2 sentences max
    text = "，".join(parts)
    if len(text) > 120:
        text = text[:120] + "..."
    
    return text


def _format_reflective(reason: str, target: str, user_context: str) -> str:
    """Format as reflective companion - higher intervention."""
    parts = []
    
    # Start with user's thought
    if user_context:
        key_term = _extract_key_term(user_context)
        if key_term:
            parts.append(f"你特別提到「{key_term}」")
    
    # Add insight with connection
    if reason:
        parts.append(f"這讓我想到 {reason}")
    
    # Add brief why-it-connects
    if target:
        # Keep explanation minimal
        target_short = target[:40] if len(target) > 40 else target
        parts.append(f"——兩者都在說同一種價值")
    
    # 3 sentences max
    text = "，".join(parts)
    if len(text) > 150:
        text = text[:150] + "。"
    
    return text


def _extract_key_term(text: str) -> str:
    """Extract a key term from user context for personalization."""
    if not text:
        return ""
    
    # Simple extraction: first substantial noun-like phrase
    # Look for quotes first
    import re
    quotes = re.findall(r'「([^」]+)」', text)
    if quotes:
        return quotes[0]
    
    # Otherwise, take a substantial segment
    words = text.split()
    for w in words:
        if len(w) > 3 and w not in ["我覺得", "我認為", "我特別", "可能", "說不定"]:
            return w[:10]
    
    return text[:10] if text else ""


def apply_surfacing_policy(
    anchor_quality: int,
    bridges: list[Bridge],
    messages: list[dict],
    recent_surfaced: list[str] | None = None,
) -> tuple[bool, str | None]:
    """Apply full surfacing policy and return (should_surface, formatted_bridge).
    
    This is the main entry point for applying surfacing.
    
    Args:
        anchor_quality: Quality score 0-25
        bridges: List of ranked bridges
        messages: Recent conversation messages
        recent_surfaced: Recently surfaced topics
    
    Returns:
        Tuple of (should_surface, formatted_bridge_text or None)
    """
    state = infer_conversation_state(messages)
    decision = should_surface_bridge(anchor_quality, bridges, state, recent_surfaced)
    
    if not decision.should_surface:
        return False, None
    
    if not bridges or not decision.style:
        return False, None
    
    # Get best bridge
    best_bridge = max(bridges, key=lambda b: b.score)
    user_context = messages[-1].get("content", "") if messages else ""
    
    # Format bridge
    formatted = format_bridge(best_bridge, decision.style, user_context)
    
    # Final word count check
    word_count = len(formatted.split())
    if word_count > MAX_WORDS:
        formatted = " ".join(formatted.split()[:MAX_WORDS]) + "..."
    
    return True, formatted
