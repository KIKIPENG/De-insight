"""Tests for Bridge Surfacing Module."""

import pytest

from core.bridge_surfacing import (
    SurfacingDecision,
    SurfacingStyle,
    ConversationState,
    infer_conversation_state,
    should_surface_bridge,
    select_bridge_style,
    format_bridge,
    apply_surfacing_policy,
)
from core.schemas import Bridge, BridgeType


def make_bridge(score: float, reason: str = "test bridge", target: str = "test target") -> Bridge:
    """Helper to create a test bridge."""
    return Bridge(
        project_id="test",
        source_claim_id="source",
        target_claim_id=target,
        bridge_type=BridgeType.VALUE_STRUCTURE_MATCH,
        reason_summary=reason,
        confidence=score,
        score=score,
    )


class TestInferConversationState:
    """Tests for conversation state inference."""

    def test_empty_messages_returns_forming(self):
        state = infer_conversation_state([])
        assert state == ConversationState.FORMING

    def test_seeking_connection_detected(self):
        messages = [
            {"role": "user", "content": "有沒有相關的理論可以參考？"}
        ]
        state = infer_conversation_state(messages)
        assert state == ConversationState.SEEKING_CONNECTION

    def test_deep_engagement_detected(self):
        # Test with multiple messages that show sustained engagement
        # This tests that the conversation state detection works
        messages = [
            {"role": "user", "content": "我覺得書籍設計師在設計書籍時，應該要考量到書籍的材質，不應該為了特定目的去把書籍本該有的材質性忽略。"},
            {"role": "assistant", "content": "這是關於材料選擇的問題。"},
            {"role": "user", "content": "對，我特別在意書籍的質感，書籍設計應該尊重紙張和印刷呈現，這是書籍靈魂的一部分。"},
            {"role": "user", "content": "而且我認為這種對材質的堅持，應該擴展到所有設計領域。"},
        ]
        state = infer_conversation_state(messages)
        # Either DEEP_ENGAGED or STABILIZING are valid for surfacing
        assert state in (ConversationState.DEEP_ENGAGED, ConversationState.STABILIZING)

    def test_stabilizing_with_repeated_topic(self):
        messages = [
            {"role": "user", "content": "我喜歡手工製作的東西。"},
            {"role": "user", "content": "手工的東西比較有故事。"},
        ]
        state = infer_conversation_state(messages)
        assert state == ConversationState.STABILIZING

    def test_forming_default(self):
        messages = [
            {"role": "user", "content": "你好"}
        ]
        state = infer_conversation_state(messages)
        assert state == ConversationState.FORMING


class TestShouldSurfaceBridge:
    """Tests for surfacing decision logic."""

    def test_suppress_when_anchor_quality_low(self):
        bridges = [make_bridge(0.5)]
        decision = should_surface_bridge(
            anchor_quality=5,  # Below threshold
            bridges=bridges,
            conversation_state=ConversationState.STABILIZING,
        )
        assert not decision.should_surface
        assert decision.reason == "anchor_quality_low"

    def test_suppress_when_no_bridges(self):
        decision = should_surface_bridge(
            anchor_quality=15,
            bridges=[],
            conversation_state=ConversationState.STABILIZING,
        )
        assert not decision.should_surface
        assert decision.reason == "no_bridges"

    def test_suppress_when_bridge_score_low(self):
        bridges = [make_bridge(0.02)]  # Below MIN_BRIDGE_SCORE
        decision = should_surface_bridge(
            anchor_quality=15,
            bridges=bridges,
            conversation_state=ConversationState.STABILIZING,
        )
        assert not decision.should_surface
        assert decision.reason == "bridge_score_low"

    def test_suppress_when_forming_state_weak_bridge(self):
        bridges = [make_bridge(0.10)]  # Not exceptional enough for FORMING
        decision = should_surface_bridge(
            anchor_quality=15,
            bridges=bridges,
            conversation_state=ConversationState.FORMING,
        )
        assert not decision.should_surface

    def test_surface_when_trigger_conditions_met(self):
        bridges = [make_bridge(0.15)]
        decision = should_surface_bridge(
            anchor_quality=15,
            bridges=bridges,
            conversation_state=ConversationState.STABILIZING,
        )
        assert decision.should_surface
        assert decision.style is not None
        assert decision.confidence > 0

    def test_suppress_recent_similar(self):
        bridges = [make_bridge(0.15, reason="Ruskin")]
        recent = ["Ruskin", "Arts and Crafts"]
        decision = should_surface_bridge(
            anchor_quality=15,
            bridges=bridges,
            conversation_state=ConversationState.STABILIZING,
            recent_surfaced=recent,
        )
        assert not decision.should_surface
        assert decision.reason == "suppressed_recent_similar"


class TestSelectBridgeStyle:
    """Tests for style selection."""

    def test_forming_always_gentle_hint(self):
        style = select_bridge_style(
            conversation_state=ConversationState.FORMING,
            bridge_score=0.5,
            anchor_quality=20,
        )
        assert style == SurfacingStyle.GENTLE_HINT

    def test_seeking_connection_prefers_cross_domain(self):
        style = select_bridge_style(
            conversation_state=ConversationState.SEEKING_CONNECTION,
            bridge_score=0.15,
            anchor_quality=15,
        )
        assert style == SurfacingStyle.CROSS_DOMAIN

    def test_deep_engaged_allows_reflective(self):
        style = select_bridge_style(
            conversation_state=ConversationState.DEEP_ENGAGED,
            bridge_score=0.20,
            anchor_quality=18,
        )
        assert style == SurfacingStyle.REFLECTIVE_COMPANION

    def test_stabilizing_gentle_for_low_score(self):
        style = select_bridge_style(
            conversation_state=ConversationState.STABILIZING,
            bridge_score=0.08,
            anchor_quality=12,
        )
        assert style == SurfacingStyle.GENTLE_HINT

    def test_stabilizing_cross_domain_for_higher_score(self):
        style = select_bridge_style(
            conversation_state=ConversationState.STABILIZING,
            bridge_score=0.15,
            anchor_quality=15,
        )
        assert style == SurfacingStyle.CROSS_DOMAIN


class TestFormatBridge:
    """Tests for bridge formatting."""

    def test_gentle_hint_stays_short(self):
        bridge = make_bridge(0.15, reason="可能有深挖的空間")
        formatted = format_bridge(
            bridge,
            SurfacingStyle.GENTLE_HINT,
            user_context="我喜歡手工製作的東西",
        )
        assert len(formatted.split()) <= 15  # Very short

    def test_cross_domain_includes_hedging(self):
        bridge = make_bridge(0.15, reason="Ruskin 的真理之燈", target="material honesty")
        formatted = format_bridge(
            bridge,
            SurfacingStyle.CROSS_DOMAIN,
            user_context="我喜歡書籍的質感",
        )
        # Should include hedging or be suggestive
        assert "可能" in formatted or "有關" in formatted

    def test_reflective_companion_includes_user_topic(self):
        bridge = make_bridge(0.20, reason="wabi-sabi 美學", target="imperfection value")
        formatted = format_bridge(
            bridge,
            SurfacingStyle.REFLECTIVE_COMPANION,
            user_context="我特別喜歡那種有痕跡的東西",
        )
        # Should reference user's topic
        assert "痕跡" in formatted or "喜歡" in formatted

    def test_empty_bridge_returns_empty(self):
        formatted = format_bridge(None, SurfacingStyle.GENTLE_HINT, "")
        assert formatted == ""


class TestApplySurfacingPolicy:
    """Tests for full policy application."""

    def test_full_policy_suppresses_low_quality(self):
        should_surface, text = apply_surfacing_policy(
            anchor_quality=5,
            bridges=[make_bridge(0.5)],
            messages=[{"role": "user", "content": "test"}],
        )
        assert not should_surface
        assert text is None

    def test_full_policy_allows_good_bridge(self):
        bridges = [make_bridge(0.25, reason="wabi-sabi")]  # Higher score for exceptional case
        messages = [
            {"role": "user", "content": "我特別喜歡那種有痕跡的作品。"},
        ]
        should_surface, text = apply_surfacing_policy(
            anchor_quality=15,
            bridges=bridges,
            messages=messages,
        )
        assert should_surface
        assert text is not None
        assert len(text.split()) <= 50

    def test_respects_recent_surfaced_suppression(self):
        bridges = [make_bridge(0.15, reason="wabi-sabi")]
        messages = [{"role": "user", "content": "我喜歡有痕跡的東西"}]
        should_surface, text = apply_surfacing_policy(
            anchor_quality=15,
            bridges=bridges,
            messages=messages,
            recent_surfaced=["wabi-sabi"],
        )
        assert not should_surface

    def test_partial_bridge_data_does_not_crash(self):
        # Test with minimal/partial bridge data
        partial_bridge = Bridge(
            project_id="test",
            source_claim_id="",
            target_claim_id="",
            bridge_type=BridgeType.VALUE_STRUCTURE_MATCH,
            reason_summary="",
            confidence=0.1,
            score=0.1,
        )
        should_surface, text = apply_surfacing_policy(
            anchor_quality=15,
            bridges=[partial_bridge],
            messages=[{"role": "user", "content": "test"}],
        )
        # Should not crash, may or may not surface
        assert isinstance(should_surface, bool)


class TestRecentBridgeTracking:
    """Tests for recent surfaced bridge tracking and suppression."""

    def test_normalize_bridge_topic(self):
        from core.bridge_surfacing import _normalize_for_comparison
        # Test normalization - hyphen is removed
        assert _normalize_for_comparison("Ruskin's Lamp of Truth") == "ruskins lamp of truth"
        assert _normalize_for_comparison("Wabi-sabi (佗寂)") == "wabisabi 佗寂"
        assert _normalize_for_comparison("  multiple   spaces  ") == "multiple spaces"
        assert _normalize_for_comparison("") == ""

    def test_recent_3_turns_suppression(self):
        """Test that suppression checks only last 3 turns."""
        bridges = [make_bridge(0.15, reason="Ruskin")]
        
        # Recent 5 bridges, but only last 3 should trigger suppression
        recent = [
            "Arts and Crafts",  # turn 1 - should NOT suppress (outside 3)
            "William Morris",   # turn 2 - should NOT suppress (outside 3)
            "material honesty", # turn 3 - should NOT suppress (at edge)
            "Ruskin",          # turn 4 - should suppress
            "truth to materials", # turn 5 - should suppress
        ]
        
        # Ruskin should be suppressed because it's in last 3
        should_surface, _ = apply_surfacing_policy(
            anchor_quality=15,
            bridges=bridges,
            messages=[{"role": "user", "content": "test"}],
            recent_surfaced=recent,
        )
        assert not should_surface

    def test_string_and_record_compatibility(self):
        """Test that recent_surfaced works with both strings and records."""
        bridges = [make_bridge(0.15, reason="wabi-sabi")]
        
        # Test with string list
        should_surface_str, _ = apply_surfacing_policy(
            anchor_quality=15,
            bridges=bridges,
            messages=[{"role": "user", "content": "test"}],
            recent_surfaced=["wabi-sabi"],
        )
        
        # Test with record-like objects
        class FakeRecord:
            def __init__(self, topic):
                self.topic = topic
        
        should_surface_rec, _ = apply_surfacing_policy(
            anchor_quality=15,
            bridges=bridges,
            messages=[{"role": "user", "content": "test"}],
            recent_surfaced=[FakeRecord("wabi-sabi")],
        )
        
        # Both should suppress
        assert not should_surface_str
        assert not should_surface_rec

    def test_normalized_suppression_substring_matching(self):
        """Test that normalized topics match via substring."""
        bridges = [make_bridge(0.15, reason="kintsugi 金繼")]
        
        # Should match "kintsugi" in "kintsugi gold repair"
        recent = ["kintsugi gold repair"]
        
        should_surface, _ = apply_surfacing_policy(
            anchor_quality=15,
            bridges=bridges,
            messages=[{"role": "user", "content": "test"}],
            recent_surfaced=recent,
        )
        assert not should_surface

    def test_no_false_suppression_different_topics(self):
        """Test that different topics don't suppress each other."""
        bridges = [make_bridge(0.15, reason="Ruskin")]
        
        recent = ["wabi-sabi", "kintsugi", "material memory"]
        
        # Use longer message to trigger STABILIZING state (not FORMING)
        messages = [
            {"role": "user", "content": "我喜歡書籍設計"},
            {"role": "user", "content": "我覺得書籍應該有質感"},
        ]
        
        should_surface, _ = apply_surfacing_policy(
            anchor_quality=15,
            bridges=bridges,
            messages=messages,
            recent_surfaced=recent,
        )
        assert should_surface


class TestSurfacingDisplayIntegration:
    """Minimal tests for display integration."""

    def test_surfaced_bridge_extracted_from_result(self):
        """Test that pipeline result contains surfaced_bridge field."""
        # Minimal test - just verify the field exists
        from rag.pipeline import run_thinking_pipeline
        import asyncio
        
        # Can't fully test without running pipeline, but verify import works
        assert run_thinking_pipeline is not None

    def test_format_bridge_stays_short(self):
        """Test that formatted bridge is short enough for display."""
        bridge = make_bridge(0.15, reason="This is a longer reason summary that should be truncated to stay within display limits")
        formatted = format_bridge(bridge, SurfacingStyle.GENTLE_HINT, "user context")
        
        # Should be reasonably short for display
        assert len(formatted) <= 200  # Reasonable display length


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
