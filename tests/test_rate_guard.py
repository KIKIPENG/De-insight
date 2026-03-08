"""RateGuard 單元測試 — Phase A hard gates。

Gate D: 速率控制與熔斷器。
所有測試都是 hard-fail，不允許 skip 或 expected failure。
"""

from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from rag.rate_guard import (
    BreakerState,
    RateGuard,
    RateLimitError,
    TokenBucket,
    get_rate_guard,
    reset_rate_guard,
)


# ═══════════════════════════════════════════════════════════════════
# Section 1: TokenBucket
# ═══════════════════════════════════════════════════════════════════


class TestTokenBucket:

    @pytest.mark.asyncio
    async def test_acquire_single_token(self):
        bucket = TokenBucket(rpm=60)
        ok = await bucket.acquire(timeout=1.0)
        assert ok

    @pytest.mark.asyncio
    async def test_acquire_exhausts_tokens(self):
        bucket = TokenBucket(rpm=2)
        ok1 = await bucket.acquire(timeout=1.0)
        ok2 = await bucket.acquire(timeout=1.0)
        assert ok1
        assert ok2
        # Third should timeout quickly
        ok3 = await bucket.acquire(timeout=0.1)
        assert not ok3

    @pytest.mark.asyncio
    async def test_tokens_refill(self):
        bucket = TokenBucket(rpm=60)
        # Exhaust many
        for _ in range(10):
            await bucket.acquire(timeout=0.1)
        # Wait for refill
        await asyncio.sleep(0.2)
        ok = await bucket.acquire(timeout=0.5)
        assert ok

    def test_available_property(self):
        bucket = TokenBucket(rpm=10)
        assert bucket.available >= 1


# ═══════════════════════════════════════════════════════════════════
# Section 2: RateGuard acquire
# ═══════════════════════════════════════════════════════════════════


class TestRateGuardAcquire:

    @pytest.mark.asyncio
    async def test_acquire_yields_request_id(self):
        guard = RateGuard(rpm=60, max_concurrency=2)
        async with guard.acquire("test/endpoint") as req_id:
            assert isinstance(req_id, str)
            assert len(req_id) > 0

    @pytest.mark.asyncio
    async def test_acquire_tracks_queue_depth(self):
        guard = RateGuard(rpm=60, max_concurrency=2)
        async with guard.acquire("test") as _:
            assert guard.queue_depth >= 0  # may be 0 by the time we check
        assert guard.queue_depth == 0

    @pytest.mark.asyncio
    async def test_acquire_records_events(self):
        guard = RateGuard(rpm=60, max_concurrency=2)
        async with guard.acquire("test/ep") as _:
            pass
        events = guard.recent_events(10)
        assert len(events) >= 1
        assert events[-1]["endpoint"] == "test/ep"
        assert events[-1]["status"] == "ok"

    @pytest.mark.asyncio
    async def test_acquire_records_error_event(self):
        guard = RateGuard(rpm=60, max_concurrency=2)
        with pytest.raises(ValueError):
            async with guard.acquire("test/fail") as _:
                raise ValueError("test error")
        events = guard.recent_events(10)
        assert events[-1]["status"] == "error"
        assert "test error" in events[-1]["error"]


# ═══════════════════════════════════════════════════════════════════
# Section 3: Circuit Breaker
# ═══════════════════════════════════════════════════════════════════


class TestCircuitBreaker:

    @pytest.mark.asyncio
    async def test_breaker_starts_closed(self):
        guard = RateGuard(rpm=60, max_concurrency=2, breaker_threshold=3)
        assert guard.breaker_state == BreakerState.CLOSED

    @pytest.mark.asyncio
    async def test_breaker_opens_after_threshold(self):
        guard = RateGuard(
            rpm=60, max_concurrency=2,
            breaker_threshold=3, breaker_cooldown=60.0,
        )
        for i in range(3):
            with pytest.raises(RuntimeError):
                async with guard.acquire("test") as _:
                    raise RuntimeError(f"fail {i}")

        assert guard.breaker_state == BreakerState.OPEN

    @pytest.mark.asyncio
    async def test_breaker_open_rejects_requests(self):
        guard = RateGuard(
            rpm=60, max_concurrency=2,
            breaker_threshold=2, breaker_cooldown=60.0,
        )
        # Trip the breaker
        for _ in range(2):
            with pytest.raises(RuntimeError):
                async with guard.acquire("test") as _:
                    raise RuntimeError("fail")

        assert guard.breaker_state == BreakerState.OPEN

        # Next request should be rejected with RateLimitError
        with pytest.raises(RateLimitError, match="Circuit breaker OPEN"):
            async with guard.acquire("test") as _:
                pass

    @pytest.mark.asyncio
    async def test_breaker_recovers_after_cooldown(self):
        guard = RateGuard(
            rpm=60, max_concurrency=2,
            breaker_threshold=2, breaker_cooldown=0.1,  # 100ms cooldown for test
        )
        for _ in range(2):
            with pytest.raises(RuntimeError):
                async with guard.acquire("test") as _:
                    raise RuntimeError("fail")

        assert guard.breaker_state == BreakerState.OPEN

        # Wait for cooldown
        await asyncio.sleep(0.15)

        # Should transition to HALF_OPEN then CLOSED on success
        async with guard.acquire("test") as _:
            pass

        assert guard.breaker_state == BreakerState.CLOSED

    @pytest.mark.asyncio
    async def test_breaker_remaining_seconds(self):
        guard = RateGuard(
            rpm=60, max_concurrency=2,
            breaker_threshold=1, breaker_cooldown=10.0,
        )
        with pytest.raises(RuntimeError):
            async with guard.acquire("test") as _:
                raise RuntimeError("fail")

        remaining = guard.breaker_remaining_seconds
        assert remaining > 0
        assert remaining <= 10.0


# ═══════════════════════════════════════════════════════════════════
# Section 4: call_with_retry
# ═══════════════════════════════════════════════════════════════════


class TestCallWithRetry:

    @pytest.mark.asyncio
    async def test_retry_succeeds_on_first_try(self):
        guard = RateGuard(rpm=60, max_concurrency=2)
        fn = AsyncMock(return_value="ok")
        result = await guard.call_with_retry("test", fn)
        assert result == "ok"
        fn.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_retry_on_transient_error(self):
        guard = RateGuard(rpm=60, max_concurrency=2)
        call_count = 0

        async def flaky():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("connection refused")
            return "recovered"

        result = await guard.call_with_retry("test", flaky, max_retries=3)
        assert result == "recovered"
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_no_retry_on_permanent_error(self):
        guard = RateGuard(rpm=60, max_concurrency=2)

        async def fail_permanent():
            raise ValueError("invalid format")  # not transient

        with pytest.raises(ValueError, match="invalid format"):
            await guard.call_with_retry("test", fail_permanent, max_retries=3)

    @pytest.mark.asyncio
    async def test_no_retry_on_rate_limit_error(self):
        """RateLimitError (breaker/throttle) should not be retried."""
        guard = RateGuard(
            rpm=60, max_concurrency=2,
            breaker_threshold=1, breaker_cooldown=60.0,
        )
        # Trip the breaker first
        with pytest.raises(RuntimeError):
            async with guard.acquire("test") as _:
                raise RuntimeError("fail")

        fn = AsyncMock(return_value="should not reach")
        with pytest.raises(RateLimitError):
            await guard.call_with_retry("test", fn, max_retries=3)
        fn.assert_not_awaited()


# ═══════════════════════════════════════════════════════════════════
# Section 5: Status & Events
# ═══════════════════════════════════════════════════════════════════


class TestStatusAndEvents:

    @pytest.mark.asyncio
    async def test_status_dict(self):
        guard = RateGuard(rpm=20, max_concurrency=1)
        s = guard.status()
        assert "breaker" in s
        assert s["breaker"] == "closed"
        assert "queue_depth" in s
        assert "available_rpm_tokens" in s
        assert "consecutive_failures" in s
        assert "total_events" in s

    @pytest.mark.asyncio
    async def test_recent_events_limit(self):
        guard = RateGuard(rpm=60, max_concurrency=2)
        for _ in range(5):
            async with guard.acquire("test") as _:
                pass
        events = guard.recent_events(3)
        assert len(events) == 3

    @pytest.mark.asyncio
    async def test_event_has_latency(self):
        guard = RateGuard(rpm=60, max_concurrency=2)
        async with guard.acquire("test") as _:
            await asyncio.sleep(0.01)
        events = guard.recent_events(1)
        assert events[0]["latency_ms"] > 0


# ═══════════════════════════════════════════════════════════════════
# Section 6: Singleton
# ═══════════════════════════════════════════════════════════════════


class TestSingleton:

    def setup_method(self):
        reset_rate_guard()

    def teardown_method(self):
        reset_rate_guard()

    def test_singleton_returns_same_instance(self):
        g1 = get_rate_guard()
        g2 = get_rate_guard()
        assert g1 is g2

    def test_reset_creates_new_instance(self):
        g1 = get_rate_guard()
        reset_rate_guard()
        g2 = get_rate_guard()
        assert g1 is not g2

    def test_singleton_respects_env(self):
        with patch.dict("os.environ", {"RATE_GUARD_RPM": "100", "RATE_GUARD_CONCURRENCY": "5"}):
            reset_rate_guard()
            g = get_rate_guard()
            assert g._bucket._rpm == 100


# ═══════════════════════════════════════════════════════════════════
# Section 7: Transient error detection
# ═══════════════════════════════════════════════════════════════════


class TestTransientDetection:

    def test_429_is_transient(self):
        assert RateGuard._is_transient(Exception("HTTP 429 Too Many Requests"))

    def test_rate_limit_is_transient(self):
        assert RateGuard._is_transient(Exception("rate limit exceeded"))

    def test_timeout_is_transient(self):
        assert RateGuard._is_transient(Exception("connection timeout"))

    def test_502_is_transient(self):
        assert RateGuard._is_transient(Exception("502 Bad Gateway"))

    def test_value_error_is_not_transient(self):
        assert not RateGuard._is_transient(ValueError("invalid input"))


# ═══════════════════════════════════════════════════════════════════
# Section 8: Backoff delay
# ═══════════════════════════════════════════════════════════════════


class TestBackoff:

    def test_backoff_increases(self):
        d0 = RateGuard._backoff_delay(0)
        d1 = RateGuard._backoff_delay(1)
        d2 = RateGuard._backoff_delay(2)
        # Base increases: 1, 2, 4 (plus jitter)
        assert d0 >= 1.0
        assert d1 >= 2.0
        assert d2 >= 4.0

    def test_backoff_capped_at_60(self):
        d = RateGuard._backoff_delay(10)
        assert d <= 60 + 30  # 60 base + max jitter (50% of 60)


# ═══════════════════════════════════════════════════════════════════
# Section 9: Job dedup
# ═══════════════════════════════════════════════════════════════════


class TestJobDedup:

    @pytest.mark.asyncio
    async def test_duplicate_queued_source_rejected(self, tmp_path):
        from rag.job_repository import DuplicateJobError, JobRepository

        db_path = tmp_path / "test_dedup.db"
        repo = JobRepository(db_path)
        await repo.ensure_table()

        # First job should succeed (status=queued)
        job_id1 = await repo.create_job(
            project_id="proj-1",
            source="https://example.com/article",
            source_type="url",
        )
        assert job_id1

        # Same source + project while still queued should fail
        with pytest.raises(DuplicateJobError):
            await repo.create_job(
                project_id="proj-1",
                source="https://example.com/article",
                source_type="url",
            )

    @pytest.mark.asyncio
    async def test_done_source_allows_reimport(self, tmp_path):
        from rag.job_repository import JobRepository

        db_path = tmp_path / "test_dedup_done.db"
        repo = JobRepository(db_path)
        await repo.ensure_table()

        job_id1 = await repo.create_job(
            project_id="proj-1",
            source="https://example.com/article",
            source_type="url",
        )
        await repo.update_status(job_id1, "done")

        # Same source after done should succeed (re-import / update)
        job_id2 = await repo.create_job(
            project_id="proj-1",
            source="https://example.com/article",
            source_type="url",
        )
        assert job_id2 != job_id1

    @pytest.mark.asyncio
    async def test_force_bypasses_dedup(self, tmp_path):
        from rag.job_repository import JobRepository

        db_path = tmp_path / "test_dedup_force.db"
        repo = JobRepository(db_path)
        await repo.ensure_table()

        job_id1 = await repo.create_job(
            project_id="proj-1",
            source="https://example.com/article",
            source_type="url",
        )
        # Force should bypass even queued dedup
        job_id2 = await repo.create_job(
            project_id="proj-1",
            source="https://example.com/article",
            source_type="url",
            force=True,
        )
        assert job_id2 != job_id1

    @pytest.mark.asyncio
    async def test_different_projects_not_deduped(self, tmp_path):
        from rag.job_repository import JobRepository

        db_path = tmp_path / "test_dedup2.db"
        repo = JobRepository(db_path)
        await repo.ensure_table()

        job1 = await repo.create_job(
            project_id="proj-1",
            source="https://example.com/article",
            source_type="url",
        )
        job2 = await repo.create_job(
            project_id="proj-2",
            source="https://example.com/article",
            source_type="url",
        )
        assert job1 != job2

    @pytest.mark.asyncio
    async def test_failed_job_allows_resubmit(self, tmp_path):
        from rag.job_repository import JobRepository

        db_path = tmp_path / "test_dedup3.db"
        repo = JobRepository(db_path)
        await repo.ensure_table()

        job1 = await repo.create_job(
            project_id="proj-1",
            source="test-source",
            source_type="text",
        )
        # Mark as failed
        await repo.update_status(job1, "failed", last_error="test error")

        # Should be able to resubmit
        job2 = await repo.create_job(
            project_id="proj-1",
            source="test-source",
            source_type="text",
        )
        assert job2 != job1


# ═══════════════════════════════════════════════════════════════════
# Section 10: EMBEDDING_LOCAL_ONLY enforcement
# ═══════════════════════════════════════════════════════════════════


class TestEmbeddingLocalOnly:

    def setup_method(self):
        from embeddings.service import reset_embedding_service
        reset_embedding_service()

    def teardown_method(self):
        from embeddings.service import reset_embedding_service
        reset_embedding_service()

    def test_local_only_allows_gguf(self):
        with patch.dict("os.environ", {"EMBEDDING_LOCAL_ONLY": "1", "EMBED_PROVIDER": "gguf"}):
            from embeddings.service import _enforce_local_only
            _enforce_local_only()  # should not raise

    def test_local_only_allows_local(self):
        with patch.dict("os.environ", {"EMBEDDING_LOCAL_ONLY": "1", "EMBED_PROVIDER": "local"}):
            from embeddings.service import _enforce_local_only
            _enforce_local_only()  # should not raise

    def test_local_only_rejects_remote(self):
        with patch.dict("os.environ", {"EMBEDDING_LOCAL_ONLY": "1", "EMBED_PROVIDER": "openai"}):
            from embeddings.service import _enforce_local_only
            with pytest.raises(RuntimeError, match="remote embedding provider"):
                _enforce_local_only()

    def test_local_only_disabled_allows_remote(self):
        with patch.dict("os.environ", {"EMBEDDING_LOCAL_ONLY": "0", "EMBED_PROVIDER": "openai"}):
            from embeddings.service import _enforce_local_only
            _enforce_local_only()  # should not raise


# ═══════════════════════════════════════════════════════════════════
# Section 11: Error classification (job_executor)
# ═══════════════════════════════════════════════════════════════════


class TestErrorClassification:

    def test_429_classified_as_transient(self):
        from rag.job_executor import ErrorCategory, classify_error
        cat = classify_error(Exception("HTTP 429 Too Many Requests"))
        assert cat == ErrorCategory.TRANSIENT

    def test_rate_limit_error_classified_as_transient(self):
        from rag.job_executor import ErrorCategory, classify_error
        cat = classify_error(RateLimitError("breaker open"))
        assert cat == ErrorCategory.TRANSIENT

    def test_502_classified_as_transient(self):
        from rag.job_executor import ErrorCategory, classify_error
        cat = classify_error(Exception("502 Bad Gateway"))
        assert cat == ErrorCategory.TRANSIENT

    def test_401_classified_as_permanent(self):
        from rag.job_executor import ErrorCategory, classify_error
        cat = classify_error(Exception("HTTP 401 Unauthorized"))
        assert cat == ErrorCategory.PERMANENT


# ═══════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
