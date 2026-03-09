"""RateGuard — 全域速率控制、熔斷、退避。

保護所有 outbound API 呼叫：
- Token bucket：每分鐘請求上限
- Semaphore：併發上限
- Circuit breaker：連續超限後暫停佇列
- 指數退避 + 抖動：transient 錯誤統一策略
- 結構化事件 log
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import AsyncIterator

log = logging.getLogger(__name__)


# ── 熔斷狀態 ────────────────────────────────────────────────────

class BreakerState(Enum):
    CLOSED = "closed"        # 正常
    OPEN = "open"            # 熔斷中
    HALF_OPEN = "half_open"  # 試探恢復


@dataclass
class _BreakerInfo:
    state: BreakerState = BreakerState.CLOSED
    consecutive_failures: int = 0
    open_until: float = 0.0  # monotonic timestamp
    last_failure_reason: str = ""


# ── 結構化事件 ──────────────────────────────────────────────────

@dataclass
class RequestEvent:
    request_id: str
    endpoint: str
    status: str              # "ok" | "error" | "throttled" | "breaker_open"
    retry_count: int = 0
    breaker_state: str = "closed"
    queue_depth: int = 0
    latency_ms: float = 0.0
    error: str = ""

    def log_line(self) -> str:
        return (
            f"[RateGuard] req={self.request_id} endpoint={self.endpoint} "
            f"status={self.status} retry={self.retry_count} "
            f"breaker={self.breaker_state} queue={self.queue_depth} "
            f"latency={self.latency_ms:.0f}ms"
            + (f" error={self.error}" if self.error else "")
        )


# ── Token Bucket ────────────────────────────────────────────────

class TokenBucket:
    """Thread-safe async token bucket for RPM limiting."""

    def __init__(self, rpm: int) -> None:
        self._rpm = max(rpm, 1)
        self._interval = 60.0 / self._rpm
        self._tokens = float(self._rpm)
        self._max_tokens = float(self._rpm)
        self._last_refill = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self, timeout: float = 30.0) -> bool:
        """等待直到取得 token。超時回傳 False。"""
        deadline = time.monotonic() + timeout
        while True:
            async with self._lock:
                self._refill()
                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    return True
            if time.monotonic() >= deadline:
                return False
            await asyncio.sleep(min(self._interval, 0.5))

    def _refill(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last_refill
        self._tokens = min(self._max_tokens, self._tokens + elapsed / self._interval)
        self._last_refill = now

    @property
    def available(self) -> int:
        return int(self._tokens)


# ── RateGuard 主體 ──────────────────────────────────────────────

class RateGuard:
    """全域速率控制。

    使用方式：
        guard = get_rate_guard()
        async with guard.acquire("chat/completions"):
            resp = await client.post(...)

    或帶自動重試：
        result = await guard.call_with_retry(
            "chat/completions",
            callable_fn,
        )
    """

    def __init__(
        self,
        rpm: int = 20,
        max_concurrency: int = 1,
        breaker_threshold: int = 5,
        breaker_cooldown: float = 600.0,  # 10 分鐘
    ) -> None:
        self._bucket = TokenBucket(rpm)
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._breaker = _BreakerInfo()
        self._breaker_threshold = breaker_threshold
        self._breaker_cooldown = breaker_cooldown
        self._queue_depth = 0
        self._events: list[RequestEvent] = []

    # ── 公開介面 ────────────────────────────────────────────

    @asynccontextmanager
    async def acquire(self, endpoint: str) -> AsyncIterator[str]:
        """取得呼叫權限。回傳 request_id。

        Raises RateLimitError if breaker is open or bucket timeout.
        """
        req_id = str(uuid.uuid4())[:8]
        self._queue_depth += 1
        event = RequestEvent(
            request_id=req_id,
            endpoint=endpoint,
            status="pending",
            breaker_state=self._breaker.state.value,
            queue_depth=self._queue_depth,
        )

        try:
            # 檢查熔斷器
            self._check_breaker(event)

            # 等待 token
            acquired = await self._bucket.acquire(timeout=30.0)
            if not acquired:
                event.status = "throttled"
                log.warning(event.log_line())
                self._events.append(event)
                raise RateLimitError("RPM throttle timeout")

            # 等待併發 slot
            await self._semaphore.acquire()
            t0 = time.monotonic()
            try:
                yield req_id
                # 成功
                event.status = "ok"
                event.latency_ms = (time.monotonic() - t0) * 1000
                self._on_success()
            except Exception as exc:
                event.status = "error"
                event.error = str(exc)[:200]
                event.latency_ms = (time.monotonic() - t0) * 1000
                self._on_failure(str(exc))
                raise
            finally:
                self._semaphore.release()
        finally:
            self._queue_depth -= 1
            event.queue_depth = self._queue_depth
            log.info(event.log_line())
            self._events.append(event)

    async def call_with_retry(
        self,
        endpoint: str,
        fn,
        *args,
        max_retries: int = 3,
        **kwargs,
    ):
        """帶指數退避 + 抖動的重試呼叫。"""
        last_exc = None
        for attempt in range(max_retries + 1):
            try:
                async with self.acquire(endpoint) as req_id:
                    return await fn(*args, **kwargs)
            except RateLimitError:
                raise  # 熔斷/throttle 不重試
            except Exception as exc:
                last_exc = exc
                if attempt < max_retries and self._is_transient(exc):
                    delay = self._backoff_delay(attempt)
                    log.warning(
                        "[RateGuard] Retry %d/%d for %s in %.1fs: %s",
                        attempt + 1, max_retries, endpoint, delay, exc,
                    )
                    await asyncio.sleep(delay)
                else:
                    raise
        raise last_exc  # should not reach

    # ── 熔斷器 ─────────────────────────────────────────────

    def _check_breaker(self, event: RequestEvent) -> None:
        """檢查熔斷器狀態。OPEN 時拒絕請求。"""
        b = self._breaker
        if b.state == BreakerState.OPEN:
            remaining = b.open_until - time.monotonic()
            if remaining > 0:
                event.status = "breaker_open"
                event.breaker_state = "open"
                event.error = f"breaker open, {remaining:.0f}s remaining"
                log.warning(event.log_line())
                self._events.append(event)
                raise RateLimitError(
                    f"Circuit breaker OPEN — 暫停 {remaining:.0f} 秒後自動恢復。"
                    f" 原因: {b.last_failure_reason}"
                )
            # 冷卻期結束 → HALF_OPEN
            b.state = BreakerState.HALF_OPEN
            log.info("[RateGuard] Breaker → HALF_OPEN (試探恢復)")

        event.breaker_state = b.state.value

    def _on_success(self) -> None:
        b = self._breaker
        if b.state == BreakerState.HALF_OPEN:
            log.info("[RateGuard] Breaker → CLOSED (恢復正常)")
        b.state = BreakerState.CLOSED
        b.consecutive_failures = 0

    def _on_failure(self, reason: str) -> None:
        b = self._breaker
        b.consecutive_failures += 1
        b.last_failure_reason = reason[:200]

        if b.consecutive_failures >= self._breaker_threshold:
            b.state = BreakerState.OPEN
            b.open_until = time.monotonic() + self._breaker_cooldown
            log.error(
                "[RateGuard] Breaker → OPEN (連續 %d 次失敗，暫停 %.0f 秒): %s",
                b.consecutive_failures, self._breaker_cooldown, reason[:100],
            )

    # ── 退避 ───────────────────────────────────────────────

    @staticmethod
    def _backoff_delay(attempt: int) -> float:
        """指數退避 + 抖動。"""
        base = min(2 ** attempt, 60)
        jitter = random.uniform(0, base * 0.5)
        return base + jitter

    @staticmethod
    def _is_transient(exc: Exception) -> bool:
        """判斷是否為 transient 錯誤。"""
        err = str(exc).lower()
        return any(k in err for k in (
            "429", "rate limit", "timeout", "connection",
            "502", "503", "504", "overloaded",
        ))

    # ── 狀態查詢 ───────────────────────────────────────────

    @property
    def breaker_state(self) -> BreakerState:
        return self._breaker.state

    @property
    def breaker_remaining_seconds(self) -> float:
        if self._breaker.state != BreakerState.OPEN:
            return 0.0
        return max(0.0, self._breaker.open_until - time.monotonic())

    @property
    def queue_depth(self) -> int:
        return self._queue_depth

    @property
    def available_tokens(self) -> int:
        return self._bucket.available

    def status(self) -> dict:
        return {
            "breaker": self._breaker.state.value,
            "breaker_remaining_s": round(self.breaker_remaining_seconds, 1),
            "queue_depth": self._queue_depth,
            "available_rpm_tokens": self.available_tokens,
            "consecutive_failures": self._breaker.consecutive_failures,
            "total_events": len(self._events),
        }

    def recent_events(self, n: int = 20) -> list[dict]:
        return [
            {
                "request_id": e.request_id,
                "endpoint": e.endpoint,
                "status": e.status,
                "retry": e.retry_count,
                "breaker": e.breaker_state,
                "latency_ms": round(e.latency_ms, 1),
                "error": e.error,
            }
            for e in self._events[-n:]
        ]


# ── 例外 ────────────────────────────────────────────────────────

class RateLimitError(Exception):
    """速率超限或熔斷器開啟。"""


# ── 全域單例 ────────────────────────────────────────────────────

_guard: RateGuard | None = None


def get_rate_guard() -> RateGuard:
    """取得全域 RateGuard 單例。"""
    global _guard
    if _guard is None:
        import os
        rpm = int(os.environ.get("RATE_GUARD_RPM", "100"))
        concurrency = int(os.environ.get("RATE_GUARD_CONCURRENCY", "8"))
        _guard = RateGuard(rpm=rpm, max_concurrency=concurrency)
    return _guard


def reset_rate_guard() -> None:
    global _guard
    _guard = None
