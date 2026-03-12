"""Health monitoring — tracks API calls, errors, rate limits, and system health."""

import logging
import time
from typing import Optional

log = logging.getLogger(__name__)


class HealthMonitor:
    """Singleton for tracking API health and system status."""

    _instance: Optional["HealthMonitor"] = None

    def __new__(cls) -> "HealthMonitor":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return

        self.api_calls_total: int = 0
        self.api_errors: list[dict] = []  # Keep last 20 errors
        self.rate_limits_hit: int = 0
        self.last_rate_limit_at: Optional[float] = None
        self.last_error: Optional[str] = None
        self.session_start: float = time.time()
        self._max_errors: int = 20
        self._initialized = True

    def record_api_call(self) -> None:
        """Record a successful API call."""
        self.api_calls_total += 1
        log.debug(f"API call recorded. Total: {self.api_calls_total}")

    def record_error(self, error_type: str, message: str) -> None:
        """
        Record an API error.

        Args:
            error_type: Type of error (e.g. "ConnectError", "APIError", "Timeout")
            message: Error message
        """
        self.last_error = message
        error_entry = {
            "timestamp": time.time(),
            "type": error_type,
            "message": message[:200],  # Keep message reasonably short
        }
        self.api_errors.append(error_entry)

        # Keep only last 20 errors
        if len(self.api_errors) > self._max_errors:
            self.api_errors = self.api_errors[-self._max_errors :]

        log.warning(f"API error recorded: {error_type} - {message[:100]}")

    def record_rate_limit(self, retry_after: Optional[float] = None) -> None:
        """
        Record a rate limit response (HTTP 429).

        Args:
            retry_after: Optional seconds to wait before retry
        """
        self.rate_limits_hit += 1
        self.last_rate_limit_at = time.time()
        self.record_error(
            "RateLimit",
            f"Rate limited (retry after {retry_after}s)" if retry_after else "Rate limited",
        )
        log.warning(f"Rate limit hit. Total: {self.rate_limits_hit}")

    def reset_session(self) -> None:
        """Reset counters for a new session."""
        self.api_calls_total = 0
        self.api_errors.clear()
        self.rate_limits_hit = 0
        self.last_rate_limit_at = None
        self.last_error = None
        self.session_start = time.time()
        log.info("Health monitor reset for new session")

    def get_summary(self) -> dict:
        """
        Get current health status summary.

        Returns:
            Dict with keys: api_calls_total, api_errors, rate_limits_hit,
                           last_rate_limit_at, last_error, session_start,
                           session_duration_secs
        """
        return {
            "api_calls_total": self.api_calls_total,
            "api_errors": list(self.api_errors),  # Copy to avoid mutation
            "rate_limits_hit": self.rate_limits_hit,
            "last_rate_limit_at": self.last_rate_limit_at,
            "last_error": self.last_error,
            "session_start": self.session_start,
            "session_duration_secs": time.time() - self.session_start,
        }


def get_health_monitor() -> HealthMonitor:
    """Get the global HealthMonitor singleton."""
    return HealthMonitor()
