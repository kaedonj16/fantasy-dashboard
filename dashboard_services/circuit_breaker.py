"""
Simple in-process circuit breaker for external HTTP APIs.

Usage:
    breaker = CircuitBreaker("tank01", failure_threshold=5, reset_timeout=300)

    if breaker.is_open():
        return None  # fail fast

    try:
        result = requests.get(url, timeout=10)
        breaker.record_success()
        return result
    except Exception as e:
        breaker.record_failure()
        raise

States:
  CLOSED  - normal operation; failures are counted
  OPEN    - failing fast; no requests sent until reset_timeout seconds pass
  HALF    - one probe request allowed; closes on success, reopens on failure
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Dict

logger = logging.getLogger(__name__)

_CLOSED = "closed"
_OPEN   = "open"
_HALF   = "half-open"


class CircuitBreaker:
    """Thread-safe circuit breaker for a single named dependency."""

    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        reset_timeout: float = 300.0,  # seconds before moving OPEN → HALF-OPEN
    ):
        self.name              = name
        self.failure_threshold = failure_threshold
        self.reset_timeout     = reset_timeout

        self._state    = _CLOSED
        self._failures = 0
        self._opened_at: float = 0.0
        self._lock     = threading.Lock()

    # ── Public API ────────────────────────────────────────────────────────────

    def is_open(self) -> bool:
        """Return True when the caller should skip the request (fail fast)."""
        with self._lock:
            if self._state == _CLOSED:
                return False
            if self._state == _OPEN:
                if time.monotonic() - self._opened_at >= self.reset_timeout:
                    self._state = _HALF
                    logger.info("[circuit-breaker:%s] → HALF-OPEN (probe allowed)", self.name)
                    return False   # allow the probe request through
                return True
            # HALF-OPEN: allow exactly one probe
            return False

    def record_success(self) -> None:
        with self._lock:
            if self._state in (_HALF, _OPEN):
                logger.info("[circuit-breaker:%s] → CLOSED (recovered)", self.name)
            self._state    = _CLOSED
            self._failures = 0

    def record_failure(self) -> None:
        with self._lock:
            self._failures += 1
            if self._state == _HALF or self._failures >= self.failure_threshold:
                self._state     = _OPEN
                self._opened_at = time.monotonic()
                logger.warning(
                    "[circuit-breaker:%s] → OPEN after %d failure(s); "
                    "will retry in %.0fs",
                    self.name, self._failures, self.reset_timeout,
                )

    @property
    def state(self) -> str:
        return self._state


# ── Module-level breaker registry ─────────────────────────────────────────────
_registry: Dict[str, CircuitBreaker] = {}
_registry_lock = threading.Lock()


def get_breaker(name: str, **kwargs) -> CircuitBreaker:
    """Return (or create) a named circuit breaker. Kwargs forwarded to __init__."""
    with _registry_lock:
        if name not in _registry:
            _registry[name] = CircuitBreaker(name, **kwargs)
        return _registry[name]
