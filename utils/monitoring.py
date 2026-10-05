"""Consolidated utils module: monitoring.

in-memory error and performance monitors

Merged from: utils/error_monitor.py, utils/perf_monitor.py.
Old import paths keep working via compatibility shims.
"""


# ======================================================================
# From utils/error_monitor.py
# ======================================================================

"""In-memory error counter so silent degradation is visible in production.

The app has hundreds of broad exception handlers that log and move on; when a
feature starts failing (a card silently disappearing, a background job dying)
nothing surfaces it. This module installs a logging handler that counts every
WARNING-or-higher record (and any record carrying exc_info) that reaches the
logging system, keyed by logger name + message shape, and exposes a snapshot
for an admin endpoint.

Limits: records logged below the logger's effective level (e.g. the many
logger.debug(..., exc_info=True) handlers under a default INFO level) never
reach any handler, so they are not counted. This monitors warnings and errors,
which is where genuine production degradation shows up.
"""
import logging
import threading
import time
from typing import List

_ERROR_LOCK = threading.Lock()
_COUNTS: dict = {}          # key -> {"count", "level", "logger", "sample", "last_ts"}
_ERROR_STARTED_AT = time.time()
_ERROR_MAX_KEYS = 500             # hard cap so a pathological message flood can't grow unbounded
_INSTALLED = False


def _key_for(record: logging.LogRecord) -> str:
    # Group by logger + level + the message template when available (record.msg
    # before %-interpolation), so "failed for league 123" and "... 456" collapse
    # into one bucket.
    msg = record.msg if isinstance(record.msg, str) else str(record.msg)
    return f"{record.name}|{record.levelname}|{msg[:160]}"


class ErrorCounterHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        try:
            if record.levelno < logging.WARNING and not record.exc_info:
                return
            key = _key_for(record)
            with _ERROR_LOCK:
                entry = _COUNTS.get(key)
                if entry is None:
                    if len(_COUNTS) >= _ERROR_MAX_KEYS:
                        return
                    entry = {
                        "count": 0,
                        "level": record.levelname,
                        "logger": record.name,
                        "sample": record.getMessage()[:300],
                        "first_ts": time.time(),
                    }
                    _COUNTS[key] = entry
                entry["count"] += 1
                entry["last_ts"] = time.time()
        except Exception:
            # A monitoring handler must never take the app down.
            pass


def install() -> None:
    """Attach the counter to the root logger (idempotent)."""
    global _INSTALLED
    if _INSTALLED:
        return
    handler = ErrorCounterHandler(level=logging.DEBUG)
    logging.getLogger().addHandler(handler)
    _INSTALLED = True


def snapshot_errors(limit: int = 100) -> dict:
    """Current error counts, most frequent first."""
    with _ERROR_LOCK:
        items: List[dict] = [
            {
                "logger": e["logger"],
                "level": e["level"],
                "count": e["count"],
                "sample": e["sample"],
                "first_seen": e["first_ts"],
                "last_seen": e.get("last_ts", e["first_ts"]),
            }
            for e in _COUNTS.values()
        ]
    items.sort(key=lambda x: x["count"], reverse=True)
    return {
        "since": _ERROR_STARTED_AT,
        "uptime_seconds": round(time.time() - _ERROR_STARTED_AT, 1),
        "distinct_errors": len(items),
        "errors": items[: max(1, int(limit))],
    }


def reset_error_monitor() -> None:
    """Clear all counts (used by tests and the admin endpoint)."""
    with _ERROR_LOCK:
        _COUNTS.clear()


# ======================================================================
# From utils/perf_monitor.py
# ======================================================================

"""In-memory per-endpoint request timing so slow paths are visible in prod.

Companion to error_monitor: that surfaces failures, this surfaces slowness. The
app builds heavy per-league contexts on demand, so a cold cache after a deploy
can make an endpoint crawl with nothing to point at. This accumulates count /
total / max / slow-count per Flask endpoint and exposes a snapshot for an admin
endpoint, so "which routes are slow" stops being a guess.

Cheap by design: fixed-size dict keyed by endpoint, O(1) per request, no
per-request allocation beyond the key lookup. A request slower than SLOW_MS is
also counted separately and logged once by the caller.
"""
import threading
import time

_PERF_LOCK = threading.Lock()
_STATS: dict = {}          # endpoint -> {count,total_ms,max_ms,slow,err,last_ts}
_PERF_STARTED_AT = time.time()
_PERF_MAX_KEYS = 800            # cap so an attacker spamming unique paths can't grow it unbounded

# A request at or beyond this many milliseconds is "slow" (counted + logged).
SLOW_MS = 1500.0


def record(endpoint: str, method: str, duration_ms: float, status: int = 200) -> None:
    """Fold one finished request into the per-endpoint stats."""
    try:
        key = f"{method} {endpoint}"
        with _PERF_LOCK:
            e = _STATS.get(key)
            if e is None:
                if len(_STATS) >= _PERF_MAX_KEYS:
                    return
                e = {"count": 0, "total_ms": 0.0, "max_ms": 0.0, "slow": 0, "err": 0, "last_ts": 0.0}
                _STATS[key] = e
            e["count"] += 1
            e["total_ms"] += duration_ms
            if duration_ms > e["max_ms"]:
                e["max_ms"] = duration_ms
            if duration_ms >= SLOW_MS:
                e["slow"] += 1
            if status >= 500:
                e["err"] += 1
            e["last_ts"] = time.time()
    except Exception:
        # Monitoring must never break a request.
        pass


def snapshot_perf(limit: int = 100, sort: str = "total") -> dict:
    """Per-endpoint timing, slowest first.

    sort: 'total' (cumulative time, default), 'avg', 'max', or 'slow'.
    """
    with _PERF_LOCK:
        items: List[dict] = []
        for key, e in _STATS.items():
            count = e["count"] or 1
            items.append({
                "endpoint": key,
                "count": e["count"],
                "avg_ms": round(e["total_ms"] / count, 1),
                "max_ms": round(e["max_ms"], 1),
                "total_ms": round(e["total_ms"], 1),
                "slow_count": e["slow"],
                "error_count": e["err"],
                "last_seen": e["last_ts"],
            })
    keyfn = {
        "avg": lambda x: x["avg_ms"],
        "max": lambda x: x["max_ms"],
        "slow": lambda x: x["slow_count"],
        "total": lambda x: x["total_ms"],
    }.get(sort, lambda x: x["total_ms"])
    items.sort(key=keyfn, reverse=True)
    return {
        "since": _PERF_STARTED_AT,
        "uptime_seconds": round(time.time() - _PERF_STARTED_AT, 1),
        "slow_threshold_ms": SLOW_MS,
        "distinct_endpoints": len(items),
        "endpoints": items[: max(1, int(limit))],
    }


def reset_perf_monitor() -> None:
    with _PERF_LOCK:
        _STATS.clear()
