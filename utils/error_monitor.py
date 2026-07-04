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

_LOCK = threading.Lock()
_COUNTS: dict = {}          # key -> {"count", "level", "logger", "sample", "last_ts"}
_STARTED_AT = time.time()
_MAX_KEYS = 500             # hard cap so a pathological message flood can't grow unbounded
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
            with _LOCK:
                entry = _COUNTS.get(key)
                if entry is None:
                    if len(_COUNTS) >= _MAX_KEYS:
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


def snapshot(limit: int = 100) -> dict:
    """Current error counts, most frequent first."""
    with _LOCK:
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
        "since": _STARTED_AT,
        "uptime_seconds": round(time.time() - _STARTED_AT, 1),
        "distinct_errors": len(items),
        "errors": items[: max(1, int(limit))],
    }


def reset() -> None:
    """Clear all counts (used by tests and the admin endpoint)."""
    with _LOCK:
        _COUNTS.clear()
