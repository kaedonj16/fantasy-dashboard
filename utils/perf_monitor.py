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
from typing import List

_LOCK = threading.Lock()
_STATS: dict = {}          # endpoint -> {count,total_ms,max_ms,slow,err,last_ts}
_STARTED_AT = time.time()
_MAX_KEYS = 800            # cap so an attacker spamming unique paths can't grow it unbounded

# A request at or beyond this many milliseconds is "slow" (counted + logged).
SLOW_MS = 1500.0


def record(endpoint: str, method: str, duration_ms: float, status: int = 200) -> None:
    """Fold one finished request into the per-endpoint stats."""
    try:
        key = f"{method} {endpoint}"
        with _LOCK:
            e = _STATS.get(key)
            if e is None:
                if len(_STATS) >= _MAX_KEYS:
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


def snapshot(limit: int = 100, sort: str = "total") -> dict:
    """Per-endpoint timing, slowest first.

    sort: 'total' (cumulative time, default), 'avg', 'max', or 'slow'.
    """
    with _LOCK:
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
        "since": _STARTED_AT,
        "uptime_seconds": round(time.time() - _STARTED_AT, 1),
        "slow_threshold_ms": SLOW_MS,
        "distinct_endpoints": len(items),
        "endpoints": items[: max(1, int(limit))],
    }


def reset() -> None:
    with _LOCK:
        _STATS.clear()
