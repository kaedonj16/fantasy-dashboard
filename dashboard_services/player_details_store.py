"""Cross-worker shared cache for player-modal payloads via Redis.

The /api/player-details payload, the per-player value-history rows, and the
modal's cheap eligibility / breakout flags used to live only in each gunicorn
worker's process-local dicts -- so every sibling worker, every worker recycle
(``--max-requests 1000`` + jitter), and every deploy repaid the full build.
That per-worker ceiling is why a "cached" modal open still measured 3.5-4.5s
in production: the first repeat on the other worker was a cold build, and
the value-history cache held only 6 players.

This module stores those small JSON payloads in the already-provisioned
Redis (see render.yaml REDIS_URL), following the fail-soft contract of
``dashboard_services/league_ctx_store.py``: Redis down, slow, corrupt,
unconfigured, or disabled via ``PLAYER_DETAILS_REDIS=0`` all degrade to the
pre-existing in-process caches. Payloads here are plain JSON (dicts, lists,
strings, numbers, bools, None) by construction -- never pickled.

Freshness is the caller's contract: every envelope carries the build
timestamp and the caller passes its own TTL; an entry older than that TTL
is a miss on read even if Redis has not expired it yet.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
from typing import Optional

logger = logging.getLogger(__name__)

ENVELOPE_VERSION = 1
DEFAULT_MAX_BYTES = 512 * 1024

_LOGGED_ONCE: set = set()
_LOGGED_LOCK = threading.Lock()


def _log_once(reason: str, message: str) -> None:
    with _LOGGED_LOCK:
        if reason in _LOGGED_ONCE:
            return
        _LOGGED_ONCE.add(reason)
    logger.info("[player-details-redis] %s", message)


def enabled() -> bool:
    """Kill-switch + configuration gate, checked on every call."""
    if os.getenv("PLAYER_DETAILS_REDIS", "1").strip() == "0":
        return False
    return bool((os.environ.get("REDIS_URL") or "").strip())


def max_bytes() -> int:
    try:
        return max(1024, int(os.getenv(
            "PLAYER_DETAILS_REDIS_MAX_BYTES", str(DEFAULT_MAX_BYTES))))
    except (TypeError, ValueError):
        return DEFAULT_MAX_BYTES


def make_key(namespace: str, key_parts) -> str:
    """Stable Redis key for a namespace + tuple-ish cache key.

    The parts are repr'd (not JSON'd) so tuples containing account ids,
    seasons and value-table timestamps hash exactly like the in-process
    dict keys they mirror.
    """
    raw = repr((str(namespace), tuple(key_parts))).encode()
    return "pd:" + str(namespace) + ":" + hashlib.sha256(raw).hexdigest()[:32]


def _redis_client():
    """Fresh client per operation (fork-safe), mirroring league_ctx_store.

    Timeouts are bounded so a hung Redis can never stall a modal open; the
    payloads are small (tens of KB), so a 2s socket timeout is generous.
    """
    url = (os.environ.get("REDIS_URL") or "").strip()
    if not url:
        return None
    try:
        import redis  # type: ignore

        return redis.from_url(url, socket_timeout=2.0, socket_connect_timeout=1.0)
    except Exception:
        return None


def get_json(namespace: str, key_parts, ttl_seconds: float) -> Optional[dict]:
    """Return the stored envelope ``{"ts": float, "payload": ...}`` or None.

    None on any miss: unconfigured, disabled, absent, corrupt, a version
    mismatch, or an entry older than ``ttl_seconds``.
    """
    if not enabled():
        return None
    client = _redis_client()
    if client is None:
        return None
    try:
        raw = client.get(make_key(namespace, key_parts))
    except Exception as exc:
        _log_once("get", f"Redis get failed ({type(exc).__name__}); "
                          "continuing without the shared cache")
        return None
    if not raw:
        return None
    try:
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        envelope = json.loads(raw)
    except Exception:
        return None
    if not isinstance(envelope, dict):
        return None
    if envelope.get("v") != ENVELOPE_VERSION or "payload" not in envelope:
        return None
    try:
        ts = float(envelope.get("ts") or 0)
    except (TypeError, ValueError):
        return None
    if ts <= 0:
        return None
    import time as _time

    if ttl_seconds is not None and _time.time() - ts >= float(ttl_seconds):
        return None
    return {"ts": ts, "payload": envelope["payload"]}


def set_json(namespace: str, key_parts, payload, ts: float,
             ttl_seconds: int) -> bool:
    """Store ``payload`` (JSON-round-trippable) with its build timestamp."""
    if not enabled():
        return False
    try:
        blob = json.dumps({
            "v": ENVELOPE_VERSION,
            "ts": float(ts or 0),
            "payload": payload,
        })
    except (TypeError, ValueError):
        return False
    if len(blob.encode("utf-8")) > max_bytes():
        _log_once("size", "payload over the shared-cache size cap; "
                          "skipping Redis store")
        return False
    client = _redis_client()
    if client is None:
        return False
    try:
        client.setex(make_key(namespace, key_parts), int(ttl_seconds), blob)
        return True
    except Exception as exc:
        _log_once("set", f"Redis set failed ({type(exc).__name__}); "
                         "continuing without the shared cache")
        return False
