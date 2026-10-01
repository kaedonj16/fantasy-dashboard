"""Cross-worker persistence for weekly Sleeper projections via Redis.

Weekly projections are league-independent: every league context in every
gunicorn worker needs the same ``{pid: entry}`` map for a (season, week).
The on-disk cache under ``cache/projections/`` is ephemeral on Render, so a
fresh instance (deploy, restart, OOM kill) used to refetch all 18 weeks from
Sleeper -- concurrently, once per worker / league build.  This module keeps
a JSON copy in the already-provisioned Redis so a cold instance can restore
the file cache instead of hammering Sleeper.

Unlike league contexts, projection maps are plain JSON (they are written to
disk as JSON today), so the payload needs no pickle: an envelope of
``{"saved_at": <float>, "data": {pid: entry, ...}}``.  The loading side
writes the restored data back to the file cache and stamps the file mtime
with ``saved_at`` so all existing TTL/staleness logic applies unchanged.

Fail-soft by contract: every function here returns ``None``/``False``
instead of raising.  Redis down, slow, corrupt, unconfigured, or disabled
via ``PROJ_REDIS=0`` all degrade to the pre-existing fetch path.
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from typing import Optional

logger = logging.getLogger(__name__)

STORAGE_TTL_SECONDS = 30 * 24 * 3600  # 30 days; staleness is judged by mtime
DEFAULT_MAX_BYTES = 16 * 1024 * 1024

_LOGGED_ONCE: set = set()
_LOGGED_LOCK = threading.Lock()


def _log_once(reason: str, message: str) -> None:
    with _LOGGED_LOCK:
        if reason in _LOGGED_ONCE:
            return
        _LOGGED_ONCE.add(reason)
    logger.info("[proj-redis] %s", message)


def _disabled() -> bool:
    return os.getenv("PROJ_REDIS", "1").strip() == "0"


def enabled() -> bool:
    """Kill-switch + configuration gate, checked on every call."""
    if _disabled():
        return False
    return bool((os.environ.get("REDIS_URL") or "").strip())


def max_bytes() -> int:
    try:
        return max(1024, int(os.getenv("PROJ_REDIS_MAX_BYTES", str(DEFAULT_MAX_BYTES))))
    except (TypeError, ValueError):
        return DEFAULT_MAX_BYTES


def proj_key(season, week) -> str:
    return f"proj:{int(season)}:{int(week)}"


def _redis_client():
    """Fresh client per operation (fork-safe), mirroring league_ctx_store.

    Timeouts are bounded so a hung Redis can never stall a request for long.
    Returns None when REDIS_URL is unset or the client cannot be built; the
    kill switch is enforced by the callers so tests can substitute a fake
    client via this seam.
    """
    url = (os.environ.get("REDIS_URL") or "").strip()
    if not url:
        return None
    try:
        import redis  # type: ignore

        return redis.from_url(url, socket_timeout=3.0, socket_connect_timeout=1.5)
    except Exception:
        return None


def save(season, week, data) -> bool:
    """Persist a week projection map; False on any failure (never raises)."""
    if _disabled():
        return False
    if not isinstance(data, dict):
        return False
    envelope = {"saved_at": time.time(), "data": data}
    try:
        blob = json.dumps(envelope, ensure_ascii=False).encode("utf-8")
    except (TypeError, ValueError) as exc:
        _log_once(
            "serialize",
            f"projection map is not JSON-serializable ({type(exc).__name__}); "
            "skipping Redis store",
        )
        return False
    cap = max_bytes()
    if len(blob) > cap:
        _log_once(
            "size",
            f"projection map is {len(blob)} bytes, over the {cap}-byte cap; "
            "skipping Redis store",
        )
        return False
    client = _redis_client()
    if client is None:
        return False
    try:
        client.setex(proj_key(season, week), STORAGE_TTL_SECONDS, blob)
        return True
    except Exception as exc:
        _log_once("save", f"Redis store failed ({type(exc).__name__}); continuing without it")
        return False


def load(season, week) -> Optional[tuple]:
    """Return ``(saved_at, data)`` for a stored week map, or None.

    Corrupt or structurally invalid payloads are indistinguishable from a
    miss: the caller falls through to the normal fetch path.
    """
    if _disabled():
        return None
    client = _redis_client()
    if client is None:
        return None
    try:
        raw = client.get(proj_key(season, week))
    except Exception as exc:
        _log_once("load", f"Redis load failed ({type(exc).__name__}); continuing without it")
        return None
    if not raw:
        return None
    try:
        if isinstance(raw, (bytes, bytearray)):
            raw = raw.decode("utf-8")
        envelope = json.loads(raw)
    except (UnicodeDecodeError, ValueError):
        return None
    if not isinstance(envelope, dict):
        return None
    data = envelope.get("data")
    if not isinstance(data, dict):
        return None
    try:
        saved_at = float(envelope.get("saved_at") or 0)
    except (TypeError, ValueError):
        return None
    return (saved_at, data)
