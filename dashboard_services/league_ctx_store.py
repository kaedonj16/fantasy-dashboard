"""Cross-worker persistence for built league contexts via Redis.

``build_league_context`` costs 7-28s of provider fan-out per league, but the
result used to live only in each gunicorn worker's process-local
``DASHBOARD_CACHE`` -- so the build was repaid per worker, per worker recycle
(``--max-requests``), per deploy, and per cache eviction. This module stores
the league-specific slice of a built context in the already-provisioned Redis
so a worker that misses its local cache can hydrate from a sibling's build
instead of rebuilding.

Trust model: Redis is internal-only (see render.yaml) and these keys are
written exclusively by the app itself, right after a successful build. The
payload is a pickled envelope because the context carries pandas DataFrames
(``df_weekly``/``team_stats``/``injury_df``/``activity_df``), which JSON
cannot represent. League-independent globals (the ~38MB players payload,
``players_map``, indexes, the model value table, rookie rankings) are NOT
stored -- the loading worker reattaches its own copies.

Fail-soft by contract: every function here returns ``None``/``False`` instead
of raising. Redis down, slow, corrupt, unconfigured, or disabled via
``LEAGUE_CTX_REDIS=0`` all degrade to the pre-existing local build path.
"""
from __future__ import annotations

import hashlib
import logging
import os
import pickle
import threading
from typing import Optional

logger = logging.getLogger(__name__)

ENVELOPE_VERSION = 1
DEFAULT_MAX_BYTES = 16 * 1024 * 1024

_LOGGED_ONCE: set = set()
_LOGGED_LOCK = threading.Lock()


def _log_once(reason: str, message: str) -> None:
    with _LOGGED_LOCK:
        if reason in _LOGGED_ONCE:
            return
        _LOGGED_ONCE.add(reason)
    logger.info("[league-ctx-redis] %s", message)


def enabled() -> bool:
    """Kill-switch + configuration gate, checked on every call."""
    if os.getenv("LEAGUE_CTX_REDIS", "1").strip() == "0":
        return False
    return bool((os.environ.get("REDIS_URL") or "").strip())


def max_bytes() -> int:
    try:
        return max(1024, int(os.getenv("LEAGUE_CTX_REDIS_MAX_BYTES", str(DEFAULT_MAX_BYTES))))
    except (TypeError, ValueError):
        return DEFAULT_MAX_BYTES


def ctx_key(platform, season, league_id) -> str:
    raw = f"{str(platform).lower()}\0{int(season)}\0{league_id}".encode()
    return "league_ctx:" + hashlib.sha256(raw).hexdigest()[:32]


def _redis_client():
    """Fresh client per operation (fork-safe), mirroring espn_draft_relay.

    Timeouts are bounded so a hung Redis can never stall a request for long;
    the payload can be a few MB, so the socket timeout is a bit roomier than
    the relay's.
    """
    url = (os.environ.get("REDIS_URL") or "").strip()
    if not url:
        return None
    try:
        import redis  # type: ignore

        return redis.from_url(url, socket_timeout=3.0, socket_connect_timeout=1.5)
    except Exception:
        return None


def dump_envelope(generation: int, built_at: float, ctx_slice: dict) -> Optional[bytes]:
    """Pickle the envelope, or None when unpicklable / over the size cap."""
    envelope = {
        "v": ENVELOPE_VERSION,
        "generation": int(generation or 0),
        "built_at": float(built_at or 0),
        "ctx": ctx_slice,
    }
    try:
        blob = pickle.dumps(envelope, protocol=4)
    except Exception as exc:
        _log_once(
            "pickle",
            f"ctx slice is not picklable ({type(exc).__name__}); skipping Redis store",
        )
        return None
    cap = max_bytes()
    if len(blob) > cap:
        _log_once(
            "size",
            f"ctx slice is {len(blob)} bytes, over the {cap}-byte cap; skipping Redis store",
        )
        return None
    return blob


def parse_envelope(raw) -> Optional[dict]:
    """Unpickle + structurally validate an envelope, or None."""
    if not raw:
        return None
    try:
        envelope = pickle.loads(raw)
    except Exception:
        return None
    if not isinstance(envelope, dict):
        return None
    if envelope.get("v") != ENVELOPE_VERSION:
        return None
    if not isinstance(envelope.get("ctx"), dict):
        return None
    try:
        float(envelope.get("built_at") or 0)
        int(envelope.get("generation") or 0)
    except (TypeError, ValueError):
        return None
    return envelope


def save(platform, season, league_id, blob: bytes, ttl_seconds: int) -> bool:
    client = _redis_client()
    if client is None:
        return False
    try:
        client.setex(ctx_key(platform, season, league_id), int(ttl_seconds), blob)
        return True
    except Exception as exc:
        _log_once("save", f"Redis store failed ({type(exc).__name__}); continuing without it")
        return False


def load(platform, season, league_id) -> Optional[dict]:
    client = _redis_client()
    if client is None:
        return None
    try:
        raw = client.get(ctx_key(platform, season, league_id))
    except Exception as exc:
        _log_once("load", f"Redis load failed ({type(exc).__name__}); continuing without it")
        return None
    return parse_envelope(raw)
