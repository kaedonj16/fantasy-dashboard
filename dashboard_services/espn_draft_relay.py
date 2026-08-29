"""ESPN live-draft relay: in-memory / Redis snapshot store for the desktop extension.

The Chrome/Edge extension observes the open ESPN draft room and POSTs picks to
``/api/draft/espn-relay`` with the Draft Room session (same-origin). Observe-only
— never talks to ESPN and never submits picks.
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from typing import Any, Dict, Mapping, Optional

logger = logging.getLogger(__name__)

# Draft-night window: long enough for a slow snake, short enough to rotate.
DEFAULT_TTL_SECONDS = 12 * 60 * 60
_STORE_LOCK = threading.Lock()
_STORE: Dict[str, Dict[str, Any]] = {}


def _store_key(league_id: str, season: int, *, platform: str = "espn") -> str:
    plat = (platform or "espn").strip().lower() or "espn"
    return f"{plat}_relay:{str(league_id).strip()}:{int(season)}"


def put_relay_snapshot(
    league_id: str,
    season: int,
    snapshot: Mapping[str, Any],
    *,
    source: str = "relay",
    platform: str = "espn",
) -> Dict[str, Any]:
    """Store the latest normalized (or raw-ready) relay payload for a draft."""
    key = _store_key(league_id, season, platform=platform)
    entry = {
        "league_id": str(league_id),
        "season": int(season),
        "source": str(source or "relay"),
        "updated_at": int(time.time()),
        "payload": dict(snapshot),
    }
    with _STORE_LOCK:
        _STORE[key] = entry
    _redis_put(key, entry)
    return entry


def get_relay_snapshot(
    league_id: str, season: int, *, platform: str = "espn"
) -> Optional[Dict[str, Any]]:
    key = _store_key(league_id, season, platform=platform)
    with _STORE_LOCK:
        local = _STORE.get(key)
    if local:
        return local
    return _redis_get(key)


def clear_relay_snapshot(league_id: str, season: int, *, platform: str = "espn") -> None:
    key = _store_key(league_id, season, platform=platform)
    with _STORE_LOCK:
        _STORE.pop(key, None)
    _redis_delete(key)


def _redis_client():
    url = (os.environ.get("REDIS_URL") or "").strip()
    if not url:
        return None
    try:
        import redis  # type: ignore
        return redis.from_url(url, socket_timeout=1.5, socket_connect_timeout=1.5)
    except Exception:
        return None


def _redis_put(key: str, entry: Mapping[str, Any]) -> None:
    client = _redis_client()
    if not client:
        return
    try:
        client.setex(key, DEFAULT_TTL_SECONDS, json.dumps(entry, separators=(",", ":")))
    except Exception as exc:
        logger.info("[espn-relay] redis put skipped error_type=%s", type(exc).__name__)


def _redis_get(key: str) -> Optional[Dict[str, Any]]:
    client = _redis_client()
    if not client:
        return None
    try:
        raw = client.get(key)
        if not raw:
            return None
        data = json.loads(raw)
        if isinstance(data, dict):
            with _STORE_LOCK:
                _STORE[key] = data
            return data
    except Exception as exc:
        logger.info("[espn-relay] redis get skipped error_type=%s", type(exc).__name__)
    return None


def _redis_delete(key: str) -> None:
    client = _redis_client()
    if not client:
        return
    try:
        client.delete(key)
    except Exception:
        pass


def merge_live_with_relay(
    live_payload: Mapping[str, Any],
    relay_entry: Optional[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Prefer the longer/fresher pick list between REST live and stored relay."""
    out = dict(live_payload or {})
    if not relay_entry or not isinstance(relay_entry.get("payload"), Mapping):
        return out
    relay = dict(relay_entry["payload"])
    live_picks = live_payload.get("picks") if isinstance(live_payload.get("picks"), list) else []
    relay_picks = relay.get("picks") if isinstance(relay.get("picks"), list) else []

    def _count(picks: list) -> int:
        n = 0
        for p in picks:
            if not isinstance(p, Mapping):
                continue
            pid = p.get("player_id") or p.get("external_player_id")
            if pid in (None, "", "0", "-1"):
                continue
            n += 1
        return n

    live_n = _count(live_picks)
    relay_n = _count(relay_picks)
    if relay_n > live_n or (relay_n == live_n and relay_n > 0 and live_n == 0):
        out["picks"] = relay_picks
        out["picks_observed"] = True
        out["live_detail_present"] = True
        out["relay_source"] = relay.get("source") or relay_entry.get("source") or "relay"
        out["relay_updated_at"] = relay_entry.get("updated_at")
        if relay.get("status"):
            out["status"] = relay["status"]
        if relay.get("in_progress") is not None:
            out["in_progress"] = relay["in_progress"]
        if relay.get("fingerprint"):
            out["fingerprint"] = relay["fingerprint"]
    return out
