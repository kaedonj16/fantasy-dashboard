"""
Compare Roles API endpoint.

Routes:
    GET /api/player-role/<sleeper_id>?season=<int>&window=season|last4|lastseason

Returns the role classification, target map, gap map, QB rushing split,
timeline and comparables assembled by dashboard_services.player_roles.
Responses are cached in-process (bounded, ~15 min TTL) because a payload
build scans the position pool and the season's play-by-play once.
"""
from __future__ import annotations

import logging
import threading
import time
from datetime import datetime

from flask import Blueprint, jsonify, request

logger = logging.getLogger(__name__)

player_roles_bp = Blueprint("player_roles", __name__)

_CACHE: dict = {}  # key -> (ts, payload)
_CACHE_TTL = 900
_CACHE_MAX = 64
_CACHE_LOCK = threading.Lock()


def _cache_get(key):
    with _CACHE_LOCK:
        hit = _CACHE.get(key)
        if hit and time.time() - hit[0] < _CACHE_TTL:
            return hit[1]
    return None


def _cache_put(key, payload) -> None:
    with _CACHE_LOCK:
        _CACHE[key] = (time.time(), payload)
        while len(_CACHE) > _CACHE_MAX:
            oldest = min(_CACHE, key=lambda k: _CACHE[k][0])
            del _CACHE[oldest]


def clear_cache() -> None:
    with _CACHE_LOCK:
        _CACHE.clear()


@player_roles_bp.route("/api/player-role/<sleeper_id>")
def api_player_role(sleeper_id: str):
    season_raw = (request.args.get("season") or "").strip()
    if season_raw.isdigit():
        season = int(season_raw)
    else:
        season = datetime.now().year
    window = (request.args.get("window") or "season").strip().lower()
    if window not in ("season", "last4", "lastseason"):
        window = "season"

    key = (str(sleeper_id), season, window)
    cached = _cache_get(key)
    if cached is not None:
        return jsonify(cached)
    try:
        from dashboard_services.player_roles import build_player_payload

        payload = build_player_payload(str(sleeper_id), season, window)
    except Exception:
        logger.exception("[player-role] build failed for %s", sleeper_id)
        return jsonify({"player_id": str(sleeper_id),
                        "error": "Failed to build player role"}), 500
    if payload is None:
        return jsonify({"player_id": str(sleeper_id),
                        "error": "No role data for this player"}), 404
    _cache_put(key, payload)
    resp = jsonify(payload)
    resp.headers["Cache-Control"] = "private, max-age=300"
    return resp
