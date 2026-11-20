"""
Compare Roles API endpoint.

Routes:
    GET /api/player-role/<sleeper_id>?season=<int>&window=season|last4|lastseason
    GET /api/player-roles/labels?season=<int>&position=QB|RB|WR|TE

The single-player route returns the role classification, target map,
gap map, QB rushing split, timeline and comparables assembled by
dashboard_services.player_roles. Responses are cached in-process
(bounded, ~15 min TTL) because a payload build scans the position pool
and the season's play-by-play once.

The labels route is the bulk companion for row decoration (Start/Sit
board, Rankings): one call returns {sleeper_id: {job, security}} for
the whole pool, season window only, backed by the engine's own bounded
labels cache -- never one request per row.
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
    started = time.time()
    try:
        from dashboard_services.player_roles import build_player_payload

        payload = build_player_payload(str(sleeper_id), season, window)
    except Exception:
        logger.exception("[player-role] build failed for %s", sleeper_id)
        return jsonify({"player_id": str(sleeper_id),
                        "error": "Failed to build player role",
                        "code": "role_build_failed"}), 500
    logger.info("[player-role] built %s season=%s window=%s in %.2fs",
                sleeper_id, season, window, time.time() - started)
    if payload is None:
        return jsonify({"player_id": str(sleeper_id),
                        "error": "No role data for this player",
                        "code": "no_role_data"}), 404
    _cache_put(key, payload)
    resp = jsonify(payload)
    resp.headers["Cache-Control"] = "private, max-age=300"
    return resp


@player_roles_bp.route("/api/player-roles/labels")
def api_player_role_labels():
    season_raw = (request.args.get("season") or "").strip()
    if season_raw.isdigit():
        season = int(season_raw)
    else:
        season = datetime.now().year
    position = (request.args.get("position") or "").strip().upper()
    if position not in ("QB", "RB", "WR", "TE"):
        position = ""
    started = time.time()
    try:
        from dashboard_services.player_roles import get_role_labels

        all_labels = get_role_labels(season)
    except Exception:
        logger.exception("[player-role-labels] build failed season=%s", season)
        return jsonify({"season": season, "window": "season",
                        "error": "Failed to build role labels",
                        "code": "role_labels_build_failed"}), 500
    labels = {}
    for pid, entry in (all_labels or {}).items():
        if position and entry.get("pos") != position:
            continue
        labels[pid] = {"job": entry.get("job") or "",
                       "security": entry.get("security") or ""}
    logger.info("[player-role-labels] served season=%s position=%s "
                "labels=%d in %.2fs", season, position or "ALL",
                len(labels), time.time() - started)
    resp = jsonify({"season": season, "window": "season", "labels": labels})
    resp.headers["Cache-Control"] = "private, max-age=300"
    return resp
