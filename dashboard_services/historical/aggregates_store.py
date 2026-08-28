"""Request-path loader for historical_profile_aggregates.json.

json + pathlib only. Flask / pandas / parquet stay out. The in-memory cache
keys on file mtime so a cron rebuild is visible on the next request.
"""
from __future__ import annotations

import json
from typing import Any, Mapping, Optional

from utils.paths import PLAYER_HISTORY_DIR

PROFILE_PATH = PLAYER_HISTORY_DIR / "historical_profile_aggregates.json"
CAREER_PATH_OVERLAY_PATH = PLAYER_HISTORY_DIR / "career_path_overlay.json"

_CACHE: dict[str, Any] = {"mtime": None, "data": None}


def _merge_career_path_overlay(data: dict, overlay: Mapping[str, Any]) -> dict:
    """Stamp prior top-12 counts and bounce-back rates onto loaded aggregates."""
    if not overlay:
        return data
    pre = data.get("preseason_profiles") if isinstance(data.get("preseason_profiles"), dict) else {}
    by_player = pre.get("by_player") if isinstance(pre.get("by_player"), dict) else {}
    counts = overlay.get("prior_top12_count") if isinstance(overlay.get("prior_top12_count"), dict) else {}
    for pid, count in counts.items():
        rec = by_player.get(str(pid))
        if isinstance(rec, dict) and count is not None:
            rec["prior_top12_count"] = count
    repeat = data.get("repeat_and_breakout") if isinstance(data.get("repeat_and_breakout"), dict) else {}
    bounce = overlay.get("bounce_back") if isinstance(overlay.get("bounce_back"), dict) else {}
    for pos, block in bounce.items():
        if not isinstance(block, dict):
            continue
        dest = repeat.get(pos)
        if isinstance(dest, dict):
            dest.update(block)
    return data


def load_profile_aggregates(*, path: Optional[Any] = None) -> dict:
    """Return the precomputed JSON, or ``{}`` when the file is missing."""
    target = path if path is not None else PROFILE_PATH
    try:
        mtime = target.stat().st_mtime
    except OSError:
        return {}
    overlay_mtime = None
    if path is None:
        try:
            overlay_mtime = CAREER_PATH_OVERLAY_PATH.stat().st_mtime
        except OSError:
            overlay_mtime = None
    cache_key = (mtime, overlay_mtime)
    cached = _CACHE.get("data")
    if cached is not None and _CACHE.get("mtime") == cache_key and path is None:
        return cached
    try:
        data = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(data, dict):
        return {}
    if path is None:
        try:
            overlay = json.loads(CAREER_PATH_OVERLAY_PATH.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            overlay = {}
        if isinstance(overlay, dict) and overlay:
            _merge_career_path_overlay(data, overlay)
        _CACHE["mtime"] = cache_key
        _CACHE["data"] = data
    return data


def stamp_historical_on_payload(payload: Mapping[str, Any]) -> dict:
    """Attach compact historical fields. Never raises into the request path."""
    out = dict(payload or {})
    try:
        from dashboard_services.historical.board import attach_historical_signals

        aggs = load_profile_aggregates()
        if not aggs:
            out["historical_available"] = False
            return out
        attach_historical_signals(out.get("players") or [], aggs)
        out["historical_available"] = True
        out["historical_phase"] = aggs.get("phase")
        out["historical_descriptive_only"] = True
    except Exception:
        out["historical_available"] = False
    return out
