"""Request-path loader for historical_profile_aggregates.json.

json + pathlib only. Flask / pandas / parquet stay out. The in-memory cache
keys on file mtime so a cron rebuild is visible on the next request.
"""
from __future__ import annotations

import json
from typing import Any, Mapping, Optional

from utils.paths import PLAYER_HISTORY_DIR

PROFILE_PATH = PLAYER_HISTORY_DIR / "historical_profile_aggregates.json"

_CACHE: dict[str, Any] = {"mtime": None, "data": None}


def load_profile_aggregates(*, path: Optional[Any] = None) -> dict:
    """Return the precomputed JSON, or ``{}`` when the file is missing."""
    target = path if path is not None else PROFILE_PATH
    try:
        mtime = target.stat().st_mtime
    except OSError:
        return {}
    cached = _CACHE.get("data")
    if cached is not None and _CACHE.get("mtime") == mtime and path is None:
        return cached
    try:
        data = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(data, dict):
        return {}
    if path is None:
        _CACHE["mtime"] = mtime
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
