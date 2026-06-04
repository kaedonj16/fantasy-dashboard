# dashboard_services/stats_fetcher.py
from __future__ import annotations

import json
import os
import time
from typing import Dict, Any, List

CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "cache", "season_stats")
os.makedirs(CACHE_DIR, exist_ok=True)

STATS_CACHE_TTL = 7 * 24 * 60 * 60  # 1 week


def _season_cache_path(season: int) -> str:
    return os.path.join(CACHE_DIR, f"season_{season}_weekly_stats.json")


def _load_season_from_disk(season: int) -> Dict[str, Any] | None:
    path = _season_cache_path(season)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r") as f:
            payload = json.load(f)
        ts = payload.get("ts")
        data = payload.get("data")
        if not ts or data is None:
            return None
        if time.time() - ts > STATS_CACHE_TTL:
            return None
        return data
    except Exception:
        return None


def _save_season_to_disk(season: int, data: Dict[str, Any]) -> None:
    path = _season_cache_path(season)
    payload = {"ts": time.time(), "data": data}
    try:
        with open(path, "w") as f:
            json.dump(payload, f)
    except Exception as e:
        print(f"[stats_fetcher] Failed to save season {season} stats: {e}")


def fetch_weekly_stats_for_season(season: int) -> Dict[int, List[Dict[str, Any]]]:
    """
    Return {week: [player_stat_obj, ...]} for the given season.

    NOT YET IMPLEMENTED. The per-week fetch (Sleeper / FantasyPros / PFR) was
    never wired up. The previous version returned empty lists for every week
    and wrote those empties to the week-long disk cache, silently masking the
    missing data. Until the fetch is implemented this raises so callers fail
    loudly instead of receiving (and caching) empty stats.

    Disk-cache helpers (``_load_season_from_disk`` / ``_save_season_to_disk``)
    remain available for the real implementation.
    """
    cached = _load_season_from_disk(season)
    if cached:
        return cached

    raise NotImplementedError(
        "fetch_weekly_stats_for_season is not implemented; wire up the per-week "
        "stats source before calling it."
    )
