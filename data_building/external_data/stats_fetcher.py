# dashboard_services/stats_fetcher.py
from __future__ import annotations

import json
import os
import requests  # or your preferred HTTP client
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

    This is intentionally generic. Inside the loop you hit Sleeper / FantasyPros / PFR.
    """
    cached = _load_season_from_disk(season)
    if cached:
        return cached

    season_stats: Dict[int, List[Dict[str, Any]]] = {}

    # TODO: adjust range(max_week+1) etc
    for week in range(1, 19):  # 1..18 regular season
        # ---- EXAMPLE (Sleeper-ish; adjust URL/params/keys to your actual source) ----
        # url = f"https://api.sleeper.app/v1/stats/nfl/regular/{week}"
        # resp = requests.get(url, timeout=20)
        # resp.raise_for_status()
        # week_data = resp.json()
        week_data: List[Dict[str, Any]] = []  # placeholder
        # --------------------------------------------------------------------------
        season_stats[week] = week_data

    _save_season_to_disk(season, season_stats)
    return season_stats
