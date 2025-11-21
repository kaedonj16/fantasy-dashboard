# dashboard_services/sleeper_bulk_stats.py

import json
import os
import requests
import time
from typing import Any, Dict, List

SLEEPER_BASE = "https://api.sleeper.app"

WEEK_CACHE_TTL = 24 * 60 * 60  # 1 day, adjust as you want
CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "cache", "sleeper_stats")
os.makedirs(CACHE_DIR, exist_ok=True)


def _week_cache_path(season: int, week: int) -> str:
    return os.path.join(CACHE_DIR, f"sleeper_stats_{season}_week_{week}.json")


def _is_cache_fresh(path: str) -> bool:
    if not os.path.exists(path):
        return False
    mtime = os.path.getmtime(path)
    return (time.time() - mtime) <= WEEK_CACHE_TTL


def fetch_week_stats(season: int, week: int, *, force_refresh: bool = False) -> List[Dict[str, Any]]:
    """
    Fetch stats for a single NFL week from Sleeper.

    Uses a JSON file cache per (season, week).
    If the cache is corrupt or not a list, it is ignored and overwritten.
    """
    cache_path = _week_cache_path(season, week)

    # 1) Try cache
    if not force_refresh and _is_cache_fresh(cache_path):
        try:
            with open(cache_path, "r") as f:
                data = json.load(f)

            if isinstance(data, dict):
                return data

            print(f"[sleeper_bulk_stats] Cache for {season} wk{week} is not a dict, got {type(data)}. Refetching...")
        except json.JSONDecodeError as e:
            print(f"[sleeper_bulk_stats] Corrupt JSON cache for {season} wk{week} at {cache_path}: {e}. Deleting...")
        except Exception as e:
            print(f"[sleeper_bulk_stats] Error reading cache for {season} wk{week}: {e}. Refetching...")

        # If we get here, cache is bad → delete it so we can rewrite
        try:
            os.remove(cache_path)
        except OSError:
            pass

    # 2) Fetch from Sleeper API
    url = f"{SLEEPER_BASE}/v1/stats/nfl/regular/{season}/{week}"  # or the correct stats endpoint
    resp = requests.get(url, params={"season": season}, timeout=20)
    resp.raise_for_status()
    data = resp.json()

    if not isinstance(data, dict):
        print(f"[sleeper_bulk_stats] WARNING: Sleeper returned non-dict for {season} wk{week}: {type(data)}")
        # You can choose to return [] or wrap it; I'd do empty to avoid blowing up downstream
        data = []

    # 3) Save clean JSON to cache
    try:
        with open(cache_path, "w") as f:
            json.dump(data, f)
    except Exception as e:
        print(f"[sleeper_bulk_stats] Failed to write cache for {season} wk{week} at {cache_path}: {e}")

    return data


def fetch_season_stats(season: int, weeks: List[int]) -> Dict[int, List[Dict[str, Any]]]:
    """
    Fetch stats for multiple weeks in a season.

    Returns: {week: [stats_dict, ...], ...}
    """
    out: Dict[int, List[Dict[str, Any]]] = {}
    for w in weeks:
        out[w] = fetch_week_stats(season, w)
    return out
