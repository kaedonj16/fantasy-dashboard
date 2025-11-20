# dashboard_services/usage_cache.py

from __future__ import annotations

import json
import os
import time
from typing import Dict, Any, Optional

from dashboard_services.player_value import (
    build_tank01_usage_map_from_players_index,
)
from dashboard_services.utils import load_players_index

# ~1 week in seconds
USAGE_CACHE_TTL = 7 * 24 * 60 * 60

# In-memory cache: { season: { "ts": float, "data": {...} } }
_USAGE_CACHE: Dict[int, Dict[str, Any]] = {}

# Optional: disk cache (so you don't lose it on restart)
CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "cache")
os.makedirs(CACHE_DIR, exist_ok=True)


def _usage_cache_path(season: int) -> str:
    return os.path.join(CACHE_DIR, f"tank01_usage_season_{season}.json")


def _load_usage_from_disk(season: int) -> Optional[Dict[str, Any]]:
    path = _usage_cache_path(season)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r") as f:
            payload = json.load(f)
        ts = payload.get("ts")
        data = payload.get("data")
        if ts is None or data is None:
            return None

        # TEMP: log what we actually read
        print(f"[usage_cache] Loaded disk cache from {os.path.abspath(path)} with {len(data)} players")

        # If you want to force rebuild while debugging, just comment this TTL block:
        if time.time() - ts > USAGE_CACHE_TTL:
            print("[usage_cache] Disk cache expired, will rebuild")
            return None

        return payload
    except Exception as e:
        print(f"[usage_cache] Failed to load disk cache: {e!r}")
        return None



def _save_usage_to_disk(season: int, ts: float, data: Dict[str, Any]) -> None:
    path = _usage_cache_path(season)
    abs_path = os.path.abspath(path)
    print(f"[usage_cache] Writing usage cache to {abs_path} with {len(data)} players")

    payload = {"ts": ts, "data": data}

    # Optional: sanity check a couple of keys so you don't spam the logs
    sample_keys = list(data.keys())[:5]
    print(f"[usage_cache] Sample players in payload: {sample_keys}")

    try:
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)
            f.flush()
        print(f"[usage_cache] Successfully wrote cache file to {abs_path}")
    except Exception as e:
        print(f"[usage_cache] Failed to write disk cache to {abs_path}: {e!r}")



def get_usage_map_for_season(season: int) -> Dict[str, Dict[str, Any]]:
    """
    Weekly-cached map of player name -> usage metrics from Tank01.

    - If we have a fresh in-memory cache: return it.
    - Else try disk cache.
    - Else rebuild from Tank01 and cache to memory + disk.

    Returned shape (per player):
    {
      "Kyren Williams": {
        "games": 11,
        "avg_off_snap_pct": 0.78,
        "avg_targets": 4.2,
        "avg_carries": 16.1,
        "ppr_ppg": 19.3,
        "half_ppr_ppg": 17.0,
        "std_ppg": 6.4,
        ...
      },
      ...
    }
    """
    now = time.time()

    # 1) Check in-memory cache
    entry = _USAGE_CACHE.get(season)
    if entry and now - entry["ts"] <= USAGE_CACHE_TTL:
        return entry["data"]

    # 2) Try disk cache
    disk_entry = _load_usage_from_disk(season)
    if disk_entry:
        _USAGE_CACHE[season] = disk_entry
        return disk_entry["data"]

    # 3) Rebuild from Tank01
    players_index = load_players_index()
    print(f"[usage_cache] Rebuilding Tank01 usage map for season {season}...")
    data = build_tank01_usage_map_from_players_index(players_index, season=season)

    entry = {"ts": now, "data": data}
    _USAGE_CACHE[season] = entry
    _save_usage_to_disk(season, now, data)

    return data
