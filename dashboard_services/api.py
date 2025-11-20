from __future__ import annotations

import functools
import json
import os
import requests
import time
from datetime import date
from pathlib import Path
from typing import Any, List


def _make_hashable(x: Any):
    """
    Recursively turn lists/dicts/sets into hashable structures
    so they can be used as cache keys.
    """
    if isinstance(x, (str, int, float, bool, type(None))):
        return x
    if isinstance(x, (list, tuple)):
        return tuple(_make_hashable(i) for i in x)
    if isinstance(x, dict):
        # sort items so order doesn't matter
        return tuple(sorted((k, _make_hashable(v)) for k, v in x.items()))
    if isinstance(x, set):
        return tuple(sorted(_make_hashable(i) for i in x))
    # fallback for weird/custom objects
    return repr(x)


def ttl_cache(ttl: int = 300):
    """
    Simple in-memory TTL cache.
    Keyed by (function name, normalized args, normalized kwargs).
    """

    def decorator(func):
        _cache = {}  # { key: (timestamp, value) }

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            key = (
                func.__name__,
                _make_hashable(args),
                _make_hashable(kwargs),
            )
            now = time.time()

            if key in _cache:
                ts, value = _cache[key]
                if now - ts < ttl:
                    return value

            value = func(*args, **kwargs)
            _cache[key] = (now, value)
            return value

        return wrapper

    return decorator


SLEEPER_BASE = "https://api.sleeper.app/v1"
SCORING_DEFAULTS = {
    # Passing
    "twoPointConversions": 2, "passYards": 0.04, "passAttempts": -0.5, "passTD": 4,
    "passCompletions": 1, "passInterceptions": -2,
    # Receiving
    "pointsPerReception": 1, "receivingYards": 0.1, "receivingTD": 6, "targets": 0.1,
    # Rushing
    "carries": 0.2, "rushYards": 0.1, "rushTD": 6, "fumbles": -2,
    # Kicking
    "fgMade": 3, "fgMissed": -1, "xpMade": 1, "xpMissed": -1,
}
TANK01_HOST = "tank01-nfl-live-in-game-real-time-statistics-nfl.p.rapidapi.com"
BASE = f"https://{TANK01_HOST}"
TANK01_API_KEY = os.getenv("TANK01_API_KEY")

if not TANK01_API_KEY:
    raise RuntimeError("TANK01_API_KEY is not set. Export it or hardcode it temporarily.")


@ttl_cache(ttl=300)
def _avatar_from_users(users: list[dict], owner_id: Optional[str]) -> Optional[str]:
    if not owner_id:
        return None
    u = next((u for u in users if u.get("user_id") == owner_id), None)
    if not u:
        return None
    meta = u.get("metadata") or {}
    avatar_meta = meta.get("avatar")
    profile_id = u.get("avatar")
    if avatar_meta:
        return avatar_meta
    if profile_id:
        return f"https://sleepercdn.com/avatars/{profile_id}"
    return None


def fetch_json(path: str, timeout: int = 25) -> dict:
    url = f"{SLEEPER_BASE}{path}"
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    return r.json()


@ttl_cache(ttl=300)
def get_league(league_id: str) -> List[dict]:
    return fetch_json(f"/league/{league_id}")


@ttl_cache(ttl=300)
def get_users(league_id: str) -> List[dict]:
    return fetch_json(f"/league/{league_id}/users")


@ttl_cache(ttl=300)
def get_rosters(league_id: str) -> List[dict]:
    return fetch_json(f"/league/{league_id}/rosters")


@ttl_cache(ttl=300)
def get_matchups(league_id: str, week: int) -> List[dict]:
    return fetch_json(f"/league/{league_id}/matchups/{week}")


@ttl_cache(ttl=300)
def get_nfl_state() -> dict:
    return fetch_json("/state/nfl") or {}


@ttl_cache(ttl=300)
def get_nfl_players() -> dict:
    return fetch_json("/players/nfl") or {}


@ttl_cache(ttl=300)
def get_transactions(league_id: str, week: int) -> List[dict]:
    return fetch_json(f"/league/{league_id}/transactions/{week}")


@ttl_cache(ttl=300)
def get_bracket(league_id: str, bracket: str) -> List[dict]:
    return fetch_json(f"/league/{league_id}/{bracket}_bracket")


@ttl_cache(ttl=300)
def get_traded_picks(league_id: str) -> List[dict]:
    return fetch_json(f"/league/{league_id}/traded_picks")


@ttl_cache(ttl=300)
def get_nfl_games_for_week_raw(week: int, season: int, season_type: str = "reg") -> list[dict]:
    url = f"{BASE}/getNFLGamesForWeek"
    params = {"week": week, "seasonType": season_type, "season": season}
    resp = requests.get(url, headers=_headers(TANK01_API_KEY), params=params, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    return data.get("body") or data


def _avatar_url(avatar_id: str) -> Union[str, None]:
    if not avatar_id:
        return None
    return f"{avatar_id}"


def _headers(rapidapi_key: str) -> Dict[str, str]:
    return {
        "x-rapidapi-host": TANK01_HOST,
        "x-rapidapi-key": str(rapidapi_key),
    }


# cache_utils.py (or near top of your existing module)
def ttl_cache(ttl: int = 300):
    """
    Simple in-memory TTL cache.
    Keyed by (function name, args, kwargs).
    """

    def decorator(func):
        _cache = {}  # { key: (timestamp, value) }

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Build a hashable cache key
            key = (func.__name__, args, tuple(sorted(kwargs.items())))
            now = time.time()

            if key in _cache:
                ts, value = _cache[key]
                if now - ts < ttl:
                    return value

            value = func(*args, **kwargs)
            _cache[key] = (now, value)
            return value

        return wrapper

    return decorator


class Tank01Error(Exception):
    pass


def get_tank01_player_gamelogs(tank_player_id: str, season: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Returns a list of Tank01 game log objects for a given player.
    Safely handles cases where Tank01 returns a string instead of a list.
    """

    if not TANK01_API_KEY:
        raise Tank01Error("TANK01_API_KEY environment variable is not set.")

    url = f"{BASE}/getNFLGamesForPlayer"  # <-- adjust if needed

    querystring = {"playerID": str(tank_player_id)}
    if season is not None:
        querystring["season"] = str(season)

    headers = {
        "x-rapidapi-key": TANK01_API_KEY,
        "x-rapidapi-host": TANK01_HOST,
    }

    resp = requests.get(url, headers=headers, params=querystring, timeout=20)
    resp.raise_for_status()

    data = resp.json()

    if data.get("statusCode") != 200:
        # Log but do NOT hard-crash the pipeline
        print(f"[Tank01] Non-200 status for player {tank_player_id}: {data}")
        return []

    body = data.get("body", [])

    # ---- SAFETY CHECK (fixes your JSN error) ----
    if not isinstance(body, list):
        print(f"[Tank01] Unexpected body type for {tank_player_id}: {type(body)} -> {body}")
        return []
    # --------------------------------------------

    # Make sure each entry is a dict
    cleaned = [g for g in body if isinstance(g, dict)]

    return cleaned
