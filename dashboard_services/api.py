from __future__ import annotations

import functools
import os
from typing import Any, List, Dict, Optional, Union

import requests

# ---- League context globals ----
SCORING_SETTINGS: Dict[str, Any] = {}
ROSTER_POSITIONS: List[str] = []
LEAGUE_SETTINGS: Dict[str, Any] = {}
TOTAL_ROSTERS: int = 0
SLEEPER_BASE = "https://api.sleeper.app/v1"
SCORING_DEFAULTS = {
    # Passing
    "twoPointConversions": 2,
    "passYards": 0.04,
    "passAttempts": -0.5,
    "passTD": 4,
    "passCompletions": 1,
    "passInterceptions": -2,
    # Receiving
    "pointsPerReception": 1,
    "receivingYards": 0.1,
    "receivingTD": 6,
    "targets": 0.1,
    # Rushing
    "carries": 0.2,
    "rushYards": 0.1,
    "rushTD": 6,
    "fumbles": -2,
    # Kicking
    "fgMade": 3,
    "fgMissed": -1,
    "xpMade": 1,
    "xpMissed": -1,
}
TANK01_HOST = "tank01-nfl-live-in-game-real-time-statistics-nfl.p.rapidapi.com"
BASE = f"https://{TANK01_HOST}"
TANK01_API_KEY = os.getenv("TANK01_API_KEY")
FOOTBALLGUYS_TEAM_LOG_URL = "https://www.footballguys.com/stats/game-logs/teams"
LEAGUE_HISTORY_CACHE: dict[str, dict] = {}
LEAGUE_HISTORY_TTL = 60 * 60 * 12  # 12 hours

# NOTE: Don't check TANK01_API_KEY at import time - only when actually calling Tank01 API
# This allows model training and other scripts to import this module without requiring the key

# Reuse a single Session and a single headers dict for all Tank01 calls
SESSION = requests.Session()


def _make_hashable(x: Any):
    """
    Recursively turn lists/dicts/sets into hashable structures
    so they can be used as cache keys.
    (Currently not used in ttl_cache but kept for compatibility.)
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


# dashboard_services/api.py

cache = {}


def _freeze(obj):
    """Recursively convert unhashable types into hashable ones for cache keys."""
    if isinstance(obj, (list, tuple)):
        return tuple(_freeze(x) for x in obj)
    if isinstance(obj, dict):
        # sort keys so order is stable
        return tuple(sorted((k, _freeze(v)) for k, v in obj.items()))
    if isinstance(obj, set):
        return tuple(sorted(_freeze(x) for x in obj))
    return obj  # assume hashable


def ttl_cache(ttl: int = 300):
    """
    Simple in-memory TTL cache.
    Keyed by (function name, args, kwargs).

    This version is lightweight and works fine for the simple
    argument types used by the functions below.
    """

    def decorator(func):
        _cache: Dict[Any, tuple[float, Any]] = {}

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            frozen_args = _freeze(args)
            frozen_kwargs = _freeze(kwargs)
            key = (func.__name__, frozen_args, frozen_kwargs)

            if key in _cache:
                return _cache[key]

            result = func(*args, **kwargs)
            _cache[key] = result
            return result

        # expose cache and a convenience clearer if you ever want it
        wrapper._cache = _cache

        def clear_cache():
            _cache.clear()

        wrapper.clear_cache = clear_cache

        return wrapper

    return decorator


def _headers(rapidapi_key: str) -> Dict[str, str]:
    return {
        "x-rapidapi-host": TANK01_HOST,
        "x-rapidapi-key": str(rapidapi_key),
    }


TANK01_HEADERS = _headers(TANK01_API_KEY) if TANK01_API_KEY else {}


@ttl_cache(ttl=300)
def avatar_from_users(platform, users: list[dict], owner_id: Optional[str]) -> Optional[str]:
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
        if platform == "sleeper":
            return f"https://sleepercdn.com/avatars/{profile_id}"
        return f"{profile_id}"
    return None


def fetch_json(path: str, timeout: int = 25) -> dict:
    url = f"{SLEEPER_BASE}{path}"
    r = SESSION.get(url, timeout=timeout)
    r.raise_for_status()
    return r.json()


@ttl_cache(ttl=300)
def get_league(league_id: str) -> dict:
    """
    Fetch a Sleeper league and cache league-wide context in module globals.

    Populates:
      - SCORING_SETTINGS
      - ROSTER_POSITIONS
      - LEAGUE_SETTINGS
      - TOTAL_ROSTERS
    """
    global SCORING_SETTINGS, ROSTER_POSITIONS, LEAGUE_SETTINGS, TOTAL_ROSTERS

    league = fetch_json(f"/league/{league_id}") or {}

    if isinstance(league, dict):
        SCORING_SETTINGS = league.get("scoring_settings") or {}
        ROSTER_POSITIONS = league.get("roster_positions") or []
        LEAGUE_SETTINGS = league.get("settings") or {}
        TOTAL_ROSTERS = int(league.get("total_rosters") or 0)

    return league


def get_scoring_settings() -> Dict[str, Any]:
    """
    Raw scoring_settings from Sleeper for the current league.
    """
    return SCORING_SETTINGS


def get_effective_scoring_settings() -> Dict[str, float]:
    """
    Defaults overlaid with league-specific scoring.
    League scoring overrides defaults.
    """
    merged = dict(SCORING_DEFAULTS)
    merged.update(SCORING_SETTINGS or {})
    return merged


def get_roster_positions() -> List[str]:
    return ROSTER_POSITIONS


def get_league_settings() -> Dict[str, Any]:
    return LEAGUE_SETTINGS


def get_total_rosters() -> int:
    return TOTAL_ROSTERS


def set_league_globals(
        scoring_settings: Optional[Dict[str, Any]] = None,
        roster_positions: Optional[List[str]] = None,
        league_settings: Optional[Dict[str, Any]] = None,
        total_rosters: Optional[int] = None,
) -> None:
    """
    Allow external callers (e.g. ESPN integration) to populate the league globals
    that are normally populated by a Sleeper get_league() call.
    """
    global SCORING_SETTINGS, ROSTER_POSITIONS, LEAGUE_SETTINGS, TOTAL_ROSTERS
    if scoring_settings is not None:
        SCORING_SETTINGS = scoring_settings
    if roster_positions is not None:
        ROSTER_POSITIONS = roster_positions
    if league_settings is not None:
        LEAGUE_SETTINGS = league_settings
    if total_rosters is not None:
        TOTAL_ROSTERS = int(total_rosters)


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
def get_drafts(league_id: str) -> List[dict]:
    return fetch_json(f"/league/{league_id}/drafts")


@ttl_cache(ttl=300)
def get_nfl_games_for_week_raw(week: int, season: int, season_type: str = "reg") -> list[dict]:
    url = f"{BASE}/getNFLGamesForWeek"
    params = {"week": week, "seasonType": season_type, "season": season}
    resp = SESSION.get(url, headers=TANK01_HEADERS, params=params, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    return data.get("body") or data


def _avatar_url(avatar_id: str) -> Union[str, None]:
    if not avatar_id:
        return None
    return f"{avatar_id}"


@ttl_cache(ttl=300)
def get_sleeper_user_by_username(username: str) -> dict | None:
    username = (username or "").strip()
    if not username:
        return None

    resp = requests.get(f"{SLEEPER_BASE}/user/{username}", timeout=10)
    if resp.status_code == 404:
        return None
    resp.raise_for_status()
    data = resp.json()
    return data if isinstance(data, dict) and data.get("user_id") else None


@ttl_cache(ttl=300)
def get_sleeper_user_leagues(user_id: str, season: int, sport: str = "nfl") -> list[dict]:
    resp = requests.get(
        f"{SLEEPER_BASE}/user/{user_id}/leagues/{sport}/{season}",
        timeout=10,
    )
    resp.raise_for_status()
    data = resp.json()
    return data if isinstance(data, list) else []


class Tank01Error(Exception):
    pass


@ttl_cache(ttl=300)
def get_tank01_player_gamelogs(
        tank_player_id: str,
        season: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Call Tank01 and return a list of game dicts for a given player.

    Normalizes Tank01's body which may be:
      - a list of games, or
      - a dict keyed by gameId -> gameDict
    """

    if not TANK01_API_KEY:
        raise Tank01Error("TANK01_API_KEY environment variable is not set.")

    url = f"{BASE}/getNFLGamesForPlayer"  # or whatever your real endpoint is

    querystring: Dict[str, Any] = {
        "playerID": str(tank_player_id),
    }
    if season is not None:
        querystring["season"] = str(season)

    resp = SESSION.get(url, headers=TANK01_HEADERS, params=querystring, timeout=20)
    if resp.status_code != 200:
        raise Tank01Error(f"Tank01 API error {resp.status_code}: {resp.text[:200]}")

    data = resp.json()
    status_code = data.get("statusCode")
    if status_code != 200:
        raise Tank01Error(f"Tank01 returned statusCode={status_code}: {data}")

    raw_body = data.get("body") or {}

    if isinstance(raw_body, list):
        games: List[Dict[str, Any]] = [g for g in raw_body if isinstance(g, dict)]
    elif isinstance(raw_body, dict):
        games = [g for g in raw_body.values() if isinstance(g, dict)]
    else:
        print(
            f"[Tank01] Unexpected body type for {tank_player_id}: "
            f"{type(raw_body)} -> {raw_body}"
        )
        games = []

    return games


@ttl_cache(ttl=300)
def get_nfl_scores_for_date(game_date: str) -> dict:
    """
    Wraps Tank01 getNFLScoresOnly.

    game_date: 'YYYYMMDD' string, e.g. '20251204'
    Returns: body dict from Tank01 (gameID -> gameDict)
    """
    url = f"{BASE}/getNFLScoresOnly"
    params = {"gameDate": game_date, "topPerformers": "true"}

    resp = SESSION.get(url, headers=TANK01_HEADERS, params=params, timeout=20)
    resp.raise_for_status()
    data = resp.json() or {}
    return data.get("body") or {}


@ttl_cache(ttl=300)
def fetch_tank_boxscore(game_id: str, session: Optional[requests.Session] = None) -> dict:
    """
    Fetch a single live boxscore from Tank01 for game_id like '20251207_PIT@BAL'.
    Returns the parsed JSON body.
    """
    sess = session or requests.Session()

    params = {"gameID": game_id}

    url = f"{BASE}/getNFLBoxScore"
    resp = sess.get(url, headers=TANK01_HEADERS, params=params, timeout=5)
    resp.raise_for_status()
    data = resp.json()

    # Tank01 usually wraps payload in 'body'
    if isinstance(data, dict) and "body" in data:
        return data["body"]
    return data


def build_team_game_lookup(scores_body: dict) -> dict[str, dict]:
    """
    Given Tank01 scores body (gameID -> game dict),
    return a map: teamAbv -> game dict.

    Example:
      'DAL' -> { ... full game dict ... }
      'DET' -> { ... same game dict ... }
    """
    team_map: dict[str, dict] = {}

    for game in scores_body.values():
        if not isinstance(game, dict):
            continue
        home = game.get("home")
        away = game.get("away")
        if home:
            team_map[str(home)] = game
        if away:
            team_map[str(away)] = game

    return team_map


def build_league_history_map(platform: str, league_id: str, season: int) -> dict[int, str]:
    """
    Returns:
        {season_int: league_id}

    Walks backward through previous_league_id for Sleeper.
    """
    platform = str(platform or "").strip().lower()

    if platform != "sleeper":
        return {int(season): str(league_id).strip()}

    cache_key = f"{platform}:{league_id}"

    # check cache
    # cached = LEAGUE_HISTORY_CACHE.get(cache_key)
    # now = time.time()
    # if cached and (now - cached["ts"] < LEAGUE_HISTORY_TTL):
    #     return cached["map"]

    season_map: dict[int, str] = {}
    seen: set[str] = set()

    cursor_league_id = str(league_id).strip()
    season_cursor = int(season)

    while cursor_league_id and cursor_league_id not in seen:
        seen.add(cursor_league_id)
        try:
            league = get_league(cursor_league_id) or {}

        except Exception:
            break

        league_season = None
        try:
            league_season = int(league.get("season"))
        except Exception:
            pass

        resolved_league_id = str(league.get("league_id") or cursor_league_id).strip()

        if league_season:
            season_map[league_season] = resolved_league_id

        prev_id = str(league.get("previous_league_id") or "").strip()
        if not prev_id:
            break

        cursor_league_id = prev_id
        season_cursor = (league_season - 1) if league_season else (season_cursor - 1)

    # cache it
    # LEAGUE_HISTORY_CACHE[cache_key] = {
    #     "ts": now,
    #     "map": season_map,
    # }

    return season_map


def resolve_league_id_for_season(
        platform: str,
        league_id: str,
        current_season: int,
        target_season: int,
) -> str:
    """
    Returns the correct league_id for a given season.

    Uses cached season map instead of walking every time.
    Falls back safely if not found.
    """
    platform = str(platform or "").strip().lower()

    if platform != "sleeper":
        return str(league_id).strip()

    # Check if we're in offseason and should use previous season logic
    try:
        from dashboard_services.api import get_nfl_state
        nfl_state = get_nfl_state() or {}
        current_nfl_season = int(nfl_state.get("season", current_season))
        season_type = str(nfl_state.get("season_type", "")).lower().strip()

        # Determine which season to actually use for league resolution
        effective_season = target_season
        if current_nfl_season > current_season and season_type in {"offseason", "pre"}:
            # We're in offseason before current season has started, use previous season
            effective_season = target_season - 1
        elif current_nfl_season == current_season and season_type == "offseason":
            # Current season is over, use completed season
            effective_season = target_season
        elif season_type in {"offseason", "pre"}:
            # We're in some form of offseason, try previous season
            effective_season = target_season - 1
    except:
        effective_season = target_season

    season_map = build_league_history_map(platform, league_id, current_season)

    # exact match with effective season
    if effective_season in season_map:
        return season_map[effective_season]

    # fallback: closest older season
    older = [s for s in season_map if s <= effective_season]
    if older:
        return season_map[max(older)]

    # fallback: closest newer season
    newer = [s for s in season_map if s >= effective_season]
    if newer:
        return season_map[min(newer)]

    # final fallback
    return str(league_id).strip()


@ttl_cache(ttl=300)
def fetch_team_game_logs_html(team_abv: str, season: int) -> str:
    """
    Fetch Footballguys team game logs page for a given NFL team and season.
    team_abv: 'ATL', 'DAL', etc.
    """
    params = {
        "team": team_abv.upper(),
        "year": str(season),
    }
    resp = SESSION.get(FOOTBALLGUYS_TEAM_LOG_URL, params=params, timeout=20)
    resp.raise_for_status()
    return resp.text
