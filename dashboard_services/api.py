from __future__ import annotations

import functools
import logging
import os
import threading
import time
from typing import Any, List, Dict, Optional, Union

import requests
from flask import g, has_app_context

from dashboard_services.circuit_breaker import get_breaker

logger = logging.getLogger(__name__)

# Shared circuit breaker for all Tank01 calls
_tank01_breaker = get_breaker("tank01", failure_threshold=5, reset_timeout=300)

# ---- League context (request-scoped) ----
# League config (scoring/roster/settings) is stored per-request via Flask's ``g``
# so concurrent requests for *different* leagues never clobber each other. Outside
# a request/app context (cron jobs, scripts, background data builds) it falls back
# to a thread-local, which is likewise isolated per thread.
_thread_local = threading.local()


def _league_state() -> Dict[str, Any]:
    """Return the per-request (or per-thread) league-config store."""
    if has_app_context():
        state = getattr(g, "_league_state", None)
        if state is None:
            state = {}
            g._league_state = state
        return state
    state = getattr(_thread_local, "league_state", None)
    if state is None:
        state = {}
        _thread_local.league_state = state
    return state


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
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

SESSION = requests.Session()
retry_strategy = Retry(
    total=3,
    backoff_factor=1,
    status_forcelist=[429, 500, 502, 503, 504],
)
adapter = HTTPAdapter(
    pool_connections=20,  # Increase from default 10
    pool_maxsize=20,      # Increase from default 10  
    max_retries=retry_strategy
)
SESSION.mount("http://", adapter)
SESSION.mount("https://", adapter)


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
                ts, cached_result = _cache[key]
                if time.time() - ts < ttl:
                    return cached_result
                del _cache[key]

            result = func(*args, **kwargs)
            _cache[key] = (time.time(), result)
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
            return f"https://sleepercdn.com/avatars/thumbs/{profile_id}"
        return f"{profile_id}"
    return None


def fetch_json(path: str, timeout: int = 25, retries: int = 3) -> dict:
    url = f"{SLEEPER_BASE}{path}"
    last_err: Exception = RuntimeError("fetch_json: no attempts made")
    for attempt in range(retries):
        try:
            r = SESSION.get(url, timeout=timeout)
            if r.status_code == 429:
                wait = 2 ** attempt
                logger.warning("Sleeper rate-limited, retrying in %ds (attempt %d/%d)", wait, attempt + 1, retries)
                time.sleep(wait)
                continue
            r.raise_for_status()
            return r.json()
        except requests.RequestException as e:
            last_err = e
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
    raise last_err


@ttl_cache(ttl=300)
def _fetch_league(league_id: str) -> dict:
    """Cached raw Sleeper league fetch (no side effects)."""
    return fetch_json(f"/league/{league_id}") or {}


def get_league(league_id: str) -> dict:
    """
    Fetch a Sleeper league and populate the request-scoped league context.

    Populates (for the current request/thread only):
      - scoring_settings
      - roster_positions
      - league_settings
      - total_rosters

    The HTTP fetch is cached, but the context is repopulated on *every* call
    (including cache hits) so a request always sees its own league's config.
    """
    league = _fetch_league(league_id)

    if isinstance(league, dict) and league:
        set_league_globals(
            scoring_settings=league.get("scoring_settings") or {},
            roster_positions=league.get("roster_positions") or [],
            league_settings=league.get("settings") or {},
            total_rosters=int(league.get("total_rosters") or 0),
        )

    return league


def get_effective_scoring_settings() -> Dict[str, float]:
    """
    Defaults overlaid with league-specific scoring.
    League scoring overrides defaults.
    """
    merged = dict(SCORING_DEFAULTS)
    merged.update(_league_state().get("scoring_settings") or {})
    return merged


def get_roster_positions() -> List[str]:
    return _league_state().get("roster_positions") or []


def get_league_settings() -> Dict[str, Any]:
    return _league_state().get("league_settings") or {}


def get_total_rosters() -> int:
    return int(_league_state().get("total_rosters") or 0)


def set_league_globals(
        scoring_settings: Optional[Dict[str, Any]] = None,
        roster_positions: Optional[List[str]] = None,
        league_settings: Optional[Dict[str, Any]] = None,
        total_rosters: Optional[int] = None,
) -> None:
    """
    Populate the request-scoped league context (e.g. from the ESPN integration,
    which does not go through the Sleeper ``get_league()`` path).
    """
    state = _league_state()
    if scoring_settings is not None:
        state["scoring_settings"] = scoring_settings
    if roster_positions is not None:
        state["roster_positions"] = roster_positions
    if league_settings is not None:
        state["league_settings"] = league_settings
    if total_rosters is not None:
        state["total_rosters"] = int(total_rosters)


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


@ttl_cache(ttl=30)
def get_drafts(league_id: str) -> List[dict]:
    """League draft list (status + scheduled start_time). Short TTL so the
    imminent-draft banner catches reschedules (start pushed back 15 min / 1 hr)
    within ~30s instead of waiting out a long cache."""
    return fetch_json(f"/league/{league_id}/drafts")


@ttl_cache(ttl=8)
def get_draft(draft_id: str) -> dict:
    """Sleeper draft metadata (status, type, settings incl. reversal_round, draft_order)."""
    return fetch_json(f"/draft/{draft_id}") or {}


@ttl_cache(ttl=1)
def get_draft_picks(draft_id: str) -> List[dict]:
    """Live/completed picks for a Sleeper draft. 1s TTL so every 2s poll sees fresh data."""
    return fetch_json(f"/draft/{draft_id}/picks") or []


@ttl_cache(ttl=300)
def get_nfl_games_for_week_raw(week: int, season: int, season_type: str = "reg") -> list[dict]:
    if _tank01_breaker.is_open():
        logger.warning("[Tank01] Circuit OPEN - skipping getNFLGamesForWeek w%s s%s", week, season)
        return []
    url = f"{BASE}/getNFLGamesForWeek"
    params = {"week": week, "seasonType": season_type, "season": season}
    try:
        resp = SESSION.get(url, headers=TANK01_HEADERS, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        _tank01_breaker.record_success()
        return data.get("body") or data
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 429:
            logger.warning("[Tank01] Rate limited - getNFLGamesForWeek w%s s%s", week, season)
            _tank01_breaker.record_failure()
            return []
        logger.error("[Tank01] HTTP %s - getNFLGamesForWeek w%s s%s", e.response.status_code, week, season)
        _tank01_breaker.record_failure()
        raise
    except requests.exceptions.RequestException as e:
        logger.error("[Tank01] Request error - getNFLGamesForWeek w%s s%s: %s", week, season, e)
        _tank01_breaker.record_failure()
        return []
    except Exception as e:
        logger.exception("[Tank01] Unexpected error - getNFLGamesForWeek w%s s%s", week, season)
        _tank01_breaker.record_failure()
        return []


def avatar_url(avatar_id: str) -> Union[str, None]:
    if not avatar_id:
        return None
    return f"{avatar_id}"


@ttl_cache(ttl=300)
def get_sleeper_user_by_username(username: str) -> dict | None:
    username = (username or "").strip()
    if not username:
        return None

    resp = SESSION.get(f"{SLEEPER_BASE}/user/{username}", timeout=10)
    if resp.status_code == 404:
        return None
    resp.raise_for_status()
    data = resp.json()
    return data if isinstance(data, dict) and data.get("user_id") else None


@ttl_cache(ttl=300)
def get_sleeper_user_leagues(user_id: str, season: int, sport: str = "nfl") -> list[dict]:
    resp = SESSION.get(
        f"{SLEEPER_BASE}/user/{user_id}/leagues/{sport}/{season}",
        timeout=10,
    )
    resp.raise_for_status()
    data = resp.json()
    return data if isinstance(data, list) else []


class Tank01Error(Exception):
    pass


@ttl_cache(ttl=300)
def get_nfl_scores_for_date(game_date: str) -> dict:
    """
    Wraps Tank01 getNFLScoresOnly.

    game_date: 'YYYYMMDD' string, e.g. '20251204'
    Returns: body dict from Tank01 (gameID -> gameDict)
    """
    url = f"{BASE}/getNFLScoresOnly"
    params = {"gameDate": game_date, "topPerformers": "true"}

    if _tank01_breaker.is_open():
        logger.warning("[Tank01] Circuit OPEN - skipping getNFLScoresOnly %s", game_date)
        return {}
    try:
        resp = SESSION.get(url, headers=TANK01_HEADERS, params=params, timeout=20)
        resp.raise_for_status()
        data = resp.json() or {}
        _tank01_breaker.record_success()
        return data.get("body") or {}
    except requests.exceptions.HTTPError as e:
        _tank01_breaker.record_failure()
        _status = getattr(e.response, "status_code", None)
        if _status == 429:
            logger.warning("[Tank01] Rate limited - getNFLScoresOnly %s", game_date)
            return {}
        # Scores are non-critical page furniture; a 403/4xx/5xx from the
        # upstream provider must not bubble up and 500 the whole dashboard.
        # Log and degrade gracefully to an empty scoreboard.
        logger.error("[Tank01] HTTP %s - getNFLScoresOnly %s", _status, game_date)
        return {}
    except requests.exceptions.RequestException as e:
        logger.error("[Tank01] Request error - getNFLScoresOnly %s: %s", game_date, e)
        _tank01_breaker.record_failure()
        return {}
    except Exception as e:
        logger.exception("[Tank01] Unexpected error - getNFLScoresOnly %s", game_date)
        _tank01_breaker.record_failure()
        return {}


@ttl_cache(ttl=300)
def fetch_tank_boxscore(game_id: str, session: Optional[requests.Session] = None) -> dict:
    """
    Fetch a single live boxscore from Tank01 for game_id like '20251207_PIT@BAL'.
    Returns the parsed JSON body.
    """
    sess = session or requests.Session()

    params = {"gameID": game_id}

    if _tank01_breaker.is_open():
        logger.warning("[Tank01] Circuit OPEN - skipping getNFLBoxScore %s", game_id)
        return {}
    try:
        url = f"{BASE}/getNFLBoxScore"
        resp = sess.get(url, headers=TANK01_HEADERS, params=params, timeout=5)
        resp.raise_for_status()
        data = resp.json()
        _tank01_breaker.record_success()
        if isinstance(data, dict) and "body" in data:
            return data["body"]
        return data
    except requests.exceptions.HTTPError as e:
        _tank01_breaker.record_failure()
        if e.response.status_code == 429:
            logger.warning("[Tank01] Rate limited - getNFLBoxScore %s", game_id)
            return {}
        logger.error("[Tank01] HTTP %s - getNFLBoxScore %s", e.response.status_code, game_id)
        raise
    except requests.exceptions.RequestException as e:
        logger.error("[Tank01] Request error - getNFLBoxScore %s: %s", game_id, e)
        _tank01_breaker.record_failure()
        return {}
    except Exception as e:
        logger.exception("[Tank01] Unexpected error - getNFLBoxScore %s", game_id)
        _tank01_breaker.record_failure()
        return {}


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
    For ESPN, the same league_id is reused across years - probe prior seasons directly.
    """
    platform = str(platform or "").strip().lower()

    if platform == "espn":
        cache_key = f"espn:{league_id}"
        now = time.time()
        cached = LEAGUE_HISTORY_CACHE.get(cache_key)
        if cached and (now - cached["ts"] < LEAGUE_HISTORY_TTL):
            return cached["map"]

        season_map: dict[int, str] = {int(season): str(league_id)}
        from dashboard_services.providers.espn_api import _league_cached as _espn_league
        consecutive_failures = 0
        for yr in range(int(season) - 1, max(int(season) - 10, 2010) - 1, -1):
            try:
                _espn_league(yr, str(league_id))
                season_map[yr] = str(league_id)
                consecutive_failures = 0
            except Exception:
                consecutive_failures += 1
                if consecutive_failures >= 2:
                    break

        LEAGUE_HISTORY_CACHE[cache_key] = {"ts": now, "map": season_map}
        return season_map

    if platform != "sleeper":
        return {int(season): str(league_id).strip()}

    cache_key = f"{platform}:{league_id}"

    now = time.time()
    cached = LEAGUE_HISTORY_CACHE.get(cache_key)
    if cached and (now - cached["ts"] < LEAGUE_HISTORY_TTL):
        return cached["map"]

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
            logger.debug("suppressed exception", exc_info=True)

        resolved_league_id = str(league.get("league_id") or cursor_league_id).strip()

        if league_season:
            season_map[league_season] = resolved_league_id

        prev_id = str(league.get("previous_league_id") or "").strip()
        if not prev_id:
            break

        cursor_league_id = prev_id
        season_cursor = (league_season - 1) if league_season else (season_cursor - 1)

    LEAGUE_HISTORY_CACHE[cache_key] = {"ts": now, "map": season_map}

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
    except Exception:
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
