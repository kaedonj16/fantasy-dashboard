from __future__ import annotations

import glob
import json
import numpy as np
import os
import pandas as pd
import re
import requests
import time
from bs4 import BeautifulSoup
from collections import defaultdict, Counter
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Dict, Optional, Any, Callable, List

from dashboard_services.api import (
    get_nfl_games_for_week_raw,
    get_transactions,
    get_rosters,
    get_users,
    get_traded_picks,
    get_nfl_state,
    get_nfl_players, fetch_team_game_logs_html, fetch_tank_boxscore,
)

# ------------------------------------------------
# Core paths / constants
# ------------------------------------------------

# Project root (adjust if your structure is different)
ROOT_DIR = Path(__file__).resolve().parents[1]

# Shared data directory
DATA_DIR = ROOT_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Cache directory
CACHE_DIR = ROOT_DIR / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

BETTER_OUTWARD_METRICS = ["PF", "PA", "MAX", "MIN", "AVG", "STD"]
BETTER_OUTWARD_SIGNS = np.array([1, -1, 1, 1, 1, -1], dtype=float)
SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}
PUNCT_RE = re.compile(r"[^a-z0-9 ]+")
WS_RE = re.compile(r"\s+")
STATUS_NOT_STARTED = "not_started"
STATUS_IN_PROGRESS = "in_progress"
STATUS_FINAL = "final"

TEAM_ALIASES = {
    "jax": "JAX", "jacksonville": "JAX", "gb": "GB", "gnb": "GB", "nwe": "NE", "ne": "NE",
    "sfo": "SF", "sf": "SF", "kan": "KC", "kc": "KC", "tam": "TB", "tb": "TB",
    "was": "WAS", "was football team": "WAS", "wsh": "WAS",
    "lv": "LV", "oak": "LV", "sd": "LAC", "lac": "LAC", "la chargers": "LAC",
    "stl": "LAR", "lar": "LAR", "la rams": "LAR", "no": "NO", "nor": "NO",
    "bal": "BAL", "cin": "CIN", "pit": "PIT", "cle": "CLE", "buf": "BUF", "mia": "MIA",
    "nyj": "NYJ", "nyg": "NYG", "phi": "PHI", "dal": "DAL", "wasdc": "WAS",
    "min": "MIN", "chi": "CHI", "det": "DET", "atl": "ATL", "car": "CAR", "norleans": "NO",
    "sea": "SEA", "den": "DEN", "ari": "ARI", "hou": "HOU", "ten": "TEN", "ind": "IND",
}

DST_CANON = {
    "49ers": "SF",
    "Patriots": "NE",
    "Giants": "NYG",
    "Jets": "NYJ",
    "Commanders": "WAS",
    "Chargers": "LAC",
    "Rams": "LAR",
    "Raiders": "LV",
    "Saints": "NO",
    # ... extend as needed
}

TANK01_HOST = "tank01-nfl-live-in-game-real-time-statistics-nfl.p.rapidapi.com"
BASE = f"https://{TANK01_HOST}"
SCHEDULE_CACHE: dict[tuple[int, int], dict] = {}
SCHEDULE_TTL = 60 * 10  # seconds

TANK01_API_HOST = "tank01-nfl-live-in-game-real-time-statistics-nfl.p.rapidapi.com"
TANK01_API_KEY = "a31667ff00msh6d542faa96aa36bp1513aajsn612c819feca4"  # consider env var in prod

NFL_TEAMS = [
    "ARI", "ATL", "BAL", "BUF", "CAR", "CHI", "CIN", "CLE", "DAL", "DEN", "DET", "GB",
    "HOU", "IND", "JAX", "KC", "LAC", "LAR", "LV", "MIA", "MIN", "NE", "NO", "NYG", "NYJ",
    "PHI", "PIT", "SEA", "SF", "TB", "TEN", "WAS",
]


# ------------------------------------------------
# Small helpers
# ------------------------------------------------

def _headers(api_key: str) -> dict:
    return {
        "x-rapidapi-host": TANK01_HOST,
        "x-rapidapi-key": api_key,
    }


def scoring_key(scoring_params: Dict) -> str:
    payload = json.dumps(scoring_params or {}, sort_keys=True, separators=(",", ":"))
    import hashlib
    return hashlib.md5(payload.encode("utf-8")).hexdigest()[:10]


def z_better_outward(team_stats: pd.DataFrame,
                     metrics=BETTER_OUTWARD_METRICS,
                     signs=BETTER_OUTWARD_SIGNS) -> pd.DataFrame:
    Z = (team_stats[metrics] - team_stats[metrics].mean()) / team_stats[metrics].std(ddof=0)
    return Z * signs


def safe_owner_name(roster_map: dict, rid) -> str:
    return roster_map.get(str(rid), f"Roster {rid}")


# ------------------------------------------------
# Paths
# ------------------------------------------------

def path_week_schedule(season: int, week: int) -> str:
    # one file per season/week/day
    return os.path.join(
        CACHE_DIR,
        f"schedule/schedule_s{season}_w{week}_d{date.today().isoformat()}.json",
    )


def path_players_index() -> str:
    return os.path.join(CACHE_DIR, "players_index.json")


def path_relevant_index() -> str:
    return os.path.join(CACHE_DIR, "players_index_relevant.json")


def path_usage_table() -> str:
    return os.path.join(DATA_DIR, f"usage_table_{date.today().isoformat()}.json")


def path_engine_table() -> str:
    return os.path.join(DATA_DIR, f"engine_values_{date.today().isoformat()}.csv")


def path_model_value_table() -> str:
    return os.path.join(DATA_DIR, f"model_values_{date.today().isoformat()}.json")


def path_teams_index() -> str:
    return os.path.join(CACHE_DIR, "teams_index.json")


def path_idp_index() -> str:
    return os.path.join(CACHE_DIR, "idp_players_index.json")


def path_week_proj(season: int, week: int) -> str:
    return os.path.join(
        CACHE_DIR,
        f"projections/projections_s{season}_w{week}_d{date.today().isoformat()}.json",
    )


def path_week_stats(season: int, week: int) -> str:
    return os.path.join(CACHE_DIR, f"stats/week_stats_s{season}_w{week}.json")


def path_fantasycalc_values() -> str:
    return os.path.join(DATA_DIR, f"fantasycalc_api_values_{date.today().isoformat()}.csv")


def path_dynastyprocess_values() -> str:
    return os.path.join(DATA_DIR, f"dynastyprocess_values_{date.today().isoformat()}.csv")


# ------------------------------------------------
# JSON / table IO
# ------------------------------------------------

def read_json(path: str) -> Optional[dict]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return None


def write_json(path, data):
    """
    Safely writes a JSON object to disk.
    Works with both string and Path objects.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    tmp.replace(p)


def load_players_index() -> Optional[Dict]:
    """Returns the cached player index (Sleeper ↔ Tank01/name/team) or None."""
    return read_json(path_players_index())


def load_relevant_index() -> Optional[Dict]:
    return read_json(path_relevant_index())


def load_usage_table() -> Optional[Dict]:
    return read_json(path_usage_table())


def load_engine_table():
    try:
        df = pd.read_csv(path_engine_table())
    except Exception:
        df = None
    return df


def load_model_value_table():
    # Return ONLY the parsed JSON data, not the path object
    return read_json(path_model_value_table())


def load_teams_index() -> Optional[Dict]:
    """Returns the cached teams index or None."""
    return read_json(path_teams_index())


def load_idp_index() -> Optional[Dict]:
    """Returns the cached teams index or None."""
    return read_json(path_idp_index())


def load_week_stats(season: int, week: int) -> Optional[Dict]:
    """Returns cached weekly stats or None."""
    return read_json(path_week_stats(season, week))


def load_week_sched(season: int, week: int) -> Optional[Dict]:
    """Returns cached weekly stats or None."""
    return read_json(path_week_schedule(season, week))


def save_players_index(index_data: Dict) -> None:
    """index_data example: { sleeper_id: { 'name': ..., 'team': ..., 'tank01_id': ... }, ... }"""
    write_json(path_players_index(), index_data)


def load_week_schedule(season: int, w: int):
    """
    Load schedule for (season, week), fetching and caching if needed.
    """
    week_path = Path(path_week_schedule(season, w))
    if not week_path.exists():
        # FIX: use keyword args so order is correct
        get_week_schedule_cached(season=season, week=w, fetch_fn=get_nfl_games_for_week_raw)

    with open(week_path, "r", encoding="utf-8") as f:
        schedule = json.load(f)

    return schedule


def load_week_projection(season: int, w: int, force_refresh: bool = False) -> Optional[Dict]:
    """
    Load projections cache for (season, week), fetching if missing.
    """
    proj_path = Path(path_week_proj(season, w))
    if not proj_path.exists() or force_refresh:
        get_week_projections_cached(season, w, fetch_week_from_tank01, force_refresh=force_refresh)

    with open(proj_path, "r", encoding="utf-8") as f:
        projections = json.load(f)

    return projections


def save_week_projections(season: int, week: int, proj_map: Dict[str, float]) -> None:
    write_json(path_week_proj(season, week), proj_map)


def save_week_schedule(season: int, week: int, data: List[Dict]) -> None:
    write_json(path_week_schedule(season, week), data)


# ------------------------------------------------
# Tank01 – indexes & projections
# ------------------------------------------------

def cache_tank01_sleeper_index(
    rapidapi_key: str,
    cache_path: Path = CACHE_DIR / "players_index.json",
    force_refresh: bool = False,
) -> Dict[str, dict]:
    """
    Returns { sleeperbotid(str): { 'name': str, 'team': str, 'tankId': str } }
    """
    if cache_path.exists() and not force_refresh:
        cached = read_json(str(cache_path))
        if isinstance(cached, dict) and cached:
            return cached

    url = f"{BASE}/getNFLPlayerList"
    r = requests.get(url, headers=_headers(rapidapi_key), timeout=30)
    r.raise_for_status()
    data = r.json()

    # Tank01 returns players under "body" (adjust if needed)
    players = data.get("body") or data.get("players") or []

    idx: Dict[str, dict] = {}
    for p in players:
        # Try multiple key variants Tank01 may use
        sleeper = (
            p.get("sleeperBotID")
            or p.get("sleeperId")
            or p.get("sleeper_id")
            or p.get("sleeperid")
        )
        if not sleeper:  # skip if no Sleeper id available
            continue
        sleeper = str(sleeper)

        tank_id = str(p.get("playerID") or p.get("playerId") or p.get("id") or "")
        name = p.get("espnName", p.get("fullName", p.get("name", "")))
        team = p.get("team", p.get("proTeam", ""))

        idx[sleeper] = {
            "name": name or "",
            "team": team or "",
            "tankId": tank_id,
        }

    write_json(str(cache_path), idx)
    return idx


def get_week_projections_cached(
    season: int,
    week: int,
    fetch_fn: Callable[[int, int], Dict[str, float]],
    force_refresh: bool = False,
) -> Dict[str, float]:
    """
    fetch_fn should call Tank01 /getNFLProjections and return:
      { sleeper_id: projected_points, ... }
    """
    cache_path = get_or_refresh_projection_path(season, week)

    if os.path.exists(cache_path) and not force_refresh:
        return load_week_projection(season, week) or {}

    data = fetch_fn(season, week)
    save_week_projections(season, week, proj_map=data)
    return data


def get_or_refresh_projection_path(season: int, week: int) -> str:
    today = date.today()
    pattern = os.path.join(
        CACHE_DIR,
        f"projections/projections_s{season}_w{week}_d*.json",
    )
    matches = glob.glob(pattern)

    # If a prior file exists, check its date
    if matches:
        # Assume only one, but handle more than one just in case
        for file in matches:
            try:
                # extract the date part between `_d` and `.json`
                basename = os.path.basename(file)
                date_str = basename.split("_d")[1].replace(".json", "")
                file_date = datetime.strptime(date_str, "%Y-%m-%d").date()

                if file_date == today:
                    # Good — today's data exists
                    return file
                else:
                    # Old — remove it
                    os.remove(file)
            except Exception:
                # If parsing fails, just delete it
                os.remove(file)

    # If nothing exists or old file was removed, return today's fresh filename
    return path_week_proj(season, week)


def get_or_refresh_schedule_path(season: int, week: int) -> Optional[str]:
    today = date.today()
    pattern = os.path.join(
        CACHE_DIR,
        f"schedule/schedule_s{season}_w{week}_d*.json",
    )
    matches = glob.glob(pattern)

    if matches:
        for file in matches:
            try:
                basename = os.path.basename(file)
                # schedule_s2024_w3_d2025-11-19.json
                date_part = basename.split("_d", 1)[1].replace(".json", "")
                file_date = datetime.strptime(date_part, "%Y-%m-%d").date()

                if file_date == today:
                    # keep today's file
                    return file

                # delete files from previous days
                os.remove(file)

            except Exception:
                # if parsing fails, delete it
                os.remove(file)

    # no valid file for today → caller will fetch & save
    return None


def get_week_schedule_cached(
    season: int,
    week: int,
    fetch_fn: Callable[[int, int, str], List[Dict]],
    season_type: str = "reg",
) -> List[Dict]:
    """
    fetch_fn should call Tank01 /getNFLSchedule (or your schedule endpoint)
    and return a list[dict] of games.
    """
    cache_path = get_or_refresh_schedule_path(season, week)

    if cache_path is not None:
        # Just load via your schedule loader
        return load_week_schedule(season, week)

    # no cache for today → fetch and save
    data = fetch_fn(week, season, season_type)
    save_week_schedule(season, week, data)
    return data


def get_players_index_cached(rapidapi_key: str) -> Dict[str, Dict[str, Any]]:
    """
    Fetches the Tank01 player list (or loads from cache if already saved locally).
    Returns a mapping: { sleeper_id: { 'name': str, 'team': str, 'tank01_id': str } }
    """
    cache_dir = CACHE_DIR
    cache_dir.mkdir(exist_ok=True)
    cache_path = cache_dir / "tank01-players_index.json"

    # If cache exists locally, just load it
    if cache_path.exists():
        with cache_path.open("r", encoding="utf-8") as f:
            return json.load(f)

    # Otherwise, call Tank01 API
    url = f"https://{TANK01_API_HOST}/getNFLPlayerList"
    headers = {
        "x-rapidapi-host": TANK01_API_HOST,
        "x-rapidapi-key": rapidapi_key,
    }

    print("📡 Fetching Tank01 player list...")
    resp = requests.get(url, headers=headers, timeout=30)
    if resp.status_code != 200:
        raise RuntimeError(f"TANK01 API error {resp.status_code}: {resp.text[:200]}")

    data = resp.json().get("body", [])
    index = {}

    for p in data:
        sid = str(p.get("sleeperId") or p.get("sleeperbotid") or "")
        if not sid:
            continue
        index[sid] = {
            "name": p.get("longName") or p.get("name") or "",
            "team": p.get("team") or "",
            "tank01_id": p.get("playerID") or p.get("id") or "",
        }

    # Save to disk
    with cache_path.open("w", encoding="utf-8") as f:
        json.dump(index, f, indent=2)

    print(f"✅ Cached {len(index)} players to {cache_path}")
    return index


def canon_team(t: Optional[str]) -> Optional[str]:
    if not t:
        return None
    t0 = t.strip()
    # e.g., "49ers D/ST" => "49ers"
    if "D/ST" in t0 or "DST" in t0:
        t0 = t0.replace("D/ST", "").replace("DST", "").strip()
    up = TEAM_ALIASES.get(t0.lower(), t0.upper())
    # If still not a 2–3 letter code but a nickname like "49ers", map to code
    return DST_CANON.get(up, up)


def norm_name(s: str) -> str:
    """
    Lowercase, remove punctuation, collapse spaces, strip suffixes like Jr., III.
    Also converts 'Lastname, Firstname' -> 'firstname lastname'.
    """
    s = s.strip()
    if "," in s and s.count(",") == 1:
        last, first = [x.strip() for x in s.split(",", 1)]
        s = f"{first} {last}"
    s = s.replace("’", "'").replace("`", "'")
    s = PUNCT_RE.sub(" ", s.lower())
    s = WS_RE.sub(" ", s).strip()
    # remove suffix tokens at the end
    parts = s.split()
    while parts and parts[-1] in SUFFIXES:
        parts.pop()
    return " ".join(parts)


def get_league_rostered_player_ids(league_id: str) -> Dict[str, List[str]]:
    """Return {str(roster_id): [player_id,...]} for all roster + IR slots."""
    rosters = get_rosters(league_id) or []
    by_roster: Dict[str, List[str]] = {}
    for r in rosters:
        rid = str(r.get("roster_id"))
        main = r.get("players") or []
        reserve = r.get("reserve") or []
        by_roster[rid] = [str(p) for p in (list(main) + list(reserve)) if p]
    return by_roster


def streak_class(row) -> str:
    typ = (row.get("StreakType") or "").upper()
    ln = int(row.get("StreakLen") or 0)
    if typ == "W" and ln >= 2:
        return "streak-hot"
    if typ == "L" and ln >= 2:
        return "streak-cold"
    return ""


def fetch_week_from_tank01(season: int, week: int) -> dict[str, float]:
    """
    Fetches Tank01 projections for all players for a given week/season,
    then returns a dict { sleeper_id: projected_points }.
    """
    scoring_params = {
        "twoPointConversions": 2,
        "passYards": 0.04,
        "passTD": 4,
        "passInterceptions": -2,
        "pointsPerReception": 1,
        "rushYards": 0.1,
        "rushTD": 6,
        "fumbles": -2,
        "receivingYards": 0.1,
        "receivingTD": 6,
    }
    url = f"https://{TANK01_API_HOST}/getNFLProjections"
    params = {
        "week": week,
        "archiveSeason": season,
        "itemFormat": "list",
        **scoring_params,
    }

    headers = {
        "x-rapidapi-host": TANK01_API_HOST,
        "x-rapidapi-key": TANK01_API_KEY,
    }

    print(f"📡 Fetching Tank01 projections for Week {week}...")
    resp = requests.get(url, headers=headers, params=params, timeout=20)

    if resp.status_code != 200:
        print(f"⚠️ Tank01 API error {resp.status_code}: {resp.text[:200]}")
        return {}

    data = resp.json()
    # Tank01 sometimes stores projections in data["body"] or directly under "list"
    body = data.get("body") or data.get("list") or []
    if isinstance(body, dict):
        body = list(body.values())

    # Load cached player index for matching Sleeper IDs
    players_idx = load_players_index()

    if not players_idx:
        print("⚠️ No cached players index found. Run get_players_index_cached() first.")
        return {}

    proj_map = map_weekly_projections_to_sleeper(body, players_idx)

    print(f"✅ Retrieved {len(proj_map)} player projections for Week {week}")
    return proj_map


def map_weekly_projections_to_sleeper(
    weekly_rows: List[dict],
    idx_sleeper: Dict[str, dict],
) -> Dict[str, float]:
    """
    Convert Tank01 rows -> { sleeper_id: projected_points }.

    For your Tank01 shape (list of [team_rows, player_rows]):
      weekly_rows[0] = team rows (for DEF)
      weekly_rows[1] = player rows (for skill players)

    We map:
      • teamID -> Sleeper DEF id via teams_index
      • playerID -> Sleeper pid via players_index.tankId
    """
    out: Dict[str, float] = {}

    teams_index = load_teams_index() or {}
    players_index = load_players_index() or {}

    # Teams / DEF (index 0)
    if len(weekly_rows) > 0:
        for row in weekly_rows[0]:
            team_id_raw = row.get("teamID")
            proj = row.get("fantasyPointsDefault")
            if proj is None:
                continue

            team_key = next(
                (k for k, v in teams_index.items() if v.get("teamId") == str(team_id_raw)),
                None,
            )
            if team_key:
                out[str(team_key)] = float(proj)

    # Players (index 1)
    if len(weekly_rows) > 1:
        for row in weekly_rows[1]:
            tank_id = row.get("playerID")
            proj_raw = row.get("fantasyPointsDefault") or {}
            proj = proj_raw.get("PPR") if isinstance(proj_raw, dict) else proj_raw
            if proj is None:
                continue

            pid = next(
                (k for k, v in players_index.items() if v.get("tankId") == str(tank_id)),
                None,
            )
            if pid:
                out[str(pid)] = float(proj)

    return out


# ------------------------------------------------
# NFL team metadata / byes
# ------------------------------------------------

def _espn_logo_url(team_abv: str) -> str:
    # ESPN logo fallback (500px)
    return f"https://a.espncdn.com/i/teamlogos/nfl/500/{team_abv.lower()}.png"


def _safe_get(d: dict, *keys, default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def get_bye_week(team: dict, season: int) -> Optional[int]:
    """
    team: one object from payload['body']
    season: e.g., 2025
    returns: bye week as int, or None if missing
    """
    bye_map = team.get("byeWeeks") or {}
    # keys may be strings ("2025") and values may be ["8"] (strings)
    weeks = bye_map.get(str(season)) or bye_map.get(season)
    if not weeks:
        return None
    # Some seasons can list multiple byes; take the first valid int
    for w in weeks:
        try:
            return int(w)
        except (TypeError, ValueError):
            continue
    return None


def byes_for_season(payload: dict, season: int) -> dict[str, Optional[int]]:
    """
    payload: the full response with a 'body' list
    returns: { teamAbv: bye_week_or_None }
    """
    out = {}
    for team in payload.get("body", []):
        abv = team.get("teamAbv")
        out[abv] = get_bye_week(team, season)
    return out


def cache_tank01_teams_index(
    rapidapi_key: str,
    season: int,
    cache_path: Path,
    force_refresh: bool = False,
) -> Dict[str, dict]:
    """
    Builds and caches:
      { teamAbv: { 'teamId': str, 'byeWeek': int|None, 'Logo': str } } for all 32 NFL teams.
    """
    if cache_path.exists() and not force_refresh:
        cached = read_json(str(cache_path))
        if isinstance(cached, dict) and cached:
            return cached

    url = f"{BASE}/getNFLTeams"
    params = {"sortBy": "teamID", "season": season}
    rs = requests.get(url, params=params, headers=_headers(rapidapi_key), timeout=30)
    rs.raise_for_status()

    bye_weeks = byes_for_season(rs.json(), season)
    index: Dict[str, dict] = {}

    body = rs.json().get("body", []) or []

    for team_abv in NFL_TEAMS:
        # Tank01 uses WSH for Washington; map WAS -> WSH when looking up
        lookup_abv = "WSH" if team_abv == "WAS" else team_abv
        team_id = None
        for obj in body:
            if obj.get("teamAbv") == lookup_abv:
                team_id = obj.get("teamID")
                break

        index[team_abv] = {
            "teamId": team_id,
            "byeWeek": bye_weeks.get(lookup_abv),
            "Logo": _espn_logo_url(team_abv),
        }

    write_json(str(cache_path), index)
    return index


# ------------------------------------------------
# Game / player status for a given week
# ------------------------------------------------

BEFORE_WINDOW = 5 * 60        # 5 minutes
IN_WINDOW = 3 * 60 * 60

def normalize_game_status_from_tank01(game: dict, now: datetime | None = None) -> str:
    """
    Returns: "pre", "in", or "post" using:
      * kickoff time
      * a 5-minute early window
      * a 3-hour game window
      * Tank01 status fallback
    """

    if now is None:
        now = datetime.now(timezone.utc)

    # Parse kickoff time
    kickoff = None
    try:
        raw = game.get("gameTime_epoch")
        if raw not in (None, ""):
            kickoff_ts = float(raw)
            kickoff = datetime.fromtimestamp(kickoff_ts, tz=timezone.utc)
    except Exception:
        kickoff = None

    if kickoff is not None:
        delta = (now - kickoff).total_seconds()

        # --- PRE-GAME ---
        # More than 5 minutes before kickoff
        if delta < -BEFORE_WINDOW:
            return "pre"

        # --- IN-GAME ---
        # From 5 minutes before kickoff to up to 3 hours after kickoff
        if -BEFORE_WINDOW <= delta <= IN_WINDOW:
            return "in"

        # --- POST-GAME ---
        if delta > IN_WINDOW:
            return "post"

    # --- FALLBACK: Use Tank01’s status fields ---

    status = (game.get("gameStatus") or "").lower().strip()
    code = str(game.get("gameStatusCode") or "").strip()

    # Completed / final → post
    if code == "2" or "final" in status or "completed" in status:
        return "post"

    # In progress / live → in
    if code == "1" or "in progress" in status or "live" in status:
        return "in"

    # Scheduled → pre
    if code == "0" or "scheduled" in status:
        return "pre"

    # Default
    return "pre"


def build_games_by_team(games: list[dict]) -> dict[str, dict]:
    """
    games -> { team_abbr: { 'status': 'pre' | 'in' | 'post', 'game': game_obj } }
    """
    games_by_team: dict[str, dict] = {}
    for g in games:
        home = g.get("home")  # e.g. "NE"
        away = g.get("away")  # e.g. "NYJ"
        norm_status = normalize_game_status_from_tank01(g)  # 'pre' | 'in' | 'post'

        if home:
            games_by_team[home] = {"status": norm_status, "game": g}
        if away:
            games_by_team[away] = {"status": norm_status, "game": g}

    return games_by_team


def build_status_by_pid(
    players_info: dict[str, dict],
    games_by_team: dict[str, dict],
    teams_index: dict[str, dict],
    current_week: int,
    idp_players_info: Optional[dict[str, dict]] = None,
) -> dict[str, str]:
    """
    players_info:     { pid: { 'team': 'NYJ', ... }, ... }  # offensive / regular players
    idp_players_info: { pid: { 'team': 'NYJ', ... }, ... }  # IDP players
    teams_index:      { 'BUF': { 'teamId': '4', 'byeWeek': 7, ... }, ... }
    games_by_team:    { 'NYJ': { 'status': 'pre'|'in'|'post', ... }, ... }
    """
    status_by_pid: dict[str, str] = {}

    # Merge offensive + IDP indexes into a single view
    combined_players: dict[str, dict] = {}
    combined_players.update(players_info or {})
    if idp_players_info:
        combined_players.update(idp_players_info)

    # 1) All player pids (offense + IDP)
    for pid, info in combined_players.items():
        team = info.get("team")

        if not team:
            status_by_pid[pid] = STATUS_FINAL
            continue

        game = games_by_team.get(team)
        if not game:
            # bye or missing schedule
            status_by_pid[pid] = STATUS_FINAL
            continue

        t_status = game.get("status")  # 'pre' | 'in' | 'post'

        if t_status == "pre":
            status_by_pid[pid] = STATUS_NOT_STARTED
        elif t_status == "in":
            status_by_pid[pid] = STATUS_IN_PROGRESS
        elif t_status == "post":
            status_by_pid[pid] = STATUS_FINAL
        else:
            status_by_pid[pid] = STATUS_NOT_STARTED

    # 2) Defenses (teams_index)
    for team_code, team_info in teams_index.items():
        pid = team_code  # DEF pid matches team code

        # Don't overwrite if already assigned (very defensive, just in case)
        if pid in status_by_pid:
            continue

        game = games_by_team.get(team_code)

        if not game:
            bye_week = team_info.get("byeWeek")

            if bye_week == current_week:
                status_by_pid[pid] = "BYE"
            else:
                status_by_pid[pid] = STATUS_FINAL

            continue

        t_status = game.get("status")

        if t_status == "pre":
            status_by_pid[pid] = STATUS_NOT_STARTED
        elif t_status == "in":
            status_by_pid[pid] = STATUS_IN_PROGRESS
        elif t_status == "post":
            status_by_pid[pid] = STATUS_FINAL
        else:
            status_by_pid[pid] = STATUS_NOT_STARTED

    return status_by_pid


def build_matchup_player(
    pid: str,
    proj_map: dict[str, float],
    actual_map: dict[str, float],
    status_by_pid: dict[str, str],
) -> dict:
    base = _from_players_map(pid)  # existing helper in your codebase
    # base has: name, pos, nfl, etc.

    player = {
        "pid": pid,
        "name": base["name"],
        "pos": base["pos"],
        "nfl": base["nfl"],
        "projection": proj_map.get(pid),
        "actual": actual_map.get(pid),
        "status": status_by_pid.get(pid, STATUS_NOT_STARTED),
    }
    return decorate_player_display(player)


def build_status_for_week(
    season: int,
    week: int,
    players_index: dict[str, dict],
    teams_index: dict[str, dict],
    idp_player_index: dict[str, dict] = None,
) -> dict[str, str]:
    games = get_nfl_games_for_week(week, season)
    games_by_team = build_games_by_team(games)
    return build_status_by_pid(players_index, games_by_team, teams_index, week, idp_player_index if idp_player_index else None)


def decorate_player_display(player: dict) -> dict:
    status = player["status"]
    proj = player.get("projection")
    actual = player.get("actual")

    if proj is None:
        proj = 0.0
    if actual is None:
        actual = 0.0

    display = {
        "projection_value": None,
        "actual_value": None,
        "projection_muted": False,
    }

    # 1) not started: projection (muted) + 0.0 actual
    if status == STATUS_NOT_STARTED:
        display["projection_value"] = proj
        display["actual_value"] = 0.0
        display["projection_muted"] = True

    # 2) in progress: only actual
    elif status == STATUS_IN_PROGRESS:
        display["projection_value"] = None
        display["actual_value"] = actual

    # 3) final (including 0): only actual
    elif status == STATUS_FINAL:
        display["projection_value"] = None
        display["actual_value"] = actual

    # Leave BYE and any other status as "actual only"
    return {**player, **display}


def get_nfl_games_for_week(
    week: int,
    season: int,
    season_type: str = "reg",
) -> list[dict]:
    return get_week_schedule_cached(
        season=season,
        week=week,
        fetch_fn=get_nfl_games_for_week_raw,
        season_type=season_type,
    )


def pinfo_for_pid(
    pid: str,
    players_index: dict[str, dict],
    teams_index: dict[str, dict],
    players: dict[str, dict],
) -> dict:
    """
    Build a display object for a player or DEF using:
      - players_index: {pid: {name, team, tankId}}
      - teams_index:   { 'BUF': { teamId, byeWeek, Logo }, ... } for DEF
      - players:       Sleeper players map {pid: {...}} with 'pos'
    """
    info = players_index.get(pid, {})
    team_info = teams_index.get(pid, {})

    # name from your players_index, fallback to pid
    name = info.get("name") or pid

    # nfl team code (BAL, DET, BUF, etc.)
    nfl = info.get("team") or team_info.get("team") or (pid if pid in teams_index else None)

    # position (string)
    pos = ""
    if players and pid in players:
        player_obj = players[pid]  # full Sleeper dict
        pos = player_obj.get("pos") or player_obj.get("position") or ""
    elif pid in teams_index:
        pos = "DEF"

    return {
        "pid": pid,
        "name": name,
        "pos": pos,
        "nfl": nfl,
    }


def build_teams_overview(
    rosters: List[dict],
    users_list: List[dict],
    picks_by_roster: Dict[str, List[dict]],
    players: Dict[str, dict],
    players_index: Dict[str, dict],
    teams_index: Dict[str, dict],
) -> List[dict]:
    teams_ctx: List[dict] = []
    users_by_id = {str(u["user_id"]): u for u in users_list}

    for r in rosters:
        rid = str(r["roster_id"])
        owner_id = str(r.get("owner_id") or "")
        user = users_by_id.get(owner_id, {})

        # record from roster.settings
        settings = r.get("settings", {}) or {}
        wins = int(settings.get("wins", 0))
        losses = int(settings.get("losses", 0))
        ties = int(settings.get("ties", 0))
        record = f"{wins}-{losses}"
        if ties:
            record += f"-{ties}"

        starters_pids = r.get("starters", []) or []
        players_pids = r.get("players", []) or []
        ir_pids = r.get("reserve", []) or []
        taxi_pids = r.get("taxi", []) or []

        starter_set = set(starters_pids)
        ir_set = set(ir_pids)
        taxi_set = set(taxi_pids)

        bench_pids = [
            pid
            for pid in players_pids
            if pid not in starter_set and pid not in ir_set and pid not in taxi_set
        ]

        def enrich_list(pids: List[str]) -> List[dict]:
            return [pinfo_for_pid(pid, players_index, teams_index, players) for pid in pids]

        teams_ctx.append({
            "roster_id": rid,
            "name": user.get("metadata", {}).get("team_name")
                    or user.get("display_name")
                    or f"Team {rid}",
            "username": user.get("username") or "",
            "avatar": user.get("avatar_url") or user.get("avatar"),
            "record": record,
            "starters": enrich_list(starters_pids),
            "bench": enrich_list(bench_pids),
            "ir": enrich_list(ir_pids),
            "taxi": enrich_list(taxi_pids),
            "picks": picks_by_roster.get(rid, []),
        })

    teams_ctx.sort(key=lambda t: t["name"].lower())
    return teams_ctx


def bucket_for_slot(slot: int, num_teams: int = 10) -> str:
    """
    Map a pick number within the round (1..N) into 'early'/'mid'/'late'.
    Tuned for 10-team by default.
    """
    if slot <= 0:
        return "late"

    if num_teams == 10:
        if 1 <= slot <= 3:
            return "early"
        elif 4 <= slot <= 6:
            return "mid"
        else:
            return "late"

    # Generic fallback: split into thirds
    third = max(1, num_teams // 3)
    if slot <= third:
        return "early"
    elif slot <= 2 * third:
        return "mid"
    else:
        return "late"


# ------------------------------------------------
# Cache clearing helpers
# ------------------------------------------------

def clear_activity_cache_for_league(league_id: str) -> None:
    """
    Clear only the caches relevant to the Activity page for a given league:
      - transactions
      - traded picks
      - users/rosters
    """
    league_id = str(league_id)

    # 1) Clear transactions for all weeks
    if hasattr(get_transactions, "_cache"):
        keys_to_del = []
        for key in list(get_transactions._cache.keys()):
            func_name, args, kwargs = key
            if func_name != "get_transactions":
                continue
            if len(args) >= 1 and str(args[0]) == league_id:
                keys_to_del.append(key)
        for k in keys_to_del:
            get_transactions._cache.pop(k, None)

    # 2) Clear traded picks for this league
    if hasattr(get_traded_picks, "_cache"):
        keys_to_del = []
        for key in list(get_traded_picks._cache.keys()):
            func_name, args, kwargs = key
            if func_name != "get_traded_picks":
                continue
            if len(args) >= 1 and str(args[0]) == league_id:
                keys_to_del.append(key)
        for k in keys_to_del:
            get_traded_picks._cache.pop(k, None)

    if hasattr(get_users, "_cache"):
        get_users.clear_cache()

    if hasattr(get_rosters, "_cache"):
        print("clearing rosters")
        get_rosters.clear_cache()


def clear_standings_cache_for_league() -> None:
    if hasattr(get_nfl_state, "_cache"):
        get_nfl_state.clear_cache()

    if hasattr(get_nfl_players, "_cache"):
        get_nfl_players.clear_cache()

    if hasattr(get_users, "_cache"):
        get_users.clear_cache()

    if hasattr(get_rosters, "_cache"):
        get_rosters.clear_cache()


def clear_teams_cache_for_league() -> None:
    if hasattr(get_users, "_cache"):
        get_users.clear_cache()

    if hasattr(get_rosters, "_cache"):
        get_rosters.clear_cache()


def clear_weekly_cache_for_league() -> None:

    # print("INSIDE clear_weekly_cache_for_league in", __name__)
    week_schedule = load_week_sched(season, week)
    live_game_ids = get_live_game_ids_for_today(week_schedule)
    build_and_save_week_stats_for_league(load_teams_index(), season, week, live_game_ids)
    print("get_rosters repr:", get_rosters)
    print("get_rosters attrs:", [a for a in dir(get_rosters) if "cache" in a])
    if hasattr(get_nfl_state, "_cache"):
        get_nfl_state.clear_cache()

    if hasattr(get_nfl_players, "_cache"):
        get_nfl_players.clear_cache()

    if hasattr(get_users, "_cache"):
        get_users.clear_cache()

    if hasattr(get_rosters, "_cache"):
        get_rosters.clear_cache()
        print("cleared rosters")

    if hasattr(get_nfl_state, "_cache"):
        get_nfl_state.clear_cache()


def _safe_int(val: Any) -> int:
    if val is None:
        return 0
    if isinstance(val, (int, float)):
        return int(val)
    s = str(val).strip()
    if not s:
        return 0
    try:
        return int(float(s))
    except ValueError:
        return 0


def build_tank_player_index(players_index: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for sleeper_id, meta in players_index.items():
        tank_id = meta.get("tankId")
        if not tank_id:
            continue
        out[str(tank_id)] = {
            "name": meta.get("name", ""),
            "team": meta.get("team", ""),
            "pos": meta.get("pos", ""),
        }
    return out


def _normalize_qb_stats(ps: Dict[str, Any]) -> Dict[str, int]:
    passing = ps.get("Passing") or {}
    rushing = ps.get("Rushing") or {}
    return {
        "pass_yds": _safe_int(passing.get("passYds")),
        "pass_td": _safe_int(passing.get("passTD")),
        "int": _safe_int(passing.get("int")),
        "rush_att": _safe_int(rushing.get("carries")),
        "rush_yds": _safe_int(rushing.get("rushYds")),
        "rush_td": _safe_int(rushing.get("rushTD")),
    }


def _normalize_skill_stats(ps: Dict[str, Any]) -> Dict[str, int]:
    rushing = ps.get("Rushing") or {}
    receiving = ps.get("Receiving") or {}
    return {
        "rush_att": _safe_int(rushing.get("carries")),
        "rush_yds": _safe_int(rushing.get("rushYds")),
        "rush_td": _safe_int(rushing.get("rushTD")),
        "rec": _safe_int(receiving.get("receptions")),
        "rec_yds": _safe_int(receiving.get("recYds")),
        "rec_td": _safe_int(receiving.get("recTD")),
    }


def _normalize_dst_stats(team_block: Dict[str, Any]) -> Dict[str, int]:
    """
    Normalize team DST block into DEF stats.
    Example fields from Tank01:
      {
        "teamAbv": "WSH",
        "defTD": "0",
        "defensiveInterceptions": "0",
        "sacks": "4",
        "ydsAllowed": "313",
        "fumblesRecovered": "0",
        "ptsAllowed": "31",
        "safeties": "0"
      }
    """
    return {
        "def_td": _safe_int(team_block.get("defTD")),
        "def_int": _safe_int(team_block.get("defensiveInterceptions")),
        "sacks": _safe_int(team_block.get("sacks")),
        "yds_allowed": _safe_int(team_block.get("ydsAllowed")),
        "fumbles_recovered": _safe_int(team_block.get("fumblesRecovered")),
        "pts_allowed": _safe_int(team_block.get("ptsAllowed")),
        "safeties": _safe_int(team_block.get("safeties")),
    }


def build_live_stats_for_game_from_tank(
    boxscore: dict,
    players_index: dict,
) -> dict:
    """
    Returns:
      {
        "MIN": {
          "QB": {...},
          "RB": {...},
          "WR": {...},
          "TE": {...},
          "DEF": { "team": {...} }
        },
        "WSH": { ... }
      }
    """

    tank_idx = build_tank_player_index(players_index)

    # Note DEF is included here
    out: dict[str, dict[str, dict[str, dict]]] = defaultdict(
        lambda: {"QB": {}, "RB": {}, "WR": {}, "TE": {}, "DEF": {}}
    )

    # --- Player-level offense stats ---
    player_stats = boxscore.get("playerStats") or {}
    if isinstance(player_stats, dict):
        for tank_id, ps in player_stats.items():
            meta = tank_idx.get(str(tank_id))
            if not meta:
                continue

            team = meta.get("team") or ps.get("teamAbv") or ps.get("team")
            pos = meta.get("pos")
            name_key = (meta.get("name") or ps.get("longName") or "").lower()

            if not team or not pos or not name_key:
                continue

            if pos == "QB":
                stats = _normalize_qb_stats(ps)
                out[team]["QB"][name_key] = stats
            elif pos in ("RB", "WR", "TE"):
                stats = _normalize_skill_stats(ps)
                out[team][pos][name_key] = stats
            else:
                # still ignoring K/IDP; handled via team DST for DEF
                continue

    # --- Team DST → DEF (named as "DEF") ---
    dst_block = boxscore.get("DST") or {}
    if isinstance(dst_block, dict):
        for side in ("home", "away"):
            team_block = dst_block.get(side)
            if not isinstance(team_block, dict):
                continue
            team_abv = team_block.get("teamAbv") or team_block.get("team")
            if not team_abv:
                continue

            def_stats = _normalize_dst_stats(team_block)
            # Put under DEF, with a "team" key
            out[team_abv]["DEF"]["team"] = def_stats

    return out

def merge_live_stats_into_league_week_stats(
    league_week_stats: dict,
    live_stats: dict,
) -> None:
    """
    Mutates league_week_stats in-place, overwriting or adding stats
    for any players that appear in live_stats.
    Shape of both dicts:

      league_week_stats[team][pos][player_name] = {stat_dict}
    """
    for team, pos_map in live_stats.items():
        team_bucket = league_week_stats.setdefault(team, {})
        for pos, players in pos_map.items():
            pos_bucket = team_bucket.setdefault(pos, {})
            for name_key, stat_dict in players.items():
                # Overwrite existing or add new
                pos_bucket[name_key] = stat_dict


def get_live_game_ids_for_today(
    schedule: Iterable[Dict[str, Any]],
    today: date | None = None,
) -> List[str]:
    """
    Return Tank01 gameIDs for games:
      - with gameDate = today's date
      - AND whose epoch time indicates the game is currently within
        the live window (kickoff to +3 hours)
    """
    if today is None:
        today = date.today()

    today_str = today.strftime("%Y%m%d")
    live_ids: List[str] = []

    now = time.time()
    three_hours = 4 * 60 * 60  # 10800 seconds

    for game in schedule:
        if not isinstance(game, dict):
            continue

        # Must match today's date
        if str(game.get("gameDate") or "") != today_str:
            continue

        # Must have an epoch value
        raw_epoch = game.get("gameTime_epoch")
        if raw_epoch is None:
            continue

        try:
            game_epoch = float(raw_epoch)
        except (ValueError, TypeError):
            continue

        if game_epoch <= now <= (game_epoch + three_hours):
            gid = game.get("gameID")
            if gid:
                live_ids.append(str(gid))
    print("here")
    return live_ids if live_ids else []


def overlay_idp_stats_from_sleeper(
    league_week_stats: Dict[str, Dict[str, Dict[str, Dict[str, float]]]],
    season: int,
    week: int,
    teams_index: Dict[str, Dict[str, Any]],
) -> None:
    """
    Mutates league_week_stats in place by adding IDP stats from Sleeper.

    Uses:
      - idp_players_index.json (sleeperId -> {name, team, pos, ...})
      - sleeper_stats_s{season}_w{week}_*.json (sleeperId -> stat dict)

    Resulting structure matches existing week_stats:
      league_week_stats[team_abv][pos][player_name_lower] = { ...idp_stats... }
    """
    # ----- Load IDP player index -----
    idp_index = load_idp_index()
    if not isinstance(idp_index, dict):
        print("[week_stats][IDP] idp_players_index.json is not a dict, skipping.")
        return

    # ----- Find the Sleeper stats file for this season/week -----
    pattern = CACHE_DIR / f"sleeper_stats/sleeper_stats_s{season}_w{week}_*.json"
    candidates = sorted(glob.glob(str(pattern)))
    if not candidates:
        # fallback: maybe no date suffix
        fallback = CACHE_DIR / f"sleeper_stats_s{season}_w{week}.json"
        if fallback.exists():
            candidates = [str(fallback)]
        else:
            print(f"[week_stats][IDP] No sleeper stats file matching {pattern} or {fallback}")
            return

    sleeper_stats_path = Path(candidates[-1])  # latest match
    print(f"[week_stats][IDP] Using Sleeper stats file: {sleeper_stats_path.name}")

    try:
        with sleeper_stats_path.open("r", encoding="utf-8") as f:
            sleeper_stats = json.load(f)
    except Exception as e:
        print(f"[week_stats][IDP] Failed to load {sleeper_stats_path}: {e}")
        return

    if not isinstance(sleeper_stats, dict):
        print("[week_stats][IDP] Sleeper stats JSON not a dict, skipping.")
        return

    # ----- Build a small mapping to handle WSH/WAS, and validate teams -----
    valid_teams = set(teams_index.keys())
    # normalize WSH → WAS in the stats structure
    def normalize_team_abv(abv: str) -> str:
        if abv == "WSH":
            return "WAS"
        return abv

    # ----- Walk through each Sleeper IDP in the stats -----
    IDP_BUCKET = "IDP"
    added_count = 0

    # ----- Walk through each Sleeper IDP in the stats -----
    for sleeper_id, raw_stats in sleeper_stats.items():
        meta = idp_index.get(str(sleeper_id))
        if not meta:
            continue  # not an IDP or not in index

        name = (meta.get("name") or "").strip()
        team_abv = (meta.get("team") or "").strip().upper()
        pos = (meta.get("pos") or "").strip().upper()  # DB/DL/LB

        if not name or not team_abv:
            continue

        team_abv = normalize_team_abv(team_abv)
        if team_abv not in valid_teams:
            # Team not known in teams_index; skip to avoid junk keys
            continue

        name_key = name.lower()

        # Clean stats: keep numeric values
        if not isinstance(raw_stats, dict):
            continue

        stats_clean: Dict[str, float | str] = {}
        for k, v in raw_stats.items():
            if isinstance(v, (int, float)):
                stats_clean[k] = float(v)

        if not stats_clean:
            continue

        # Optionally keep their defensive position inside the stat blob
        stats_clean.setdefault("pos", pos)

        # Ensure team bucket exists
        team_bucket = league_week_stats.setdefault(team_abv, {})
        # Put ALL defensive players under "IDP"
        idp_bucket = team_bucket.setdefault(IDP_BUCKET, {})

        existing = idp_bucket.get(name_key, {})
        if isinstance(existing, dict):
            existing.update(stats_clean)
            idp_bucket[name_key] = existing
        else:
            idp_bucket[name_key] = stats_clean

        added_count += 1

    print(f"[week_stats][IDP] Added/updated {added_count} IDP stat lines.")


def build_and_save_week_stats_for_league(
    teams_index: Dict[str, Dict[str, Any]],
    season: int,
    week: int,
    live_game_ids: Optional[Iterable[str]] = None,
) -> Path:
    """
    1) Build baseline week stats from your existing HTML pipeline.
    2) Optionally overlay live Tank01 stats for any provided gameIDs.
    3) Overlay IDP stats from Sleeper using idp_players_index + sleeper_stats file.
    """
    league_week_stats: Dict[str, Dict[str, Dict[str, Dict[str, float]]]] = {}

    print(f"[week_stats] building stats for the week (season={season}, week={week})")
    for team_abv in teams_index.keys():
        orig_team_abv = team_abv
        if team_abv == "WSH":
            team_abv = "WAS"
        try:
            html = fetch_team_game_logs_html(team_abv, season)
            pos_player_stats = parse_team_week_pos_player_stats(html, week)
            league_week_stats[team_abv] = pos_player_stats
        except Exception as e:
            print(f"[week_stats] Error for {orig_team_abv} week {week}: {e}")
            league_week_stats[team_abv] = {}

    # ---------- Overlay live Tank01 stats (optional) ----------
    if live_game_ids:
        # Load players_index once, reuse
        players_index = load_players_index()

        # Normalize single string -> list
        if isinstance(live_game_ids, str):
            live_game_ids = [live_game_ids]

        session = requests.Session()

        for game_id in live_game_ids:
            try:
                print(f"[week_stats] fetching live Tank01 stats for game {game_id}")
                boxscore = fetch_tank_boxscore(game_id, session=session)
                live_stats = build_live_stats_for_game_from_tank(boxscore, players_index)
                merge_live_stats_into_league_week_stats(league_week_stats, live_stats)
            except Exception as e:
                print(f"[week_stats] Tank01 error for {game_id}: {e}")

    # ---------- Overlay IDP stats from Sleeper ----------
    overlay_idp_stats_from_sleeper(
        league_week_stats=league_week_stats,
        season=season,
        week=week,
        teams_index=teams_index,
    )

    out_path = path_week_stats(season, week)
    write_json(out_path, league_week_stats)
    print(f"[week_stats] Wrote → {out_path}")
    return out_path



def normalize_name(name: str) -> str:
    suffixes = {"jr", "jr.", "sr", "sr.", "ii", "iii", "iv", "v"}

    if not name:
        return ""
    s = name.lower()

    # collapse whitespace
    s = re.sub(r"\s+", " ", s).strip()

    # drop suffix tokens like jr, sr, ii, iii, etc.
    parts = s.split(" ")
    parts = [p for p in parts if p not in suffixes]

    return " ".join(parts)


def parse_team_week_pos_player_stats(
    html: str,
    target_week: int,
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    soup = BeautifulSoup(html, "html.parser")

    heading_to_pos = {
        "Quarterbacks": "QB",
        "Running Backs": "RB",
        "Wide Receivers": "WR",
        "Tight Ends": "TE",
    }

    result: Dict[str, Dict[str, Dict[str, Any]]] = {}

    for h2 in soup.find_all("h2"):
        heading = h2.get_text(strip=True)
        pos_code = heading_to_pos.get(heading)
        if not pos_code:
            continue

        table = h2.find_next("table")
        if not table:
            continue

        pos_players = parse_position_table_for_week(table, pos_code, target_week)
        result[pos_code] = pos_players

    return result


def parse_position_table_for_week(
    table,
    pos_code: str,
    target_week: int,
) -> Dict[str, Dict[str, Any]]:
    thead = table.find("thead")
    tbody = table.find("tbody")
    if not thead or not tbody:
        return {}

    head_rows = thead.find_all("tr")
    if not head_rows:
        return {}

    week_hdr_cells = head_rows[0].find_all("th")
    week_col_idx = None
    target_label = f"Wk {target_week}".lower()

    for i, th in enumerate(week_hdr_cells):
        txt = th.get_text(strip=True).lower()
        if txt == target_label:
            week_col_idx = i
            break

    if week_col_idx is None:
        return {}

    pos_players: Dict[str, Dict[str, Any]] = {}

    for tr in tbody.find_all("tr"):
        cells = tr.find_all("td")
        if len(cells) <= week_col_idx:
            continue

        name_cell = cells[0]
        link = name_cell.find("a")
        player_name = (
            link.get_text(strip=True) if link is not None else name_cell.get_text(strip=True)
        )
        if not player_name:
            continue
        player_name = normalize_name(player_name)

        stat_cell = cells[week_col_idx]
        plain_text = stat_cell.get_text(strip=True)
        if plain_text == "" or plain_text == "0":
            continue

        lines = list(stat_cell.stripped_strings)
        stats = parse_stat_lines_for_pos(lines, pos_code)
        if stats:
            pos_players[player_name] = stats

    return pos_players


def parse_stat_lines_for_pos(lines, pos_code: str) -> Dict[str, Any]:
    """
    lines: list of text lines from the cell for that week, e.g.
      QB: ["320-2-1", "3-19-0"]
      RB: ["8-69-0", "1-6-0"]
    """

    def _parse_nums(line: str) -> list[int]:
        line = line.strip()
        if not line:
            return []

        parts = line.split("-")
        nums: list[int] = []
        i = 0
        while i < len(parts):
            part = parts[i].strip()
            if part == "":
                if i + 1 < len(parts) and parts[i + 1].strip():
                    nums.append(-int(parts[i + 1].strip()))
                    i += 2
                else:
                    i += 1
            else:
                nums.append(int(part))
                i += 1
            if len(nums) >= 3:
                break
        return nums

    nums_by_line: list[list[int]] = []
    for line in lines:
        nums = _parse_nums(line)
        if nums:
            nums_by_line.append(nums)

    stats: Dict[str, Any] = {}

    if pos_code == "QB":
        if len(nums_by_line) >= 1 and len(nums_by_line[0]) >= 3:
            py, ptd, ints = nums_by_line[0][:3]
            stats.update({"pass_yds": py, "pass_td": ptd, "int": ints})
        if len(nums_by_line) >= 2 and len(nums_by_line[1]) >= 3:
            ra, ry, rtd = nums_by_line[1][:3]
            stats.update({"rush_att": ra, "rush_yds": ry, "rush_td": rtd})

    elif pos_code in {"RB", "WR"}:
        if len(nums_by_line) >= 1 and len(nums_by_line[0]) >= 3:
            ra, ry, rtd = nums_by_line[0][:3]
            stats.update({"rush_att": ra, "rush_yds": ry, "rush_td": rtd})
        if len(nums_by_line) >= 2 and len(nums_by_line[1]) >= 3:
            rec, r_yards, rtd2 = nums_by_line[1][:3]
            stats.update({"rec": rec, "rec_yds": r_yards, "rec_td": rtd2})

    elif pos_code == "TE":
        if len(nums_by_line) >= 1 and len(nums_by_line[0]) >= 3:
            rec, r_yards, rtd = nums_by_line[0][:3]
            stats.update({"rec": rec, "rec_yds": r_yards, "rec_td": rtd})

    else:
        if len(nums_by_line) >= 1 and len(nums_by_line[0]) >= 3:
            a, b, c = nums_by_line[0][:3]
            stats.update({"stat1": a, "stat2": b, "stat3": c})

    return stats


def count_roster_positions(positions: list[str]) -> dict[str, int]:
    """
    Takes a Sleeper roster_positions array and returns a count of each slot type.
    Example:
      ['QB','RB','RB','WR','WR','TE','FLEX',...] → {'QB':1,'RB':2,'WR':2,...}
    """
    return dict(Counter(positions))
