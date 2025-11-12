from __future__ import annotations

import json
import numpy as np
import os
import pandas as pd
import re
import requests
from pathlib import Path
from typing import Dict, Optional

from .api import _headers

CACHE_DIR = Path("cache")
BETTER_OUTWARD_METRICS = ["PF", "PA", "MAX", "MIN", "AVG", "STD"]
BETTER_OUTWARD_SIGNS = np.array([1, -1, 1, 1, 1, -1], dtype=float)
SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}
PUNCT_RE = re.compile(r"[^a-z0-9 ]+")
WS_RE = re.compile(r"\s+")

TEAM_ALIASES = {
    "jax": "JAX", "jacksonville": "JAX", "gb": "GB", "gnb": "GB", "nwe": "NE", "ne": "NE",
    "sfo": "SF", "sf": "SF", "kan": "KC", "kc": "KC", "tam": "TB", "tb": "TB",
    "was": "WAS", "was football team": "WAS", "wsh": "WAS",
    "lv": "LV", "oak": "LV", "sd": "LAC", "lac": "LAC", "la chargers": "LAC",
    "stl": "LAR", "lar": "LAR", "la rams": "LAR", "no": "NO", "nor": "NO",
    "bal": "BAL", "cin": "CIN", "pit": "PIT", "cle": "CLE", "buf": "BUF", "mia": "MIA",
    "nyj": "NYJ", "nyg": "NYG", "phi": "PHI", "dal": "DAL", "wasdc": "WAS",
    "min": "MIN", "chi": "CHI", "det": "DET", "atl": "ATL", "car": "CAR", "norleans": "NO",
    "sea": "SEA", "den": "DEN", "ari": "ARI", "hou": "HOU", "ten": "TEN", "ind": "IND"
}
DST_CANON = {
    "49ers": "SF", "Patriots": "NE", "Giants": "NYG", "Jets": "NYJ", "Commanders": "WAS",
    "Chargers": "LAC", "Rams": "LAR", "Raiders": "LV", "Saints": "NO",
    # ... add as needed
}
TANK01_HOST = "tank01-nfl-live-in-game-real-time-statistics-nfl.p.rapidapi.com"
BASE = f"https://{TANK01_HOST}"


def scoring_key(scoring_params: Dict) -> str:
    payload = json.dumps(scoring_params or {}, sort_keys=True, separators=(",", ":"))
    import hashlib
    return hashlib.md5(payload.encode("utf-8")).hexdigest()[:10]


def write_json(path: str, data: dict) -> None:
    tmp = path + ".tmp"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def z_better_outward(team_stats: pd.DataFrame,
                     metrics=BETTER_OUTWARD_METRICS,
                     signs=BETTER_OUTWARD_SIGNS) -> pd.DataFrame:
    Z = (team_stats[metrics] - team_stats[metrics].mean()) / team_stats[metrics].std(ddof=0)
    return Z * signs


def safe_owner_name(roster_map: dict, rid) -> str:
    return roster_map.get(str(rid), f"Roster {rid}")


def path_players_index() -> str:
    return os.path.join(CACHE_DIR, "players_index.json")


def path_teams_index() -> str:
    return os.path.join(CACHE_DIR, "teams_index.json")


def path_week_proj(season: int, week: int) -> str:
    return os.path.join(CACHE_DIR, f"projections_s{season}_w{week}.json")


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


def load_teams_index() -> Optional[Dict]:
    """Returns the cached player index (Sleeper ↔ Tank01/name/team) or None."""
    return read_json(path_teams_index())


def save_players_index(index_data: Dict) -> None:
    """index_data example: { sleeper_id: { 'name': ..., 'team': ..., 'tank01_id': ... }, ... }"""
    write_json(path_players_index(), index_data)


def load_week_projections(season: int, week: int) -> Optional[Dict[str, float]]:
    """
    Returns { sleeper_id: projected_points } for that week, scoring, season — or None.
    """
    data = read_json(path_week_proj(season, week))
    return data if isinstance(data, dict) else None


def save_week_projections(season: int, week: int, proj_map: Dict[str, float]) -> None:
    write_json(path_week_proj(season, week), proj_map)


def cache_tank01_sleeper_index(rapidapi_key: str,
                               cache_path: pathlib.Path = CACHE_DIR / "players_index.json",
                               force_refresh: bool = False) -> Dict[str, dict]:
    """
    Returns { sleeperbotid(str): { 'name': str, 'team': str, 'tankId': str } }
    """
    if cache_path.exists() and not force_refresh:
        cached = read_json(cache_path)
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
        sleeper = (p.get("sleeperBotID") or p.get("sleeperId") or
                   p.get("sleeper_id") or p.get("sleeperid"))
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

    write_json(cache_path, idx)
    return idx


def get_week_projections_cached(
        season: int,
        week: int,
        fetch_fn: Callable[[int, int, Dict], Dict[str, float]],
        force_refresh: bool = False,
) -> Dict[str, float]:
    """
    fetch_fn should call Tank01 /getNFLProjections with scoring params and return:
      { sleeper_id: projected_points, ... }
    """
    if not force_refresh:
        cached = load_week_projections(season, week)
        if cached is not None:
            return cached
    data = fetch_fn(season, week)
    save_week_projections(season, week, proj_map=data)
    return data


def save_week_projections(season: int, week: int, proj_map: Dict[str, float]) -> None:
    path = path_week_proj(season, week)
    write_json(path, proj_map)


def get_players_index_cached(rapidapi_key: str, week: int) -> Dict[str, Dict[str, Any]]:
    """
    Fetches the Tank01 player list (or loads from cache if already saved locally).
    Returns a mapping: { sleeper_id: { 'name': str, 'team': str, 'tank01_id': str } }
    """

    CACHE_DIR = "cache"
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_path = f"{CACHE_DIR}/tank01-players_index.json"

    # If cache exists locally, just load it
    if os.path.exists(cache_path):
        with open(cache_path, "r") as f:
            return json.load(f)

    # Otherwise, call Tank01 API
    url = "https://tank01-nfl-live-in-game-real-time-statistics-nfl.p.rapidapi.com/getNFLPlayerList"
    headers = {
        "x-rapidapi-host": "tank01-nfl-live-in-game-real-time-statistics-nfl.p.rapidapi.com",
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
    with open(cache_path, "w") as f:
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


def _streak_class(row) -> str:
    typ = (row.get("StreakType") or "").upper()
    ln = int(row.get("StreakLen") or 0)
    if typ == "W" and ln >= 2:
        return "streak-hot"
    if typ == "L" and ln >= 2:
        return "streak-cold"
    return ""


TANK01_API_HOST = "tank01-nfl-live-in-game-real-time-statistics-nfl.p.rapidapi.com"
TANK01_API_KEY = "a31667ff00msh6d542faa96aa36bp1513aajsn612c819feca4"


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
        **scoring_params,  # scoring rules like passYards, rushTD, etc.
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

    # Map Tank01 players → Sleeper IDs using your helper
    proj_map = map_weekly_projections_to_sleeper(body, players_idx)

    print(f"✅ Retrieved {len(proj_map)} player projections for Week {week}")
    return proj_map


def map_weekly_projections_to_sleeper(weekly_rows: List[dict],
                                      idx_sleeper: Dict[str, dict]) -> Dict[str, float]:
    """
    Convert Tank01 rows -> { sleeper_id: projected_points }
    Prefer 'sleeperbotid' if present; fallback to name/team matching against the cached index.
    """
    # Build reverse helpers for fallback matching
    by_name_team_to_sleeper = {}
    for sleeper_id, info in idx_sleeper.items():
        key = (info.get("name", "").strip().lower(), info.get("team", "").upper())
        by_name_team_to_sleeper[key] = sleeper_id

    out: Dict[str, float] = {}
    for row in weekly_rows[0]:
        teams = load_teams_index()
        teamId = next((k for k, v in teams.items() if v.get("teamId") == row.get("teamID")), None)
        proj = row.get("fantasyPointsDefault")
        if proj is None:
            continue
        if teamId:
            out[str(teamId)] = float(proj)
            continue

    for row in weekly_rows[1]:
        players = load_players_index()
        playerId = next((k for k, v in players.items() if v.get("tankId") == row.get("playerID")), None)
        proj = row.get("fantasyPointsDefault").get("PPR")
        if proj is None:
            continue

        if playerId:
            out[str(playerId)] = float(proj)
            continue

    return out


NFL_TEAMS = [
    "ARI", "ATL", "BAL", "BUF", "CAR", "CHI", "CIN", "CLE", "DAL", "DEN", "DET", "GB",
    "HOU", "IND", "JAX", "KC", "LAC", "LAR", "LV", "MIA", "MIN", "NE", "NO", "NYG", "NYJ",
    "PHI", "PIT", "SEA", "SF", "TB", "TEN", "WAS"
]


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
    team: one object from payload['body'] (like the ARI dict you showed)
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
        force_refresh: bool = False
) -> Dict[str, dict]:
    """
    Builds and caches:
      { teamAbv: { 'byeWeek': int|None, 'espnLogo1': str } } for all 32 NFL teams.

    Strategy:
      • Pull /getNFLPlayerList once; use first seen logo per team when available (espnLogo1/espnLogo).
      • For each team, try a Tank01 team-schedule call to discover the bye week.
      • Fallback logo is ESPN CDN if Tank01 doesn’t provide one.
    """
    if cache_path.exists() and not force_refresh:
        cached = read_json(cache_path)
        if isinstance(cached, dict) and cached:
            return cached

    index: Dict[str, dict] = {}
    teamId = None
    url = f"{BASE}/getNFLTeams"
    params = {"sortBy": "teamID", "season": season}
    rs = requests.get(url, params=params, headers=_headers(rapidapi_key), timeout=30)
    bye_weeks = byes_for_season(rs.json(), season)
    for team in NFL_TEAMS:
        if rs.ok:
            for obj in rs.json().get("body"):
                if team == "WAS":
                    team = "WSH"
                if obj.get("teamAbv") == team:
                    teamId = obj.get("teamID")
        index[team] = {
            "teamId": teamId,
            "byeWeek": bye_weeks.get(team),
            "Logo": _espn_logo_url(team)
        }

    write_json(cache_path, index)
    return index
