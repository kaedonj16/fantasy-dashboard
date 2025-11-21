from __future__ import annotations

import glob
import json
import numpy as np
import os
import pandas as pd
import re
import requests
import time
from datetime import date
from pathlib import Path
from typing import Dict, Optional

from dashboard_services.api import get_nfl_games_for_week_raw

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
SCHEDULE_CACHE: dict[tuple[int, int], dict] = {}
SCHEDULE_TTL = 60 * 10


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


def path_week_schedule(season: int, week: int) -> str:
    # one file per season/week – schedule itself doesn’t need the date in the name
    return os.path.join(CACHE_DIR, f"schedule/schedule_s{season}_w{week}_d{date.today()}.json")


def path_players_index() -> str:
    return os.path.join(CACHE_DIR, "players_index.json")


def path_teams_index() -> str:
    return os.path.join(CACHE_DIR, "teams_index.json")


def path_week_proj(season: int, week: int) -> str:
    return os.path.join(CACHE_DIR, f"projections/projections_s{season}_w{week}_d{date.today()}.json")


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


def save_week_schedule(season: int, week: int, data: List[Dict]) -> None:
    write_json(path_week_schedule(season, week), data)


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
) -> Dict[str, float]:
    """
    fetch_fn should call Tank01 /getNFLProjections with scoring params and return:
      { sleeper_id: projected_points, ... }
    """
    cache_path = get_or_refresh_projection_path(season, week)

    if os.path.exists(cache_path):
        return load_week_projections(season, week)
    else:
        data = fetch_fn(season, week)
        save_week_projections(season, week, proj_map=data)
    return data


def get_or_refresh_projection_path(season: int, week: int) -> str:
    today = date.today()
    pattern = os.path.join(CACHE_DIR, f"projections/projections_s{season}_w{week}_d*.json")
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


def get_or_refresh_schedule_path(season: int, week: int) -> str:
    today = date.today()
    pattern = os.path.join(CACHE_DIR, f"schedule/schedule_s{season}_w{week}_d*.json")
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

    # no valid file for today → return today's file path
    return path_week_schedule(season, week)  # MUST create _dYYYY-MM-DD.json


def get_week_schedule_cached(
        week: int,
        season: int,
        fetch_fn: Callable[[int, int, str], List[Dict]],
        season_type: str = "reg",
) -> List[Dict]:
    """
    fetch_fn should call Tank01 /getNFLSchedule (or your schedule endpoint)
    and return a list[dict] of games.
    """
    cache_path = get_or_refresh_schedule_path(season, week)

    if os.path.exists(cache_path):
        # your own loader – or json.load(open(cache_path))
        return load_week_schedule(season, week)

    # no cache for today → fetch and save
    data = fetch_fn(week, season, season_type)
    save_week_schedule(season, week, data)  # make this write to cache_path internally
    return data


def get_players_index_cached(rapidapi_key: str) -> Dict[str, Dict[str, Any]]:
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


def streak_class(row) -> str:
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


STATUS_NOT_STARTED = "not_started"
STATUS_IN_PROGRESS = "in_progress"
STATUS_FINAL = "final"


def normalize_game_status_from_tank01(game: dict) -> str:
    """
    Convert Tank01's gameStatus/gameStatusCode into 'pre' | 'in' | 'post'.
    """
    raw = (game.get("gameStatus") or "").lower()
    code = str(game.get("gameStatusCode") or "").strip()

    # From your sample:
    # gameStatus: 'Scheduled' -> pre
    # gameStatus: 'Final'     -> post
    # (If they ever send 'In Progress' / code 1, treat as in)
    if raw == "final" or code == "2":
        return "post"
    if raw == "scheduled" or code == "0":
        return "pre"
    if "progress" in raw or code == "1":
        return "in"

    # Fallback
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
        current_week: int
) -> dict[str, str]:
    """
    players_info: { pid: { 'nfl': 'NYJ', ... }, ... }
    teams_index:  { 'BUF': { 'teamId': '4', 'byeWeek': 7, ... }, ... }
    games_by_team: { 'NYJ': { 'status': 'pre'|'in'|'post', ... }, ... }
    """
    status_by_pid: dict[str, str] = {}

    # --- 1) Offensive players / normal players ---
    for pid, info in players_info.items():
        # Prefer 'nfl', fall back to 'team' if present
        team = info.get("team")

        if not team:
            status_by_pid[pid] = STATUS_FINAL
            continue

        game = games_by_team.get(team)
        if not game:
            # bye or missing schedule
            status_by_pid[pid] = STATUS_FINAL
            continue

        t_status = game["status"]  # 'pre' | 'in' | 'post'

        if t_status == "pre":
            status_by_pid[pid] = STATUS_NOT_STARTED
        elif t_status == "in":
            status_by_pid[pid] = STATUS_IN_PROGRESS
        elif t_status == "post":
            status_by_pid[pid] = STATUS_FINAL
        else:
            status_by_pid[pid] = STATUS_NOT_STARTED

    # --- 2) Defenses (teams_index) ---
    # keys in teams_index are team codes: "BUF", "NE", "DET", etc.
    for team_code, team_info in teams_index.items():
        # Def PID is usually the team code itself ("BUF", "NE", etc.)
        pid = team_code

        # Don't overwrite if you *already* assigned a status for this pid
        if pid in status_by_pid:
            continue

        game = games_by_team.get(team_code)

        if not game:
            # No game found for this team in the schedule.
            bye_week = team_info.get("byeWeek")

            if bye_week == current_week:
                # Explicit bye this week – you can keep STATUS_FINAL here
                # because your UI already checks byeWeek == w to show "BYE".
                status_by_pid[pid] = "BYE"
            else:
                # No game and not the bye week: treat as FINAL / safe fallback.
                status_by_pid[pid] = STATUS_FINAL

            continue

        # If there *is* a game, map its status
        t_status = game["status"]

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
    base = _from_players_map(pid)  # whatever you were already using
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
      - players_index: {pid: {name, team, tankId, byeWeek}}
      - teams_index:   { 'BUF': { teamId, byeWeek, Logo }, ... } for DEF
      - players:       Sleeper players map {pid: {...}} with 'position'
    """
    info = players_index.get(pid, {})
    team_info = teams_index.get(pid, {})

    # name from your players_index, fallback to pid
    name = info.get("name") or pid

    # nfl team code (BAL, DET, BUF, etc.)
    # for players, use players_index["team"]; for DEF, use teams_index
    nfl = info.get("team") or team_info.get("team") or (pid if pid in teams_index else None)

    # position (string)
    pos = ""
    if players and pid in players:
        player_obj = players[pid]  # full Sleeper dict
        # prefer 'position'; fallback to fantasy_positions[0] if you use that
        pos = player_obj.get("pos") or ""
    elif pid in teams_index:
        pos = "DEF"  # treat team IDs as defenses

    return {
        "pid": pid,
        "name": name,
        "pos": pos,  # now a simple string ("QB", "RB", etc.)
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
            pid for pid in players_pids
            if pid not in starter_set
               and pid not in ir_set
               and pid not in taxi_set
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
