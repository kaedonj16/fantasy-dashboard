"""Consolidated utils module: game_status.

NFL game status, live-game detection, and week-status builders (split from utils/utils.py).
"""

# --- imports carried over from utils/utils.py ---
import glob
import json
import os
import re
import threading as _threading
import requests
import time
import traceback
import uuid
from contextlib import contextmanager as _contextmanager
from bs4 import BeautifulSoup
from collections import OrderedDict as _OrderedDict, defaultdict
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Dict, Optional, Any, Callable, List, Iterable, TYPE_CHECKING
from dashboard_services.api import (
    _fetch_league,
    get_nfl_games_for_week_raw,
    get_transactions,
    get_rosters,
    get_users,
    get_traded_picks,
    get_nfl_state,
    get_nfl_players, fetch_team_game_logs_html, fetch_tank_boxscore, get_matchups,
)
from dashboard_services.display_names import team_label_from_user, username_from_user

# ======================================================================
# From utils/utils.py (split per consolidation map)
# ======================================================================

# --- utils/utils.py L993 ---
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

# --- utils/utils.py L1005 ---
def streak_class(row) -> str:
    typ = (row.get("StreakType") or "").upper()
    ln = int(row.get("StreakLen") or 0)
    if typ == "W" and ln >= 2:
        return "streak-hot"
    if typ == "L" and ln >= 2:
        return "streak-cold"
    return ""

# --- utils/utils.py L1015 ---
def streak_edge_class(streak) -> str:
    """Granular edge-signal class for a streak string like 'W3' / 'L2'.

    Returns ``streak-w1`` / ``streak-w2`` / ``streak-w3`` / ``streak-w4plus``
    (or the ``streak-l*`` equivalents); ``''`` when the streak is empty or
    unparseable. W1/L1 are intentionally included so the caller can render
    them as a whisper; 4+ caps the intensity scale.
    """
    m = re.match(r"^([WLwl])\s*(\d+)\s*$", str(streak or "").strip())
    if not m:
        return ""
    n = int(m.group(2))
    if n <= 0:
        return ""
    side = "w" if m.group(1).upper() == "W" else "l"
    band = "1" if n == 1 else "2" if n == 2 else "3" if n == 3 else "4plus"
    return f"streak-{side}{band}"

# --- utils/utils.py L1370 ---
BEFORE_WINDOW = 5 * 60  # 5 minutes

# --- utils/utils.py L1371 ---
IN_WINDOW = 3 * 60 * 60

# --- utils/utils.py L1374 ---
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

# --- utils/utils.py L1434 ---
def game_has_started(game: Optional[dict], now: datetime | None = None) -> bool:
    """True when a Tank01/schedule game is live or final.

    Explicit ``gameStatusCode`` 0 (scheduled) normally wins even if a stale
    ``gameTime_epoch`` would otherwise look like the game already ended —
    that is what painted last year's box scores on the Week 1 preview.

    Exception: when ``gameDate`` is a calendar day before today, treat the
    game as started/final even if Tank01 still reports code 0. Matchup rows
    already date-correct the game line to "Final"; without this exception the
    box-score line stays blank after Thursday night while the schedule lags.
    """
    if not game or not isinstance(game, dict):
        return False
    if now is None:
        now = datetime.now(timezone.utc)
    code = str(game.get("gameStatusCode") or "").strip()
    if code in ("1", "2"):
        return True
    if code == "0":
        game_date = str(game.get("gameDate") or "")[:8]
        if len(game_date) == 8 and game_date.isdigit():
            # Local calendar day — same basis as format_team_game_line's today_str.
            today_str = (
                now.astimezone().strftime("%Y%m%d")
                if getattr(now, "tzinfo", None)
                else now.strftime("%Y%m%d")
            )
            if game_date < today_str:
                return True
        return False
    return normalize_game_status_from_tank01(game, now=now) in ("in", "post")

# --- utils/utils.py L1468 ---
def finished_game_ids_for_week(season: int, week: int) -> list[str]:
    """IDs of the week's games that are final (Tank01/schedule status ``post``).

    Used by the live advanced-metrics refresh to tell "a new game just finished"
    from "nothing changed", so a rebuild happens the moment a slate goes final
    rather than at the next daily cron. Any load/parse failure yields ``[]``.
    """
    from utils.data_cache import load_week_schedule
    try:
        sched = load_week_schedule(int(season), int(week)) or []
    except Exception:
        return []
    out: list[str] = []
    for g in sched:
        if not isinstance(g, dict):
            continue
        if normalize_game_status_from_tank01(g) == "post":
            gid = g.get("gameID") or g.get("gameId") or g.get("id")
            if gid:
                out.append(str(gid))
    return sorted(set(out))

# --- utils/utils.py L1490 ---
def week_has_final_game(season: int, week: int) -> bool:
    """True when at least one of the week's games has gone final."""
    if not week or int(week) < 1:
        return False
    return bool(finished_game_ids_for_week(season, week))

# --- utils/utils.py L1497 ---
def resolve_adv_metrics_completed_week(season: int, current_week: int) -> int:
    """Highest week whose finished games should feed the metrics snapshot.

    Returns ``current_week`` once that in-progress week has at least one final
    game (so a just-completed slate is included immediately), otherwise
    ``current_week - 1`` (the last fully-finished week). Never below 0. Both the
    live refresh and the daily cron call this so they agree on which week the
    day's snapshot row covers and neither regresses the other's write.
    """
    cw = max(0, int(current_week or 0))
    if cw >= 1 and week_has_final_game(season, cw):
        return cw
    return max(0, cw - 1)

# --- utils/utils.py L1512 ---
def build_games_by_team(games: list[dict]) -> dict[str, dict]:
    """
    games -> { team_abbr: { 'status': 'pre' | 'in' | 'post', 'game': game_obj } }
    """
    from utils.nfl import team_abbr_keys
    games_by_team: dict[str, dict] = {}
    for g in games:
        home = g.get("home")  # e.g. "NE"
        away = g.get("away")  # e.g. "NYJ"
        norm_status = normalize_game_status_from_tank01(g)  # 'pre' | 'in' | 'post'
        entry = {"status": norm_status, "game": g}

        for raw in (home, away):
            if not raw:
                continue
            for key in team_abbr_keys(raw):
                games_by_team[key] = entry

    return games_by_team

# --- utils/utils.py L1532 ---
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
    from utils.nfl import lookup_team_map
    from utils.data_cache import STATUS_NOT_STARTED
    from utils.data_cache import STATUS_IN_PROGRESS
    from utils.data_cache import STATUS_FINAL
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

        game = lookup_team_map(games_by_team, team)
        if not game:
            if not games_by_team:
                # Schedule data missing entirely — assume games haven't started
                # so projections are shown instead of wall-to-wall 0.0 actuals.
                status_by_pid[pid] = STATUS_NOT_STARTED
            else:
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

        game = lookup_team_map(games_by_team, team_code)

        if not game:
            if not games_by_team:
                status_by_pid[pid] = STATUS_NOT_STARTED
            else:
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

# --- utils/utils.py L1617 ---
def build_status_for_week(
        season: int,
        week: int,
        players_index: dict[str, dict],
        teams_index: dict[str, dict],
        idp_player_index: dict[str, dict] = None,
) -> dict[str, str]:
    games = get_nfl_games_for_week(week, season)
    games_by_team = build_games_by_team(games)
    return build_status_by_pid(players_index, games_by_team, teams_index, week,
                               idp_player_index if idp_player_index else None)

# --- utils/utils.py L1630 ---
def decorate_player_display(player: dict) -> dict:
    from utils.data_cache import STATUS_NOT_STARTED
    from utils.data_cache import STATUS_IN_PROGRESS
    from utils.data_cache import STATUS_FINAL
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

# --- utils/utils.py L1666 ---
def get_nfl_games_for_week(
        week: int,
        season: int,
        season_type: str = "reg",
) -> list[dict]:
    from utils.data_cache import get_week_schedule_cached
    return get_week_schedule_cached(
        season=season,
        week=week,
        fetch_fn=get_nfl_games_for_week_raw,
        season_type=season_type,
    )

# --- utils/utils.py L1679 ---
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
    from utils.nfl import def_team_logo_urls
    from utils.nfl import canon_team
    info = players_index.get(pid, {})
    team_info = teams_index.get(pid, {})

    # name from your players_index, fallback to pid
    name = info.get("name") or pid

    # nfl team code (BAL, DET, BUF, etc.) — WAS, never WSH
    nfl = info.get("team") or team_info.get("team") or (pid if pid in teams_index else None)
    if nfl:
        nfl = canon_team(nfl) or nfl

    # position (string)
    pos = ""
    if players and pid in players:
        player_obj = players[pid]  # full Sleeper dict
        pos = player_obj.get("pos") or player_obj.get("position") or ""
    elif pid in teams_index:
        pos = "DEF"

    out = {
        "pid": pid,
        "name": name,
        "pos": pos,
        "nfl": nfl,
    }
    # DEF/DST: surface the NFL team logo so callers can render a crest instead
    # of a missing Sleeper headshot (DEF ids are team abbreviations).
    if str(pos or "").upper() in ("DEF", "DST", "D/ST") and nfl:
        _local, _espn = def_team_logo_urls(nfl)
        if _espn:
            out["logo"] = _espn
            out["logo_local"] = _local
    elif team_info.get("Logo"):
        # Team-abbr pid looked up purely from teams_index.
        out["logo"] = team_info.get("Logo")
        out["logo_local"] = f"/static/images/team_logos/{(nfl or pid)}.png"
    return out

# --- utils/utils.py L1732 ---
def build_teams_overview(
        rosters: List[dict],
        users_list: List[dict],
        picks_by_roster: Dict[str, List[dict]],
        players: Dict[str, dict],
        players_index: Dict[str, dict],
        teams_index: Dict[str, dict],
        platform: str,
) -> List[dict]:
    teams_ctx: List[dict] = []
    users_by_id = {str(u["user_id"]): u for u in users_list}
    users_by_rid = {str(u.get("roster_id")): u for u in users_list if u.get("roster_id") is not None}

    def normalize_pos(pos: str) -> str:
        p = (pos or "").strip().upper()
        if p == "PK":
            return "K"
        if p in ("D/ST", "DST", "DEF"):
            return "DEF"
        return p

    # Traditional ESPN-ish ordering target
    SLOT_ORDER = ["QB", "RB", "RB", "WR", "WR", "TE", "FLEX", "K", "DEF"]
    ORDER_RANK = {"QB": 0, "RB": 1, "WR": 2, "TE": 3, "FLEX": 4, "K": 5, "DEF": 6}

    def sort_starters_espn(starter_pids: List[str]) -> List[str]:
        """
        Reorders the given starter player IDs into:
        QB, RB, RB, WR, WR, TE, FLEX, K, DEF (best-effort).
        Extras get appended in a stable order.
        """
        # Build (pid, pos) list
        enriched = []
        for pid in starter_pids:
            p = pinfo_for_pid(pid, players_index, teams_index, players) or {}
            pos = normalize_pos(p.get("pos") or p.get("position") or "")
            enriched.append((pid, pos))

        # Buckets
        buckets = {"QB": [], "RB": [], "WR": [], "TE": [], "K": [], "DEF": [], "OTHER": []}
        for pid, pos in enriched:
            if pos in buckets:
                buckets[pos].append(pid)
            else:
                buckets["OTHER"].append(pid)

        ordered: List[str] = []

        # Fill fixed slots in the classic order
        used = set()

        def take(bucket_key: str) -> str | None:
            arr = buckets.get(bucket_key, [])
            while arr:
                pid = arr.pop(0)
                if pid not in used:
                    used.add(pid)
                    return pid
            return None

        def take_flex() -> str | None:
            # Prefer RB/WR/TE in that order for FLEX (you can swap priority if you want)
            for k in ("RB", "WR", "TE"):
                pid = take(k)
                if pid:
                    return pid
            return None

        for slot in SLOT_ORDER:
            if slot == "FLEX":
                pid = take_flex()
            else:
                pid = take(slot)
            if pid:
                ordered.append(pid)

        # Append any remaining starters (superflex, extra flex, IDP, etc.)
        leftovers: List[str] = []
        # remaining known buckets (in a sensible rank order)
        for key in ("QB", "RB", "WR", "TE", "K", "DEF"):
            for pid in buckets[key]:
                if pid not in used:
                    leftovers.append(pid)
                    used.add(pid)
        # others last
        for pid in buckets["OTHER"]:
            if pid not in used:
                leftovers.append(pid)
                used.add(pid)

        # Keep stable relative ordering for anything we didn't consume
        return ordered + leftovers

    _DISPLAY_POSITIONS = {"QB", "RB", "WR", "TE", "K", "DEF"}

    def enrich_list(pids: List[str]) -> List[dict]:
        result = []
        for pid in pids:
            p = pinfo_for_pid(pid, players_index, teams_index, players)
            pos = normalize_pos(p.get("pos") or p.get("position") or "")
            if not pos or pos in _DISPLAY_POSITIONS:
                result.append(p)
        return result

    for r in rosters:
        rid = str(r["roster_id"])
        owner_id = str(r.get("owner_id") or "")
        user = users_by_rid.get(rid) or users_by_id.get(owner_id, {})

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

        # ESPN: re-sort starters into traditional layout
        if (platform or "").lower().strip() == "espn":
            starters_pids = sort_starters_espn(list(starters_pids))

        starter_set = set(starters_pids)
        ir_set = set(ir_pids)
        taxi_set = set(taxi_pids)

        bench_pids = [
            pid for pid in players_pids
            if pid not in starter_set and pid not in ir_set and pid not in taxi_set
        ]

        teams_ctx.append({
            "roster_id": rid,
            "name": team_label_from_user(user, r, fallback=f"Team {rid}"),
            "username": username_from_user(user),
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
