from __future__ import annotations

import json
from datetime import datetime, date
from itertools import zip_longest
from pathlib import Path
from typing import Dict, List, Any, Optional

from dashboard_services.api import (
    get_matchups,
    get_users,
    get_rosters,
    avatar_from_users,
    get_nfl_scores_for_date,
    build_team_game_lookup, get_league_settings, get_bracket,
)
from dashboard_services.utils import write_json, load_week_schedule, load_teams_index, load_week_stats, normalize_name

STATUS_NOT_STARTED = "not_started"
STATUS_IN_PROGRESS = "in_progress"
STATUS_FINAL = "final"


def get_owner_id(rosters: Optional[list[dict]] = None, roster_id: Optional[str] = None) -> Optional[str]:
    return next((r["owner_id"] for r in rosters if str(r.get("roster_id")) == str(roster_id)), None)


def build_matchup_preview(
        league_id: str,
        week: int,
        roster_map: Dict[str, str],
        players_map: Dict[str, Dict[str, str]],
) -> List[dict]:
    mlist = get_matchups(league_id, week) or []
    if not mlist:
        return []

    # Pre-fetch users/rosters once instead of per team
    users = get_users(league_id) or []
    rosters = get_rosters(league_id) or []

    # Pull league settings to find playoff start week
    settings = get_league_settings()
    playoff_week_start = int(settings.get("playoff_week_start") or 0)

    # Brackets
    winners_bracket = get_bracket(league_id, "winners") or []
    losers_bracket  = get_bracket(league_id, "losers") or []

    # figure out league size from rosters / roster_map
    if rosters:
        num_teams = len({str(r.get("roster_id")) for r in rosters})
    else:
        num_teams = len(roster_map) if roster_map else 0

    # expected number of head-to-head games in regular season
    expected_matchups = max(1, num_teams // 2) if num_teams else None

    # Precompute maps for fast lookup
    owner_id_by_rid: Dict[str, Optional[str]] = {}
    record_by_rid: Dict[str, tuple[int, int]] = {}
    for r in rosters:
        rid_str = str(r.get("roster_id"))
        owner_id_by_rid[rid_str] = r.get("owner_id")
        r_settings = r.get("settings") or {}
        record_by_rid[rid_str] = (
            r_settings.get("wins", 0),
            r_settings.get("losses", 0),
        )

    username_by_owner: Dict[str, Optional[str]] = {
        u["user_id"]: u.get("display_name") for u in users if "user_id" in u
    }

    avatar_cache: Dict[Optional[str], Any] = {}

    def get_avatar(owner_id: Optional[str]) -> Any:
        if owner_id not in avatar_cache:
            avatar_cache[owner_id] = avatar_from_users(users, owner_id) if owner_id is not None else None
        return avatar_cache[owner_id]

    def _from_players_map(pid: str) -> Dict[str, str]:
        if players_map:
            info = players_map.get(pid)
        else:
            info = None
        if info:
            name = info.get("name") or pid
            nfl = info.get("team") or "FA"
            pos = info.get("pos") or (
                info.get("fantasy_positions", [""])[0]
                if info.get("fantasy_positions")
                else ""
            )
            return {"name": name, "nfl": nfl, "pos": pos}
        if pid.isalpha() and 2 <= len(pid) <= 3:
            return {"name": f"{pid} D/ST", "nfl": pid, "pos": "DEF"}
        return {"name": pid, "nfl": "FA", "pos": ""}

    def _pinfo(pid: str, pts_map: Dict[str, float]) -> dict:
        base = _from_players_map(pid)
        pts = pts_map.get(pid) if pts_map else None
        return {
            "pid": pid,
            "name": base["name"],
            "pos": base["pos"],
            "nfl": base["nfl"],
            "pts": pts,
        }

    def _team_block_from_match_row(row: dict) -> dict:
        rid = str(row.get("roster_id"))
        starters_raw = [s for s in (row.get("starters") or []) if s]
        pts_map = {str(k): v for k, v in (row.get("players_points") or {}).items()}
        s_infos: List[dict] = [_pinfo(str(pid), pts_map) for pid in starters_raw]
        pts_total = float(row["points"]) if isinstance(row.get("points"), (int, float)) else None

        wins, losses = record_by_rid.get(rid, (0, 0))
        owner_id = owner_id_by_rid.get(rid)
        username = username_by_owner.get(owner_id)

        return {
            "name": roster_map.get(rid, f"Roster {rid}"),
            "starters": s_infos,
            "pts_total": pts_total,
            "avatar": get_avatar(owner_id),
            "record": f"{wins}-{losses}",
            "username": username,
        }

    def _team_block_tbd(rid: Optional[int | str]) -> dict:
        rid_str = str(rid) if rid is not None else None
        name = "TBD"
        record = "-"

        if rid_str and rid_str in roster_map:
            name = roster_map[rid_str]
        if rid_str and rid_str in record_by_rid:
            w, l = record_by_rid[rid_str]
            record = f"{w}-{l}"

        owner_id = owner_id_by_rid.get(rid_str)
        return {
            "name": name,
            "starters": [],
            "pts_total": None,
            "avatar": get_avatar(owner_id) if owner_id else None,
            "record": record,
            "username": username_by_owner.get(owner_id) if owner_id else None,
        }

    # ------------------------------------------------------------------
    # PLAYOFF BRANCH – winners + losers bracket
    # ------------------------------------------------------------------
    is_playoff_week = (
        bool(playoff_week_start)
        and week >= playoff_week_start
        and (winners_bracket or losers_bracket)
    )

    if is_playoff_week:
        # Map roster_id -> matchup row for *this* fantasy week
        by_rid: Dict[str, dict] = {}
        for row in mlist:
            rid_str = str(row.get("roster_id"))
            if rid_str not in by_rid:
                by_rid[rid_str] = row

        # Combine winners + losers for result resolution
        all_brackets = list(winners_bracket) + list(losers_bracket)

        # result_by_match[m] = {"w": roster_id or None, "l": roster_id or None}
        result_by_match: Dict[int, Dict[str, Optional[int]]] = {}

        for b in all_brackets:
            try:
                mid = int(b.get("m"))
            except (TypeError, ValueError):
                continue

            entry = result_by_match.setdefault(mid, {"w": None, "l": None})

            w_team = b.get("w")
            l_team = b.get("l")

            if w_team is not None:
                try:
                    entry["w"] = int(w_team)
                except (TypeError, ValueError):
                    pass

            if l_team is not None:
                try:
                    entry["l"] = int(l_team)
                except (TypeError, ValueError):
                    pass

        def _resolve_slot(b: dict, slot_key: str, from_key: str) -> Optional[int]:
            """
            Resolve t1 / t2 from either a direct value or a from-spec:
            - tX: direct roster id
            - tX_from: {"w": match_no} or {"l": match_no}
            """
            direct = b.get(slot_key)
            if direct is not None:
                try:
                    return int(direct)
                except (TypeError, ValueError):
                    return None

            from_spec = b.get(from_key)
            if not isinstance(from_spec, dict) or not from_spec:
                return None

            if "w" in from_spec:
                prev_m = from_spec["w"]
                try:
                    prev_m = int(prev_m)
                except (TypeError, ValueError):
                    return None
                return result_by_match.get(prev_m, {}).get("w")

            if "l" in from_spec:
                prev_m = from_spec["l"]
                try:
                    prev_m = int(prev_m)
                except (TypeError, ValueError):
                    return None
                return result_by_match.get(prev_m, {}).get("l")

            return None

        # Round mapping: playoff_week_start -> round 1, etc.
        current_round = (week - playoff_week_start) + 1

        def _build_round_matchups(bracket_list: List[dict]) -> List[dict]:
            out_matches: List[dict] = []
            for b in bracket_list:
                if b.get("r") != current_round:
                    continue

                mid = b.get("m")

                t1_rid = _resolve_slot(b, "t1", "t1_from")
                t2_rid = _resolve_slot(b, "t2", "t2_from")

                left_row = by_rid.get(str(t1_rid)) if t1_rid is not None else None
                right_row = by_rid.get(str(t2_rid)) if t2_rid is not None else None

                if left_row is not None:
                    left = _team_block_from_match_row(left_row)
                else:
                    left = _team_block_tbd(t1_rid)

                if right_row is not None:
                    right = _team_block_from_match_row(right_row)
                else:
                    right = _team_block_tbd(t2_rid)

                out_matches.append({
                    "matchup_id": mid,
                    "left": left,
                    "right": right,
                })
            return out_matches

        # Build matches for both winners & losers in this round
        playoff_out: List[dict] = []
        playoff_out.extend(_build_round_matchups(winners_bracket))
        playoff_out.extend(_build_round_matchups(losers_bracket))

        # In playoffs we usually want *all* games (winners + losers),
        # so do NOT cap by expected_matchups here.
        return playoff_out

    # ------------------------------------------------------------------
    # REGULAR SEASON BRANCH – existing logic
    # ------------------------------------------------------------------
    by_mid: Dict[Any, List[dict]] = {}
    for m in mlist:
        mid = m.get("matchup_id")
        by_mid.setdefault(mid, []).append(m)

    out = []
    for mid, rows in by_mid.items():
        if not rows:
            continue
        rows_sorted = sorted(rows, key=lambda r: str(r.get("roster_id")))
        left = _team_block_from_match_row(rows_sorted[0])
        right = (
            _team_block_from_match_row(rows_sorted[1])
            if len(rows_sorted) > 1
            else {
                "name": "TBD",
                "avatar": None,
                "starters": [],
                "pts_total": None,
                "record": "-",
                "username": None,
            }
        )
        out.append({"matchup_id": mid, "left": left, "right": right})

    # In regular season, we can still cap by expected_matchups if desired
    if expected_matchups is not None:
        return out[:expected_matchups]
    return out



def render_matchup_carousel_weeks(
    slides_by_week: dict[int, str],
    dashboard: bool,
    active_week: int | None = None,
) -> str:
    """
    Render a single matchup carousel card.

    slides_by_week: {week: "<div class='m-slide'>...</div>..."}
    active_week: which week's slides to show inside the track.
    """
    if not slides_by_week:
        slides_html = "<div class='m-empty'>No matchups</div>"
    else:
        # pick active week if given, else first key
        if active_week is None:
            active_week = sorted(slides_by_week.keys())[0]
        slides_html = slides_by_week.get(active_week) or "<div class='m-empty'>No matchups</div>"

    central = "central" if dashboard else ""
    style = "max-width:800px;" if not dashboard else ""

    return f"""
      <div class="card matchup-carousel {central}" data-section="matchups" style="{style} margin-bottom:30px;">
        <div class="m-nav">
          <h2>Matchup Preview</h2>
          <div class="m-controls">
            <button class="m-btn m-btn-prev" type="button">‹ Prev</button>
            <button class="m-btn m-btn-next" type="button">Next ›</button>
          </div>
        </div>
        <div class="m-carousel">
          <div class="m-track">
            {slides_html}
          </div>
        </div>
      </div>
    """



def add_bye_weeks_to_players():
    player_path = Path("cache/players_index.json")
    team_path = Path("cache/teams_index.json")
    with open(player_path, "r") as file:
        players = json.load(file)
    with open(team_path, "r") as file:
        teams = json.load(file)
    for player_id, player_data in players.items():
        team_abv = player_data.get("team")
        if team_abv in teams:
            player_data["byeWeek"] = teams[team_abv]["byeWeek"]

    write_json(player_path, players)


def team_live_totals(
        team: dict,
        status_by_pid: dict[str, str],
        projections: dict[str, float],
) -> tuple[float, float]:
    """
    actual_total:
        sum of all actual points for starters (p['pts'])
    live_proj_total:
        - players not started  -> use projection
        - players started/finished -> use actual
    """
    actual_total = 0.0
    live_proj_total = 0.0

    starters = team.get("starters") or []
    projections = projections or {}

    for p in starters:
        pid = p.get("pid")

        actual = float(p.get("pts") or 0.0)
        actual_total += actual

        status = status_by_pid.get(pid, STATUS_NOT_STARTED)
        proj_val = float(projections.get(pid, 0.0))

        # Use == for strings, not `is`
        if status in (STATUS_IN_PROGRESS, STATUS_FINAL):
            # started or finished → use actual
            live_proj_total += actual
        else:
            # not started (or unknown) → use projection
            live_proj_total += proj_val

    return actual_total, live_proj_total



def compute_team_projections_for_weeks(
        matchups_by_week: dict[int, list[dict]],
        statuses_by_week: dict[int, dict],
        projections_by_week: dict[int, dict],
        roster_map: dict[str, str],  # roster_id -> owner name
) -> dict[tuple[int, str], float]:
    """
    Returns {(week, roster_id): live_proj_total}

    Assumes:
      - statuses_by_week: {week: {"statuses": {pid: status_str}}}
      - projections_by_week: {week: {"projections": {pid: proj_val}}}
      - matchups_by_week: {week: [ { "left": {...}, "right": {...} }, ... ]}
    """
    proj_by_roster: dict[tuple[int, str], float] = {}

    # reverse: owner display name -> roster_id (fallback if roster_id missing)
    owner_to_rid = {owner: rid for rid, owner in roster_map.items()}

    for week, matchups in matchups_by_week.items():
        # per-week statuses
        week_status_bundle = statuses_by_week.get(week) or {}
        week_status_by_pid = (week_status_bundle.get("statuses") or {}) if isinstance(week_status_bundle, dict) else {}

        # per-week projections
        if isinstance(projections_by_week, dict):
            week_proj_container = projections_by_week.get(week) or {}
            week_proj_map = week_proj_container.get("projections") or {}
        else:
            # fallback – treat as already a flat {pid: proj_val}
            week_proj_map = projections_by_week or {}

        for m in matchups:
            for side in ("left", "right"):
                team = m.get(side) or {}
                rid = team.get("roster_id")

                # fallback if roster_id not in the team obj
                if rid is None:
                    rid = owner_to_rid.get(team.get("name", ""))

                if rid is None:
                    continue

                _, live_proj_total = team_live_totals(
                    team,
                    week_status_by_pid,
                    week_proj_map,
                )
                proj_by_roster[(week, str(rid))] = live_proj_total

    return proj_by_roster



def build_team_schedule_lookup(games: List[dict]) -> Dict[str, dict]:
    """
    Given a list of game dicts from Tank01 getNFLGamesForWeek,
    build a lookup: team_abv -> that week's game dict.

    Each team appears at most once per week, so mapping is safe.
    """
    lookup: Dict[str, dict] = {}
    for g in games:
        home = (g["home"] or "").upper()
        away = (g["away"] or "").upper()

        if home:
            lookup[home] = g
        if away:
            lookup[away] = g

    return lookup


def parse_game_datetime(game_time_str: str) -> datetime:
    """
    Convert Tank01 game date/time into a real datetime object.

    game_date: "20251204"
    game_time_str: "8:15p" or "1:00a" (Tank01 style)
    """
    time_str = game_time_str.strip().lower()

    # Add missing "m"
    if time_str.endswith("a") or time_str.endswith("p"):
        time_str += "m"  # "8:15p" → "8:15pm"

    dt = datetime.strptime(f"{time_str}", "%I:%M%p")
    return dt


def build_defense_rankings(teams_index: dict) -> dict:
    """
    Returns a dictionary ranking all teams by defensive metrics:
      - opp_pass_yds_pg (lower = better)
      - opp_rush_yds_pg (lower = better)

    Output:
      {
        "ATL": {"opp_pass_yds_pg": 14, "opp_rush_yds_pg": 17},
        "DAL": {...},
        ...
      }
    """
    pass_list = []
    rush_list = []

    for abbr, info in teams_index.items():
        opp_pass = info.get("opp_pass_yds_pg")
        opp_rush = info.get("opp_rush_yds_pg")

        if opp_pass is not None:
            pass_list.append((abbr, float(opp_pass)))
        if opp_rush is not None:
            rush_list.append((abbr, float(opp_rush)))

    pass_sorted = sorted(pass_list, key=lambda x: x[1])
    rush_sorted = sorted(rush_list, key=lambda x: x[1])

    rankings = {abbr: {} for abbr in teams_index.keys()}

    for rank, (abbr, _) in enumerate(pass_sorted, start=1):
        rankings[abbr]["opp_pass_yds_pg"] = rank

    for rank, (abbr, _) in enumerate(rush_sorted, start=1):
        rankings[abbr]["opp_rush_yds_pg"] = rank

    return rankings


def format_player_stats(
        teams_stats: Dict[str, Dict[str, Dict[str, Dict[str, Any]]]],
        team: str,
        pos: str,
        player: str,
) -> Optional[str]:
    """
    Returns a compact stat line with no player name, e.g.:
      "4 rec, 61 yds, 1 td"

    For offensive positions (QB/RB/WR/TE) this behaves exactly as before.
    For IDP, all defensive positions (DL/DE/DT/LB/DB/CB/S, etc.) are pulled
    from the single "IDP" bucket per team in week_stats.

    Drops any 0-value TD lines and skips empty lines.
    """
    # Map raw position -> bucket used in week_stats
    defensive_positions = {
        "DL", "DE", "DT",      # line
        "EDGE",
        "LB", "ILB", "OLB",    # linebackers
        "DB", "CB", "S", "FS", "SS",  # secondary
    }

    lookup_pos = "IDP" if pos in defensive_positions or pos == "IDP" else pos

    team_data = teams_stats.get(team)
    if not team_data:
        return None

    pos_data = team_data.get(lookup_pos)
    if not pos_data:
        return None

    player_stats = pos_data.get(normalize_name(player))
    if not player_stats:
        return None

    def phrase(v: int | float, singular: str, plural: str) -> str:
        v_int = int(v)
        return f"{v_int} {singular if v_int == 1 else plural}"

    parts: list[str] = []

    # ---------------- QB / RB / WR / TE (unchanged behavior) ----------------
    if lookup_pos == "QB":
        py = player_stats.get("pass_yds", 0)
        ptd = player_stats.get("pass_td", 0)
        ints = player_stats.get("int", 0)
        ra = player_stats.get("rush_att", 0)
        ry = player_stats.get("rush_yds", 0)
        rtd = player_stats.get("rush_td", 0)

        if py:
            parts.append(phrase(py, "pass yd", "pass yds"))
        if ptd > 0:
            parts.append(phrase(ptd, "td", "tds"))
        if ints:
            parts.append(phrase(ints, "int", "ints"))
        if ra:
            parts.append(phrase(ra, "car", "car"))
        if ry:
            parts.append(phrase(ry, "rush yd", "rush yds"))
        if rtd > 0:
            parts.append(phrase(rtd, "rush td", "rush tds"))

    elif lookup_pos in {"RB", "WR", "TE"}:
        ra = player_stats.get("rush_att", 0)
        ry = player_stats.get("rush_yds", 0)
        rtd = player_stats.get("rush_td", 0)
        rec = player_stats.get("rec", 0)
        rec_yds = player_stats.get("rec_yds", 0)
        rec_td = player_stats.get("rec_td", 0)

        if rec:
            parts.append(phrase(rec, "rec", "rec"))
        if rec_yds:
            parts.append(phrase(rec_yds, "yd", "yds"))
        if rec_td > 0:
            parts.append(phrase(rec_td, "td", "tds"))

        if ra:
            parts.append(phrase(ra, "car", "car"))
        if ry:
            parts.append(phrase(ry, "yd", "yds"))
        if rtd > 0:
            parts.append(phrase(rtd, "td", "tds"))

    # ---------------- IDP branch (new) ----------------
    elif lookup_pos == "IDP":
        # Common Sleeper IDP fields from your example + typical ones
        tkl = player_stats.get("idp_tkl", 0)
        tkl_solo = player_stats.get("idp_tkl_solo", 0)
        tkl_ast = player_stats.get("idp_tkl_ast", 0)
        qb_hit = player_stats.get("idp_qb_hit", 0)
        ff = player_stats.get("idp_ff", 0) or player_stats.get("idp_forced_fum", 0)
        sack = (
            player_stats.get("idp_sack")
            or player_stats.get("idp_sk")
            or player_stats.get("idp_sacks")
            or 0
        )
        int_def = player_stats.get("idp_int", 0)
        pd = player_stats.get("idp_pd", 0) or player_stats.get("idp_pass_def", 0)

        # Tackles
        if tkl:
            # total tackles line, e.g. "2 tkl"
            parts.append(phrase(tkl, "tkl", "tkl"))
            # optional breakdown in parentheses " (1 solo, 1 ast)"
            breakdown_bits = []
            if tkl_solo:
                breakdown_bits.append(phrase(tkl_solo, "solo", "solo"))
            if tkl_ast:
                breakdown_bits.append(phrase(tkl_ast, "ast", "ast"))
            if breakdown_bits:
                parts[-1] += f" ({', '.join(breakdown_bits)})"

        # Sacks
        if sack:
            parts.append(phrase(sack, "sack", "sacks"))

        # Forced fumbles
        if ff:
            parts.append(phrase(ff, "FF", "FF"))

        # QB hits
        if qb_hit:
            parts.append(phrase(qb_hit, "QB hit", "QB hits"))

        # Interceptions
        if int_def:
            parts.append(phrase(int_def, "int", "ints"))

        # Passes defended
        if pd:
            parts.append(phrase(pd, "PD", "PD"))

        # Fallback: if we somehow didn't pick anything but the dict has numbers,
        # add a compact "key=value" dump to avoid returning "no stats".
        if not parts:
            for k, v in player_stats.items():
                if isinstance(v, (int, float)) and v != 0:
                    parts.append(f"{k}={int(v)}")

    # ---------------- Other positions (unchanged generic behavior) ----------------
    else:
        for k, v in player_stats.items():
            if isinstance(v, int) and v != 0:
                parts.append(f"{k}={v}")

    if not parts:
        return "no stats"

    return ", ".join(parts)



def build_offense_rankings(teams_index: dict) -> dict:
    """
    Returns a dictionary ranking all teams by offensive metrics:
      - rush_yds_rank  (higher rush_yds_pg = better, rank 1 is best)
      - pass_yds_rank  (higher pass_yds_pg = better, rank 1 is best)
      - total_off_rank (combined yards + TDs, rank 1 is best)

    teams_index example:
      {
        "ARI": {
          "rush_yds_pg": 100.6,
          "pass_yds_pg": 236.7,
          "rush_td_pg": 0.75,
          "pass_td_pg": 1.58,
          ...
        },
        ...
      }

    Output:
      {
        "ARI": {
          "rush_yds_rank": 14,
          "pass_yds_rank": 10,
          "total_off_rank": 8,
        },
        "ATL": {...},
        ...
      }
    """

    TD_WEIGHT = 40.0  # treat 1 TD per game ~ 40 yards; tweak if desired

    rush_list = []
    pass_list = []
    total_list = []

    for abbr, info in teams_index.items():
        rush_yds = info.get("rush_yds_pg")
        pass_yds = info.get("pass_yds_pg")
        rush_td = info.get("rush_td_pg")
        pass_td = info.get("pass_td_pg")

        # rushing yards list
        if rush_yds is not None:
            rush_list.append((abbr, float(rush_yds)))

        # passing yards list
        if pass_yds is not None:
            pass_list.append((abbr, float(pass_yds)))

        # total offense list (need at least both yardage numbers)
        if rush_yds is not None and pass_yds is not None:
            r_y = float(rush_yds)
            p_y = float(pass_yds)
            r_td = float(rush_td) if rush_td is not None else 0.0
            p_td = float(pass_td) if pass_td is not None else 0.0

            total_yards = r_y + p_y
            tds_pg = r_td + p_td
            total_score = total_yards + TD_WEIGHT * tds_pg

            total_list.append((abbr, total_score))

    # Sort: higher is better for offense
    rush_sorted = sorted(rush_list, key=lambda x: x[1], reverse=True)
    pass_sorted = sorted(pass_list, key=lambda x: x[1], reverse=True)
    total_sorted = sorted(total_list, key=lambda x: x[1], reverse=True)

    rankings = {abbr: {} for abbr in teams_index.keys()}

    for rank, (abbr, _) in enumerate(rush_sorted, start=1):
        rankings[abbr]["rush_yds_rank"] = rank

    for rank, (abbr, _) in enumerate(pass_sorted, start=1):
        rankings[abbr]["pass_yds_rank"] = rank

    for rank, (abbr, _) in enumerate(total_sorted, start=1):
        rankings[abbr]["total_off_rank"] = rank

    return rankings


def render_matchup_slide(
        season: str,
        m: dict,
        w: int,
        proj_week: int,
        status_by_pid: dict[str, str],
        projections: dict[str, float],
        players: dict,
        teams: dict,
        team_game_lookup: dict,
) -> str:
    """One slide with rows like:
       [Left Name] [Left Pts/Proj] [Right Pts/Proj] [Right Name]
    """
    proj = w > proj_week

    # Heavy stuff: do once per call
    teams_index = load_teams_index()
    defense_ranks = build_defense_rankings(teams_index)
    offense_ranks = build_offense_rankings(teams_index)
    week_stats = load_week_stats(season, w)
    team_schedule_lookup = build_team_schedule_lookup(load_week_schedule(season, w))

    # Projections for this week (dict {pid: proj_val})
    if isinstance(projections, dict):
        week_proj_container = projections.get(w) or {}
        week_proj_map = week_proj_container.get("projections") or {}
    else:
        week_proj_map = projections or {}

    # Cache NFL score lookups per date
    score_cache: dict[str, dict] = {}

    def get_team_game_from_scores(game_date_str: str, team_abv: str) -> Optional[dict]:
        """
        Lazily fetch scores for a given date once, then reuse for all players.
        """
        if not game_date_str:
            return None
        if game_date_str not in score_cache:
            scores_body = get_nfl_scores_for_date(game_date_str)
            score_cache[game_date_str] = build_team_game_lookup(scores_body) if scores_body else {}
        return score_cache[game_date_str].get(team_abv)

    today_str = date.today().strftime("%Y%m%d")
    now_dt = datetime.now()

    def team_head(t, proj_mode: bool):
        ava = t.get("avatar") or ""
        img = f"<img class='avatar' src='{ava}' onerror=\"this.style.display='none'\">" if ava else ""

        if not proj_mode:
            points = f"{t['pts_total']:.2f}" if isinstance(t.get("pts_total"), (int, float)) else "—"
            return f"""
        <div class="m-team">
          {img}
          <div>
            <div class="name left">{t['name']}</div>
            <div>{t['record']} • @{t['username']}</div>
          </div>
          <span class="num">{points}</span>
        </div>
        """

        actual_total, live_proj_total = team_live_totals(
            t,
            status_by_pid,
            week_proj_map,
        )

        return f"""
        <div class="m-team">
          {img}
          <div>
            <div class="name left">{t['name']}</div>
            <div>{t['record']} • @{t['username']}</div>
          </div>
          <div style="display:grid;grid-template-columns:1;justify-items: center;">
            <span class="num">{actual_total:.1f}</span>
            <span class="proj" style="opacity:0.4;text-align:center;">{live_proj_total:.1f}</span>
          </div>
        </div>
        """

    def team_head_2nd(t, proj_mode: bool):
        ava = t.get("avatar") or ""
        img = f"<img class='avatar' src='{ava}' onerror=\"this.style.display='none'\">" if ava else ""

        if not proj_mode:
            points = f"{t['pts_total']:.2f}" if isinstance(t.get("pts_total"), (int, float)) else "—"
            return f"""
        <div class="m-team">
          <span class="num">{points}</span>
          <div class="right">
            <div class="name">{t['name']}</div>
            <div>@{t['username']} • {t['record']}</div>
          </div>
          {img}
        </div>
        """

        actual_total, live_proj_total = team_live_totals(
            t,
            status_by_pid,
            week_proj_map,
        )
        return f"""
        <div class="m-team">
          <div style="display:grid;grid-template-columns:1;justify-items: center;">
            <span class="num">{actual_total:.1f}</span>
            <span class="proj" style="opacity:0.4;text-align:center;">{live_proj_total:.1f}</span>
          </div>
          <div class="right">
            <div class="name">{t['name']}</div>
            <div>@{t['username']} • {t['record']}</div>
          </div>
          {img}
        </div>
        """

    def format_team_game_line(team_abv: str, game: dict, pos: str, side: str) -> str:
        if not team_abv or not game:
            return ""

        home = str(game.get("home") or "").upper()
        away = str(game.get("away") or "").upper()
        t_up = team_abv.upper()
        if t_up not in (home, away):
            return ""

        is_home = (t_up == home)
        opp = away if is_home else home
        status_code = str(game.get("gameStatusCode") or "0")  # '0' scheduled, '1' live, '2' final
        game_date = str(game.get("gameDate") or game.get("gameID", "")[:8])  # '20251204'
        game_time = str(game.get("gameTime") or "")  # '8:15p'

        # quick status correction by date
        if game_date < today_str:
            status_code = "2"
        elif game_date == today_str and game_time:
            try:
                if parse_game_datetime(game_time) > now_dt:
                    # future kick within same date – treat as scheduled
                    status_code = "0"
            except ValueError:
                pass

        if status_code == "0":
            dow = ""
            if game_date:
                try:
                    dt = datetime.strptime(game_date, "%Y%m%d")
                    dow = dt.strftime("%a")
                except ValueError:
                    pass

            display_time = game_time
            if display_time.endswith("p"):
                display_time = display_time[:-1] + " pm"
            elif display_time.endswith("a"):
                display_time = display_time[:-1] + " am"

            opp_ranks = defense_ranks.get(opp, {})
            off_ranks = offense_ranks.get(opp, {})

            opp_rank = None
            if pos in ["QB", "WR", "TE"]:
                opp_rank = opp_ranks.get("opp_pass_yds_pg")
            elif pos == "RB":
                opp_rank = opp_ranks.get("opp_rush_yds_pg")
            elif pos == "DEF":
                opp_rank = off_ranks.get("total_off_rank")

            suffix = f" (#{opp_rank})" if opp_rank is not None else ""
            prefix = ("@ " + opp + suffix) if not is_home else ("vs " + opp + suffix)
            return " ".join(x for x in [dow, display_time, prefix] if x).strip()

        # For live/final, pull from scores API once per date
        game_date_std = game_date  # already YYYYMMDD
        score_str = ""
        score_game = get_team_game_from_scores(game_date_std, team_abv)

        if score_game:
            if is_home:
                my_pts = score_game.get("homePts")
                opp_pts = score_game.get("awayPts")
            else:
                my_pts = score_game.get("awayPts")
                opp_pts = score_game.get("homePts")

            if my_pts is not None and opp_pts is not None:
                score_str = f"{my_pts}-{opp_pts}"

        if status_code == "1":
            line_score = game.get("lineScore") or {}
            period = line_score.get("period", "")
            clock = game.get("gameClock", "")
            prefix = "@ " + opp if not is_home else "vs " + opp
            extra = " ".join(x for x in [period, clock, prefix] if x).strip()

            if side == "right":
                return f"{score_str} {extra} <span class='live-dot'></span>".strip()
            return f"<span class='live-dot'></span>{score_str} {extra}".strip()

        if status_code == "2":
            prefix = "@ " + opp if not is_home else "vs " + opp
            if score_str:
                return f"Final {prefix} {score_str}"
            return "Final"

        return ""

    def player_bits(
            p,
            side: str,
            left_side: bool,
    ):
        if not p:
            return "", 0.0, None, False, None

        pid = p.get("pid")
        name = p.get("name", "")
        nfl = p.get("nfl", "")
        pos = p.get("pos")
        if pos not in ["QB","RB", "WR", "TE", "K", "DEF"]:
            pos = "IDP"
        if pos == "IDP":
            team_stats = week_stats.get(nfl, {})
            pos_data = team_stats.get(pos, {})
            player_stats = pos_data.get(normalize_name(name),{})
            actual = player_stats.get('pts_idp', 0.0)
        else:
            actual = p.get("pts") or 0.0

        proj_val = week_proj_map.get(pid, 0.0)
        is_bye = False

        player_index = players.get(pid) or teams.get(pid)
        if player_index:
            if proj_val == 0.0 and player_index.get("byeWeek") == w:
                is_bye = True

        status = status_by_pid.get(pid, STATUS_NOT_STARTED)
        if status == "BYE":
            is_bye = True

        # decide what to show
        if is_bye:
            display_actual = 0.0
            display_proj = None
        elif status == STATUS_NOT_STARTED:
            display_actual = 0.0
            display_proj = proj_val
        elif status == STATUS_IN_PROGRESS:
            display_actual = actual
            display_proj = proj_val
        elif status == STATUS_FINAL:
            display_actual = actual
            display_proj = None
        else:
            display_actual = 0.0 if actual is None else actual
            display_proj = proj_val

        # game / stats
        game_line = ""
        stats = None
        if nfl:
            team_code = str(nfl).upper()
            game = None

            if team_game_lookup:
                game = team_game_lookup.get(team_code)
            if game is None and team_schedule_lookup:
                game = team_schedule_lookup.get(team_code)

            # normalized name (special-case Ken Walker)
            lookup_name = "ken walker" if name == "Kenneth Walker" else name

            if game:
                game_line = format_team_game_line(team_code, game, pos, side)
            stats = format_player_stats(
                week_stats,
                team_code,
                pos,
                lookup_name,
            )

        meta_content = f"&nbsp;{nfl}"

        if left_side:
            if is_bye:
                cell = (
                    f"<div class='p {side}' style='opacity:0.4;'>"
                    f"<span class='pos-badge {pos}'>{pos}</span>"
                    f"<span class='pname'>{name}</span>"
                    f"<span class='meta'>{meta_content}</span>"
                    f"</div>"
                )
            else:
                cell = (
                    f"<div class='p {side}'>"
                    f"<span class='pos-badge {pos}'>{pos}</span>"
                    f"<div style='display: flex;flex-direction: column;'>"
                    f"<div><span class='pname'>{name} </span>"
                    f"<span class='meta'> {meta_content}</span></div>"
                    f"<span class='meta'>{game_line}</span></div>"
                    f"</div>"
                )
        else:
            if is_bye:
                cell = (
                    f"<div class='p {side}' style='justify-content:flex-end; opacity:0.4;'>"
                    f"<span class='meta'>{meta_content}</span>"
                    f"<span class='pname'>{name}</span>"
                    f"<span class='pos-badge {pos}'>{pos}</span>"
                    f"</div>"
                )
            else:
                cell = (
                    f"<div class='p {side}' style='justify-content:flex-end;'>"
                    f"<div style='display:flex;flex-direction:column-reverse;'>"
                    f"<span class='meta'>{game_line}</span>"
                    f"<div><span class='meta'>{meta_content} </span>"
                    f"<span class='pname'> {name}</span></div></div>"
                    f"<span class='pos-badge {pos}'>{pos}</span>"
                    f"</div>"
                )

        return cell, float(display_actual), display_proj, is_bye, (stats if stats else None)

    rows_html: List[str] = []

    for L, R in zip_longest(
            m["left"].get("starters", []),
            m["right"].get("starters", []),
            fillvalue=None,
    ):
        left_cell, left_actual, left_proj, left_is_bye, left_stats = player_bits(
            L, "left", True
        )
        right_cell, right_actual, right_proj, right_is_bye, right_stats = player_bits(
            R, "right", False
        )

        la = 0.0 if left_is_bye else left_actual
        ra = 0.0 if right_is_bye else right_actual

        left_more = la > ra
        right_more = ra > la

        def score_stack(actual_val, proj_val, side: str, is_bye: bool, more: bool) -> str:
            if is_bye:
                return (
                    "<div class='num-stack' style='display:grid'>"
                    f"<span class='num mid {side}' style='opacity:0.4;'>BYE</span>"
                    "</div>"
                )
            if proj_val is None:
                cls = f"num mid {side}" + (" more" if more else "")
                return (
                    "<div class='num-stack' style='display:grid'>"
                    f"<span class='{cls}'>{actual_val:.1f}</span>"
                    "</div>"
                )

            cls_actual = f"num mid {side}" + (" more" if more else "")
            return (
                "<div class='num-stack' style='display:grid'>"
                f"<span class='{cls_actual}'>{actual_val:.1f}</span>"
                f"<span class='num mid {side} proj' style='opacity:0.4;'>{proj_val:.1f}</span>"
                "</div>"
            )

        def stat_stack(stats, side: str) -> str:
            if stats is None:
                return "<div></div>"
            if side == "left":
                return (
                    "<div class='p right' style='display: grid;'>"
                    f"<span class='meta' style='display:flex;justify-content:flex-end'>{stats}</span></div>"
                )
            return (
                "<div class='p left'>"
                f"<span class='meta'>{stats}</span></div>"
            )

        left_points_html = score_stack(left_actual, left_proj, "l", left_is_bye, left_more)
        right_points_html = score_stack(right_actual, right_proj, "r", right_is_bye, right_more)
        points = f"{left_points_html}{right_points_html}"

        rows_html.append(
            f"""<div class="m-row">
                  {left_cell}
                  {stat_stack(left_stats, "left")}
                  {points}
                  {stat_stack(right_stats, "right")}
                  {right_cell}
                </div>"""
        )

    return f"""
    <div class="m-slide">
      <div class="m-head">
        {team_head(m['left'], proj)}
        <div class="m-vs">vs</div>
        {team_head_2nd(m['right'], proj)}
      </div>
      <div class="m-body">
        <div class="m-combo">
          {''.join(rows_html)}
        </div>
      </div>
    </div>
    """
