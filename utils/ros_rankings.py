"""Rest-of-season (ROS) rankings: forward production for the rest of the year.

One computation, three consumers: the /players ROS mode (fields stamped on
the /api/league-players payload by app.py's overlay), and the public
/rankings/rest-of-season pages. Never a second opinion: every surface reads
the map this module builds.

Formula (backtested on the 2025 season, ~/workspace/ros-rankings/
BACKTEST_RESULTS.md; pooled MAE 2.374 vs 2.452 projection-only-everywhere
and 2.584 for the originally pitched 65/35 blend):

  * RB / WR / TE: ROS PPG = forward projection PPG alone. Sleeper's forward
    weekly projections already incorporate recent form, so blending form in
    again double-counts it over a rest-of-season window.
  * QB: ROS PPG = 40% forward projection PPG + 30% last-4-played form PPG
    + 30% season PPG. A thin form sample (< 4 played games in the window)
    shrinks the form weight toward the projection, the same shrinkage the
    start-score base blend uses.
  * ROS Pts = ROS PPG x projected games left. Games left counts the window
    weeks where the player's team plays (non-bye) and a projection feed
    exists, minus the leading zero-run of weeks out for an injured returner
    (the waiver path's weeks-out read on the projection series).
  * Two sub-windows ride along on the same weekly numbers: Stretch (window
    start through Week 14) and Playoffs (Weeks 15-17), each with its own
    PPG / total / games and positional rank. Weeks-out and bye handling
    apply inside each window independently.

Eligibility (the playoff boards' gates): at least 5 games played this
season AND a real role: the team's #1 at his position (by season points),
or RB >= 12 touches/game, or WR/TE >= 5.5 targets/game. IR / PUP / NFI /
suspended players are excluded entirely. A player who fails any gate gets
NO ROS fields, so the board hides him in ROS mode rather than ranking him
on a number the model does not stand behind.

Schedule rank is display only: the average Schedule Rankings matchup value
of the player's remaining opponents, ranked across teams (1 = softest
remaining schedule). It never enters the score: the weekly projections
already price the opponent in.

Cross-position order (the All view) is by VOR total: (window PPG minus
the replacement PPG at his position) x window games, with the replacement
convention in ros_replacement_index below. The All view's tier dividers
are computed from those VOR totals (compute_ros_vor_tier_thresholds), NOT
from the positional totals: a mixed list keyed to raw totals stamps each
player with his own position's scale, so the dividers repeat and run out
of order (a QB whose raw total is elite-scale but whose VOR is mid-pack
reads T1 in the middle of the list).

Scoring basis is standard PPR (rec = 1.0), the same basis the public pages
state and the backtest used. The compute reads only the cached files the
start-score bundle already loads (weekly stat files, weekly projection
files, schedules): it never fetches and never computes per page request.
``load_ros_map`` is signature-cached (file mtimes), so the overlay in
app.py pays the pool-wide pass at most once per data change, and the
overlay's own memo then serves every request.
"""
from __future__ import annotations

import logging
import os
from datetime import date
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

ROS_POSITIONS = ("QB", "RB", "WR", "TE")
ROS_END_WEEK = 17
PLAYOFF_START_WEEK = 15
STRETCH_END_WEEK = 14

MIN_GAMES_PLAYED = 5
RB_MIN_TOUCHES_PER_GAME = 12.0
REC_MIN_TARGETS_PER_GAME = 5.5

QB_W_PROJ = 0.40
QB_W_FORM = 0.30
QB_W_SEASON = 0.30
FORM_WINDOW = 4

PPR_SCORING = {"rec": 1.0}

# Season-ending / unavailable statuses. A plain "Out" is a weekly status,
# not a season verdict: those players stay eligible and their missed weeks
# come out of the games count through the projection zero-run instead.
SEASON_OUT_MARKERS = ("IR", "PUP", "NFI", "SUS")


# ── Pure pieces (unit-tested directly) ──────────────────────────────────────

def is_season_out(injury_status) -> bool:
    status = str(injury_status or "").upper()
    return any(marker in status for marker in SEASON_OUT_MARKERS)


def qb_ros_ppg(fwd_ppg: float, form_ppg: Optional[float],
               form_games: int, season_ppg: Optional[float]) -> float:
    """QB ROS PPG: 40% forward projection + 30% last-4 form + 30% season.

    The form weight scales with the games actually in the form window
    (form_games / 4) and the released weight lands on the projection, so a
    2-game form sample carries half the form weight. A missing form or
    season line folds its weight onto the projection as well.
    """
    fwd = float(fwd_ppg or 0.0)
    w_form = QB_W_FORM * max(0, min(FORM_WINDOW, int(form_games or 0))) / FORM_WINDOW
    if form_ppg is None:
        w_form = 0.0
    w_season = QB_W_SEASON if season_ppg is not None else 0.0
    w_proj = 1.0 - w_form - w_season
    return (w_proj * fwd
            + w_form * float(form_ppg or 0.0)
            + w_season * float(season_ppg or 0.0))


def window_projection_stats(series: List[Optional[float]]) -> Optional[dict]:
    """Forward PPG / total / games for one window from a weekly series.

    ``series`` is the player's projected points for the window's non-bye
    weeks in order, one entry per week whose projection feed exists; None
    means the player is absent from that week's feed. The leading run of
    ~zero-or-absent weeks is weeks out for an injured returner: those weeks
    leave the count entirely. Absent weeks after the return are skipped,
    not scored as zero games (the waiver forward-PPG read: only weeks where
    the player actually has a projection count). Returns None when no
    playable week remains.
    """
    from utils.waivers import weeks_out_from_projections

    if not series:
        return None
    weeks_out = weeks_out_from_projections(series, treat_missing_as_out=True)
    remaining = list(series[weeks_out:])
    present = [max(0.0, float(v)) for v in remaining if v is not None]
    if not present:
        return None
    games = len(present)
    total = sum(present)
    return {"fwd_ppg": total / games, "total": total, "games": games,
            "weeks_out": weeks_out}


def passes_role_gate(position: str, *, is_team_top: bool,
                     touches_per_game: Optional[float],
                     targets_per_game: Optional[float]) -> bool:
    """The playoff boards' role gate: team #1 at the position, or a real
    workload (RB >= ~12 touches/game, WR/TE >= ~5.5 targets/game). QBs pass
    only as their team's #1: a backup's projection spike is not a role."""
    if is_team_top:
        return True
    pos = str(position or "").upper()
    if pos == "RB":
        return touches_per_game is not None and touches_per_game >= RB_MIN_TOUCHES_PER_GAME
    if pos in ("WR", "TE"):
        return targets_per_game is not None and targets_per_game >= REC_MIN_TARGETS_PER_GAME
    return False


def compute_player_ros(position: str, *, played_games: int,
                       season_ppg: Optional[float], last4_ppg: Optional[float],
                       last4_games: int, touches_per_game: Optional[float],
                       targets_per_game: Optional[float], is_team_top: bool,
                       season_out: bool,
                       windows: Dict[str, List[Optional[float]]],
                       sched_rank: Optional[int]) -> Optional[dict]:
    """ROS fields for one player, or None when he is not ROS-eligible.

    ``windows`` maps "all" / "stretch" / "playoffs" to that window's weekly
    projection series (see window_projection_stats). Positional ranks are
    NOT set here: assign_ros_ranks fills them once the whole pool exists.
    A sub-window with no playable week left simply carries no fields.
    """
    pos = str(position or "").upper()
    if pos not in ROS_POSITIONS or season_out:
        return None
    if int(played_games or 0) < MIN_GAMES_PLAYED:
        return None
    if not passes_role_gate(pos, is_team_top=is_team_top,
                            touches_per_game=touches_per_game,
                            targets_per_game=targets_per_game):
        return None
    all_stats = window_projection_stats(windows.get("all") or [])
    if not all_stats or all_stats["total"] <= 0:
        return None

    out: dict = {}
    for window_key, prefix in (("all", "ros"), ("stretch", "ros_stretch"),
                               ("playoffs", "ros_playoff")):
        stats = all_stats if window_key == "all" else window_projection_stats(
            windows.get(window_key) or [])
        if not stats or stats["games"] <= 0:
            continue
        if pos == "QB":
            ppg = qb_ros_ppg(stats["fwd_ppg"], last4_ppg, last4_games, season_ppg)
        else:
            ppg = stats["fwd_ppg"]
        if ppg <= 0:
            continue
        out[f"{prefix}_ppg"] = round(ppg, 2)
        out[f"{prefix}_total"] = round(ppg * stats["games"], 1)
        out[f"{prefix}_games"] = stats["games"]
    if "ros_ppg" not in out:
        return None
    if sched_rank is not None:
        out["ros_sched_rank"] = int(sched_rank)
    return out


def assign_ros_ranks(ros_by_pid: Dict[str, dict],
                     positions_by_pid: Dict[str, str]) -> None:
    """Positional ranks per window, by that window's total (ties broken by
    PPG, then player id for stability). Mutates ``ros_by_pid`` in place."""
    for prefix in ("ros", "ros_stretch", "ros_playoff"):
        total_key, ppg_key, rank_key = (
            f"{prefix}_total", f"{prefix}_ppg", f"{prefix}_pos_rank")
        by_pos: Dict[str, List[Tuple[str, dict]]] = {}
        for pid, fields in ros_by_pid.items():
            if total_key not in fields:
                continue
            pos = str(positions_by_pid.get(pid) or "").upper()
            by_pos.setdefault(pos, []).append((pid, fields))
        for rows in by_pos.values():
            rows.sort(key=lambda kv: (-kv[1][total_key], -kv[1][ppg_key], kv[0]))
            for i, (_pid, fields) in enumerate(rows, 1):
                fields[rank_key] = i


def _ros_tier_thresholds_for_values(rows: List[Tuple[Optional[str], float]]) -> list:
    """Scale-normalized drop thresholds for one list of (position, value).

    The shared drop-based algorithm is calibrated for dynasty values (its
    span cap and fallback live around 1000), while ROS numbers top out
    near 200, so the values are scaled onto that range for the computation
    and the thresholds scaled back. A pool too thin or too smooth to tier
    returns [] (no dividers) rather than dynasty-scale numbers.
    """
    from utils.tier_thresholds import FALLBACK_THRESHOLDS, compute_tier_thresholds

    top = max((v for _pos, v in rows), default=0.0)
    if len(rows) < 6 or top <= 0:
        return []
    factor = 1000.0 / top
    remapped = [
        {"position": pos, "value": v * factor, "sf_value": v * factor}
        for pos, v in rows
    ]
    thresholds = compute_tier_thresholds(remapped, "1qb", 10)
    if len(thresholds) < 2 or list(thresholds) == list(FALLBACK_THRESHOLDS):
        return []
    return [round(v / factor, 1) for v in thresholds]


def compute_ros_tier_thresholds(ros_by_pid: Dict[str, dict],
                                positions_by_pid: Dict[str, str]) -> dict:
    """Tier thresholds from the ROS totals actually displayed, per window.

    Follows the redraft pattern (PR #3031): thresholds are computed on the
    same numbers the board sorts by, never the dynasty scale. These key
    the position-filtered views, which order by the window total.
    Returns {"all": [...], "stretch": [...], "playoffs": [...]}.
    """
    out: dict = {}
    for window_key, prefix in (("all", "ros"), ("stretch", "ros_stretch"),
                               ("playoffs", "ros_playoff")):
        total_key = f"{prefix}_total"
        rows = [(positions_by_pid.get(pid), fields[total_key])
                for pid, fields in ros_by_pid.items() if total_key in fields]
        out[window_key] = _ros_tier_thresholds_for_values(rows)
    return out


# ── Cross-position VOR: replacement convention + All-view tiers ─────────────
# One convention, read by every ROS surface: the public overall page calls
# ros_replacement_ppg directly, the payload's All-view tier thresholds are
# computed from ros_vor_totals, and the /players board mirrors the same
# index math in JS (prRosReplacementMap in static/rankings.js).
#
# Starters per team and the scarcity multipliers are the value model's
# (data_building/player_value.py: STARTERS with QB 2 in Superflex, and
# REPLACEMENT_MULT). ONE exception: QB in 1QB. The 1.5x QB multiplier is
# calibrated for the dynasty model's normalized strength scale, not raw
# points; applied to raw ROS PPGs it prices the replacement QB deep in the
# QB2 band, which leaves an elite QB's edge looking like an elite RB's.
# In a 1QB league the actual fallback is the streaming tier right at the
# starter line, so QB replacement in 1QB follows the site's VORP helpers
# instead (utils/trade.py stamp_value_metrics): the player AT the starter
# line, rank = starters x league size. SF keeps the multiplier: with two
# starters per team, QB scarcity is real and priced.
ROS_STARTERS = {"QB": 1, "RB": 2, "WR": 2, "TE": 1}
ROS_REPLACEMENT_MULT = {"QB": 1.50, "RB": 1.20, "WR": 1.20, "TE": 1.25}
ROS_LEAGUE_TYPES = ("1qb", "sf")
ROS_LEAGUE_SIZES = (8, 10, 12, 14)
_WINDOW_PREFIXES = (("all", "ros"), ("stretch", "ros_stretch"),
                    ("playoffs", "ros_playoff"))


def ros_replacement_index(position: str, league_size: int, league_type: str) -> int:
    """0-based index into a position's PPG-sorted (desc) pool locating the
    replacement-level player, for one league shape."""
    pos = str(position or "").upper()
    size = int(league_size or 10)
    starters = ROS_STARTERS.get(pos, 1)
    if pos == "QB" and str(league_type or "1qb").lower() == "sf":
        return int(2 * size * ROS_REPLACEMENT_MULT["QB"])
    if pos == "QB":
        # 1QB: the player at the starter line (utils/trade.py VORP rank).
        return int(round(starters * size)) - 1
    return int(starters * size * ROS_REPLACEMENT_MULT.get(pos, 1.2))


def ros_replacement_ppg(ppgs_desc: List[float], position: str,
                        league_size: int, league_type: str) -> Optional[float]:
    """Replacement PPG from a position's PPGs sorted desc (None if empty)."""
    if not ppgs_desc:
        return None
    idx = ros_replacement_index(position, league_size, league_type)
    idx = max(0, min(idx, len(ppgs_desc) - 1))
    return float(ppgs_desc[idx])


def ros_vor_totals(ros_by_pid: Dict[str, dict],
                   positions_by_pid: Dict[str, str],
                   league_size: int, league_type: str) -> Dict[str, Dict[str, float]]:
    """{window: {pid: VOR total}} for one league shape.

    VOR total = (window PPG - replacement PPG at his position) x window
    games: the All view's cross-position ordering metric.
    """
    out: Dict[str, Dict[str, float]] = {}
    for window_key, prefix in _WINDOW_PREFIXES:
        ppg_key, games_key = f"{prefix}_ppg", f"{prefix}_games"
        by_pos: Dict[str, List[float]] = {}
        for pid, fields in ros_by_pid.items():
            if ppg_key in fields:
                pos = str(positions_by_pid.get(pid) or "").upper()
                by_pos.setdefault(pos, []).append(float(fields[ppg_key]))
        repl: Dict[str, float] = {}
        for pos, ppgs in by_pos.items():
            ppgs.sort(reverse=True)
            base = ros_replacement_ppg(ppgs, pos, league_size, league_type)
            if base is not None:
                repl[pos] = base
        window_vor: Dict[str, float] = {}
        for pid, fields in ros_by_pid.items():
            if ppg_key not in fields or games_key not in fields:
                continue
            base = repl.get(str(positions_by_pid.get(pid) or "").upper())
            if base is None:
                continue
            window_vor[pid] = round(
                (float(fields[ppg_key]) - base) * int(fields[games_key]), 1)
        out[window_key] = window_vor
    return out


def compute_ros_vor_tier_thresholds(ros_by_pid: Dict[str, dict],
                                    positions_by_pid: Dict[str, str]) -> dict:
    """Tier thresholds for the All view, computed on VOR totals.

    The All view orders by VOR total, so its dividers must be thresholds
    on VOR totals: one monotonic T1..Tn sequence for the mixed list.
    Keyed per window x league type x league size, the same shape the
    client looks up for dynasty tiers:
    {"all": {"1qb": {"10": [...]}, "sf": {...}}, "stretch": ..., ...}.
    """
    out: dict = {}
    for window_key, _prefix in _WINDOW_PREFIXES:
        by_type: dict = {}
        for league_type in ROS_LEAGUE_TYPES:
            by_size: dict = {}
            for size in ROS_LEAGUE_SIZES:
                vors = ros_vor_totals(
                    ros_by_pid, positions_by_pid, size, league_type)[window_key]
                rows = [(positions_by_pid.get(pid), v) for pid, v in vors.items()]
                by_size[str(size)] = _ros_tier_thresholds_for_values(rows)
            by_type[league_type] = by_size
        out[window_key] = by_type
    return out


# ── Loader: cached files in, ROS map out ────────────────────────────────────

_ROS_CACHE: dict = {"key": None, "payload": None}


def _files_signature(season: int, start_week: int) -> tuple:
    """mtime signature over every cached file the ROS compute reads."""
    from utils.data_cache import CACHE_DIR, path_week_proj, path_week_schedule

    parts: List[Tuple[str, float]] = []
    stats_dir = os.path.join(CACHE_DIR, "sleeper_stats")
    try:
        for name in sorted(os.listdir(stats_dir)):
            if name.startswith(f"sleeper_stats_s{season}_w"):
                try:
                    parts.append((name, os.path.getmtime(os.path.join(stats_dir, name))))
                except OSError:
                    parts.append((name, -1.0))
    except OSError:
        pass
    for w in range(start_week, ROS_END_WEEK + 1):
        for path in (path_week_proj(season, w), path_week_schedule(season, w)):
            try:
                parts.append((os.path.basename(path), os.path.getmtime(path)))
            except OSError:
                parts.append((os.path.basename(path), -1.0))
    return tuple(parts)


def load_ros_map(season: Optional[int] = None) -> dict:
    """The full ROS payload block: per-player fields + window + thresholds.

    Returns {"players": {pid: fields}, "window": {"start_week", "end_week"},
    "tier_thresholds_ros": {...}, "tier_thresholds_ros_vor": {...},
    "updated": "Month D, YYYY"}. Empty players
    when the window has closed (past Week 17) or the cached feeds are not
    there yet: the board then simply has no ROS rows, never fabricated ones.
    Signature-cached; the heavy pass runs at most once per data change per
    process. Never raises.
    """
    try:
        if not season:
            from dashboard_services.api import get_nfl_state
            season = int((get_nfl_state() or {}).get("season") or date.today().year)
        season = int(season)
    except Exception:
        return {"players": {}, "window": None, "tier_thresholds_ros": {}, "tier_thresholds_ros_vor": {},
                "updated": None}
    try:
        from utils.season_qualification import qualification_policy
        completed = [int(w) for w in qualification_policy(season).completed_weeks]
    except Exception:
        completed = []
    start_week = (max(completed) + 1) if completed else 1
    updated = date.today().strftime("%B %d, %Y")
    if start_week > ROS_END_WEEK:
        return {"players": {}, "window": {"start_week": start_week,
                                          "end_week": ROS_END_WEEK},
                "tier_thresholds_ros": {}, "tier_thresholds_ros_vor": {},
                "updated": updated}

    key = (season, start_week, _files_signature(season, start_week))
    if _ROS_CACHE.get("key") == key and _ROS_CACHE.get("payload") is not None:
        return _ROS_CACHE["payload"]

    payload = _build_ros_map(season, start_week, updated)
    if payload is None:
        # Build failed: serve the empty shape for this call but do NOT
        # cache it, so the next request retries instead of pinning an
        # empty board to this data signature.
        return {"players": {}, "window": {"start_week": start_week,
                                          "end_week": ROS_END_WEEK},
                "tier_thresholds_ros": {}, "tier_thresholds_ros_vor": {},
                "updated": updated}
    _ROS_CACHE["key"] = key
    _ROS_CACHE["payload"] = payload
    return payload


def _build_ros_map(season: int, start_week: int, updated: str) -> Optional[dict]:
    try:
        from data_building.start_score_bundle import derive_start_sit_inputs
        from utils.data_cache import _read_week_projection_file
        from utils.fantasy_scoring import week_stat_points
        from utils.projections import weekly_projection_points
        from utils.utils import load_players_index, load_week_schedule
        from utils.schedule_ease import norm_sched_team
        import glob as _glob
        import json as _json
        from utils.data_cache import CACHE_DIR

        index = load_players_index() or {}
        played_inputs = derive_start_sit_inputs(season, dict(PPR_SCORING)) or {}

        # Usage + team-points pass over the completed weeks' stat files:
        # touches / targets per game for the role gate, and each team's #1
        # at a position by season points.
        usage: Dict[str, dict] = {}
        team_points: Dict[Tuple[str, str], Dict[str, float]] = {}
        stat_paths = sorted(_glob.glob(os.path.join(
            CACHE_DIR, "sleeper_stats", f"sleeper_stats_s{season}_w*.json")))
        for path in stat_paths:
            try:
                week = int(path.rsplit("_w", 1)[1].split(".")[0])
            except (IndexError, ValueError):
                continue
            if week >= start_week:
                continue  # only completed weeks carry played stats
            try:
                with open(path) as fh:
                    rows = _json.load(fh)
            except Exception:
                continue
            if not isinstance(rows, dict):
                continue
            for pid, st in rows.items():
                if not isinstance(st, dict) or st.get("pts_ppr") is None:
                    continue
                pid = str(pid)
                meta = index.get(pid) or {}
                pos = str(meta.get("pos") or meta.get("position") or "").upper()
                if pos not in ROS_POSITIONS:
                    continue
                u = usage.setdefault(pid, {"touches": 0.0, "targets": 0.0})
                u["touches"] += float(st.get("rush_att") or 0) + float(st.get("rec") or 0)
                u["targets"] += float(st.get("rec_tgt") or 0)
                team = str(meta.get("team") or "").upper()
                if team:
                    try:
                        pts = float(week_stat_points(st, dict(PPR_SCORING), ""))
                    except Exception:
                        pts = 0.0
                    bucket = team_points.setdefault((team, pos), {})
                    bucket[pid] = bucket.get(pid, 0.0) + pts
        team_top: set = set()
        for bucket in team_points.values():
            if bucket:
                team_top.add(max(bucket.items(), key=lambda kv: kv[1])[0])

        # Remaining schedules: team -> {week: opponent} over the window.
        team_opps: Dict[str, Dict[int, str]] = {}
        for w in range(start_week, ROS_END_WEEK + 1):
            try:
                games = load_week_schedule(season, w) or []
            except Exception:
                games = []
            for g in games:
                if not isinstance(g, dict):
                    continue
                home = norm_sched_team(str(g.get("home_team") or g.get("home") or ""))
                away = norm_sched_team(str(g.get("away_team") or g.get("away") or ""))
                if home and away:
                    team_opps.setdefault(home, {})[w] = away
                    team_opps.setdefault(away, {})[w] = home

        # ROS schedule rank per (position, team): average Schedule Rankings
        # matchup value of the remaining opponents, ranked across teams
        # (1 = softest). Display only.
        sched_rank: Dict[Tuple[str, str], int] = {}
        try:
            from app import _matchup_rank_table
            from utils.nfl import rank_values
            for pos in ROS_POSITIONS:
                _ranks, _total, info, _is_z = _matchup_rank_table(season, pos)
                team_values: Dict[str, float] = {}
                for team, opps in team_opps.items():
                    vals = []
                    for w in range(start_week, ROS_END_WEEK + 1):
                        opp = opps.get(w)
                        row = (info or {}).get(opp) if opp else None
                        if row and isinstance(row.get("rank_value"), (int, float)):
                            vals.append(float(row["rank_value"]))
                    if vals:
                        team_values[team] = sum(vals) / len(vals)
                ranks, _n = rank_values(team_values)
                for team, rank in ranks.items():
                    sched_rank[(pos, team)] = int(rank)
        except Exception:
            logger.debug("[ros] schedule rank unavailable", exc_info=True)
            sched_rank = {}

        # Weekly projection series per player, split into the three windows.
        week_maps: Dict[int, dict] = {}
        for w in range(start_week, ROS_END_WEEK + 1):
            week_maps[w] = _read_week_projection_file(season, w) or {}

        def _series(pid: str, pos: str, team: str, weeks) -> List[Optional[float]]:
            out: List[Optional[float]] = []
            for w in weeks:
                week_map = week_maps.get(w) or {}
                if not week_map:
                    continue  # feed not published for this week yet: not a week
                opps = team_opps.get(team)
                if opps is not None and w not in opps:
                    continue  # bye week: the team does not play
                out.append(weekly_projection_points(week_map, pid, dict(PPR_SCORING), pos))
            return out

        all_weeks = list(range(start_week, ROS_END_WEEK + 1))
        stretch_weeks = [w for w in all_weeks if w <= STRETCH_END_WEEK]
        playoff_weeks = [w for w in all_weeks if w >= PLAYOFF_START_WEEK]

        ros_by_pid: Dict[str, dict] = {}
        positions_by_pid: Dict[str, str] = {}
        for pid, meta in index.items():
            if not isinstance(meta, dict):
                continue
            pid = str(pid)
            pos = str(meta.get("pos") or meta.get("position") or "").upper()
            if pos not in ROS_POSITIONS:
                continue
            played = played_inputs.get(pid) or {}
            played_games = int(played.get("played_games") or 0)
            if played_games <= 0:
                continue
            team = str(meta.get("team") or "").upper()
            u = usage.get(pid) or {"touches": 0.0, "targets": 0.0}
            windows = {
                "all": _series(pid, pos, team, all_weeks),
                "stretch": _series(pid, pos, team, stretch_weeks),
                "playoffs": _series(pid, pos, team, playoff_weeks),
            }
            if not windows["all"]:
                continue
            fields = compute_player_ros(
                pos,
                played_games=played_games,
                season_ppg=played.get("season_ppg"),
                last4_ppg=played.get("last4_ppg"),
                last4_games=min(FORM_WINDOW, played_games),
                touches_per_game=(u["touches"] / played_games) if played_games else None,
                targets_per_game=(u["targets"] / played_games) if played_games else None,
                is_team_top=pid in team_top,
                season_out=is_season_out(meta.get("injury_status") or meta.get("status")),
                windows=windows,
                sched_rank=sched_rank.get((pos, norm_sched_team(team))),
            )
            if fields:
                ros_by_pid[pid] = fields
                positions_by_pid[pid] = pos

        assign_ros_ranks(ros_by_pid, positions_by_pid)
        return {
            "players": ros_by_pid,
            "window": {"start_week": start_week, "end_week": ROS_END_WEEK},
            "tier_thresholds_ros": compute_ros_tier_thresholds(ros_by_pid, positions_by_pid),
            "tier_thresholds_ros_vor": compute_ros_vor_tier_thresholds(ros_by_pid, positions_by_pid),
            "updated": updated,
        }
    except Exception:
        logger.exception("[ros] ROS map build failed")
        return None
