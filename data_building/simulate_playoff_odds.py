"""
Monte Carlo playoff odds simulator.

Three modes:
  preseason / offseason  - no game data yet; project team strength from
                           each team's current roster player values, then
                           simulate the full regular season.
  in-season              - fetch remaining schedule from the API and simulate
                           from the current standings using each team's
                           historical scoring distribution (avg ± std).
  complete               - season is over; return 100 / 0 based on actual
                           final standings (no simulation needed).

All simulation paths are vectorised across n_sims with NumPy so 10 000 runs
finish in under a second.
"""
from __future__ import annotations

import logging
from collections import defaultdict
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

_MIN_STD      = 8.0    # floor on std dev
_N_SIMS       = 10_000
_BASE_STD     = 20.0   # preseason std dev - no per-team variance data yet
_BENCH_SLOTS  = {"BN", "IR", "TAXI"}

# Conservative defaults for players with no prior-season stats (rookies, etc.)
# These are realistic first-year averages in a typical PPR league.
_ROOKIE_PPG: dict[str, float] = {
    "QB": 14.0,  # few rookies start; those who do land around here
    "RB": 7.5,   # split backfields / limited role
    "WR": 6.5,   # steep learning curve
    "TE": 4.5,   # slowest position to develop
}
_ROOKIE_PPG_DEFAULT = 6.0  # catch-all for K, DEF, unknown


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def simulate_playoff_odds(
    ctx: dict,
    platform: str = "sleeper",
    n_sims: int = _N_SIMS,
    seed: Optional[int] = None,
) -> list[dict]:
    """
    Return playoff odds for every team in the league.

    Each result dict:
        roster_id, team_name, wins, losses, ties,
        playoff_pct, bye_pct, first_seed_pct, miss_pct,
        avg_final_wins, avg_final_losses, n_sims, is_complete, is_projected
    """
    settings           = ctx.get("league_settings") or {}
    playoff_week_start = int(settings.get("playoff_week_start") or 15)
    playoff_teams      = int(settings.get("playoff_teams") or 6)
    current_week       = int(ctx.get("current_week") or 0)
    season             = int(ctx.get("season") or 0)
    league_id          = str(ctx.get("league_id") or "")
    regular_season_end = playoff_week_start - 1

    team_stats = ctx.get("team_stats")
    has_games  = team_stats is not None and not team_stats.empty

    # ── Case 1: no game data yet (preseason / offseason of new year) ─────────
    if not has_games:
        teams = _estimate_from_rosters(ctx)
        if not teams:
            return []
        remaining_weeks = list(range(1, playoff_week_start))
        matchups_by_week = _fetch_remaining_schedule(
            platform, league_id, season, remaining_weeks
        )
        if not matchups_by_week:
            matchups_by_week = _random_schedule(teams, remaining_weeks, seed)
        result = _run_mc(teams, matchups_by_week, playoff_teams, n_sims, seed)
        for r in result:
            r["is_projected"] = True
        return result

    # ── Case 2: season complete ───────────────────────────────────────────────
    teams = _build_teams(team_stats)
    if not teams:
        return []

    if current_week > regular_season_end:
        return _actual_results(teams, playoff_teams)

    # ── Case 3: in-season projection ─────────────────────────────────────────
    remaining_weeks = list(range(current_week + 1, playoff_week_start))
    matchups_by_week = _fetch_remaining_schedule(
        platform, league_id, season, remaining_weeks
    )
    if not matchups_by_week:
        matchups_by_week = _random_schedule(teams, remaining_weeks, seed)

    result = _run_mc(teams, matchups_by_week, playoff_teams, n_sims, seed)
    for r in result:
        r["is_projected"] = False
    return result


# ---------------------------------------------------------------------------
# Roster-value projection (preseason / offseason)
# ---------------------------------------------------------------------------

_LINEUP_EFFICIENCY = 1.0    # position-aware selection already handles constraints
_FLEX_POSITIONS    = {"FLEX", "WR/RB/TE", "RB/WR/TE", "W/R/T"}
_FLEX_ELIGIBLE     = {"RB", "WR", "TE"}
_SUPER_FLEX_POS    = {"SUPER_FLEX", "SUPERFLEX", "QB/WR/RB/TE", "OP"}
_SUPER_FLEX_ELIG   = {"QB", "RB", "WR", "TE"}


def _position_aware_lineup(
    pids: list,
    ppg_map: dict,
    pos_map: dict,
    roster_positions: list,
) -> float:
    """
    Project weekly scoring using position-constrained optimal lineup.

    Fills fixed position slots (QB, RB, WR, TE, K, DEF) first, then fills
    FLEX/SuperFlex slots with the best remaining eligible player.
    Applies a small efficiency discount for bye weeks / injuries / suboptimal
    starts so the result reflects expected avg rather than theoretical max.
    """
    # Tally starting slots by type
    fixed_slots: dict[str, int] = {}
    flex_slots   = 0
    sflex_slots  = 0
    for slot in roster_positions:
        s = str(slot).upper()
        if s in _BENCH_SLOTS:
            continue
        if s in _SUPER_FLEX_POS:
            sflex_slots += 1
        elif s in _FLEX_POSITIONS:
            flex_slots += 1
        else:
            fixed_slots[s] = fixed_slots.get(s, 0) + 1

    # Resolve each player to (pos, ppg)
    by_pos: dict[str, list[float]] = {}
    for pid in pids:
        info = ppg_map.get(str(pid))
        if info:
            pos = info["pos"]
            ppg = info["ppg"] if info["ppg"] > 0 else _ROOKIE_PPG.get(info["pos"], _ROOKIE_PPG_DEFAULT)
        else:
            pos = pos_map.get(str(pid), "")
            ppg = _ROOKIE_PPG.get(pos, _ROOKIE_PPG_DEFAULT)
        if pos:
            by_pos.setdefault(pos, []).append(ppg)
    for pos in by_pos:
        by_pos[pos].sort(reverse=True)

    used: dict[str, int] = {}
    total = 0.0

    # Fill fixed slots
    for slot_pos, count in fixed_slots.items():
        pool = by_pos.get(slot_pos, [])
        for _ in range(count):
            i = used.get(slot_pos, 0)
            if i < len(pool):
                total += pool[i]
            used[slot_pos] = used.get(slot_pos, 0) + 1

    # Fill FLEX (RB/WR/TE eligible)
    flex_pool = sorted(
        (ppg for pos in _FLEX_ELIGIBLE for ppg in by_pos.get(pos, [])[used.get(pos, 0):]),
        reverse=True,
    )
    for i in range(flex_slots):
        if i < len(flex_pool):
            total += flex_pool[i]
            # mark one slot used for accounting (best-effort)
    remaining_after_flex = flex_pool[flex_slots:]

    # Fill SuperFlex (QB/RB/WR/TE eligible)
    sflex_pool = sorted(
        list(by_pos.get("QB", [])[used.get("QB", 0):]) + remaining_after_flex,
        reverse=True,
    )
    for i in range(sflex_slots):
        if i < len(sflex_pool):
            total += sflex_pool[i]

    return total * _LINEUP_EFFICIENCY


def _estimate_from_rosters(ctx: dict) -> list[dict]:
    """
    Build synthetic team scoring profiles from prior-season actual PPG.

    For each player on a roster, looks up their PPG from last season's usage
    cache (same source as the player modal). Players with no prior-season stats
    fall back to a conservative position-based default.

    Uses position-aware lineup selection so the projected avg reflects the
    actual starting constraints (1 QB, 2 RB, etc.) rather than best-ball.
    A small efficiency discount accounts for bye weeks, injuries, and
    suboptimal lineup decisions.
    """
    rosters          = ctx.get("rosters") or []
    roster_map       = ctx.get("roster_map") or {}
    roster_positions = ctx.get("roster_positions") or []

    if not rosters:
        return []

    # Detect scoring format from league settings
    rec_pts = float((ctx.get("scoring_settings") or {}).get("rec") or 0)
    if rec_pts >= 1.0:
        ppg_key = "ppr_ppg"
    elif rec_pts >= 0.5:
        ppg_key = "half_ppr_ppg"
    else:
        ppg_key = "std_scoring_ppg"

    # Load per-player PPG from the same usage_rows cache the player modal uses.
    # Try current year first, fall back to prior year (mirrors player modal logic).
    ppg_map: dict[str, dict] = {}   # player_id → {ppg, pos}
    try:
        import os as _os, json as _json
        from datetime import date as _date
        _cache_dir = _os.path.join(_os.path.dirname(__file__), "..", "cache", "player_history")
        _year = _date.today().year
        _usage_data = None
        for _y in [_year, _year - 1]:
            _path = _os.path.join(_cache_dir, f"usage_rows_{_y}.json")
            if _os.path.exists(_path):
                with open(_path) as _f:
                    _usage_data = _json.load(_f)
                break
        if _usage_data:
            for p in _usage_data:
                pid = str(p.get("id") or "")
                if not pid:
                    continue
                ppg = float((p.get("usage") or {}).get(ppg_key) or 0)
                pos = str(p.get("position") or "").upper()
                ppg_map[pid] = {"ppg": ppg, "pos": pos}
    except Exception as exc:
        logger.warning("[playoff_odds] Could not load usage_rows cache: %s", exc)

    # Load player positions from DB as fallback for rookies not in usage table
    pos_map: dict[str, str] = {}
    try:
        from dashboard_services.db import get_conn
        with get_conn() as conn:
            rows = conn.execute(
                "SELECT player_id, position FROM player_values WHERE position IS NOT NULL"
            ).fetchall()
        pos_map = {str(r["player_id"]): str(r["position"]).upper() for r in rows}
    except Exception:
        pass

    if not ppg_map and not pos_map:
        return []

    # Build projected avg for each roster using position-aware lineup selection
    teams: list[dict] = []
    for roster in rosters:
        rid  = roster.get("roster_id")
        pids = roster.get("players") or []

        projected_avg = _position_aware_lineup(
            pids, ppg_map, pos_map, roster_positions
        )

        name = (
            roster_map.get(str(rid))
            or roster_map.get(int(rid) if rid is not None else -1)
            or f"Team {rid}"
        )
        teams.append({
            "roster_id": int(rid),
            "name":      name,
            "wins":      0,
            "losses":    0,
            "ties":      0,
            "pf":        0.0,
            "avg":       round(projected_avg, 1),
            "std":       _BASE_STD,
        })
    return teams


# ---------------------------------------------------------------------------
# In-season team data from team_stats DataFrame
# ---------------------------------------------------------------------------

def _build_teams(team_stats) -> list[dict]:
    teams = []
    for rid in team_stats.index:
        row    = team_stats.loc[rid]
        wins   = int(row.get("Wins",   0) or 0)
        losses = int(row.get("Losses", 0) or 0)
        ties   = int(row.get("Ties",   0) or 0)
        pf     = float(row.get("PF",   0) or 0)
        avg    = float(row.get("AVG",  80) or 80)
        std    = max(float(row.get("STD", 15) or 15), _MIN_STD)
        teams.append({
            "roster_id": int(rid),
            "name":      str(row.get("owner", f"Team {rid}")),
            "wins":  wins, "losses": losses, "ties": ties,
            "pf": pf, "avg": avg, "std": std,
        })
    return teams


# ---------------------------------------------------------------------------
# Completed-season result (100 % / 0 %)
# ---------------------------------------------------------------------------

def _actual_results(teams: list[dict], playoff_teams: int) -> list[dict]:
    n_byes   = 2 if playoff_teams >= 4 else 0
    ranked   = sorted(teams, key=lambda t: (-t["wins"], -t["pf"]))
    made     = {t["roster_id"] for t in ranked[:playoff_teams]}
    bye_set  = {t["roster_id"] for t in ranked[:n_byes]}
    top_seed = ranked[0]["roster_id"] if ranked else None

    return [{
        "roster_id":        t["roster_id"],
        "team_name":        t["name"],
        "wins":             t["wins"],
        "losses":           t["losses"],
        "ties":             t["ties"],
        "playoff_pct":      100.0 if t["roster_id"] in made    else 0.0,
        "bye_pct":          100.0 if t["roster_id"] in bye_set else 0.0,
        "first_seed_pct":   100.0 if t["roster_id"] == top_seed else 0.0,
        "miss_pct":         0.0   if t["roster_id"] in made    else 100.0,
        "avg_final_wins":   float(t["wins"]),
        "avg_final_losses": float(t["losses"]),
        "n_sims":           0,
        "is_complete":      True,
        "is_projected":     False,
    } for t in teams]


# ---------------------------------------------------------------------------
# Schedule helpers
# ---------------------------------------------------------------------------

def _fetch_remaining_schedule(
    platform: str,
    league_id: str,
    season: int,
    weeks: list[int],
) -> dict[int, list[tuple[int, int]]]:
    if not league_id or not weeks:
        return {}
    try:
        from dashboard_services.platform_api import get_matchups
    except ImportError:
        return {}

    result: dict[int, list[tuple[int, int]]] = {}
    for week in weeks:
        try:
            raw = get_matchups(platform, league_id, week, season) or []
        except Exception:
            continue
        by_mid: dict = defaultdict(list)
        for m in raw:
            mid = m.get("matchup_id")
            rid = m.get("roster_id")
            if mid and rid is not None:
                by_mid[mid].append(int(rid))
        pairs = [(v[0], v[1]) for v in by_mid.values() if len(v) == 2]
        if pairs:
            result[week] = pairs
    return result


def _random_schedule(
    teams: list[dict],
    weeks: list[int],
    seed: Optional[int],
) -> dict[int, list[tuple[int, int]]]:
    rng = np.random.default_rng(seed)
    ids = [t["roster_id"] for t in teams]
    return {
        week: [
            (sh[i], sh[i + 1])
            for sh in [rng.permutation(ids).tolist()]
            for i in range(0, len(sh) - 1, 2)
        ]
        for week in weeks
    }


# ---------------------------------------------------------------------------
# Monte Carlo engine (vectorised)
# ---------------------------------------------------------------------------

def _run_mc(
    teams: list[dict],
    matchups_by_week: dict[int, list[tuple[int, int]]],
    playoff_teams: int,
    n_sims: int,
    seed: Optional[int],
) -> list[dict]:
    rng = np.random.default_rng(seed)
    n   = len(teams)
    idx = {t["roster_id"]: i for i, t in enumerate(teams)}

    avgs = np.array([t["avg"] for t in teams], dtype=np.float32)
    stds = np.array([t["std"] for t in teams], dtype=np.float32)

    wins = np.tile([t["wins"] for t in teams], (n_sims, 1)).astype(np.float32)
    pf   = np.tile([t["pf"]   for t in teams], (n_sims, 1)).astype(np.float32)

    n_byes          = 2 if playoff_teams >= 4 else 0
    remaining_weeks = len(matchups_by_week)

    for week_pairs in matchups_by_week.values():
        for (rid_a, rid_b) in week_pairs:
            ia = idx.get(rid_a)
            ib = idx.get(rid_b)
            if ia is None or ib is None:
                continue
            sa = np.maximum(
                rng.normal(avgs[ia], stds[ia], n_sims).astype(np.float32), 0
            )
            sb = np.maximum(
                rng.normal(avgs[ib], stds[ib], n_sims).astype(np.float32), 0
            )
            a_wins = sa > sb
            wins[:, ia] += a_wins.astype(np.float32)
            wins[:, ib] += (~a_wins).astype(np.float32)
            pf[:, ia]   += sa
            pf[:, ib]   += sb

    # Rank by wins desc, pf desc (wins dominate)
    rank_key  = wins * 1e6 + pf
    team_rank = np.argsort(np.argsort(-rank_key, axis=1), axis=1)  # 0 = best

    in_playoffs = (team_rank < playoff_teams).mean(axis=0) * 100
    got_bye     = (team_rank < n_byes).mean(axis=0) * 100 if n_byes else np.zeros(n)
    is_first    = (team_rank == 0).mean(axis=0) * 100

    init_wins   = np.array([t["wins"]   for t in teams], dtype=np.float32)
    init_losses = np.array([t["losses"] for t in teams], dtype=np.float32)
    avg_wins    = wins.mean(axis=0)
    avg_losses  = init_losses + remaining_weeks - (avg_wins - init_wins)

    return [{
        "roster_id":        t["roster_id"],
        "team_name":        t["name"],
        "wins":             t["wins"],
        "losses":           t["losses"],
        "ties":             t["ties"],
        "playoff_pct":      round(float(in_playoffs[i]),  1),
        "bye_pct":          round(float(got_bye[i]),      1),
        "first_seed_pct":   round(float(is_first[i]),     1),
        "miss_pct":         round(100 - float(in_playoffs[i]), 1),
        "avg_final_wins":   round(float(avg_wins[i]),     1),
        "avg_final_losses": round(float(avg_losses[i]),   1),
        "sim_avg":          round(float(avgs[i]),         1),
        "sim_std":          round(float(stds[i]),         1),
        "n_sims":           n_sims,
        "is_complete":      False,
    } for i, t in enumerate(teams)]
