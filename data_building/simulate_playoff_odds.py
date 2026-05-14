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

import hashlib
import logging
import math
from collections import defaultdict
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

_MIN_STD      = 8.0    # floor on std dev
_N_SIMS       = 10_000
_BENCH_SLOTS  = {"BN", "IR", "TAXI"}

# Conservative defaults for players with no prior-season stats (rookies, etc.)
_ROOKIE_PPG: dict[str, float] = {
    "QB": 14.0,
    "RB": 7.5,
    "WR": 6.5,
    "TE": 4.5,
}
_ROOKIE_PPG_DEFAULT = 6.0

# Per-position weekly std dev model: std = multiplier * ppg + base
# Derived from empirical PPR weekly score distributions.
_POS_STD: dict[str, tuple[float, float]] = {
    "QB":  (0.28, 3.0),
    "RB":  (0.45, 2.0),
    "WR":  (0.50, 2.0),
    "TE":  (0.42, 1.5),
    "K":   (0.00, 4.0),
    "DEF": (0.00, 5.5),
}
_POS_STD_DEFAULT = (0.42, 2.0)

_FLEX_POSITIONS = {"FLEX", "WR/RB/TE", "RB/WR/TE", "W/R/T"}
_FLEX_ELIGIBLE  = {"RB", "WR", "TE"}
_SUPER_FLEX_POS = {"SUPER_FLEX", "SUPERFLEX", "QB/WR/RB/TE", "OP"}


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

    # Deterministic seed per league+season so odds don't drift on each reload.
    # Caller-supplied seed overrides (useful for testing).
    if seed is None and league_id:
        seed = int(hashlib.md5(f"{league_id}:{season}".encode()).hexdigest(), 16) % (2 ** 32)

    # Build division map from roster settings (Sleeper: roster.settings.division, 1-indexed)
    # Only used if the league has divisions configured.
    n_divisions = int((ctx.get("league_settings") or {}).get("divisions") or 0)
    division_map: dict[int, int] = {}
    if n_divisions >= 2:
        for r in (ctx.get("rosters") or []):
            rid = r.get("roster_id")
            div = int((r.get("settings") or {}).get("division") or 1)
            if rid is not None:
                division_map[int(rid)] = div

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
            matchups_by_week = _fallback_schedule(teams, remaining_weeks, division_map)
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
        matchups_by_week = _fallback_schedule(teams, remaining_weeks, division_map)

    result = _run_mc(teams, matchups_by_week, playoff_teams, n_sims, seed)
    for r in result:
        r["is_projected"] = False
    return result


# ---------------------------------------------------------------------------
# Roster-value projection (preseason / offseason)
# ---------------------------------------------------------------------------

def _player_std(pos: str, ppg: float) -> float:
    m, b = _POS_STD.get(pos, _POS_STD_DEFAULT)
    return m * ppg + b


def _team_std_from_starters(starters: list[tuple[str, float]]) -> float:
    """Estimate team weekly std dev from position-based player variance."""
    variance = sum(_player_std(pos, ppg) ** 2 for pos, ppg in starters)
    return max(math.sqrt(variance), _MIN_STD)


def _position_aware_lineup(
    pids: list,
    ppg_map: dict,
    pos_map: dict,
    roster_positions: list,
) -> tuple[float, list[tuple[str, float]]]:
    """
    Project weekly scoring using position-constrained optimal lineup.

    Fills fixed position slots (QB, RB, WR, TE, K, DEF) first, then fills
    FLEX/SuperFlex with the best remaining eligible player.

    Returns (projected_avg, starters) where starters is a list of (pos, ppg)
    for each starting slot — used to estimate per-team std dev.
    """
    # Tally starting slots by type
    fixed_slots: dict[str, int] = {}
    flex_slots  = 0
    sflex_slots = 0
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
            ppg = info["ppg"] if info["ppg"] > 0 else _ROOKIE_PPG.get(pos, _ROOKIE_PPG_DEFAULT)
        else:
            pos = pos_map.get(str(pid), "")
            ppg = _ROOKIE_PPG.get(pos, _ROOKIE_PPG_DEFAULT)
        if pos:
            by_pos.setdefault(pos, []).append(ppg)
    for pos in by_pos:
        by_pos[pos].sort(reverse=True)

    used: dict[str, int] = {}
    starters: list[tuple[str, float]] = []

    # Fill fixed slots
    for slot_pos, count in fixed_slots.items():
        pool = by_pos.get(slot_pos, [])
        for _ in range(count):
            i = used.get(slot_pos, 0)
            ppg = pool[i] if i < len(pool) else 0.0
            starters.append((slot_pos, ppg))
            used[slot_pos] = i + 1

    # Fill FLEX (RB/WR/TE eligible)
    flex_pool = sorted(
        [(pos, ppg) for pos in _FLEX_ELIGIBLE
         for ppg in by_pos.get(pos, [])[used.get(pos, 0):]],
        key=lambda x: x[1], reverse=True,
    )
    for i in range(flex_slots):
        if i < len(flex_pool):
            starters.append(flex_pool[i])
    remaining_after_flex = flex_pool[flex_slots:]

    # Fill SuperFlex (QB/RB/WR/TE eligible)
    sflex_pool = sorted(
        [(("QB", ppg)) for ppg in by_pos.get("QB", [])[used.get("QB", 0):]]
        + remaining_after_flex,
        key=lambda x: x[1], reverse=True,
    )
    for i in range(sflex_slots):
        if i < len(sflex_pool):
            starters.append(sflex_pool[i])

    total = sum(ppg for _, ppg in starters)
    return total, starters


def _estimate_from_rosters(ctx: dict) -> list[dict]:
    """
    Build synthetic team scoring profiles from prior-season actual PPG.

    For each player on a roster, looks up their PPG from last season's usage
    cache (same source as the player modal). Players with no prior-season stats
    fall back to a conservative position-based default.

    Uses position-aware lineup selection and per-team std dev estimated from
    the position composition of the starting lineup.
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
    # Try current year first, fall back to prior year.
    ppg_map: dict[str, dict] = {}
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

    # Load player positions from DB as fallback for rookies not in usage cache
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

    teams: list[dict] = []
    for roster in rosters:
        rid  = roster.get("roster_id")
        pids = roster.get("players") or []

        projected_avg, starters = _position_aware_lineup(
            pids, ppg_map, pos_map, roster_positions
        )
        projected_std = _team_std_from_starters(starters)

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
            "std":       round(projected_std, 1),
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


def _fallback_schedule(
    teams: list[dict],
    weeks: list[int],
    division_map: dict[int, int],
) -> dict[int, list[tuple[int, int]]]:
    """Route to divisional or plain round-robin based on whether divisions exist."""
    if division_map:
        return _divisional_schedule(teams, weeks, division_map)
    return _round_robin_schedule(teams, weeks)


def _round_robin_schedule(
    teams: list[dict],
    weeks: list[int],
) -> dict[int, list[tuple[int, int]]]:
    """
    Balanced round-robin (circle method): fix one team, rotate the rest.
    Produces N-1 unique rounds then cycles — matches Sleeper's default.
    """
    ids = [t["roster_id"] for t in teams]
    n   = len(ids)
    if n < 2:
        return {}

    if n % 2 == 1:
        ids = ids + [None]
        n += 1

    fixed    = ids[0]
    rotating = ids[1:]
    n_rounds = n - 1

    schedule: dict[int, list[tuple[int, int]]] = {}
    for week_idx, week in enumerate(weeks):
        r   = week_idx % n_rounds
        rot = rotating[-r:] + rotating[:-r] if r else rotating[:]
        pairs: list[tuple[int, int]] = []
        if fixed is not None and rot[0] is not None:
            pairs.append((fixed, rot[0]))
        for j in range(1, n // 2):
            a, b = rot[j], rot[n - 2 - j]
            if a is not None and b is not None:
                pairs.append((a, b))
        if pairs:
            schedule[week] = pairs
    return schedule


def _divisional_schedule(
    teams: list[dict],
    weeks: list[int],
    division_map: dict[int, int],
) -> dict[int, list[tuple[int, int]]]:
    """
    Divisional schedule: intra-division pairs appear twice, inter-division once.

    Builds a pool of all desired matchups for the season, then greedily
    packs them into weekly slots (each team plays once per week).
    Mirrors the frequency weighting of Sleeper's divisional scheduling.
    """
    from collections import defaultdict as _dd
    by_div: dict[int, list[int]] = _dd(list)
    for t in teams:
        div = division_map.get(t["roster_id"], 1)
        by_div[div].append(t["roster_id"])

    if len(by_div) < 2:
        return _round_robin_schedule(teams, weeks)

    divisions = list(by_div.values())
    n_teams   = len(teams)
    n_weeks   = len(weeks)
    n_per_week = n_teams // 2

    # Intra-division pairs (each appears twice in pool)
    intra_pairs: list[tuple[int, int]] = []
    for div_ids in divisions:
        for i in range(len(div_ids)):
            for j in range(i + 1, len(div_ids)):
                intra_pairs.append((div_ids[i], div_ids[j]))

    # Inter-division pairs (each appears once; cycle to fill remaining weeks)
    inter_pairs: list[tuple[int, int]] = []
    for d1 in range(len(divisions)):
        for d2 in range(d1 + 1, len(divisions)):
            for a in divisions[d1]:
                for b in divisions[d2]:
                    inter_pairs.append((a, b))

    # Pool: intra × 2 first so intra games are scheduled early,
    # then inter cycling to fill all weeks
    pool = list(intra_pairs) * 2 + list(inter_pairs)
    target = n_per_week * n_weeks
    while len(pool) < target:
        pool.extend(inter_pairs)
    pool = pool[:target]  # trim excess (cycles already guaranteed coverage)

    # Greedy weekly matching: each team appears at most once per round
    schedule: dict[int, list[tuple[int, int]]] = {}
    remaining = pool
    for week in weeks:
        round_pairs: list[tuple[int, int]] = []
        used: set[int] = set()
        leftover: list[tuple[int, int]] = []
        for a, b in remaining:
            if a not in used and b not in used:
                round_pairs.append((a, b))
                used.add(a)
                used.add(b)
            else:
                leftover.append((a, b))
        if round_pairs:
            schedule[week] = round_pairs
        remaining = leftover
        if not remaining:
            break
    return schedule


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

    n_byes = 2 if playoff_teams >= 4 else 0

    # Count scheduled games per team — with odd-team leagues one team per round
    # gets a bye, so they play fewer games than len(matchups_by_week).
    games_per_team = np.zeros(n, dtype=np.float32)

    for week_pairs in matchups_by_week.values():
        for (rid_a, rid_b) in week_pairs:
            ia = idx.get(rid_a)
            ib = idx.get(rid_b)
            if ia is None or ib is None:
                continue
            games_per_team[ia] += 1
            games_per_team[ib] += 1
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
    # Use per-team games scheduled so bye weeks don't inflate projected losses.
    avg_losses  = init_losses + games_per_team - (avg_wins - init_wins)

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
        "n_sims":           n_sims,
        "is_complete":      False,
    } for i, t in enumerate(teams)]
