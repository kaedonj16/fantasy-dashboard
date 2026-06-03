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
_WEEKLY_BLEND = 0.30   # in-season weight on weekly projections vs historical avg

# Weekly fantasy team totals are right-skewed (boom weeks, thin left tail),
# not Gaussian. We model each team's weekly score as a skew-normal with a
# fixed positive shape, rescaled to preserve the team's (mean, std). alpha=2
# gives a realistic skewness of ~0.45 while keeping the inputs the rest of the
# model already computes.
_SKEW_ALPHA   = 2.0
_SKEW_DELTA   = _SKEW_ALPHA / math.sqrt(1.0 + _SKEW_ALPHA * _SKEW_ALPHA)
# Mean/std of the raw skew-normal latent z (used to standardize before scaling)
_SKEW_Z_MEAN  = _SKEW_DELTA * math.sqrt(2.0 / math.pi)
_SKEW_Z_STD   = math.sqrt(1.0 - 2.0 * _SKEW_DELTA * _SKEW_DELTA / math.pi)

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


def _n_byes(playoff_teams: int) -> int:
    """
    Number of first-round byes implied by the bracket size.

    Even playoff sizes use a "half-bracket" model (Sleeper standard):
      byes = playoff_teams - largest_power_of_2_≤_playoff_teams
      4  → 0 byes  (4 = 2^2, full bracket)
      6  → 2 byes  (6 - 4 = 2; top 2 skip R1)
      8  → 0 byes  (8 = 2^3, full bracket)
      10 → 2 byes  (10 - 8 = 2)
      12 → 4 byes  (12 - 8 = 4)

    Odd playoff sizes use a standard single-elimination bracket
    (next power of 2 fills the empty slots with byes):
      5  → 3 byes  (8 - 5 = 3; top 3 skip R1, teams 4 & 5 play)
      7  → 1 bye   (8 - 7 = 1; top seed skips R1, teams 2-7 play)
    """
    if playoff_teams < 2:
        return 0
    if playoff_teams % 2 == 0:
        largest_pow2 = 1 << (playoff_teams.bit_length() - 1)
        return playoff_teams - largest_pow2
    # Odd: next power of 2 ≥ playoff_teams
    import math as _math
    next_pow2 = 1 << _math.ceil(_math.log2(playoff_teams))
    return next_pow2 - playoff_teams


# ---------------------------------------------------------------------------
# Public helper — PPG map for use by other modules (e.g. archetype engine)
# ---------------------------------------------------------------------------

def build_sim_state(ctx: dict, platform: str = "sleeper") -> Optional[dict]:
    """
    Build everything needed for trade-swap simulations without running them.

    Mirrors the branching logic of simulate_playoff_odds:
      - Preseason / offseason: project team strength from rosters (FP projections)
      - In-season: historical team avg blended with next-week projections
      - Season complete: returns None (no simulation needed)

    Returns a state dict with keys:
        teams             — list of team dicts (roster_id, avg, std, wins, …)
        matchups          — {week: [(rid_a, rid_b), …]}
        playoff_teams     — int
        seed              — int | None
        ppg_map           — {player_id: {ppg, pos}}
        pos_map           — {player_id: position}
        roster_positions  — list of slot strings from league settings
        roster_pid_map    — {roster_id: [player_id, …]}
    """
    settings           = ctx.get("league_settings") or {}
    playoff_week_start = int(settings.get("playoff_week_start") or 15)
    playoff_teams      = int(settings.get("playoff_teams") or 6)
    current_week       = int(ctx.get("current_week") or 0)
    season             = int(ctx.get("season") or 0)
    league_id          = str(ctx.get("league_id") or "")
    regular_season_end = playoff_week_start - 1

    seed: Optional[int] = None
    if league_id:
        seed = int(hashlib.md5(f"{league_id}:{season}".encode()).hexdigest(), 16) % (2 ** 32)

    n_divisions  = int((ctx.get("league_settings") or {}).get("divisions") or 0)
    division_map: dict[int, int] = {}
    if n_divisions >= 2:
        for r in (ctx.get("rosters") or []):
            rid = r.get("roster_id")
            div = int((r.get("settings") or {}).get("division") or 1)
            if rid is not None:
                division_map[int(rid)] = div

    ppg_map, pos_map = build_ppg_map(ctx)
    roster_positions = ctx.get("roster_positions") or []

    team_stats = ctx.get("team_stats")
    has_games  = team_stats is not None and not team_stats.empty

    if not has_games:
        # Share the exact PPG map used by swap sims so baseline == no-op swap.
        # Offseason team avg is 100% projection, so swaps use blend = 1.0.
        teams = _estimate_from_rosters(ctx, ppg_map=ppg_map, pos_map=pos_map)
        if not teams:
            return None
        remaining_weeks = list(range(1, playoff_week_start))
        blend_factor = 1.0
    else:
        if current_week > regular_season_end:
            return None  # season complete — no simulation
        teams = _build_teams(team_stats)
        if not teams:
            return None
        remaining_weeks = list(range(current_week + 1, playoff_week_start))
        # In-season avg blends historical results with weekly projections.
        # Swaps must apply the same blend factor to the projection delta so a
        # no-op swap leaves the avg unchanged.
        blend_factor = _WEEKLY_BLEND
        _blend_weekly_projections(teams, ctx, season, current_week + 1, blend=blend_factor)

    matchups = _resolve_schedule(
        platform, league_id, season, remaining_weeks, teams, division_map
    )

    roster_pid_map: dict[int, list[str]] = {}
    for r in (ctx.get("rosters") or []):
        rid = r.get("roster_id")
        if rid is not None:
            roster_pid_map[int(rid)] = [str(p) for p in (r.get("players") or [])]

    return {
        "teams":            teams,
        "matchups":         matchups,
        "playoff_teams":    playoff_teams,
        "seed":             seed,
        "ppg_map":          ppg_map,
        "pos_map":          pos_map,
        "roster_positions": roster_positions,
        "roster_pid_map":   roster_pid_map,
        "blend":            blend_factor,
    }


def run_base_simulation(sim_state: dict, n_sims: int = 2000) -> dict[int, float]:
    """
    Run the Monte Carlo simulation for the pre-built state.
    Returns {roster_id: playoff_pct (0–100)}.
    """
    result = _run_mc(
        sim_state["teams"], sim_state["matchups"],
        sim_state["playoff_teams"], n_sims, sim_state.get("seed"),
    )
    return {r["roster_id"]: r["playoff_pct"] for r in result}


def simulate_with_swap(
    sim_state: dict,
    viewer_roster_id: int,
    viewer_pids_after: list[str],
    n_sims: int = 10_000,
) -> tuple[float, float]:
    """
    Re-run the simulation with viewer's roster replaced by viewer_pids_after.

    Only the viewer's avg/std are recomputed; all other teams are unchanged,
    keeping the simulation fast (numpy-vectorised). The simulation reuses the
    same deterministic seed as the baseline (run_base_simulation) so that the
    *only* difference between the two runs is the viewer's swapped lineup —
    a common-random-numbers variance reduction. This makes the playoff-odds
    delta a clean before/after comparison instead of two independent noisy
    draws subtracted from each other.

    Returns (playoff_pct 0–100, new_avg_ppg).

    The viewer's new avg is computed as a *marginal* adjustment:

        new_avg = current_avg + blend * (proj_lineup(after) - proj_lineup(before))

    Both projection lineups use the same PPG map, so the map cancels in the
    difference and a no-op swap (after == before) leaves the avg untouched —
    yielding a zero delta. Offseason uses blend = 1.0 (avg is 100% projection,
    so new_avg collapses to proj_lineup(after)); in-season uses the same blend
    factor applied to the team's blended baseline, so only the trade's true
    effect moves the odds.
    """
    ppg_map          = sim_state["ppg_map"]
    pos_map          = sim_state["pos_map"]
    roster_positions = sim_state["roster_positions"]
    blend            = float(sim_state.get("blend", 1.0))

    viewer_team = next(
        (t for t in sim_state["teams"] if t["roster_id"] == viewer_roster_id), None
    )
    current_avg = viewer_team["avg"] if viewer_team else 0.0

    proj_after, after_starters = _position_aware_lineup(
        viewer_pids_after, ppg_map, pos_map, roster_positions
    )

    if blend >= 1.0:
        new_avg = proj_after
    else:
        before_pids = sim_state.get("roster_pid_map", {}).get(viewer_roster_id, [])
        proj_before, _ = _position_aware_lineup(
            before_pids, ppg_map, pos_map, roster_positions
        )
        new_avg = current_avg + blend * (proj_after - proj_before)

    new_std = max(_team_std_from_starters(after_starters), _MIN_STD)

    teams = [
        {**t, "avg": round(new_avg, 1), "std": round(new_std, 1)}
        if t["roster_id"] == viewer_roster_id else t
        for t in sim_state["teams"]
    ]

    result = _run_mc(teams, sim_state["matchups"], sim_state["playoff_teams"],
                     n_sims, sim_state.get("seed"))
    for r in result:
        if r["roster_id"] == viewer_roster_id:
            return r["playoff_pct"], new_avg

    return 0.0, new_avg


def build_ppg_map(ctx: dict) -> tuple[dict, dict]:
    """
    Build (ppg_map, pos_map).

    Priority order:
      1. Sleeper weekly projections for the upcoming week — same source used by
         _blend_weekly_projections, so simulate_with_swap sees the same player
         PPGs as the base simulation's team-avg blend.
      2. FantasyPros season projections — preseason baseline or gap-filler.
      3. Prior-season usage_rows cache — fills any remaining gaps.

    Returns:
        ppg_map  — {str(player_id): {"ppg": float, "pos": str}}
        pos_map  — {str(player_id): str(position)}   (position fallback)
    """
    season       = int(ctx.get("season") or 0)
    current_week = int(ctx.get("current_week") or 0)
    _ss    = ctx.get("scoring_settings") or {}
    _rss   = ctx.get("raw_scoring_settings") or {}
    rec_pts = float(_ss.get("rec") or _ss.get("pointsPerReception") or 0)
    if rec_pts >= 1.0:
        scoring  = "ppr"
        ppg_key  = "ppr_ppg"
    elif rec_pts >= 0.5:
        scoring  = "half_ppr"
        ppg_key  = "half_ppr_ppg"
    else:
        scoring  = "std"
        ppg_key  = "std_scoring_ppg"

    # pos_map built first — needed as position fallback for all PPG sources
    pos_map: dict = {}
    try:
        from dashboard_services.db import get_conn
        with get_conn() as conn:
            rows = conn.execute(
                "SELECT player_id, position FROM player_values WHERE position IS NOT NULL"
            ).fetchall()
        pos_map = {str(r["player_id"]): str(r["position"]).upper() for r in rows}
    except Exception:
        pass

    ppg_map: dict = {}

    # Priority 1: Sleeper weekly projections (in-season — same source as the
    # team-avg blend so simulate_with_swap is consistent with the base sim)
    if current_week > 0 and season > 0:
        try:
            from utils.utils import fetch_week_projections, pick_proj_variant
            multi_map = fetch_week_projections(season, current_week + 1, _rss)
            variant   = pick_proj_variant(_rss)
            for pid, variants in (multi_map or {}).items():
                if not isinstance(variants, dict):
                    continue
                pts = variants.get(variant) or variants.get("ppr") or 0.0
                if pts > 0:
                    ppg_map[str(pid)] = {
                        "ppg": float(pts),
                        "pos": pos_map.get(str(pid), ""),
                    }
        except Exception:
            pass

    # Priority 2: FantasyPros season projections (preseason or gap-filler)
    try:
        from data_building.fetch_projections import fetch_fp_season_projections
        fp_data = fetch_fp_season_projections(season, scoring)
        for pid, info in fp_data.items():
            if str(pid) not in ppg_map and info.get("ppg", 0) > 0:
                ppg_map[str(pid)] = {"ppg": info["ppg"], "pos": info.get("pos", "")}
    except Exception:
        pass

    # Priority 3: Prior-season usage cache
    try:
        import os as _os, json as _json
        from datetime import date as _date
        _cache_dir = _os.path.join(_os.path.dirname(__file__), "..", "cache", "player_history")
        _year = season or _date.today().year
        for _y in [_year, _year - 1]:
            _path = _os.path.join(_cache_dir, f"usage_rows_{_y}.json")
            if _os.path.exists(_path):
                with open(_path) as _f:
                    _usage_data = _json.load(_f)
                for p in _usage_data:
                    pid = str(p.get("id") or "")
                    if not pid or pid in ppg_map:
                        continue
                    ppg = float((p.get("usage") or {}).get(ppg_key) or 0)
                    pos = str(p.get("position") or "").upper()
                    if ppg > 0:
                        ppg_map[pid] = {"ppg": ppg, "pos": pos}
                break
    except Exception:
        pass

    return ppg_map, pos_map


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
        matchups_by_week = _resolve_schedule(
            platform, league_id, season, remaining_weeks, teams, division_map
        )
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
    matchups_by_week = _resolve_schedule(
        platform, league_id, season, remaining_weeks, teams, division_map
    )

    # Blend historical team avg with Sleeper's upcoming-week player projections.
    # Using 30 % weekly projection + 70 % season average keeps the estimate
    # stable while still reflecting injuries and matchups.
    _blend_weekly_projections(teams, ctx, season, current_week + 1)

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

    # Per-position averages from real projections — used as fallback for
    # players with no FP data (rookies, injured, newly signed).
    _pos_totals: dict[str, list] = {}
    for _info in ppg_map.values():
        _p, _g = _info.get("pos", ""), _info.get("ppg", 0)
        if _p and _g > 0:
            if _p not in _pos_totals:
                _pos_totals[_p] = [_g, 1]
            else:
                _pos_totals[_p][0] += _g
                _pos_totals[_p][1] += 1
    pos_fallback = {p: v[0] / v[1] for p, v in _pos_totals.items()}

    # Resolve each player to (pos, ppg)
    by_pos: dict[str, list[float]] = {}
    for pid in pids:
        info = ppg_map.get(str(pid))
        if info:
            pos = info["pos"]
            ppg = info["ppg"] if info["ppg"] > 0 else (
                pos_fallback.get(pos) or _ROOKIE_PPG.get(pos, _ROOKIE_PPG_DEFAULT)
            )
        else:
            pos = pos_map.get(str(pid), "")
            ppg = pos_fallback.get(pos) or _ROOKIE_PPG.get(pos, _ROOKIE_PPG_DEFAULT)
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


def _estimate_from_rosters(
    ctx: dict,
    ppg_map: Optional[dict] = None,
    pos_map: Optional[dict] = None,
) -> list[dict]:
    """
    Build synthetic team scoring profiles for preseason simulation.

    PPG source priority (highest accuracy first):
      1. FantasyPros consensus season projections ÷ 17
         (accounts for off-season moves, role changes, analyst consensus)
      2. Prior-season usage_rows cache (same source as player modal)
         (fallback when FP fetch fails or player is absent from FP)
      3. Position-based rookie default (last resort)

    Uses position-aware lineup selection and per-team std dev estimated from
    the position composition of the projected starting lineup.

    When ppg_map/pos_map are supplied (by build_sim_state) they are used
    verbatim instead of being rebuilt, so the baseline team averages and the
    per-trade swap simulations share one identical PPG source — a no-op swap
    then produces a zero delta and real trades show only their true effect.
    """
    rosters          = ctx.get("rosters") or []
    roster_map       = ctx.get("roster_map") or {}
    roster_positions = ctx.get("roster_positions") or []
    season           = int(ctx.get("season") or 0)

    if not rosters:
        return []

    # If a prebuilt PPG map is supplied, skip the internal build entirely.
    if ppg_map is not None:
        if pos_map is None:
            pos_map = {}
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
                "wins":      0, "losses": 0, "ties": 0, "pf": 0.0,
                "avg":       round(projected_avg, 1),
                "std":       round(projected_std, 1),
            })
        return teams

    # Detect scoring format — Sleeper uses "rec", non-Sleeper uses "pointsPerReception"
    _ss = ctx.get("scoring_settings") or {}
    rec_pts = float(_ss.get("rec") or _ss.get("pointsPerReception") or 0)
    if rec_pts >= 1.0:
        scoring  = "ppr"
        ppg_key  = "ppr_ppg"
    elif rec_pts >= 0.5:
        scoring  = "half_ppr"
        ppg_key  = "half_ppr_ppg"
    else:
        scoring  = "std"
        ppg_key  = "std_scoring_ppg"

    # ── Source 1: FantasyPros season projections ──────────────────────────────
    ppg_map: dict[str, dict] = {}
    try:
        from data_building.fetch_projections import fetch_fp_season_projections
        fp_data = fetch_fp_season_projections(season, scoring)
        for pid, info in fp_data.items():
            if info.get("ppg", 0) > 0:
                ppg_map[str(pid)] = {"ppg": info["ppg"], "pos": info.get("pos", "")}
        if ppg_map:
            logger.info("[playoff_odds] Using FP projections: %d players", len(ppg_map))
    except Exception as exc:
        logger.warning("[playoff_odds] FP projections unavailable: %s", exc)

    # ── Source 2: prior-season usage_rows (fills gaps / full fallback) ────────
    try:
        import os as _os, json as _json
        from datetime import date as _date
        _cache_dir = _os.path.join(_os.path.dirname(__file__), "..", "cache", "player_history")
        _year = season or _date.today().year
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
                if not pid or pid in ppg_map:
                    continue  # already have a FP projection for this player
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
# In-season weekly projection blend
# ---------------------------------------------------------------------------

def _blend_weekly_projections(
    teams: list[dict],
    ctx: dict,
    season: int,
    next_week: int,
    blend: float = _WEEKLY_BLEND,
) -> None:
    """
    Update each team's avg in-place by blending historical avg (1-blend) with
    their projected optimal lineup for next_week (blend).

    Requires ctx["rosters"] for player-roster mapping. Silently skips if
    Sleeper projections are unavailable (network error, off-season, etc.).
    """
    if next_week < 1 or blend <= 0:
        return

    raw_ss = ctx.get("raw_scoring_settings") or {}

    try:
        from utils.utils import fetch_week_projections, pick_proj_variant
        multi_map = fetch_week_projections(season, next_week, raw_ss)
    except Exception as exc:
        logger.warning("[playoff_odds] Sleeper weekly proj unavailable: %s", exc)
        return

    if not multi_map:
        return

    variant = pick_proj_variant(raw_ss)

    roster_positions = ctx.get("roster_positions") or []
    # Build roster_id → player_ids lookup
    rid_to_pids: dict[int, list] = {
        int(r.get("roster_id")): list(r.get("players") or [])
        for r in (ctx.get("rosters") or [])
        if r.get("roster_id") is not None
    }
    # pos_map for position fallback
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

    # Flatten multi-variant projections to a single pts value per player
    week_ppg: dict[str, dict] = {}
    for pid, variants in multi_map.items():
        if not isinstance(variants, dict):
            continue
        pts = variants.get(variant) or variants.get("ppr") or 0.0
        if pts > 0:
            week_ppg[pid] = {"ppg": pts, "pos": pos_map.get(pid, "")}

    updated = 0
    for team in teams:
        rid  = team["roster_id"]
        pids = rid_to_pids.get(rid)
        if not pids:
            continue
        proj_score, _ = _position_aware_lineup(pids, week_ppg, pos_map, roster_positions)
        if proj_score <= 0:
            continue
        historical_avg  = team["avg"]
        team["avg"]     = round(blend * proj_score + (1 - blend) * historical_avg, 1)
        updated += 1

    if updated:
        logger.info(
            "[playoff_odds] Blended weekly proj (%.0f%%) for %d teams (week %d)",
            blend * 100, updated, next_week,
        )


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
    n_byes   = _n_byes(playoff_teams)
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


def _resolve_schedule(
    platform: str,
    league_id: str,
    season: int,
    weeks: list[int],
    teams: list[dict],
    division_map: dict[int, int],
) -> dict[int, list[tuple[int, int]]]:
    """Single schedule source shared by every simulation path.

    Prefers the real published schedule and switches to it the moment it is
    decided — per week. Any week the platform hasn't published yet is filled
    from the deterministic fallback so coverage is complete and identical
    across all callers. Once the real schedule is fully published, every
    simulation uses it verbatim.
    """
    real = _fetch_remaining_schedule(platform, league_id, season, weeks)
    missing = [w for w in weeks if w not in real]
    if not missing:
        return real
    fallback = _fallback_schedule(teams, missing, division_map)
    # Real takes precedence per week; fallback only fills undecided weeks.
    return {**fallback, **real}


def _round_robin_schedule(
    teams: list[dict],
    weeks: list[int],
) -> dict[int, list[tuple[int, int]]]:
    """
    Balanced round-robin (circle method): fix one team, rotate the rest.
    Produces N-1 unique rounds then cycles — matches Sleeper's default.

    Teams are sorted by roster_id so the generated schedule is identical
    regardless of the order teams are passed in. This keeps the fallback
    schedule universal across every caller (playoff-odds page, archetype
    swap sims, etc.) so all simulations share one schedule.
    """
    ids = sorted(t["roster_id"] for t in teams)
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
            a, b = rot[j], rot[n - 1 - j]
            if a is not None and b is not None and a != b:
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
    for t in sorted(teams, key=lambda x: x["roster_id"]):
        div = division_map.get(t["roster_id"], 1)
        by_div[div].append(t["roster_id"])

    if len(by_div) < 2:
        return _round_robin_schedule(teams, weeks)

    # Iterate divisions in sorted key order so the schedule is deterministic
    # and identical across all callers (one universal schedule).
    divisions = [sorted(by_div[d]) for d in sorted(by_div)]
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

def _sample_scores(
    rng: "np.random.Generator",
    mean: float,
    std: float,
    n_sims: int,
) -> "np.ndarray":
    """Draw n_sims weekly scores from a right-skewed distribution.

    Two improvements over a plain Normal draw:

    1. Skew-normal shape — weekly fantasy totals are right-skewed (occasional
       boom weeks, a thin floor), so a Gaussian understates upside ceilings.
       The latent skew-normal z = delta*|u0| + sqrt(1-delta^2)*u1 is
       standardised and rescaled to the requested (mean, std), so the mean and
       variance are preserved exactly while injecting realistic positive skew.

    2. Antithetic variates — the symmetric component u1 is mirrored between the
       first and second half of the sims (z uses +u1 / -u1 over the same |u0|).
       Sim i and sim i+half are therefore mirror seasons, which cancels much of
       the sampling noise in win counts and points-for, tightening playoff-odds
       estimates (and trade deltas) without raising n_sims.
    """
    half = (n_sims + 1) // 2
    u0 = rng.standard_normal(half).astype(np.float32)
    u1 = rng.standard_normal(half).astype(np.float32)
    c  = np.float32(math.sqrt(1.0 - _SKEW_DELTA * _SKEW_DELTA))
    d  = np.float32(_SKEW_DELTA)
    half_normal = np.abs(u0)
    z_pos = d * half_normal + c * u1
    z_neg = d * half_normal - c * u1          # antithetic on the symmetric part
    z = np.concatenate([z_pos, z_neg])[:n_sims]
    z_std = (z - np.float32(_SKEW_Z_MEAN)) / np.float32(_SKEW_Z_STD)
    scores = np.float32(mean) + np.float32(std) * z_std
    return np.maximum(scores, 0).astype(np.float32)


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

    n_byes = _n_byes(playoff_teams)

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
            sa = _sample_scores(rng, avgs[ia], stds[ia], n_sims)
            sb = _sample_scores(rng, avgs[ib], stds[ib], n_sims)
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
