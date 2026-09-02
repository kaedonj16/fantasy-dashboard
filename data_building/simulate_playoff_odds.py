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
finish in under a second. Trade-suggestion ranking freezes one baseline
season and re-scores only the teams a package actually moves, so a slate of
packages costs one full sim plus cheap overlays instead of a full replay
each time.
"""
from __future__ import annotations

import hashlib
import logging
import math
import threading
import time
from collections import defaultdict
from typing import Optional

import numpy as np

from utils.lineup_slots import RESTRICTED_FLEX_SLOTS, canonicalize_slot, slot_eligible_positions

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

_FLEX_ELIGIBLE  = {"RB", "WR", "TE"}

# Used when a provider leaves roster_positions empty (ESPN/Yahoo/MFL adapter
# bugs). Prefer a standard redraft lineup over silent 0.0% Proj% / flat odds.
_DEFAULT_STARTING_SLOTS = ["QB", "RB", "RB", "WR", "WR", "TE", "FLEX", "K", "DEF"]


def _tally_starting_slots(roster_positions):
    """Count dedicated / restricted-flex / FLEX / Superflex slots."""
    fixed_slots: dict[str, int] = {}
    flex_slots = sflex_slots = 0
    restricted = {name: 0 for name in RESTRICTED_FLEX_SLOTS}
    for slot in roster_positions or []:
        s = canonicalize_slot(slot)
        if s in _BENCH_SLOTS:
            continue
        if s == "SUPER_FLEX":
            sflex_slots += 1
        elif s == "FLEX":
            flex_slots += 1
        elif s in restricted:
            restricted[s] += 1
        elif s:
            fixed_slots[s] = fixed_slots.get(s, 0) + 1
    return fixed_slots, flex_slots, sflex_slots, restricted


def _tally_starting_slots_or_default(roster_positions):
    """Like ``_tally_starting_slots``, but never returns an empty lineup."""
    fixed_slots, flex_slots, sflex_slots, restricted = _tally_starting_slots(
        roster_positions
    )
    if fixed_slots or flex_slots or sflex_slots or any(restricted.values()):
        return fixed_slots, flex_slots, sflex_slots, restricted, False
    logger.warning(
        "[playoff_odds] empty roster_positions; using default %s lineup",
        "/".join(_DEFAULT_STARTING_SLOTS),
    )
    fixed_slots, flex_slots, sflex_slots, restricted = _tally_starting_slots(
        _DEFAULT_STARTING_SLOTS
    )
    return fixed_slots, flex_slots, sflex_slots, restricted, True


def _consume_eligible(pool, n, eligible):
    """Take up to ``n`` (pos, ppg) rows whose pos is in ``eligible``."""
    taken, rest = [], []
    for pos, ppg in pool:
        if len(taken) < n and pos in eligible:
            taken.append((pos, ppg))
        else:
            rest.append((pos, ppg))
    return taken, rest

# Per-week injury hazard: the probability that a given starter misses that
# week's game. When a starter is out, a bench player replaces them at a
# fraction of their output, so the team loses (1 - _INJURY_REPLACEMENT) of the
# starter's projected PPG for that week. Rates are per-game "miss next game"
# approximations from empirical NFL availability by position (RBs miss the most,
# QBs the least). This injects realistic week-to-week downside that the smooth
# scoring distribution alone doesn't capture.
_INJURY_HAZARD: dict[str, float] = {
    "QB": 0.03, "RB": 0.07, "WR": 0.05, "TE": 0.05, "K": 0.01, "DEF": 0.0,
}
_INJURY_HAZARD_DEFAULT = 0.05
_INJURY_REPLACEMENT    = 0.45   # replacement plays at ~45% of the starter's PPG

# Injuries last multiple weeks, not one. When a starter goes down we sample a
# duration (in weeks) from this distribution; while out, the substitution loss
# applies every week. The per-week ONSET rate is the position hazard divided by
# the mean duration, so the total expected games missed stays ≈ the hazard while
# the misses now cluster into realistic multi-week absences.
_INJURY_DURATION_CHOICES = (1, 2, 3, 4, 6, 8)
_INJURY_DURATION_PROBS   = (0.50, 0.20, 0.12, 0.08, 0.06, 0.04)
_INJURY_MEAN_DURATION    = sum(c * p for c, p in zip(_INJURY_DURATION_CHOICES, _INJURY_DURATION_PROBS))

# Fantasy matchups occasionally tie (identical rounded scores). Treat scores
# within this margin as a tie worth half a win to each side.
_TIE_MARGIN = 0.05


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

# In-process cache of built sim states. The Playoff Impact endpoint rebuilds
# this on every trade edit, but the heavy parts (league context, ~14 weeks of
# projections, per-team week profiles) only change when the rosters, week, or
# settings change — so we key on exactly that and reuse otherwise.
_SIM_STATE_CACHE: dict = {}
_SIM_STATE_TTL   = 600        # seconds
_SIM_STATE_MAX   = 64         # cap entries to bound memory


def _ctx_signature(ctx: dict, platform: str) -> str:
    league_id = str(ctx.get("league_id") or "")
    season    = int(ctx.get("season") or 0)
    cw        = int(ctx.get("current_week") or 0)
    settings  = ctx.get("league_settings") or {}
    h = hashlib.md5()
    h.update(f"{settings.get('playoff_week_start')}:{settings.get('playoff_teams')};".encode())
    for r in sorted(ctx.get("rosters") or [], key=lambda x: x.get("roster_id") or 0):
        pids = ",".join(sorted(str(p) for p in (r.get("players") or [])))
        h.update(f"{r.get('roster_id')}:{pids};".encode())
    return f"{platform}:{league_id}:{season}:{cw}:{h.hexdigest()}"


def _evict_sim_cache() -> None:
    if len(_SIM_STATE_CACHE) <= _SIM_STATE_MAX:
        return
    # Drop the oldest entries down to the cap.
    for key in sorted(_SIM_STATE_CACHE, key=lambda k: _SIM_STATE_CACHE[k][0])[
        : len(_SIM_STATE_CACHE) - _SIM_STATE_MAX
    ]:
        _SIM_STATE_CACHE.pop(key, None)


def build_sim_state(ctx: dict, platform: str = "sleeper", use_cache: bool = True) -> Optional[dict]:
    """Cached wrapper around the sim-state build (see _build_sim_state_impl).

    Reuses a recently built state when the league, week, settings, and every
    roster are unchanged; the swap functions never mutate the state, so sharing
    it across requests is safe.
    """
    if not use_cache:
        return _build_sim_state_impl(ctx, platform)
    sig = _ctx_signature(ctx, platform)
    now = time.time()
    hit = _SIM_STATE_CACHE.get(sig)
    if hit and (now - hit[0]) < _SIM_STATE_TTL:
        return hit[1]
    state = _build_sim_state_impl(ctx, platform)
    if state is not None:
        _SIM_STATE_CACHE[sig] = (now, state)
        _evict_sim_cache()
    return state


def _build_sim_state_impl(ctx: dict, platform: str = "sleeper") -> Optional[dict]:
    """
    Build everything needed for trade-swap simulations without running them.

    Mirrors the branching logic of simulate_playoff_odds:
      - Preseason / offseason: project team strength from Sleeper projections
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

    season_ppg_map, pos_map = build_ppg_map(ctx)
    roster_positions = ctx.get("roster_positions") or []
    raw_ss           = ctx.get("raw_scoring_settings") or {}

    team_stats = ctx.get("team_stats")
    has_games = team_stats_have_played_games(team_stats)
    roster_map = ctx.get("roster_map") or {}

    if not has_games:
        # Preseason / offseason / Week-1 seed: project from rosters. blend = 1.0
        # → per-week mean is the pure weekly Sleeper projection.
        teams = _estimate_from_rosters(ctx, ppg_map=season_ppg_map, pos_map=pos_map)
        if not teams:
            return None
        remaining_weeks = list(range(1, playoff_week_start))
        blend_factor = 1.0
        hist_avg_by_rid = {t["roster_id"]: 0.0 for t in teams}
        hist_std_by_rid = {t["roster_id"]: 0.0 for t in teams}
    else:
        if current_week > regular_season_end:
            return None  # season complete — no simulation
        teams = _build_teams(team_stats, roster_map)
        if not teams:
            return None
        remaining_weeks = list(range(current_week + 1, playoff_week_start))
        # In-season: per-week mean blends that week's projection with the team's
        # season-to-date average (captured here BEFORE any mutation) and realized
        # weekly std. The per-week profiles apply the blend themselves.
        blend_factor = _WEEKLY_BLEND
        hist_avg_by_rid = {t["roster_id"]: float(t["avg"]) for t in teams}
        hist_std_by_rid = {t["roster_id"]: float(t["std"]) for t in teams}

    matchups = _resolve_schedule(
        platform, league_id, season, remaining_weeks, teams, division_map
    )

    roster_pid_map: dict[int, list[str]] = {}
    for r in (ctx.get("rosters") or []):
        rid = r.get("roster_id")
        if rid is not None:
            roster_pid_map[int(rid)] = [str(p) for p in (r.get("players") or [])]

    # Per-week Sleeper projection maps and the resulting per-team week profiles.
    week_ppg_maps = _build_week_ppg_maps(
        season, remaining_weeks, raw_ss, pos_map, season_ppg_map
    )
    week_profiles = _compute_week_profiles(
        roster_pid_map, week_ppg_maps, pos_map, roster_positions,
        hist_avg_by_rid, hist_std_by_rid, blend_factor,
    )

    return {
        "teams":            teams,
        "matchups":         matchups,
        "playoff_teams":    playoff_teams,
        "seed":             seed,
        "ppg_map":          season_ppg_map,
        "pos_map":          pos_map,
        "roster_positions": roster_positions,
        "roster_pid_map":   roster_pid_map,
        "blend":            blend_factor,
        "week_ppg_maps":    week_ppg_maps,
        "week_profiles":    week_profiles,
        "hist_avg_by_rid":  hist_avg_by_rid,
        "hist_std_by_rid":  hist_std_by_rid,
    }


def _viewer_week_mean(week_profiles: dict, viewer_roster_id: int) -> float:
    """Average of the viewer's per-week means — their effective projected PPG."""
    means = [
        wp[viewer_roster_id]["mean"]
        for wp in week_profiles.values()
        if viewer_roster_id in wp
    ]
    return round(sum(means) / len(means), 1) if means else 0.0


def run_base_simulation(sim_state: dict, n_sims: int = 2000) -> dict[int, float]:
    """
    Run the Monte Carlo simulation for the pre-built state.
    Returns {roster_id: playoff_pct (0–100)}.

    Caches the weekly score tensor on ``sim_state`` so subsequent
    ``simulate_with_swap`` calls reuse every unchanged team's season instead of
    replaying the whole league.
    """
    freeze = _ensure_freeze(sim_state, n_sims)
    return {r["roster_id"]: r["playoff_pct"] for r in freeze["result"]}


def simulate_with_swap(
    sim_state: dict,
    viewer_roster_id: int,
    viewer_pids_after: list[str],
    n_sims: int = 10_000,
) -> tuple[float, float]:
    """
    Re-run the simulation with viewer's roster replaced by viewer_pids_after.

    Only the teams whose lineups actually change (the viewer, and the
    counterparty when we can infer one) are re-scored. Every other team's
    weekly scores are reused from a frozen baseline drawn with the same seed,
    so the playoff-odds delta is a common-random-numbers before/after — and
    ranking a slate of suggestion packages costs one full league sim plus a
    cheap overlay per package, not a full 10k-season replay each time.

    Returns (playoff_pct 0–100, new_avg_ppg).

    Both sides of the deal are reflected: the give/get are derived from the
    diff between the viewer's current roster and viewer_pids_after, and the
    counterparty (the team that currently owns the incoming players) is
    re-rostered with the players the viewer sends out. This matches
    simulate_swap_impact, so a partner that sits on the viewer's schedule or in
    their seeding race is correctly strengthened — otherwise the displayed
    suggestion delta is systematically too favorable (the sent players would
    simply vanish from the league).

    The viewer's new avg is computed as a *marginal* adjustment:

        new_avg = current_avg + blend * (proj_lineup(after) - proj_lineup(before))

    Both projection lineups use the same PPG map, so the map cancels in the
    difference and a no-op swap (after == before) leaves the avg untouched —
    yielding a zero delta. Offseason uses blend = 1.0 (avg is 100% projection,
    so new_avg collapses to proj_lineup(after)); in-season uses the same blend
    factor applied to the team's blended baseline, so only the trade's true
    effect moves the odds.
    """
    overrides = {viewer_roster_id: viewer_pids_after}
    # Reflect the counterparty: derive give/get from the roster diff and hand the
    # sent players to whoever currently owns the incoming ones.
    roster_pid_map = sim_state.get("roster_pid_map") or {}
    before_set = set(roster_pid_map.get(viewer_roster_id, []))
    after_set  = set(viewer_pids_after)
    give_pids  = before_set - after_set
    get_pids   = list(after_set - before_set)
    counterparty = _infer_counterparty(roster_pid_map, viewer_roster_id, get_pids)
    if counterparty is not None:
        cp_pids = roster_pid_map.get(counterparty, [])
        overrides[counterparty] = list((set(cp_pids) - set(get_pids)) | give_pids)

    after_profiles = _override_profiles(sim_state, overrides)
    new_avg = _viewer_week_mean(after_profiles, viewer_roster_id)
    result = _standings_with_overrides(sim_state, after_profiles, overrides, n_sims)
    for r in result:
        if r["roster_id"] == viewer_roster_id:
            return r["playoff_pct"], new_avg

    return 0.0, new_avg


def _viewer_profiles_for_roster(sim_state: dict, viewer_roster_id: int, pids: list) -> dict:
    """Per-week profile for the viewer's hypothetical roster (one entry per week)."""
    hist_avg = float((sim_state.get("hist_avg_by_rid") or {}).get(viewer_roster_id, 0.0))
    hist_std = float((sim_state.get("hist_std_by_rid") or {}).get(viewer_roster_id, 0.0))
    blend    = float(sim_state.get("blend", 1.0))
    pos_map  = sim_state["pos_map"]
    rpos     = sim_state["roster_positions"]
    out: dict[int, dict] = {}
    for week, ppg_map in (sim_state.get("week_ppg_maps") or {}).items():
        out[week] = _team_week_profile(
            pids, ppg_map, pos_map, rpos, hist_avg, hist_std, blend
        )
    return out


def _override_profiles(sim_state: dict, overrides: dict) -> dict:
    """Clone the base week profiles, replacing each {roster_id: pids} override.

    Every non-overridden team is untouched, so before/after sims differ by
    exactly the trade (common random numbers).
    """
    base = sim_state.get("week_profiles") or {}
    per_rid_wk = {
        rid: _viewer_profiles_for_roster(sim_state, rid, pids)
        for rid, pids in overrides.items()
    }
    merged: dict[int, dict] = {}
    for week, wp in base.items():
        nwp = dict(wp)
        for rid, wkprof in per_rid_wk.items():
            if week in wkprof:
                nwp[rid] = wkprof[week]
        merged[week] = nwp
    return merged



def _infer_counterparty(roster_pid_map: dict, viewer_roster_id: int, get_pids: list) -> Optional[int]:
    """The roster that currently owns the most of the incoming players, if any."""
    gset = set(get_pids)
    counts: dict[int, int] = {}
    for rid, pids in (roster_pid_map or {}).items():
        if rid == viewer_roster_id:
            continue
        c = len(gset & set(pids))
        if c:
            counts[rid] = c
    return max(counts, key=counts.get) if counts else None


def simulate_swap_impact(
    sim_state: dict,
    viewer_roster_id: int,
    give_pids: list[str],
    get_pids: list[str],
    n_sims: int = 2000,
) -> dict:
    """
    Compute before/after playoff metrics for a trade.

    Returns a dict:
        before: {playoff_pct, avg_final_wins, avg_ppg}
        after:  {playoff_pct, avg_final_wins, avg_ppg}
        delta:  {playoff_pct, avg_final_wins, avg_ppg}
        league_avg_ppg: float  (median projected PPG across all teams)
        available: bool        (False when season is complete / no sim possible)
    """
    if not sim_state:
        return {"available": False}

    base_profiles    = sim_state.get("week_profiles") or {}

    viewer_team = next(
        (t for t in sim_state["teams"] if t["roster_id"] == viewer_roster_id), None
    )
    if viewer_team is None:
        return {"available": False}

    # Before simulation (base profiles already reflect the viewer's current roster)
    freeze = _ensure_freeze(sim_state, n_sims)
    before_results = freeze["result"]
    before_row = next((r for r in before_results if r["roster_id"] == viewer_roster_id), {})

    # Compute after roster
    current_pids = sim_state.get("roster_pid_map", {}).get(viewer_roster_id, [])
    current_pids_set = set(current_pids)
    # Detect give IDs that aren't actually on the viewer's roster — the removal
    # will be a no-op for those players, which the caller should surface to the user.
    missing_give_ids = [p for p in give_pids if p not in current_pids_set]
    after_set = (current_pids_set - set(give_pids)) | set(get_pids)
    after_pids = list(after_set)

    # Reflect both sides of the deal: the players you send away strengthen the
    # team you receive from (and vice versa), which matters when they're on your
    # schedule or in your seeding race.
    overrides = {viewer_roster_id: after_pids}
    counterparty = _infer_counterparty(
        sim_state.get("roster_pid_map", {}), viewer_roster_id, get_pids
    )
    if counterparty is not None:
        cp_pids = sim_state.get("roster_pid_map", {}).get(counterparty, [])
        overrides[counterparty] = list((set(cp_pids) - set(get_pids)) | set(give_pids))

    after_profiles = _override_profiles(sim_state, overrides)
    after_results = _standings_with_overrides(sim_state, after_profiles, overrides, n_sims)
    after_row = next((r for r in after_results if r["roster_id"] == viewer_roster_id), {})

    import statistics
    league_avgs = [
        _viewer_week_mean(base_profiles, t["roster_id"]) for t in sim_state["teams"]
    ]
    league_avgs = [a for a in league_avgs if a > 0]
    league_avg_ppg = round(statistics.median(league_avgs), 1) if league_avgs else 0.0

    new_avg      = _viewer_week_mean(after_profiles, viewer_roster_id)
    before_po    = round(float(before_row.get("playoff_pct",    0)), 1)
    before_wins  = round(float(before_row.get("avg_final_wins", 0)), 1)
    before_ppg   = _viewer_week_mean(base_profiles, viewer_roster_id)
    before_top3  = round(float(before_row.get("top3_pick_pct",  0)), 1)
    after_po     = round(float(after_row.get("playoff_pct",     0)), 1)
    after_wins   = round(float(after_row.get("avg_final_wins",  0)), 1)
    after_ppg    = round(new_avg, 1)
    after_top3   = round(float(after_row.get("top3_pick_pct",   0)), 1)

    return {
        "available": True,
        "before": {"playoff_pct": before_po, "avg_final_wins": before_wins,
                   "avg_ppg": before_ppg, "top3_pick_pct": before_top3},
        "after":  {"playoff_pct": after_po,  "avg_final_wins": after_wins,
                   "avg_ppg": after_ppg, "top3_pick_pct": after_top3},
        "delta":  {
            "playoff_pct":    round(after_po    - before_po,    1),
            "avg_final_wins": round(after_wins  - before_wins,  1),
            "avg_ppg":        round(after_ppg   - before_ppg,   1),
            "top3_pick_pct":  round(after_top3  - before_top3,  1),
        },
        "league_avg_ppg":   league_avg_ppg,
        "missing_give_ids": missing_give_ids,
    }


def shape_playoff_impact_for_league(result: dict, is_redraft: bool) -> dict:
    """Redraft has no future draft capital or multi-year age window.

    Drop top-3 pick odds and leave ``outlook`` unset so the Playoff Impact
    card only shows this-season odds, wins, and PPG.
    """
    out = dict(result or {})
    out["scoring_type"] = "redraft" if is_redraft else "dynasty"
    if not is_redraft:
        return out
    for bucket in ("before", "after", "delta"):
        row = dict(out.get(bucket) or {})
        row.pop("top3_pick_pct", None)
        out[bucket] = row
    out["outlook"] = None
    return out


def build_ppg_map(ctx: dict) -> tuple[dict, dict]:
    """
    Build (ppg_map, pos_map).

    Point projections come from Sleeper weekly lines only (upcoming week, or
    week 1 in preseason). Players Sleeper does not project are omitted.

    Returns:
        ppg_map  — {str(player_id): {"ppg": float, "pos": str}}
        pos_map  — {str(player_id): str(position)}   (position fallback)
    """
    season       = int(ctx.get("season") or 0)
    current_week = int(ctx.get("current_week") or 0)
    _rss   = ctx.get("raw_scoring_settings") or {}

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
        logger.debug("suppressed exception", exc_info=True)

    ppg_map: dict = {}

    # Priority 1: Sleeper projections — the primary source at ALL points.
    # In-season we use the upcoming week; offseason/preseason (current_week == 0)
    # we use week 1 of the season as a representative weekly baseline. This keeps
    # the simulation on Sleeper's numbers year-round, with the usage cache only
    # filling players Sleeper doesn't cover.
    if season > 0:
        proj_week = current_week + 1 if current_week > 0 else 1
        try:
            from utils.utils import fetch_week_projections
            multi_map = fetch_week_projections(season, proj_week, _rss)
            from utils.fantasy_scoring import projection_points
            for pid, variants in (multi_map or {}).items():
                if not isinstance(variants, dict):
                    continue
                pts = projection_points(variants, _rss, pos_map.get(str(pid), ""))
                if pts > 0:
                    ppg_map[str(pid)] = {
                        "ppg": float(pts),
                        "pos": pos_map.get(str(pid), ""),
                    }
        except Exception:
            logger.debug("suppressed exception", exc_info=True)

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
    has_games = team_stats_have_played_games(team_stats)
    roster_map = ctx.get("roster_map") or {}

    roster_positions = ctx.get("roster_positions") or []

    # ── Case 1: no game data yet (preseason / offseason / Week-1 seed) ───────
    if not has_games:
        # Build the PPG map Sleeper-first (same source as in-season / swap sims)
        # so projections come from Sleeper at all points.
        season_ppg_map, pos_map = build_ppg_map(ctx)
        teams = _estimate_from_rosters(ctx, ppg_map=season_ppg_map, pos_map=pos_map)
        if not teams:
            return []
        remaining_weeks = list(range(1, playoff_week_start))
        matchups_by_week = _resolve_schedule(
            platform, league_id, season, remaining_weeks, teams, division_map
        )
        hist_avg = {t["roster_id"]: 0.0 for t in teams}
        hist_std = {t["roster_id"]: 0.0 for t in teams}
        week_profiles = _build_ctx_week_profiles(
            ctx, platform, league_id, season, remaining_weeks, 1.0,
            hist_avg, hist_std, pos_map, season_ppg_map, roster_positions,
        )
        result = _run_mc(teams, matchups_by_week, week_profiles, playoff_teams, n_sims, seed)
        for r in result:
            r["is_projected"] = True
        return result

    # ── Case 2: season complete ───────────────────────────────────────────────
    teams = _build_teams(team_stats, roster_map)
    if not teams:
        return []

    if current_week > regular_season_end:
        return _actual_results(teams, playoff_teams)

    # ── Case 3: in-season projection ─────────────────────────────────────────
    remaining_weeks = list(range(current_week + 1, playoff_week_start))
    matchups_by_week = _resolve_schedule(
        platform, league_id, season, remaining_weeks, teams, division_map
    )

    # Each remaining week uses that week's Sleeper projection blended with the
    # team's season-to-date average (captured before any mutation), so byes and
    # week-specific matchups flow through while keeping realized-scoring signal.
    season_ppg_map, pos_map = build_ppg_map(ctx)
    hist_avg = {t["roster_id"]: float(t["avg"]) for t in teams}
    hist_std = {t["roster_id"]: float(t["std"]) for t in teams}
    week_profiles = _build_ctx_week_profiles(
        ctx, platform, league_id, season, remaining_weeks, _WEEKLY_BLEND,
        hist_avg, hist_std, pos_map, season_ppg_map, roster_positions,
    )

    result = _run_mc(teams, matchups_by_week, week_profiles, playoff_teams, n_sims, seed)
    for r in result:
        r["is_projected"] = False
    return result


def _build_ctx_week_profiles(
    ctx, platform, league_id, season, remaining_weeks, blend,
    hist_avg_by_rid, hist_std_by_rid, pos_map, season_ppg_map, roster_positions,
) -> dict:
    """Shared per-week profile builder for the standalone playoff-odds page."""
    raw_ss = ctx.get("raw_scoring_settings") or {}
    roster_pid_map = {
        int(r["roster_id"]): [str(p) for p in (r.get("players") or [])]
        for r in (ctx.get("rosters") or []) if r.get("roster_id") is not None
    }
    week_ppg_maps = _build_week_ppg_maps(
        season, remaining_weeks, raw_ss, pos_map, season_ppg_map
    )
    return _compute_week_profiles(
        roster_pid_map, week_ppg_maps, pos_map, roster_positions,
        hist_avg_by_rid, hist_std_by_rid, blend,
    )


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
    # Tally starting slots by type. Empty provider slots used to return 0 PPG
    # for every team (Value Rankings Proj% stuck at 0.0%); fall back to a
    # standard lineup so projections still differentiate rosters.
    fixed_slots, flex_slots, sflex_slots, restricted, _ = _tally_starting_slots_or_default(
        roster_positions
    )

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

    leftover = sorted(
        [(pos, ppg) for pos in _FLEX_ELIGIBLE
         for ppg in by_pos.get(pos, [])[used.get(pos, 0):]],
        key=lambda x: x[1], reverse=True,
    )
    for name in RESTRICTED_FLEX_SLOTS:
        taken, leftover = _consume_eligible(
            leftover, restricted.get(name, 0), slot_eligible_positions(name),
        )
        starters.extend(taken)
    taken, leftover = _consume_eligible(leftover, flex_slots, _FLEX_ELIGIBLE)
    starters.extend(taken)

    sflex_pool = sorted(
        [("QB", ppg) for ppg in by_pos.get("QB", [])[used.get("QB", 0):]]
        + leftover,
        key=lambda x: x[1], reverse=True,
    )
    for i in range(sflex_slots):
        if i < len(sflex_pool):
            starters.append(sflex_pool[i])

    total = sum(ppg for _, ppg in starters)
    return total, starters


def _lineup_with_replacements(
    pids: list,
    ppg_map: dict,
    pos_map: dict,
    roster_positions: list,
) -> tuple[float, list[tuple[str, float]], list[float]]:
    """Optimal lineup plus, for each starter, the best benched replacement.

    Returns (total, starters, replacements) where:
      - starters       — [(pos, ppg), …] for each starting slot
      - replacements   — [replacement_ppg, …] aligned to starters: the projected
                         points of the best healthy rostered player eligible for
                         that slot if the starter is unavailable that week.

    Used by the injury model so an injured starter is replaced by the next best
    available player on the roster (the realistic loss = starter − replacement),
    not a flat fraction. Replacements are computed independently per starter
    (single-injury assumption), which slightly understates rare multi-injury
    weeks at one position — an acceptable approximation.
    """
    # Tally starting slots by type (mirrors _position_aware_lineup, including
    # the empty-slots default so injury modeling stays aligned with Proj%).
    fixed_slots, flex_slots, sflex_slots, restricted, _ = _tally_starting_slots_or_default(
        roster_positions
    )

    # Per-position fallback averages for players lacking a projection
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
    # Each starter recorded as (pos, ppg, slot_type) so we can find its replacement
    starters_full: list[tuple[str, float, str]] = []

    for slot_pos, count in fixed_slots.items():
        pool = by_pos.get(slot_pos, [])
        for _ in range(count):
            i = used.get(slot_pos, 0)
            ppg = pool[i] if i < len(pool) else 0.0
            starters_full.append((slot_pos, ppg, slot_pos))
            used[slot_pos] = i + 1

    leftover = sorted(
        [(pos, ppg) for pos in _FLEX_ELIGIBLE
         for ppg in by_pos.get(pos, [])[used.get(pos, 0):]],
        key=lambda x: x[1], reverse=True,
    )
    for name in RESTRICTED_FLEX_SLOTS:
        taken, leftover = _consume_eligible(
            leftover, restricted.get(name, 0), slot_eligible_positions(name),
        )
        for pos, ppg in taken:
            starters_full.append((pos, ppg, name))
            used[pos] = used.get(pos, 0) + 1
    taken, leftover = _consume_eligible(leftover, flex_slots, _FLEX_ELIGIBLE)
    for pos, ppg in taken:
        starters_full.append((pos, ppg, "FLEX"))
        used[pos] = used.get(pos, 0) + 1

    sflex_pool = sorted(
        [("QB", ppg) for ppg in by_pos.get("QB", [])[used.get("QB", 0):]]
        + leftover,
        key=lambda x: x[1], reverse=True,
    )
    for i in range(sflex_slots):
        if i < len(sflex_pool):
            pos, ppg = sflex_pool[i]
            starters_full.append((pos, ppg, "SFLEX"))
            used[pos] = used.get(pos, 0) + 1

    # Best benched player per position (first beyond the starters already used)
    def _best_bench(positions: set) -> float:
        best = 0.0
        for p in positions:
            pool = by_pos.get(p, [])
            i = used.get(p, 0)
            if i < len(pool):
                best = max(best, pool[i])
        return best

    starters: list[tuple[str, float]] = []
    replacements: list[float] = []
    for pos, ppg, slot_type in starters_full:
        if slot_type == "FLEX":
            repl = _best_bench(_FLEX_ELIGIBLE)
        elif slot_type == "SFLEX":
            repl = _best_bench({"QB"} | _FLEX_ELIGIBLE)
        elif slot_type in RESTRICTED_FLEX_SLOTS:
            repl = _best_bench(set(slot_eligible_positions(slot_type)))
        else:
            repl = _best_bench({slot_type})
        starters.append((pos, ppg))
        replacements.append(repl)

    total = sum(ppg for _, ppg in starters)
    return total, starters, replacements


def _team_week_profile(
    pids: list,
    ppg_map: dict,
    pos_map: dict,
    roster_positions: list,
    hist_avg: float,
    hist_std: float,
    blend: float,
) -> dict:
    """Build a single team's (mean, std, injury params) for one week.

    mean = blend × week_projection + (1−blend) × season_avg

    Preseason uses blend = 1.0 (pure projection — the optimal-lineup total from
    Sleeper's weekly projections); in-season blends the week's projection with
    the season-to-date average.

    Injury loss per starter = (starter − best replacement) put on the same scale
    as the projection term (× blend), so a no-op stays a no-op and in-season
    injuries are discounted consistently with the projection's weight.
    """
    proj, starters, repls = _lineup_with_replacements(
        pids, ppg_map, pos_map, roster_positions
    )
    mean = blend * proj + (1.0 - blend) * hist_avg
    if hist_std and hist_std > 0:
        std = max(hist_std, _MIN_STD)
    else:
        std = max(_team_std_from_starters(starters), _MIN_STD)
    scale = blend
    lost = np.array(
        [max(s_ppg - r_ppg, 0.0) * scale for (_, s_ppg), r_ppg in zip(starters, repls)],
        dtype=np.float32,
    )
    haz = np.array(
        [_INJURY_HAZARD.get(str(pos).upper(), _INJURY_HAZARD_DEFAULT) for pos, _ in starters],
        dtype=np.float32,
    )
    return {"mean": round(mean, 2), "std": round(std, 2), "lost": lost, "haz": haz}


def _build_week_ppg_maps(
    season: int,
    weeks: list[int],
    raw_scoring_settings: dict,
    pos_map: dict,
    season_ppg_map: dict,
) -> dict[int, dict]:
    """Fetch each week's Sleeper projections so the sim uses week-specific PPG.

    Weekly maps are disk-cached (get_week_projections_cached) so repeated sims
    are cheap. A player's bye week naturally yields no weekly entry (PPG 0), so
    the optimal lineup routes around it. If a week's Sleeper map is too sparse
    (future weeks not yet published preseason), that week falls back to the
    robust season baseline map so team strength never collapses to zero.
    """
    out: dict[int, dict] = {}
    try:
        from utils.utils import (
            fetch_week_projections, get_week_projections_cached,
        )
    except Exception:
        return {w: season_ppg_map for w in weeks}

    from utils.fantasy_scoring import projection_points
    for w in weeks:
        wk_map: dict = {}
        try:
            multi = get_week_projections_cached(
                season, w,
                lambda s, wk: fetch_week_projections(s, wk, raw_scoring_settings),
            ) or {}
            for pid, variants in multi.items():
                if not isinstance(variants, dict):
                    continue
                pts = projection_points(variants, raw_scoring_settings, pos_map.get(str(pid), ""))
                if pts > 0:
                    wk_map[str(pid)] = {"ppg": float(pts), "pos": pos_map.get(str(pid), "")}
        except Exception:
            wk_map = {}
        # Too sparse to be a real weekly slate → use the season baseline instead.
        out[w] = wk_map if len(wk_map) >= 50 else season_ppg_map
    return out


def _compute_week_profiles(
    roster_pid_map: dict[int, list],
    week_ppg_maps: dict[int, dict],
    pos_map: dict,
    roster_positions: list,
    hist_avg_by_rid: dict[int, float],
    hist_std_by_rid: dict[int, float],
    blend: float,
) -> dict[int, dict]:
    """Per-week (mean, std, injury params) for every team."""
    profiles: dict[int, dict] = {}
    for week, ppg_map in week_ppg_maps.items():
        wp: dict[int, dict] = {}
        for rid, pids in roster_pid_map.items():
            wp[rid] = _team_week_profile(
                pids, ppg_map, pos_map, roster_positions,
                hist_avg_by_rid.get(rid, 0.0), hist_std_by_rid.get(rid, 0.0),
                blend,
            )
        profiles[week] = wp
    return profiles


def _estimate_from_rosters(
    ctx: dict,
    ppg_map: Optional[dict] = None,
    pos_map: Optional[dict] = None,
) -> list[dict]:
    """
    Build synthetic team scoring profiles for preseason simulation.

    PPG source: Sleeper weekly projections aggregated into an active-week PPG.
    Players Sleeper does not project contribute 0.

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
                "starters":  starters,
            })
        return teams

    # ── Source 1: canonical Sleeper season projection ────────────────────────
    ppg_map: dict[str, dict] = {}
    try:
        from utils.projection_resolver import resolve_projected_ppg_many
        from utils.utils import load_players_index
        _players_idx = load_players_index() or {}
        _roster_ids = {str(pid) for roster in (ctx.get("rosters") or [])
                       for pid in (roster.get("players") or [])}
        _positions = {pid: str((_players_idx.get(pid) or {}).get("pos") or "")
                      for pid in _roster_ids}
        sleeper_data = resolve_projected_ppg_many(
            _roster_ids, ctx.get("raw_scoring_settings") or ctx.get("scoring_settings") or {},
            season, positions=_positions)
        for pid, info in sleeper_data.items():
            if (info.get("ppg") or 0) > 0:
                ppg_map[str(pid)] = {"ppg": info["ppg"], "pos": _positions.get(str(pid), "")}
        if ppg_map:
            logger.info("[playoff_odds] Using Sleeper projections: %d players", len(ppg_map))
    except Exception as exc:
        logger.warning("[playoff_odds] Sleeper projections unavailable: %s", exc)

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
        logger.debug("suppressed exception", exc_info=True)

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
            "starters":  starters,
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
        from utils.utils import fetch_week_projections
        multi_map = fetch_week_projections(season, next_week, raw_ss)
    except Exception as exc:
        logger.warning("[playoff_odds] Sleeper weekly proj unavailable: %s", exc)
        return

    if not multi_map:
        return

    from utils.fantasy_scoring import projection_points

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
        logger.debug("suppressed exception", exc_info=True)

    # Flatten multi-variant projections to a single pts value per player
    week_ppg: dict[str, dict] = {}
    for pid, variants in multi_map.items():
        if not isinstance(variants, dict):
            continue
        pts = projection_points(variants, raw_ss, pos_map.get(str(pid), ""))
        if pts > 0:
            week_ppg[pid] = {"ppg": pts, "pos": pos_map.get(pid, "")}

    updated = 0
    for team in teams:
        rid  = team["roster_id"]
        pids = rid_to_pids.get(rid)
        if not pids:
            continue
        proj_score, starters = _position_aware_lineup(pids, week_ppg, pos_map, roster_positions)
        if proj_score <= 0:
            continue
        historical_avg  = team["avg"]
        team["avg"]     = round(blend * proj_score + (1 - blend) * historical_avg, 1)
        # Keep the projected starters so the Monte Carlo engine can model
        # per-week injury hazard for this team in-season too.
        team["starters"] = starters
        updated += 1

    if updated:
        logger.info(
            "[playoff_odds] Blended weekly proj (%.0f%%) for %d teams (week %d)",
            blend * 100, updated, next_week,
        )


# ---------------------------------------------------------------------------
# In-season team data from team_stats DataFrame
# ---------------------------------------------------------------------------

def team_stats_have_played_games(team_stats) -> bool:
    """True only when at least one team has a real game on the board.

    Week 1 seeds a 0-0 ``team_stats`` frame so Standings is not blank before
    any leg finalizes. That frame must NOT flip the sim into the in-season
    path — otherwise ``_build_teams`` would run on a RangeIndex and mint
    fake roster_ids (0..N-1), which then mis-attribute playoff odds to the
    Front Office Report / trade-window card while the odds *table* (keyed by
    team name) still looks coherent.
    """
    if team_stats is None:
        return False
    try:
        if getattr(team_stats, "empty", True):
            return False
    except Exception:
        return False
    for col in ("G", "Wins", "Losses", "Ties"):
        if col not in getattr(team_stats, "columns", []):
            continue
        try:
            if float(team_stats[col].fillna(0).sum()) > 0:
                return True
        except Exception:
            continue
    if "PF" in getattr(team_stats, "columns", []):
        try:
            if float(team_stats["PF"].fillna(0).sum()) > 0:
                return True
        except Exception:
            pass
    return False


def _roster_id_for_owner(owner: str, roster_map: Optional[dict], index_val) -> Optional[int]:
    """Map a standings owner name (or index) onto a real league roster_id."""
    name = str(owner or "").strip()
    if roster_map and name:
        for rid, mapped in roster_map.items():
            if str(mapped or "").strip() == name:
                try:
                    return int(rid)
                except (TypeError, ValueError):
                    continue
    try:
        rid = int(index_val)
    except (TypeError, ValueError):
        return None
    if not roster_map:
        return rid
    # Only accept the index when it is a real roster key — never a RangeIndex
    # slot that merely collides with a team id (the Week-1 seed bug).
    if str(rid) in roster_map:
        return rid
    try:
        if rid in roster_map:
            return rid
    except Exception:
        pass
    return None


def _build_teams(team_stats, roster_map: Optional[dict] = None) -> list[dict]:
    """Build sim team rows from standings, keyed by real roster_id.

    ``team_stats`` is owner-oriented with a default RangeIndex. Never treat
    that index as roster_id when a roster_map is available — resolve through
    the owner name instead.
    """
    teams = []
    seen: set[int] = set()
    for idx in team_stats.index:
        row = team_stats.loc[idx]
        owner = str(row.get("owner", "") or "")
        rid = _roster_id_for_owner(owner, roster_map, idx)
        if rid is None or rid in seen:
            continue
        seen.add(rid)
        wins = int(row.get("Wins", 0) or 0)
        losses = int(row.get("Losses", 0) or 0)
        ties = int(row.get("Ties", 0) or 0)
        pf = float(row.get("PF", 0) or 0)
        avg = float(row.get("AVG", 80) or 80)
        std = max(float(row.get("STD", 15) or 15), _MIN_STD)
        teams.append({
            "roster_id": rid,
            "name": owner or f"Team {rid}",
            "wins": wins, "losses": losses, "ties": ties,
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


def _sample_injury_duration(rng: "np.random.Generator", shape) -> "np.ndarray":
    """Draw injury durations (weeks) from the empirical distribution.

    Always draws a full `shape` array (even where unused) so the rng stream is
    consumed identically every week — preserving common random numbers between
    before/after trade sims.
    """
    u = rng.random(shape, dtype=np.float32)
    dur = np.ones(shape, dtype=np.int16)
    thresh = 0.0
    for choice, prob in zip(_INJURY_DURATION_CHOICES, _INJURY_DURATION_PROBS):
        dur[u >= thresh] = choice
        thresh += prob
    return dur


def _apply_injuries(
    rng: "np.random.Generator",
    scores: "np.ndarray",
    lost: Optional["np.ndarray"],
    onset_haz: Optional["np.ndarray"],
    state: Optional["np.ndarray"],
    n_sims: int,
) -> tuple["np.ndarray", Optional["np.ndarray"]]:
    """Subtract multi-week injury losses and advance per-starter injury state.

    `state` is an (n_sims, k) int array of weeks-remaining-out per starter slot,
    carried across the schedule. Each week a healthy starter newly goes down with
    its position ONSET rate and is then out for a sampled duration; while out, the
    substitution loss (starter − best bench replacement) applies. Returns the
    adjusted scores and the updated state. The rng is shared so before/after sims
    stay paired.
    """
    if lost is None or state is None or lost.shape[0] == 0:
        return scores, state
    k = lost.shape[0]
    cur = state[:, :k]
    currently_out = cur > 0
    onset = (~currently_out) & (rng.random((n_sims, k), dtype=np.float32) < onset_haz)
    dur = _sample_injury_duration(rng, (n_sims, k))
    cur = np.where(onset, dur, cur)
    is_out = cur > 0
    lost_pts = (is_out * lost).sum(axis=1)
    state[:, :k] = np.maximum(cur - 1, 0)   # decrement for next week
    return np.maximum(scores - lost_pts, 0).astype(np.float32), state


# Module-level lock so overlapping suggestion requests (rapid strategy-chip
# clicks) share one freeze build instead of each replaying the full season.
_FREEZE_LOCK = threading.Lock()


def _rng_for_team(seed: Optional[int], roster_id: int) -> "np.random.Generator":
    """Independent deterministic RNG stream per roster.

    Splitting the seed by roster_id lets a trade swap re-simulate only the
    teams whose lineups changed while every other team's weekly scores stay
    bit-identical (common random numbers) — without replaying the whole league.
    """
    if seed is None:
        return np.random.default_rng()
    mix = (int(seed) ^ ((int(roster_id) + 1) * 0x9E3779B9)) & 0xFFFFFFFF
    return np.random.default_rng(mix)


def _playing_weeks_by_rid(
    matchups_by_week: dict[int, list[tuple[int, int]]],
) -> dict[int, list[int]]:
    out: dict[int, list[int]] = defaultdict(list)
    for week in sorted(matchups_by_week.keys()):
        seen: set[int] = set()
        for a, b in matchups_by_week[week]:
            if a not in seen:
                out[a].append(week)
                seen.add(a)
            if b not in seen:
                out[b].append(week)
                seen.add(b)
    return out


def _team_kmax(week_profiles: dict, roster_id: int) -> int:
    k = 0
    for wp in week_profiles.values():
        p = wp.get(roster_id)
        if p is not None:
            k = max(k, int(p["lost"].shape[0]))
    return k


def _score_one_team(
    roster_id: int,
    playing_weeks: list[int],
    week_profiles: dict,
    fb_avg: float,
    fb_std: float,
    n_sims: int,
    seed: Optional[int],
    kmax: int,
) -> dict[int, "np.ndarray"]:
    """Draw this team's weekly scores from its own RNG stream.

    Injury state carries across the weeks this team actually plays (bye weeks
    are skipped, matching the historical matchup-loop behaviour).
    """
    rng = _rng_for_team(seed, roster_id)
    state = np.zeros((n_sims, kmax), dtype=np.int16) if kmax else None
    out: dict[int, np.ndarray] = {}
    for week in playing_weeks:
        p = (week_profiles.get(week) or {}).get(roster_id)
        if p is not None:
            mean, std = float(p["mean"]), float(p["std"])
            lost = p["lost"]
            haz = p["haz"]
            onset = haz / np.float32(_INJURY_MEAN_DURATION) if haz.shape[0] else haz
        else:
            mean, std, lost, onset = fb_avg, fb_std, None, None
        sa = _sample_scores(rng, mean, std, n_sims)
        sa, state = _apply_injuries(rng, sa, lost, onset, state, n_sims)
        out[week] = sa
    return out


def _simulate_week_scores(
    teams: list[dict],
    matchups_by_week: dict[int, list[tuple[int, int]]],
    week_profiles: dict[int, dict],
    n_sims: int,
    seed: Optional[int],
    playing_weeks: Optional[dict[int, list[int]]] = None,
) -> tuple[dict[int, dict[int, "np.ndarray"]], "np.ndarray", dict[int, list[int]]]:
    """Draw weekly scores for every team.

    Returns ``(score_map, games_per_team, playing_weeks)`` where
    ``score_map[week][roster_id]`` is an ``(n_sims,)`` array.
    """
    n = len(teams)
    idx = {t["roster_id"]: i for i, t in enumerate(teams)}
    if playing_weeks is None:
        playing_weeks = _playing_weeks_by_rid(matchups_by_week)
    games_per_team = np.zeros(n, dtype=np.float32)
    for rid, weeks in playing_weeks.items():
        i = idx.get(rid)
        if i is not None:
            games_per_team[i] = float(len(weeks))
    fb_avg = {t["roster_id"]: float(t.get("avg", 0.0)) for t in teams}
    fb_std = {
        t["roster_id"]: max(float(t.get("std", _MIN_STD)), _MIN_STD) for t in teams
    }
    score_map: dict[int, dict[int, np.ndarray]] = defaultdict(dict)
    for t in teams:
        rid = t["roster_id"]
        sm = _score_one_team(
            rid,
            playing_weeks.get(rid, []),
            week_profiles,
            fb_avg[rid],
            fb_std[rid],
            n_sims,
            seed,
            _team_kmax(week_profiles, rid),
        )
        for week, arr in sm.items():
            score_map[week][rid] = arr
    return score_map, games_per_team, playing_weeks


def _pack_mc_results(
    teams: list[dict],
    wins: "np.ndarray",
    pf: "np.ndarray",
    ties_gained: "np.ndarray",
    games_per_team: "np.ndarray",
    playoff_teams: int,
    n_sims: int,
) -> list[dict]:
    n = len(teams)
    n_byes = _n_byes(playoff_teams)
    # Rank by wins desc, pf desc (wins dominate). Compute in float64: wins*1e6
    # reaches ~1.3e7 in a full season, past float32's exact-integer range
    # (2^24 ≈ 1.68e7, ULP ≥ 1.0 above ~8.4e6), which silently collapses the PF
    # tiebreaker for teams within ~1 pt — flipping seeding in tight bubble races.
    rank_key  = wins.astype(np.float64) * 1e6 + pf.astype(np.float64)
    team_rank = np.argsort(np.argsort(-rank_key, axis=1), axis=1)  # 0 = best

    in_playoffs = (team_rank < playoff_teams).mean(axis=0) * 100
    got_bye     = (team_rank < n_byes).mean(axis=0) * 100 if n_byes else np.zeros(n)
    is_first    = (team_rank == 0).mean(axis=0) * 100

    # Draft-pick outlook — the inverse of playoff success. Dynasty/rookie draft
    # order is reverse standings, so the worst finishers hold the best picks.
    # rank n-1 (dead last)  → pick 1.01;  rank >= n-3 → a top-3 pick.
    pick_one    = (team_rank == n - 1).mean(axis=0) * 100
    top3_pick   = (team_rank >= max(0, n - 3)).mean(axis=0) * 100
    # Projected draft slot: best record (rank 0) drafts last (slot n),
    # worst record (rank n-1) drafts first (slot 1).
    avg_slot    = (n - team_rank).mean(axis=0)

    init_wins   = np.array([t["wins"]   for t in teams], dtype=np.float32)
    init_losses = np.array([t["losses"] for t in teams], dtype=np.float32)
    init_ties   = np.array([t["ties"]   for t in teams], dtype=np.float32)
    avg_wins    = wins.mean(axis=0)
    avg_ties_gained = ties_gained.mean(axis=0)
    avg_final_ties  = init_ties + avg_ties_gained
    # Use per-team games scheduled so bye weeks don't inflate projected losses.
    # `wins` already credits 0.5 per tie, so subtract that half back out — a tie is
    # a tie, not half a loss.
    avg_losses  = (init_losses + games_per_team
                   - (avg_wins - init_wins) - 0.5 * avg_ties_gained)

    # Mathematical clinch / elimination (consistent with the sim's top-N-by-record
    # seeding). A team has CLINCHED a berth if, even losing out while everyone who
    # could pass them wins out, fewer than `playoff_teams` teams can finish ahead.
    # It's ELIMINATED if at least `playoff_teams` teams are guaranteed ahead even
    # if it wins out. Bounds are conservative (ignore the joint schedule and
    # tiebreakers) so they never falsely declare certainty.
    rem      = games_per_team               # remaining games per team
    t_min    = init_wins                    # worst case: lose out
    t_max    = init_wins + rem              # best case: win out
    opp_max  = init_wins + rem
    clinched   = np.zeros(n, dtype=bool)
    eliminated = np.zeros(n, dtype=bool)
    for i in range(n):
        can_be_ahead     = int(np.sum(opp_max >= t_min[i])) - 1   # exclude self
        guaranteed_ahead = int(np.sum(init_wins > t_max[i]))
        clinched[i]   = can_be_ahead < playoff_teams
        eliminated[i] = guaranteed_ahead >= playoff_teams

    # A projection should never read a literal 100% or 0% unless the spot is
    # mathematically settled: with finite sims a saturated outcome just means the
    # sample never hit the rare collapse/surge. Clamp the undecided ones.
    def _playoff_pct(i: int) -> float:
        if clinched[i]:
            return 100.0
        if eliminated[i]:
            return 0.0
        return round(min(99.9, max(0.1, float(in_playoffs[i]))), 1)

    return [{
        "roster_id":        t["roster_id"],
        "team_name":        t["name"],
        "wins":             t["wins"],
        "losses":           t["losses"],
        "ties":             t["ties"],
        "playoff_pct":      _playoff_pct(i),
        "bye_pct":          round(min(99.9, float(got_bye[i])), 1),
        "first_seed_pct":   round(min(99.9, float(is_first[i])), 1),
        "pick_one_pct":     round(min(99.9, float(pick_one[i])), 1),
        "top3_pick_pct":    round(min(99.9, float(top3_pick[i])), 1),
        "avg_draft_slot":   round(float(avg_slot[i]),     1),
        "miss_pct":         round(100 - _playoff_pct(i), 1),
        "avg_final_wins":   round(float(avg_wins[i]),     1),
        "avg_final_losses": round(float(avg_losses[i]),   1),
        "avg_final_ties":   round(float(avg_final_ties[i]), 1),
        "n_sims":           n_sims,
        "is_complete":      False,
    } for i, t in enumerate(teams)]


def _standings_from_score_map(
    teams: list[dict],
    matchups_by_week: dict[int, list[tuple[int, int]]],
    score_map: dict[int, dict[int, "np.ndarray"]],
    playoff_teams: int,
    n_sims: int,
    games_per_team: "np.ndarray",
    overlays: Optional[dict[int, dict[int, "np.ndarray"]]] = None,
) -> list[dict]:
    """Resolve the remaining schedule from frozen (optionally overlaid) scores."""
    n = len(teams)
    idx = {t["roster_id"]: i for i, t in enumerate(teams)}
    wins = np.tile([t["wins"] for t in teams], (n_sims, 1)).astype(np.float32)
    pf   = np.tile([t["pf"]   for t in teams], (n_sims, 1)).astype(np.float32)
    ties_gained = np.zeros((n_sims, n), dtype=np.float32)

    def _score(week: int, rid: int) -> "np.ndarray":
        if overlays:
            ov = overlays.get(rid)
            if ov is not None and week in ov:
                return ov[week]
        return score_map[week][rid]

    for week in sorted(matchups_by_week.keys()):
        for rid_a, rid_b in matchups_by_week[week]:
            ia = idx.get(rid_a)
            ib = idx.get(rid_b)
            if ia is None or ib is None:
                continue
            sa = _score(week, rid_a)
            sb = _score(week, rid_b)
            tie    = np.abs(sa - sb) < _TIE_MARGIN
            a_wins = (sa > sb) & ~tie
            b_wins = (sb > sa) & ~tie
            tie_f = tie.astype(np.float32)
            wins[:, ia] += a_wins.astype(np.float32) + 0.5 * tie_f
            wins[:, ib] += b_wins.astype(np.float32) + 0.5 * tie_f
            ties_gained[:, ia] += tie_f
            ties_gained[:, ib] += tie_f
            pf[:, ia] += sa
            pf[:, ib] += sb

    return _pack_mc_results(
        teams, wins, pf, ties_gained, games_per_team, playoff_teams, n_sims,
    )


def _ensure_freeze(sim_state: dict, n_sims: int) -> dict:
    """Draw one baseline season and cache weekly scores on ``sim_state``.

    Trade-suggestion ranking calls this once, then overlays 1–2 re-scored
    teams per package. The cache is keyed by ``n_sims`` so a 2k suggestion
    freeze is never reused for a 10k standings-page run (or vice versa).
    """
    freeze = sim_state.get("_mc_freeze")
    if freeze is not None and freeze.get("n_sims") == n_sims:
        return freeze
    with _FREEZE_LOCK:
        freeze = sim_state.get("_mc_freeze")
        if freeze is not None and freeze.get("n_sims") == n_sims:
            return freeze
        score_map, games, playing = _simulate_week_scores(
            sim_state["teams"],
            sim_state["matchups"],
            sim_state.get("week_profiles") or {},
            n_sims,
            sim_state.get("seed"),
        )
        result = _standings_from_score_map(
            sim_state["teams"],
            sim_state["matchups"],
            score_map,
            sim_state["playoff_teams"],
            n_sims,
            games,
        )
        freeze = {
            "n_sims": n_sims,
            "score_map": score_map,
            "result": result,
            "games_per_team": games,
            "playing_weeks": playing,
        }
        sim_state["_mc_freeze"] = freeze
        return freeze


def _standings_with_overrides(
    sim_state: dict,
    after_profiles: dict,
    overrides: dict,
    n_sims: int,
) -> list[dict]:
    """Replay standings after re-scoring only the teams in ``overrides``."""
    freeze = _ensure_freeze(sim_state, n_sims)
    if not overrides:
        return freeze["result"]
    teams = sim_state["teams"]
    fb_avg = {t["roster_id"]: float(t.get("avg", 0.0)) for t in teams}
    fb_std = {
        t["roster_id"]: max(float(t.get("std", _MIN_STD)), _MIN_STD) for t in teams
    }
    playing = freeze["playing_weeks"]
    seed = sim_state.get("seed")
    overlays: dict[int, dict[int, np.ndarray]] = {}
    for rid in overrides:
        overlays[rid] = _score_one_team(
            rid,
            playing.get(rid, []),
            after_profiles,
            fb_avg.get(rid, 0.0),
            fb_std.get(rid, _MIN_STD),
            n_sims,
            seed,
            _team_kmax(after_profiles, rid),
        )
    return _standings_from_score_map(
        teams,
        sim_state["matchups"],
        freeze["score_map"],
        sim_state["playoff_teams"],
        n_sims,
        freeze["games_per_team"],
        overlays=overlays,
    )


def _run_mc(
    teams: list[dict],
    matchups_by_week: dict[int, list[tuple[int, int]]],
    week_profiles: dict[int, dict],
    playoff_teams: int,
    n_sims: int,
    seed: Optional[int],
) -> list[dict]:
    """Monte Carlo over the remaining schedule.

    Each team is scored from its own RNG stream (so a later swap can redo just
    that team). Each week pulls that team's week-specific (mean, std, injury
    params) from week_profiles[week][roster_id], so the sim uses Sleeper's
    projection for that exact week (handling byes and matchup-specific
    strength) and the per-week injury substitution loss. Falls back to a team's
    stored avg/std if a week profile is missing.
    """
    score_map, games, _ = _simulate_week_scores(
        teams, matchups_by_week, week_profiles, n_sims, seed,
    )
    return _standings_from_score_map(
        teams, matchups_by_week, score_map, playoff_teams, n_sims, games,
    )
