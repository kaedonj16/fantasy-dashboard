"""
Multitask prediction outputs for the breakout engine.

Derives three additional predictions from existing component scores:
  hit_probability  - P(top-12 fantasy finish at position next season)
  cumulative_ppr   - expected total PPR points over the next 2 seasons
  peak_ppr         - expected peak single-season PPR

All three are heuristic models calibrated per position, designed to be
interpretable and consistent with the existing rule-based scoring engine.
"""
from __future__ import annotations

from typing import Optional

# Base breakout hit rates calibrated specifically for the NON-ESTABLISHED
# candidate pool (players who haven't yet had a big season — the "emerging"
# tier the engine targets).  These are lower than NFL-wide base rates because
# established stars are excluded from scoring:  backtest 2022+2023 shows
# actual breakout rate ≈ 17% overall, with the 10-20% predicted bucket
# landing at only 2-8% actual.
_BASE_HIT_RATE: dict[str, float] = {
    "QB": 0.25,   # was 0.38 — QB candidates are mostly backups / 2nd-year starters
    "RB": 0.12,   # was 0.22
    "WR": 0.09,   # was 0.15 — largest positional pool, hardest to break through
    "TE": 0.12,   # was 0.20
}

# Positional PPR floor used when usage data is missing entirely.
# Set near the top-12 floor for each position (2024 actuals):
# WR top-12 floor ~241, TE ~145, RB ~245, QB ~288.
_BASE_SEASON_PPR: dict[str, float] = {
    "QB":  290.0,
    "RB":  245.0,
    "WR":  240.0,
    "TE":  145.0,
}

# PPR points per target/carry — fitted from 2024 season actuals
# (players with ≥8 games; rushing pts isolated from receiving for RBs).
# WR: 1.73 | TE: 1.76 | RB targets: 1.56 | RB carries: 0.61
_TARGET_PPR_RATE: dict[str, float] = {
    "WR": 1.75,
    "TE": 1.75,
    "RB": 1.60,
    "QB": 0.0,   # QBs scored via pass yards/TDs, not receiver targets
}
_CARRY_PPR_RATE: dict[str, float] = {
    "RB": 0.62,   # rush yards + rush TDs per carry (2024 avg: 0.61)
    "WR": 0.15,   # end-arounds / gadget plays
    "TE": 0.05,
    "QB": 0.0,
}
# Snap-share → season PPR fallback scale (fitted from 2024 actuals).
# Used only when no target/carry projection is available.
_SNAP_PPR_SCALE: dict[str, float] = {
    "QB":  252.0,
    "RB":  313.0,
    "WR":  186.0,
    "TE":  127.0,
}

# Per-position uplift applied to the raw season-PPR estimate before returning.
# Calibrated from 2022-2023 backtests after fixing the source-stats format bug
# (pre-existing usage_rows files stored avg_targets per game, not season totals,
# causing all WR/RB/TE to hit the PPR floor and appear ~20% under-predicted).
# Once source stats are correctly populated, the base estimate is well-calibrated:
# 2023→2024 bias is +0.2 ppg without uplift. QB retains a small uplift because
# snap-share data is unavailable in the pre-existing format (avg_off_snap_pct=0).
_SEASON_PPR_UPLIFT: dict[str, float] = {
    "QB": 1.08,
    "RB": 1.00,
    "WR": 1.00,
    "TE": 1.00,
}


def _estimate_season_ppr(
    position: str,
    projected_usage: dict,
    efficiency_metrics: Optional[dict],
    prev_usage: Optional[dict],
) -> float:
    """
    Estimate single-season PPR from projected usage + efficiency.
    Falls back to snap-share scaling when target/carry data is thin.
    A per-position uplift (_SEASON_PPR_UPLIFT) corrects the systematic
    downward bias observed in 2022-2023 backtests.
    """
    pos = (position or "WR").upper()
    base = _BASE_SEASON_PPR.get(pos, 150.0)
    uplift = _SEASON_PPR_UPLIFT.get(pos, 1.15)

    proj_targets = float(projected_usage.get("targets") or projected_usage.get("projected_targets") or 0)
    proj_carries = float(projected_usage.get("carries") or projected_usage.get("projected_carries") or 0)
    proj_snaps   = float(projected_usage.get("snap_share") or projected_usage.get("projected_snap_share") or 0)

    # Use previous season as floor when projections are thin
    if proj_targets == 0 and proj_carries == 0 and prev_usage:
        proj_targets = float(prev_usage.get("targets") or 0)
        proj_carries = float(prev_usage.get("carries") or 0)
        proj_snaps   = float(prev_usage.get("snap_share") or 0)

    # QBs score via passing stats, not targets/carries — bypass that branch entirely
    if pos == "QB":
        if proj_snaps > 0:
            return proj_snaps * _SNAP_PPR_SCALE["QB"] * uplift
        if prev_usage:
            prev_pass_att = float(prev_usage.get("pass_attempts") or 0)
            if prev_pass_att > 0:
                # Estimate snap share: a full-time starter throws ~580 attempts/season
                est_snaps = min(prev_pass_att / 580.0, 1.0)
                return est_snaps * _SNAP_PPR_SCALE["QB"] * uplift
        return base * 0.5 * uplift

    if proj_targets > 0 or proj_carries > 0:
        ppr = (
            proj_targets * _TARGET_PPR_RATE.get(pos, 7.0)
            + proj_carries * _CARRY_PPR_RATE.get(pos, 1.8)
        )
        # Efficiency multiplier: above-average efficiency gets a 10% boost
        if efficiency_metrics:
            ypt = float(efficiency_metrics.get("yards_per_target") or 0)
            ypc = float(efficiency_metrics.get("yards_per_carry") or 0)
            if (pos in ("WR", "TE") and ypt >= 9.0) or (pos == "RB" and ypc >= 4.8):
                ppr *= 1.10
        return max(ppr, base * 0.4) * uplift

    if proj_snaps > 0:
        return proj_snaps * _SNAP_PPR_SCALE.get(pos, 350.0) * uplift

    return base * 0.5 * uplift  # thin data → conservative estimate


def calculate_hit_probability(
    breakout_score: float,
    readiness_score: float,
    confidence_score: float,
    position: str,
    opportunity_score: float = 0.0,
    role_trajectory_score: float = 0.0,
) -> float:
    """
    P(player finishes top-12 at position next season).

    Calibrated from 2022-2024 backtest data (639 paired player-seasons).
    Uses piecewise linear interpolation over empirical hit rates by score
    bucket, then applies opportunity and trajectory modifiers.
    """
    # Empirical hit rates by score bucket (from 2022-2024 backtest):
    # breakout_score → p(top-12 finish at position)
    # Smoothed to enforce monotonicity; small-sample high buckets capped conservatively.
    _SCORE_CURVE: dict[str, list[tuple[float, float]]] = {
        "WR": [(0, 0.00), (40, 0.02), (55, 0.03), (65, 0.07), (72, 0.25), (80, 0.33), (100, 0.40)],
        "RB": [(0, 0.01), (40, 0.04), (55, 0.05), (65, 0.15), (72, 0.40), (80, 0.50), (100, 0.58)],
        "TE": [(0, 0.00), (40, 0.05), (50, 0.18), (60, 0.30), (70, 0.38), (80, 0.48), (100, 0.55)],
        "QB": [(0, 0.05), (45, 0.10), (55, 0.12), (65, 0.22), (72, 0.38), (80, 0.48), (100, 0.55)],
    }

    pos = (position or "WR").upper()
    score = max(0.0, min(100.0, float(breakout_score)))
    curve = _SCORE_CURVE.get(pos, _SCORE_CURVE["WR"])

    # Piecewise linear interpolation
    base_prob = curve[0][1]
    for i in range(len(curve) - 1):
        s0, p0 = curve[i]
        s1, p1 = curve[i + 1]
        if s0 <= score <= s1:
            t = (score - s0) / (s1 - s0) if s1 > s0 else 0.0
            base_prob = p0 + t * (p1 - p0)
            break
    else:
        base_prob = curve[-1][1]

    # Opportunity modifier: significant vacated opportunity provides additional
    # upside for WR/TE/RB — they have a clear path to volume.
    opp = float(opportunity_score or 0.0)
    if opp >= 60 and pos in ("WR", "TE", "RB"):
        opp_boost = min(0.08, (opp - 60) / 40 * 0.10)
        base_prob = min(0.90, base_prob + opp_boost)

    # Role trajectory modifier: strong ascending trajectory reduces bust risk.
    traj = float(role_trajectory_score or 0.0)
    if traj >= 70:
        traj_boost = min(0.05, (traj - 70) / 30 * 0.06)
        base_prob = min(0.90, base_prob + traj_boost)

    return round(min(max(base_prob, 0.01), 0.95), 3)


def calculate_cumulative_ppr(
    position: str,
    projected_usage: dict,
    efficiency_metrics: Optional[dict],
    prev_usage: Optional[dict],
    readiness_score: float,
    age: Optional[float],
) -> float:
    """
    Expected PPR fantasy points over the next 2 seasons.

    Season 1 is the primary projection; Season 2 is discounted by an
    age-based factor (young players develop, veterans decline).
    A readiness discount is applied to players with low readiness scores
    to account for the risk of not capturing projected opportunity.
    """
    season1 = _estimate_season_ppr(position, projected_usage, efficiency_metrics, prev_usage)

    # Readiness discount: low readiness means the player may not capitalise.
    # Floor raised to 0.75 — even low-readiness candidates have plausible usage
    # and the 0.60 floor caused systematic PPG underprediction in backtests.
    readiness_factor = 0.75 + (readiness_score / 100.0) * 0.25  # range [0.75, 1.00]
    season1 *= readiness_factor

    age_f = float(age or 24)
    if age_f < 23:
        year2_factor = 1.15   # rookie / sophomore — development upside
    elif age_f < 26:
        year2_factor = 1.05   # prime entry — slight improvement expected
    elif age_f < 29:
        year2_factor = 0.95   # peak years — mild decline
    else:
        year2_factor = 0.78   # late career — meaningful decay

    season2 = season1 * year2_factor
    return round(season1 + season2, 1)


def calculate_peak_ppr(
    cumulative_ppr: float,
    role_trajectory_score: float,
    readiness_score: float,
) -> float:
    """
    Expected peak single-season PPR in the 2-season window.

    Peak ≥ average season, with an upside multiplier driven by role
    trajectory (how much opportunity could grow) and readiness (how
    well the player can convert that opportunity).
    """
    avg_season = cumulative_ppr / 2.0

    # Upside multiplier: high-trajectory + high-readiness players can
    # significantly outperform their average projection in a best-case year.
    trajectory_factor = role_trajectory_score / 100.0
    readiness_factor  = readiness_score / 100.0
    upside = 1.0 + trajectory_factor * readiness_factor * 0.45

    return round(avg_season * upside, 1)


def compute_multitask_predictions(
    position: str,
    breakout_score: float,
    readiness_score: float,
    confidence_score: float,
    role_trajectory_score: float,
    projected_usage: dict,
    efficiency_metrics: Optional[dict],
    prev_usage: Optional[dict],
    age: Optional[float],
    competition_threat: float = 0.0,
    opportunity_score: float = 0.0,
) -> dict:
    """
    Compute all three multitask predictions in one call.

    Returns dict with keys: hit_probability, cumulative_ppr, peak_ppr.
    """
    hit_prob = calculate_hit_probability(
        breakout_score, readiness_score, confidence_score, position,
        opportunity_score=opportunity_score,
        role_trajectory_score=role_trajectory_score,
    )
    cum_ppr = calculate_cumulative_ppr(
        position, projected_usage, efficiency_metrics, prev_usage, readiness_score, age
    )
    peak = calculate_peak_ppr(cum_ppr, role_trajectory_score, readiness_score)

    # Recover season-1 PPR from cumulative: cumulative = s1 * (1 + year2_factor)
    age_f = float(age or 24)
    if age_f < 23:
        _y2 = 1.15
    elif age_f < 26:
        _y2 = 1.05
    elif age_f < 29:
        _y2 = 0.95
    else:
        _y2 = 0.78
    season1_ppr = round(cum_ppr / (1.0 + _y2), 1)

    # Floor: a breakout candidate already has an established role — their
    # baseline projection should be at least their prior season rate.
    # Skipped when genuine competition is present (≥0.38 threat) since the
    # player may legitimately lose their starting role.
    # Requires ≥10 games of prior data for a reliable sample.
    if prev_usage and competition_threat < 0.38:
        prior_ppg   = float(prev_usage.get("ppr_ppg") or 0)
        prior_games = int(prev_usage.get("games") or 0)
        if prior_ppg > 0 and prior_games >= 10:
            floor_season_ppr = prior_ppg * 17
            if season1_ppr < floor_season_ppr:
                ratio       = floor_season_ppr / max(season1_ppr, 1.0)
                cum_ppr     = round(cum_ppr * ratio, 1)
                peak        = round(calculate_peak_ppr(cum_ppr, role_trajectory_score, readiness_score), 1)
                season1_ppr = round(floor_season_ppr, 1)

    return {
        "hit_probability": hit_prob,
        "cumulative_ppr":  cum_ppr,
        "season1_ppr":     season1_ppr,
        "peak_ppr":        peak,
    }
