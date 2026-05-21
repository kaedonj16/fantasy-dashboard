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
# Calibrated from 2022-2023 backtests: the model systematically underpredicts
# by ~20% (mean predicted 4.7 ppg vs actual 5.8 ppg for the candidate pool).
# Root cause: projected_usage reconstructed from component_details gives a
# conservative baseline; actual breakout players typically beat the baseline by
# more than their opportunity share alone would suggest.
_SEASON_PPR_UPLIFT: dict[str, float] = {
    "QB": 1.10,
    "RB": 1.20,
    "WR": 1.22,
    "TE": 1.18,
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
) -> float:
    """
    P(player finishes top-12 at position next season).

    Uses the breakout score as the primary signal (0-100 scale → 0-1),
    weighted with readiness and confidence as modifiers, then calibrated
    to per-position base rates via a power-law transform.
    """
    pos = (position or "WR").upper()
    base = _BASE_HIT_RATE.get(pos, 0.18)

    # Composite signal: weighted blend of three most predictive components
    raw_signal = (
        (breakout_score / 100.0) * 0.55
        + (readiness_score / 100.0) * 0.30
        + (confidence_score / 100.0) * 0.15
    )

    # Power-law scaling relative to the "neutral" breakout level (score=50)
    # At signal=0.50 → multiplier=1.0 (stays at base rate)
    # At signal=1.00 → multiplier≈2.8 (strong breakout)
    # At signal=0.25 → multiplier≈0.42 (weak signal)
    neutral = 0.50
    if raw_signal >= neutral:
        multiplier = (raw_signal / neutral) ** 1.5
    else:
        multiplier = (raw_signal / neutral) ** 0.7

    prob = base * multiplier
    return round(min(max(prob, 0.01), 0.95), 3)


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
) -> dict:
    """
    Compute all three multitask predictions in one call.

    Returns dict with keys: hit_probability, cumulative_ppr, peak_ppr.
    """
    hit_prob = calculate_hit_probability(
        breakout_score, readiness_score, confidence_score, position
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

    return {
        "hit_probability": hit_prob,
        "cumulative_ppr":  cum_ppr,
        "season1_ppr":     season1_ppr,
        "peak_ppr":        peak,
    }
