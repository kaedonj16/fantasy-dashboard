"""Pure asset-tier classification and multi-for-one trade adjustment.

Extracted from app.py so this logic can be unit-tested without the pandas/DB
stack.

``asset_tier`` maps a fantasy value to a tier number (T1 elite ... TN) given a
set of thresholds. ``apply_tier_stack_adjustment`` applies the depth penalty for
the side of an unequal trade that gives up more players: the extra players are
discounted toward true waiver-wire value, mutating the passed-in side dicts in
place with tier/effective-value annotations.
"""
from __future__ import annotations

from utils.tier_thresholds import FALLBACK_THRESHOLDS

# Number of tiers the classifier caps out at (T1 elite ... T9 catch-all).
NUM_TIERS = 9


def build_tier_caps(num_tiers: int) -> dict:
    """Per-tier value-retention caps, linearly interpolated from 1.0 (T1)
    down to 0.38 (bottom tier). Returns {tier: cap}."""
    high, low = 1.0, 0.38
    if num_tiers <= 1:
        return {1: 1.0}
    return {t: round(high - (high - low) * (t - 1) / (num_tiers - 1), 3)
            for t in range(1, num_tiers + 1)}


def asset_tier(value: float, thresholds: list = None) -> int:
    t = thresholds if thresholds is not None else FALLBACK_THRESHOLDS
    for i, threshold in enumerate(t):
        if value >= threshold:
            return min(i + 1, NUM_TIERS)
    return NUM_TIERS  # catch-all T9


def apply_tier_stack_adjustment(side_a: dict, side_b: dict,
                                 tier_thresholds: list = None,
                                 is_sf: bool = False,
                                 value_table: list = None,
                                 league_size: int = 10) -> None:
    """
    Depth penalty for the bigger side in unequal trades.

    The side giving up more players is discounted by a fraction of the value of
    true waiver-wire players (ranked well below the roster cutoff).  The first
    reference player sits at roughly rank (league_size × 38) and each successive
    extra player steps deeper; 50% of that value is applied as the penalty.

    Bigger side: effective_total = raw_total - sum(bench_values).
    Smaller side: effective_total = raw_total (no adjustment).
    """
    a_count = len(side_a.get("breakdown") or []) or len(side_a.get("player_values") or [])
    b_count = len(side_b.get("breakdown") or []) or len(side_b.get("player_values") or [])

    thresholds = tier_thresholds if tier_thresholds is not None else FALLBACK_THRESHOLDS

    # Annotate tiers first (informational, no multiplier applied)
    for side in (side_a, side_b):
        for item in (side.get("breakdown") or []):
            val = item.get("value", 0.0)
            item["tier"]            = asset_tier(val, thresholds)
            item["stack_mult"]      = 1.0
            item["effective_value"] = round(val, 1)

    if a_count == b_count:
        return

    delta   = abs(a_count - b_count)
    smaller = side_a if a_count < b_count else side_b
    bigger  = side_b if a_count < b_count else side_a

    # Build sorted value list for bench-rank lookup
    sorted_vals: list = []
    if value_table:
        sorted_vals = sorted(
            [float(p.get("value") or 0) for p in value_table
             if isinstance(p, dict) and float(p.get("value") or 0) > 10],
            reverse=True,
        )

    # Rank well below the roster cutoff so the penalty reflects true waiver-wire value
    base_rank      = league_size * 38   # ~380 for 10-team (below 27-spot roster cutoff)
    _BENCH_BASE    = 80.0               # fallback when value_table unavailable
    _BENCH_STEP    = -5.0               # each extra player is worth slightly less
    _PENALTY_FRAC  = 0.5                # apply 50% of waiver-wire value as the penalty

    def _bench_value(i: int) -> float:
        rank = min(len(sorted_vals) if sorted_vals else 9999, base_rank + i * 10)
        if sorted_vals:
            idx = min(rank - 1, len(sorted_vals) - 1)
            return sorted_vals[idx] * _PENALTY_FRAC
        return max(20.0, (_BENCH_BASE + i * _BENCH_STEP) * _PENALTY_FRAC)

    bench_total = sum(_bench_value(i) for i in range(delta))

    bigger["effective_total"]  = float(bigger.get("raw_total") or 0.0) - bench_total
    bigger["adjustment"]       = -bench_total
    smaller["effective_total"] = float(smaller.get("raw_total") or 0.0)
    smaller["adjustment"]      = 0.0
