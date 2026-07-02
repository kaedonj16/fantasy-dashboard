"""Pure waiver-pickup scoring.

Extracted from app.py so the composite score can be unit-tested without the
pandas/DB stack.

``waiver_pickup_score`` ranks free-agent pickups by combining raw model value
with three bonuses: recent 7-day trend, breakout score, and an age curve that
peaks at each position's prime and decays past it.
"""
from __future__ import annotations

# Age past which each position starts losing the age bonus.
WAIVER_PRIME_MAX = {"QB": 33, "RB": 26, "WR": 28, "TE": 29}


def waiver_pickup_score(c: dict, waiver_breakout: dict, prime_max: dict = WAIVER_PRIME_MAX) -> float:
    """Composite waiver-pickup score: value + trend + breakout + age bonuses.

    Extracted from two identical request-scoped closures; the caller passes the
    per-request ``waiver_breakout`` map so behavior is unchanged.
    """
    val = c["value"]
    age = c["age"] or 0
    pos = c["position"]
    rank_chg = c["rank_change_7d"] or 0
    bscore = waiver_breakout.get(c["player_id"], 0)
    prime = prime_max.get(pos, 28)

    # Trend bonus: up to +60 for strong 7d movement
    trend_bonus = min(rank_chg * 4, 60) if rank_chg and rank_chg > 0 else 0
    # Breakout bonus: up to +50
    breakout_bonus = min(bscore * 0.5, 50)
    # Age bonus: peak age = +30, every year past prime = -10
    age_bonus = 30 - max(0, (age - prime) * 10) if age else 0

    return val + trend_bonus + breakout_bonus + age_bonus
