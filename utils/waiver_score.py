"""Pure waiver-pickup scoring and signal classification.

Extracted from app.py so the ranking model can be unit-tested without the
pandas/DB stack, and shared by both waiver surfaces (the /api/waiver-candidates
endpoint and the offseason dashboard card) so they rank and label identically.

Design goals for a *waiver target* list (vs. a plain dynasty-value list):

  * Value informs the ranking but must not dominate it. A saturating curve
    compresses the gap between a 1500-value veteran and a 250-value breakout so
    that opportunity signals can lift an emerging player above a static one.
  * Recent role growth (usage spikes: snaps / touches / targets rising over the
    last few weeks) is the single strongest "add him now" signal, so it feeds
    the score directly — not just the badge.
  * Age is a smooth curve: ascending-young players are rewarded progressively
    and past-prime players decay (bounded), rather than a hard cliff at prime.
"""
from __future__ import annotations

# Age past which each position starts losing the age bonus (peak dynasty window).
WAIVER_PRIME_MAX = {"QB": 33, "RB": 26, "WR": 28, "TE": 29}

# Minimum last-3-week-vs-season rise, per stat, to count as a usage spike. A
# candidate whose delta hits its stat's threshold has a usage ratio of 1.0.
USAGE_SPIKE_MIN = {"snap_pct": 8.0, "touches": 3.0, "targets": 2.0}


def value_component(val) -> float:
    """Saturating value contribution (0 .. ~120).

    120 * v / (v + 500): concave, so value stays monotonic but its gaps compress
    (e.g. 100->20, 300->45, 500->60, 800->74, 1500->90). This keeps a high-value
    free agent attractive without letting static value bury emerging players.
    """
    try:
        v = max(0.0, float(val or 0))
    except (TypeError, ValueError):
        return 0.0
    return 120.0 * v / (v + 500.0)


def usage_ratio(stat, delta) -> float:
    """Usage-spike magnitude as a multiple of the stat's spike threshold.

    Returns 0.0 when there is no usage data. A player exactly at the threshold
    scores 1.0; twice the threshold scores 2.0.
    """
    if not stat or delta is None:
        return 0.0
    thr = USAGE_SPIKE_MIN.get(stat, 3.0)
    if thr <= 0:
        return 0.0
    try:
        return max(0.0, float(delta) / thr)
    except (TypeError, ValueError):
        return 0.0


def waiver_pickup_score(c: dict, waiver_breakout: dict,
                        prime_max: dict = WAIVER_PRIME_MAX) -> float:
    """Composite waiver-pickup score: value + usage + trend + breakout + age.

    ``c`` is a candidate dict with keys: value, age, position, rank_change_7d,
    player_id, and optionally usage_stat / usage_delta (weekly usage trend). The
    breakout score is looked up from ``waiver_breakout`` by player_id.
    """
    try:
        val = float(c.get("value") or 0)
    except (TypeError, ValueError):
        val = 0.0
    age = c.get("age") or 0
    pos = c.get("position")
    rank_chg = c.get("rank_change_7d") or 0
    bscore = waiver_breakout.get(c.get("player_id"), 0) or 0
    prime = prime_max.get(pos, 28)

    # Value: saturating base so it informs but doesn't dominate.
    value_pts = value_component(val)

    # Usage spike: recent role growth, the strongest "add now" signal. Hitting a
    # stat's spike threshold is +30; ~1.7x threshold caps at +50.
    usage_pts = min(usage_ratio(c.get("usage_stat"), c.get("usage_delta")) * 30.0, 50.0)

    # Weekly rank trend: reward risers (+3.5/spot, cap +45); mildly penalize
    # players falling out of relevance (-1.5/spot, floor -15).
    if rank_chg > 0:
        trend_pts = min(rank_chg * 3.5, 45.0)
    else:
        trend_pts = max(rank_chg * 1.5, -15.0)

    # Breakout opportunity model score: up to +45.
    breakout_pts = min(bscore * 0.5, 45.0)

    # Age: smooth youth reward / past-prime decay, both bounded.
    if not age:
        age_pts = 0.0
    else:
        gap = prime - age  # + = younger than prime
        if gap >= 0:
            age_pts = min(22.0 + gap * 2.0, 36.0)
        else:
            age_pts = max(22.0 + gap * 7.0, -22.0)

    return value_pts + usage_pts + trend_pts + breakout_pts + age_pts


def waiver_signal(c: dict, waiver_breakout: dict,
                  prime_max: dict = WAIVER_PRIME_MAX) -> "tuple[str, str]":
    """Return (badge_class, label) describing why a candidate is interesting.

    Shared by both waiver surfaces. The usage-spike branch is a no-op for
    candidates without usage data (e.g. the offseason card), so those simply
    fall through to the breakout/trend/value/age labels.
    """
    rank_chg = c.get("rank_change_7d") or 0
    age = c.get("age") or 0
    pos = c.get("position")
    bscore = waiver_breakout.get(c.get("player_id"), 0) or 0
    prime = prime_max.get(pos, 28)
    try:
        val = float(c.get("value") or 0)
    except (TypeError, ValueError):
        val = 0.0

    if usage_ratio(c.get("usage_stat"), c.get("usage_delta")) >= 1.0:
        return ("signal-usage", "Usage Spike")
    if bscore >= 55:
        return ("signal-breakout", "Breakout")
    if rank_chg >= 8:
        return ("signal-rising", "Rising Fast")
    if rank_chg >= 3:
        return ("signal-rising", "Trending Up")
    if age and age < prime - 2 and val >= 300:
        return ("signal-value", "Value Play")
    if age and age > prime + 2:
        return ("signal-aging", "Sell Window")
    return ("signal-hold", "Available")
