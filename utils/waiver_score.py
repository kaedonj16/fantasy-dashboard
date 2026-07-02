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

# NFL injury/roster statuses that vacate opportunity for the players behind them
# on the depth chart, weighted by how likely the injured player is to miss time
# (and thus how much of the role opens up). Even a QUESTIONABLE tag ahead is a
# real bump — the backup is one setback from the job — so it scores, just less
# than a confirmed absence.
VACANCY_SEVERITY = {
    "IR": 1.0, "PUP": 1.0, "NFI": 1.0, "SUSP": 0.9, "OUT": 0.85,
    "DOUBTFUL": 0.5, "QUESTIONABLE": 0.3,
}

# Severity at/above which the vacancy is treated as a genuine "role is open now"
# (confirmed/likely absence -> "Next Man Up"); below it is a softer bump.
VACANCY_STRONG = 0.5


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


def build_depth_index(full_players: dict) -> dict:
    """Group a Sleeper players map by (team, position) for depth-chart lookups.

    Returns ``{(TEAM, POS): [{"pid", "depth_order", "status"}, ...]}`` where
    ``status`` is the player's injury_status (falling back to roster status).
    """
    idx: dict = {}
    for pid, p in (full_players or {}).items():
        if not isinstance(p, dict):
            continue
        team = str(p.get("team") or "").upper()
        pos = str(p.get("position") or "").upper()
        if not team or not pos:
            continue
        idx.setdefault((team, pos), []).append({
            "pid": str(pid),
            "depth_order": p.get("depth_chart_order"),
            "status": p.get("injury_status") or p.get("status") or "",
        })
    return idx


def injured_ahead(depth_order, teammates) -> list:
    """Vacating injury statuses of same-position teammates ranked AHEAD.

    ``depth_order`` is this candidate's depth_chart_order (1 = starter); a falsy
    value is treated as deep so any injured starter ahead still counts.
    ``teammates`` is an iterable of ``{"depth_order", "status"}`` dicts for the
    same team + position. Only statuses in VACANCY_SEVERITY are returned.
    """
    mine = depth_order or 99
    out = []
    for t in teammates:
        o = t.get("depth_order") or 99
        st = str(t.get("status") or "").upper()
        if o < mine and st in VACANCY_SEVERITY:
            out.append(st)
    return out


def injured_ahead_for_player(pid, full_players: dict, depth_index: dict) -> list:
    """Convenience wrapper: vacating statuses ahead of ``pid`` on its depth chart."""
    fp = full_players or {}
    p = fp.get(pid) or fp.get(str(pid)) or {}
    team = str(p.get("team") or "").upper()
    pos = str(p.get("position") or "").upper()
    if not team or not pos:
        return []
    group = (depth_index or {}).get((team, pos)) or []
    teammates = [g for g in group if g.get("pid") != str(pid)]
    return injured_ahead(p.get("depth_chart_order"), teammates)


def depth_chart_vacancy_score(statuses) -> float:
    """Points for injured players sitting ahead on the depth chart (0 .. 55).

    The most severe vacancy dominates — an injured starter directly ahead is the
    strongest waiver signal there is — and additional injured bodies ahead add a
    little more on top.
    """
    sev = sorted(
        (VACANCY_SEVERITY.get(str(s).upper(), 0.0) for s in (statuses or [])),
        reverse=True,
    )
    sev = [x for x in sev if x > 0]
    if not sev:
        return 0.0
    return min(sev[0] * 40.0 + sum(sev[1:]) * 8.0, 55.0)


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

    # Injury vacancy: injured players ahead on the depth chart free up the role
    # directly (a starter on IR makes the backup a must-add). Up to +55.
    injury_pts = depth_chart_vacancy_score(c.get("injured_ahead"))

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

    return value_pts + injury_pts + usage_pts + trend_pts + breakout_pts + age_pts


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

    inj_sev = max((VACANCY_SEVERITY.get(str(s).upper(), 0.0)
                   for s in (c.get("injured_ahead") or [])), default=0.0)

    # A confirmed/likely absence ahead is the most actionable signal — the role
    # is vacated now — so it outranks even a usage spike.
    if inj_sev >= VACANCY_STRONG:
        return ("signal-injury", "Next Man Up")
    if usage_ratio(c.get("usage_stat"), c.get("usage_delta")) >= 1.0:
        return ("signal-usage", "Usage Spike")
    if bscore >= 55:
        return ("signal-breakout", "Breakout")
    if rank_chg >= 8:
        return ("signal-rising", "Rising Fast")
    if rank_chg >= 3:
        return ("signal-rising", "Trending Up")
    # A softer injury bump (e.g. the starter is Questionable): worth flagging,
    # but weaker than a confirmed out or a real usage/trend signal.
    if inj_sev > 0:
        return ("signal-injury-soft", "Bumped Up")
    if age and age < prime - 2 and val >= 300:
        return ("signal-value", "Value Play")
    if age and age > prime + 2:
        return ("signal-aging", "Sell Window")
    return ("signal-hold", "Available")
