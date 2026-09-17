"""Authoritative opponent-adjusted defense-vs-position calculations.

All Schedule Assistant consumers use the multiplier produced here.  A value of
1.10 means a defense increased its opponents' pre-game expectation by 10%; a
value of .90 means it suppressed it by 10%.  Functions are deliberately pure
so rebuilds and corrected-stat replays are deterministic and idempotent.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict

BASELINE_WEIGHTS = {"recent": .50, "earlier": .30, "prior": .20}
MIN_PARTICIPATION = {"QB": {"attempts": 8, "snaps": 15},
                     "RB": {"touches": 3, "snaps": 8},
                     "WR": {"targets": 2, "routes": 5, "snaps": 8},
                     "TE": {"targets": 1, "routes": 5, "snaps": 8}}
WINSOR_MULTIPLIER = (.50, 1.50)
EARLY_SEASON_CURRENT_WEIGHTS = {0: 0., 1: .25, 2: .40, 3: .55, 4: .70, 5: .85}


def season_weights(completed_through_week: int, *, blend: bool = True):
    if not blend:
        return 0., 1.
    current = EARLY_SEASON_CURRENT_WEIGHTS.get(max(0, int(completed_through_week)), 1.)
    return 1. - current, current


def scoring_profile_hash(settings):
    canonical = json.dumps(settings or {}, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


def rating_cache_key(season, completed_week, settings, position):
    return f"{int(season)}:{int(completed_week)}:{scoring_profile_hash(settings)}:{position.upper()}"


def blend_value(previous, current, completed_week, *, blend=True):
    if previous is None:
        return (current, "current") if current is not None else (None, "unavailable")
    if current is None:
        return previous, "prior"
    pw, cw = season_weights(completed_week, blend=blend)
    return pw * float(previous) + cw * float(current), "blended" if pw else "current"


def rank_values(values):
    ordered = sorted(((t, v) for t, v in values.items() if v is not None),
                     key=lambda x: (-float(x[1]), x[0]))
    return {t: i + 1 for i, (t, _) in enumerate(ordered)}, len(ordered)


def normalize_schedule(values):
    """Return 0..100 scores from full-precision schedule multipliers."""
    usable = [float(v) for v in values.values() if v is not None]
    if not usable:
        return {}
    lo, hi = min(usable), max(usable)
    if math.isclose(lo, hi):
        return {k: 50. for k, v in values.items() if v is not None}
    return {k: 100. * (float(v) - lo) / (hi - lo) for k, v in values.items() if v is not None}


def rank_team_schedules(team_opponents, values):
    averages = {team: sum(vs) / len(vs) for team, opponents in team_opponents.items()
                if (vs := [float(values[o]) for o in opponents if o and values.get(o) is not None])}
    ranks, total = rank_values(averages)
    return {t: (ranks[t], total, round(v, 4)) for t, v in averages.items()}


def meaningful_participation(row):
    """Conservative OR thresholds; unavailable opportunity falls back to points."""
    pos = str(row.get("position") or row.get("pos") or "").upper()
    observed = False
    for field, minimum in MIN_PARTICIPATION.get(pos, {}).items():
        value = row.get(field)
        if value is not None:
            observed = True
            if float(value or 0) >= minimum:
                return True
    return (not observed) and float(row.get("fantasy_points") or row.get("pts") or 0) > 0


def pregame_baseline(history, position_baseline, weights=None):
    """Leak-free stabilized expectation from records strictly before a game.

    ``history`` is already chronological and contains only qualifying games.
    The prior/replacement component is never zero; uncertainty reports how much
    of the estimate rests on fewer than four current-season games.
    """
    weights = weights or BASELINE_WEIGHTS
    current = [r for r in history if r.get("is_current_season", True)]
    prior = [r for r in history if not r.get("is_current_season", True)]
    recent, earlier = current[-4:], current[:-4]
    replacement = max(float(position_baseline or 0), .1)
    components = {
        "recent": sum(float(r["fantasy_points"]) for r in recent) / len(recent) if recent else None,
        "earlier": sum(float(r["fantasy_points"]) for r in earlier) / len(earlier) if earlier else None,
        "prior": sum(float(r["fantasy_points"]) for r in prior[-8:]) / len(prior[-8:]) if prior else replacement,
    }
    # Shift missing component weight to the role/position replacement rather
    # than silently returning zero. Current data progressively displaces prior.
    n = len(current)
    prior_scale = max(0., 1. - n / 8.)
    configured = dict(weights)
    configured["prior"] *= prior_scale
    numerator = denominator = 0.
    for key, weight in configured.items():
        value = components[key] if components[key] is not None else replacement
        numerator += weight * value
        denominator += weight
    expected = numerator / denominator if denominator else replacement
    reliability = min(1., n / 6.)
    return expected, reliability


def game_adjustment(actual, expected):
    expected = float(expected)
    if expected <= 0:
        return None
    multiplier = float(actual) / expected
    return {"actual": float(actual), "expected": expected,
            "points_over_expected": float(actual) - expected,
            "multiplier": multiplier, "adjusted_percent": (multiplier - 1.) * 100.}


def aggregate_defense_games(games, prior_multiplier=1., prior_weight=4.):
    """Opportunity weighted, winsorized and shrunk positional defense result."""
    valid = [g for g in games if float(g.get("expected") or 0) > 0]
    if not valid:
        return None
    weighted = weight_sum = opps = 0.
    for index, g in enumerate(valid):
        raw = float(g["actual"]) / float(g["expected"])
        clipped = min(WINSOR_MULTIPLIER[1], max(WINSOR_MULTIPLIER[0], raw))
        reliability = float(g.get("reliability", 1.))
        opportunity = max(1., float(g.get("opportunities") or g["expected"]))
        recency = .9 + .1 * (index + 1) / len(valid)
        weight = float(g["expected"]) * reliability * recency
        weighted += clipped * weight
        weight_sum += weight
        opps += opportunity
    observed = weighted / weight_sum
    shrunk = (weight_sum * observed + prior_weight * float(prior_multiplier)) / (weight_sum + prior_weight)
    actual = sum(float(g["actual"]) for g in valid)
    expected = sum(float(g["expected"]) for g in valid)
    confidence = "high" if len(valid) >= 8 and weight_sum >= 80 else "medium" if len(valid) >= 4 else "low"
    return {"raw_allowed_per_game": actual / len(valid),
            "expected_opponent_points": expected / len(valid),
            "adjusted_points_over_expected": (actual - expected) / len(valid),
            "observed_multiplier": observed, "adjusted_multiplier": shrunk,
            "adjusted_percent": (shrunk - 1.) * 100., "sample_size": len(valid),
            "opportunity_count": opps, "reliable_sample_weight": weight_sum,
            "prior_weight": prior_weight, "confidence": confidence}
