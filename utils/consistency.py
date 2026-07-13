"""Weekly consistency / boom-bust profiles from a player's game-by-game scores.

Season averages hide *how* a player gets there: a steady 14-a-week WR and a
6-or-26 lottery ticket can share a PPG. This module turns a list of weekly
fantasy points into the shape managers actually decide on - a floor, a ceiling,
a 0-100 consistency score, and boom/bust rates against position-aware thresholds.

Pure and dependency-free (no NumPy): the app-specific part is loading the weekly
scores; the math here is fully unit-tested.
"""
from __future__ import annotations

import math
from typing import Optional

# Per-position weekly thresholds (PPR points): at/above `boom` is a smash week,
# below `bust` is a week that likely lost you the matchup. Tuned to typical
# startable-week distributions.
_POS_THRESHOLDS: dict[str, tuple[float, float]] = {
    "QB": (25.0, 15.0),
    "RB": (20.0, 8.0),
    "WR": (20.0, 8.0),
    "TE": (15.0, 5.0),
    "K":  (12.0, 5.0),
    "DEF": (12.0, 3.0),
}
_DEFAULT_THRESHOLDS = (18.0, 7.0)

_MIN_GAMES = 3   # below this the profile is flagged small-sample

# Weeks of current-season data at which the profile is fully "molded" to this
# season and stops leaning on last year (a full crossfade by mid-season).
BLEND_FULL_SEASON = 8


def _label_for(consistency: int, boom_rate: float, bust_rate: float, small_sample: bool) -> str:
    """Map a consistency score + boom/bust rates to a one-word profile label."""
    if small_sample:
        return "Small sample"
    if boom_rate >= 0.30 and bust_rate >= 0.30:
        return "Boom or bust"
    if consistency >= 62:
        return "Steady"
    if consistency >= 40:
        return "Balanced"
    return "Volatile"


def _percentile(sorted_vals: "list[float]", p: float) -> Optional[float]:
    """Linear-interpolated percentile (p in 0..100) of a pre-sorted list."""
    n = len(sorted_vals)
    if n == 0:
        return None
    if n == 1:
        return sorted_vals[0]
    k = (n - 1) * (p / 100.0)
    lo = math.floor(k)
    hi = math.ceil(k)
    if lo == hi:
        return sorted_vals[int(k)]
    return sorted_vals[lo] * (hi - k) + sorted_vals[hi] * (k - lo)


def consistency_profile(weekly_points: "list[float]", position: str = "") -> Optional[dict]:
    """Boom-bust profile from a player's weekly fantasy scores.

    ``weekly_points`` should already exclude byes / games not played (None). Games
    the player was active but scored ~0 count - they're real bad weeks.

    Returns None if there are no games. Otherwise a dict with ``games``, ``mean``,
    ``floor`` (20th pct), ``ceiling`` (80th pct), ``std``, ``cv``,
    ``consistency`` (0-100, higher = steadier), ``boom_rate``, ``bust_rate``,
    ``label``, and ``small_sample``.
    """
    pts = [float(p) for p in weekly_points if p is not None]
    n = len(pts)
    if n == 0:
        return None

    mean = sum(pts) / n
    var = sum((p - mean) ** 2 for p in pts) / n
    std = math.sqrt(var)
    cv = (std / mean) if mean > 0 else 0.0

    ordered = sorted(pts)
    floor = _percentile(ordered, 20)
    ceiling = _percentile(ordered, 80)

    boom_t, bust_t = _POS_THRESHOLDS.get(str(position).upper(), _DEFAULT_THRESHOLDS)
    boom_rate = sum(1 for p in pts if p >= boom_t) / n
    bust_rate = sum(1 for p in pts if p < bust_t) / n

    # Consistency: coefficient of variation inverted onto 0-100. A CV of ~0.3 is
    # very steady (~70), ~0.6 is streaky (~40), >=1.0 is a pure lottery ticket.
    consistency = round(100 * max(0.0, min(1.0, 1.0 - cv)))

    small_sample = n < _MIN_GAMES
    label = _label_for(consistency, boom_rate, bust_rate, small_sample)

    return {
        "games": n,
        "mean": round(mean, 1),
        "floor": round(floor, 1) if floor is not None else None,
        "ceiling": round(ceiling, 1) if ceiling is not None else None,
        "std": round(std, 1),
        "cv": round(cv, 2),
        "consistency": consistency,
        "boom_rate": round(boom_rate, 2),
        "bust_rate": round(bust_rate, 2),
        "label": label,
        "small_sample": small_sample,
    }


def blended_consistency_profile(
    current: "list[float]",
    prior: "list[float]",
    position: str = "",
    *,
    prior_season: Optional[int] = None,
    full_season: int = BLEND_FULL_SEASON,
) -> Optional[dict]:
    """Consistency profile that starts from last season and molds to this one.

    With no current-season games it returns last season's profile (tagged with
    its year). As current games accumulate, each metric crossfades linearly from
    prior toward current, weighting current by ``min(1, games / full_season)`` -
    so by ~mid-season the profile is entirely the current year. With no prior
    data it's just the current-season profile.
    """
    cur = consistency_profile(current, position)
    pri = consistency_profile(prior, position)

    if cur is None:
        if pri is None:
            return None
        out = dict(pri)
        out["season"] = prior_season   # pure last-season profile; tag the year
        out["blended"] = False
        return out
    if pri is None:
        return cur  # no history to lean on

    n_cur = cur["games"]
    w = max(0.0, min(1.0, n_cur / float(full_season)))
    if w >= 1.0:
        return cur  # fully molded to the current season

    def bl(key: str) -> float:
        return w * cur[key] + (1.0 - w) * pri[key]

    consistency = round(bl("consistency"))
    boom_rate = round(bl("boom_rate"), 2)
    bust_rate = round(bl("bust_rate"), 2)
    return {
        "games": n_cur,
        "prior_games": pri["games"],
        "mean": round(bl("mean"), 1),
        "floor": round(bl("floor"), 1),
        "ceiling": round(bl("ceiling"), 1),
        "std": round(bl("std"), 1),
        "cv": round(bl("cv"), 2),
        "consistency": consistency,
        "boom_rate": boom_rate,
        "bust_rate": bust_rate,
        "label": _label_for(consistency, boom_rate, bust_rate, False),
        "small_sample": False,
        "blended": True,
        "prior_season": prior_season,
        "weight_current": round(w, 2),
    }
