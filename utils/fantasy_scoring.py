"""Pure fantasy-points scoring from a Sleeper stats line.

Extracted from app.py so the points calculation can be unit-tested without the
pandas/DB stack. Uses Sleeper's stat/scoring key names; every scoring value
falls back to a standard PPR-ish default when the league omits it.
"""
from __future__ import annotations


_DEFAULT_RATES = {
    "pass_yd": 0.04, "pass_td": 4.0, "pass_int": -2.0,
    "rush_yd": 0.1, "rush_td": 6.0, "rec": 0.0,
    "rec_yd": 0.1, "rec_td": 6.0, "fum_lost": -2.0,
}


def _rate(settings: dict, key: str) -> float:
    """Respect explicit zero scoring; only use defaults when a key is absent."""
    value = settings[key] if key in settings else _DEFAULT_RATES.get(key, 0.0)
    try:
        return float(value or 0)
    except (TypeError, ValueError):
        return 0.0


def score_stats(s: dict, ss: dict, pos: str = "") -> float:
    """Compute points from a projected Sleeper stat line and league settings.

    In addition to the common defaults, every projected stat with an exact
    scoring-settings key is included (first downs, two-point conversions,
    returns, completions, sacks, etc.). Milestone bonuses are applied once.
    """
    s = s or {}
    ss = ss or {}
    p = 0.0
    handled = set(_DEFAULT_RATES)
    for key in handled:
        p += float(s.get(key) or 0) * _rate(ss, key)
    # Sleeper uses matching names for most custom stat/rate pairs.
    for key, value in s.items():
        if key in handled or key.startswith("bonus_") or key not in ss:
            continue
        try:
            p += float(value or 0) * float(ss.get(key) or 0)
        except (TypeError, ValueError):
            continue
    if str(pos).upper() == "TE":
        p += float(s.get("rec") or 0) * _rate(ss, "bonus_rec_te")
    py = s.get("pass_yd") or 0
    ry = s.get("rush_yd") or 0
    ey = s.get("rec_yd") or 0
    rr = ry + ey
    if py >= 400: p += (ss.get("bonus_pass_yd_400") or 0)
    elif py >= 300: p += (ss.get("bonus_pass_yd_300") or 0)
    if ry >= 200: p += (ss.get("bonus_rush_yd_200") or 0)
    elif ry >= 100: p += (ss.get("bonus_rush_yd_100") or 0)
    if ey >= 200: p += (ss.get("bonus_rec_yd_200") or 0)
    elif ey >= 100: p += (ss.get("bonus_rec_yd_100") or 0)
    if rr >= 200: p += (ss.get("bonus_rush_rec_yd_200") or 0)
    elif rr >= 100: p += (ss.get("bonus_rush_rec_yd_100") or 0)
    return p


def projection_points(entry: dict, scoring_settings: dict, pos: str = "") -> float:
    """Select exact league scoring from a cached multi-variant projection."""
    if isinstance(entry, (int, float)):
        return float(entry)
    if not isinstance(entry, dict):
        return 0.0
    raw = entry.get("raw_stats")
    if isinstance(raw, dict) and scoring_settings:
        return round(score_stats(raw, scoring_settings, pos), 2)
    from utils.proj_variant import pick_proj_variant
    variant = pick_proj_variant(scoring_settings or {})
    return float(entry.get(variant) or entry.get("ppr") or 0.0)
