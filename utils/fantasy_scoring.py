"""Pure fantasy-points scoring from a Sleeper stats line.

Extracted from app.py so the points calculation can be unit-tested without the
pandas/DB stack. Uses Sleeper's stat/scoring key names; every scoring value
falls back to a standard PPR-ish default when the league omits it.
"""
from __future__ import annotations


def score_stats(s: dict, ss: dict) -> float:
    """Compute fantasy points from a Sleeper stats dict using Sleeper key names."""
    s = s or {}
    ss = ss or {}
    p = 0.0
    p += (s.get("pass_yd") or 0) * (ss.get("pass_yd") or 0.04)
    p += (s.get("pass_td") or 0) * (ss.get("pass_td") or 4.0)
    p += (s.get("pass_int") or 0) * (ss.get("pass_int") or -2.0)
    p += (s.get("rush_yd") or 0) * (ss.get("rush_yd") or 0.1)
    p += (s.get("rush_td") or 0) * (ss.get("rush_td") or 6.0)
    p += (s.get("rec") or 0) * (ss.get("rec") or 0)
    p += (s.get("rec_yd") or 0) * (ss.get("rec_yd") or 0.1)
    p += (s.get("rec_td") or 0) * (ss.get("rec_td") or 6.0)
    p += (s.get("fum_lost") or 0) * (ss.get("fum_lost") or -2.0)
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
