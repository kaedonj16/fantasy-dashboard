"""Single start/sit ranking formula: projection times six capped multipliers.

The START badges, optimal-lineup banner, and Compare card all rank on this
score so they cannot contradict each other. Kept as a pure function so the
math is unit-tested without Flask.
"""
from __future__ import annotations

from typing import Optional


def compute_start_score(
    proj_pts: float,
    *,
    on_bye: bool = False,
    recent_ppg: float = 0.0,
    season_ppg: float = 0.0,
    def_rank: Optional[float] = None,
    def_total: Optional[int] = None,
    usage_delta: Optional[float] = None,
    usage_season_avg: Optional[float] = None,
    injury_status: Optional[str] = None,
    implied_total: Optional[float] = None,
    bust_rate: Optional[float] = None,
) -> tuple[float, dict, Optional[str]]:
    """Return ``(score, score_factors, demotion)``.

    Factors: proj, form, matchup, usage, avail, vegas, floor. Bye and OUT/IR
    zero the score. Every non-projection signal is a capped multiplier.
    """
    form = mu = usage = avail = vegas = floor = 1.0
    demotion = None
    try:
        proj = float(proj_pts or 0)
    except (TypeError, ValueError):
        proj = 0.0
    if on_bye:
        return 0.0, {
            "proj": proj, "form": 1.0, "matchup": 1.0, "usage": 1.0,
            "avail": 1.0, "vegas": 1.0, "floor": 1.0,
        }, "bye"

    try:
        recent = float(recent_ppg or 0)
    except (TypeError, ValueError):
        recent = 0.0
    try:
        season = float(season_ppg or 0)
    except (TypeError, ValueError):
        season = 0.0
    if recent > 0 and season > 0:
        form = min(1.10, max(0.90, recent / season))

    if def_rank and def_total and def_total > 1:
        ease = (float(def_total) - float(def_rank)) / (float(def_total) - 1)
        mu = 0.90 + ease * 0.20

    if usage_delta is not None and usage_season_avg:
        rel = float(usage_delta) / max(float(usage_season_avg), 1.0)
        usage = min(1.05, max(0.95, 1.0 + rel * 0.25))

    status = (injury_status or "").upper()
    if any(k in status for k in ("OUT", "IR", "SUSP", "DOUBT", "PUP", "DNP")):
        avail = 0.0
        demotion = "out"
    elif "QUESTION" in status or status in ("GTD", "Q"):
        avail = 0.85
        demotion = "questionable"

    if implied_total is not None:
        try:
            imp = float(implied_total)
        except (TypeError, ValueError):
            imp = None
        if imp is not None:
            if imp <= 17:
                vegas = 0.94
                demotion = demotion or "low_total"
            elif imp >= 27:
                vegas = 1.04

    if bust_rate is not None:
        try:
            floor = min(1.10, max(0.90, 1.0 + (0.5 - float(bust_rate)) * 0.4))
        except (TypeError, ValueError):
            floor = 1.0

    score = proj * form * mu * usage * avail * vegas * floor
    return score, {
        "proj": proj,
        "form": round(form, 3),
        "matchup": round(mu, 3),
        "usage": round(usage, 3),
        "avail": round(avail, 3),
        "vegas": round(vegas, 3),
        "floor": round(floor, 3),
    }, demotion
