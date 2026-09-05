"""Single start/sit ranking formula: projection times capped multipliers.

The START badges, optimal-lineup banner, and Compare card all rank on this
score so they cannot contradict each other. Kept as a pure function so the
math is unit-tested without Flask.

Weekly projections already bake in the opponent, so defensive matchup rank is
*not* re-multiplied into the score (that used to double-count). Matchup stays on
the row as a chip only. Weather and Vegas *are* applied here because those
signals are usually missing from raw projection feeds.
"""
from __future__ import annotations

from typing import Optional

# Notable weather → position multipliers (only hurts; never boosts).
# Wind hurts pass game / kickers most; RBs are mostly insulated.
_WEATHER_MULT = {
    "wind": {"QB": 0.92, "WR": 0.94, "TE": 0.94, "K": 0.90, "RB": 0.99, "DEF": 0.97},
    "precip": {"QB": 0.95, "WR": 0.95, "TE": 0.96, "K": 0.93, "RB": 0.98, "DEF": 0.96},
    "cold": {"QB": 0.97, "WR": 0.97, "TE": 0.97, "K": 0.94, "RB": 0.98, "DEF": 0.97},
}
_WEATHER_DEFAULT = {"QB": 0.96, "WR": 0.96, "TE": 0.96, "K": 0.94, "RB": 0.98, "DEF": 0.97}


def _neutral_factors(proj: float) -> dict:
    return {
        "proj": proj,
        "form": 1.0,
        "matchup": 1.0,
        "usage": 1.0,
        "avail": 1.0,
        "vegas": 1.0,
        "floor": 1.0,
        "weather": 1.0,
    }


def _weather_mult(weather_kind: Optional[str], position: Optional[str]) -> float:
    if not weather_kind:
        return 1.0
    kind = str(weather_kind).lower().strip()
    # Open-Meteo tag uses kind "weather" as a generic fallback; treat like light cold.
    if kind == "weather":
        kind = "cold"
    pos = (position or "").upper().strip()
    table = _WEATHER_MULT.get(kind) or _WEATHER_DEFAULT
    return float(table.get(pos) or table.get("WR") or 0.96)


def _vegas_mult(implied_total: float, position: Optional[str]) -> float:
    """Position-aware Vegas nudge from implied team total."""
    pos = (position or "").upper().strip()
    pass_catcher = pos in ("QB", "WR", "TE")
    if implied_total <= 17:
        if pass_catcher:
            return 0.92
        if pos == "RB":
            return 0.96
        if pos == "K":
            return 0.94
        return 0.94
    if implied_total >= 27:
        if pass_catcher:
            return 1.05
        if pos == "RB":
            return 1.02
        if pos == "K":
            return 1.03
        return 1.04
    return 1.0


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
    weather_kind: Optional[str] = None,
    position: Optional[str] = None,
    apply_matchup: bool = False,
) -> tuple[float, dict, Optional[str]]:
    """Return ``(score, score_factors, demotion)``.

    Factors: proj, form, matchup, usage, avail, vegas, floor, weather.
    Bye and OUT/IR zero the score. Non-projection signals are capped multipliers.

    ``apply_matchup`` defaults to False because weekly projections already
    reflect the opponent. Pass True only for matchup-neutral projection feeds.
    ``def_rank`` / ``def_total`` are still accepted so callers can pass them
    without branching; they only affect the score when ``apply_matchup`` is True.
    """
    form = mu = usage = avail = vegas = floor = weather = 1.0
    demotion = None
    try:
        proj = float(proj_pts or 0)
    except (TypeError, ValueError):
        proj = 0.0
    if on_bye:
        return 0.0, _neutral_factors(proj), "bye"

    try:
        recent = float(recent_ppg or 0)
    except (TypeError, ValueError):
        recent = 0.0
    try:
        season = float(season_ppg or 0)
    except (TypeError, ValueError):
        season = 0.0
    if recent > 0 and season > 0:
        # Mild form nudge — projections often already react to hot/cold streaks.
        form = min(1.08, max(0.92, recent / season))

    if apply_matchup and def_rank and def_total and def_total > 1:
        ease = (float(def_total) - float(def_rank)) / (float(def_total) - 1)
        # Residual only (±3%): even matchup-neutral feeds shouldn't swing hard.
        mu = 0.97 + ease * 0.06

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
            vegas = _vegas_mult(imp, position)
            if vegas < 1.0:
                demotion = demotion or "low_total"

    if bust_rate is not None:
        try:
            floor = min(1.10, max(0.90, 1.0 + (0.5 - float(bust_rate)) * 0.4))
        except (TypeError, ValueError):
            floor = 1.0

    weather = _weather_mult(weather_kind, position)
    if weather < 1.0:
        demotion = demotion or "weather"

    score = proj * form * mu * usage * avail * vegas * floor * weather
    return score, {
        "proj": proj,
        "form": round(form, 3),
        "matchup": round(mu, 3),
        "usage": round(usage, 3),
        "avail": round(avail, 3),
        "vegas": round(vegas, 3),
        "floor": round(floor, 3),
        "weather": round(weather, 3),
    }, demotion
