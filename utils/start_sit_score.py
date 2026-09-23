"""Single start/sit ranking formula: projection times capped multipliers.

The START badges, optimal-lineup banner, and Compare card all rank on this
score so they cannot contradict each other. Kept as a pure function so the
math is unit-tested without Flask.

Weekly projections already bake in the opponent, so defensive matchup rank is
*not* re-multiplied into the score (that used to double-count). Matchup stays on
the row as a chip only. Weather and Vegas *are* applied here because those
signals are usually missing from raw projection feeds.

Offensive-line quality is only *partly* reflected in projection feeds, so it is
applied as a deliberately small residual (±4%), not a full multiplier — the same
double-count caution as matchup. Callers pass the position-relevant 0-100 index
(run block for RB, pass block for QB/WR/TE); 50 is league-average and neutral.
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
        "oline": 1.0,
        "expected_plays": 1.0,
        "role": 1.0,
        "def_injuries": 1.0,
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


def _lerp_clamped(x: float, x0: float, x1: float, y0: float, y1: float) -> float:
    """Linear interpolation of x in [x0, x1] → [y0, y1], clamped at the ends."""
    if x <= x0:
        return y0
    if x >= x1:
        return y1
    return y0 + (y1 - y0) * (x - x0) / (x1 - x0)


def _vegas_mult(implied_total: float, position: Optional[str]) -> float:
    """Position-aware Vegas nudge from implied team total.

    Piecewise-linear: ramps from the low-total haircut up to neutral over
    17→20 and from neutral up to the high-total boost over 24→27. The old
    step function jumped ~8% between 17.0 and 17.1 implied total, so a
    0.1-point Vegas line move could flip a start/sit call; now the nudge is
    continuous in the total.
    """
    pos = (position or "").upper().strip()
    pass_catcher = pos in ("QB", "WR", "TE")
    if pass_catcher:
        low, high = 0.92, 1.05
    elif pos == "RB":
        low, high = 0.96, 1.02
    elif pos == "K":
        low, high = 0.94, 1.03
    else:
        low, high = 0.94, 1.04
    if implied_total < 20.0:
        return _lerp_clamped(implied_total, 17.0, 20.0, low, 1.0)
    if implied_total > 24.0:
        return _lerp_clamped(implied_total, 24.0, 27.0, 1.0, high)
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
    oline_index: Optional[float] = None,
    apply_matchup: bool = False,
    expected_team_plays: Optional[float] = None,
    league_average_plays: Optional[float] = None,
    role_confidence: Optional[float] = None,
    defensive_injury_impact: Optional[float] = None,
) -> tuple[float, dict, Optional[str]]:
    """Return ``(score, score_factors, demotion)``.

    Factors: proj, form, matchup, usage, avail, vegas, floor, weather, oline.
    Bye and OUT/IR zero the score. Non-projection signals are capped multipliers.

    ``oline_index`` is the player's position-relevant 0-100 O-line rating (run
    block for RB, pass block for QB/WR/TE, composite otherwise); 50 is neutral.
    It is applied as a small ±4% residual because projections already reflect
    line quality in part. Pass None to leave the factor neutral.

    ``apply_matchup`` defaults to False because weekly projections already
    reflect the opponent. Pass True only for matchup-neutral projection feeds.
    ``def_rank`` / ``def_total`` are still accepted so callers can pass them
    without branching; they only affect the score when ``apply_matchup`` is True.
    """
    form = mu = usage = avail = vegas = floor = weather = oline = plays = role = def_inj = 1.0
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
        # Soft residual only (−5%). Questionable players usually play, starting
        # them is normally fine, and weekly projections often already bake in
        # some risk — a 15% haircut was flipping too many start/sit calls.
        avail = 0.95
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

    if oline_index is not None:
        try:
            e = max(0.0, min(100.0, float(oline_index))) / 100.0
            # 0.96 (worst line) .. 1.04 (best line); ~1.0 at league-average (50).
            oline = 0.96 + e * 0.08
            if oline < 1.0:
                demotion = demotion or "oline"
        except (TypeError, ValueError):
            oline = 1.0

    if expected_team_plays is not None and league_average_plays:
        try:
            # Pace affects opportunity, but feeds and projections are correlated;
            # apply only half the relative delta and cap the residual at ±5%.
            relative = float(expected_team_plays) / float(league_average_plays) - 1.0
            plays = min(1.05, max(0.95, 1.0 + relative * 0.5))
            if plays < 1.0:
                demotion = demotion or "low_play_volume"
        except (TypeError, ValueError, ZeroDivisionError):
            plays = 1.0
    if role_confidence is not None:
        try:
            confidence = min(1.0, max(0.0, float(role_confidence)))
            # Confidence widens/narrows the range more than it moves the mean.
            role = 0.97 + 0.03 * confidence
            if confidence < 0.45:
                demotion = demotion or "volatile_role"
        except (TypeError, ValueError):
            role = 1.0
    if defensive_injury_impact is not None:
        try:
            # Caller supplies a quality-weighted 0..1 impact; absence is neutral.
            def_inj = 1.0 + 0.03 * min(1.0, max(0.0, float(defensive_injury_impact)))
        except (TypeError, ValueError):
            def_inj = 1.0

    score = proj * form * mu * usage * avail * vegas * floor * weather * oline * plays * role * def_inj
    return score, {
        "proj": proj,
        "form": round(form, 3),
        "matchup": round(mu, 3),
        "usage": round(usage, 3),
        "avail": round(avail, 3),
        "vegas": round(vegas, 3),
        "floor": round(floor, 3),
        "weather": round(weather, 3),
        "oline": round(oline, 3),
        "expected_plays": round(plays, 3),
        "role": round(role, 3),
        "def_injuries": round(def_inj, 3),
    }, demotion


def likely_range(expected: float, role_confidence: Optional[float] = None) -> tuple[float, float]:
    """Conservative display range; confidence changes width, never fake mean precision."""
    try:
        mean = max(0.0, float(expected))
        confidence = 0.5 if role_confidence is None else min(1.0, max(0.0, float(role_confidence)))
    except (TypeError, ValueError):
        return (0.0, 0.0)
    width = mean * (0.22 + (1.0 - confidence) * 0.18)
    return round(max(0.0, mean - width), 1), round(mean + width, 1)
