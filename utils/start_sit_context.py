"""Pure adapters from existing weekly/team caches into Start/Sit inputs."""
from __future__ import annotations

from statistics import mean, pstdev

from utils.nfl_stadiums import normalize_nfl_team


def expected_plays_context(team_rows: dict, team: str, opponent: str, nfl_avg) -> dict:
    """Blend actual offense plays with opponent plays faced; no invented pace."""
    team = normalize_nfl_team(team)
    opponent = normalize_nfl_team(opponent)
    try:
        avg = float(nfl_avg)
    except (TypeError, ValueError):
        return {}
    own = (team_rows or {}).get(team) or {}
    opp = (team_rows or {}).get(opponent) or {}
    try:
        offense = float(own["off_plays_pg"])
        allowed = float(opp.get("plays_faced_l4_pg") or opp["plays_faced_pg"])
    except (KeyError, TypeError, ValueError):
        return {}
    # Shrink both observations toward league average, then blend evenly. This
    # guards against one anomalous recent game while retaining real possession.
    expected = mean((0.7 * offense + 0.3 * avg, 0.7 * allowed + 0.3 * avg))
    return {"expected_team_plays": round(expected, 1), "league_average_plays": round(avg, 1),
            "source": "team_play_volume"}


def role_confidence_from_trend(trend: dict) -> float | None:
    """0..1 recent role stability, preserving confirmed promotions."""
    series = trend.get("series") if isinstance(trend, dict) else None
    if not isinstance(series, list) or len(series) < 2:
        return None
    try:
        values = [float(v) for v in series[-3:]]
    except (TypeError, ValueError):
        return None
    level = max(1.0, mean(values))
    stability = max(0.0, 1.0 - pstdev(values) / level)
    # A promoted player with two consecutive elevated readings is not punished
    # for the old low baseline that made them interesting in the first place.
    if len(values) >= 2 and values[-1] >= values[-2] >= mean(series[:-2] or values):
        stability = max(stability, 0.75)
    return round(min(1.0, stability), 3)
