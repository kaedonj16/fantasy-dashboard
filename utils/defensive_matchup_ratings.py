"""Shared policy and pure calculations for defense-vs-position ratings.

The Schedule Assistant intentionally ranks the same points-allowed value that
it displays.  Keeping the blend and SOS math here prevents the player grid and
rankings view from quietly drifting apart.
"""
from __future__ import annotations

import hashlib
import json


EARLY_SEASON_CURRENT_WEIGHTS = {
    0: 0.00,
    1: 0.25,
    2: 0.40,
    3: 0.55,
    4: 0.70,
    5: 0.85,
}


def season_weights(completed_through_week: int, *, blend: bool = True) -> tuple[float, float]:
    """Return ``(previous, current)`` weights for the selected season."""
    if not blend:
        return 0.0, 1.0
    current = EARLY_SEASON_CURRENT_WEIGHTS.get(max(0, int(completed_through_week)), 1.0)
    return 1.0 - current, current


def scoring_profile_hash(settings: dict | None) -> str:
    """Stable short identity for every scoring category, including K/DST."""
    canonical = json.dumps(settings or {}, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def rating_cache_key(season: int, completed_week: int, settings: dict | None, position: str) -> str:
    return f"{int(season)}:{int(completed_week)}:{scoring_profile_hash(settings)}:{position.upper()}"


def blend_value(previous, current, completed_week: int, *, blend: bool = True):
    """Blend available values without ever treating missing data as zero."""
    if previous is None:
        return (current, "current") if current is not None else (None, "unavailable")
    if current is None:
        return previous, "prior"
    prior_w, current_w = season_weights(completed_week, blend=blend)
    return prior_w * float(previous) + current_w * float(current), "blended" if prior_w else "current"


def rank_values(values: dict[str, float | None]) -> tuple[dict[str, int], int]:
    """Rank higher points allowed as easier (#1), omitting unavailable teams."""
    ordered = sorted(((team, value) for team, value in values.items() if value is not None),
                     key=lambda item: (-float(item[1]), item[0]))
    return {team: index + 1 for index, (team, _) in enumerate(ordered)}, len(ordered)


def rank_team_schedules(team_opponents: dict[str, list[str | None]], values: dict[str, float | None]):
    """Rank schedules from mean underlying FPA; byes/missing opponents are excluded."""
    averages = {}
    for team, opponents in team_opponents.items():
        usable = [float(values[o]) for o in opponents if o and values.get(o) is not None]
        if usable:
            averages[team] = sum(usable) / len(usable)
    ranks, total = rank_values(averages)
    return {team: (ranks[team], total, round(value, 2)) for team, value in averages.items()}
