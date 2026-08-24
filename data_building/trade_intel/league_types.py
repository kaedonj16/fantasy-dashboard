"""Canonical Sleeper league-type values used by trade-intel pipelines."""
from enum import IntEnum


class LeagueType(IntEnum):
    """Values published by ``Sleeper league.settings.type``."""

    REDRAFT = 0
    KEEPER = 1
    DYNASTY = 2


CALIBRATABLE_LEAGUE_TYPES = (LeagueType.REDRAFT, LeagueType.DYNASTY)


def calibration_mode(league_type: int) -> str:
    """Return the value-market name, rejecting keeper and unknown formats."""
    try:
        normalized = LeagueType(league_type)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Unsupported league type for calibration: {league_type!r}") from exc
    if normalized is LeagueType.REDRAFT:
        return "redraft"
    if normalized is LeagueType.DYNASTY:
        return "dynasty"
    raise ValueError("Keeper leagues cannot be used to calibrate redraft values")
