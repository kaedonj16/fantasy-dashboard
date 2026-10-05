"""Canonical Sleeper league-type values used by trade-intel pipelines."""
from enum import IntEnum


class LeagueType(IntEnum):
    """Values published by ``Sleeper league.settings.type``."""

    REDRAFT = 0
    KEEPER = 1
    DYNASTY = 2


CALIBRATABLE_LEAGUE_TYPES = (LeagueType.REDRAFT, LeagueType.DYNASTY)

# league_type int -> per-format bucket name on trade_intel_player_stats.league_format.
FORMAT_NAMES = {
    int(LeagueType.REDRAFT): "redraft",
    int(LeagueType.KEEPER): "keeper",
    int(LeagueType.DYNASTY): "dynasty",
}


def format_name(league_type: int | None) -> str:
    """Map a Sleeper league_type int to a stats-bucket name.

    0 -> "redraft", 1 -> "keeper", 2 -> "dynasty"; anything unknown (missing
    league row, NULL, future values) folds into the "all" aggregate bucket.
    """
    if league_type is None:
        return "all"
    try:
        return FORMAT_NAMES[int(league_type)]
    except (KeyError, TypeError, ValueError):
        return "all"


def league_format_sql_param(league_format: str) -> int | None:
    """Map a UI ``dynasty``/``redraft``/``keeper``/``all`` filter onto ``trade_intel_leagues.league_type``.

    Crawler contract is Sleeper's: 0 = redraft, 1 = keeper, 2 = dynasty.
    Keeper is stored (crawler/discovery include it) but still excluded from
    value calibration; see ``calibration_mode``.
    """
    lf = str(league_format or "all").strip().lower()
    if lf == "dynasty":
        return int(LeagueType.DYNASTY)
    if lf == "redraft":
        return int(LeagueType.REDRAFT)
    if lf == "keeper":
        return int(LeagueType.KEEPER)
    return None


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
