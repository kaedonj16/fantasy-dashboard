"""Compatibility shim: utils.game_conditions now lives in utils.start_sit.

Re-exports every public name so existing imports keep working.
New code should import from utils.start_sit directly.
"""
from utils.start_sit import (  # noqa: F401,F403
    logger,
    implied_team_total,
    total_tag,
    weather_tag,
    parse_tank01_odds,
    parse_open_meteo_daily,
    _to_float,
    _norm,
    _first_book,
    _ODDS_CACHE,
    _ODDS_TTL,
    _WEATHER_CACHE,
    _WEATHER_TTL,
    fetch_week_odds,
    fetch_game_weather,
    build_week_conditions,
    _day_offset,
)

__all__ = ['logger', 'implied_team_total', 'total_tag', 'weather_tag', 'parse_tank01_odds', 'parse_open_meteo_daily', '_to_float', '_norm', '_first_book', '_ODDS_CACHE', '_ODDS_TTL', '_WEATHER_CACHE', '_WEATHER_TTL', 'fetch_week_odds', 'fetch_game_weather', 'build_week_conditions', '_day_offset']


# --- monkeypatch propagation (see utils/_shim.py) ---
from utils._shim import propagate_sets_to as _propagate_sets_to
import importlib as _importlib
_propagate_sets_to(__name__, _importlib.import_module("utils.start_sit"))
