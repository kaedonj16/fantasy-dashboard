"""Compatibility shim: utils.defensive_matchup_ratings now lives in utils.nfl.

Re-exports every public name so existing imports keep working.
New code should import from utils.nfl directly.
"""
from utils.nfl import (  # noqa: F401,F403
    BASELINE_WEIGHTS,
    MIN_PARTICIPATION,
    WINSOR_MULTIPLIER,
    EARLY_SEASON_CURRENT_WEIGHTS,
    season_weights,
    scoring_profile_hash,
    rating_cache_key,
    blend_value,
    rank_values,
    normalize_schedule,
    rank_team_schedules,
    meaningful_participation,
    pregame_baseline,
    game_adjustment,
    aggregate_defense_games,
)

__all__ = ['BASELINE_WEIGHTS', 'MIN_PARTICIPATION', 'WINSOR_MULTIPLIER', 'EARLY_SEASON_CURRENT_WEIGHTS', 'season_weights', 'scoring_profile_hash', 'rating_cache_key', 'blend_value', 'rank_values', 'normalize_schedule', 'rank_team_schedules', 'meaningful_participation', 'pregame_baseline', 'game_adjustment', 'aggregate_defense_games']


# --- monkeypatch propagation (see utils/_shim.py) ---
from utils._shim import propagate_sets_to as _propagate_sets_to
import importlib as _importlib
_propagate_sets_to(__name__, _importlib.import_module("utils.nfl"))
