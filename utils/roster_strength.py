"""Compatibility shim: utils.roster_strength now lives in utils.trade.

Re-exports every public name so existing imports keep working.
New code should import from utils.trade directly.
"""
from utils.trade import (  # noqa: F401,F403
    weighted_pos_strength,
    CORE_POSITIONS,
    _player_position,
    roster_pos_value_lists,
    rank_rosters_by_position,
    strength_percentile,
    average_league_percentiles,
    ROSTER_COMPONENT_WEIGHTS,
    fit_roster_component_weights,
    positional_strength_profile,
    STARTER_THRESHOLD,
    DEPTH_FLOOR,
    _SF_QB_THRESHOLD_MULT,
    derive_league_thresholds,
    dedicated_starter_counts,
)

__all__ = ['weighted_pos_strength', 'CORE_POSITIONS', '_player_position', 'roster_pos_value_lists', 'rank_rosters_by_position', 'strength_percentile', 'average_league_percentiles', 'ROSTER_COMPONENT_WEIGHTS', 'fit_roster_component_weights', 'positional_strength_profile', 'STARTER_THRESHOLD', 'DEPTH_FLOOR', '_SF_QB_THRESHOLD_MULT', 'derive_league_thresholds', 'dedicated_starter_counts']


# --- monkeypatch propagation (see utils/_shim.py) ---
from utils._shim import propagate_sets_to as _propagate_sets_to
import importlib as _importlib
_propagate_sets_to(__name__, _importlib.import_module("utils.trade"))
