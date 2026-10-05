"""Compatibility shim: utils.nfl_context now lives in utils.nfl.

Re-exports every public name so existing imports keep working.
New code should import from utils.nfl directly.
"""
from utils.nfl import (  # noqa: F401,F403
    VALID_PHASES,
    calendar_nfl_season,
    _calendar_phase,
    normalize_nfl_state,
    nfl_state_is_stale,
    nfl_state_last_good_at,
    season_cache_key,
    current_sample_weight,
)

__all__ = ['VALID_PHASES', 'calendar_nfl_season', '_calendar_phase', 'normalize_nfl_state', 'nfl_state_is_stale', 'nfl_state_last_good_at', 'season_cache_key', 'current_sample_weight']


# --- monkeypatch propagation (see utils/_shim.py) ---
from utils._shim import propagate_sets_to as _propagate_sets_to
import importlib as _importlib
_propagate_sets_to(__name__, _importlib.import_module("utils.nfl"))
