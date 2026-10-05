"""Compatibility shim: utils.nfl_stadiums now lives in utils.nfl.

Re-exports every public name so existing imports keep working.
New code should import from utils.nfl directly.
"""
from utils.nfl import (  # noqa: F401,F403
    STADIUMS,
    ALIASES,
    _FULL_NAMES,
    _NICKNAMES,
    _COLD_WEEK_START,
    normalize_team,
    normalize_nfl_team,
    stadium_coords,
    game_environment,
)

__all__ = ['STADIUMS', 'ALIASES', '_FULL_NAMES', '_NICKNAMES', '_COLD_WEEK_START', 'normalize_team', 'normalize_nfl_team', 'stadium_coords', 'game_environment']


# --- monkeypatch propagation (see utils/_shim.py) ---
from utils._shim import propagate_sets_to as _propagate_sets_to
import importlib as _importlib
_propagate_sets_to(__name__, _importlib.import_module("utils.nfl"))
