"""Compatibility shim: utils.nfl_teams now lives in utils.nfl.

Re-exports every public name so existing imports keep working.
New code should import from utils.nfl directly.
"""
from utils.nfl import (  # noqa: F401,F403
    TEAM_FULL_NAMES,
    get_team_full_name,
)

__all__ = ['TEAM_FULL_NAMES', 'get_team_full_name']


# --- monkeypatch propagation (see utils/_shim.py) ---
from utils._shim import propagate_sets_to as _propagate_sets_to
import importlib as _importlib
_propagate_sets_to(__name__, _importlib.import_module("utils.nfl"))
