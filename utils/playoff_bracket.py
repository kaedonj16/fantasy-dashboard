"""Compatibility shim: utils.playoff_bracket now lives in utils.standings.

Re-exports every public name so existing imports keep working.
New code should import from utils.standings directly.
"""
from utils.standings import (  # noqa: F401,F403
    _rid,
    _mid,
    _pts,
    pair_matchup_sides,
    _winner_loser,
    derive_bracket_from_matchups,
    project_bracket_from_seeds,
    derive_or_project_bracket,
)

__all__ = ['_rid', '_mid', '_pts', 'pair_matchup_sides', '_winner_loser', 'derive_bracket_from_matchups', 'project_bracket_from_seeds', 'derive_or_project_bracket']


# --- monkeypatch propagation (see utils/_shim.py) ---
from utils._shim import propagate_sets_to as _propagate_sets_to
import importlib as _importlib
_propagate_sets_to(__name__, _importlib.import_module("utils.standings"))
