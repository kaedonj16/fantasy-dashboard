"""Compatibility shim: utils.matchup_schedule now lives in utils.league.

Re-exports every public name so existing imports keep working.
New code should import from utils.league directly.
"""
from utils.league import (  # noqa: F401,F403
    _starters_look_like_full_roster,
    lineup_from_roster,
    synthetic_week_matchups,
    last_finalized_week,
    resolve_matchup_week,
)

__all__ = ['_starters_look_like_full_roster', 'lineup_from_roster', 'synthetic_week_matchups', 'last_finalized_week', 'resolve_matchup_week']


# --- monkeypatch propagation (see utils/_shim.py) ---
from utils._shim import propagate_sets_to as _propagate_sets_to
import importlib as _importlib
_propagate_sets_to(__name__, _importlib.import_module("utils.league"))
