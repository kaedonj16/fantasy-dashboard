"""Compatibility shim: utils.league_scoring now lives in utils.league.

Re-exports every public name so existing imports keep working.
New code should import from utils.league directly.
"""
from utils.league import (  # noqa: F401,F403
    logger,
    _ALIASES,
    _DEFAULTS,
    _PER_UNIT_YARD_KEYS,
    _TRANSITIONAL_ALIASES,
    assign_scoring_rate,
    stamp_scoring_aliases,
    normalize_league_scoring,
)

__all__ = ['logger', '_ALIASES', '_DEFAULTS', '_PER_UNIT_YARD_KEYS', '_TRANSITIONAL_ALIASES', 'assign_scoring_rate', 'stamp_scoring_aliases', 'normalize_league_scoring']


# --- monkeypatch propagation (see utils/_shim.py) ---
from utils._shim import propagate_sets_to as _propagate_sets_to
import importlib as _importlib
_propagate_sets_to(__name__, _importlib.import_module("utils.league"))
