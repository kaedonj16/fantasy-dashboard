"""Compatibility shim: utils.consistency now lives in utils.players.

Re-exports every public name so existing imports keep working.
New code should import from utils.players directly.
"""
from utils.players import (  # noqa: F401,F403
    _POS_THRESHOLDS,
    _DEFAULT_THRESHOLDS,
    _MIN_GAMES,
    BLEND_FULL_SEASON,
    _label_for,
    _percentile,
    consistency_profile,
    blended_consistency_profile,
)

__all__ = ['_POS_THRESHOLDS', '_DEFAULT_THRESHOLDS', '_MIN_GAMES', 'BLEND_FULL_SEASON', '_label_for', '_percentile', 'consistency_profile', 'blended_consistency_profile']


# --- monkeypatch propagation (see utils/_shim.py) ---
from utils._shim import propagate_sets_to as _propagate_sets_to
import importlib as _importlib
_propagate_sets_to(__name__, _importlib.import_module("utils.players"))
