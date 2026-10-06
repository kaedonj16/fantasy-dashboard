"""Compatibility shim: utils.pick_score now lives in utils.draft.

Re-exports every public name so existing imports keep working.
New code should import from utils.draft directly.
"""
from utils.draft import (  # noqa: F401,F403
    PS_WEIGHTS,
    PS_AGE_PEAKS,
    ps_tier_of,
    starter_counts,
    empirical_slot_allocation,
    compute_pick_score,
)

__all__ = ['PS_WEIGHTS', 'PS_AGE_PEAKS', 'ps_tier_of', 'starter_counts', 'empirical_slot_allocation', 'compute_pick_score']


# --- monkeypatch propagation (see utils/_shim.py) ---
from utils._shim import propagate_sets_to as _propagate_sets_to
import importlib as _importlib
_propagate_sets_to(__name__, _importlib.import_module("utils.draft"))
