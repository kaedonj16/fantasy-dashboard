"""Compatibility shim: utils.tier_thresholds now lives in utils.trade.

Re-exports every public name so existing imports keep working.
New code should import from utils.trade directly.
"""
from utils.trade import (  # noqa: F401,F403
    FALLBACK_THRESHOLDS,
    ELITE_RANK_CUTOFFS,
    MAX_DISPLAY_TIERS,
    compute_tier_thresholds,
)

__all__ = ['FALLBACK_THRESHOLDS', 'ELITE_RANK_CUTOFFS', 'MAX_DISPLAY_TIERS', 'compute_tier_thresholds']


# --- monkeypatch propagation (see utils/_shim.py) ---
from utils._shim import propagate_sets_to as _propagate_sets_to
import importlib as _importlib
_propagate_sets_to(__name__, _importlib.import_module("utils.trade"))
