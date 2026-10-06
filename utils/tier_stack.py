"""Compatibility shim: utils.tier_stack now lives in utils.trade.

Re-exports every public name so existing imports keep working.
New code should import from utils.trade directly.
"""
from utils.trade import (  # noqa: F401,F403
    NUM_TIERS,
    build_tier_caps,
    asset_tier,
    apply_tier_stack_adjustment,
)

__all__ = ['NUM_TIERS', 'build_tier_caps', 'asset_tier', 'apply_tier_stack_adjustment']


# --- monkeypatch propagation (see utils/_shim.py) ---
from utils._shim import propagate_sets_to as _propagate_sets_to
import importlib as _importlib
_propagate_sets_to(__name__, _importlib.import_module("utils.trade"))
