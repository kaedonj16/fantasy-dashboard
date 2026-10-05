"""Compatibility shim: utils.player_tiers now lives in utils.trade.

Re-exports every public name so existing imports keep working.
New code should import from utils.trade directly.
"""
from utils.trade import (  # noqa: F401,F403
    SKILL_POS,
    positional_ranks,
    pos_category,
    roster_position_counts,
    starter_gap_needs,
    startable_surplus,
    ceiling_needs,
    consolidate_target_allowed,
)

__all__ = ['SKILL_POS', 'positional_ranks', 'pos_category', 'roster_position_counts', 'starter_gap_needs', 'startable_surplus', 'ceiling_needs', 'consolidate_target_allowed']


# --- monkeypatch propagation (see utils/_shim.py) ---
from utils._shim import propagate_sets_to as _propagate_sets_to
import importlib as _importlib
_propagate_sets_to(__name__, _importlib.import_module("utils.trade"))
