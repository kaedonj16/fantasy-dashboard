"""Compatibility shim: utils.pick_slots now lives in utils.draft.

Re-exports every public name so existing imports keep working.
New code should import from utils.draft directly.
"""
from utils.draft import (  # noqa: F401,F403
    placements_from_bracket,
    compute_pick_slots,
    slots_from_regular_season,
    pick_label,
    avg_pick_value_for_round,
    bucket_for_slot,
    pick_value_from_table,
    is_pick_asset_id,
    parse_pick_asset,
)

__all__ = ['placements_from_bracket', 'compute_pick_slots', 'slots_from_regular_season', 'pick_label', 'avg_pick_value_for_round', 'bucket_for_slot', 'pick_value_from_table', 'is_pick_asset_id', 'parse_pick_asset']


# --- monkeypatch propagation (see utils/_shim.py) ---
from utils._shim import propagate_sets_to as _propagate_sets_to
import importlib as _importlib
_propagate_sets_to(__name__, _importlib.import_module("utils.draft"))
