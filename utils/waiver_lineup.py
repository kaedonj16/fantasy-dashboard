"""Compatibility shim: utils.waiver_lineup now lives in utils.waivers.

Re-exports every public name so existing imports keep working.
New code should import from utils.waivers directly.
"""
from utils.waivers import (  # noqa: F401,F403
    MIN_MEANINGFUL_GAIN,
    PickupEvaluation,
    _slot_counts,
    _ASSIGN_ORDER,
    _assign_locked,
    _counts_to_positions,
    optimal_starters_and_points,
    _starter_slots,
    evaluate_pickup_week,
    _bench_slots,
    evaluate_pickup,
    _classify_outcome,
)

__all__ = ['MIN_MEANINGFUL_GAIN', 'PickupEvaluation', '_slot_counts', '_ASSIGN_ORDER', '_assign_locked', '_counts_to_positions', 'optimal_starters_and_points', '_starter_slots', 'evaluate_pickup_week', '_bench_slots', 'evaluate_pickup', '_classify_outcome']


# --- monkeypatch propagation (see utils/_shim.py) ---
from utils._shim import propagate_sets_to as _propagate_sets_to
import importlib as _importlib
_propagate_sets_to(__name__, _importlib.import_module("utils.waivers"))
