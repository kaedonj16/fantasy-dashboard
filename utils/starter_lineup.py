"""Compatibility shim: utils.starter_lineup now lives in utils.lineups.

Re-exports every public name so existing imports keep working.
New code should import from utils.lineups directly.
"""
from utils.lineups import (  # noqa: F401,F403
    _grade_slot,
    starter_slots,
    derive_starters_from_slots,
)

__all__ = ['_grade_slot', 'starter_slots', 'derive_starters_from_slots']


# --- monkeypatch propagation (see utils/_shim.py) ---
from utils._shim import propagate_sets_to as _propagate_sets_to
import importlib as _importlib
_propagate_sets_to(__name__, _importlib.import_module("utils.lineups"))
