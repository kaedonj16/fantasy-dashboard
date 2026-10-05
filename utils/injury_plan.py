"""Compatibility shim: utils.injury_plan now lives in utils.digest.

Re-exports every public name so existing imports keep working.
New code should import from utils.digest directly.
"""
from utils.digest import (  # noqa: F401,F403
    _VALUE_STASH,
    _VALUE_HOLD,
    status_weeks_band,
    resolve_weeks_out,
    injury_plan,
    ir_capacity,
)

__all__ = ['_VALUE_STASH', '_VALUE_HOLD', 'status_weeks_band', 'resolve_weeks_out', 'injury_plan', 'ir_capacity']


# --- monkeypatch propagation (see utils/_shim.py) ---
from utils._shim import propagate_sets_to as _propagate_sets_to
import importlib as _importlib
_propagate_sets_to(__name__, _importlib.import_module("utils.digest"))
