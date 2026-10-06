"""Compatibility shim: utils.roster_compliance now lives in utils.lineups.

Re-exports every public name so existing imports keep working.
New code should import from utils.lineups directly.
"""
from utils.lineups import (  # noqa: F401,F403
    IR_SLOT_ELIGIBLE,
    effective_taxi_slots,
    roster_compliance_issues,
)

__all__ = ['IR_SLOT_ELIGIBLE', 'effective_taxi_slots', 'roster_compliance_issues']


# --- monkeypatch propagation (see utils/_shim.py) ---
from utils._shim import propagate_sets_to as _propagate_sets_to
import importlib as _importlib
_propagate_sets_to(__name__, _importlib.import_module("utils.lineups"))
