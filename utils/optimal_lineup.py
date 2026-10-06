"""Compatibility shim: utils.optimal_lineup now lives in utils.lineups.

Re-exports every public name so existing imports keep working.
New code should import from utils.lineups directly.
"""
from utils.lineups import (  # noqa: F401,F403
    _pid,
    starting_slots,
    assign_optimal_lineup,
    assign_fixed_lineup,
    analyze_lineup,
    analyze_team_week,
    compute_optimal_lineup,
)

__all__ = ['_pid', 'starting_slots', 'assign_optimal_lineup', 'assign_fixed_lineup', 'analyze_lineup', 'analyze_team_week', 'compute_optimal_lineup']


# --- monkeypatch propagation (see utils/_shim.py) ---
from utils._shim import propagate_sets_to as _propagate_sets_to
import importlib as _importlib
_propagate_sets_to(__name__, _importlib.import_module("utils.lineups"))
