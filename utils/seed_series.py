"""Compatibility shim: utils.seed_series now lives in utils.standings.

Re-exports every public name so existing imports keep working.
New code should import from utils.standings directly.
"""
from utils.standings import (  # noqa: F401,F403
    seed_series_for,
)

__all__ = ['seed_series_for']


# --- monkeypatch propagation (see utils/_shim.py) ---
from utils._shim import propagate_sets_to as _propagate_sets_to
import importlib as _importlib
_propagate_sets_to(__name__, _importlib.import_module("utils.standings"))
