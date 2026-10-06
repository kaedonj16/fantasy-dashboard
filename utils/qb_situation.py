"""Compatibility shim: utils.qb_situation now lives in utils.start_sit.

Re-exports every public name so existing imports keep working.
New code should import from utils.start_sit directly.
"""
from utils.start_sit import (  # noqa: F401,F403
    qb_situation_chip,
)

__all__ = ['qb_situation_chip']


# --- monkeypatch propagation (see utils/_shim.py) ---
from utils._shim import propagate_sets_to as _propagate_sets_to
import importlib as _importlib
_propagate_sets_to(__name__, _importlib.import_module("utils.start_sit"))
