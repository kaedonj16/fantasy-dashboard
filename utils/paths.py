"""Compatibility shim: utils.paths now lives in utils.data_cache.

Re-exports every public name so existing imports keep working.
New code should import from utils.data_cache directly.
"""
from utils.data_cache import (  # noqa: F401,F403
    ROOT_DIR,
    DATA_DIR,
    CACHE_DIR,
    PLAYER_HISTORY_DIR,
    PLAYER_INVESTMENT_DIR,
)

__all__ = ['ROOT_DIR', 'DATA_DIR', 'CACHE_DIR', 'PLAYER_HISTORY_DIR', 'PLAYER_INVESTMENT_DIR']


# --- monkeypatch propagation (see utils/_shim.py) ---
from utils._shim import propagate_sets_to as _propagate_sets_to
import importlib as _importlib
_propagate_sets_to(__name__, _importlib.import_module("utils.data_cache"))
