"""Compatibility shim: utils.viewer_resolve now lives in utils.league.

Re-exports every public name so existing imports keep working.
New code should import from utils.league directly.
"""
from utils.league import (  # noqa: F401,F403
    normalize_sleeper_username,
    resolve_viewer_for_league,
)

__all__ = ['normalize_sleeper_username', 'resolve_viewer_for_league']


# --- monkeypatch propagation (see utils/_shim.py) ---
from utils._shim import propagate_sets_to as _propagate_sets_to
import importlib as _importlib
_propagate_sets_to(__name__, _importlib.import_module("utils.league"))
