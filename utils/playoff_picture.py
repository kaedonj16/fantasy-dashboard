"""Compatibility shim: utils.playoff_picture now lives in utils.standings.

Re-exports every public name so existing imports keep working.
New code should import from utils.standings directly.
"""
from utils.standings import (  # noqa: F401,F403
    BYE,
    CLINCHED,
    IN,
    BUBBLE,
    ELIMINATED,
    bye_count,
    _ordinal,
    _games_back,
    compute_playoff_picture,
    _scenario,
)

__all__ = ['BYE', 'CLINCHED', 'IN', 'BUBBLE', 'ELIMINATED', 'bye_count', '_ordinal', '_games_back', 'compute_playoff_picture', '_scenario']


# --- monkeypatch propagation (see utils/_shim.py) ---
from utils._shim import propagate_sets_to as _propagate_sets_to
import importlib as _importlib
_propagate_sets_to(__name__, _importlib.import_module("utils.standings"))
