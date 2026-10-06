"""Compatibility shim: utils.scorezone_demo now lives in utils.scorezone.

Re-exports every public name so existing imports keep working.
New code should import from utils.scorezone directly.
"""
from utils.scorezone import (  # noqa: F401,F403
    DEMO_GAME_SECONDS,
    DEMO_SCORING,
    demo_rng,
    demo_script,
    demo_fold,
    demo_pts,
)

__all__ = ['DEMO_GAME_SECONDS', 'DEMO_SCORING', 'demo_rng', 'demo_script', 'demo_fold', 'demo_pts']


# --- monkeypatch propagation (see utils/_shim.py) ---
from utils._shim import propagate_sets_to as _propagate_sets_to
import importlib as _importlib
_propagate_sets_to(__name__, _importlib.import_module("utils.scorezone"))
