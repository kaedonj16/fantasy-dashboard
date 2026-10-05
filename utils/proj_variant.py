"""Compatibility shim: utils.proj_variant now lives in utils.projections.

Re-exports every public name so existing imports keep working.
New code should import from utils.projections directly.
"""
from utils.projections import (  # noqa: F401,F403
    pick_proj_variant,
    pick_proj_variant_from_draft_scoring,
)

__all__ = ['pick_proj_variant', 'pick_proj_variant_from_draft_scoring']


# --- monkeypatch propagation (see utils/_shim.py) ---
from utils._shim import propagate_sets_to as _propagate_sets_to
import importlib as _importlib
_propagate_sets_to(__name__, _importlib.import_module("utils.projections"))
