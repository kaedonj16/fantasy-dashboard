"""Compatibility shim: utils.draft_capital now lives in utils.league.

Re-exports every public name so existing imports keep working.
New code should import from utils.league directly.
"""
from utils.league import (  # noqa: F401,F403
    _NO_DRAFT_CAPITAL,
    _HAS_PICK_FEED,
    normalize_draft_capital_platform,
    provider_exposes_draft_capital,
    has_future_draft_capital,
    draft_capital_unavailable_copy,
)

__all__ = ['_NO_DRAFT_CAPITAL', '_HAS_PICK_FEED', 'normalize_draft_capital_platform', 'provider_exposes_draft_capital', 'has_future_draft_capital', 'draft_capital_unavailable_copy']


# --- monkeypatch propagation (see utils/_shim.py) ---
from utils._shim import propagate_sets_to as _propagate_sets_to
import importlib as _importlib
_propagate_sets_to(__name__, _importlib.import_module("utils.league"))
