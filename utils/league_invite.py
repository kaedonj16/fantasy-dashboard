"""Compatibility shim: utils.league_invite now lives in utils.league.

Re-exports every public name so existing imports keep working.
New code should import from utils.league directly.
"""
from utils.league import (  # noqa: F401,F403
    _PLATFORMS,
    normalize_invite_platform,
    league_invite_path,
    league_invite_url,
    dashboard_after_invite,
    is_league_plan_buyer,
)

__all__ = ['_PLATFORMS', 'normalize_invite_platform', 'league_invite_path', 'league_invite_url', 'dashboard_after_invite', 'is_league_plan_buyer']


# --- monkeypatch propagation (see utils/_shim.py) ---
from utils._shim import propagate_sets_to as _propagate_sets_to
import importlib as _importlib
_propagate_sets_to(__name__, _importlib.import_module("utils.league"))
