"""Compatibility shim: utils.player_identity now lives in utils.players.

Re-exports every public name so existing imports keep working.
New code should import from utils.players directly.
"""
from utils.players import (  # noqa: F401,F403
    _ID_FIELDS,
    _ROLE_POSITIONS,
    _VERIFIED_GIVEN_ALIASES,
    normalize_player_name,
    PlayerIdentityResolver,
    resolve_player_identity,
)

__all__ = ['_ID_FIELDS', '_ROLE_POSITIONS', '_VERIFIED_GIVEN_ALIASES', 'normalize_player_name', 'PlayerIdentityResolver', 'resolve_player_identity']


# --- monkeypatch propagation (see utils/_shim.py) ---
from utils._shim import propagate_sets_to as _propagate_sets_to
import importlib as _importlib
_propagate_sets_to(__name__, _importlib.import_module("utils.players"))
