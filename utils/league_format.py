"""Compatibility shim: utils.league_format now lives in utils.league.

Re-exports every public name so existing imports keep working.
New code should import from utils.league directly.
"""
from utils.league import (  # noqa: F401,F403
    _truthy,
    _norm_type,
    _SNAKE_DRAFT_TYPES,
    _AUCTION_DRAFT_TYPES,
    _draft_rounds,
    _primary_draft,
    is_auction_draft,
    auction_budget,
    is_best_ball,
    _REDRAFT_TYPE_INTS,
    _KEEPER_TYPE_INTS,
    _DYNASTY_TYPE_INTS,
    _REDRAFT_LABELS,
    _KEEPER_LABELS,
    classify_league_roster_format,
    detect_league_format,
)

__all__ = ['_truthy', '_norm_type', '_SNAKE_DRAFT_TYPES', '_AUCTION_DRAFT_TYPES', '_draft_rounds', '_primary_draft', 'is_auction_draft', 'auction_budget', 'is_best_ball', '_REDRAFT_TYPE_INTS', '_KEEPER_TYPE_INTS', '_DYNASTY_TYPE_INTS', '_REDRAFT_LABELS', '_KEEPER_LABELS', 'classify_league_roster_format', 'detect_league_format']


# --- monkeypatch propagation (see utils/_shim.py) ---
from utils._shim import propagate_sets_to as _propagate_sets_to
import importlib as _importlib
_propagate_sets_to(__name__, _importlib.import_module("utils.league"))
