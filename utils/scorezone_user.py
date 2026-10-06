"""Compatibility shim: utils.scorezone_user now lives in utils.scorezone.

Re-exports every public name so existing imports keep working.
New code should import from utils.scorezone directly.
"""
from utils.scorezone import (  # noqa: F401,F403
    MAX_USER_LEAGUES,
    owner_id_variants,
    match_viewer_roster,
    portfolio_from_account_leagues,
    portfolio_from_sleeper_leagues,
    resolve_portfolio_viewer_roster,
)

__all__ = ['MAX_USER_LEAGUES', 'owner_id_variants', 'match_viewer_roster', 'portfolio_from_account_leagues', 'portfolio_from_sleeper_leagues', 'resolve_portfolio_viewer_roster']


# --- monkeypatch propagation (see utils/_shim.py) ---
from utils._shim import propagate_sets_to as _propagate_sets_to
import importlib as _importlib
_propagate_sets_to(__name__, _importlib.import_module("utils.scorezone"))
