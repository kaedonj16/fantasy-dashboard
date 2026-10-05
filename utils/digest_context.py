"""Compatibility shim: utils.digest_context now lives in utils.digest.

Re-exports every public name so existing imports keep working.
New code should import from utils.digest directly.
"""
from utils.digest import (  # noqa: F401,F403
    logger,
    DYNASTY_MOVE_MIN,
    LEAGUEWIDE_MOVE_MIN,
    uses_long_term_value,
    _FAILED,
    DigestRunCache,
    _load_league_bundle,
    in_season,
    team_display_name,
    value_column,
    player_value,
    filter_movers,
    mover_notes,
    matchup_for_roster,
    _sum_proj,
    _win_prob_from_starters,
    trade_insight_for_roster,
    breakout_for_roster,
    roster_core,
    _name,
)

__all__ = ['logger', 'DYNASTY_MOVE_MIN', 'LEAGUEWIDE_MOVE_MIN', 'uses_long_term_value', '_FAILED', 'DigestRunCache', '_load_league_bundle', 'in_season', 'team_display_name', 'value_column', 'player_value', 'filter_movers', 'mover_notes', 'matchup_for_roster', '_sum_proj', '_win_prob_from_starters', 'trade_insight_for_roster', 'breakout_for_roster', 'roster_core', '_name']


# --- monkeypatch propagation (see utils/_shim.py) ---
from utils._shim import propagate_sets_to as _propagate_sets_to
import importlib as _importlib
_propagate_sets_to(__name__, _importlib.import_module("utils.digest"))
