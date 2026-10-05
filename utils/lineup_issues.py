"""Compatibility shim: utils.lineup_issues now lives in utils.lineups.

Re-exports every public name so existing imports keep working.
New code should import from utils.lineups directly.
"""
from utils.lineups import (  # noqa: F401,F403
    SERIOUS_INJURY_STATUSES,
    SWAP_EXCLUDED_STATUSES,
    EMPTY_SLOT_IDS,
    locked_teams_from_games,
    locked_teams_for_week,
    find_lineup_issues,
    projection_upgrades,
    format_lineup_lock_swap,
    format_lineup_lock_swaps,
    pair_start_sit_swaps,
    summarize_issues,
)

__all__ = ['SERIOUS_INJURY_STATUSES', 'SWAP_EXCLUDED_STATUSES', 'EMPTY_SLOT_IDS', 'locked_teams_from_games', 'locked_teams_for_week', 'find_lineup_issues', 'projection_upgrades', 'format_lineup_lock_swap', 'format_lineup_lock_swaps', 'pair_start_sit_swaps', 'summarize_issues']


# --- monkeypatch propagation (see utils/_shim.py) ---
from utils._shim import propagate_sets_to as _propagate_sets_to
import importlib as _importlib
_propagate_sets_to(__name__, _importlib.import_module("utils.lineups"))
