"""Compatibility shim: utils.cross_league_actions now lives in utils.digest.

Re-exports every public name so existing imports keep working.
New code should import from utils.digest directly.
"""
from utils.digest import (  # noqa: F401,F403
    _PRIORITY,
    action_priority,
    make_action,
    rank_cross_league_actions,
    lineup_actions_from_issues,
    injury_stash_action,
    WAIVER_MIN_VALUE_MULT,
    WAIVER_RANK_CEILING_12,
    WAIVER_SF_QB_CEILING_12,
    WAIVER_VETERAN_RANK_CEILING_12,
    _SKILL_POS,
    _OUT_STATUS,
    _FA_TEAMS,
    _POS_RANK_RE,
    waiver_value_threshold,
    _clamp_league_size,
    waiver_rank_ceiling,
    parse_pos_rank,
    waiver_add_clears_quality_bar,
    waiver_add_detail,
    _waiver_need_context,
    select_waiver_add,
    waiver_pickup_action,
    _ROSTER_ISSUE_RANK,
    _ROSTER_ISSUE_TITLE,
    roster_slot_action,
    calendar_action,
)

__all__ = ['_PRIORITY', 'action_priority', 'make_action', 'rank_cross_league_actions', 'lineup_actions_from_issues', 'injury_stash_action', 'WAIVER_MIN_VALUE_MULT', 'WAIVER_RANK_CEILING_12', 'WAIVER_SF_QB_CEILING_12', 'WAIVER_VETERAN_RANK_CEILING_12', '_SKILL_POS', '_OUT_STATUS', '_FA_TEAMS', '_POS_RANK_RE', 'waiver_value_threshold', '_clamp_league_size', 'waiver_rank_ceiling', 'parse_pos_rank', 'waiver_add_clears_quality_bar', 'waiver_add_detail', '_waiver_need_context', 'select_waiver_add', 'waiver_pickup_action', '_ROSTER_ISSUE_RANK', '_ROSTER_ISSUE_TITLE', 'roster_slot_action', 'calendar_action']


# --- monkeypatch propagation (see utils/_shim.py) ---
from utils._shim import propagate_sets_to as _propagate_sets_to
import importlib as _importlib
_propagate_sets_to(__name__, _importlib.import_module("utils.digest"))
