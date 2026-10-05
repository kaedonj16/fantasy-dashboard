"""Compatibility shim: utils.league_payload now lives in utils.league.

Re-exports every public name so existing imports keep working.
New code should import from utils.league directly.
"""
from utils.league import (  # noqa: F401,F403
    format_sleeper_league_option,
    get_most_recent_valid_draft_for_season,
    build_roster_map,
    _FILLED_ROSTER_MIN_PLAYERS,
    _LIVE_DRAFT_STATUSES,
    _INCOMPLETE_DRAFT_STATUSES,
    _RAW_INCOMPLETE_DRAFT_STATUSES,
    _REDRAFT_KEEPER_TYPES,
    _REDRAFT_KEEPER_LABELS,
    _norm_status,
    _as_epoch_ms,
    _is_known_redraft_or_keeper,
    _looks_dynasty,
    _explicit_startup_incomplete,
    rosters_look_undrafted,
    draft_start_ms,
    startup_draft_phase,
    startup_draft_pending,
    show_matchup_preview,
    draft_countdown_copy,
    top_board_preview,
)

__all__ = ['format_sleeper_league_option', 'get_most_recent_valid_draft_for_season', 'build_roster_map', '_FILLED_ROSTER_MIN_PLAYERS', '_LIVE_DRAFT_STATUSES', '_INCOMPLETE_DRAFT_STATUSES', '_RAW_INCOMPLETE_DRAFT_STATUSES', '_REDRAFT_KEEPER_TYPES', '_REDRAFT_KEEPER_LABELS', '_norm_status', '_as_epoch_ms', '_is_known_redraft_or_keeper', '_looks_dynasty', '_explicit_startup_incomplete', 'rosters_look_undrafted', 'draft_start_ms', 'startup_draft_phase', 'startup_draft_pending', 'show_matchup_preview', 'draft_countdown_copy', 'top_board_preview']


# --- monkeypatch propagation (see utils/_shim.py) ---
from utils._shim import propagate_sets_to as _propagate_sets_to
import importlib as _importlib
_propagate_sets_to(__name__, _importlib.import_module("utils.league"))
