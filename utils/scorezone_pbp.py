"""Compatibility shim: utils.scorezone_pbp now lives in utils.scorezone.

Re-exports every public name so existing imports keep working.
New code should import from utils.scorezone directly.
"""
from utils.scorezone import (  # noqa: F401,F403
    _s,
    _normalize_name,
    _extract_first_initial_last,
    _resolve_player_name,
    PLAY_STATE_VALID,
    PLAY_STATE_NO_PLAY,
    PLAY_STATE_NULLIFIED,
    PLAY_STATE_OVERTURNED,
    PLAY_STATE_CORRECTED,
    _detect_play_state,
    _is_no_play,
    _extract_target_from_text,
    _TWO_POINT_MARKER,
    _TWO_POINT_RESULT,
    _2PT_NAME,
    _2PT_PASS,
    _2PT_RUSH_ACTION,
    _2PT_RUSH,
    _two_point_segments,
    two_point_attempted,
    two_point_succeeded,
    parse_two_point_conversion,
    _CONV_ROLE_TO_IDENTITY_ROLE,
    _extract_rusher_from_text,
    _opponent_team,
    _first,
    _play_text,
    _fumble_lost_fumbler,
    _stat_line_nonzero,
    _normalize_player_delta,
    _iter_player_stats,
    _raw_pbp_list,
    extract_pbp_plays,
    game_situation_from_plays,
    pbp_boxscore_mismatches,
    normalize_nfl_game_status,
    build_games_snapshot,
    demo_play_text,
)

__all__ = ['_s', '_normalize_name', '_extract_first_initial_last', '_resolve_player_name', 'PLAY_STATE_VALID', 'PLAY_STATE_NO_PLAY', 'PLAY_STATE_NULLIFIED', 'PLAY_STATE_OVERTURNED', 'PLAY_STATE_CORRECTED', '_detect_play_state', '_is_no_play', '_extract_target_from_text', '_TWO_POINT_MARKER', '_TWO_POINT_RESULT', '_2PT_NAME', '_2PT_PASS', '_2PT_RUSH_ACTION', '_2PT_RUSH', '_two_point_segments', 'two_point_attempted', 'two_point_succeeded', 'parse_two_point_conversion', '_CONV_ROLE_TO_IDENTITY_ROLE', '_extract_rusher_from_text', '_opponent_team', '_first', '_play_text', '_fumble_lost_fumbler', '_stat_line_nonzero', '_normalize_player_delta', '_iter_player_stats', '_raw_pbp_list', 'extract_pbp_plays', 'game_situation_from_plays', 'pbp_boxscore_mismatches', 'normalize_nfl_game_status', 'build_games_snapshot', 'demo_play_text']


# --- monkeypatch propagation (see utils/_shim.py) ---
from utils._shim import propagate_sets_to as _propagate_sets_to
import importlib as _importlib
_propagate_sets_to(__name__, _importlib.import_module("utils.scorezone"))
