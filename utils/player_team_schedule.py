"""Compatibility shim: utils.player_team_schedule now lives in utils.players.

Re-exports every public name so existing imports keep working.
New code should import from utils.players directly.
"""
from utils.players import (  # noqa: F401,F403
    logger,
    _TEAM_SCHEDULE_CACHE,
    _TEAM_SCHEDULE_TTL,
    _BOX_PAYLOAD_CACHE,
    _BOX_PAYLOAD_TTL_LIVE,
    _BOX_PAYLOAD_TTL_FINAL,
    _BOX_PAYLOAD_TTL_SCHEDULED,
    _POST_WEEK_LABELS,
    _TEAM_CANON,
    _TEAM_ABBR_ALIASES,
    _safe_float,
    _safe_int,
    _block_num,
    _canon,
    _team_keys,
    _teams_match,
    _lookup_team_map,
    tank_boxscore_game_id,
    _fmt_kickoff,
    _fmt_date_label,
    _status_from_game,
    _result_for_team,
    _enrich_from_scores,
    _logo_for,
    _row_from_game,
    _bye_row,
    build_team_schedule,
    resolve_team_for_season,
    _POS_ORDER,
    _player_identity,
    _stat_bundle,
    _has_any,
    _group_columns,
    _cell_value,
    shape_boxscore_payload,
    get_shaped_boxscore,
)

__all__ = ['logger', '_TEAM_SCHEDULE_CACHE', '_TEAM_SCHEDULE_TTL', '_BOX_PAYLOAD_CACHE', '_BOX_PAYLOAD_TTL_LIVE', '_BOX_PAYLOAD_TTL_FINAL', '_BOX_PAYLOAD_TTL_SCHEDULED', '_POST_WEEK_LABELS', '_TEAM_CANON', '_TEAM_ABBR_ALIASES', '_safe_float', '_safe_int', '_block_num', '_canon', '_team_keys', '_teams_match', '_lookup_team_map', 'tank_boxscore_game_id', '_fmt_kickoff', '_fmt_date_label', '_status_from_game', '_result_for_team', '_enrich_from_scores', '_logo_for', '_row_from_game', '_bye_row', 'build_team_schedule', 'resolve_team_for_season', '_POS_ORDER', '_player_identity', '_stat_bundle', '_has_any', '_group_columns', '_cell_value', 'shape_boxscore_payload', 'get_shaped_boxscore']


# --- monkeypatch propagation (see utils/_shim.py) ---
from utils._shim import propagate_sets_to as _propagate_sets_to
import importlib as _importlib
_propagate_sets_to(__name__, _importlib.import_module("utils.players"))
