"""Compatibility shim: utils.draft_grade now lives in utils.draft.

Re-exports every public name so existing imports keep working.
New code should import from utils.draft directly.
"""
from utils.draft import (  # noqa: F401,F403
    clamp01,
    dr_grade_letter,
    dr_letter_to_score,
    dr_rookie_team_score,
    dr_slot_eligible,
    _FLEX_COVERS,
    _has_flex_for,
    dr_lineup_score,
    dr_optimal_lineup,
    dr_avg_top_n,
    dr_league_lineup_avg,
    dr_starter_metric_avg,
    dr_peer_starter_avg,
    dr_weighted_pick_score,
    dr_peer_value_ps,
    dr_resolve_strength_baseline,
    DR_SPLIT_STARTUP,
    DR_SPLIT_REDRAFT,
    DR_CONSTRUCTION_STARTUP,
    DR_CONSTRUCTION_REDRAFT,
    dr_grade_split,
    dr_construction_mix,
    dr_team_grade_score,
    dr_apply_field_curve,
)

__all__ = ['clamp01', 'dr_grade_letter', 'dr_letter_to_score', 'dr_rookie_team_score', 'dr_slot_eligible', '_FLEX_COVERS', '_has_flex_for', 'dr_lineup_score', 'dr_optimal_lineup', 'dr_avg_top_n', 'dr_league_lineup_avg', 'dr_starter_metric_avg', 'dr_peer_starter_avg', 'dr_weighted_pick_score', 'dr_peer_value_ps', 'dr_resolve_strength_baseline', 'DR_SPLIT_STARTUP', 'DR_SPLIT_REDRAFT', 'DR_CONSTRUCTION_STARTUP', 'DR_CONSTRUCTION_REDRAFT', 'dr_grade_split', 'dr_construction_mix', 'dr_team_grade_score', 'dr_apply_field_curve']


# --- monkeypatch propagation (see utils/_shim.py) ---
from utils._shim import propagate_sets_to as _propagate_sets_to
import importlib as _importlib
_propagate_sets_to(__name__, _importlib.import_module("utils.draft"))
