"""Compatibility shim: utils.waiver_score now lives in utils.waivers.

Re-exports every public name so existing imports keep working.
New code should import from utils.waivers directly.
"""
from utils.waivers import (  # noqa: F401,F403
    WAIVER_PRIME_MAX,
    USAGE_SPIKE_MIN,
    VACANCY_SEVERITY,
    VACANCY_STRONG,
    INJURY_DURATION_WEEKS,
    _OUT_STATUSES,
    SERIOUS_INJURY_STATUSES,
    is_seriously_hurt,
    waiver_injury_note,
    drop_seriously_hurt,
    WaiverWeights,
    WEIGHTS,
    _pos_depth,
    _discounted_weeks,
    value_component,
    projection_component,
    usage_ratio,
    build_depth_index,
    _will_play,
    depth_analysis,
    depth_analysis_for_player,
    injured_ahead,
    injured_ahead_for_player,
    _proximity_weight,
    strip_bye_weeks,
    weeks_out_from_projections,
    WEEK_MISSING,
    WEEK_ZERO,
    WEEK_BYE,
    WEEK_OUT,
    WEEK_PLAYING,
    classify_projection_week,
    return_timeline,
    expected_vacated_points,
    depth_chart_vacancy_score,
    self_injury_multiplier,
    blended_trend,
    adaptive_trend_thresholds,
    schedule_bonus,
    schedule_urgency,
    roster_needs_drop,
    pick_waiver_push_candidate,
    waiver_push_copy,
    FAAB_SCORE_LOW,
    FAAB_SCORE_HIGH,
    _FAAB_PHASE_MULT,
    faab_intensity,
    _faab_pct_bands,
    _faab_rationale,
    faab_bid_bands,
    faab_recommendation,
    replacement_levels,
    scarcity_multiplier,
    waiver_pickup_score,
    waiver_signal,
    positional_need_scores,
    need_multiplier,
    HORIZONS,
    horizon_weights,
    BREAKOUT_FLOOR,
    credible_opportunity,
    passes_candidate_floor,
)

__all__ = ['WAIVER_PRIME_MAX', 'USAGE_SPIKE_MIN', 'VACANCY_SEVERITY', 'VACANCY_STRONG', 'INJURY_DURATION_WEEKS', '_OUT_STATUSES', 'SERIOUS_INJURY_STATUSES', 'is_seriously_hurt', 'waiver_injury_note', 'drop_seriously_hurt', 'WaiverWeights', 'WEIGHTS', '_pos_depth', '_discounted_weeks', 'value_component', 'projection_component', 'usage_ratio', 'build_depth_index', '_will_play', 'depth_analysis', 'depth_analysis_for_player', 'injured_ahead', 'injured_ahead_for_player', '_proximity_weight', 'strip_bye_weeks', 'weeks_out_from_projections', 'WEEK_MISSING', 'WEEK_ZERO', 'WEEK_BYE', 'WEEK_OUT', 'WEEK_PLAYING', 'classify_projection_week', 'return_timeline', 'expected_vacated_points', 'depth_chart_vacancy_score', 'self_injury_multiplier', 'blended_trend', 'adaptive_trend_thresholds', 'schedule_bonus', 'schedule_urgency', 'roster_needs_drop', 'pick_waiver_push_candidate', 'waiver_push_copy', 'FAAB_SCORE_LOW', 'FAAB_SCORE_HIGH', '_FAAB_PHASE_MULT', 'faab_intensity', '_faab_pct_bands', '_faab_rationale', 'faab_bid_bands', 'faab_recommendation', 'replacement_levels', 'scarcity_multiplier', 'waiver_pickup_score', 'waiver_signal', 'positional_need_scores', 'need_multiplier', 'HORIZONS', 'horizon_weights', 'BREAKOUT_FLOOR', 'credible_opportunity', 'passes_candidate_floor']


# --- monkeypatch propagation (see utils/_shim.py) ---
from utils._shim import propagate_sets_to as _propagate_sets_to
import importlib as _importlib
_propagate_sets_to(__name__, _importlib.import_module("utils.waivers"))
