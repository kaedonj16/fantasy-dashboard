"""Compatibility shim: utils.start_sit_score now lives in utils.start_sit.

Re-exports every public name so existing imports keep working.
New code should import from utils.start_sit directly.
"""
from utils.start_sit import (  # noqa: F401,F403
    _WEATHER_MULT,
    _WEATHER_DEFAULT,
    _neutral_factors,
    _weather_mult,
    _lerp_clamped,
    _vegas_mult,
    ELITE_FORM_FLOOR_PPG,
    FORM_BLEND_WEIGHT,
    FORM_BLEND_WINDOW,
    FORM_SCALES,
    PROJ_CAP_RATIO,
    RETURN_DISCOUNT_LADDER,
    USAGE_SCALES,
    USAGE_STAT_KEYS,
    START_SIT_POOL_EXCLUDED_STATUSES,
    bottom_teams_by_implied_total,
    cap_projection,
    compute_start_score,
    form_usage_inputs,
    lead_adjusted_last4_ppg,
    lead_adjusted_score_inputs,
    lead_filtered_last4_ppg,
    likely_range,
    return_adjusted_projection,
    return_discount_multiplier,
    snap_share_trend,
    start_sit_pool_excluded,
    usage_value_from_stats,
)

__all__ = ['_WEATHER_MULT', '_WEATHER_DEFAULT', '_neutral_factors', '_weather_mult', '_lerp_clamped', '_vegas_mult', 'ELITE_FORM_FLOOR_PPG', 'FORM_BLEND_WEIGHT', 'FORM_BLEND_WINDOW', 'FORM_SCALES', 'PROJ_CAP_RATIO', 'RETURN_DISCOUNT_LADDER', 'USAGE_SCALES', 'USAGE_STAT_KEYS', 'START_SIT_POOL_EXCLUDED_STATUSES', 'bottom_teams_by_implied_total', 'cap_projection', 'compute_start_score', 'form_usage_inputs', 'lead_adjusted_last4_ppg', 'lead_adjusted_score_inputs', 'lead_filtered_last4_ppg', 'likely_range', 'return_adjusted_projection', 'return_discount_multiplier', 'start_sit_pool_excluded', 'usage_value_from_stats']


# --- monkeypatch propagation (see utils/_shim.py) ---
from utils._shim import propagate_sets_to as _propagate_sets_to
import importlib as _importlib
_propagate_sets_to(__name__, _importlib.import_module("utils.start_sit"))
