"""Compatibility shim: utils.waiver_big_game now lives in utils.waivers.

Re-exports every public name so existing imports keep working.
New code should import from utils.waivers directly.
"""
from utils.waivers import (  # noqa: F401,F403
    PositionProductionThresholds,
    POSITION_PRODUCTION_THRESHOLDS,
    DEFAULT_PRODUCTION_THRESHOLDS,
    production_thresholds,
    BigGameConfig,
    CONFIG,
    GameContext,
    BigGameAssessment,
    resolve_expectation,
    absolute_component,
    relative_component,
    performance_surprise,
    _resolve_pts,
    _delta_component,
    _level_component,
    role_sustainability,
    _td_fraction,
    _explosive_fraction,
    _hot_efficiency,
    _fmt_delta,
    classify,
    assess_big_game,
    _basis_phrase,
    discovery_key,
    _STATUS_RANK,
    merge_assessment,
)

__all__ = ['PositionProductionThresholds', 'POSITION_PRODUCTION_THRESHOLDS', 'DEFAULT_PRODUCTION_THRESHOLDS', 'production_thresholds', 'BigGameConfig', 'CONFIG', 'GameContext', 'BigGameAssessment', 'resolve_expectation', 'absolute_component', 'relative_component', 'performance_surprise', '_resolve_pts', '_delta_component', '_level_component', 'role_sustainability', '_td_fraction', '_explosive_fraction', '_hot_efficiency', '_fmt_delta', 'classify', 'assess_big_game', '_basis_phrase', 'discovery_key', '_STATUS_RANK', 'merge_assessment']


# --- monkeypatch propagation (see utils/_shim.py) ---
from utils._shim import propagate_sets_to as _propagate_sets_to
import importlib as _importlib
_propagate_sets_to(__name__, _importlib.import_module("utils.waivers"))
