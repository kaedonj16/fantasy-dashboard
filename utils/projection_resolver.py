"""Compatibility shim: utils.projection_resolver now lives in utils.projections.

Re-exports every public name so existing imports keep working.
New code should import from utils.projections directly.
"""
from utils.projections import (  # noqa: F401,F403
    PROJECTION_CACHE_VERSION,
    SEASON_AVERAGE,
    WEEKLY,
    POINTS_PER_GAME,
    POINTS,
    _LOG,
    _MAX_PLAUSIBLE_PPG,
    _DEFAULT_MAX_PLAUSIBLE_PPG,
    scoring_fingerprint,
    projection_cache_key,
    ProjectionResult,
    _positive,
    _valid_ppg,
    _sleeper_week_value,
    _season_total_projection,
    resolve_projected_ppg,
    resolve_projected_ppg_many,
)

__all__ = ['PROJECTION_CACHE_VERSION', 'SEASON_AVERAGE', 'WEEKLY', 'POINTS_PER_GAME', 'POINTS', '_LOG', '_MAX_PLAUSIBLE_PPG', '_DEFAULT_MAX_PLAUSIBLE_PPG', 'scoring_fingerprint', 'projection_cache_key', 'ProjectionResult', '_positive', '_valid_ppg', '_sleeper_week_value', '_season_total_projection', 'resolve_projected_ppg', 'resolve_projected_ppg_many']


# --- monkeypatch propagation (see utils/_shim.py) ---
from utils._shim import propagate_sets_to as _propagate_sets_to
import importlib as _importlib
_propagate_sets_to(__name__, _importlib.import_module("utils.projections"))
