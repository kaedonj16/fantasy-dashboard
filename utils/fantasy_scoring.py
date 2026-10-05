"""Compatibility shim: utils.fantasy_scoring now lives in utils.projections.

Re-exports every public name so existing imports keep working.
New code should import from utils.projections directly.
"""
from utils.projections import (  # noqa: F401,F403
    _DEFAULT_RATES,
    completed_points_summary,
    _rate,
    score_stats,
    _WEEK_STATS_TO_SLEEPER,
    week_stats_line_points,
    _sleeper_standard_points,
    projection_points,
    week_stat_points,
    weekly_projection_points,
)

__all__ = ['_DEFAULT_RATES', 'completed_points_summary', '_rate', 'score_stats', '_WEEK_STATS_TO_SLEEPER', 'week_stats_line_points', '_sleeper_standard_points', 'projection_points', 'week_stat_points', 'weekly_projection_points']


# --- monkeypatch propagation (see utils/_shim.py) ---
from utils._shim import propagate_sets_to as _propagate_sets_to
import importlib as _importlib
_propagate_sets_to(__name__, _importlib.import_module("utils.projections"))
