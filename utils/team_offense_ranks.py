"""Compatibility shim: utils.team_offense_ranks now lives in utils.nfl.

Re-exports every public name so existing imports keep working.
New code should import from utils.nfl directly.
"""
from utils.nfl import (  # noqa: F401,F403
    STAT_KEYS,
    PROJECTION_DIVISOR,
    _STAT_KEY_ALIASES,
    _stat_value,
    REG_WEEKS,
    canon_team,
    _row_season_week,
    _row_scores,
    aggregate_completed_games,
    fully_completed_weeks,
    aggregate_sleeper_team_weeks,
    aggregate_sleeper_team_weeks_for_teams,
    read_csv_team_totals,
    _per_game,
    build_offense_table,
    compute_team_offense,
    competition_ranks,
    RANK_METRICS,
    rank_offense_table,
    ranked_metric,
)

__all__ = ['STAT_KEYS', 'PROJECTION_DIVISOR', '_STAT_KEY_ALIASES', '_stat_value', 'REG_WEEKS', 'canon_team', '_row_season_week', '_row_scores', 'aggregate_completed_games', 'fully_completed_weeks', 'aggregate_sleeper_team_weeks', 'aggregate_sleeper_team_weeks_for_teams', 'read_csv_team_totals', '_per_game', 'build_offense_table', 'compute_team_offense', 'competition_ranks', 'RANK_METRICS', 'rank_offense_table', 'ranked_metric']


# --- monkeypatch propagation (see utils/_shim.py) ---
from utils._shim import propagate_sets_to as _propagate_sets_to
import importlib as _importlib
_propagate_sets_to(__name__, _importlib.import_module("utils.nfl"))
