"""Compatibility shim: utils.defense_vs_position now lives in utils.nfl.

Re-exports every public name so existing imports keep working.
New code should import from utils.nfl directly.
"""
from utils.nfl import (  # noqa: F401,F403
    POSITIONS,
    REG_WEEKS,
    EFF_LABELS,
    canon_team,
    _as_float,
    _row_scores,
    completed_defense_games,
    table_fingerprint,
    _blank_pos_totals,
    aggregate_defense_stats,
    _efficiency,
    _competition_ranks,
    build_defense_vs_position,
)

__all__ = ['POSITIONS', 'REG_WEEKS', 'EFF_LABELS', 'canon_team', '_as_float', '_row_scores', 'completed_defense_games', 'table_fingerprint', '_blank_pos_totals', 'aggregate_defense_stats', '_efficiency', '_competition_ranks', 'build_defense_vs_position']


# --- monkeypatch propagation (see utils/_shim.py) ---
from utils._shim import propagate_sets_to as _propagate_sets_to
import importlib as _importlib
_propagate_sets_to(__name__, _importlib.import_module("utils.nfl"))
