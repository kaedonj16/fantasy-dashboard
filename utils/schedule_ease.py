"""Compatibility shim: utils.schedule_ease now lives in utils.nfl.

Re-exports every public name so existing imports keep working.
New code should import from utils.nfl directly.
"""
from utils.nfl import (  # noqa: F401,F403
    SCHED_TEAM_ALIAS,
    norm_sched_team,
    sched_rank_color,
    matchup_cell_ease,
)

__all__ = ['SCHED_TEAM_ALIAS', 'norm_sched_team', 'sched_rank_color', 'matchup_cell_ease']


# --- monkeypatch propagation (see utils/_shim.py) ---
from utils._shim import propagate_sets_to as _propagate_sets_to
import importlib as _importlib
_propagate_sets_to(__name__, _importlib.import_module("utils.nfl"))
