"""Compatibility shim: utils.scorezone_stats now lives in utils.scorezone.

Re-exports every public name so existing imports keep working.
New code should import from utils.scorezone directly.
"""
from utils.scorezone import (  # noqa: F401,F403
    rz_num,
    rz_safe_epoch,
    _pick,
    rz_stat_line_from_ps,
    resolve_boxscore_player_stats,
    rz_def_stat_line,
)

__all__ = ['rz_num', 'rz_safe_epoch', '_pick', 'rz_stat_line_from_ps', 'resolve_boxscore_player_stats', 'rz_def_stat_line']


# --- monkeypatch propagation (see utils/_shim.py) ---
from utils._shim import propagate_sets_to as _propagate_sets_to
import importlib as _importlib
_propagate_sets_to(__name__, _importlib.import_module("utils.scorezone"))
