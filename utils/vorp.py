"""Compatibility shim: utils.vorp now lives in utils.trade.

Re-exports every public name so existing imports keep working.
New code should import from utils.trade directly.
"""
from utils.trade import (  # noqa: F401,F403
    VALUE_STARTERS,
    VALUE_FLEX_ALLOC,
    POINTS_PER_WIN_DEFAULT,
    PROJ_SEASON_GAMES,
    _POS_NORM,
    normalize_position,
    points_per_win,
    stamp_value_metrics,
    projected_season_pts,
    projected_vorp_map,
)

__all__ = ['VALUE_STARTERS', 'VALUE_FLEX_ALLOC', 'POINTS_PER_WIN_DEFAULT', 'PROJ_SEASON_GAMES', '_POS_NORM', 'normalize_position', 'points_per_win', 'stamp_value_metrics', 'projected_season_pts', 'projected_vorp_map']


# --- monkeypatch propagation (see utils/_shim.py) ---
from utils._shim import propagate_sets_to as _propagate_sets_to
import importlib as _importlib
_propagate_sets_to(__name__, _importlib.import_module("utils.trade"))
