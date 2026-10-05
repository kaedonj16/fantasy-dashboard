"""Compatibility shim: utils.standings_viz now lives in utils.standings.

Re-exports every public name so existing imports keep working.
New code should import from utils.standings directly.
"""
from utils.standings import (  # noqa: F401,F403
    _LUCK,
    _UNLUCK,
    _NEU,
    _ACCENT,
    _LBL,
    _esc,
    luck_quadrant_svg,
    _median,
    value_age_svg,
)

__all__ = ['_LUCK', '_UNLUCK', '_NEU', '_ACCENT', '_LBL', '_esc', 'luck_quadrant_svg', '_median', 'value_age_svg']


# --- monkeypatch propagation (see utils/_shim.py) ---
from utils._shim import propagate_sets_to as _propagate_sets_to
import importlib as _importlib
_propagate_sets_to(__name__, _importlib.import_module("utils.standings"))
