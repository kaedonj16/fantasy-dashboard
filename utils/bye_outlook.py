"""Compatibility shim: utils.bye_outlook now lives in utils.lineups.

Re-exports every public name so existing imports keep working.
New code should import from utils.lineups directly.
"""
from utils.lineups import (  # noqa: F401,F403
    _SKILL_POSITIONS,
    build_bye_outlook,
    _fmt_pos_counts,
    summarize_bye_outlook,
    trade_bye_coverage_warnings,
)

__all__ = ['_SKILL_POSITIONS', 'build_bye_outlook', '_fmt_pos_counts', 'summarize_bye_outlook', 'trade_bye_coverage_warnings']


# --- monkeypatch propagation (see utils/_shim.py) ---
from utils._shim import propagate_sets_to as _propagate_sets_to
import importlib as _importlib
_propagate_sets_to(__name__, _importlib.import_module("utils.lineups"))
