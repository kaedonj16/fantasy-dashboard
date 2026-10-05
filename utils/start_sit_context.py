"""Compatibility shim: utils.start_sit_context now lives in utils.start_sit.

Re-exports every public name so existing imports keep working.
New code should import from utils.start_sit directly.
"""
from utils.start_sit import (  # noqa: F401,F403
    _ABSENCE_SKILL_POS,
    _ABSENCE_DEF_POS,
    _OL_POSITIONS,
    _ABSENCE_STATUSES,
    _ABSENCE_STATUS_LABEL,
    _absence_entry,
    productive_pids_from_weekly_points,
    starting_lineman_pids,
    build_absence_index,
    absence_notes,
    expected_plays_context,
    role_confidence_from_trend,
)

__all__ = ['_ABSENCE_SKILL_POS', '_ABSENCE_DEF_POS', '_OL_POSITIONS', '_ABSENCE_STATUSES', '_ABSENCE_STATUS_LABEL', '_absence_entry', 'productive_pids_from_weekly_points', 'starting_lineman_pids', 'build_absence_index', 'absence_notes', 'expected_plays_context', 'role_confidence_from_trend']


# --- monkeypatch propagation (see utils/_shim.py) ---
from utils._shim import propagate_sets_to as _propagate_sets_to
import importlib as _importlib
_propagate_sets_to(__name__, _importlib.import_module("utils.start_sit"))
