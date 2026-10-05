"""Compatibility shim: utils.season_qualification now lives in utils.projections.

Re-exports every public name so existing imports keep working.
New code should import from utils.projections directly.
"""
from utils.projections import (  # noqa: F401,F403
    FULL_GAMES_MIN,
    FULL_VOLUME_MINS,
    _is_regular,
    _is_final,
    completed_regular_season_rounds,
    scaled_minimum,
    player_completed_weeks,
    bulk_player_completed_weeks,
    player_sample_note,
    player_qualification_note,
    QualificationPolicy,
    qualification_policy,
)

__all__ = ['FULL_GAMES_MIN', 'FULL_VOLUME_MINS', '_is_regular', '_is_final', 'completed_regular_season_rounds', 'scaled_minimum', 'player_completed_weeks', 'bulk_player_completed_weeks', 'player_sample_note', 'player_qualification_note', 'QualificationPolicy', 'qualification_policy']


# --- monkeypatch propagation (see utils/_shim.py) ---
from utils._shim import propagate_sets_to as _propagate_sets_to
import importlib as _importlib
_propagate_sets_to(__name__, _importlib.import_module("utils.projections"))
