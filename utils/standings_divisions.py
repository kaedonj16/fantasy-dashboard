"""Compatibility shim: utils.standings_divisions now lives in utils.standings.

Re-exports every public name so existing imports keep working.
New code should import from utils.standings directly.
"""
from utils.standings import (  # noqa: F401,F403
    logger,
    _LAST_GOOD_DIV_MAP,
    _divisions_configured,
    roster_division_map,
    div_map_for_ctx,
    division_name_map,
    active_divisions,
    resolve_divisions,
    division_win_pct,
    playoff_seed_order,
    assign_playoff_seeds,
    _norm_rid,
    division_records,
    format_record,
    format_record_html,
    division_records_for_ctx,
    is_division_game,
)

__all__ = ['logger', '_LAST_GOOD_DIV_MAP', '_divisions_configured', 'roster_division_map', 'div_map_for_ctx', 'division_name_map', 'active_divisions', 'resolve_divisions', 'division_win_pct', 'playoff_seed_order', 'assign_playoff_seeds', '_norm_rid', 'division_records', 'format_record', 'format_record_html', 'division_records_for_ctx', 'is_division_game']


# --- monkeypatch propagation (see utils/_shim.py) ---
from utils._shim import propagate_sets_to as _propagate_sets_to
import importlib as _importlib
_propagate_sets_to(__name__, _importlib.import_module("utils.standings"))
