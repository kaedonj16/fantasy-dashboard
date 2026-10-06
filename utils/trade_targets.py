"""Compatibility shim: utils.trade_targets now lives in utils.trade.

Re-exports every public name so existing imports keep working.
New code should import from utils.trade directly.
"""
from utils.trade import (  # noqa: F401,F403
    POSITIONS,
    PEAK_AGE,
    NEED_RANK_FRACTION,
    QUALITY_STARTER_MULT,
    _HOLE_FILL_FRAC,
    _MAX_PICK_IN_CEILING,
    MAX_TARGETS,
    MAX_PER_POS_HARD,
    MAX_PER_POS_SOFT,
    _MIN_ASSET,
    one_for_one_chip,
    package_ceiling,
    affordability_multiplier,
    infer_roster_window,
    age_fit_multiplier,
    availability_multiplier,
    _STRENGTH_PAD,
    _padded_vals,
    strength_gain,
    classify_position_needs,
    detect_needed_positions,
    detect_surplus_positions,
    complementary_multiplier,
    need_summary,
    annotate_owner_depth,
    _fill_bar,
    fit_reason,
    _candidate_score,
    rank_position_candidates,
    select_trade_targets,
)

__all__ = ['POSITIONS', 'PEAK_AGE', 'NEED_RANK_FRACTION', 'QUALITY_STARTER_MULT', '_HOLE_FILL_FRAC', '_MAX_PICK_IN_CEILING', 'MAX_TARGETS', 'MAX_PER_POS_HARD', 'MAX_PER_POS_SOFT', '_MIN_ASSET', 'one_for_one_chip', 'package_ceiling', 'affordability_multiplier', 'infer_roster_window', 'age_fit_multiplier', 'availability_multiplier', '_STRENGTH_PAD', '_padded_vals', 'strength_gain', 'classify_position_needs', 'detect_needed_positions', 'detect_surplus_positions', 'complementary_multiplier', 'need_summary', 'annotate_owner_depth', '_fill_bar', 'fit_reason', '_candidate_score', 'rank_position_candidates', 'select_trade_targets']


# --- monkeypatch propagation (see utils/_shim.py) ---
from utils._shim import propagate_sets_to as _propagate_sets_to
import importlib as _importlib
_propagate_sets_to(__name__, _importlib.import_module("utils.trade"))
