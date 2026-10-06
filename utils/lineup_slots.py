"""Compatibility shim: utils.lineup_slots now lives in utils.lineups.

Re-exports every public name so existing imports keep working.
New code should import from utils.lineups directly.
"""
from utils.lineups import (  # noqa: F401,F403
    FLEX_SLOT_NAMES,
    RB_WR_SLOT_NAMES,
    WR_TE_SLOT_NAMES,
    RB_TE_SLOT_NAMES,
    RESTRICTED_FLEX_SLOTS,
    SUPERFLEX_SLOT_NAMES,
    DEF_SLOT_NAMES,
    SKILL_POSITIONS,
    BENCH_SLOT_NAMES,
    SLOT_ELIGIBILITY,
    normalize_slot_name,
    canonicalize_slot,
    canonicalize_slots,
    count_lineup_slots,
    slot_total,
    slot_eligible_positions,
    flex_count,
    superflex_count,
    restricted_flex_counts,
    is_superflex_lineup,
    is_restricted_flex_slot,
    start_sit_pos,
    start_sit_groups,
    _split_pair,
    starter_need_counts,
)

__all__ = ['FLEX_SLOT_NAMES', 'RB_WR_SLOT_NAMES', 'WR_TE_SLOT_NAMES', 'RB_TE_SLOT_NAMES', 'RESTRICTED_FLEX_SLOTS', 'SUPERFLEX_SLOT_NAMES', 'DEF_SLOT_NAMES', 'SKILL_POSITIONS', 'BENCH_SLOT_NAMES', 'SLOT_ELIGIBILITY', 'normalize_slot_name', 'canonicalize_slot', 'canonicalize_slots', 'count_lineup_slots', 'slot_total', 'slot_eligible_positions', 'flex_count', 'superflex_count', 'restricted_flex_counts', 'is_superflex_lineup', 'is_restricted_flex_slot', 'start_sit_pos', 'start_sit_groups', '_split_pair', 'starter_need_counts']


# --- monkeypatch propagation (see utils/_shim.py) ---
from utils._shim import propagate_sets_to as _propagate_sets_to
import importlib as _importlib
_propagate_sets_to(__name__, _importlib.import_module("utils.lineups"))
