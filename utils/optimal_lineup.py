"""Pure optimal-lineup solver.

Extracted from app.py so the greedy starter-selection logic can be unit-tested
without importing the full application (pandas / DB) stack.

Given each player's projected/scored points, their positions, and a league's
roster slots, ``compute_optimal_lineup`` returns the set of player ids that
maximizes total points under the slot constraints (single-position slots first,
then restricted flex, then FLEX, then SUPER_FLEX), plus that optimal total.
"""
from __future__ import annotations

from collections import defaultdict

from utils.lineup_slots import (
    DEF_SLOT_NAMES,
    RESTRICTED_FLEX_SLOTS,
    canonicalize_slot,
    slot_eligible_positions,
)

# Fixed, mandatory single-position slots, filled before flex pools.
_SINGLE_SLOTS = ["QB", "RB", "WR", "TE", "K", "DEF", "DL", "LB", "DB"]

# Narrower flex first so a TE cannot steal a WR/RB-only spot.
_FLEX_FILL_ORDER = list(RESTRICTED_FLEX_SLOTS) + ["FLEX", "SUPER_FLEX"]


def compute_optimal_lineup(pts_map, player_positions, roster_positions, all_pids):
    """Greedy optimal-lineup solver.

    Args:
        pts_map: {player_id: points}. Missing/None -> 0.
        player_positions: {player_id: position string}.
        roster_positions: list of slot strings (e.g. ["QB", "RB", "RB", "FLEX"]).
            Provider aliases (OP, WRRB_FLEX, DST, …) are canonicalized.
            Restricted flex (WRRB_FLEX / W/R / WR/TE) stays distinct from FLEX.
        all_pids: iterable of candidate player ids.

    Returns:
        (set_of_optimal_starter_ids, optimal_total_pts).
    """
    slot_counts = defaultdict(int)
    for s in roster_positions:
        canon = canonicalize_slot(s)
        if canon:
            slot_counts[canon] += 1

    by_pos = defaultdict(list)
    for pid in all_pids:
        pos = str(player_positions.get(pid) or "").upper()
        if pos in DEF_SLOT_NAMES:
            pos = "DEF"
        by_pos[pos].append((pid, float(pts_map.get(pid) or 0)))
    for pos in by_pos:
        by_pos[pos].sort(key=lambda x: x[1], reverse=True)

    used, starters = set(), []

    # Fill mandatory single-position slots first
    for pos in _SINGLE_SLOTS:
        n = slot_counts.get(pos, 0)
        for pid, _ in by_pos[pos]:
            if n <= 0:
                break
            if pid not in used:
                starters.append(pid)
                used.add(pid)
                n -= 1

    for slot in _FLEX_FILL_ORDER:
        n = slot_counts.get(slot, 0)
        if n <= 0:
            continue
        eligible = slot_eligible_positions(slot)
        pool = sorted(
            [(pid, pts) for pos in eligible for pid, pts in by_pos[pos] if pid not in used],
            key=lambda x: x[1], reverse=True,
        )
        for pid, _ in pool[:n]:
            starters.append(pid)
            used.add(pid)

    opt_set = set(starters)
    opt_pts = sum(float(pts_map.get(pid) or 0) for pid in opt_set)
    return opt_set, round(opt_pts, 2)
