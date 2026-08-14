"""Canonical, exact fantasy-lineup assignment.

Provider slot names are normalized here, at the boundary.  The solver uses a
maximum-weight bipartite assignment (dynamic programming over lineup slots), so
restricted flexes and multi-position players cannot be mishandled by greedy
"fixed slots first" selection.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Iterable, Mapping

BENCH_SLOTS = frozenset({"BN", "BENCH", "IR", "RESERVE", "TAXI"})
SKILL = frozenset({"QB", "RB", "WR", "TE"})

_ALIASES = {
    "SF": "SUPER_FLEX", "SFLEX": "SUPER_FLEX", "SUPERFLEX": "SUPER_FLEX",
    "SUPER FLEX": "SUPER_FLEX", "OP": "SUPER_FLEX", "QB_RB_WR_TE": "SUPER_FLEX",
    "Q_RB_WR_TE": "SUPER_FLEX", "FLEX": "FLEX", "RB_WR_TE": "FLEX",
    "RBWRTE": "FLEX", "W_R_T": "FLEX", "WRRB_FLEX": "WR_RB_FLEX",
    "RB_WR_FLEX": "WR_RB_FLEX", "RBWR": "WR_RB_FLEX", "WR_RB": "WR_RB_FLEX",
    "WRTE_FLEX": "WR_TE_FLEX", "WR_TE": "WR_TE_FLEX", "REC_FLEX": "WR_TE_FLEX",
    "DST": "DEF", "D/ST": "DEF",
}

_ELIGIBILITY = {
    "FLEX": frozenset({"RB", "WR", "TE"}),
    "WR_RB_FLEX": frozenset({"WR", "RB"}),
    "WR_TE_FLEX": frozenset({"WR", "TE"}),
    "SUPER_FLEX": SKILL,
}


def normalize_slot(slot: object) -> str:
    raw = str(slot or "").strip().upper().replace("-", "_")
    return _ALIASES.get(raw, raw)


def slot_eligibility(slot: object) -> frozenset[str]:
    """Return the explicit position set for one normalized starting slot."""
    normalized = normalize_slot(slot)
    if normalized in BENCH_SLOTS:
        return frozenset()
    return _ELIGIBILITY.get(normalized, frozenset({normalized}) if normalized else frozenset())


def _positions(value: object) -> frozenset[str]:
    if isinstance(value, str):
        parts = value.replace("/", ",").split(",")
    elif isinstance(value, Iterable):
        parts = value
    else:
        parts = ()
    return frozenset(str(p).strip().upper() for p in parts if str(p).strip())


def compute_optimal_lineup(
    pts_map: Mapping, player_positions: Mapping, roster_positions: Iterable, all_pids: Iterable,
):
    """Return ``(starter_ids, total)`` for the exact maximum-value assignment.

    Values may be projections, market values, or realized scores. Missing and
    non-finite values are conservatively treated as zero.
    """
    slots = [slot_eligibility(s) for s in roster_positions]
    slots = [s for s in slots if s]
    players = []
    for pid in dict.fromkeys(all_pids):
        try:
            score = float(pts_map.get(pid) or 0.0)
            if score != score:  # NaN
                score = 0.0
        except (TypeError, ValueError):
            score = 0.0
        players.append((pid, _positions(player_positions.get(pid)), score))

    # Most restricted slots first reduces the memoized state space. Index is a
    # stable tie-break, making equal-value results deterministic.
    slots.sort(key=lambda eligible: (len(eligible), tuple(sorted(eligible))))

    @lru_cache(maxsize=None)
    def solve(player_i: int, slot_mask: int):
        if player_i == len(players):
            return 0.0, ()
        pid, positions, score = players[player_i]
        best = solve(player_i + 1, slot_mask)  # bench this player
        for slot_i, eligible in enumerate(slots):
            if slot_mask & (1 << slot_i) or not positions.intersection(eligible):
                continue
            tail_score, tail_ids = solve(player_i + 1, slot_mask | (1 << slot_i))
            candidate = (score + tail_score, (pid,) + tail_ids)
            if candidate[0] > best[0]:
                best = candidate
        return best

    total, ids = solve(0, 0)
    return set(ids), round(total, 2)
