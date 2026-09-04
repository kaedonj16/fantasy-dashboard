"""Derive a legal starting lineup when a provider has no live starter flags.

MFL roster exports do not include the current lineup. Weekly results do
when a week has been scored. Offseason and pre-week boards still need a
slot-legal starter list so Start/Sit and Optimal Lineup have something
to compare against.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence

from utils.lineup_slots import BENCH_SLOT_NAMES, canonicalize_slot, canonicalize_slots


def _grade_slot(slot: str) -> str:
    """Map a canonical slot onto the names ``dr_slot_eligible`` understands."""
    s = canonicalize_slot(slot)
    if s == "SUPER_FLEX":
        return "SF"
    return s


def starter_slots(roster_positions: Optional[Iterable]) -> List[str]:
    """Starting slots only (no bench / IR / taxi)."""
    out: List[str] = []
    for slot in canonicalize_slots(roster_positions):
        if slot in BENCH_SLOT_NAMES:
            continue
        out.append(_grade_slot(slot))
    return out


def derive_starters_from_slots(
    player_ids: Sequence[str],
    roster_positions: Optional[Iterable],
    pos_by_pid: Optional[Dict[str, str]] = None,
    score_by_pid: Optional[Dict[str, float]] = None,
) -> List[str]:
    """Fill starter slots greedily from ``player_ids``.

    Uses the same most-restrictive-slot-first fill as draft grades. Scores
    default to 0 so position eligibility alone decides when projections
    are missing. Returns starter ids in slot-fill order (not roster order).
    """
    slots = starter_slots(roster_positions)
    if not slots or not player_ids:
        return []
    pos_map = pos_by_pid or {}
    score_map = score_by_pid or {}
    players = []
    seen: set[str] = set()
    for raw in player_ids:
        pid = str(raw or "").strip()
        if not pid or pid in seen:
            continue
        seen.add(pid)
        pos = str(pos_map.get(pid) or "").upper()
        if pos == "PK":
            pos = "K"
        if pos in ("DST", "D/ST", "D-ST"):
            pos = "DEF"
        try:
            score = float(score_map.get(pid) or 0.0)
        except (TypeError, ValueError):
            score = 0.0
        players.append({"id": pid, "pos": pos, "ppg": score, "val": score})
    if not players:
        return []
    from utils.draft_grade import dr_optimal_lineup

    chosen = dr_optimal_lineup(players, slots)
    # Preserve slot-fill preference: walk the greedy set in input order.
    return [pid for pid in (str(p["id"]) for p in players) if pid in chosen]
