"""Canonical lineup-slot names across Sleeper, ESPN, Yahoo, and MFL.

Providers do not share one name for FLEX, Superflex, or D/ST. Scoring,
start/sit, waiver-need, optimal-lineup, and playoff sims must treat those
aliases as the same slot or the same league grades differently depending on
which platform supplied its settings.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional

# Skill FLEX (RB/WR/TE). Never includes a QB — Superflex is a separate pool.
FLEX_SLOT_NAMES = {
    "FLEX",
    "RB_WR_FLEX", "RB_WR_TE", "WR_RB", "WR_TE", "RB_WR",
    "WRRB_FLEX", "WRTE_FLEX", "RBWR_FLEX", "RBWRTE", "RBWR",
    "REC_FLEX",
    "W_R_T", "W_R", "WR_RB_TE", "RB_WR_TE_FLEX",
}

# QB-eligible FLEX. "OP" is ESPN's offensive-player slot.
SUPERFLEX_SLOT_NAMES = {
    "SUPER_FLEX", "SUPERFLEX", "SFLEX", "OP",
    "QB_RB_WR_TE", "Q_RB_WR_TE", "Q_W_R_T", "QB_WR_RB_TE",
}

DEF_SLOT_NAMES = {"DEF", "DST", "D_ST", "D_S_T"}

SKILL_POSITIONS = {"QB", "RB", "WR", "TE"}
BENCH_SLOT_NAMES = {"BN", "BE", "BENCH", "IR", "TAXI", "RESERVE"}


def normalize_slot_name(slot) -> str:
    """Uppercase a slot and collapse '/', '-', spaces to underscores."""
    s = str(slot or "").upper().strip()
    for ch in ("-", "/", " ", "."):
        s = s.replace(ch, "_")
    while "__" in s:
        s = s.replace("__", "_")
    return s.strip("_")


def canonicalize_slot(slot) -> str:
    """Map a provider slot name onto one canonical token.

    Unknown slots pass through normalized (so IDP names like DL/LB/DB keep
    working). Empty input stays empty.
    """
    s = normalize_slot_name(slot)
    if not s:
        return ""
    if s in FLEX_SLOT_NAMES:
        return "FLEX"
    if s in SUPERFLEX_SLOT_NAMES:
        return "SUPER_FLEX"
    if s in DEF_SLOT_NAMES:
        return "DEF"
    if s in {"BE", "BENCH"}:
        return "BN"
    if s == "RESERVE":
        return "IR"
    return s


def canonicalize_slots(roster_positions: Optional[Iterable]) -> List[str]:
    """Canonicalize a league's slot list, dropping empty entries."""
    out: List[str] = []
    for slot in roster_positions or []:
        s = canonicalize_slot(slot)
        if s:
            out.append(s)
    return out


def count_lineup_slots(roster_positions: Optional[Iterable]) -> Dict[str, int]:
    """Count canonical slots in a league's starting-lineup list."""
    counts: Dict[str, int] = {}
    for s in canonicalize_slots(roster_positions):
        counts[s] = counts.get(s, 0) + 1
    return counts


def slot_total(slot_counts: Optional[Dict[str, int]], names: Iterable[str]) -> int:
    """Sum equivalent slots whether the dict is keyed by canonical or alias names."""
    counts = slot_counts or {}
    wanted = {normalize_slot_name(n) for n in names}
    total = 0
    for key, n in counts.items():
        canon = canonicalize_slot(key)
        raw = normalize_slot_name(key)
        if raw in wanted or canon in wanted:
            total += int(n or 0)
    return total


def flex_count(roster_positions: Optional[Iterable] = None,
               slot_counts: Optional[Dict[str, int]] = None) -> int:
    if slot_counts is not None:
        return slot_total(slot_counts, FLEX_SLOT_NAMES | {"FLEX"})
    return count_lineup_slots(roster_positions).get("FLEX", 0)


def superflex_count(roster_positions: Optional[Iterable] = None,
                    slot_counts: Optional[Dict[str, int]] = None) -> int:
    if slot_counts is not None:
        return slot_total(slot_counts, SUPERFLEX_SLOT_NAMES | {"SUPER_FLEX"})
    return count_lineup_slots(roster_positions).get("SUPER_FLEX", 0)


def is_superflex_lineup(roster_positions: Optional[Iterable] = None,
                        slot_counts: Optional[Dict[str, int]] = None) -> bool:
    """True when the lineup starts a Superflex / OP / 2QB slot."""
    return superflex_count(roster_positions, slot_counts) > 0


def start_sit_pos(pos) -> str:
    """Canonical Start/Sit bucket: QB/RB/WR/TE/K/DEF, else empty."""
    s = canonicalize_slot(pos)
    if s in SKILL_POSITIONS or s in {"K", "DEF"}:
        return s
    return ""


def start_sit_groups(slot_counts: Optional[Dict[str, int]] = None,
                     roster_positions: Optional[Iterable] = None) -> List[str]:
    """Position groups the Start/Sit advisor should rank for this lineup."""
    counts = slot_counts if slot_counts is not None else count_lineup_slots(roster_positions)
    groups = ["QB", "RB", "WR", "TE"]
    if int((counts or {}).get("K") or 0) > 0:
        groups.append("K")
    if int((counts or {}).get("DEF") or 0) > 0:
        groups.append("DEF")
    return groups


def starter_need_counts(roster_positions: Optional[Iterable],
                        extra_depth: int = 1) -> Dict[str, int]:
    """How many players a waiver-aware roster should have at each skill position.

    Dedicated starters plus a fair share of FLEX (split RB/WR) plus Superflex as
    extra QB, then ``extra_depth`` bench insurance so a 2-RB league still wants
    a 3rd RB. Superflex is *not* added to RB/WR — those slots start a QB.
    """
    counts = count_lineup_slots(roster_positions)
    flex = counts.get("FLEX", 0)
    sf = counts.get("SUPER_FLEX", 0)
    rb_flex = flex // 2
    wr_flex = flex - rb_flex
    extra = max(0, int(extra_depth))
    return {
        "QB": max(1, counts.get("QB", 0) + sf) + extra,
        "RB": max(1, counts.get("RB", 0) + rb_flex) + extra,
        "WR": max(1, counts.get("WR", 0) + wr_flex) + extra,
        "TE": max(1, counts.get("TE", 0)) + extra,
    }
