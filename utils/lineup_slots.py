"""Canonical lineup-slot names across Sleeper, ESPN, Yahoo, MFL, and Fleaflicker.

Providers do not share one name for FLEX, Superflex, or D/ST. Scoring,
start/sit, waiver-need, optimal-lineup, and playoff sims must treat those
aliases as the same slot or the same league grades differently depending on
which platform supplied its settings.

Restricted flex (WR/RB only, WR/TE, RB/TE) stays distinct from standard FLEX
(RB/WR/TE). Collapsing ``WRRB_FLEX`` / Yahoo ``W/R`` into FLEX lets a TE start
in a slot that cannot hold one.
"""
from __future__ import annotations

from typing import Dict, FrozenSet, Iterable, List, Optional

# Skill FLEX (RB/WR/TE). Never includes a QB — Superflex is a separate pool.
# Restricted two-position flexes live in their own sets below.
FLEX_SLOT_NAMES = {
    "FLEX",
    "RB_WR_TE", "WR_RB_TE", "RBWRTE", "WRRBTE", "WRRBTE_FLEX",
    "W_R_T", "RB_WR_TE_FLEX", "RBWRTE_FLEX",
}

# RB or WR only (Sleeper WRRB_FLEX, Yahoo W/R, ESPN RB/WR).
RB_WR_SLOT_NAMES = {
    "RB_WR", "WR_RB", "WRRB_FLEX", "RBWR_FLEX", "RBWR",
    "W_R", "RB_WR_FLEX", "WR_RB_FLEX",
}

# WR or TE only (Sleeper REC_FLEX, Yahoo W/T, ESPN WR/TE).
WR_TE_SLOT_NAMES = {
    "WR_TE", "TE_WR", "REC_FLEX", "WRTE_FLEX", "W_T", "WR_TE_FLEX",
}

# RB or TE only (Yahoo R/T, Fleaflicker RB/TE).
RB_TE_SLOT_NAMES = {
    "RB_TE", "TE_RB", "R_T", "RBTE_FLEX", "RB_TE_FLEX",
}

RESTRICTED_FLEX_SLOTS = ("RB_WR", "WR_TE", "RB_TE")

# QB-eligible FLEX. "OP" is ESPN's offensive-player slot.
SUPERFLEX_SLOT_NAMES = {
    "SUPER_FLEX", "SUPERFLEX", "SFLEX", "OP",
    "QB_RB_WR_TE", "Q_RB_WR_TE", "Q_W_R_T", "QB_WR_RB_TE",
    # Fleaflicker sometimes lists eligible positions in other orders.
    "RB_WR_TE_QB", "WR_RB_TE_QB",
}

DEF_SLOT_NAMES = {"DEF", "DST", "D_ST", "D_S_T"}

SKILL_POSITIONS = {"QB", "RB", "WR", "TE"}
BENCH_SLOT_NAMES = {"BN", "BE", "BENCH", "IR", "TAXI", "RESERVE"}

SLOT_ELIGIBILITY: Dict[str, FrozenSet[str]] = {
    "QB": frozenset({"QB"}),
    "RB": frozenset({"RB"}),
    "WR": frozenset({"WR"}),
    "TE": frozenset({"TE"}),
    "K": frozenset({"K"}),
    "DEF": frozenset({"DEF"}),
    "RB_WR": frozenset({"RB", "WR"}),
    "WR_TE": frozenset({"WR", "TE"}),
    "RB_TE": frozenset({"RB", "TE"}),
    "FLEX": frozenset({"RB", "WR", "TE"}),
    "SUPER_FLEX": frozenset({"QB", "RB", "WR", "TE"}),
}


def normalize_slot_name(slot) -> str:
    """Uppercase a slot and collapse '/', '-', '+', spaces to underscores."""
    s = str(slot or "").upper().strip()
    for ch in ("-", "/", "+", " ", "."):
        s = s.replace(ch, "_")
    while "__" in s:
        s = s.replace("__", "_")
    return s.strip("_")


def canonicalize_slot(slot) -> str:
    """Map a provider slot name onto one canonical token.

    Unknown slots pass through normalized (so IDP names like DL/LB/DB keep
    working). Empty input stays empty. Restricted flex aliases stay distinct
    from standard FLEX.
    """
    s = normalize_slot_name(slot)
    if not s:
        return ""
    if s in SUPERFLEX_SLOT_NAMES:
        return "SUPER_FLEX"
    if s in FLEX_SLOT_NAMES:
        return "FLEX"
    if s in RB_WR_SLOT_NAMES:
        return "RB_WR"
    if s in WR_TE_SLOT_NAMES:
        return "WR_TE"
    if s in RB_TE_SLOT_NAMES:
        return "RB_TE"
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


def slot_eligible_positions(slot) -> FrozenSet[str]:
    """Positions that can fill ``slot`` after canonicalization."""
    s = canonicalize_slot(slot)
    if not s:
        return frozenset()
    return SLOT_ELIGIBILITY.get(s, frozenset({s}))


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


def restricted_flex_counts(roster_positions: Optional[Iterable] = None,
                           slot_counts: Optional[Dict[str, int]] = None) -> Dict[str, int]:
    """Counts of RB_WR / WR_TE / RB_TE after canonicalization."""
    counts = slot_counts if slot_counts is not None else count_lineup_slots(roster_positions)
    return {
        "RB_WR": slot_total(counts, RB_WR_SLOT_NAMES | {"RB_WR"}),
        "WR_TE": slot_total(counts, WR_TE_SLOT_NAMES | {"WR_TE"}),
        "RB_TE": slot_total(counts, RB_TE_SLOT_NAMES | {"RB_TE"}),
    }


def is_superflex_lineup(roster_positions: Optional[Iterable] = None,
                        slot_counts: Optional[Dict[str, int]] = None) -> bool:
    """True when the lineup starts a Superflex / OP / 2QB slot."""
    return superflex_count(roster_positions, slot_counts) > 0


def is_restricted_flex_slot(slot) -> bool:
    return canonicalize_slot(slot) in RESTRICTED_FLEX_SLOTS


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


def _split_pair(n: int) -> tuple[int, int]:
    """Split ``n`` interchangeable slots; the odd one goes to the second side."""
    n = max(0, int(n or 0))
    return n // 2, n - (n // 2)


def starter_need_counts(roster_positions: Optional[Iterable],
                        extra_depth: int = 1) -> Dict[str, int]:
    """How many players a waiver-aware roster should have at each skill position.

    Dedicated starters plus a fair share of FLEX (split RB/WR), restricted flex
    split across its two eligible positions, and Superflex as extra QB, then
    ``extra_depth`` bench insurance so a 2-RB league still wants a 3rd RB.
    Superflex is *not* added to RB/WR — those slots start a QB.
    """
    counts = count_lineup_slots(roster_positions)
    flex = counts.get("FLEX", 0)
    sf = counts.get("SUPER_FLEX", 0)
    rb_wr = counts.get("RB_WR", 0)
    wr_te = counts.get("WR_TE", 0)
    rb_te = counts.get("RB_TE", 0)
    rb_from_flex, wr_from_flex = _split_pair(flex)
    rb_from_rb_wr, wr_from_rb_wr = _split_pair(rb_wr)
    wr_from_wr_te, te_from_wr_te = _split_pair(wr_te)
    rb_from_rb_te, te_from_rb_te = _split_pair(rb_te)
    extra = max(0, int(extra_depth))
    return {
        "QB": max(1, counts.get("QB", 0) + sf) + extra,
        "RB": max(1, counts.get("RB", 0) + rb_from_flex + rb_from_rb_wr + rb_from_rb_te) + extra,
        "WR": max(1, counts.get("WR", 0) + wr_from_flex + wr_from_rb_wr + wr_from_wr_te) + extra,
        "TE": max(1, counts.get("TE", 0) + te_from_wr_te + te_from_rb_te) + extra,
    }
