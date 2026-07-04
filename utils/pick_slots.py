"""Pure rookie-pick slot logic: bracket parsing, draft-order computation, and
pick display labels.

Extracted from app.py so the ordering rules can be unit-tested without the
pandas/DB stack. The draft order is the reverse of final overall standings:
non-playoff teams first (worst regular-season record gets slot 1), then
playoff teams ordered by playoff finish (earliest eliminated first, champion
last).
"""
from typing import Dict, Optional, Set, Tuple


def placements_from_bracket(winners_bracket: list) -> Tuple[Set[int], Dict[int, int]]:
    """Parse a Sleeper winners bracket into playoff participation and finish.

    Returns (playoff_roster_ids, {roster_id: final_placement}). Sleeper sets
    "p" on the decisive matchup for each placement: the winner takes placement
    p, the loser p + 1. Roster ids appear as direct integers in the t1/t2/w/l
    fields; anything else (TBD references like {"w": ...} dicts) is ignored.
    """
    playoff_rids: Set[int] = set()
    placements: Dict[int, int] = {}
    for m in winners_bracket or []:
        if not isinstance(m, dict):
            continue
        for key in ("t1", "t2", "w", "l"):
            v = m.get(key)
            if isinstance(v, int) and v > 0:
                playoff_rids.add(v)
        p = m.get("p")
        if p is None:
            continue
        try:
            p = int(p)
        except (TypeError, ValueError):
            continue
        w = m.get("w")
        l = m.get("l")
        if isinstance(w, int) and w > 0:
            placements[w] = p
        if isinstance(l, int) and l > 0:
            placements[l] = p + 1
    return playoff_rids, placements


def compute_pick_slots(
    reg_ranks: Dict[int, int],
    playoff_rids: Set[int],
    playoff_placements: Dict[int, int],
) -> Dict[int, int]:
    """Rookie draft slots from regular-season ranks and playoff finishes.

    Non-playoff teams get slots 1..k ordered worst-to-best regular season;
    playoff teams get the remaining slots ordered worst-to-best playoff finish
    (champion picks last). Returns {} when there are no playoff placements to
    anchor the order (callers fall back to regular-season-only slots).
    """
    if not playoff_placements:
        return {}

    non_playoff = sorted(
        ((rid, rank) for rid, rank in reg_ranks.items() if rid not in playoff_rids),
        key=lambda x: x[1],  # highest rank number = worst record
        reverse=True,
    )
    playoff_ordered = sorted(
        playoff_placements.items(),
        key=lambda x: x[1],  # highest placement number = worst finish
        reverse=True,
    )

    slot_map: Dict[int, int] = {}
    slot = 1
    for rid, _ in non_playoff:
        slot_map[rid] = slot
        slot += 1
    for rid, _ in playoff_ordered:
        slot_map[rid] = slot
        slot += 1
    return slot_map


def slots_from_regular_season(reg_ranks: Dict[int, int], total_teams: Optional[int] = None) -> Dict[int, int]:
    """Fallback draft order from regular-season standings only: the worst
    record (highest rank number) picks first."""
    total = total_teams or len(reg_ranks)
    return {rid: total - rank + 1 for rid, rank in reg_ranks.items()}


def pick_label(year: int, rnd: int, exact_slot: Optional[int] = None) -> str:
    """Display label for a rookie pick: "2026 1.03" when the exact slot is
    known, "2026 1st (Mid)" otherwise, "Pick" when year/round are missing."""
    if not year or not rnd:
        return "Pick"
    if exact_slot is not None:
        return f"{year} {rnd}.{exact_slot:02d}"
    suffix = {1: "st", 2: "nd", 3: "rd"}.get(rnd, "th")
    return f"{year} {rnd}{suffix} (Mid)"


def avg_pick_value_for_round(by_id: dict, season: int, rnd: int) -> float:
    """Average model value of all picks matching season + round prefix.
    Pick value keys look like "2026_1_03"."""
    prefix = f"{season}_{rnd}_"
    vals = [v for k, v in by_id.items() if k.startswith(prefix)]
    return (sum(vals) / len(vals)) if vals else 0.0


def is_pick_asset_id(asset_id) -> bool:
    """A draft-pick asset id looks like '2026_1_01' or '2026_1_early'
    (year_round_slotOrBucket). Player ids are bare numeric Sleeper ids."""
    parts = str(asset_id or "").split("_")
    if len(parts) < 2:
        return False
    yr = parts[0]
    return len(yr) == 4 and yr.isdigit() and parts[1].isdigit()


def parse_pick_asset(pick_id) -> Optional[dict]:
    """Parse a pick asset id into its parts plus a display name.

    Returns {"season", "round", "slot" (int or None), "slot_raw" (the third
    segment verbatim, e.g. "01" or "early"), "bucket" ("Early"/"Mid"/"Late" or
    None), "name"} or None when the id is not a recognizable pick.
    Names: "2026 1.03" (exact slot), "2026 1st (Early)" (bucket),
    "2026 1st" (round only).
    """
    if not is_pick_asset_id(pick_id):
        return None
    parts = str(pick_id).split("_")
    try:
        yr = int(parts[0])
        rnd = int(parts[1])
    except (ValueError, IndexError):
        return None
    third = parts[2] if len(parts) >= 3 else ""
    sfx = {1: "st", 2: "nd", 3: "rd"}.get(rnd, "th")
    bkt = {"early": "Early", "mid": "Mid", "late": "Late"}.get(third.lower())
    if bkt:
        name = f"{yr} {rnd}{sfx} ({bkt})"
    elif third.isdigit():
        name = f"{yr} {rnd}.{int(third):02d}"
    else:
        name = f"{yr} {rnd}{sfx}"
    return {
        "season": yr,
        "round": rnd,
        "slot": int(third) if third.isdigit() else None,
        "slot_raw": third,
        "bucket": bkt,
        "name": name,
    }
