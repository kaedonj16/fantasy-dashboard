"""Deterministic matchup pairing when a platform has not published a week yet."""
from __future__ import annotations

from typing import List


def synthetic_week_matchups(rosters: List[dict], week: int) -> List[dict]:
    """Round-robin pairs shaped like a Sleeper matchup payload.

    Mirrors ``data_building.simulate_playoff_odds._round_robin_schedule`` so the
    Season Hub preview and the playoff-odds sim agree on undecided weeks.
    """
    ids = sorted(
        {int(r["roster_id"]) for r in (rosters or []) if r.get("roster_id") is not None},
    )
    if len(ids) < 2:
        return []
    by_rid = {
        int(r["roster_id"]): r
        for r in (rosters or [])
        if r.get("roster_id") is not None
    }
    n = len(ids)
    if n % 2 == 1:
        ids = ids + [None]  # bye slot
        n += 1
    fixed = ids[0]
    rotating = ids[1:]
    n_rounds = n - 1
    r = (max(1, int(week)) - 1) % n_rounds
    rot = rotating[-r:] + rotating[:-r] if r else rotating[:]
    pairs: List[tuple] = []
    if fixed is not None and rot[0] is not None:
        pairs.append((fixed, rot[0]))
    for j in range(1, n // 2):
        a, b = rot[j], rot[n - 1 - j]
        if a is not None and b is not None and a != b:
            pairs.append((a, b))
    out: List[dict] = []
    for mid, (left_id, right_id) in enumerate(pairs, start=1):
        for rid in (left_id, right_id):
            roster = by_rid.get(int(rid)) or {}
            out.append({
                "matchup_id": mid,
                "roster_id": rid,
                "points": None,
                "players": list(roster.get("players") or []),
                "starters": list(roster.get("starters") or []),
                "players_points": {},
            })
    return out
