"""Deterministic matchup pairing when a platform has not published a week yet."""
from __future__ import annotations

from typing import List


def _starters_look_like_full_roster(starters: List[str], players: List[str]) -> bool:
    return bool(players) and len(starters) >= len(players) and len(players) > 9


def lineup_from_roster(roster: dict, *, starter_slots: int = 9) -> tuple[List[str], List[str]]:
    """Return (starters, bench) canonical ids from a normalized roster dict."""
    players = [str(p) for p in (roster.get("players") or []) if p]
    if not players:
        return [], []

    stored_starters = [str(s) for s in (roster.get("starters") or []) if s]
    reserve = {str(r) for r in (roster.get("reserve") or []) if r}

    # Prefer the platform lineup when it looks real. ``reserve`` is IR-only
    # (Sleeper/ESPN/Yahoo); subtracting it from the full roster would pull
    # the bench into the matchup starters.
    if stored_starters and not _starters_look_like_full_roster(stored_starters, players):
        starters = [p for p in stored_starters if p not in reserve]
    elif reserve:
        starters = [p for p in players if p not in reserve]
    else:
        starters = list(stored_starters)

    if not starters:
        # All-BN / unset lineup, or empty starter field — never leave matchups blank.
        if stored_starters and not _starters_look_like_full_roster(stored_starters, players):
            starters = stored_starters
        elif stored_starters:
            starters = stored_starters[:starter_slots]
        else:
            starters = players[:starter_slots]
    elif _starters_look_like_full_roster(starters, players):
        starters = starters[:starter_slots]

    bench = [p for p in players if p not in set(starters)]
    return starters, bench


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
            starters, _bench = lineup_from_roster(roster)
            out.append({
                "matchup_id": mid,
                "roster_id": rid,
                "points": None,
                "players": list(roster.get("players") or []),
                "starters": starters,
                "players_points": {},
            })
    return out


def resolve_matchup_week(current_week, matchups_by_week=None) -> int:
    """Week to paint on the dashboard / scout.

    Sleeper's NFL state can still be ``week=0`` in the days before kickoff
    while Yahoo / ESPN already publish a Week 1 scoreboard. The weekly hub
    already uses ``current_week or 1``; the dashboard was looking up week 0
    and rendering an empty "No matchups" carousel.
    """
    by_week = matchups_by_week if isinstance(matchups_by_week, dict) else {}
    try:
        week = int(current_week or 0)
    except (TypeError, ValueError):
        week = 0

    def _rows(w):
        return by_week.get(w) or by_week.get(str(w)) or []

    if week > 0:
        if _rows(week) or not by_week:
            return week
    if _rows(1):
        return 1
    populated = []
    for key in by_week:
        try:
            w = int(key)
        except (TypeError, ValueError):
            continue
        if w > 0 and _rows(w):
            populated.append(w)
    if populated:
        return min(populated)
    return max(1, week)
