"""Derive a Sleeper-shaped winners bracket from weekly matchup rows.

Fleaflicker and MFL do not publish a bracket object. Playoff weeks still
have paired matchups. This helper turns those rows into the
``{r, m, t1, t2, t1_from, t2_from, w, l}`` shape ``playoff_bracket``
already renders.

When playoff weeks have not been played yet, ``project_bracket_from_seeds``
builds a first-round pairing from standings order so the Playoff Picture
tab is not empty all season.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Sequence


def _rid(row: dict) -> Optional[int]:
    raw = row.get("roster_id")
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def _mid(row: dict, fallback: int) -> int:
    try:
        return int(row.get("matchup_id") or fallback)
    except (TypeError, ValueError):
        return fallback


def _pts(row: dict) -> Optional[float]:
    raw = row.get("points")
    if raw is None:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def pair_matchup_sides(rows: Iterable[dict]) -> List[tuple[int, dict, dict]]:
    """Group roster rows that share a ``matchup_id`` into two-team games."""
    grouped: Dict[int, List[dict]] = defaultdict(list)
    for i, row in enumerate(rows or [], 1):
        if not isinstance(row, dict):
            continue
        if _rid(row) is None:
            continue
        grouped[_mid(row, i)].append(row)
    out: List[tuple[int, dict, dict]] = []
    for mid, sides in grouped.items():
        if len(sides) < 2:
            continue
        out.append((mid, sides[0], sides[1]))
    return sorted(out, key=lambda g: g[0])


def _winner_loser(left: dict, right: dict) -> tuple[Optional[int], Optional[int]]:
    p1, p2 = _pts(left), _pts(right)
    r1, r2 = _rid(left), _rid(right)
    if p1 is None or p2 is None or r1 is None or r2 is None:
        return None, None
    # Unplayed / 0-0 games stay undecided.
    if p1 == 0 and p2 == 0:
        return None, None
    if p1 > p2:
        return r1, r2
    if p2 > p1:
        return r2, r1
    return None, None


def derive_bracket_from_matchups(
    matchups_by_week: Dict[Any, Sequence[dict]],
    playoff_week_start: int,
    *,
    kind: str = "winners",
    max_rounds: int = 4,
) -> List[Dict[str, Any]]:
    """Build bracket rounds from playoff-week matchup rows.

    ``kind`` other than ``winners`` returns [] (consolation is not derived).
    """
    if str(kind or "winners").lower() != "winners":
        return []
    try:
        start = int(playoff_week_start)
    except (TypeError, ValueError):
        return []
    if start <= 0:
        return []

    weeks = []
    for raw in (matchups_by_week or {}):
        try:
            week = int(raw)
        except (TypeError, ValueError):
            continue
        if start <= week < start + max_rounds:
            weeks.append(week)
    weeks = sorted(set(weeks))

    out: List[Dict[str, Any]] = []
    for i, week in enumerate(weeks):
        rows = matchups_by_week.get(week) or matchups_by_week.get(str(week)) or []
        for mid, left, right in pair_matchup_sides(rows):
            winner, loser = _winner_loser(left, right)
            out.append({
                "r": i + 1,
                "m": mid,
                "t1": _rid(left),
                "t2": _rid(right),
                "t1_from": None,
                "t2_from": None,
                "w": winner,
                "l": loser,
                "derived": True,
            })
    return out


def project_bracket_from_seeds(
    seed_roster_ids: Sequence[Any],
    *,
    playoff_teams: int = 6,
) -> List[Dict[str, Any]]:
    """First-round pairings from standings order when playoff weeks are empty.

    Standard field: 4 teams (1v4, 2v3), 6 teams (3v6 and 4v5; 1 and 2 bye),
    8 teams (1v8, 4v5, 2v7, 3v6). Other sizes pair the bottom of the field
    and leave the top as byes.
    """
    seeds: List[int] = []
    for raw in seed_roster_ids or []:
        try:
            seeds.append(int(raw))
        except (TypeError, ValueError):
            continue
    try:
        n = int(playoff_teams or 0)
    except (TypeError, ValueError):
        n = 0
    if n <= 0:
        n = len(seeds)
    seeds = seeds[:n]
    if len(seeds) < 2:
        return []

    # Number of first-round games = field size minus next power-of-two byes.
    import math
    bracket = 1 << int(math.ceil(math.log2(max(len(seeds), 2))))
    byes = bracket - len(seeds)
    playing = seeds[byes:]
    if len(playing) < 2:
        return []
    games = []
    lo, hi = 0, len(playing) - 1
    mid = 1
    while lo < hi:
        games.append({
            "r": 1,
            "m": mid,
            "t1": playing[lo],
            "t2": playing[hi],
            "t1_from": None,
            "t2_from": None,
            "w": None,
            "l": None,
            "derived": True,
            "projected": True,
        })
        mid += 1
        lo += 1
        hi -= 1
    return games


def derive_or_project_bracket(
    *,
    matchups_by_week: Optional[Dict[Any, Sequence[dict]]] = None,
    playoff_week_start: int = 15,
    seed_roster_ids: Optional[Sequence[Any]] = None,
    playoff_teams: int = 6,
    kind: str = "winners",
) -> List[Dict[str, Any]]:
    """Prefer real playoff-week games; otherwise project the first round."""
    actual = derive_bracket_from_matchups(
        matchups_by_week or {}, playoff_week_start, kind=kind,
    )
    if actual:
        return actual
    if str(kind or "winners").lower() != "winners":
        return []
    return project_bracket_from_seeds(seed_roster_ids or [], playoff_teams=playoff_teams)
