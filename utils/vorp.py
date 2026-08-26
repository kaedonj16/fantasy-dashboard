"""Pure VORP / replacement-level helpers.

Fantasy VORP is season points minus a replacement-level starter at the same
position. Replacement rank is (starters × teams + FLEX share), matching the
league-aware math in ``data_building.advanced_metrics``.

These helpers are importable without the DB/Flask stack so unit tests can pin
the Kraft-style case: a top-10 projected TE must not inherit last season's
injury-shortened totals as VORP next to upcoming-season proj PPG.
"""
from __future__ import annotations

import os
from typing import Any, Dict, Iterable, List, Mapping, Optional

# Standard starters per team used to locate the replacement-level player.
# FLEX is modeled as one extra RB/WR/TE slot per team, split by typical usage.
VALUE_STARTERS = {"QB": 1.0, "RB": 2.0, "WR": 3.0, "TE": 1.0}
VALUE_FLEX_ALLOC = {"RB": 0.45, "WR": 0.45, "TE": 0.10}
# Marginal PPR points worth one head-to-head win (≈ weekly team-score stdev).
POINTS_PER_WIN_DEFAULT = 28.0
PROJ_SEASON_GAMES = 17.0

_POS_NORM = {"HB": "RB", "FB": "RB", "SE": "WR", "FL": "WR"}


def normalize_position(pos: Optional[str]) -> Optional[str]:
    """Return the canonical fantasy position for a raw position string."""
    if not pos:
        return pos
    upper = str(pos).upper()
    return _POS_NORM.get(upper, upper)


def points_per_win() -> float:
    try:
        v = float(os.getenv("POINTS_PER_WIN", "").strip())
        return v if v > 0 else POINTS_PER_WIN_DEFAULT
    except (TypeError, ValueError):
        return POINTS_PER_WIN_DEFAULT


def stamp_value_metrics(
    recs: List[Dict[str, Any]],
    num_teams: int = 12,
    starters: Optional[Dict[str, float]] = None,
    points_per_win_value: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """Stamp ``vorp``, ``war``, and position ranks onto recs with ``position`` + ``pts``.

    Mutates ``recs`` in place and returns them. Replacement is the player ranked
    at ``round(starters×teams + flex_share×teams)`` within the position.
    """
    teams = int(num_teams) if num_teams and num_teams > 0 else 12
    start_slots = {**VALUE_STARTERS, **(starters or {})}
    ppw = (
        points_per_win_value
        if (points_per_win_value and points_per_win_value > 0)
        else points_per_win()
    )

    pool_by_pos: Dict[str, List[float]] = {}
    for rec in recs:
        pos = rec.get("position")
        if pos not in start_slots:
            continue
        pool_by_pos.setdefault(pos, []).append(float(rec.get("pts") or 0.0))

    repl_pts: Dict[str, float] = {}
    for pos, base in start_slots.items():
        rank = base * teams + VALUE_FLEX_ALLOC.get(pos, 0.0) * teams
        pool = sorted(pool_by_pos.get(pos, []), reverse=True)
        if not pool:
            repl_pts[pos] = 0.0
            continue
        idx = int(round(rank)) - 1
        idx = max(0, min(len(pool) - 1, idx))
        repl_pts[pos] = pool[idx]

    for rec in recs:
        pos = rec.get("position")
        if pos not in start_slots:
            continue
        vorp = float(rec.get("pts") or 0.0) - repl_pts.get(pos, 0.0)
        rec["vorp"] = round(vorp, 3)
        rec["war"] = round(vorp / ppw, 3)

    for pos in start_slots:
        pos_recs = [r for r in recs if r.get("position") == pos]
        for key in ("vorp", "war"):
            for i, rec in enumerate(
                sorted(pos_recs, key=lambda x: x[key], reverse=True), 1
            ):
                rec[f"{key}_rank"] = i
    return recs


def projected_season_pts(
    player: Mapping[str, Any],
    season_games: float = PROJ_SEASON_GAMES,
) -> Optional[float]:
    """Upcoming-season points: ``proj_pts``, else ``proj_ppg × season_games``.

    Last-season actuals (``ppg``, ``total_pts``) are ignored on purpose — those
    are the injury-shortened numbers that made a TE10 project as −40 VORP.
    """
    pts = player.get("proj_pts")
    try:
        if pts is not None and float(pts) > 0:
            return float(pts)
    except (TypeError, ValueError):
        pass
    ppg = player.get("proj_ppg")
    try:
        if ppg is not None and float(ppg) > 0:
            return float(ppg) * float(season_games)
    except (TypeError, ValueError):
        pass
    return None


def projected_vorp_map(
    players: Iterable[Mapping[str, Any]],
    num_teams: int = 12,
    starters: Optional[Dict[str, float]] = None,
    season_games: float = PROJ_SEASON_GAMES,
) -> Dict[str, float]:
    """Player-id → VORP from projected season points (draft/rankings overlay).

    Uses the same replacement rank as historical VORP, but the inputs are
    upcoming-season projections so a top-10 projected TE cannot inherit last
    year's missed-game totals.
    """
    recs: List[Dict[str, Any]] = []
    for p in players or []:
        pos = normalize_position(str(p.get("position") or ""))
        if pos not in VALUE_STARTERS:
            continue
        pts = projected_season_pts(p, season_games)
        if pts is None:
            continue
        pid = str(p.get("id") or p.get("player_id") or "")
        if not pid:
            continue
        recs.append({"player_id": pid, "position": pos, "pts": pts})
    if not recs:
        return {}
    stamp_value_metrics(recs, num_teams=num_teams, starters=starters)
    return {r["player_id"]: float(r["vorp"]) for r in recs}
