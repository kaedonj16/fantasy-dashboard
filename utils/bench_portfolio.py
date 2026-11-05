"""Bench as portfolio (Frontier 7).

A bench spot is a held option: its value is the path by which the
player becomes startable. Two benches with the same market value
are not the same bench. The backup one injury away from a starter's
job, on the team whose starter YOU roster (a handcuff: his path
pays out on your own roster), is worth more to you than the same
player is to the field; the third-stringer behind two healthy
starters is worth almost nothing no matter what a value table says.

This module prices that path, transparently and additively:

* one-away: depth order 2 at his NFL position (+30), order 3 (+10);
* handcuff premium: the starter ahead of him is on the viewer's own
  roster (+25);
* live window: the starter ahead is out — with an ESPN return
  horizon, +3 per week of coverage (capped at 6 weeks); with only
  a role-state "starter out" signal and no honest date, +10;
* role momentum: promoted (+20) or returning (+10) this week;
  benched (-20).

Market value breaks ties. The output is a ranked portfolio plus
the funding order the GM brief's claims consume: when a claim needs
a drop, the spot that should fund it is the one with the WEAKEST
path, not merely the lowest sticker price.

Fail soft: no roster, no depth data and no role states yields an
empty portfolio and callers keep their previous order.
"""
from __future__ import annotations

from typing import Any, Callable, Optional

_ONE_AWAY = 30
_TWO_AWAY = 10
_HANDCUFF = 25
_WINDOW_PER_WEEK = 3
_WINDOW_CAP_WEEKS = 6
_WINDOW_NO_DATE = 10
_PROMOTED = 20
_RETURNING = 10
_BENCHED = -20


def _players(ctx: dict) -> dict:
    players = ctx.get("players") or {}
    return players if isinstance(players, dict) else {}


def _values(ctx: dict) -> dict:
    out = {}
    for row in ctx.get("model_value_table") or []:
        if not isinstance(row, dict):
            continue
        pid = str(row.get("player_id") or row.get("id") or "")
        if not pid:
            continue
        val = row.get("value_sf") or row.get("value") \
            or row.get("value_1qb")
        try:
            out[pid] = float(val) if val is not None else 0.0
        except (TypeError, ValueError):
            out[pid] = 0.0
    return out


def _depth_order(row: dict) -> Optional[int]:
    order = row.get("depth_chart_order")
    try:
        return int(order) if order is not None else None
    except (TypeError, ValueError):
        return None


def _starter_ahead(players: dict, row: dict) -> Optional[str]:
    """The pid of the player directly ahead on the depth chart."""
    team = str(row.get("team") or "").upper()
    pos = str(row.get("position") or row.get("pos") or "").upper()
    order = _depth_order(row)
    if not team or not pos or not order or order <= 1:
        return None
    best = None
    best_order = 0
    for pid, other in players.items():
        if not isinstance(other, dict) or str(pid) == str(row.get("_pid")):
            continue
        if str(other.get("team") or "").upper() != team:
            continue
        if str(other.get("position") or other.get("pos")
               or "").upper() != pos:
            continue
        o = _depth_order(other)
        if o is not None and best_order < o < order:
            best, best_order = str(pid), o
    return best


def bench_portfolio(ctx: dict, viewer_roster_id, *,
                    week: Optional[int] = None,
                    role_states: Optional[dict] = None,
                    weeks_out_fn: Optional[Callable] = None,
                    ) -> list:
    """Ranked end-of-bench rows for the viewer (best path first).

    Each row: ``{player_id, name, position, team, value,
    path_score, reasons, handcuff_of}``. Starters, reserve and taxi
    residents are not bench spots and never appear.
    """
    if not isinstance(ctx, dict):
        return []
    roster = None
    for r in ctx.get("rosters") or []:
        if isinstance(r, dict) and str(r.get("roster_id")) == str(
                viewer_roster_id):
            roster = r
            break
    if roster is None:
        return []
    players = _players(ctx)
    values = _values(ctx)
    if week is None:
        week = ctx.get("current_week") or ctx.get("week") or 0
    if role_states is None:
        try:
            from utils.role_state import get_role_states

            role_states = get_role_states(ctx, int(week or 0))
        except Exception:
            role_states = {}
    if weeks_out_fn is None:
        try:
            from dashboard_services.injury_return import (
                weeks_out_for_player,
            )

            weeks_out_fn = weeks_out_for_player
        except Exception:
            weeks_out_fn = lambda pid: None  # noqa: E731
    starters = {str(p) for p in (roster.get("starters") or [])}
    reserve = {str(p) for p in (roster.get("reserve") or [])}
    taxi = {str(p) for p in (roster.get("taxi") or [])}
    mine = {str(p) for p in (roster.get("players") or [])}
    rows = []
    for pid in mine:
        if pid in starters or pid in reserve or pid in taxi:
            continue
        row = dict(players.get(pid) or {})
        row["_pid"] = pid
        pos = str(row.get("position") or row.get("pos") or "").upper()
        order = _depth_order(row)
        score = 0
        reasons = []
        handcuff_of = None
        ahead = _starter_ahead(players, row) if order and order > 1 \
            else None
        if order == 2:
            score += _ONE_AWAY
            reasons.append("One injury away from the starting job")
        elif order == 3:
            score += _TWO_AWAY
            reasons.append("Two deep on his depth chart")
        if ahead is not None and ahead in mine:
            score += _HANDCUFF
            handcuff_of = ahead
            ahead_name = str((players.get(ahead) or {}).get(
                "full_name") or (players.get(ahead) or {}).get("name")
                or "your starter")
            reasons.append(f"Handcuff to your own {ahead_name}")
        if ahead is not None:
            weeks_out = None
            try:
                weeks_out = weeks_out_fn(ahead)
            except Exception:
                weeks_out = None
            ahead_state = str((role_states.get(ahead) or {}).get(
                "state") or "")
            if weeks_out:
                gained = int(min(float(weeks_out), _WINDOW_CAP_WEEKS)
                             * _WINDOW_PER_WEEK)
                score += gained
                reasons.append(
                    f"The starter ahead is out about "
                    f"{float(weeks_out):g} more weeks")
            elif ahead_state == "starter_out":
                score += _WINDOW_NO_DATE
                reasons.append("The starter ahead is out this week")
        state = str((role_states.get(pid) or {}).get("state") or "")
        if state == "promoted":
            score += _PROMOTED
            reasons.append("Promoted into a starting role this week")
        elif state == "returning":
            score += _RETURNING
            reasons.append("Returning this week")
        elif state == "benched":
            score += _BENCHED
            reasons.append("Lost his starting job")
        rows.append({
            "player_id": pid,
            "name": str(row.get("full_name") or row.get("name") or pid),
            "position": pos,
            "team": str(row.get("team") or "").upper(),
            "value": values.get(pid, 0.0),
            "path_score": score,
            "reasons": reasons,
            "handcuff_of": handcuff_of,
        })
    rows.sort(key=lambda r: (-r["path_score"], -r["value"], r["name"]))
    return rows


def drop_order(ctx: dict, viewer_roster_id, **kwargs) -> list:
    """Player ids in funding order: weakest path to value first.

    The GM brief's claim funding consumes this: the drop that costs
    the roster least is the bench spot with no live path, which is
    not always the cheapest sticker on the bench.
    """
    rows = bench_portfolio(ctx, viewer_roster_id, **kwargs)
    ranked = sorted(rows, key=lambda r: (r["path_score"], r["value"],
                                         r["name"]))
    return [r["player_id"] for r in ranked]


__all__ = ["bench_portfolio", "drop_order"]
