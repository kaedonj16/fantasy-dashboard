"""Keeper-cost pricing (Frontier 3).

Wave 1 taught the site WHO is a designated keeper (untradeable).
What it never priced is what a keeper COSTS: a player kept at a
Round 6 cost and the same player kept at a Round 1 cost are
different assets wearing the same market value, and every team's
projected keeper set decides who is actually gettable next year.
In a keeper league, trade values that ignore the round are wrong.

The cost machinery already exists for the keeper page
(``dashboard_services.pages.keeper_page.compute_league_keepers``:
draft history + the surplus optimizer, per team). This module is
the trade layer's reader of it:

* ``keeper_costs(ctx)`` -> ``{pid: {cost_round, adp_overall,
  roster_id, projected}}`` for the league's projected keeper sets,
  memoized on the ctx. Non-keeper formats and missing draft
  history yield {} — no cost is ever invented for a player the
  optimizer would not keep, and redraft leagues see nothing.
* ``keeper_block_for_package(ctx, send, receive)`` -> the block
  ``trade_partner_checks.evaluate_package`` attaches: per-side
  keeper surplus (the settled ``keeper_surplus_value`` curve:
  value gained by keeping at the cost round instead of drafting
  at ADP), so a deal sending away cheap keepers is visibly more
  expensive than its sticker values say.
"""
from __future__ import annotations

from typing import Any, Optional


def _league_size(ctx: dict) -> int:
    try:
        n = int(ctx.get("total_rosters") or 0)
    except (TypeError, ValueError):
        n = 0
    if n <= 0:
        n = len(ctx.get("rosters") or [])
    return n or 12


def keeper_costs(ctx: dict, *, platform: Optional[str] = None,
                 league_id: Optional[str] = None) -> dict:
    """Projected keeper costs for the league, or {} (fail closed)."""
    if not isinstance(ctx, dict):
        return {}
    memo = ctx.get("_keeper_costs")
    if isinstance(memo, dict):
        return memo
    out: dict = {}
    try:
        from dashboard_services.keeper_core import is_keeper_format

        if is_keeper_format(ctx):
            from dashboard_services.pages.keeper_page import (
                compute_league_keepers,
            )

            platform = platform or str(ctx.get("platform") or "sleeper")
            league_id = league_id or str(ctx.get("league_id") or "")
            payload = compute_league_keepers(
                ctx, platform=platform, league_id=league_id)
            for row in (payload or {}).get("kept") or []:
                if not isinstance(row, dict):
                    continue
                pid = str(row.get("id") or "")
                cost = row.get("costRound")
                if not pid or cost is None:
                    continue
                out[pid] = {
                    "cost_round": int(cost),
                    "adp_overall": row.get("adpOverall"),
                    "roster_id": str(row.get("rosterId") or ""),
                    "projected": bool(row.get("projected")),
                }
    except Exception:
        out = {}
    try:
        ctx["_keeper_costs"] = out
    except Exception:
        pass
    return out


def _asset_pid(asset: Any) -> str:
    if not isinstance(asset, dict):
        return ""
    return str(asset.get("id") or asset.get("player_id") or "")


def _side(assets: list, costs: dict, league_size: int) -> tuple:
    from utils.draft import keeper_surplus_value

    rows = []
    total = 0.0
    for asset in assets or []:
        pid = _asset_pid(asset)
        entry = costs.get(pid)
        if not entry:
            continue
        surplus = keeper_surplus_value(
            entry.get("adp_overall"), entry["cost_round"], league_size)
        rows.append({
            "player_id": pid,
            "cost_round": entry["cost_round"],
            "surplus_value": (round(float(surplus), 2)
                              if surplus is not None else None),
        })
        if surplus is not None:
            total += float(surplus)
    return rows, round(total, 2)


def keeper_block_for_package(ctx: dict, send: list, receive: list,
                             costs: Optional[dict] = None,
                             ) -> Optional[dict]:
    """The keeper-cost block for one package, or None.

    None when the league has no priced keepers in the deal at all:
    the block appears only when it has something true to say.
    """
    if not isinstance(ctx, dict):
        return None
    costs = costs if costs is not None else keeper_costs(ctx)
    if not costs:
        return None
    size = _league_size(ctx)
    send_rows, send_total = _side(send, costs, size)
    recv_rows, recv_total = _side(receive, costs, size)
    if not send_rows and not recv_rows:
        return None
    notes = []
    for label, rows in (("Sends", send_rows), ("Receives", recv_rows)):
        cheap = [r for r in rows
                 if r["surplus_value"] and r["surplus_value"] > 0]
        if cheap:
            rounds = ", ".join(
                f"Round {r['cost_round']}" for r in cheap)
            notes.append(
                f"{label} keeper value beyond the sticker price "
                f"(kept at {rounds})")
    return {
        "send": send_rows,
        "receive": recv_rows,
        "send_surplus": send_total,
        "receive_surplus": recv_total,
        "net_surplus_to_partner": round(send_total - recv_total, 2),
        "notes": notes,
    }


__all__ = ["keeper_costs", "keeper_block_for_package"]
