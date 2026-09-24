"""Backup/third-string QB signal for start/sit rows.

Reads the live Sleeper QB depth chart for a team and finds the effective
starter: the lowest depth_chart_order QB expected to play (same "will play"
rule the waiver depth-chart logic uses). When that is the QB2 or deeper, the
team's pass-catchers are catching passes from a backup, which is the signal
this chip carries.

Pure stdlib + utils.waiver_score so it stays importable without Flask.
"""
from __future__ import annotations

from typing import Optional

from utils.waiver_score import _will_play


def qb_situation_chip(team: Optional[str], depth_index: dict,
                      full_players: dict) -> Optional[dict]:
    """Return a ``{"label", "kind", "note"}`` chip dict, or None.

    Returns None when the QB1 is starting, when the chart is missing or
    incomplete, or on any error, so callers degrade to no chip instead of a
    wrong one.
    """
    if not team or not depth_index:
        return None
    try:
        group = (depth_index or {}).get((str(team).upper(), "QB")) or []
        ranked = []
        for g in group:
            try:
                order = int(g.get("depth_order"))
            except (TypeError, ValueError):
                continue
            ranked.append((order, g))
        if not ranked:
            return None
        ranked.sort(key=lambda t: t[0])
        starter = next((g for _, g in ranked if _will_play(g.get("status"))), None)
        if not starter:
            return None
        order = next(o for o, g in ranked if g is starter)
        if order <= 1:
            return None
        pid = str(starter.get("pid") or "")
        pdata = (full_players or {}).get(pid) or {}
        name = (pdata.get("full_name") or "").strip()
        if order == 2:
            return {"label": "Backup QB", "kind": "qb2",
                    "note": f"{name or 'The backup'} (QB2) is starting for "
                            f"{str(team).upper()} with the starter out"}
        return {"label": "3rd-string QB", "kind": "qb3",
                "note": f"{name or 'A third-stringer'} (QB{order}) is starting for "
                        f"{str(team).upper()} with the top QBs out"}
    except Exception:
        return None
