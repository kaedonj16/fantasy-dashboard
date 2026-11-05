"""Trade-suggestion extras (Parity 3).

The suggestions engine (context_builders.build_trade_suggestions_
context) prices a deal and estimates acceptance. Four judgments it
did not make, added here as one post-pass over the finished list so
the engine itself stays untouched:

* Coverage justification: which of the viewer's future starter
  holes (the GM brief's coverage map) the target's position
  addresses, named by week.
* Asset conflicts across the list: the same viewer asset spent in
  two live suggestions is flagged on both — one asset, one plan
  (the brief's sequencing rule, applied to the suggestion list).
* Counter ladders: for each deal, the sweetener ceiling (how much
  more the viewer can add before the deal leaves the shared fair
  band) with named bench sweeteners under it, and the refuse-
  outright rules (a counter that removes the target, or that asks
  for a designated keeper, is a non-starter).
* Race / deadline weighting: with a known trade deadline from the
  league-rules record, late-window deals with contending partners
  (buyers pay up late) and re-tooling partners (sellers move late)
  earn a bounded score weight. Unknown format or deadline: no
  weight, no note — fail closed.

Everything is annotation except the race weight, which adjusts
``suggestion_score`` by the stated points and re-sorts.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)

FAIR_BAND_MAX = 1.40   # mirrors trade_partner_checks.FAIR_BAND_MAX
_LATE_WINDOW_WEEKS = 2
_LATE_CONTENDER = 0.4
_LATE_SELLER = 0.2


def _pid(asset: Any) -> str:
    if not isinstance(asset, dict):
        return ""
    return str(asset.get("id") or asset.get("player_id") or "")


def _coverage_by_position(ctx: dict, viewer_roster_id) -> dict:
    """{position: [hole weeks]} from the brief's coverage map."""
    try:
        from dashboard_services.gm_brief import build_coverage

        coverage, _roles = build_coverage(ctx, viewer_roster_id)
    except Exception:
        return {}
    out: dict = {}
    for hole in (coverage or {}).get("holes") or []:
        if not isinstance(hole, dict):
            continue
        pos = str(hole.get("position") or "").upper()
        week = hole.get("week")
        if pos and isinstance(week, int):
            out.setdefault(pos, []).append(week)
    for weeks in out.values():
        weeks.sort()
    return out


def _annotate_coverage(partner: dict, holes_by_pos: dict) -> None:
    targets = partner.get("targets_they_have") or []
    if not targets or not holes_by_pos:
        return
    pos = str((targets[0] or {}).get("position") or "").upper()
    weeks = holes_by_pos.get(pos) or []
    if not weeks:
        return
    text = f"Addresses your {pos} hole in Week {weeks[0]}"
    if len(weeks) > 1:
        text += f" (+{len(weeks) - 1} more week" \
                f"{'s' if len(weeks) > 2 else ''})"
    partner["coverage"] = {
        "position": pos, "hole_weeks": weeks[:4], "text": text,
    }


def _annotate_conflicts(partners: list) -> None:
    spent: dict = {}
    for partner in partners:
        for asset in partner.get("targets_viewer_sends") or []:
            pid = _pid(asset)
            if pid:
                spent.setdefault(pid, []).append(
                    str(partner.get("team_name") or ""))
    for partner in partners:
        conflicts = {}
        for asset in partner.get("targets_viewer_sends") or []:
            pid = _pid(asset)
            others = [n for n in spent.get(pid, [])
                      if n != str(partner.get("team_name") or "")]
            if pid and others:
                conflicts[pid] = sorted(set(others))
        if conflicts:
            partner["asset_conflicts"] = conflicts


def _sweetener_pool(ctx: dict, viewer_roster: dict,
                    values: dict, locked: set, spent_ids: set) -> list:
    starters = {str(p) for p in (viewer_roster.get("starters") or [])}
    pool = []
    for pid in (viewer_roster.get("players") or []):
        pid = str(pid)
        if pid in starters or pid in locked or pid in spent_ids:
            continue
        mv = values.get(pid) or {}
        try:
            value = float(mv.get("value") or mv.get("model_value")
                          or mv.get("trade_value") or 0)
        except (TypeError, ValueError):
            value = 0.0
        if value <= 0:
            continue
        pool.append({
            "id": pid,
            "name": str(mv.get("name") or pid),
            "position": str(mv.get("position") or "").upper(),
            "value": round(value, 1),
        })
    pool.sort(key=lambda r: r["value"])
    return pool


def _annotate_ladder(partner: dict, pool: list, locked_rows: list,
                     target: dict) -> None:
    give = float(partner.get("value_you_give") or 0)
    get = float(partner.get("value_you_get") or 0)
    ceiling = max(0.0, round(get * FAIR_BAND_MAX - give, 1))
    sweeteners = []
    running = 0.0
    for cand in pool:
        if len(sweeteners) >= 2 or running + cand["value"] > ceiling:
            continue
        sweeteners.append(cand)
        running += cand["value"]
    refuse = [{
        "rule": "target_must_remain",
        "text": (f"A counter without "
                 f"{target.get('name') or 'the target'} is a "
                 "different deal, not a counter"),
    }]
    if locked_rows:
        names = ", ".join(r["name"] for r in locked_rows[:3])
        refuse.append({
            "rule": "never_send_designated_keeper",
            "text": (f"Refuse any counter asking for {names}: "
                     "designated keepers do not move"),
        })
    partner["counter_ladder"] = {
        "ceiling_value": ceiling,
        "sweeteners": sweeteners,
        "refuse_outright": refuse,
    }


def _weeks_to_deadline(ctx: dict) -> Optional[int]:
    try:
        from dashboard_services.gm_brief import get_rules

        rules = get_rules(ctx)
    except Exception:
        return None
    if (rules or {}).get("source") in (None, "default_unverified"):
        return None
    deadline = (rules or {}).get("trade_deadline")
    if not deadline:
        return None
    try:
        from datetime import date, datetime

        if isinstance(deadline, datetime):
            deadline = deadline.date()
        elif isinstance(deadline, str):
            deadline = date.fromisoformat(deadline[:10])
        days = (deadline - date.today()).days
    except Exception:
        return None
    return max(0, days // 7) if days >= 0 else None


def _annotate_race(partner: dict, weeks_left: Optional[int]) -> float:
    if weeks_left is None or weeks_left > _LATE_WINDOW_WEEKS:
        return 0.0
    contention = str(partner.get("partner_contention") or "").lower()
    if contention in ("contender", "bubble"):
        points, note = _LATE_CONTENDER, (
            "Deadline pressure: a contending partner pays up "
            "inside the final two weeks")
    elif contention in ("rebuilder", "rebuilding", "seller"):
        points, note = _LATE_SELLER, (
            "Deadline pressure: a re-tooling partner moves "
            "players late")
    else:
        return 0.0
    partner["race_weight"] = {
        "points": points, "weeks_to_deadline": weeks_left,
        "note": note,
    }
    return points


def annotate_suggestions(ctx: dict, partners: list,
                         viewer_roster: dict, values: dict) -> list:
    """Run the four annotations over a finished suggestion list.

    Mutates and returns ``partners`` (re-sorted when a race weight
    moved a score). Any layer that cannot be computed honestly is
    simply absent from the entries it would have touched.
    """
    if not partners:
        return partners
    viewer_roster_id = str((viewer_roster or {}).get("roster_id") or "")
    holes = _coverage_by_position(ctx, viewer_roster_id) \
        if viewer_roster_id else {}
    try:
        from utils.league_rules import keeper_locked_ids

        locked = {str(p) for p in keeper_locked_ids(ctx)}
    except Exception:
        locked = set()
    locked_rows = []
    for pid in sorted(locked):
        mv = values.get(pid) or {}
        locked_rows.append({"id": pid,
                            "name": str(mv.get("name") or pid)})
    weeks_left = _weeks_to_deadline(ctx)
    weighted = False
    for partner in partners:
        if not isinstance(partner, dict):
            continue
        _annotate_coverage(partner, holes)
        target = (partner.get("targets_they_have") or [{}])[0]
        spent_ids = {_pid(a) for a in
                     (partner.get("targets_viewer_sends") or [])}
        pool = _sweetener_pool(ctx, viewer_roster, values, locked,
                               spent_ids)
        _annotate_ladder(partner, pool, locked_rows, target)
        points = _annotate_race(partner, weeks_left)
        if points:
            partner["suggestion_score"] = round(
                float(partner.get("suggestion_score") or 0) + points, 3)
            weighted = True
    _annotate_conflicts(partners)
    if weighted:
        partners.sort(
            key=lambda p: float(p.get("suggestion_score") or 0),
            reverse=True)
    return partners


__all__ = ["annotate_suggestions"]
