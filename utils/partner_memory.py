"""Per-manager acceptance memory (Frontier 4).

Wave 2's acceptance model conditions on the deal in front of the
partner: ratio, fit, surplus, contention, and a league-wide
frequency prior from the trade database. What it cannot see is the
partner himself: this manager always needs a QB back, that one has
never taken a 2-for-1, a third consistently wins his deals on value.
The league's own completed-trade history knows all of that. This
module reads it.

Shape of the evidence (provider transaction rows, Sleeper-shaped,
which the Fleaflicker adapter also normalizes to): a completed trade
carries ``roster_ids``, ``adds`` {player_id: receiving roster} and
``drops`` {player_id: sending roster}. From those, per partner:

* sample: completed trades involving the partner;
* QB appetite: share of his trades in which he RECEIVED a QB;
* shape history: how often he was the one-asset side of an uneven
  deal (sent 1, received 2+), which is the shape managers most
  often refuse on principle;
* value tilt: mean (value he sent) / (value he received) across his
  trades, priced at CURRENT model values. That is an approximation
  and the profile says so (``value_basis: "current_values"``):
  historical prices are not stored, so the tilt is only used when
  the sample is real and the tilt is clear.

The adjustment is deliberately small and bounded ([-8, +8] points
into the shared conditioned acceptance) and it FAILS CLOSED: fewer
than ``MIN_SAMPLE`` completed trades with this partner yields no
adjustment and no notes. Nothing here invents a tendency the
history does not show.
"""
from __future__ import annotations

from typing import Any, Optional

MIN_SAMPLE = 3
_MAX_ADJ = 8


def _sides(trade: dict) -> dict:
    """{roster_id: {"gets": [pid], "sends": [pid]}} for one trade row."""
    adds = trade.get("adds") or {}
    drops = trade.get("drops") or {}
    sides: dict = {}
    for pid, to_rid in adds.items():
        sides.setdefault(str(to_rid), {"gets": [], "sends": []})[
            "gets"].append(str(pid))
    for pid, from_rid in drops.items():
        sides.setdefault(str(from_rid), {"gets": [], "sends": []})[
            "sends"].append(str(pid))
    return sides


def completed_trades(transactions: Any) -> list:
    """Normalized completed trades from raw transaction rows."""
    out = []
    for row in transactions or []:
        if not isinstance(row, dict):
            continue
        if str(row.get("type") or "").lower() != "trade":
            continue
        status = str(row.get("status") or "complete").lower()
        if status not in ("complete", "completed"):
            continue
        sides = _sides(row)
        if len(sides) < 2:
            continue
        out.append({"sides": sides,
                    "picks": len(row.get("draft_picks") or [])})
    return out


def partner_profile(transactions: Any, partner_roster_id,
                    *, values_by_id: Optional[dict] = None,
                    positions_by_id: Optional[dict] = None) -> Optional[dict]:
    """This partner's trading tendencies, or None under the sample
    gate. See the module docstring for what each field means."""
    trades = [t for t in completed_trades(transactions)
              if str(partner_roster_id) in t["sides"]]
    n = len(trades)
    if n < MIN_SAMPLE:
        return None
    rid = str(partner_roster_id)
    qb_received = 0
    uneven_taken = 0
    uneven_offered = 0
    tilts = []
    for trade in trades:
        mine = trade["sides"][rid]
        gets, sends = mine["gets"], mine["sends"]
        if positions_by_id:
            if any(str(positions_by_id.get(pid) or "").upper() == "QB"
                   for pid in gets):
                qb_received += 1
        if len(sends) == 1 and len(gets) >= 2:
            uneven_taken += 1
        if len(gets) == 1 and len(sends) >= 2:
            uneven_offered += 1
        if values_by_id:
            sent = sum(float(values_by_id.get(pid) or 0) for pid in sends)
            got = sum(float(values_by_id.get(pid) or 0) for pid in gets)
            if sent > 0 and got > 0:
                tilts.append(sent / got)
    profile = {
        "sample": n,
        "qb_received_share": (round(qb_received / n, 2)
                              if positions_by_id else None),
        "uneven_taken_share": round(uneven_taken / n, 2),
        "uneven_offered_share": round(uneven_offered / n, 2),
        "value_tilt": (round(sum(tilts) / len(tilts), 3) if tilts
                       else None),
        "value_basis": "current_values" if tilts else None,
    }
    return profile


def profiles_by_roster(transactions: Any, *,
                       values_by_id: Optional[dict] = None,
                       positions_by_id: Optional[dict] = None) -> dict:
    """{roster_id: profile} for every partner over the sample gate."""
    rids = set()
    for trade in completed_trades(transactions):
        rids.update(trade["sides"].keys())
    out = {}
    for rid in rids:
        profile = partner_profile(
            transactions, rid, values_by_id=values_by_id,
            positions_by_id=positions_by_id)
        if profile:
            out[rid] = profile
    return out


def partner_acceptance_adj(profile: Optional[dict], *,
                           partner_receives_positions: Optional[list] = None,
                           partner_sends_count: int = 0,
                           partner_receives_count: int = 0,
                           ) -> tuple:
    """(points, notes): the bounded history adjustment for one deal,
    described entirely from the partner's seat.

    * He takes QBs: when the package hands him a QB and at least
      half his completed trades brought him one, +4.
    * The 2-for-1 shape: when he would send 1 and receive 2+, a
      history of taking that shape earns +3; never having taken it
      across a real sample costs -4.
    * Value tilt: a partner whose deals average sending clearly
      less than he receives (tilt <= 0.90) prices fair offers below
      his bar, -3; one who pays up (tilt >= 1.05) earns +3.
    """
    if not profile:
        return 0, []
    points = 0
    notes = []
    positions = [str(p or "").upper()
                 for p in (partner_receives_positions or [])]
    qb_share = profile.get("qb_received_share")
    if "QB" in positions and qb_share is not None and qb_share >= 0.5:
        points += 4
        notes.append(
            f"Has taken a QB in {int(qb_share * 100)}% of his "
            f"{profile['sample']} completed trades")
    if partner_sends_count == 1 and partner_receives_count >= 2:
        taken = profile.get("uneven_taken_share") or 0
        if taken > 0:
            points += 3
            notes.append("Has taken a 2-for-1 shape before")
        elif profile.get("sample", 0) >= 4:
            points -= 4
            notes.append(
                f"Has never taken a 2-for-1 in {profile['sample']} "
                "completed trades")
    tilt = profile.get("value_tilt")
    if tilt is not None:
        if tilt <= 0.90:
            points -= 3
            notes.append(
                "Historically receives more value than he sends "
                "(at current values)")
        elif tilt >= 1.05:
            points += 3
            notes.append(
                "Historically pays up in his deals (at current values)")
    return max(-_MAX_ADJ, min(_MAX_ADJ, points)), notes


def fetch_profiles(ctx: dict) -> dict:
    """Serve-time helper: this league's partner profiles, or {}.

    Fetches the league's completed-week transactions through the
    shared service cache and prices them at the ctx's model values.
    Memoized on the ctx; every failure is an empty dict (the caller
    simply runs without memory, exactly as before this layer).
    """
    if not isinstance(ctx, dict):
        return {}
    memo = ctx.get("_partner_memory")
    if isinstance(memo, dict):
        return memo
    out: dict = {}
    try:
        platform = str(ctx.get("platform") or "").strip().lower()
        league_id = ctx.get("league_id")
        season = ctx.get("season")
        week = int(ctx.get("current_week") or ctx.get("week") or 0)
        if platform and league_id is not None and season and week > 1:
            from dashboard_services.service import (
                get_transactions_by_week,
            )

            by_week = get_transactions_by_week(
                str(league_id), list(range(1, week)),
                platform=platform, season=int(season)) or {}
            rows = []
            for week_rows in by_week.values():
                rows.extend(week_rows or [])
            values, positions = {}, {}
            for row in ctx.get("model_value_table") or []:
                if not isinstance(row, dict):
                    continue
                pid = str(row.get("player_id") or row.get("id") or "")
                if not pid:
                    continue
                val = row.get("value_sf") or row.get("value") \
                    or row.get("value_1qb")
                try:
                    values[pid] = float(val) if val is not None else 0.0
                except (TypeError, ValueError):
                    values[pid] = 0.0
                positions[pid] = row.get("position") or row.get("pos")
            out = profiles_by_roster(
                rows, values_by_id=values or None,
                positions_by_id=positions or None)
    except Exception:
        out = {}
    try:
        ctx["_partner_memory"] = out
    except Exception:
        pass
    return out


__all__ = [
    "MIN_SAMPLE",
    "completed_trades",
    "partner_profile",
    "profiles_by_roster",
    "partner_acceptance_adj",
    "fetch_profiles",
]
