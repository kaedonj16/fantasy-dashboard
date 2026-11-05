"""League culture model (Frontier 6).

Advice engines tend to assume a generic league. Real leagues have
customs, and the customs change the right move: in a league where
every roster carries three QBs, your third QB is a trade asset, not
a wasted bench spot; in a league where defenses stream weekly, a
DST claim is a one-week rental and should be priced like one. The
customs are visible in data the site already touches: roster
composition (who hoards what) and the transaction history (wire
churn, defense streaming, trade activity).

Signals (each carries its evidence; a signal with no data behind it
is absent, never guessed):

* ``qb_hoarding``: mean QBs rostered per team vs the format's norm.
* ``wire_churn``: add/drop moves per team per week.
* ``dst_streaming``: share of add/drop moves that are defenses.
* ``trade_activity``: completed trades so far.

Thresholds are stated constants, not learned parameters: they name
the point where the custom is unmistakable in the evidence.
"""
from __future__ import annotations

from typing import Any, Optional

# Mean rostered QBs per team at which the league is hoarding: in a
# 1QB league a third QB is a stash, so a league averaging close to
# three is hoarding; superflex leagues legitimately carry more.
_QB_HOARD_MEAN_1QB = 2.6
_QB_HOARD_MEAN_SF = 3.6
# Defenses as a share of all add/drop player moves.
_DST_STREAM_SHARE = 0.12
_DST_STREAM_MIN_MOVES = 8
_CHURN_HIGH = 6.0     # moves per team per week
_CHURN_LOW = 2.0
_TRADES_ACTIVE = 8


def _is_superflex(ctx: dict) -> bool:
    rp = ctx.get("roster_positions") or []
    slots = [str(s).upper().replace(" ", "_") for s in rp]
    if any("SUPER" in s or s in ("QB_FLEX", "OP") for s in slots):
        return True
    return str(ctx.get("league_type") or "").lower() == "sf"


def _positions_by_id(ctx: dict) -> dict:
    out = {}
    players = ctx.get("players") or {}
    if isinstance(players, dict):
        for pid, row in players.items():
            if isinstance(row, dict):
                pos = row.get("position") or row.get("pos")
                if pos:
                    out[str(pid)] = str(pos).upper()
    if not out:
        for row in ctx.get("model_value_table") or []:
            if isinstance(row, dict):
                pid = str(row.get("player_id") or row.get("id") or "")
                pos = row.get("position") or row.get("pos")
                if pid and pos:
                    out[pid] = str(pos).upper()
    return out


def qb_hoarding(ctx: dict) -> Optional[dict]:
    """The QB-hoarding signal from roster composition alone."""
    rosters = [r for r in (ctx.get("rosters") or [])
               if isinstance(r, dict)]
    if len(rosters) < 4:
        return None
    positions = _positions_by_id(ctx)
    if not positions:
        return None
    counts = []
    for roster in rosters:
        n = sum(1 for pid in (roster.get("players") or [])
                if positions.get(str(pid)) == "QB")
        counts.append(n)
    if not counts:
        return None
    mean = sum(counts) / len(counts)
    threshold = _QB_HOARD_MEAN_SF if _is_superflex(ctx) \
        else _QB_HOARD_MEAN_1QB
    return {
        "mean_qbs_per_team": round(mean, 2),
        "threshold": threshold,
        "hoarding": mean >= threshold,
        "evidence": (f"Teams roster {mean:.1f} QBs on average "
                     f"across {len(counts)} rosters"),
    }


def transaction_signals(transactions: Any, *, num_teams: int = 0,
                        weeks: int = 0,
                        positions_by_id: Optional[dict] = None) -> dict:
    """Churn / streaming / trade-activity signals from raw rows."""
    out: dict = {}
    if not transactions:
        return out
    positions = positions_by_id or {}
    moves = 0
    dst_moves = 0
    trades = 0
    seen_weeks = set()
    for row in transactions:
        if not isinstance(row, dict):
            continue
        ttype = str(row.get("type") or "").lower()
        if row.get("week") is not None:
            seen_weeks.add(row.get("week"))
        if ttype == "trade":
            if str(row.get("status") or "complete").lower() in (
                    "complete", "completed"):
                trades += 1
            continue
        if ttype not in ("waiver", "free_agent", "add_drop", "add"):
            continue
        for pid in list((row.get("adds") or {}).keys()) \
                + list((row.get("drops") or {}).keys()):
            moves += 1
            if str(positions.get(str(pid)) or "").upper() in (
                    "DEF", "DST"):
                dst_moves += 1
    n_weeks = weeks or len(seen_weeks) or 1
    if moves and num_teams:
        per = moves / max(1, num_teams) / max(1, n_weeks)
        level = ("high" if per >= _CHURN_HIGH
                 else "low" if per <= _CHURN_LOW else "normal")
        out["wire_churn"] = {
            "moves_per_team_week": round(per, 2), "level": level,
            "evidence": (f"{moves} add/drop moves over {n_weeks} "
                         f"weeks across {num_teams} teams"),
        }
    if moves >= _DST_STREAM_MIN_MOVES:
        share = dst_moves / moves if moves else 0.0
        out["dst_streaming"] = {
            "dst_move_share": round(share, 3),
            "streaming": share >= _DST_STREAM_SHARE,
            "evidence": (f"{dst_moves} of {moves} add/drop moves "
                         "were defenses"),
        }
    if trades:
        out["trade_activity"] = {
            "completed_trades": trades,
            "active": trades >= _TRADES_ACTIVE,
            "evidence": f"{trades} completed trades this season",
        }
    return out


def culture_profile(ctx: dict, transactions: Any = None) -> dict:
    """The league's culture profile: only evidenced signals."""
    if not isinstance(ctx, dict):
        return {}
    out: dict = {}
    hoarding = qb_hoarding(ctx)
    if hoarding:
        out["qb_hoarding"] = hoarding
    if transactions:
        num_teams = len(ctx.get("rosters") or [])
        out.update(transaction_signals(
            transactions, num_teams=num_teams,
            positions_by_id=_positions_by_id(ctx)))
    return out


def fetch_culture(ctx: dict) -> dict:
    """Serve-time profile incl. transaction signals, memoized on
    the ctx and fail-soft to the roster-only profile."""
    if not isinstance(ctx, dict):
        return {}
    memo = ctx.get("_league_culture")
    if isinstance(memo, dict):
        return memo
    transactions = None
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
            transactions = []
            for week_rows in by_week.values():
                transactions.extend(week_rows or [])
    except Exception:
        transactions = None
    profile = culture_profile(ctx, transactions)
    try:
        ctx["_league_culture"] = profile
    except Exception:
        pass
    return profile


def claim_culture_note(profile: dict, claim: dict) -> Optional[str]:
    """The one-line bend a culture signal applies to a claim card.

    Only two bends exist, both factual: a QB claim in a hoarding
    league (the stash holds trade value there), and a DST claim in
    a streaming league (a one-week rental, priced like one).
    """
    if not profile or not isinstance(claim, dict):
        return None
    pos = str(claim.get("position") or "").upper()
    hoarding = profile.get("qb_hoarding") or {}
    if pos == "QB" and hoarding.get("hoarding"):
        return (f"QBs are hoarded in this league "
                f"({hoarding.get('mean_qbs_per_team')} per team): "
                "a stashed QB holds trade value here")
    streaming = profile.get("dst_streaming") or {}
    if pos in ("DEF", "DST") and streaming.get("streaming"):
        return ("Defenses stream weekly in this league: treat this "
                "as a one-week rental, not a hold")
    return None


__all__ = [
    "qb_hoarding",
    "transaction_signals",
    "culture_profile",
    "fetch_culture",
    "claim_culture_note",
]
