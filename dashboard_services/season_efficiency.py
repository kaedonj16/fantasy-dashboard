"""Shared, cached league lineup-efficiency aggregation.

The Matchups "Lineup" tab (dashboard_services/pages/optimal_page.py) computes
actual-vs-optimal points per team per week. Standings, the team modal and the
Weekly Recap awards want the same numbers, so this module factors the per-team
season aggregation into one place and caches it.

Efficiency is actual starter points divided by the optimal-lineup points, using
final provider results and the league's roster rules, so it never changes once a
week is finalized. The cache is therefore keyed on the set of completed weeks:
it stays valid until a new week finalizes, at which point the key changes and the
result is recomputed.
"""
from __future__ import annotations

import logging
import time
from datetime import datetime

from utils.optimal_lineup import analyze_team_week

logger = logging.getLogger(__name__)

# key -> result. Small: one entry per (league, season, completed-weeks) snapshot.
_CACHE: dict = {}
_CACHE_MAX = 64


def _analyze_team_week(matchup_rows, rid, players, slots, *, week=None):
    """Actual/optimal for one roster using the Lineup tab's shared adapter."""
    out = analyze_team_week(matchup_rows, rid, players, slots, week=week)
    if not out.get("complete"):
        return out
    return {"week": week_int(out.get("week")), "actual": float(out["actual"]),
            "optimal": float(out["optimal"]), "missed": float(out["missed"]),
            "eff": (float(out["efficiency"]) if out.get("efficiency") is not None else None)}


def week_int(w):
    try:
        return int(w)
    except (TypeError, ValueError):
        return None


def compute_league_season_efficiency(ctx: dict) -> dict:
    """Return per-team season efficiency for the whole league.

    Shape::

        {
          "completed_weeks": [1, 2, ...],
          "by_rid": {
            "<rid>": {
              "actual": float, "optimal": float, "missed": float,
              "eff": float | None,          # 0-100, actual/optimal
              "weeks": [{"week": int, "actual": float, "optimal": float, "eff": float|None}],
            }
          }
        }

    Cached on (platform, league_id, season, completed-weeks). Returns an empty
    ``by_rid`` when no weeks are finalized yet.
    """
    from app import get_players_index_global
    from dashboard_services.platform_api import get_matchups
    from dashboard_services.pages.optimal_page import verified_completed_weeks

    platform = ctx.get("platform") or "sleeper"
    season = int(ctx.get("season") or datetime.now().year)
    league_id = ctx.get("league_id") or ""
    slots = ctx.get("roster_positions") or []
    rosters = ctx.get("rosters") or []
    roster_map = ctx.get("roster_map") or {}

    completed = verified_completed_weeks(
        ctx.get("df_weekly"),
        season_complete=bool(ctx.get("season_complete")),
        matchups_by_week=ctx.get("matchups_by_week"),
    )
    empty = {"completed_weeks": [], "by_rid": {}}
    if not completed:
        return {**empty, "state": "no_completed_week", "incomplete": {}}

    requested = ctx.get("efficiency_weeks")
    if requested:
        wanted = {week_int(w) for w in requested}
        completed = [w for w in completed if w in wanted]
        if not completed:
            return {**empty, "state": "no_completed_week", "incomplete": {}}

    cache_key = (platform, str(league_id), season, tuple(completed))
    cached = _CACHE.get(cache_key)
    # Complete results get a modest correction window; incomplete/provider
    # failures retry quickly instead of being frozen until another week ends.
    if cached is not None:
        ttl = 300 if cached["value"].get("state") == "complete" else 0
        if time.monotonic() - cached["at"] < ttl:
            return cached["value"]
    retry_incomplete = bool(cached and cached["value"].get("state") != "complete")

    players = get_players_index_global() or {}
    matchup_cache = ctx.setdefault("optimal_matchups_by_week", {})
    for week in completed:
        if retry_incomplete or week not in matchup_cache or matchup_cache.get(week) is None:
            try:
                matchup_cache[week] = get_matchups(platform, league_id, week, season) or []
            except (LookupError, ValueError, RuntimeError, OSError) as exc:
                logger.warning("season-efficiency matchup unavailable league=%s week=%s: %s",
                               league_id, week, exc)
                matchup_cache[week] = None

    by_rid, incomplete = {}, {}
    for roster in rosters:
        rid = str(roster.get("roster_id") or "")
        if not rid:
            continue
        weeks = []
        for w in completed:
            wk = _analyze_team_week(matchup_cache.get(w), rid, players, slots, week=w)
            if wk.get("complete") is False:
                incomplete.setdefault(rid, {})[w] = wk.get("reason") or "historical data incomplete"
            else:
                if wk["week"] is None:
                    wk["week"] = w
                weeks.append(wk)
        actual = sum(wk["actual"] for wk in weeks)
        optimal = sum(wk["optimal"] for wk in weeks)
        by_rid[rid] = {
            "actual": actual,
            "optimal": optimal,
            "missed": sum(wk["missed"] for wk in weeks),
            "eff": (actual / optimal * 100) if optimal > 0 else None,
            "weeks": weeks,
        }

    available = sum(bool(v["weeks"]) for v in by_rid.values())
    state = "complete" if available == len(by_rid) and not incomplete else ("partial" if available else "incomplete")
    result = {"completed_weeks": completed, "by_rid": by_rid, "state": state,
              "incomplete": incomplete}
    if len(_CACHE) >= _CACHE_MAX:
        _CACHE.clear()
    _CACHE[cache_key] = {"at": time.monotonic(), "value": result}
    return result
