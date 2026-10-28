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

The cache has two tiers with the same key and TTL discipline: an in-process
dict, and a JSON file in the shared CACHE_DIR so results survive the constant
worker recycles (one gunicorn worker, --max-requests). Complete results keep a
300s correction window. Incomplete/partial results previously had a 0s TTL,
which re-paid the full provider refetch (one uncached get_matchups per week)
on EVERY render of a standings/recap page; they now get an honest 60s TTL —
the stored state still reports itself incomplete/partial, it is simply not
recomputed more than once a minute. Both tiers are bounded at 64 entries with
oldest-first eviction (the in-process tier previously cleared wholesale).
"""
from __future__ import annotations

import json
import logging
import threading
import time
from datetime import datetime

from utils.optimal_lineup import analyze_team_week

logger = logging.getLogger(__name__)

# key -> {"at": monotonic, "value": result}. One entry per
# (league, season, completed-weeks) snapshot.
_CACHE: dict = {}
_CACHE_MAX = 64
_CACHE_LOCK = threading.Lock()
_COMPLETE_TTL = 300.0
_INCOMPLETE_TTL = 60.0

# Disk tier (second level, survives worker recycles). Tests may point this at
# a tmp file; None means CACHE_DIR / _DISK_CACHE_NAME.
_DISK_CACHE_FILE = None
_DISK_CACHE_NAME = "season_efficiency_cache.json"


def _ttl_for(result: dict) -> float:
    return _COMPLETE_TTL if (result or {}).get("state") == "complete" else _INCOMPLETE_TTL


def _disk_key(cache_key) -> str:
    platform, league_id, season, completed = cache_key
    return json.dumps([platform, str(league_id), season, list(completed)])


def _disk_path():
    if _DISK_CACHE_FILE is not None:
        return _DISK_CACHE_FILE
    from utils.data_cache import CACHE_DIR
    return CACHE_DIR / _DISK_CACHE_NAME


def _disk_entry(cache_key):
    """Return the disk entry {"at", "value"} for the key, fresh or stale."""
    try:
        from utils.data_cache import read_json
        data = read_json(str(_disk_path())) or {}
        entry = data.get(_disk_key(cache_key))
        if isinstance(entry, dict) and "value" in entry:
            return entry
    except Exception:
        logger.debug("season-efficiency disk cache read failed", exc_info=True)
    return None


def _disk_put(cache_key, result) -> None:
    try:
        from utils.data_cache import read_json, write_json
        path = str(_disk_path())
        data = read_json(path) or {}
        if not isinstance(data, dict):
            data = {}
        data[_disk_key(cache_key)] = {"at": time.time(), "value": result}
        if len(data) > _CACHE_MAX:
            for old_key, _ in sorted(
                data.items(), key=lambda kv: float((kv[1] or {}).get("at") or 0)
            )[: len(data) - _CACHE_MAX]:
                data.pop(old_key, None)
        write_json(path, data)
    except Exception:
        logger.debug("season-efficiency disk cache write failed", exc_info=True)


def _cache_lookup(cache_key):
    """Return (fresh_value, saw_incomplete).

    fresh_value: a cached result still inside its TTL, consulting the
    in-process tier first, then the disk tier (a disk hit is promoted into
    memory with its remaining TTL preserved); None when nothing fresh.
    saw_incomplete: either tier holds an entry (fresh or stale) whose state
    is not "complete" — the signal to refetch provider rows on recompute
    rather than trust the ctx's per-ctx matchup cache.
    """
    now = time.monotonic()
    with _CACHE_LOCK:
        cached = _CACHE.get(cache_key)
    saw_incomplete = bool(
        cached is not None and cached["value"].get("state") != "complete"
    )
    if cached is not None and now - cached["at"] < _ttl_for(cached["value"]):
        return cached["value"], saw_incomplete
    entry = _disk_entry(cache_key)
    if entry is not None:
        value = entry["value"]
        wall_at = float(entry.get("at") or 0)
        if value.get("state") != "complete":
            saw_incomplete = True
        if time.time() - wall_at < _ttl_for(value):
            with _CACHE_LOCK:
                _CACHE[cache_key] = {"at": now - (time.time() - wall_at), "value": value}
            return value, saw_incomplete
    return None, saw_incomplete


def _cache_put(cache_key, result) -> None:
    with _CACHE_LOCK:
        if cache_key not in _CACHE and len(_CACHE) >= _CACHE_MAX:
            oldest = min(_CACHE, key=lambda k: _CACHE[k]["at"])
            _CACHE.pop(oldest, None)
        _CACHE[cache_key] = {"at": time.monotonic(), "value": result}
    _disk_put(cache_key, result)


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
    fresh, retry_incomplete = _cache_lookup(cache_key)
    if fresh is not None:
        return fresh

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
    _cache_put(cache_key, result)
    return result
