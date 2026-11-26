"""Autopilot scheduled runs (Phase 1 plan Step 7).

One entry point, run_scheduled_runs(), invoked by the
POST /api/cron/autopilot hook on the web service (triggered by the
Render cron services via scripts/trigger_autopilot.py, Thu evening and
Sun ~12:30 ET). The web service is where the ESPN credentials, VAPID
keys, and email config live, so the work runs there, not in the cron
container.

Per league with stored Autopilot settings and mode not Off:
- Access is re-checked through autopilot_access (pilot + PRO + usable);
  anything else is skipped silently (it renders no surface either).
- A fresh plan is built from the live roster.
- Auto leagues: the plan executes in full through autopilot_execute
  (the run is the authorization; every guardrail re-checks live).
- Approve leagues: the plan stays queued and the owner is notified
  that moves are waiting.

Off / paused / kill-switched leagues are skipped by the plan builder
and executor themselves; this runner only routes outcomes and fires
notifications. A failure in one league never stops the others.
"""
from __future__ import annotations

import logging
import threading
from datetime import datetime, time as _dtime, timedelta, timezone
from typing import Any, Callable, Dict, List, Optional
from zoneinfo import ZoneInfo

from dashboard_services import (
    autopilot_execute,
    autopilot_notify,
    autopilot_plan,
    autopilot_preview,
    autopilot_store as store,
)

logger = logging.getLogger(__name__)

ET = ZoneInfo("America/New_York")

#: The two weekly run windows, the app's single source of truth for
#: the Autopilot schedule. They mirror the Render cron services in
#: render.yaml (autopilot-thu: "0 0 * * 5", autopilot-sun: "30 16 * * 0";
#: Render schedules are UTC), so they are anchored in UTC as
#: (weekday, hour, minute) with Monday=0: Friday 00:00 UTC is the
#: Thursday-evening Eastern pass (19:00 EST / 20:00 EDT) and Sunday
#: 16:30 UTC is the Sunday Eastern pass (11:30 EST / 12:30 EDT).
RUN_WINDOWS_UTC = ((4, 0, 0), (6, 16, 30))


def next_run_after(now: Optional[datetime] = None) -> datetime:
    """The next scheduled run instant after ``now``, in Eastern time."""
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    now = now.astimezone(timezone.utc)
    best: Optional[datetime] = None
    for weekday, hh, mm in RUN_WINDOWS_UTC:
        cand = datetime.combine(
            now.date(), _dtime(hh, mm), tzinfo=timezone.utc
        ) + timedelta(days=(weekday - now.weekday()) % 7)
        if cand <= now:
            cand += timedelta(days=7)
        if best is None or cand < best:
            best = cand
    assert best is not None
    return best.astimezone(ET)


def next_run_label(now: Optional[datetime] = None) -> str:
    """User-facing next-run label, e.g. "Tomorrow, Thu 7:00 PM ET",
    "Today, Sun 12:30 PM ET", or "Sun, Nov 30 · 11:30 AM ET"."""
    nxt = next_run_after(now)
    now_et = (now or datetime.now(timezone.utc)).astimezone(ET)
    time_s = nxt.strftime("%-I:%M %p")
    day_s = nxt.strftime("%a")
    days = (nxt.date() - now_et.date()).days
    if days == 0:
        return f"Today, {day_s} {time_s} ET"
    if days == 1:
        return f"Tomorrow, {day_s} {time_s} ET"
    return f"{day_s}, {nxt.strftime('%b')} {nxt.day} · {time_s} ET"


# ---------------------------------------------------------------------------
# Per-league overlap lock: a scheduled pass and a manual "Run now" for
# the same league never run at once. Non-blocking: the second caller
# gets an "already_running" outcome instead of queueing up behind a
# full plan build.
# ---------------------------------------------------------------------------

_LEAGUE_LOCKS: Dict[tuple, threading.Lock] = {}
_LEAGUE_LOCKS_GUARD = threading.Lock()


def _league_key(account_id, platform, league_id, season) -> tuple:
    return (account_id, str(platform or "").lower(), str(league_id),
            int(season))


def _league_lock(key: tuple) -> threading.Lock:
    with _LEAGUE_LOCKS_GUARD:
        lock = _LEAGUE_LOCKS.get(key)
        if lock is None:
            lock = threading.Lock()
            _LEAGUE_LOCKS[key] = lock
        return lock


def _candidate_leagues() -> List[Dict[str, Any]]:
    """Every stored-settings league whose mode is not Off."""
    store.init_autopilot_tables()
    from dashboard_services.db import get_conn

    with get_conn() as conn:
        rows = conn.execute(
            """SELECT account_id, platform, league_id, season
               FROM autopilot_settings WHERE mode <> 'off'
               ORDER BY account_id, league_id"""
        ).fetchall()
    return [dict(r) for r in rows or []]


def run_league_pass(
    account_id, platform, league_id, season,
    *, transport: Optional[Any] = None,
) -> Dict[str, Any]:
    """One Autopilot pass for a single league: the exact code path the
    cron runs, shared with the page's "Run now" action.

    Builds a fresh plan, then follows the league's stored mode: Auto
    executes through autopilot_execute and notifies; Approve stores
    the plan and notifies that moves are waiting. Returns an outcome
    dict; never raises. Outcomes: "executed", "queued", "no_moves",
    "already_running", "no_access", "off", a build_plan status
    ("paused", "kill_switch", ...), or "error".
    """
    lock = _league_lock(_league_key(account_id, platform, league_id, season))
    if not lock.acquire(blocking=False):
        return {"outcome": "already_running", "league_id": str(league_id)}
    try:
        if not autopilot_preview.autopilot_access(
            account_id, platform, league_id, season
        ):
            return {"outcome": "no_access", "league_id": str(league_id)}
        settings = store.get_settings(account_id, platform, league_id, season)
        if str(settings.get("mode") or "off") == "off":
            return {"outcome": "off", "league_id": str(league_id)}
        built = autopilot_plan.build_plan(
            account_id, platform, league_id, season, transport=transport
        )
        status = built.get("status")
        if status not in ("planned", "no_moves"):
            return {"outcome": status, "league_id": str(league_id)}
        plan = built.get("plan") or {}
        swaps = plan.get("swaps") or []
        if not swaps:
            return {"outcome": "no_moves", "league_id": str(league_id)}
        if settings["mode"] == "auto":
            result = autopilot_execute.execute_plan(
                account_id, platform, league_id, season, plan["id"],
                transport=transport,
            )
            autopilot_notify.notify_execution(
                account_id, platform, league_id, season, result, plan
            )
            return {
                "outcome": "executed", "league_id": str(league_id),
                "swaps": len(swaps), "result_status": result.get("status"),
            }
        # approve: queue + tell the owner moves are waiting
        autopilot_notify.notify_plan_waiting(
            account_id, platform, league_id, season, plan
        )
        return {
            "outcome": "queued", "league_id": str(league_id),
            "swaps": len(swaps),
        }
    except Exception:
        logger.warning(
            "autopilot schedule: league %s failed", league_id, exc_info=True
        )
        return {"outcome": "error", "league_id": str(league_id)}
    finally:
        lock.release()


def run_scheduled_runs(
    *, transport_factory: Optional[Callable[..., Any]] = None,
) -> Dict[str, Any]:
    """Run one scheduled pass over every enabled league.

    ``transport_factory`` (tests) maps (account_id, league_id, season)
    to an ESPN transport; production passes None and espn_writes builds
    the real transport from the acting account's stored credentials.
    Returns a summary dict; never raises.
    """
    summary: Dict[str, Any] = {
        "leagues": 0, "planned": 0, "executed": 0, "queued": 0,
        "skipped": 0, "errors": 0, "details": [],
    }
    try:
        candidates = _candidate_leagues()
    except Exception:
        logger.warning("autopilot schedule: candidate query failed", exc_info=True)
        summary["errors"] += 1
        return summary
    for cand in candidates:
        account_id = cand["account_id"]
        platform = cand["platform"]
        league_id = cand["league_id"]
        season = cand["season"]
        summary["leagues"] += 1
        transport = None
        if transport_factory is not None:
            transport = transport_factory(account_id, league_id, season)
        outcome = run_league_pass(
            account_id, platform, league_id, season, transport=transport
        )
        o = outcome.get("outcome")
        detail: Dict[str, Any] = {"league_id": league_id, "outcome": o}
        if outcome.get("result_status") is not None:
            detail["result_status"] = outcome["result_status"]
        summary["details"].append(detail)
        if o == "executed":
            summary["planned"] += 1
            summary["executed"] += 1
        elif o == "queued":
            summary["planned"] += 1
            summary["queued"] += 1
        elif o == "error":
            summary["errors"] += 1
        elif o != "no_moves":
            summary["skipped"] += 1
    return summary
