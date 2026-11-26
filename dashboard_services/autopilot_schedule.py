"""Autopilot scheduled runs (Phase 1 plan Step 7).

One entry point, run_scheduled_runs(), invoked by the
POST /api/cron/autopilot hook on the web service (triggered by the
Render cron services via scripts/trigger_autopilot.py). The web
service is where the ESPN credentials, VAPID keys, and email config
live, so the work runs there, not in the cron container.

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

Schedule awareness
------------------
Run windows are derived from the week's REAL NFL slate (the nflverse
schedule rows the site already loads through
dashboard_services.nfl_game_data), not from two fixed clock times:

- Kickoffs are grouped into clusters: games starting within
  CLUSTER_GAP (2h) of each other share one window.
- A cluster's window opens RUN_LEAD (60 minutes) before its first
  kickoff, so the pass sees final inactives and still lands before
  lock. Two anchors keep their familiar times when they land close
  enough to a cluster's first kickoff (within ANCHOR_SNAP): the
  Thursday-evening pass and the Sunday late-morning pass, which are
  the legacy Render cron anchors in RUN_WINDOWS_UTC.
- A later cluster on the same day needs no own window when an earlier
  window that day is within SAME_DAY_COVERAGE of its first kickoff
  (the Sunday afternoon games ride the late-morning pass, exactly as
  before).
- Sunday-night and Monday clusters never get their own window: the
  Sunday pass sets those players while they are still unlocked.

That is what makes holiday weeks work: Thanksgiving week gets a
Wednesday-evening window (the Wed 20:00 game), a Thursday 12:00 PM ET
window ahead of the 13:00 kickoff (not 19:00, after two kickoffs), the
familiar Thursday-evening window for the night game, and a Friday
14:00 window for the Black Friday game. An international Sunday gets
its own early window ahead of the 9:30 AM ET kickoff while the main
slate keeps the late-morning pass.

Due-gating: run_scheduled_runs() only runs a league when a window is
actually due (window opened, its first kickoff still ahead) and the
window has not already been served for that league (checked against
the stored plans, see _window_served). The Render cron services fire
at every candidate trigger time the windows can need; firings that
find nothing due are cheap no-ops. When the schedule data is
unavailable the gate fails OPEN (the pass runs, as it did before
slate-derived windows): a spurious pass is harmless because the plan
builder pins locked players, while a missed pass strands a lineup.
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

#: The two legacy weekly run anchors, kept as the fallback when the
#: real slate cannot be loaded and as the snap targets that keep
#: normal weeks on their familiar times. They mirror the Render cron
#: services in render.yaml (autopilot-thu: "0 0 * * 5",
#: autopilot-sun: "30 16 * * 0"; Render schedules are UTC), so they are
#: anchored in UTC as (weekday, hour, minute) with Monday=0: Friday
#: 00:00 UTC is the Thursday-evening Eastern pass (19:00 EST /
#: 20:00 EDT) and Sunday 16:30 UTC is the Sunday Eastern pass
#: (11:30 EST / 12:30 EDT).
RUN_WINDOWS_UTC = ((4, 0, 0), (6, 16, 30))

#: A slate-derived window opens this long before its cluster's first
#: kickoff: late enough to see the final inactive lists (published
#: ~90 minutes before kickoff is too early to act on for the early
#: games; 60 minutes is the working compromise the schedule has
#: always used), early enough to finish before lock.
RUN_LEAD = timedelta(minutes=60)

#: Kickoffs this close together are one cluster and share a window
#: (the Sunday 13:00 slate; the 16:05/16:25 wave is the next cluster).
CLUSTER_GAP = timedelta(hours=2)

#: A cluster needs no own window when an earlier window the same day
#: opens within this long before its first kickoff (the Sunday
#: afternoon wave rides the late-morning pass; Thanksgiving's 16:30
#: game rides the noon pass).
SAME_DAY_COVERAGE = timedelta(hours=5)

#: A legacy anchor replaces a cluster's derived window when it opens
#: within this long before the cluster's first kickoff, so normal
#: weeks keep the familiar Thursday-evening / Sunday late-morning
#: times instead of drifting to kickoff-minus-60.
ANCHOR_SNAP = timedelta(minutes=120)


# ---------------------------------------------------------------------------
# The real slate (nflverse rows via nfl_game_data)
# ---------------------------------------------------------------------------

def _schedule_rows() -> List[dict]:
    """All nflverse schedule rows, [] when unavailable. Never raises."""
    try:
        from dashboard_services import nfl_game_data

        return nfl_game_data._nflverse_games_rows() or []
    except Exception:
        logger.debug("autopilot schedule rows unavailable", exc_info=True)
        return []


def _parse_games(rows: Optional[List[dict]] = None) -> List[Dict[str, Any]]:
    """Regular-season games from schedule rows as ET kickoffs.

    nflverse ``gameday`` is the local (Eastern) game date and
    ``gametime`` the Eastern wall-clock kickoff, so the pair parses
    directly in ET. Rows without a usable date/time are skipped.
    """
    if rows is None:
        rows = _schedule_rows()
    games: List[Dict[str, Any]] = []
    for row in rows or []:
        if str(row.get("game_type") or "").upper() != "REG":
            continue
        gameday = str(row.get("gameday") or "").strip()
        gametime = str(row.get("gametime") or "").strip()
        if not gameday or not gametime:
            continue
        try:
            kickoff = datetime.strptime(
                f"{gameday} {gametime}", "%Y-%m-%d %H:%M"
            ).replace(tzinfo=ET)
            season = int(float(str(row.get("season") or 0)))
            week = int(float(str(row.get("week") or 0)))
        except (TypeError, ValueError):
            continue
        if not season or not week:
            continue
        games.append({
            "season": season, "week": week, "kickoff": kickoff,
            "home": str(row.get("home_team") or "").strip(),
            "away": str(row.get("away_team") or "").strip(),
        })
    return games


def week_kickoffs(season: int, week: int,
                  rows: Optional[List[dict]] = None) -> List[datetime]:
    """Sorted ET kickoff instants for one regular-season week."""
    return sorted(
        g["kickoff"] for g in _parse_games(rows)
        if g["season"] == int(season) and g["week"] == int(week)
    )


def kickoffs_by_team(season: int, week: int,
                     rows: Optional[List[dict]] = None) -> Dict[str, datetime]:
    """{canonical team code: ET kickoff} for one week.

    Team codes are canonicalized with utils.nfl.norm_sched_team (the
    same normalization the start-score bundles and matchup tables
    use), so nflverse codes (LA, WSH, JAC) and site codes (LAR, WAS,
    JAX) meet in the middle. Empty when the slate is unavailable:
    callers must treat "no kickoff known" as "not schedule-locked",
    never as locked.
    """
    try:
        from utils.nfl import norm_sched_team
    except Exception:
        def norm_sched_team(t):  # type: ignore[misc]
            return str(t or "").upper().strip()

    out: Dict[str, datetime] = {}
    for g in _parse_games(rows):
        if g["season"] != int(season) or g["week"] != int(week):
            continue
        for team in (g["home"], g["away"]):
            code = norm_sched_team(team)
            if code:
                out[code] = g["kickoff"]
    return out


def schedule_locked_teams(season: int, week: int, now: datetime,
                          rows: Optional[List[dict]] = None) -> set:
    """Canonical team codes whose game has kicked off by ``now``."""
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    return {
        team for team, kickoff in kickoffs_by_team(season, week, rows=rows).items()
        if kickoff <= now
    }


# ---------------------------------------------------------------------------
# Due windows derived from the slate
# ---------------------------------------------------------------------------

def _clusters(kickoffs: List[datetime]) -> List[List[datetime]]:
    clusters: List[List[datetime]] = []
    for ko in sorted(kickoffs):
        if clusters and ko - clusters[-1][-1] <= CLUSTER_GAP:
            clusters[-1].append(ko)
        else:
            clusters.append([ko])
    return clusters


def _legacy_anchor_for(kickoff: datetime) -> Optional[datetime]:
    """The legacy anchor instant for a Thursday / Sunday kickoff date.

    Thursday clusters anchor to Friday 00:00 UTC (19:00 EST /
    20:00 EDT Thursday evening); Sunday clusters to Sunday 16:30 UTC
    (11:30 EST / 12:30 EDT). Other weekdays have no anchor.
    """
    day = kickoff.date()
    if kickoff.weekday() == 3:  # Thursday
        return datetime.combine(
            day + timedelta(days=1), _dtime(0, 0), tzinfo=timezone.utc
        ).astimezone(ET)
    if kickoff.weekday() == 6:  # Sunday
        return datetime.combine(
            day, _dtime(16, 30), tzinfo=timezone.utc
        ).astimezone(ET)
    return None


def _due_windows_for_games(games: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Due windows for one (season, week) game's list. See the module
    docstring for the rules; each window is
    {"start", "kickoff", "season", "week"} with ET datetimes, where
    "kickoff" is the first kickoff the window serves."""
    if not games:
        return []
    season = games[0]["season"]
    week = games[0]["week"]
    windows: List[Dict[str, Any]] = []
    for cluster in _clusters([g["kickoff"] for g in games]):
        first = cluster[0]
        # Sunday-night and Monday clusters never get their own window:
        # the Sunday pass sets those players while they are unlocked.
        if first.weekday() == 0:  # Monday
            continue
        if first.weekday() == 6 and first.hour >= 18:  # Sunday night
            continue
        anchor = _legacy_anchor_for(first)
        if anchor is not None and first - ANCHOR_SNAP <= anchor < first:
            if all(w["start"] != anchor for w in windows):
                windows.append({
                    "start": anchor, "kickoff": first,
                    "season": season, "week": week,
                })
            continue
        covered = any(
            w["start"].date() == first.date()
            and w["start"] < first
            and first - w["start"] <= SAME_DAY_COVERAGE
            for w in windows
        )
        if covered:
            continue
        windows.append({
            "start": first - RUN_LEAD, "kickoff": first,
            "season": season, "week": week,
        })
    windows.sort(key=lambda w: w["start"])
    return windows


def due_windows(season: int, week: int,
                rows: Optional[List[dict]] = None) -> List[Dict[str, Any]]:
    """Slate-derived due windows for one regular-season week."""
    games = [
        g for g in _parse_games(rows)
        if g["season"] == int(season) and g["week"] == int(week)
    ]
    return _due_windows_for_games(games)


def _upcoming_windows(now: datetime,
                      rows: Optional[List[dict]] = None) -> List[Dict[str, Any]]:
    """Due windows for every week with a kickoff near ``now``."""
    games = _parse_games(rows)
    horizon_lo = now - timedelta(days=1)
    horizon_hi = now + timedelta(days=14)
    keys = {
        (g["season"], g["week"]) for g in games
        if horizon_lo <= g["kickoff"] <= horizon_hi
    }
    out: List[Dict[str, Any]] = []
    for season, week in sorted(keys):
        out.extend(_due_windows_for_games([
            g for g in games if g["season"] == season and g["week"] == week
        ]))
    out.sort(key=lambda w: w["start"])
    return out


def due_window_for(season: int, now: Optional[datetime] = None,
                   rows: Optional[List[dict]] = None) -> Optional[Dict[str, Any]]:
    """The window due right now for a season, or None.

    A window is due from its start until the first kickoff it serves:
    a late firing inside that span still runs; once the kickoff has
    passed the window's players are locked and the next window owns
    the rest of the week.
    """
    now = _as_utc(now)
    for w in _upcoming_windows(now, rows=rows):
        if w["season"] != int(season):
            continue
        if w["start"] <= now < w["kickoff"]:
            return w
    return None


# ---------------------------------------------------------------------------
# Next run (slate-derived; legacy anchors as the fallback)
# ---------------------------------------------------------------------------

def _as_utc(now: Optional[datetime]) -> datetime:
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    return now.astimezone(timezone.utc)


def _legacy_next_run_after(now: datetime) -> datetime:
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


def next_run_after(now: Optional[datetime] = None,
                   rows: Optional[List[dict]] = None) -> datetime:
    """The next scheduled run instant after ``now``, in Eastern time.

    Slate-derived when the week's real schedule is available (a
    currently-due window counts: it is the run about to happen);
    the legacy anchors answer when no slate can be loaded or the
    season is over.
    """
    now = _as_utc(now)
    try:
        future = [
            w for w in _upcoming_windows(now, rows=rows)
            if w["kickoff"] > now
        ]
    except Exception:
        logger.debug("autopilot next-run slate lookup failed", exc_info=True)
        future = []
    if future:
        return min(future, key=lambda w: w["start"])["start"].astimezone(ET)
    return _legacy_next_run_after(now)


def next_run_label(now: Optional[datetime] = None,
                   rows: Optional[List[dict]] = None) -> str:
    """User-facing next-run label, e.g. "Tomorrow, Thu 7:00 PM ET",
    "Today, Thu 12:00 PM ET", or "Sun, Nov 30 · 11:30 AM ET"."""
    nxt = next_run_after(now, rows=rows)
    now_et = _as_utc(now).astimezone(ET)
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


def _window_served(account_id, platform, league_id, season,
                   window: Dict[str, Any]) -> bool:
    """Has this due window already been served for this league?

    Answered from the stored plans (no parallel tracking): a plan
    built at or after the window opened, for the window's week. For
    Auto leagues only a plan that reached execution counts (a plan
    the owner merely viewed being built on the page must not suppress
    the scheduled execution); for Approve leagues any plan built in
    the window counts, because the pass's whole job is to build,
    queue, and notify once. Any store failure answers False (fail
    open: run the pass).
    """
    try:
        settings = store.get_settings(account_id, platform, league_id, season)
        mode = str(settings.get("mode") or "off")
        if mode == "auto":
            statuses: Optional[tuple] = (
                "executed", "partial", "failed", "already_applied")
        else:
            statuses = None
        return store.plan_exists_since(
            account_id, platform, league_id, season,
            int(window["week"]), window["start"], statuses=statuses,
        )
    except Exception:
        logger.debug("autopilot served check failed", exc_info=True)
        return False


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
    *, now: Optional[datetime] = None,
    transport_factory: Optional[Callable[..., Any]] = None,
) -> Dict[str, Any]:
    """Run one scheduled pass over every enabled league.

    Due-gated against the slate-derived windows: a league runs only
    when a window for its season is due and not already served for
    that league (outcomes "not_due" / "already_served"); with no
    schedule data the gate fails open and every league runs, as
    before. Manual "Run now" never comes through here and is never
    gated.

    ``transport_factory`` (tests) maps (account_id, league_id, season)
    to an ESPN transport; production passes None and espn_writes builds
    the real transport from the acting account's stored credentials.
    Returns a summary dict; never raises.
    """
    summary: Dict[str, Any] = {
        "leagues": 0, "planned": 0, "executed": 0, "queued": 0,
        "skipped": 0, "errors": 0, "not_due": 0, "already_served": 0,
        "details": [],
    }
    now = _as_utc(now)
    try:
        rows = _schedule_rows()
    except Exception:
        rows = []
    try:
        candidates = _candidate_leagues()
    except Exception:
        logger.warning("autopilot schedule: candidate query failed", exc_info=True)
        summary["errors"] += 1
        return summary
    due_by_season: Dict[int, Optional[Dict[str, Any]]] = {}
    for cand in candidates:
        account_id = cand["account_id"]
        platform = cand["platform"]
        league_id = cand["league_id"]
        season = cand["season"]
        summary["leagues"] += 1
        if rows:
            if int(season) not in due_by_season:
                due_by_season[int(season)] = due_window_for(
                    int(season), now, rows=rows)
            window = due_by_season[int(season)]
            if window is None:
                summary["not_due"] += 1
                summary["skipped"] += 1
                summary["details"].append(
                    {"league_id": league_id, "outcome": "not_due"})
                continue
            if _window_served(account_id, platform, league_id, season, window):
                summary["already_served"] += 1
                summary["skipped"] += 1
                summary["details"].append(
                    {"league_id": league_id, "outcome": "already_served"})
                continue
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
