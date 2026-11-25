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
from typing import Any, Callable, Dict, List, Optional

from dashboard_services import (
    autopilot_execute,
    autopilot_notify,
    autopilot_plan,
    autopilot_preview,
    autopilot_store as store,
)

logger = logging.getLogger(__name__)


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
        try:
            if not autopilot_preview.autopilot_access(
                account_id, platform, league_id, season
            ):
                summary["skipped"] += 1
                summary["details"].append(
                    {"league_id": league_id, "outcome": "no_access"})
                continue
            built = autopilot_plan.build_plan(
                account_id, platform, league_id, season, transport=transport
            )
            status = built.get("status")
            if status not in ("planned", "no_moves"):
                summary["skipped"] += 1
                summary["details"].append(
                    {"league_id": league_id, "outcome": status})
                continue
            plan = built.get("plan") or {}
            swaps = plan.get("swaps") or []
            if not swaps:
                summary["details"].append(
                    {"league_id": league_id, "outcome": "no_moves"})
                continue
            summary["planned"] += 1
            settings = store.get_settings(account_id, platform, league_id, season)
            if settings["mode"] == "auto":
                result = autopilot_execute.execute_plan(
                    account_id, platform, league_id, season, plan["id"],
                    transport=transport,
                )
                autopilot_notify.notify_execution(
                    account_id, platform, league_id, season, result, plan
                )
                summary["executed"] += 1
                summary["details"].append({
                    "league_id": league_id, "outcome": "executed",
                    "result_status": result.get("status"),
                })
            else:  # approve: queue + tell the owner moves are waiting
                autopilot_notify.notify_plan_waiting(
                    account_id, platform, league_id, season, plan
                )
                summary["queued"] += 1
                summary["details"].append(
                    {"league_id": league_id, "outcome": "queued"})
        except Exception:
            logger.warning(
                "autopilot schedule: league %s failed", league_id, exc_info=True
            )
            summary["errors"] += 1
            summary["details"].append(
                {"league_id": league_id, "outcome": "error"})
    return summary
