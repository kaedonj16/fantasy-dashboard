"""Autopilot action endpoints (Phase 1).

The Autopilot page, the My Leagues card control, and the scheduled
cron hook all write through here:

    POST /api/autopilot/<platform>/<season>/<league_id>/mode
        {mode: off|approve|auto} - the shared per-league mode.
    POST /api/autopilot/<platform>/<season>/<league_id>/settings
        Any of lineups_enabled, problem_slots_only, notify_push,
        notify_email, paused, protected_player_ids.
    POST /api/autopilot/<platform>/<season>/<league_id>/execute
        {plan_id, groups: [swap indexes]} - runs autopilot_execute on
        the accepted groups, then notifies per the league's prefs.
    POST /api/autopilot/<platform>/<season>/<league_id>/replan
        Builds a fresh plan (superseding the open one) and returns it.
    POST /api/autopilot/<platform>/<season>/<league_id>/run-now
        Runs this league's scheduled pass immediately (the same
        per-league code path the cron calls) for the signed-in owner.
    POST /api/autopilot/kill-switch
        {on: bool} - the account-wide switch (store-level; surfaced
        on the Autopilot page).
    POST /api/cron/autopilot
        CRON_SECRET-authenticated scheduled run (Thu/Sun), dispatched
        on a background thread like /api/cron/notifications.

Every league endpoint is gated by autopilot_access for the session's
account and fails closed with a 404, exactly like the page: for anyone
else the surface does not exist. Execution itself re-checks every
guardrail inside autopilot_execute.
"""
from __future__ import annotations

import hmac
import logging
import os
import threading

from flask import Blueprint, jsonify, request, session

from extensions import limiter

logger = logging.getLogger(__name__)

autopilot_bp = Blueprint("autopilot", __name__)

_run_lock = threading.Lock()


def _gate(platform, season, league_id):
    """(account_id, None) when the session may act on this league, else
    (None, 404 response)."""
    account_id = session.get("account_id")
    if not account_id:
        return None, (jsonify({"error": "Not found"}), 404)
    try:
        from dashboard_services.autopilot_preview import autopilot_access

        if not autopilot_access(account_id, platform, league_id, season):
            return None, (jsonify({"error": "Not found"}), 404)
    except Exception:
        return None, (jsonify({"error": "Not found"}), 404)
    return account_id, None


def _json_body() -> dict:
    return request.get_json(silent=True) or {}


@autopilot_bp.route(
    "/api/autopilot/<platform>/<int:season>/<league_id>/mode", methods=["POST"]
)
@limiter.limit("30 per minute")
def api_autopilot_mode(platform: str, season: int, league_id: str):
    account_id, err = _gate(platform, season, league_id)
    if err:
        return err
    mode = str(_json_body().get("mode") or "").strip().lower()
    if mode not in ("off", "approve", "auto"):
        return jsonify({"ok": False, "error": "Invalid mode"}), 400
    from dashboard_services import autopilot_store as store

    settings = store.save_settings(
        account_id, platform, league_id, season, {"mode": mode}
    )
    return jsonify({"ok": True, "mode": settings["mode"]})


@autopilot_bp.route(
    "/api/autopilot/<platform>/<int:season>/<league_id>/settings", methods=["POST"]
)
@limiter.limit("30 per minute")
def api_autopilot_settings(platform: str, season: int, league_id: str):
    account_id, err = _gate(platform, season, league_id)
    if err:
        return err
    body = _json_body()
    updates = {}
    for key in ("lineups_enabled", "problem_slots_only", "notify_push",
                "notify_email", "paused"):
        if key in body:
            updates[key] = bool(body[key])
    if "protected_player_ids" in body:
        try:
            updates["protected_player_ids"] = [
                int(p) for p in (body["protected_player_ids"] or [])
            ]
        except (TypeError, ValueError):
            return jsonify({"ok": False, "error": "Invalid protected list"}), 400
    if not updates:
        return jsonify({"ok": False, "error": "No settings provided"}), 400
    from dashboard_services import autopilot_store as store

    try:
        settings = store.save_settings(
            account_id, platform, league_id, season, updates
        )
    except ValueError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400
    return jsonify({"ok": True, "settings": {
        k: settings[k] for k in (
            "mode", "lineups_enabled", "problem_slots_only", "notify_push",
            "notify_email", "paused", "protected_player_ids")
    }})


@autopilot_bp.route(
    "/api/autopilot/<platform>/<int:season>/<league_id>/execute", methods=["POST"]
)
@limiter.limit("12 per minute")
def api_autopilot_execute(platform: str, season: int, league_id: str):
    account_id, err = _gate(platform, season, league_id)
    if err:
        return err
    body = _json_body()
    try:
        plan_id = int(body.get("plan_id"))
        groups = [int(g) for g in (body.get("groups") or [])]
    except (TypeError, ValueError):
        return jsonify({"ok": False, "error": "Invalid plan or groups"}), 400
    from dashboard_services import (
        autopilot_execute,
        autopilot_notify,
        autopilot_store as store,
    )

    plan = store.get_plan(plan_id)
    result = autopilot_execute.execute_plan(
        account_id, platform, league_id, season, plan_id,
        accepted_groups=groups,
    )
    # Notifications never fail the execution response.
    autopilot_notify.notify_execution(
        account_id, platform, league_id, season, result, plan
    )
    return jsonify({"ok": True, "result": result})


@autopilot_bp.route(
    "/api/autopilot/<platform>/<int:season>/<league_id>/replan", methods=["POST"]
)
@limiter.limit("12 per minute")
def api_autopilot_replan(platform: str, season: int, league_id: str):
    account_id, err = _gate(platform, season, league_id)
    if err:
        return err
    from dashboard_services import autopilot_plan

    try:
        built = autopilot_plan.build_plan(account_id, platform, league_id, season)
    except Exception as exc:
        logger.warning("autopilot replan failed", exc_info=True)
        return jsonify({"ok": False, "error": str(exc)}), 502
    plan = built.get("plan") or {}
    return jsonify({
        "ok": True,
        "status": built.get("status"),
        "plan_id": plan.get("id"),
        "swaps": len(plan.get("swaps") or []),
    })


@autopilot_bp.route(
    "/api/autopilot/<platform>/<int:season>/<league_id>/run-now", methods=["POST"]
)
@limiter.limit("6 per minute")
def api_autopilot_run_now(platform: str, season: int, league_id: str):
    """Manual trigger of this league's scheduled pass, owner-session
    authenticated (the page's Run now button; NOT the cron hook).

    Runs the scheduler's own per-league code path: Auto executes the
    fresh plan, Approve builds + stores it and notifies. A run already
    in progress for the league (cron or manual) answers 409 instead
    of queueing a second heavy plan build behind it.
    """
    account_id, err = _gate(platform, season, league_id)
    if err:
        return err
    from dashboard_services import autopilot_schedule

    outcome = autopilot_schedule.run_league_pass(
        account_id, platform, league_id, season
    )
    o = outcome.get("outcome")
    if o == "already_running":
        return jsonify({
            "ok": False, "status": o,
            "error": "A run is already in progress for this league.",
        }), 409
    if o == "off":
        return jsonify({
            "ok": False, "status": o,
            "error": "Autopilot is off for this league.",
        }), 400
    if o == "no_access":
        return jsonify({"error": "Not found"}), 404
    if o == "error":
        return jsonify({
            "ok": False, "status": o,
            "error": "The run did not complete. Try again.",
        }), 502
    return jsonify({
        "ok": True, "status": o, "swaps": outcome.get("swaps", 0),
        "result_status": outcome.get("result_status"),
    })


@autopilot_bp.route("/api/autopilot/kill-switch", methods=["POST"])
@limiter.limit("30 per minute")
def api_autopilot_kill_switch():
    account_id = session.get("account_id")
    if not account_id:
        return jsonify({"error": "Not found"}), 404
    try:
        from dashboard_services.autopilot_preview import pilot_account_allowed

        if not pilot_account_allowed(account_id):
            return jsonify({"error": "Not found"}), 404
    except Exception:
        return jsonify({"error": "Not found"}), 404
    from dashboard_services import autopilot_store as store

    on = bool(_json_body().get("on"))
    return jsonify({"ok": True, "kill_switch": store.set_kill_switch(account_id, on)})


# ---------------------------------------------------------------------------
# Scheduled runs
# ---------------------------------------------------------------------------

def _cron_authorized(data: dict) -> bool:
    """Same convention as /api/cron/notifications: X-Admin-Secret header,
    or CRON_SECRET via the X-Cron-Secret header or the JSON body."""
    admin_secret = os.environ.get("ADMIN_SECRET", "")
    cron_secret = os.environ.get("CRON_SECRET", "")
    header_admin = request.headers.get("X-Admin-Secret", "") or ""
    header_cron = request.headers.get("X-Cron-Secret", "") or ""
    body_secret = str((data or {}).get("secret") or "")
    if admin_secret and header_admin and hmac.compare_digest(header_admin, admin_secret):
        return True
    if cron_secret:
        if header_cron and hmac.compare_digest(header_cron, cron_secret):
            return True
        if body_secret and hmac.compare_digest(body_secret, cron_secret):
            return True
    return False


def _run_schedule_background():
    try:
        from dashboard_services.autopilot_schedule import run_scheduled_runs

        summary = run_scheduled_runs()
        logger.info("[cron/autopilot] %s", summary)
    except Exception:
        logger.warning("[cron/autopilot] run failed", exc_info=True)
    finally:
        _run_lock.release()


@autopilot_bp.route("/api/cron/autopilot", methods=["POST"])
@limiter.limit("60 per minute")
def api_cron_autopilot():
    """Scheduled Autopilot pass (Thu evening + Sun ~12:30 ET crons).

    Answers 202 immediately and runs on a background thread, with the
    same non-blocking overlap guard as /api/cron/notifications, so a
    slow league read can never pile runs up.
    """
    data = _json_body()
    if not _cron_authorized(data):
        return jsonify({"error": "Forbidden"}), 403
    if not _run_lock.acquire(blocking=False):
        logger.info("[cron/autopilot] skipped, previous run still in progress")
        return jsonify({"ok": True, "skipped": "already_running"}), 202
    threading.Thread(target=_run_schedule_background, daemon=True).start()
    return jsonify({"ok": True, "queued": True}), 202
