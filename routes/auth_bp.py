"""
Auth / session routes.

Routes: /health, /set-viewer, /logout
"""
from __future__ import annotations

import logging
from datetime import datetime

from flask import (
    Blueprint, jsonify, redirect, render_template_string,
    request, session, url_for,
)

auth_bp = Blueprint("auth", __name__)
logger = logging.getLogger(__name__)


# ── Identify by username only (no league required) ────────────────────────────

@auth_bp.route("/api/identify", methods=["POST"])
def api_identify():
    """Set viewer session from a Sleeper username alone — no league needed.
    Used by the subscribe flow so guests can log in without a league context.
    Returns JSON {ok: true, username, user_id} or {error: str}.
    """
    from dashboard_services.api import get_sleeper_user
    data = request.get_json(force=True) or {}
    username = str(data.get("username") or "").strip()
    if not username:
        return jsonify({"error": "Username is required"}), 400
    try:
        user = get_sleeper_user(username)
    except Exception:
        return jsonify({"error": "Could not reach Sleeper. Try again."}), 503
    if not user:
        return jsonify({"error": "Username not found on Sleeper"}), 404
    session["viewer_username"] = user.get("username") or username
    session["viewer_user_id"] = str(user.get("user_id") or "")
    return jsonify({"ok": True, "username": session["viewer_username"], "user_id": session["viewer_user_id"]})


# ── Health probe ──────────────────────────────────────────────────────────────

@auth_bp.route("/health")
def health():
    """Uptime / readiness probe used by Render and load balancers."""
    from dashboard_services.db import get_database_url
    db_ok = False
    try:
        import psycopg
        url = get_database_url()
        with psycopg.connect(url, connect_timeout=3) as conn:
            conn.execute("SELECT 1")
        db_ok = True
    except Exception as exc:
        logger.warning("[health] DB check failed: %s", exc)

    payload = {"status": "ok" if db_ok else "degraded", "db": db_ok}
    status_code = 200 if db_ok else 503
    return jsonify(payload), status_code


# ── Viewer session ────────────────────────────────────────────────────────────

@auth_bp.route("/set-viewer", methods=["POST"])
def set_viewer():
    from app import (
        FORM_BODY, _background_seed_user, generate_recent_updates_html,
        get_league_ctx_from_cache, resolve_viewer_for_league, save_viewer_session,
    )
    league_id = (request.form.get("league_id") or "").strip()
    username = (request.form.get("username") or "").strip()
    platform = (request.form.get("platform") or "sleeper").strip().lower()
    season = int(request.form.get("season") or datetime.now().year)

    if not league_id or not username:
        return redirect(url_for("index"))

    ctx = get_league_ctx_from_cache(platform=platform, league_id=league_id, season=season)
    viewer = resolve_viewer_for_league(ctx["users"], ctx["rosters"], username)

    if not viewer:
        return render_template_string(
            FORM_BODY,
            league=league_id,
            error="Could not match that username to a team in this league.",
            recent_updates=generate_recent_updates_html(),
        )

    save_viewer_session(viewer)
    if platform == "sleeper" and viewer.get("viewer_user_id"):
        _background_seed_user(viewer["viewer_user_id"], viewer.get("viewer_username"))
    return redirect(url_for("page_dashboard", platform=platform, season=season, league_id=league_id))


# ── Logout ────────────────────────────────────────────────────────────────────

@auth_bp.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("index"))
