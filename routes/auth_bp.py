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
    """Set viewer session from a Sleeper username alone - no league needed.
    Returns JSON {ok, username, user_id, leagues:[{league_id, name, season}]}.
    """
    from dashboard_services.api import (
        get_sleeper_user_by_username as get_sleeper_user,
        get_sleeper_user_leagues,
    )
    from datetime import datetime as _dt
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

    session.permanent = True  # persist across browser restarts / notification taps (30-day lifetime)
    session["viewer_username"] = user.get("username") or username
    session["viewer_user_id"] = str(user.get("user_id") or "")

    # Fetch this user's leagues so the UI can offer a league picker
    leagues = []
    try:
        current_season = _dt.now().year
        raw = get_sleeper_user_leagues(session["viewer_user_id"], current_season)
        leagues = [
            {"league_id": str(lg.get("league_id", "")), "name": lg.get("name", "Unknown League"), "season": current_season}
            for lg in raw if lg.get("league_id")
        ]
    except Exception:
        logger.debug("suppressed exception", exc_info=True)

    return jsonify({
        "ok": True,
        "username": session["viewer_username"],
        "user_id": session["viewer_user_id"],
        "leagues": leagues,
    })


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

    # Return to the page the user was on when they signed in, if safe
    next_url = (request.form.get("next") or "").strip()
    if next_url and next_url.startswith("/") and not next_url.startswith("//"):
        return redirect(next_url)
    return redirect(url_for("page_dashboard", platform=platform, season=season, league_id=league_id))


# ── Full sign-in for a league (JSON, no navigation) ──────────────────────────

@auth_bp.route("/api/sign-in-league", methods=["POST"])
def api_sign_in_league():
    """Fully sign a viewer into a league and return JSON (no redirect).

    Same resolution as /set-viewer (username/team -> roster, full session via
    save_viewer_session), so the in-page "View in your league" flow leaves the
    user as genuinely logged in. ESPN viewer matching is optional, mirroring the
    home flow.
    """
    from app import (
        _background_seed_user, get_league_ctx_from_cache,
        resolve_viewer_for_league, save_viewer_session,
    )
    data = request.get_json(force=True) or {}
    platform  = (data.get("platform") or "sleeper").strip().lower()
    league_id = str(data.get("league_id") or "").strip()
    season    = int(data.get("season") or datetime.now().year)
    username  = str(data.get("username") or data.get("team_name") or "").strip()

    if not league_id:
        return jsonify({"ok": False, "error": "league_id required"}), 400

    try:
        ctx = get_league_ctx_from_cache(platform=platform, league_id=league_id, season=season)
    except Exception as exc:
        logger.warning("[sign-in-league] league load failed: %s", exc)
        return jsonify({"ok": False, "error": "Could not load that league."}), 400

    viewer = None
    if username:
        viewer = resolve_viewer_for_league(ctx.get("users") or [], ctx.get("rosters") or [], username)

    if not viewer:
        if platform == "espn":
            # ESPN doesn't have Sleeper-style usernames; a match is optional.
            session.permanent = True
            session["viewer_username"] = username or "ESPN Manager"
            return jsonify({"ok": True, "matched": False})
        return jsonify({"ok": False,
                        "error": "Could not match that username to a team in this league."}), 404

    save_viewer_session(viewer)
    if platform == "sleeper" and viewer.get("viewer_user_id"):
        _background_seed_user(viewer["viewer_user_id"], viewer.get("viewer_username"))

    return jsonify({
        "ok": True, "matched": True,
        "username":  viewer.get("viewer_username"),
        "user_id":   viewer.get("viewer_user_id"),
        "roster_id": viewer.get("viewer_roster_id"),
        "team_name": viewer.get("viewer_team_name"),
    })


# ── Quick-set viewer from localStorage (no league context fetch) ─────────────

@auth_bp.route("/api/quick-set-viewer", methods=["POST"])
def api_quick_set_viewer():
    """Set viewer session variables directly from trusted localStorage data.
    Skips get_league_ctx_from_cache entirely - used by the 'Continue as X'
    returning-user flow where we already know the viewer is valid.
    """
    data = request.get_json(force=True) or {}
    username  = str(data.get("username")  or "").strip()
    roster_id = str(data.get("roster_id") or "").strip()
    user_id   = str(data.get("user_id")   or "").strip()
    team_name = str(data.get("team_name") or "").strip()

    if not username:
        return jsonify({"ok": False, "error": "username required"}), 400

    session.permanent             = True
    session["viewer_username"]    = username
    if user_id:
        session["viewer_user_id"] = user_id
    if roster_id:
        session["viewer_roster_id"] = roster_id
    if team_name:
        session["viewer_team_name"] = team_name

    return jsonify({"ok": True})


# ── Set viewer roster (AJAX) ──────────────────────────────────────────────────

@auth_bp.route("/api/set-viewer-roster", methods=["POST"])
def api_set_viewer_roster():
    """Persist the selected roster_id to the session without a full page reload.
    Called by the team-selector dropdown in the trade calculator.
    """
    data = request.get_json(force=True) or {}
    roster_id = str(data.get("roster_id") or "").strip()
    if not roster_id:
        return jsonify({"error": "roster_id is required"}), 400
    session["viewer_roster_id"] = roster_id
    return jsonify({"ok": True, "roster_id": roster_id})


# ── Logout ────────────────────────────────────────────────────────────────────

@auth_bp.route("/logout")
def logout():
    session.clear()
    # Return a tiny page that strips sensitive fields from saved_viewer in
    # localStorage (so auto-restore can't silently re-login) while keeping just
    # the username so the "Continue as X" banner still shows on the home page.
    return """<!doctype html><html><head><meta charset="utf-8">
<title>Signing out…</title></head><body>
<script>
try {
  var sv = JSON.parse(localStorage.getItem('saved_viewer') || 'null');
  if (sv) {
    // Keep only enough for "Continue as X" — strip everything that would
    // allow auto-restore to silently re-authenticate.
    localStorage.setItem('saved_viewer', JSON.stringify({
      username:  sv.username  || '',
      league_id: sv.league_id || '',
      platform:  sv.platform  || 'sleeper',
      season:    sv.season    || '',
      team_name: sv.team_name || '',
      // Intentionally omit: user_id, roster_id (prevent auto-restore auth)
    }));
  }
  // Mark explicit logout so auto-restore skips this session.
  sessionStorage.setItem('_explicitLogout', '1');
} catch(_) {}
// In the installed PWA the service worker caches navigations, so without this a
// logged-in page could be served from cache after logout. Purge all caches,
// then go home so every later navigation re-fetches a fresh, logged-out page.
function _go(){ window.location.replace('/'); }
if (window.caches && caches.keys) {
  caches.keys()
    .then(function(ks){ return Promise.all(ks.map(function(k){ return caches.delete(k); })); })
    .then(_go, _go);
} else { _go(); }
</script>
</body></html>"""
