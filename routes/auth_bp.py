"""
Auth / session routes.

Routes: /health, /set-viewer, /logout
"""
from __future__ import annotations

import logging
from datetime import datetime

from flask import (
    Blueprint, current_app, jsonify, make_response, redirect,
    render_template_string, request, session, url_for,
)

from extensions import limiter

import os
import secrets
import base64
import hashlib
from urllib.parse import urlencode

auth_bp = Blueprint("auth", __name__)
logger = logging.getLogger(__name__)


# ── Identify by username only (no league required) ────────────────────────────

@auth_bp.route("/api/identify", methods=["POST"])
@limiter.limit("30 per minute")
def api_identify():
    """Set viewer session from a Sleeper username alone - no league needed.
    Returns JSON {ok, username, user_id, leagues:[{league_id, name, season}]}.

    Rate limited: each call fans out to the Sleeper API (user lookup + league
    list), so an uncapped endpoint lets anyone burn our upstream quota and
    enumerate usernames. 30/min per IP is far above legitimate use (a handful
    of calls per session).
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

    # If Google is already signed in, bridge this Sleeper identity onto the
    # account so personal PRO follows the account path (safe: refuses steal).
    if session.get("account_id") and session.get("viewer_user_id"):
        try:
            from dashboard_services.accounts import link_platform_identity
            status = link_platform_identity(
                int(session["account_id"]), "sleeper",
                session["viewer_user_id"], session.get("viewer_username"),
            )
            if status == "conflict":
                # Same as Google sign-in: do not keep browsing as a Sleeper
                # identity that already belongs to another Google account.
                logger.warning(
                    "[identify] sleeper identity conflict for acct=%s uid=%s",
                    session.get("account_id"), session.get("viewer_user_id"),
                )
                for k in ("viewer_user_id", "viewer_username", "viewer_roster_id",
                          "viewer_display_name", "viewer_team_name"):
                    session.pop(k, None)
                return jsonify({
                    "error": "That Sleeper username is already linked to another account.",
                }), 409
        except Exception:
            logger.debug("[identify] sleeper bridge failed", exc_info=True)

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
        body_html = render_template_string(
            FORM_BODY,
            username=username,
            viewed_season=season,
            league=league_id,
            error="Could not match that username to a team in this league.",
            recent_updates=generate_recent_updates_html(),
            yahoo_enabled=False,
        )
        from app import render_page
        return render_page("BR Fantasy Dashboard", None, "home", body_html, lite_js=True)

    save_viewer_session(viewer)
    session["viewer_platform"] = platform
    # Provider authorization may update a league association only when Google
    # account authentication was already explicit. Never infer account_id from
    # the provider user/team identity.
    if session.get("account_id"):
        from dashboard_services.accounts import add_user_league
        add_user_league(
            session["account_id"], platform, league_id, season=season,
            team_id=viewer.get("viewer_roster_id"),
            name=(ctx.get("league") or {}).get("name"),
        )
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
            session["viewer_platform"] = "espn"
            return jsonify({"ok": True, "matched": False})
        return jsonify({"ok": False,
                        "error": "Could not match that username to a team in this league."}), 404

    save_viewer_session(viewer)
    session["viewer_platform"] = platform
    if session.get("account_id"):
        from dashboard_services.accounts import add_user_league
        add_user_league(
            session["account_id"], platform, league_id, season=season,
            team_id=viewer.get("viewer_roster_id"),
            name=(ctx.get("league") or {}).get("name"),
        )
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
    platform  = str(data.get("platform") or "").strip().lower()
    league_id = str(data.get("league_id") or "").strip()
    try:
        season = int(data.get("season")) if data.get("season") else None
    except (TypeError, ValueError):
        season = None

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

    # Team selection can complete a provider connection made while the Google
    # account was already active (notably ESPN OTP). Without an authenticated
    # account this remains a provider-only session and performs no account lookup.
    if session.get("account_id") and platform and league_id and season and roster_id:
        from dashboard_services.accounts import add_user_league
        add_user_league(
            session["account_id"], platform, league_id, season=season,
            team_id=roster_id, name=None,
        )

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
@auth_bp.route("/reset-user")
def logout():
    session.clear()
    # One canonical local sign-out/reset path: discard both account and platform
    # viewer markers without touching any database account or league records.
    #
    # The page must never strand the user on a blank screen if its JS stalls, so
    # it carries a visible message and a <meta refresh> fallback that navigates
    # home even when scripting fails; the cache purge is also time-boxed so a
    # hung caches.delete() can't block the redirect.
    response = make_response("""<!doctype html><html><head><meta charset="utf-8">
<meta http-equiv="refresh" content="3;url=/?signed_out=1">
<title>Signing out…</title>
<style>
  html,body{height:100%;margin:0}
  body{display:flex;align-items:center;justify-content:center;
       font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif;
       color:#334155;background:#f8fafc}
  @media (prefers-color-scheme: dark){body{color:#cbd5e1;background:#0f172a}}
  .so-wrap{text-align:center}
  .so-spin{width:26px;height:26px;margin:0 auto 12px;border-radius:50%;
           border:3px solid rgba(148,163,184,.35);border-top-color:#3b82f6;
           animation:so-spin 1s linear infinite}
  @keyframes so-spin{to{transform:rotate(360deg)}}
</style></head><body>
<div class="so-wrap"><div class="so-spin"></div><div>Signing out…</div></div>
<script>
try {
  localStorage.removeItem('saved_viewer');
  localStorage.removeItem('saved_account');
  // Session storage contains transient navigation/team hand-offs. Clearing it
  // prevents a second user inheriting a roster while preserving preferences
  // such as theme, which live in localStorage under unrelated keys.
  sessionStorage.clear();
} catch(_) {}
// The service worker caches navigations, so a logged-in page could otherwise be
// served from cache after logout. Fully tear the worker down: UNREGISTER it (so it
// can't control the next navigation) and purge all its caches, then land on a
// cache-busting URL that has no cached copy to serve. Time-boxed so a hung
// teardown can never strand the user on this screen.
var _went = false;
function _go(){ if (_went) return; _went = true; window.location.replace('/?signed_out=' + Date.now()); }
setTimeout(_go, 1500);  // hard cap: redirect even if teardown stalls
var _jobs = [];
try {
  if (navigator.serviceWorker && navigator.serviceWorker.getRegistrations) {
    _jobs.push(navigator.serviceWorker.getRegistrations().then(function(rs){
      return Promise.all(rs.map(function(r){ return r.unregister(); }));
    }));
  }
} catch(_) {}
try {
  if (window.caches && caches.keys) {
    _jobs.push(caches.keys().then(function(ks){
      return Promise.all(ks.map(function(k){ return caches.delete(k); }));
    }));
  }
} catch(_) {}
if (_jobs.length) { Promise.all(_jobs).then(_go, _go); } else { _go(); }
</script>
</body></html>""")

    # Google sign-in introduced a domain-scoped session cookie so OAuth survives
    # an apex/www host change. Browsers can retain the older host-only cookie
    # alongside it; Flask only expires the currently configured domain cookie,
    # allowing that legacy cookie to restore the authenticated session on the
    # next request. Explicitly expire the host-only variant as well. Flask's
    # session interface will add the domain-scoped deletion after this response.
    if current_app.config.get("SESSION_COOKIE_DOMAIN"):
        response.delete_cookie(
            current_app.config.get("SESSION_COOKIE_NAME", "session"),
            path=current_app.config.get("SESSION_COOKIE_PATH") or "/",
            secure=current_app.config.get("SESSION_COOKIE_SECURE", False),
            httponly=current_app.config.get("SESSION_COOKIE_HTTPONLY", True),
            samesite=current_app.config.get("SESSION_COOKIE_SAMESITE"),
        )

    return response


# ────────────────────────────────────────────────────────────────────────────
# Merged from routes/google_auth_bp.py: Google OAuth 2.0 sign-in for standalone accounts.
# ────────────────────────────────────────────────────────────────────────────

def _redirect_after_google(default: str):
    """Prefer a staged home PRO checkout over the usual post-login destination."""
    try:
        from routes.billing_bp import pending_checkout_resume_path
        resume = pending_checkout_resume_path()
        if resume:
            return redirect(resume)
    except Exception:
        logger.debug("[google_auth] pending checkout check failed", exc_info=True)
    return redirect(default)


_AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
_TOKEN_URL = "https://oauth2.googleapis.com/token"


def _google_configured() -> bool:
    return bool(
        os.environ.get("GOOGLE_CLIENT_ID")
        and os.environ.get("GOOGLE_CLIENT_SECRET")
        and os.environ.get("GOOGLE_REDIRECT_URI")
    )


@auth_bp.route("/auth/google")
def google_auth_start():
    """Begin Google OAuth; redirect to the consent page."""
    if not _google_configured():
        return (
            "<p>Google sign-in is not configured on this server. Set "
            "GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET, and GOOGLE_REDIRECT_URI.</p>",
            503,
        )
    # Diagnostic: the session cookie carrying `state` is host-scoped, so the host
    # serving this request must be able to send that cookie to the callback host.
    # Warn when the current host differs from GOOGLE_REDIRECT_URI's host, which is
    # the usual cause of "sign-in doesn't stick" (state set here, missing there).
    from urllib.parse import urlparse
    cur_host = (request.host or "").split(":")[0].lower()
    cb_host = (urlparse(os.environ.get("GOOGLE_REDIRECT_URI", "")).hostname or "").lower()
    if cb_host and cur_host and cur_host != cb_host:
        logger.warning(
            "[google_auth] host mismatch: starting on %s but callback goes to %s; "
            "the state cookie may not survive unless COOKIE_DOMAIN spans both.",
            cur_host, cb_host,
        )

    state = secrets.token_urlsafe(24)
    nonce = secrets.token_urlsafe(24)
    verifier = secrets.token_urlsafe(64)
    challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).rstrip(b"=").decode()
    session["google_oauth_state"] = state
    session["google_oauth_nonce"] = nonce
    session["google_pkce_verifier"] = verifier
    session["google_auth_intent"] = "onboarding" if request.args.get("intent") == "onboarding" else "login"
    from utils.safe_url import safe_local_url
    session["google_oauth_next"] = safe_local_url(request.args.get("next"), "/")
    params = {
        "client_id": os.environ["GOOGLE_CLIENT_ID"],
        "redirect_uri": os.environ["GOOGLE_REDIRECT_URI"],
        "response_type": "code",
        "scope": "openid email profile",
        "state": state,
        "nonce": nonce,
        "code_challenge": challenge,
        "code_challenge_method": "S256",
        "access_type": "online",
        "prompt": "select_account",
    }
    return redirect(f"{_AUTH_URL}?{urlencode(params)}")


@auth_bp.route("/auth/google/callback")
@limiter.limit("30 per minute")
def google_auth_callback():
    """Handle Google's redirect: verify state, exchange code, sign the user in.

    Rate limited: a legitimate sign-in hits this once; the cap only bites
    automated probing of the token-exchange path.
    """
    import requests

    if request.args.get("error"):
        return redirect("/?google_error=" + request.args.get("error"))

    code = request.args.get("code") or ""
    state = request.args.get("state") or ""
    stored_state = session.pop("google_oauth_state", None)
    stored_nonce = session.pop("google_oauth_nonce", None)
    verifier = session.pop("google_pkce_verifier", None)
    next_url = session.pop("google_oauth_next", "/") or "/"
    from utils.safe_url import safe_local_url
    next_url = safe_local_url(next_url, "/")
    if not stored_state or stored_state != state:
        # Distinguish the two failure modes so prod logs are actionable:
        #  - stored_state is None with an otherwise empty session => the session
        #    cookie set on /auth/google was NOT sent to this callback. That is
        #    almost always a host/domain split (e.g. state set on www., callback
        #    on the apex), which host-only cookies can't cross. Fix: set
        #    COOKIE_DOMAIN so the cookie is shared across the domain, and make the
        #    /auth/google host match GOOGLE_REDIRECT_URI's host.
        #  - stored_state present but different => a genuine stale/replayed state.
        cause = "cookie_lost" if stored_state is None else "state_replayed"
        logger.warning(
            "[google_auth] state check failed (%s): host=%s had_session_keys=%s referer=%s",
            cause, (request.host or "").split(":")[0],
            bool(list(session.keys())), request.headers.get("Referer", ""),
        )
        return redirect("/?google_error=state_mismatch&why=" + cause)
    if not code:
        return redirect("/?google_error=no_code")

    try:
        tok = requests.post(
            _TOKEN_URL,
            data={
                "code": code,
                "client_id": os.environ["GOOGLE_CLIENT_ID"],
                "client_secret": os.environ["GOOGLE_CLIENT_SECRET"],
                "redirect_uri": os.environ["GOOGLE_REDIRECT_URI"],
                "grant_type": "authorization_code",
                "code_verifier": verifier,
            },
            timeout=15,
        ).json()
        raw_id_token = tok.get("id_token")
        if not raw_id_token:
            logger.error("[google_auth] no id_token (keys=%s)", sorted(tok.keys()))
            return redirect("/?google_error=token_exchange_failed")
        from google.auth.transport import requests as google_requests
        from google.oauth2 import id_token
        info = id_token.verify_oauth2_token(
            raw_id_token, google_requests.Request(), os.environ["GOOGLE_CLIENT_ID"],
        )
        if info.get("nonce") != stored_nonce:
            logger.warning("[google_auth] nonce validation failed")
            return redirect("/?google_error=invalid_nonce")
    except Exception as exc:
        logger.error("[google_auth] token verification failed (%s)", type(exc).__name__)
        return redirect("/?google_error=token_exchange_failed")

    sub = info.get("sub")
    email = info.get("email")
    if not sub:
        return redirect("/?google_error=invalid_profile")

    from dashboard_services.accounts import (
        upsert_google_account, link_platform_identity, add_user_league,
    )
    account_id, account_created = upsert_google_account(sub, email, info.get("given_name"))
    if not account_id:
        return redirect("/?google_error=account_error")

    session["account_id"] = account_id
    session["account_email"] = email
    session["account_first_name"] = info.get("given_name") or ""
    session.permanent = True

    # Product analytics: first Google sign-in counts as signup, else login.
    from dashboard_services import analytics as _analytics
    _analytics.track_event(
        _analytics.EVENT_SIGNUP if account_created else _analytics.EVENT_LOGIN,
        account_id=int(account_id),
        path="/auth/google/callback",
    )

    if account_created and email:
        try:
            from utils.welcome_email import send_signup_welcome
            send_signup_welcome(
                int(account_id),
                email=email,
                first_name=info.get("given_name") or "",
            )
        except Exception:
            logger.warning("[google_auth] signup welcome email failed", exc_info=True)

    # Private provider onboarding is validated and encrypted before Google sign-in.
    # The browser session carries only an opaque one-time token; consume it now
    # and attach the league to the canonical Google account.
    pending_provider_token = session.pop("pending_provider_connection_token", None)
    if pending_provider_token:
        try:
            from dashboard_services.accounts import (
                consume_private_provider_connection, add_provider_league_connection,
            )
            pending_provider = consume_private_provider_connection(pending_provider_token)
            if pending_provider:
                provider = str(pending_provider.get("provider") or "espn").strip().lower()
                credentials = {
                    k: v for k, v in pending_provider.items()
                    if k not in {"provider", "league_id", "season", "name", "team_id"}
                }
                team_id = pending_provider.get("team_id")
                if provider == "fleaflicker" and not team_id:
                    try:
                        from dashboard_services.providers.registry import get_provider
                        from dashboard_services.providers.fleaflicker_api import resolve_fleaflicker_team_id
                        flea_provider = get_provider("fleaflicker")
                        users = flea_provider.get_users(
                            pending_provider["league_id"],
                            pending_provider["season"],
                            token=credentials.get("token"),
                        )
                        team_id = resolve_fleaflicker_team_id(
                            users, flea_user_id=credentials.get("flea_user_id"),
                        )
                    except Exception:
                        logger.warning("[google_auth] fleaflicker team resolution failed", exc_info=True)
                add_provider_league_connection(
                    account_id, provider, pending_provider["league_id"],
                    pending_provider["season"],
                    pending_provider.get("name") or f"{provider.title()} League",
                    "private", credentials=credentials, team_id=team_id,
                )
                if provider == "fleaflicker" and team_id:
                    try:
                        from routes.link_bp import _persist_fleaflicker_viewer
                        _persist_fleaflicker_viewer(
                            pending_provider["league_id"],
                            pending_provider["season"],
                            str(team_id),
                            token=credentials.get("token"),
                        )
                    except Exception:
                        logger.warning("[google_auth] fleaflicker viewer persist failed", exc_info=True)
                session.pop("onboarding_progress", None)
                return _redirect_after_google(
                    f"/{provider}/{pending_provider['season']}/"
                    f"{pending_provider['league_id']}/dashboard"
                )
        except Exception:
            logger.warning("[google_auth] pending provider attach failed", exc_info=True)

    # An existing Sleeper session may authorize Sleeper enrichment, but Google
    # sign-in does not implicitly attach every discovered provider league. Only
    # explicit league connections create durable user_leagues associations.
    viewer_user_id = session.get("viewer_user_id")
    if viewer_user_id:
        try:
            status = link_platform_identity(
                account_id, "sleeper", str(viewer_user_id), session.get("viewer_username"),
            )
            if status == "conflict":
                # Sleeper id already belongs to another Google account -- do not
                # steal it. Clear the unverified viewer so this Google session
                # doesn't keep browsing as that Sleeper identity.
                logger.warning(
                    "[google_auth] sleeper identity conflict for acct=%s uid=%s",
                    account_id, viewer_user_id,
                )
                for k in ("viewer_user_id", "viewer_username", "viewer_roster_id",
                          "viewer_display_name", "viewer_team_name"):
                    session.pop(k, None)
        except Exception:
            logger.warning("[google_auth] sleeper bridge failed", exc_info=True)

    # Select-league-then-login: if the user picked a league before signing in,
    # attach it now and drop them straight into it.
    pending = session.pop("pending_link", None)
    if isinstance(pending, dict) and pending.get("platform") and pending.get("league_id"):
        try:
            # Home Yahoo "Continue with Google" signs in first. Yahoo still has
            # to authorize before membership is verified and the league attached.
            if (
                str(pending.get("platform") or "").strip().lower() == "yahoo"
                and not session.get("yahoo_guid")
            ):
                params = {"league_id": str(pending["league_id"])}
                team_name = str(pending.get("username") or "").strip()
                if team_name:
                    params["team_name"] = team_name
                return redirect("/auth/yahoo?" + urlencode(params))
            add_user_league(
                account_id, pending["platform"], pending["league_id"],
                season=pending.get("season"), team_id=pending.get("team_id"),
                name=pending.get("name"),
            )
            if pending["platform"] == "yahoo":
                yahoo_guid = session.get("yahoo_guid")
                if yahoo_guid:
                    try:
                        link_platform_identity(account_id, "yahoo", str(yahoo_guid))
                        from dashboard_services.providers.yahoo_api import save_league_owner
                        lookup_season = pending.get("season")
                        if lookup_season is not None:
                            save_league_owner(
                                pending["league_id"], int(lookup_season), str(yahoo_guid),
                            )
                    except Exception:
                        logger.warning("[google_auth] yahoo pending link attach failed", exc_info=True)
            # Sleeper: resolve the typed username to a team and set the viewer
            # identity, so the dashboard is personalized (and their other Sleeper
            # leagues get bridged too) -- the home flow never set a viewer session.
            uname = pending.get("username")
            if pending["platform"] == "sleeper" and uname and not session.get("viewer_user_id"):
                try:
                    from app import (
                        get_league_ctx_from_cache, resolve_viewer_for_league, save_viewer_session,
                    )
                    lctx = get_league_ctx_from_cache("sleeper", pending["league_id"], pending.get("season"))
                    viewer = resolve_viewer_for_league(lctx.get("users"), lctx.get("rosters"), uname)
                    if viewer:
                        save_viewer_session(viewer)
                        vuid = viewer.get("viewer_user_id")
                        # Persist the verified per-league roster immediately;
                        # the asynchronous backfill is only for other leagues.
                        add_user_league(
                            account_id, "sleeper", pending["league_id"],
                            season=pending.get("season"),
                            team_id=viewer.get("viewer_roster_id"),
                            name=pending.get("name"),
                        )
                        if vuid:
                            link_platform_identity(account_id, "sleeper", str(vuid), uname)
                except Exception:
                    logger.warning("[google_auth] sleeper viewer resolve failed", exc_info=True)
            elif (pending.get("team_id") or pending.get("username")) and not session.get("viewer_roster_id"):
                try:
                    from app import (
                        get_league_ctx_from_cache, resolve_viewer_for_league, save_viewer_session,
                    )
                    lctx = get_league_ctx_from_cache(
                        pending["platform"], pending["league_id"], pending.get("season"),
                    )
                    # ESPN pickers pass roster/team id (not owner SWID) plus the
                    # team name as username -- both feed resolve_viewer so Scout
                    # and other personalized tabs unlock after Google sign-in.
                    viewer = resolve_viewer_for_league(
                        lctx.get("users"), lctx.get("rosters"),
                        pending.get("username") or "",
                        user_id=str(pending["team_id"]) if pending.get("team_id") else None,
                    )
                    if viewer:
                        save_viewer_session(viewer)
                        session["viewer_platform"] = pending["platform"]
                except Exception:
                    logger.warning("[google_auth] pending team viewer resolve failed", exc_info=True)
            dest = (
                f"/{pending['platform']}/{pending.get('season') or ''}"
                f"/{pending['league_id']}/dashboard"
            )
            checkout_plan = str(pending.get("checkout_plan") or "").strip()
            if checkout_plan in {"starter", "all_pro", "hall_of_fame"}:
                dest = (
                    f"/{pending['platform']}/{pending.get('season') or ''}"
                    f"/{pending['league_id']}/pricing?plan={checkout_plan}&checkout=1"
                )
            return _redirect_after_google(dest)
        except Exception:
            logger.warning("[google_auth] pending link attach failed", exc_info=True)

    # Login never waits on a fantasy provider. Choose from saved database
    # metadata; provider refresh happens after the application has rendered.
    try:
        from dashboard_services.accounts import get_post_login_destination
        destination = get_post_login_destination(account_id)
    except Exception:
        logger.warning("[google_auth] saved league destination unavailable", exc_info=True)
        destination = None
    return _redirect_after_google(destination or next_url)


# ────────────────────────────────────────────────────────────────────────────
# Merged from routes/yahoo_auth_bp.py: Yahoo OAuth 2.0 routes.
# ────────────────────────────────────────────────────────────────────────────

def _yahoo_configured() -> bool:
    return bool(
        os.environ.get("YAHOO_CLIENT_ID")
        and os.environ.get("YAHOO_CLIENT_SECRET")
        and os.environ.get("YAHOO_REDIRECT_URI")
    )


@auth_bp.route("/auth/yahoo")
def yahoo_auth_start():
    """Begin Yahoo OAuth flow.  Redirects to Yahoo consent page."""
    from dashboard_services.providers.yahoo_api import yahoo_enabled
    if not yahoo_enabled():
        # Yahoo is turned off (Fantasy API access pending) -- don't walk the user
        # into the "application not authorized" wall.
        return redirect("/?yahoo_error=unavailable")
    if not _yahoo_configured():
        return (
            "<p>Yahoo OAuth is not configured on this server. "
            "Set YAHOO_CLIENT_ID, YAHOO_CLIENT_SECRET, and YAHOO_REDIRECT_URI.</p>",
            503,
        )

    from dashboard_services.providers.yahoo_api import get_authorization_url

    # NOTE: don't try to force the request onto the redirect_uri's apex host here.
    # The site canonicalizes the other way (apex -> www at the edge), so redirecting
    # www -> apex just ping-pongs against that and yields ERR_TOO_MANY_REDIRECTS.
    # Yahoo returns to the apex callback, the edge bounces it to www (query intact),
    # and the state cookie set here on www is present there -- so the flow completes
    # on one host without any redirect of our own.

    league_id = (request.args.get("league_id") or "").strip()
    from utils.safe_url import safe_local_url
    next_url  = safe_local_url(request.args.get("next"), "/")
    team_name = (request.args.get("team_name") or "").strip()
    # reauth=1 means we're recovering from a 403 (wrong account) -- force Yahoo's
    # account chooser so the user can pick a different account instead of being
    # silently re-authorized as the same one and hitting the same 403.
    force_login = (request.args.get("reauth") or "").strip() in ("1", "true", "yes")

    # Send Yahoo only a short, opaque state token. A JSON blob (braces, quotes,
    # spaces) in the `state` parameter trips Yahoo's authorization endpoint and
    # bounces the user to its generic "uh-oh" page, so keep the real context in
    # the server session keyed to that token and hand Yahoo just the nonce.
    state = secrets.token_urlsafe(24)
    session["yahoo_oauth_state"] = state
    session["yahoo_oauth_ctx"]   = {
        "league_id": league_id,
        "next":      next_url,
        "team_name": team_name,
    }

    auth_url = get_authorization_url(state=state, force_login=force_login)
    logger.info("[yahoo-auth] redirecting to: %s", auth_url)
    return redirect(auth_url)


@auth_bp.route("/auth/yahoo/callback")
@limiter.limit("30 per minute")
def yahoo_auth_callback():
    """Handle Yahoo OAuth callback, exchange code for tokens.

    Rate limited: a legitimate sign-in hits this once; the cap only bites
    automated probing of the token-exchange path.
    """
    from dashboard_services.providers.yahoo_api import (
        exchange_code_for_tokens, save_tokens, save_league_owner, get_login_guid,
    )

    error = request.args.get("error")
    if error:
        logger.warning("[yahoo_auth] OAuth error: %s", error)
        return redirect(f"/?yahoo_error={error}")

    code     = request.args.get("code") or ""
    state    = request.args.get("state") or ""

    # Verify the opaque state matches what we stored, then recover the context
    # from the session (it was never sent to Yahoo).
    stored_state = session.pop("yahoo_oauth_state", None)
    if not stored_state or stored_state != state:
        logger.warning("[yahoo_auth] State mismatch - possible CSRF")
        return redirect("/?yahoo_error=state_mismatch")

    ctx_data  = session.pop("yahoo_oauth_ctx", {}) or {}
    league_id = ctx_data.get("league_id") or ""
    pending_link_league = session.pop("yahoo_link_league_id", None)
    from utils.safe_url import safe_local_url
    next_url  = safe_local_url(ctx_data.get("next"), "/")
    team_name = ctx_data.get("team_name") or ""

    try:
        tok = exchange_code_for_tokens(code)
    except Exception as exc:
        logger.error("[yahoo_auth] Token exchange failed: %s", exc)
        return redirect("/?yahoo_error=token_exchange_failed")

    guid          = tok.get("xoauth_yahoo_guid") or ""
    access_token  = tok.get("access_token") or ""
    refresh_token = tok.get("refresh_token") or ""
    expires_in    = int(tok.get("expires_in") or 3600)

    # The access token is the only thing we truly can't proceed without.
    if not access_token:
        logger.error(
            "[yahoo_auth] No access_token in token response (keys=%s)", sorted(tok.keys()),
        )
        return redirect("/?yahoo_error=invalid_token_response")

    # Yahoo's fspt-r token response usually omits xoauth_yahoo_guid, and the token
    # is forbidden from the user-identity resource, so resolve the guid from the
    # league instead. If even that fails, fall back to a stable synthetic id
    # derived from the token so login still completes and the token store /
    # league-owner mapping keep working (the guid is only an identifier).
    if not guid:
        guid = get_login_guid(access_token, league_id)
    if not guid:
        import hashlib
        guid = "ytok_" + hashlib.sha256(
            (refresh_token or access_token).encode("utf-8")
        ).hexdigest()[:32]
        logger.warning("[yahoo_auth] no guid from Yahoo; using synthetic id")

    save_tokens(guid, access_token, refresh_token, expires_in)

    # Store Yahoo identity in session -- guid only. Access tokens live in the DB
    # after save_tokens; do not persist the bearer in the session cookie.
    session["yahoo_guid"]         = guid
    session.pop("yahoo_access_token", None)
    session["viewer_username"]    = team_name or guid
    session.permanent             = True

    # If we have a league_id, record this guid as an authorized owner of the
    # league (so non-owner viewers and background jobs can fetch it later) and
    # redirect into the league dashboard.
    if league_id:
        from datetime import datetime
        from dashboard_services.api import get_nfl_state
        from dashboard_services.providers.yahoo_api import resolve_league_key, get_league, get_users
        nfl_state  = get_nfl_state() or {}
        season     = int(nfl_state.get("season") or datetime.now().year)

        # Resolve the league's real, season-specific key from the leagues this
        # account actually belongs to. This both (a) confirms access before we
        # drop the user on the dashboard (a build against an inaccessible league
        # 500s on an uncaught 403) and (b) finds the correct season, since Yahoo's
        # "nfl" game code only ever points at the current season's game -- a league
        # from any other season would otherwise fail even for a real member.
        resolved = resolve_league_key(access_token, league_id)
        status = resolved.get("status")
        if status == "found":
            if resolved.get("season"):
                season = int(resolved["season"])
        elif status == "absent":
            logger.warning("[yahoo_auth] league %s not in this account's leagues", league_id)
            return redirect("/?yahoo_error=league_access_denied")
        else:
            # Couldn't list the account's leagues -- fall back to a direct fetch so
            # a current-season league (which resolves as nfl.l.<id>) still works.
            try:
                get_league(season, league_id, access_token)
            except Exception as exc:
                logger.warning("[yahoo_auth] league %s not accessible for this token: %s", league_id, exc)
                return redirect("/?yahoo_error=league_access_denied")

        try:
            save_league_owner(league_id, season, guid)
        except Exception:
            logger.warning("[yahoo_auth] save_league_owner failed", exc_info=True)
        # Yahoo OAuth authorizes Yahoo only. Attach its verified league to an app
        # account iff Google was already explicitly authenticated in this
        # session; never resolve an account by Yahoo guid/league membership.
        if session.get("account_id"):
            team_id = None
            try:
                users = get_users(season, league_id, access_token) or []
                team_id = next((
                    str(user.get("roster_id")) for user in users
                    if str(user.get("user_id") or "") == str(guid)
                    and user.get("roster_id") is not None
                ), None)
                from dashboard_services.accounts import add_user_league, link_platform_identity
                link_platform_identity(session["account_id"], "yahoo", guid, team_name or None)
                add_user_league(
                    session["account_id"], "yahoo", league_id, season=season,
                    team_id=team_id, name=resolved.get("name"),
                )
            except Exception:
                logger.warning("[yahoo_auth] account league attach failed", exc_info=True)
        try:
            from routes.billing_bp import pending_checkout_resume_path
            resume = pending_checkout_resume_path()
            if resume:
                return redirect(resume)
        except Exception:
            logger.debug("[yahoo_auth] pending checkout check failed", exc_info=True)
        return redirect(f"/yahoo/{season}/{league_id}/dashboard")

    # Link-modal resume when OAuth started without league_id in the URL.
    if pending_link_league and session.get("account_id"):
        from datetime import datetime
        from dashboard_services.api import get_nfl_state
        from dashboard_services.providers.yahoo_api import resolve_league_key, get_users
        resume_id = str(pending_link_league)
        nfl_state = get_nfl_state() or {}
        season = int(nfl_state.get("season") or datetime.now().year)
        resolved = resolve_league_key(access_token, resume_id)
        if resolved.get("season"):
            season = int(resolved["season"])
        try:
            save_league_owner(resume_id, season, guid)
        except Exception:
            logger.warning("[yahoo_auth] save_league_owner failed", exc_info=True)
        try:
            users = get_users(season, resume_id, access_token) or []
            team_id = next((
                str(user.get("roster_id")) for user in users
                if str(user.get("user_id") or "") == str(guid)
                and user.get("roster_id") is not None
            ), None)
            from dashboard_services.accounts import add_user_league, link_platform_identity
            link_platform_identity(session["account_id"], "yahoo", guid, team_name or None)
            add_user_league(
                session["account_id"], "yahoo", resume_id, season=season,
                team_id=team_id, name=resolved.get("name"),
            )
        except Exception:
            logger.warning("[yahoo_auth] account league attach failed", exc_info=True)
        try:
            from routes.billing_bp import pending_checkout_resume_path
            resume = pending_checkout_resume_path()
            if resume:
                return redirect(resume)
        except Exception:
            logger.debug("[yahoo_auth] pending checkout check failed", exc_info=True)
        return redirect(f"/yahoo/{season}/{resume_id}/dashboard")

    if pending_link_league:
        from urllib.parse import quote
        return redirect(f"/portfolio?link_yahoo={quote(str(pending_link_league))}")

    return redirect(next_url)


@auth_bp.route("/api/yahoo-validate-league")
def api_yahoo_validate_league():
    """Validate a Yahoo league ID and return its name.
    Requires the user to have already completed OAuth (yahoo_guid in session).
    """
    from dashboard_services.providers.yahoo_api import yahoo_enabled
    if not yahoo_enabled():
        return jsonify({"ok": False, "error": "Yahoo connections are temporarily unavailable."}), 503

    league_id    = (request.args.get("league_id") or "").strip()
    from dashboard_services.providers.yahoo_api import resolve_session_yahoo_token
    _, access_token = resolve_session_yahoo_token(session)

    if not league_id:
        return jsonify({"ok": False, "error": "League ID required"}), 400

    if not access_token:
        # Return a special flag so the frontend knows it needs to start OAuth
        from dashboard_services.providers.yahoo_api import yahoo_oauth_start_url
        return jsonify({
            "ok": False,
            "needs_oauth": True,
            "auth_url": yahoo_oauth_start_url(league_id=league_id, next_url="/"),
        }), 401

    try:
        from dashboard_services.providers.yahoo_api import get_league, resolve_league_key
        from datetime import datetime
        from dashboard_services.api import get_nfl_state
        nfl_state = get_nfl_state() or {}
        season    = int(nfl_state.get("season") or datetime.now().year)
        # Resolve the real season-specific key first: Yahoo's "nfl" code only
        # reaches the current season, so a prior-season league would 403 for a
        # real member without this. "absent" => account genuinely isn't in it;
        # "unknown" => couldn't list, so fall through to a direct fetch.
        resolved = resolve_league_key(access_token, league_id)
        if resolved.get("status") == "absent":
            from dashboard_services.providers.yahoo_api import yahoo_oauth_start_url
            return jsonify({
                "ok": False, "needs_oauth": True,
                "auth_url": yahoo_oauth_start_url(league_id=league_id, reauth=True, next_url="/"),
                "error": ("That Yahoo account isn't in any league with ID " + league_id +
                          ". Check the league ID, or reconnect with the account that's in it."),
            }), 401
        if resolved.get("season"):
            season = int(resolved["season"])
        name   = resolved.get("name")
        if not name:
            league = get_league(season, league_id, access_token)
            name   = league.get("name")
        return jsonify({"ok": True, "league": {"name": name, "season": season}})
    except Exception as exc:
        msg = str(exc)
        logger.warning("[yahoo] validate league %s failed: %s", league_id, msg)
        from dashboard_services.providers.yahoo_api import yahoo_auth_error_kind, yahoo_oauth_start_url
        kind = yahoo_auth_error_kind(exc)
        # 401 token_expired / 403 wrong account -- drop stale session identity and
        # send the user back through OAuth.
        if kind in ("expired", "forbidden"):
            session.pop("yahoo_access_token", None)
            if kind == "forbidden":
                session.pop("yahoo_guid", None)
            return jsonify({
                "ok": False, "needs_oauth": True,
                "auth_url": yahoo_oauth_start_url(league_id=league_id, reauth=True, next_url="/"),
                "error": (
                    "Your Yahoo login expired. Reconnect Yahoo and try again."
                    if kind == "expired" else
                    ("That Yahoo account can't access league " + league_id +
                     ". Reconnect with the Yahoo account that's in this league.")
                ),
            }), 401
        return jsonify({
            "ok": False,
            "error": "Couldn't load that Yahoo league. Double-check the league ID.",
        }), 400


@auth_bp.route("/api/yahoo-debug")
def api_yahoo_debug():
    """Return Yahoo parse diagnostics for the current league (copy/paste for support).

    Requires Yahoo OAuth in this session. Enable server log lines with
    YAHOO_API_DEBUG=1 on the host. No access tokens are included in the response.
    """
    from dashboard_services.providers.yahoo_api import (
        diagnose_league, get_valid_access_token, yahoo_enabled,
    )
    if not yahoo_enabled():
        return jsonify({"ok": False, "error": "Yahoo connections are unavailable."}), 503

    league_id = (request.args.get("league_id") or "").strip()
    # /yahoo/<season>/<league_id>/... pages -- infer from the Referer when omitted.
    if not league_id:
        ref = request.referrer or ""
        for marker in ("/yahoo/", "/api/yahoo/"):
            if marker in ref:
                tail = ref.split(marker, 1)[-1].strip("/").split("/")
                if len(tail) >= 2 and tail[1].isdigit():
                    league_id = tail[1]
                    break

    access_token = ""
    guid = session.get("yahoo_guid") or ""
    if guid:
        access_token = get_valid_access_token(guid) or ""
    session.pop("yahoo_access_token", None)

    if not access_token:
        return jsonify({"ok": False, "error": "Yahoo OAuth required.", "needs_oauth": True}), 401
    if not league_id:
        return jsonify({"ok": False, "error": "league_id required (query param or Yahoo league URL)."}), 400

    from datetime import datetime
    from dashboard_services.api import get_nfl_state
    nfl_state = get_nfl_state() or {}
    season = int(request.args.get("season") or nfl_state.get("season") or datetime.now().year)

    try:
        from dashboard_services.providers.yahoo_api import resolve_league_key
        resolved = resolve_league_key(access_token, league_id)
        if resolved.get("season"):
            season = int(resolved["season"])
    except Exception:
        logger.debug("[yahoo-debug] resolve_league_key failed", exc_info=True)

    report = diagnose_league(season, league_id, access_token)
    report["yahoo_api_debug_logging"] = (
        (os.environ.get("YAHOO_API_DEBUG") or "").strip().lower() in ("1", "true", "yes", "on")
    )
    return jsonify(report)
