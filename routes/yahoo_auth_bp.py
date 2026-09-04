"""
Yahoo OAuth 2.0 routes.

Flow:
  1. User visits /auth/yahoo?league_id=<id>&next=<url>
     → stores state in session, redirects to Yahoo consent page
  2. Yahoo redirects to /auth/yahoo/callback?code=...&state=...
     → exchanges code for tokens, saves to DB, redirects to next URL
"""
from __future__ import annotations

import logging
import os
import secrets

from flask import (
    Blueprint, jsonify, redirect, request, session, url_for,
)

yahoo_auth_bp = Blueprint("yahoo_auth", __name__)
logger = logging.getLogger(__name__)


def _yahoo_configured() -> bool:
    return bool(
        os.environ.get("YAHOO_CLIENT_ID")
        and os.environ.get("YAHOO_CLIENT_SECRET")
        and os.environ.get("YAHOO_REDIRECT_URI")
    )


@yahoo_auth_bp.route("/auth/yahoo")
def yahoo_auth_start():
    """Begin Yahoo OAuth flow.  Redirects to Yahoo consent page."""
    from dashboard_services.providers.yahoo_api import yahoo_enabled
    if not yahoo_enabled():
        # Yahoo is turned off (Fantasy API access pending) — don't walk the user
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
    # and the state cookie set here on www is present there — so the flow completes
    # on one host without any redirect of our own.

    league_id = (request.args.get("league_id") or "").strip()
    from utils.safe_url import safe_local_url
    next_url  = safe_local_url(request.args.get("next"), "/")
    team_name = (request.args.get("team_name") or "").strip()
    # reauth=1 means we're recovering from a 403 (wrong account) — force Yahoo's
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


@yahoo_auth_bp.route("/auth/yahoo/callback")
def yahoo_auth_callback():
    """Handle Yahoo OAuth callback, exchange code for tokens."""
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

    # Store Yahoo identity in session — guid only. Access tokens live in the DB
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
        # "nfl" game code only ever points at the current season's game — a league
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
            # Couldn't list the account's leagues — fall back to a direct fetch so
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


@yahoo_auth_bp.route("/api/yahoo-validate-league")
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
        # 401 token_expired / 403 wrong account — drop stale session identity and
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


@yahoo_auth_bp.route("/api/yahoo-debug")
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
    # /yahoo/<season>/<league_id>/... pages — infer from the Referer when omitted.
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
