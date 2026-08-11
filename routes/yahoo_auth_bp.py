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
    next_url  = (request.args.get("next") or "/").strip()
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
    next_url  = ctx_data.get("next") or "/"
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

    # Store Yahoo identity in session
    session["yahoo_guid"]         = guid
    session["yahoo_access_token"] = access_token
    session["viewer_username"]    = team_name or guid
    session.permanent             = True

    # If we have a league_id, record this guid as an authorized owner of the
    # league (so non-owner viewers and background jobs can fetch it later) and
    # redirect into the league dashboard.
    if league_id:
        from datetime import datetime
        from dashboard_services.api import get_nfl_state
        from dashboard_services.providers.yahoo_api import resolve_league_key, get_league
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
        return redirect(f"/yahoo/{season}/{league_id}/dashboard")

    return redirect(next_url)


@yahoo_auth_bp.route("/api/yahoo-validate-league")
def api_yahoo_validate_league():
    """Validate a Yahoo league ID and return its name.
    Requires the user to have already completed OAuth (yahoo_access_token in session).
    """
    from dashboard_services.providers.yahoo_api import yahoo_enabled
    if not yahoo_enabled():
        return jsonify({"ok": False, "error": "Yahoo connections are temporarily unavailable."}), 503

    league_id    = (request.args.get("league_id") or "").strip()
    access_token = session.get("yahoo_access_token") or ""

    if not league_id:
        return jsonify({"ok": False, "error": "League ID required"}), 400

    if not access_token:
        # Return a special flag so the frontend knows it needs to start OAuth
        return jsonify({"ok": False, "needs_oauth": True}), 401

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
            return jsonify({
                "ok": False, "needs_oauth": True,
                "auth_url": "/auth/yahoo?reauth=1",
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
        # 403 = the token is valid but the authorized Yahoo account can't see this
        # league (wrong account / not a member / wrong id or season). A stale
        # token would otherwise dead-end here forever, so drop it and send the
        # user back through OAuth to reconnect with the right account.
        if "403" in msg or "Forbidden" in msg:
            session.pop("yahoo_access_token", None)
            session.pop("yahoo_guid", None)
            return jsonify({
                "ok": False, "needs_oauth": True,
                "auth_url": "/auth/yahoo?reauth=1",
                "error": ("That Yahoo account can't access league " + league_id +
                          ". Reconnect with the Yahoo account that's in this league."),
            }), 401
        return jsonify({
            "ok": False,
            "error": "Couldn't load that Yahoo league. Double-check the league ID.",
        }), 400
