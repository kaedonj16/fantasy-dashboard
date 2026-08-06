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
    if not _yahoo_configured():
        return (
            "<p>Yahoo OAuth is not configured on this server. "
            "Set YAHOO_CLIENT_ID, YAHOO_CLIENT_SECRET, and YAHOO_REDIRECT_URI.</p>",
            503,
        )

    from dashboard_services.providers.yahoo_api import get_authorization_url

    # The registered redirect_uri lives on the apex host, so Yahoo always returns
    # the user there. Start the flow on that same host too — otherwise the session
    # cookie set here (e.g. on www.) isn't sent to the apex callback and state
    # validation fails. Bounce to the canonical host first, preserving the query.
    from urllib.parse import urlparse
    canonical_host = urlparse(os.environ.get("YAHOO_REDIRECT_URI") or "").netloc
    if canonical_host and request.host and request.host != canonical_host:
        return redirect(f"https://{canonical_host}{request.full_path}")

    league_id = (request.args.get("league_id") or "").strip()
    next_url  = (request.args.get("next") or "/").strip()
    team_name = (request.args.get("team_name") or "").strip()

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

    auth_url = get_authorization_url(state=state)
    logger.info("[yahoo-auth] redirecting to: %s", auth_url)
    return redirect(auth_url)


@yahoo_auth_bp.route("/auth/yahoo/callback")
def yahoo_auth_callback():
    """Handle Yahoo OAuth callback, exchange code for tokens."""
    from dashboard_services.providers.yahoo_api import (
        exchange_code_for_tokens, save_tokens, save_league_owner,
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

    if not guid or not access_token:
        logger.error("[yahoo_auth] Missing guid or access_token in token response")
        return redirect("/?yahoo_error=invalid_token_response")

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
        nfl_state  = get_nfl_state() or {}
        season     = int(nfl_state.get("season") or datetime.now().year)
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
    league_id    = (request.args.get("league_id") or "").strip()
    access_token = session.get("yahoo_access_token") or ""

    if not league_id:
        return jsonify({"ok": False, "error": "League ID required"}), 400

    if not access_token:
        # Return a special flag so the frontend knows it needs to start OAuth
        return jsonify({"ok": False, "needs_oauth": True}), 401

    try:
        from dashboard_services.providers.yahoo_api import get_league
        from datetime import datetime
        from dashboard_services.api import get_nfl_state
        nfl_state = get_nfl_state() or {}
        season    = int(nfl_state.get("season") or datetime.now().year)
        league    = get_league(season, league_id, access_token)
        return jsonify({"ok": True, "league": {"name": league.get("name")}})
    except Exception as exc:
        logger.warning("[yahoo] validate league %s failed: %s", league_id, exc)
        return jsonify({"ok": False, "error": str(exc)}), 400
