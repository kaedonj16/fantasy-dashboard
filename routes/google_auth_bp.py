"""Google OAuth 2.0 sign-in for standalone accounts.

Flow (mirrors the Yahoo blueprint):
  1. User visits /auth/google?next=<url>
     → stores an opaque state in the session, redirects to Google's consent page
  2. Google redirects to /auth/google/callback?code=...&state=...
     → exchanges the code, reads the profile (sub + email), upserts the account,
       sets session['account_id'], and — if a Sleeper session is already present —
       bridges that identity onto the account and backfills its leagues.

Configuration (env vars, all required to enable the flow):
  GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET, GOOGLE_REDIRECT_URI
The redirect URI must exactly match one registered on the OAuth client, e.g.
  https://brfantasyfootball.com/auth/google/callback
"""
from __future__ import annotations

import logging
import os
import secrets
import base64
import hashlib
import threading
from urllib.parse import urlencode

from flask import Blueprint, redirect, request, session

google_auth_bp = Blueprint("google_auth", __name__)
logger = logging.getLogger(__name__)

_AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
_TOKEN_URL = "https://oauth2.googleapis.com/token"


def _google_configured() -> bool:
    return bool(
        os.environ.get("GOOGLE_CLIENT_ID")
        and os.environ.get("GOOGLE_CLIENT_SECRET")
        and os.environ.get("GOOGLE_REDIRECT_URI")
    )


@google_auth_bp.route("/auth/google")
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
    session["google_oauth_next"] = (request.args.get("next") or "/").strip()
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


@google_auth_bp.route("/auth/google/callback")
def google_auth_callback():
    """Handle Google's redirect: verify state, exchange code, sign the user in."""
    import requests

    if request.args.get("error"):
        return redirect("/?google_error=" + request.args.get("error"))

    code = request.args.get("code") or ""
    state = request.args.get("state") or ""
    stored_state = session.pop("google_oauth_state", None)
    stored_nonce = session.pop("google_oauth_nonce", None)
    verifier = session.pop("google_pkce_verifier", None)
    next_url = session.pop("google_oauth_next", "/") or "/"
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
    account_id = upsert_google_account(sub, email, info.get("given_name"))
    if not account_id:
        return redirect("/?google_error=account_error")

    session["account_id"] = account_id
    session["account_email"] = email
    session["account_first_name"] = info.get("given_name") or ""
    session.permanent = True

    # Private ESPN onboarding is validated and encrypted before Google sign-in.
    # The browser session carries only an opaque one-time token; consume it now
    # and attach the league to the canonical Google account.
    pending_provider_token = session.pop("pending_provider_connection_token", None)
    if pending_provider_token:
        try:
            from dashboard_services.accounts import (
                consume_private_espn_connection, add_espn_league_connection,
            )
            pending_provider = consume_private_espn_connection(pending_provider_token)
            if pending_provider:
                add_espn_league_connection(
                    account_id, pending_provider["league_id"], pending_provider["season"],
                    pending_provider.get("name") or "ESPN League", "private",
                    swid=pending_provider["swid"], espn_s2=pending_provider["espn_s2"],
                    team_id=pending_provider.get("team_id"),
                )
                session.pop("onboarding_progress", None)
                return redirect(
                    f"/espn/{pending_provider['season']}/{pending_provider['league_id']}/dashboard"
                )
        except Exception:
            logger.warning("[google_auth] pending ESPN attach failed", exc_info=True)

    # Bridge an already-signed-in Sleeper session onto this account: attach the
    # identity and backfill its leagues so they show up immediately. Best-effort
    # — a failure here must not block sign-in.
    viewer_user_id = session.get("viewer_user_id")
    if viewer_user_id:
        try:
            link_platform_identity(
                account_id, "sleeper", str(viewer_user_id), session.get("viewer_username"),
            )
            threading.Thread(
                target=_backfill_sleeper_leagues,
                args=(account_id, str(viewer_user_id)),
                daemon=True,
            ).start()
        except Exception:
            logger.warning("[google_auth] sleeper bridge failed", exc_info=True)

    # Select-league-then-login: if the user picked a league before signing in,
    # attach it now and drop them straight into it.
    pending = session.pop("pending_link", None)
    if isinstance(pending, dict) and pending.get("platform") and pending.get("league_id"):
        try:
            add_user_league(
                account_id, pending["platform"], pending["league_id"],
                season=pending.get("season"), team_id=pending.get("team_id"),
                name=pending.get("name"),
            )
            # Sleeper: resolve the typed username to a team and set the viewer
            # identity, so the dashboard is personalized (and their other Sleeper
            # leagues get bridged too) — the home flow never set a viewer session.
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
                        if vuid:
                            link_platform_identity(account_id, "sleeper", str(vuid), uname)
                            _backfill_sleeper_leagues(account_id, str(vuid))
                except Exception:
                    logger.warning("[google_auth] sleeper viewer resolve failed", exc_info=True)
            return redirect(
                f"/{pending['platform']}/{pending.get('season') or ''}/{pending['league_id']}/dashboard"
            )
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
    return redirect(destination or next_url)


def _backfill_sleeper_leagues(account_id: int, viewer_user_id: str) -> None:
    """Copy the Sleeper user's leagues into user_leagues for this account."""
    from datetime import datetime
    from dashboard_services.accounts import add_user_league
    from dashboard_services.api import get_sleeper_user_leagues, get_nfl_state, get_rosters
    nfl_state = get_nfl_state() or {}
    season = int(nfl_state.get("season") or datetime.now().year)
    leagues = get_sleeper_user_leagues(viewer_user_id, season) or []
    if not leagues:
        leagues = get_sleeper_user_leagues(viewer_user_id, season - 1) or []
        if leagues:
            season = season - 1
    for lg in leagues:
        lid = str(lg.get("league_id") or "")
        if lid:
            team_id = None
            try:
                team_id = next((str(r.get("roster_id")) for r in (get_rosters(lid) or [])
                                if str(r.get("owner_id") or "") == str(viewer_user_id)), None)
            except Exception:
                logger.debug("[google_auth] roster backfill failed for %s", lid, exc_info=True)
            add_user_league(
                account_id, "sleeper", lid,
                season=int(lg.get("season") or season),
                team_id=team_id,
                name=lg.get("name"),
            )
