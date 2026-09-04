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
    account_id = upsert_google_account(sub, email, info.get("given_name"))
    if not account_id:
        return redirect("/?google_error=account_error")

    session["account_id"] = account_id
    session["account_email"] = email
    session["account_first_name"] = info.get("given_name") or ""
    session.permanent = True

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
                return redirect(
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
                # Sleeper id already belongs to another Google account — do not
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
                    # team name as username — both feed resolve_viewer so Scout
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
