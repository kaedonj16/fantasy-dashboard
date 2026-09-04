"""Link other platforms: attach ESPN / Yahoo leagues to the signed-in account.

Flow the modal drives:
  * ESPN  — GET /api/link/espn/preview?league_id=&season=  → {name, teams[]}
            (no per-user discovery on ESPN, so the user picks their team)
  * Yahoo — GET /api/link/yahoo/preview?league_id=&season= → {name, teams[], my_team_id}
            (requires Yahoo OAuth; the user's team is auto-detected from the guid)
  * both  — POST /api/link/add {platform, league_id, season, team_id, name}
            → writes a user_leagues row for the account

Adding requires a signed-in account (session['account_id']); everything then
shows up via /api/my-leagues (switcher + My Leagues).
"""
from __future__ import annotations

import logging
import hashlib
import re
import uuid
from datetime import datetime
from typing import Optional

from flask import Blueprint, jsonify, request, session

link_bp = Blueprint("link", __name__)
logger = logging.getLogger(__name__)

_ESPN_PUBLIC_FIELDS = {"league_id", "season"}
_ESPN_PRIVATE_FIELDS = {"league_id", "season", "swid", "espn_s2"}

# Users routinely paste a whole cookie string (or the full "Cookie:" header)
# instead of the two isolated values. Pull each cookie out of whichever field
# carries it so a blob dropped into either box still connects.
_ESPN_S2_IN_BLOB = re.compile(r'espn[_-]?s2\s*=\s*"?([^;"\s]+)', re.IGNORECASE)
_SWID_IN_BLOB = re.compile(r'\bswid\s*=\s*"?(\{[0-9A-Fa-f-]+\}|[0-9A-Fa-f-]{8,})', re.IGNORECASE)


def _extract_espn_credentials(swid_raw: str, espn_s2_raw: str) -> tuple[str, str]:
    """Return (swid, espn_s2), tolerating a pasted cookie blob in either field.

    When a ``name=value`` cookie marker is present anywhere in the two fields,
    the matched value wins; otherwise the field is treated as an already-clean
    single value and passed through unchanged (the historical happy path).
    """
    swid = (swid_raw or "").strip()
    espn_s2 = (espn_s2_raw or "").strip()
    blob = swid + "\n" + espn_s2
    m_s2 = _ESPN_S2_IN_BLOB.search(blob)
    if m_s2:
        espn_s2 = m_s2.group(1)
    m_swid = _SWID_IN_BLOB.search(blob)
    if m_swid:
        swid = m_swid.group(1)
    return swid, espn_s2


def _default_season() -> int:
    try:
        from dashboard_services.api import get_nfl_state
        return int((get_nfl_state() or {}).get("season") or datetime.now().year)
    except Exception:
        return datetime.now().year


@link_bp.route("/api/link/onboarding", methods=["GET", "POST", "DELETE"])
def link_onboarding_progress():
    """Resume non-sensitive provider selection across Google authentication."""
    if request.method == "GET":
        return jsonify({"ok": True, "progress": session.get("onboarding_progress")})
    if request.method == "DELETE":
        session.pop("onboarding_progress", None)
        return jsonify({"ok": True})
    data = request.get_json(silent=True) or {}
    allowed = {"provider", "connection_method", "league_id", "season", "step"}
    if not isinstance(data, dict) or set(data) - allowed:
        return jsonify({"ok": False, "error": "Invalid onboarding progress."}), 400
    provider = str(data.get("provider") or "").lower()
    method = str(data.get("connection_method") or "").lower()
    league_id = str(data.get("league_id") or "").strip()
    from dashboard_services.providers.registry import provider_keys
    if provider not in provider_keys() or method not in ("public", "private"):
        return jsonify({"ok": False, "error": "Invalid onboarding selection."}), 400
    session["onboarding_progress"] = {
        "provider": provider, "connection_method": method, "league_id": league_id,
        "season": data.get("season"), "step": str(data.get("step") or "league"),
    }
    return jsonify({"ok": True})


def _espn_error(exc: Exception, method: str):
    """Map ESPN failures without reflecting upstream bodies or credentials."""
    name = type(exc).__name__
    msg = str(exc).lower()
    reference = getattr(exc, "debug_reference", None) or uuid.uuid4().hex[:12]
    cause = type(exc.__cause__).__name__ if exc.__cause__ else None
    message_fingerprint = hashlib.sha256(str(exc).encode("utf-8", "replace")).hexdigest()[:12]
    logger.warning(
        "[link/espn] ref=%s method=%s error_type=%s error_module=%s cause_type=%s message_fingerprint=%s",
        reference, method, name, type(exc).__module__, cause, message_fingerprint,
    )

    def with_reference(message: str):
        return f"{message} Reference: {reference}."

    if name == "ESPNInvalidLeague" or "404" in msg:
        return with_reference("No ESPN league was found for that ID and season."), 404
    if name == "ESPNAccessDenied" or "401" in msg or "403" in msg:
        if method == "public":
            return with_reference("This ESPN league could not be accessed publicly. If it is a private "
                                  "league, connect using the Private League option."), 403
        return with_reference("ESPN rejected these credentials or the session has expired."), 403
    if name == "ESPNMalformedResponse":
        # ESPN commonly answers with an HTML login/challenge page and HTTP 200
        # when private-league cookies are expired or otherwise unusable. Treat
        # that as an authentication failure rather than returning 502: besides
        # being more accurate, some reverse proxies replace 502 JSON bodies with
        # their own HTML page, which hides this actionable message from the UI.
        if method == "private":
            return with_reference("ESPN did not accept this session. Copy fresh SWID and espn_s2 "
                                  "cookie values from an active ESPN login and try again."), 403
        return with_reference("ESPN returned incomplete league data. Check the league ID and "
                              "season, then try again."), 422
    if name == "ProviderCredentialConfigurationError":
        return with_reference("Private league connections are temporarily unavailable because "
                              "the server encryption key is not configured."), 503
    if "429" in msg or "rate" in msg:
        return with_reference("ESPN is rate limiting requests. Please wait a moment and try again."), 429
    if "timeout" in msg or "500" in msg or "502" in msg or "503" in msg:
        return with_reference("ESPN is temporarily unavailable. Please try again later."), 503
    return with_reference("ESPN returned an unexpected response. Please verify the details and try again."), 422


def _connect_espn(method: str):
    account_id = session.get("account_id")
    if not account_id:
        return jsonify({"ok": False, "error": "Sign in with Google first to link leagues."}), 401
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({"ok": False, "error": "A JSON request body is required."}), 400
    allowed = _ESPN_PUBLIC_FIELDS if method == "public" else _ESPN_PRIVATE_FIELDS
    unexpected = set(data) - allowed
    if unexpected:
        return jsonify({"ok": False, "error": "Unexpected fields for this connection method."}), 400
    league_id = str(data.get("league_id") or "").strip()
    if not league_id.isdigit():
        return jsonify({"ok": False, "error": "League ID must contain numbers only."}), 400
    try:
        season = int(data.get("season") or _default_season())
    except (TypeError, ValueError):
        return jsonify({"ok": False, "error": "Season must be a valid year."}), 400
    swid = str(data.get("swid") or "").strip() if method == "private" else None
    espn_s2 = str(data.get("espn_s2") or "").strip() if method == "private" else None
    if method == "private":
        swid, espn_s2 = _extract_espn_credentials(swid, espn_s2)
    if method == "private" and (not swid or not espn_s2):
        return jsonify({"ok": False, "error": "SWID and ESPN_S2 are required."}), 400
    try:
        from dashboard_services.providers.espn_api import connect_league
        info = connect_league(season, league_id, swid=swid, espn_s2=espn_s2)
        from dashboard_services.accounts import add_espn_league_connection
        add_espn_league_connection(
            account_id, league_id, season,
            info.get("name") or f"ESPN League {league_id}", method,
            swid=swid, espn_s2=espn_s2,
        )
    except Exception as exc:
        # Never log the submitted payload or exception text: third-party client
        # exceptions can contain cookie-bearing request details.
        logger.warning("[link/espn/%s] connection failed (%s)", method, type(exc).__name__)
        error, status = _espn_error(exc, method)
        return jsonify({"ok": False, "error": error}), status
    return jsonify({
        "ok": True, "platform": "espn", "connection_method": method,
        "league_id": league_id, "season": season, "name": info.get("name"),
        "redirect_url": f"/espn/{season}/{league_id}/dashboard",
    })


@link_bp.post("/api/link/espn/public")
def link_espn_public():
    return _connect_espn("public")


@link_bp.post("/api/link/espn/private")
def link_espn_private():
    return _connect_espn("private")


@link_bp.post("/api/link/espn/private/pending")
def link_espn_private_pending():
    """Validate and stage private credentials before Google onboarding."""
    if session.get("account_id"):
        return jsonify({"ok": False, "error": "Account is already signed in."}), 409
    data = request.get_json(silent=True) or {}
    if not isinstance(data, dict) or set(data) - _ESPN_PRIVATE_FIELDS:
        return jsonify({"ok": False, "error": "Invalid private ESPN request."}), 400
    league_id = str(data.get("league_id") or "").strip()
    swid, espn_s2 = _extract_espn_credentials(data.get("swid"), data.get("espn_s2"))
    if not league_id.isdigit() or not swid or not espn_s2:
        return jsonify({"ok": False, "error": "League ID, SWID, and ESPN_S2 are required."}), 400
    try:
        season = int(data.get("season") or _default_season())
        from dashboard_services.providers.espn_api import connect_league
        info = connect_league(season, league_id, swid=swid, espn_s2=espn_s2)
        from dashboard_services.accounts import stage_private_espn_connection
        token = stage_private_espn_connection(
            league_id, season, info.get("name") or f"ESPN League {league_id}", swid, espn_s2,
        )
        session["pending_provider_connection_token"] = token
        session["onboarding_progress"] = {
            "provider": "espn", "connection_method": "private",
            "league_id": league_id, "season": season, "step": "google",
        }
    except Exception as exc:
        logger.warning("[link/espn/private/pending] failed (%s)", type(exc).__name__)
        error, status = _espn_error(exc, "private")
        return jsonify({"ok": False, "error": error}), status
    return jsonify({
        "ok": True,
        "league_id": league_id,
        "season": season,
        "auth_url": "/auth/google?intent=onboarding&next=/",
    })


@link_bp.post("/api/link/espn/private/guest")
def link_espn_private_guest():
    """Open a staged private league without creating an application account."""
    data = request.get_json(silent=True) or {}
    league_id = str(data.get("league_id") or "").strip()
    try:
        season = int(data.get("season"))
    except (TypeError, ValueError):
        return jsonify({"ok": False, "error": "Invalid staged league."}), 400
    token = session.get("pending_provider_connection_token")
    from dashboard_services.accounts import peek_private_espn_connection
    if not peek_private_espn_connection(token, league_id, season):
        return jsonify({"ok": False, "error": "Private ESPN session expired. Validate it again."}), 401
    return jsonify({
        "ok": True,
        "redirect_url": f"/espn/{season}/{league_id}/dashboard",
    })


@link_bp.post("/api/link/espn/private/saved")
def link_espn_private_saved():
    """Open an already-linked private league using account-stored credentials."""
    account_id = session.get("account_id")
    if not account_id:
        return jsonify({"ok": False, "error": "Sign in with Google first."}), 401
    data = request.get_json(silent=True)
    if not isinstance(data, dict) or set(data) - _ESPN_PUBLIC_FIELDS:
        return jsonify({"ok": False, "error": "Only league_id and season are allowed."}), 400
    league_id = str(data.get("league_id") or "").strip()
    if not league_id.isdigit():
        return jsonify({"ok": False, "error": "League ID must contain numbers only."}), 400
    try:
        season = int(data.get("season") or _default_season())
    except (TypeError, ValueError):
        return jsonify({"ok": False, "error": "Season must be a valid year."}), 400
    from dashboard_services.accounts import get_espn_league_credentials
    credentials = get_espn_league_credentials(account_id, league_id, season) or {}
    swid, espn_s2 = credentials.get("swid"), credentials.get("espn_s2")
    if not swid or not espn_s2:
        return jsonify({
            "ok": False,
            "needs_credentials": True,
            "error": "Enter SWID and ESPN_S2 to connect this private league.",
        }), 409
    try:
        from dashboard_services.providers.espn_api import connect_league
        info = connect_league(season, league_id, swid=swid, espn_s2=espn_s2)
    except Exception as exc:
        logger.warning("[link/espn/private/saved] connection failed (%s)", type(exc).__name__)
        error, status = _espn_error(exc, "private")
        if status in (401, 403):
            from dashboard_services.accounts import mark_espn_connection_status
            mark_espn_connection_status(
                account_id, league_id, season, "reauth_required", "espn_auth_rejected",
            )
        return jsonify({
            "ok": False,
            "needs_credentials": status in (401, 403),
            "error": error,
        }), status
    return jsonify({
        "ok": True,
        "platform": "espn",
        "connection_method": "private",
        "league_id": league_id,
        "season": season,
        "name": info.get("name"),
        "redirect_url": f"/espn/{season}/{league_id}/dashboard",
    })


@link_bp.post("/api/link/espn/reconnect")
def link_espn_reconnect():
    """Refresh credentials for an owned saved league without re-entering its ID in UI."""
    account_id = session.get("account_id")
    if not account_id:
        return jsonify({"ok": False, "error": "Sign in with Google first."}), 401
    data = request.get_json(silent=True) or {}
    allowed = {"league_id", "season", "swid", "espn_s2"}
    if not isinstance(data, dict) or set(data) - allowed:
        return jsonify({"ok": False, "error": "Invalid reconnect request."}), 400
    league_id = str(data.get("league_id") or "").strip()
    swid, espn_s2 = _extract_espn_credentials(data.get("swid"), data.get("espn_s2"))
    try:
        season = int(data.get("season"))
    except (TypeError, ValueError):
        return jsonify({"ok": False, "error": "Invalid saved league."}), 400
    if not league_id.isdigit() or not swid or not espn_s2:
        return jsonify({"ok": False, "error": "SWID and ESPN_S2 are required."}), 400
    from dashboard_services.accounts import owns_user_league
    if not owns_user_league(account_id, "espn", league_id, season):
        return jsonify({"ok": False, "error": "Saved league not found."}), 404
    try:
        from dashboard_services.providers.espn_api import connect_league
        connect_league(season, league_id, swid=swid, espn_s2=espn_s2)
        from dashboard_services.accounts import replace_espn_credentials
        if not replace_espn_credentials(account_id, league_id, season, swid, espn_s2):
            return jsonify({"ok": False, "error": "Saved league not found."}), 404
    except Exception as exc:
        logger.warning("[link/espn/reconnect] failed (%s)", type(exc).__name__)
        error, status = _espn_error(exc, "private")
        return jsonify({"ok": False, "error": error}), status
    return jsonify({"ok": True, "redirect_url": f"/espn/{season}/{league_id}/dashboard"})


# ── ESPN email + one-time-code sign-in (feature-flagged, off by default) ───────
# A friendlier alternative to pasting cookies: the member enters their email,
# gets a code, enters it, and we obtain SWID + espn_s2 and run the SAME connect
# pipeline. Everything downstream of the credentials is unchanged; cookie paste
# stays as the fallback for every failure.
def _otp_error(exc: Exception):
    reference = uuid.uuid4().hex[:12]
    name = type(exc).__name__
    logger.warning("[link/espn/otp] ref=%s error_type=%s", reference, name)
    mapping = {
        "EspnLoginUnavailable": ("ESPN email sign-in is unavailable right now. Use the cookie option below.", 503),
        "EspnLoginInvalidCode": ("That code isn't right. Check your email and try again.", 400),
        "EspnLoginExpired": ("This sign-in expired. Request a new code.", 401),
        "EspnLoginCaptchaRequired": ("ESPN asked for an extra security check. Use the cookie option below.", 409),
        "EspnLoginTooManyAttempts": ("Too many tries. Start the sign-in again.", 429),
        "EspnLoginRateLimited": ("Too many code requests. Wait a few minutes and try again.", 429),
    }
    message, status = mapping.get(name, ("Couldn't complete ESPN sign-in. Use the cookie option below.", 400))
    return f"{message} Reference: {reference}.", status


@link_bp.post("/api/link/espn/otp/start")
def link_espn_otp_start():
    from dashboard_services.providers.espn_login import otp_login_enabled, get_broker
    if not otp_login_enabled():
        return jsonify({"ok": False, "error": "ESPN email sign-in isn't available."}), 404
    data = request.get_json(silent=True) or {}
    if not isinstance(data, dict) or set(data) - {"email", "league_id", "season"}:
        return jsonify({"ok": False, "error": "Invalid request."}), 400
    email = str(data.get("email") or "").strip()
    league_id = str(data.get("league_id") or "").strip()
    if not league_id.isdigit():
        return jsonify({"ok": False, "error": "League ID must contain numbers only."}), 400
    try:
        season = int(data.get("season") or _default_season())
    except (TypeError, ValueError):
        return jsonify({"ok": False, "error": "Season must be a valid year."}), 400
    try:
        login_id = get_broker().start(email)
    except Exception as exc:
        logger.warning("[link/espn/otp/start] failed (%s)", type(exc).__name__)
        error, status = _otp_error(exc)
        return jsonify({"ok": False, "error": error}), status
    return jsonify({"ok": True, "login_id": login_id, "state": "awaiting_code"})


@link_bp.post("/api/link/espn/otp/verify")
def link_espn_otp_verify():
    from dashboard_services.providers.espn_login import otp_login_enabled, get_broker
    if not otp_login_enabled():
        return jsonify({"ok": False, "error": "ESPN email sign-in isn't available."}), 404
    data = request.get_json(silent=True) or {}
    if not isinstance(data, dict) or set(data) - {"login_id", "code", "league_id", "season"}:
        return jsonify({"ok": False, "error": "Invalid request."}), 400
    login_id = str(data.get("login_id") or "").strip()
    code = str(data.get("code") or "").strip()
    league_id = str(data.get("league_id") or "").strip()
    if not login_id or not code or not league_id.isdigit():
        return jsonify({"ok": False, "error": "Enter the code from your email."}), 400
    try:
        season = int(data.get("season") or _default_season())
    except (TypeError, ValueError):
        return jsonify({"ok": False, "error": "Season must be a valid year."}), 400
    try:
        creds = get_broker().verify(login_id, code)
    except Exception as exc:
        logger.warning("[link/espn/otp/verify] failed (%s)", type(exc).__name__)
        error, status = _otp_error(exc)
        return jsonify({"ok": False, "error": error}), status
    swid, espn_s2 = creds.get("swid"), creds.get("espn_s2")
    # Same validate-then-persist path as cookie paste, tagged connection_method="otp".
    try:
        from dashboard_services.providers.espn_api import connect_league
        info = connect_league(season, league_id, swid=swid, espn_s2=espn_s2)
    except Exception as exc:
        logger.warning("[link/espn/otp/verify] connect failed (%s)", type(exc).__name__)
        error, status = _espn_error(exc, "private")
        return jsonify({"ok": False, "error": error}), status
    league_name = info.get("name") or f"ESPN League {league_id}"
    account_id = session.get("account_id")
    if account_id:
        # Stored as "private" (the credentials are private cookies; the schema's
        # CHECK allows only public/private). "otp" is surfaced in the response so
        # the client knows how the member connected, without a schema change.
        from dashboard_services.accounts import add_espn_league_connection
        add_espn_league_connection(account_id, league_id, season, league_name, "private", swid=swid, espn_s2=espn_s2)
        return jsonify({
            "ok": True, "platform": "espn", "connection_method": "otp",
            "league_id": league_id, "season": season, "name": info.get("name"),
            # Let the member pick which team is theirs before landing on the
            # dashboard; the client sets the viewer, then follows redirect_url.
            "teams": info.get("teams") or [],
            "redirect_url": f"/espn/{season}/{league_id}/dashboard",
        })
    # No account yet: stage for post-Google onboarding, mirroring /private/pending.
    from dashboard_services.accounts import stage_private_espn_connection
    token = stage_private_espn_connection(league_id, season, league_name, swid, espn_s2)
    session["pending_provider_connection_token"] = token
    session["onboarding_progress"] = {
        "provider": "espn", "connection_method": "otp",
        "league_id": league_id, "season": season, "step": "google",
    }
    return jsonify({"ok": True, "league_id": league_id, "season": season,
                    "auth_url": "/auth/google?intent=onboarding&next=/"})


@link_bp.post("/api/link/espn/otp/resend")
def link_espn_otp_resend():
    from dashboard_services.providers.espn_login import otp_login_enabled, get_broker
    if not otp_login_enabled():
        return jsonify({"ok": False, "error": "ESPN email sign-in isn't available."}), 404
    data = request.get_json(silent=True) or {}
    login_id = str(data.get("login_id") or "").strip()
    if not login_id:
        return jsonify({"ok": False, "error": "Invalid request."}), 400
    try:
        get_broker().resend(login_id)
    except Exception as exc:
        logger.warning("[link/espn/otp/resend] failed (%s)", type(exc).__name__)
        error, status = _otp_error(exc)
        return jsonify({"ok": False, "error": error}), status
    return jsonify({"ok": True, "state": "awaiting_code"})


@link_bp.route("/api/link/espn/preview")
def link_espn_preview():
    league_id = (request.args.get("league_id") or "").strip()
    if not league_id.isdigit():
        return jsonify({"ok": False, "error": "ESPN league ID must be a number."}), 400
    season = int(request.args.get("season") or _default_season())
    try:
        from dashboard_services.providers.espn_api import get_league, get_teams
        info = get_league(season, league_id)
        teams = get_teams(season, league_id)
    except Exception as exc:
        name = type(exc).__name__
        if name == "ESPNAccessDenied":
            return jsonify({"ok": False, "error": "This looks like a private ESPN league the server can't read."}), 403
        if name == "ESPNInvalidLeague":
            return jsonify({"ok": False, "error": "No ESPN league found for that ID and season."}), 404
        logger.warning("[link/espn] preview failed: %s", exc)
        return jsonify({"ok": False, "error": "Could not load that ESPN league."}), 400
    return jsonify({
        "ok": True,
        "platform": "espn",
        "league_id": str(league_id),
        "season": season,
        "name": info.get("name") or f"ESPN League {league_id}",
        "teams": teams,
    })


def _yahoo_needs_oauth_response(league_id: str, *, reauth: bool = False, error: str = "") -> tuple:
    """401 payload for Yahoo preview/validate when OAuth is required."""
    from dashboard_services.providers.yahoo_api import yahoo_oauth_start_url

    session["yahoo_link_league_id"] = league_id
    if session.get("account_id"):
        auth_url = yahoo_oauth_start_url(league_id=league_id, next_url="/portfolio", reauth=reauth)
    else:
        auth_url = yahoo_oauth_start_url(next_url="/portfolio", reauth=reauth)
    payload: dict = {"ok": False, "needs_oauth": True, "auth_url": auth_url}
    if error:
        payload["error"] = error
    return jsonify(payload), 401


def _yahoo_preview_payload(league_id: str, season: int, access_token: str, guid: str):
    """Load Yahoo league + teams for the link modal, or return an error response."""
    from dashboard_services.providers.yahoo_api import (
        get_league, get_login_guid, get_users, resolve_league_key,
    )
    resolved = resolve_league_key(access_token, league_id)
    if resolved.get("status") == "absent":
        return _yahoo_needs_oauth_response(
            league_id,
            reauth=True,
            error=("That Yahoo account isn't in any league with ID " + league_id +
                   ". Check the ID, or reconnect with the account that's in it."),
        )
    if resolved.get("season"):
        season = int(resolved["season"])
    league = get_league(season, league_id, access_token)
    users = get_users(season, league_id, access_token)
    my_guid = guid or ""
    if not my_guid or my_guid.startswith("ytok_"):
        resolved_guid = get_login_guid(access_token, league_id)
        if resolved_guid:
            my_guid = resolved_guid
            session["yahoo_guid"] = resolved_guid
    teams, my_team_id = [], None
    for u in users:
        tid = str(u.get("roster_id") or "")
        nm = (u.get("metadata") or {}).get("team_name") or u.get("display_name") or f"Team {tid}"
        teams.append({"team_id": tid, "name": nm})
        if my_guid and str(u.get("user_id")) == str(my_guid):
            my_team_id = tid
    return jsonify({
        "ok": True,
        "platform": "yahoo",
        "league_id": str(league_id),
        "season": season,
        "name": league.get("name") or f"Yahoo League {league_id}",
        "teams": teams,
        "my_team_id": my_team_id,
    })


@link_bp.route("/api/link/yahoo/preview")
def link_yahoo_preview():
    from dashboard_services.providers.yahoo_api import yahoo_enabled
    if not yahoo_enabled():
        return jsonify({"ok": False, "error": "Yahoo connections are temporarily unavailable."}), 503
    league_id = (request.args.get("league_id") or "").strip()
    if not league_id:
        return jsonify({"ok": False, "error": "Yahoo league ID required."}), 400
    from dashboard_services.providers.yahoo_api import (
        get_valid_access_token, resolve_session_yahoo_token, yahoo_auth_error_kind,
    )
    guid, access_token = resolve_session_yahoo_token(session)
    if not access_token:
        return _yahoo_needs_oauth_response(league_id)
    season = int(request.args.get("season") or _default_season())
    try:
        return _yahoo_preview_payload(league_id, season, access_token, guid)
    except Exception as exc:
        logger.warning("[link/yahoo] preview failed: %s", exc)
        kind = yahoo_auth_error_kind(exc)
        # Yahoo said token_expired even though our stored expires_at looked fine —
        # force a refresh once before bouncing the user through OAuth again.
        if kind == "expired" and guid:
            refreshed = get_valid_access_token(guid, force_refresh=True) or ""
            if refreshed:
                try:
                    return _yahoo_preview_payload(league_id, season, refreshed, guid)
                except Exception as retry_exc:
                    logger.warning("[link/yahoo] preview retry after refresh failed: %s", retry_exc)
                    kind = yahoo_auth_error_kind(retry_exc) or "expired"
        if kind in ("expired", "forbidden"):
            session.pop("yahoo_access_token", None)
            if kind == "forbidden":
                session.pop("yahoo_guid", None)
            return _yahoo_needs_oauth_response(
                league_id,
                reauth=True,
                error=(
                    "Your Yahoo login expired. Reconnect Yahoo and try again."
                    if kind == "expired" else
                    ("That Yahoo account can't access league " + league_id +
                     ". Reconnect with the account that's in this league.")
                ),
            )
        return jsonify({"ok": False, "error": "Could not load that Yahoo league (check the ID)."}), 400


@link_bp.route("/api/link/pending", methods=["POST"])
def link_pending():
    """Stash a league the user selected *before* signing in, then send them to
    Google. The callback attaches it to the new account and drops them into it —
    the select-league-then-login onboarding order."""
    data = request.get_json(force=True) or {}
    platform = (data.get("platform") or "").strip().lower()
    league_id = str(data.get("league_id") or "").strip()
    from dashboard_services.providers.registry import provider_keys
    if platform not in provider_keys() or not league_id:
        return jsonify({"ok": False, "error": "Missing platform or league_id."}), 400
    try:
        season = int(data.get("season")) if data.get("season") else _default_season()
    except (TypeError, ValueError):
        season = _default_season()
    session["pending_link"] = {
        "platform": platform,
        "league_id": league_id,
        "season": season,
        "team_id": (str(data.get("team_id")).strip() or None) if data.get("team_id") else None,
        "name": (str(data.get("name")).strip() or None) if data.get("name") else None,
        # Sleeper: the username the user typed, so the callback can resolve their
        # team and set the viewer identity (personalizes the dashboard).
        "username": (str(data.get("username")).strip() or None) if data.get("username") else None,
    }
    checkout_plan = str(data.get("checkout_plan") or "").strip()
    if checkout_plan in {"single_league", "league", "combo", "user"}:
        session["pending_link"]["checkout_plan"] = checkout_plan
        from urllib.parse import quote
        next_url = (
            f"/{platform}/{season}/{quote(league_id, safe='')}"
            f"/pricing?plan={quote(checkout_plan, safe='')}&checkout=1"
        )
        return jsonify({"ok": True, "auth_url": f"/auth/google?next={quote(next_url, safe='')}"})
    return jsonify({"ok": True, "auth_url": "/auth/google?next=/"})


@link_bp.route("/api/link/remove", methods=["POST"])
def link_remove():
    """Unlink a league from the signed-in account."""
    account_id = session.get("account_id")
    if not account_id:
        return jsonify({"ok": False, "error": "Not signed in."}), 401
    data = request.get_json(force=True) or {}
    platform = (data.get("platform") or "").strip().lower()
    league_id = str(data.get("league_id") or "").strip()
    if not platform or not league_id:
        return jsonify({"ok": False, "error": "Missing platform or league_id."}), 400
    season = data.get("season")
    try:
        season = int(season) if season else None
    except (TypeError, ValueError):
        season = None
    try:
        from dashboard_services.accounts import remove_user_league
        remove_user_league(account_id, platform, league_id, season=season)
    except Exception as exc:
        logger.warning("[link/remove] failed: %s", exc)
        return jsonify({"ok": False, "error": "Could not remove that league."}), 500
    return jsonify({"ok": True})


@link_bp.route("/api/link/add", methods=["POST"])
def link_add():
    account_id = session.get("account_id")
    if not account_id:
        return jsonify({"ok": False, "error": "Sign in with Google first to link leagues."}), 401
    data = request.get_json(force=True) or {}
    platform = (data.get("platform") or "").strip().lower()
    league_id = (str(data.get("league_id") or "")).strip()
    from dashboard_services.providers.registry import provider_keys
    if platform not in provider_keys() or not league_id:
        return jsonify({"ok": False, "error": "Missing platform or league_id."}), 400
    season = data.get("season")
    try:
        season = int(season) if season else None
    except (TypeError, ValueError):
        season = None
    team_id = (str(data.get("team_id")).strip() or None) if data.get("team_id") else None
    name = (str(data.get("name")).strip() or None) if data.get("name") else None
    try:
        from dashboard_services.accounts import add_user_league, link_platform_identity
        if platform == "sleeper":
            from dashboard_services.api import (
                get_rosters, get_sleeper_user_by_username, get_sleeper_user_leagues,
            )
            username = (str(data.get("username") or "").strip() or None)
            sleeper_user = get_sleeper_user_by_username(username) if username else None
            platform_user_id = str((sleeper_user or {}).get("user_id") or "").strip()
            if not platform_user_id:
                return jsonify({"ok": False, "error": "Could not verify that Sleeper user."}), 400
            memberships = get_sleeper_user_leagues(platform_user_id, season) or []
            if str(league_id) not in {str(lg.get("league_id")) for lg in memberships}:
                return jsonify({"ok": False, "error": "That Sleeper user is not a member of this league."}), 400
            team_id = next((str(roster.get("roster_id")) for roster in (get_rosters(league_id) or [])
                            if str(roster.get("owner_id") or "") == platform_user_id), None)
            link_platform_identity(account_id, "sleeper", platform_user_id,
                                   (sleeper_user or {}).get("username") or username)
        flea_auth_token = None
        if platform == "fleaflicker":
            try:
                from dashboard_services.accounts import get_provider_league_credentials
                from dashboard_services.providers.registry import get_provider
                from dashboard_services.providers.fleaflicker_api import resolve_fleaflicker_team_id
                creds = get_provider_league_credentials(
                    account_id, platform, league_id, season,
                ) or {}
                flea_auth_token = creds.get("token")
                flea_uid = str(creds.get("flea_user_id") or "").strip()
                if flea_uid:
                    link_platform_identity(account_id, "fleaflicker", flea_uid)
                if not team_id and (flea_uid or flea_auth_token):
                    provider = get_provider("fleaflicker")
                    lookup_season = season if season is not None else _default_season()
                    users = provider.get_users(
                        league_id, lookup_season, token=flea_auth_token,
                    )
                    team_id = resolve_fleaflicker_team_id(
                        users, team_id=team_id, flea_user_id=flea_uid or None,
                    ) or team_id
            except Exception:
                logger.warning("[link/add/fleaflicker] team resolution failed", exc_info=True)
        if platform == "yahoo":
            from dashboard_services.providers.yahoo_api import (
                resolve_league_key, resolve_session_yahoo_token, save_league_owner,
            )
            yahoo_guid, access_token = resolve_session_yahoo_token(session)
            lookup_season = season if season is not None else _default_season()
            if yahoo_guid:
                link_platform_identity(account_id, "yahoo", str(yahoo_guid))
            if access_token and league_id and yahoo_guid:
                try:
                    resolved = resolve_league_key(access_token, league_id)
                    if resolved.get("season"):
                        lookup_season = int(resolved["season"])
                        if season is None:
                            season = lookup_season
                    save_league_owner(league_id, lookup_season, str(yahoo_guid))
                except Exception:
                    logger.warning("[link/add/yahoo] save_league_owner failed", exc_info=True)
        add_user_league(account_id, platform, league_id, season=season, team_id=team_id, name=name)
        if platform == "fleaflicker":
            lookup_season = season if season is not None else _default_season()
            _persist_fleaflicker_viewer(
                league_id, lookup_season, team_id, token=flea_auth_token,
            )
    except Exception as exc:
        logger.warning("[link/add] failed: %s", exc)
        return jsonify({"ok": False, "error": "Could not save that league."}), 500
    return jsonify({"ok": True})


@link_bp.get("/api/link/mfl/preview")
def link_mfl_preview():
    """Validate a public, read-only MFL league and return its franchises."""
    league_id = str(request.args.get("league_id") or "").strip()
    try:
        season = int(request.args.get("season") or _default_season())
    except (TypeError, ValueError):
        return jsonify({"ok": False, "error": "Season must be a valid year."}), 400
    if not league_id.isdigit() or not (2000 <= season <= 2100):
        return jsonify({"ok": False, "error": "Enter a valid numeric MFL League ID and season."}), 400
    try:
        from dashboard_services.providers.registry import get_provider
        provider = get_provider("mfl")
        league = provider.get_league(league_id, season)
        users = provider.get_users(league_id, season)
    except Exception as exc:
        from dashboard_services.providers.base import ProviderAuthenticationError, LeagueNotFoundError
        logger.warning("[link/mfl] preview failed error=%s", type(exc).__name__)
        if isinstance(exc, ProviderAuthenticationError):
            return jsonify({"ok": False, "error": "This MFL league is private or requires authentication."}), 403
        if isinstance(exc, LeagueNotFoundError):
            return jsonify({"ok": False, "error": "No MFL league was found for that ID and season."}), 404
        return jsonify({"ok": False, "error": "MyFantasyLeague is temporarily unavailable."}), 503
    teams = [{"team_id": str(u.get("roster_id") or u.get("user_id")),
              "name": (u.get("metadata") or {}).get("team_name") or u.get("display_name")}
             for u in users]
    return jsonify({"ok": True, "platform": "mfl", "league_id": league_id,
                    "season": season, "name": league.get("name"), "teams": teams})


def _mfl_private_credentials(data: dict) -> dict:
    """Derive storeable MFL auth from cookie, APIKEY, and/or username+password login."""
    from dashboard_services.providers.mfl_api import login as mfl_login, normalize_mfl_cookie
    cookie = normalize_mfl_cookie(data.get("cookie") or data.get("mfl_user_id"))
    apikey = str(data.get("apikey") or "").strip()
    username = str(data.get("username") or "").strip()
    password = str(data.get("password") or "")
    season = data.get("season")
    if username and password:
        cookie = mfl_login(username, password, int(season or _default_season()))
    if not cookie and not apikey:
        raise ValueError("MFL private leagues require a login cookie and/or league APIKEY.")
    out = {}
    if cookie:
        out["cookie"] = cookie
    if apikey:
        out["apikey"] = apikey
    return out


def _provider_connect_error(exc: Exception, provider: str, method: str):
    from dashboard_services.providers.base import (
        ProviderAuthenticationError, LeagueNotFoundError, ProviderUnavailableError,
    )
    logger.warning("[link/%s/%s] failed error=%s", provider, method, type(exc).__name__)
    if isinstance(exc, ProviderAuthenticationError):
        return ({"ok": False, "error": f"This {provider.upper() if provider == 'mfl' else provider.title()} "
                 "league is private or the credentials were rejected."}, 403)
    if isinstance(exc, LeagueNotFoundError):
        label = "MyFantasyLeague" if provider == "mfl" else "Fleaflicker"
        return ({"ok": False, "error": f"No {label} league was found for that ID and season."}, 404)
    if isinstance(exc, ProviderUnavailableError) or isinstance(exc, ValueError):
        message = str(exc) if isinstance(exc, ValueError) and str(exc) else (
            "MyFantasyLeague is temporarily unavailable." if provider == "mfl"
            else "Fleaflicker is temporarily unavailable."
        )
        status = 400 if isinstance(exc, ValueError) else 503
        return ({"ok": False, "error": message}, status)
    label = "MyFantasyLeague" if provider == "mfl" else "Fleaflicker"
    return ({"ok": False, "error": f"{label} is temporarily unavailable."}, 503)


def _connect_mfl(method: str):
    account_id = session.get("account_id")
    if not account_id:
        return jsonify({"ok": False, "error": "Sign in to connect a league."}), 401
    data = request.get_json(silent=True) or {}
    allowed = {"league_id", "season"} if method == "public" else {
        "league_id", "season", "cookie", "mfl_user_id", "apikey", "username", "password",
    }
    if not isinstance(data, dict) or set(data) - allowed:
        return jsonify({"ok": False, "error": "Unexpected fields for this connection method."}), 400
    league_id = str(data.get("league_id") or "").strip()
    try:
        season = int(data.get("season") or _default_season())
    except (TypeError, ValueError):
        return jsonify({"ok": False, "error": "Season must be a valid year."}), 400
    if not league_id.isdigit() or not (2000 <= season <= 2100):
        return jsonify({"ok": False, "error": "Enter a valid numeric MFL League ID and season."}), 400
    credentials = None
    try:
        from dashboard_services.providers.registry import get_provider
        provider = get_provider("mfl")
        if method == "private":
            credentials = _mfl_private_credentials({**data, "season": season})
            info = provider.connect_league(
                league_id, season, cookie=credentials.get("cookie"), apikey=credentials.get("apikey"),
            )
        else:
            info = provider.connect_league(league_id, season)
        from dashboard_services.accounts import add_provider_league_connection
        add_provider_league_connection(
            account_id, "mfl", league_id, season,
            info.get("name") or f"MFL League {league_id}", method,
            credentials=credentials,
        )
    except Exception as exc:
        error, status = _provider_connect_error(exc, "mfl", method)
        return jsonify(error), status
    return jsonify({
        "ok": True, "platform": "mfl", "connection_method": method,
        "league_id": league_id, "season": season, "name": info.get("name"),
        "redirect_url": f"/mfl/{season}/{league_id}/dashboard",
    })


@link_bp.post("/api/link/mfl/public")
def link_mfl_public():
    return _connect_mfl("public")


@link_bp.post("/api/link/mfl/private")
def link_mfl_private():
    return _connect_mfl("private")


@link_bp.post("/api/link/mfl/private/pending")
def link_mfl_private_pending():
    data = request.get_json(silent=True) or {}
    league_id = str(data.get("league_id") or "").strip()
    try:
        season = int(data.get("season") or _default_season())
    except (TypeError, ValueError):
        return jsonify({"ok": False, "error": "Invalid private MFL request."}), 400
    if not league_id.isdigit():
        return jsonify({"ok": False, "error": "Invalid private MFL request."}), 400
    try:
        from dashboard_services.providers.registry import get_provider
        credentials = _mfl_private_credentials({**data, "season": season})
        info = get_provider("mfl").connect_league(
            league_id, season, cookie=credentials.get("cookie"), apikey=credentials.get("apikey"),
        )
        from dashboard_services.accounts import stage_private_provider_connection
        token = stage_private_provider_connection(
            "mfl", league_id, season, info.get("name") or f"MFL League {league_id}", credentials,
        )
        session["pending_provider_connection_token"] = token
        return jsonify({
            "ok": True, "provider": "mfl", "connection_method": "private",
            "league_id": league_id, "season": season, "name": info.get("name"),
            "auth_url": "/auth/google?intent=onboarding&next=/",
        })
    except Exception as exc:
        error, status = _provider_connect_error(exc, "mfl", "private")
        return jsonify(error), status


@link_bp.post("/api/link/mfl/private/guest")
def link_mfl_private_guest():
    data = request.get_json(silent=True) or {}
    league_id = str(data.get("league_id") or "").strip()
    try:
        season = int(data.get("season") or _default_season())
    except (TypeError, ValueError):
        return jsonify({"ok": False, "error": "Invalid request."}), 400
    token = session.get("pending_provider_connection_token")
    from dashboard_services.accounts import peek_private_provider_connection
    if not peek_private_provider_connection(token, "mfl", league_id, season):
        return jsonify({"ok": False, "error": "Private MFL session expired. Reconnect."}), 403
    return jsonify({
        "ok": True, "redirect_url": f"/mfl/{season}/{league_id}/dashboard",
    })


@link_bp.post("/api/link/mfl/private/saved")
def link_mfl_private_saved():
    account_id = session.get("account_id")
    if not account_id:
        return jsonify({"ok": False, "error": "Sign in to open a saved private league."}), 401
    data = request.get_json(silent=True) or {}
    league_id = str(data.get("league_id") or "").strip()
    try:
        season = int(data.get("season") or _default_season())
    except (TypeError, ValueError):
        return jsonify({"ok": False, "error": "Invalid request."}), 400
    from dashboard_services.accounts import get_provider_league_credentials, mark_provider_connection_status
    credentials = get_provider_league_credentials(account_id, "mfl", league_id, season) or {}
    if not (credentials.get("cookie") or credentials.get("apikey")):
        return jsonify({
            "ok": False,
            "error": "Enter an MFL login cookie and/or league APIKEY to connect this private league.",
        }), 400
    try:
        from dashboard_services.providers.registry import get_provider
        info = get_provider("mfl").connect_league(
            league_id, season,
            cookie=credentials.get("cookie"), apikey=credentials.get("apikey"),
        )
    except Exception as exc:
        error, status = _provider_connect_error(exc, "mfl", "private")
        try:
            mark_provider_connection_status(
                account_id, "mfl", league_id, season, "reauth_required", "mfl_auth_rejected",
            )
        except Exception:
            pass
        return jsonify(error), status
    return jsonify({
        "ok": True, "platform": "mfl", "connection_method": "private",
        "league_id": league_id, "season": season, "name": info.get("name"),
        "redirect_url": f"/mfl/{season}/{league_id}/dashboard",
    })


@link_bp.post("/api/link/mfl/reconnect")
def link_mfl_reconnect():
    account_id = session.get("account_id")
    if not account_id:
        return jsonify({"ok": False, "error": "Sign in to reconnect."}), 401
    data = request.get_json(silent=True) or {}
    allowed = {"league_id", "season", "cookie", "mfl_user_id", "apikey", "username", "password"}
    if not isinstance(data, dict) or set(data) - allowed:
        return jsonify({"ok": False, "error": "Unexpected fields for this connection method."}), 400
    league_id = str(data.get("league_id") or "").strip()
    try:
        season = int(data.get("season") or _default_season())
    except (TypeError, ValueError):
        return jsonify({"ok": False, "error": "Invalid request."}), 400
    from dashboard_services.accounts import owns_user_league, replace_provider_credentials
    if not owns_user_league(account_id, "mfl", league_id, season):
        return jsonify({"ok": False, "error": "League not found on this account."}), 404
    try:
        credentials = _mfl_private_credentials({**data, "season": season})
        from dashboard_services.providers.registry import get_provider
        get_provider("mfl").connect_league(
            league_id, season, cookie=credentials.get("cookie"), apikey=credentials.get("apikey"),
        )
        if not replace_provider_credentials(account_id, "mfl", league_id, season, credentials):
            return jsonify({"ok": False, "error": "Could not update credentials."}), 400
    except Exception as exc:
        error, status = _provider_connect_error(exc, "mfl", "private")
        return jsonify(error), status
    return jsonify({"ok": True, "redirect_url": f"/mfl/{season}/{league_id}/dashboard"})


def _flea_private_credentials(data: dict) -> dict:
    """Derive a storeable Fleaflicker token from login or a pasted token."""
    from dashboard_services.providers.fleaflicker_api import login as flea_login, normalize_auth_token
    token = normalize_auth_token(data.get("token") or data.get("authorization"))
    flea_user_id = None
    email = str(data.get("email") or "").strip()
    password = str(data.get("password") or "")
    if email and password:
        session = flea_login(email, password)
        token = session["token"]
        flea_user_id = session.get("user_id")
    if not token:
        raise ValueError("Fleaflicker private leagues require a login token (or email + password).")
    out = {"token": token}
    if flea_user_id:
        out["flea_user_id"] = str(flea_user_id)
    return out


def _resolve_fleaflicker_team(provider, league_id, season, *, token=None, team_id=None, flea_user_id=None):
    from dashboard_services.providers.fleaflicker_api import resolve_fleaflicker_team_id
    users = provider.get_users(league_id, season, token=token)
    return resolve_fleaflicker_team_id(
        users, team_id=team_id, flea_user_id=flea_user_id,
    )


def _persist_fleaflicker_viewer(
    league_id: str, season: int, team_id: Optional[str], *, token: Optional[str] = None,
) -> None:
    """Resolve a Fleaflicker team id into the Flask viewer session."""
    if not team_id:
        return
    try:
        from dashboard_services.providers.registry import get_provider
        from utils.viewer_resolve import resolve_viewer_for_league
        provider = get_provider("fleaflicker")
        users = provider.get_users(league_id, season, token=token)
        rosters = provider.get_rosters(league_id, season, token=token)
        viewer = resolve_viewer_for_league(users, rosters, "", user_id=str(team_id))
        if viewer:
            from app import save_viewer_session
            save_viewer_session(viewer)
            session["viewer_platform"] = "fleaflicker"
    except Exception:
        logger.warning("[link/fleaflicker] viewer resolution failed", exc_info=True)


def _apply_fleaflicker_guest_viewer(token: str, league_id: str, season: int) -> None:
    """Set the viewer session for a guest private connect when owner id is known."""
    from dashboard_services.accounts import peek_private_provider_connection
    pending = peek_private_provider_connection(token, "fleaflicker", league_id, season)
    if not pending:
        return
    flea_user_id = pending.get("flea_user_id")
    auth_token = pending.get("token")
    if not flea_user_id or not auth_token:
        return
    try:
        from dashboard_services.providers.registry import get_provider
        provider = get_provider("fleaflicker")
        team_id = _resolve_fleaflicker_team(
            provider, league_id, season, token=auth_token, flea_user_id=flea_user_id,
        )
        _persist_fleaflicker_viewer(league_id, season, team_id, token=auth_token)
    except Exception:
        logger.warning("[link/fleaflicker/guest] viewer resolution failed", exc_info=True)


def _connect_fleaflicker(method: str):
    account_id = session.get("account_id")
    if not account_id:
        return jsonify({"ok": False, "error": "Sign in to connect a league."}), 401
    data = request.get_json(silent=True) or {}
    allowed = {"league_id", "season", "team_id"} if method == "public" else {
        "league_id", "season", "team_id", "token", "authorization", "email", "password",
    }
    if not isinstance(data, dict) or set(data) - allowed:
        return jsonify({"ok": False, "error": "Unexpected fields for this connection method."}), 400
    league_id = str(data.get("league_id") or "").strip()
    try:
        season = int(data.get("season") or _default_season())
    except (TypeError, ValueError):
        return jsonify({"ok": False, "error": "Season must be a valid year."}), 400
    if not league_id.isdigit() or not (2000 <= season <= 2100):
        return jsonify({"ok": False, "error": "Enter a valid numeric Fleaflicker League ID and season."}), 400
    requested_team_id = (str(data.get("team_id")).strip() or None) if data.get("team_id") else None
    credentials = None
    auth_token = None
    try:
        from dashboard_services.providers.registry import get_provider
        provider = get_provider("fleaflicker")
        if method == "private":
            credentials = _flea_private_credentials(data)
            auth_token = credentials.get("token")
            info = provider.connect_league(league_id, season, token=auth_token)
        else:
            info = provider.connect_league(league_id, season)
        team_id = _resolve_fleaflicker_team(
            provider, league_id, season, token=auth_token,
            team_id=requested_team_id,
            flea_user_id=(credentials or {}).get("flea_user_id"),
        )
        from dashboard_services.accounts import add_provider_league_connection
        add_provider_league_connection(
            account_id, "fleaflicker", league_id, season,
            info.get("name") or f"Fleaflicker League {league_id}", method,
            credentials=credentials, team_id=team_id,
        )
        _persist_fleaflicker_viewer(league_id, season, team_id, token=auth_token)
    except Exception as exc:
        error, status = _provider_connect_error(exc, "fleaflicker", method)
        return jsonify(error), status
    return jsonify({
        "ok": True, "platform": "fleaflicker", "connection_method": method,
        "league_id": league_id, "season": season, "name": info.get("name"),
        "team_id": team_id,
        "redirect_url": f"/fleaflicker/{season}/{league_id}/dashboard",
    })


@link_bp.post("/api/link/fleaflicker/public")
def link_fleaflicker_public():
    return _connect_fleaflicker("public")


@link_bp.post("/api/link/fleaflicker/private")
def link_fleaflicker_private():
    return _connect_fleaflicker("private")


@link_bp.get("/api/link/fleaflicker/preview")
def link_fleaflicker_preview():
    league_id = str(request.args.get("league_id") or "").strip()
    try:
        season = int(request.args.get("season") or _default_season())
    except (TypeError, ValueError):
        return jsonify({"ok": False, "error": "Season must be a valid year."}), 400
    if not league_id.isdigit() or not (2000 <= season <= 2100):
        return jsonify({"ok": False, "error": "Enter a valid numeric Fleaflicker League ID and season."}), 400
    token = str(request.args.get("token") or "").strip()
    try:
        from dashboard_services.providers.registry import get_provider
        from dashboard_services.providers.fleaflicker_api import normalize_auth_token
        provider = get_provider("fleaflicker")
        league = provider.get_league(league_id, season, token=normalize_auth_token(token) or None)
        users = provider.get_users(league_id, season, token=normalize_auth_token(token) or None)
    except Exception as exc:
        error, status = _provider_connect_error(exc, "fleaflicker", "preview")
        return jsonify(error), status
    teams = [{"team_id": str(u.get("roster_id") or u.get("user_id")),
              "name": (u.get("metadata") or {}).get("team_name") or u.get("display_name")}
             for u in users]
    my_team_id = None
    if token:
        from dashboard_services.providers.fleaflicker_api import resolve_fleaflicker_team_id
        pending = session.get("pending_provider_connection_token")
        flea_user_id = None
        if pending:
            from dashboard_services.accounts import peek_private_provider_connection
            staged = peek_private_provider_connection(
                pending, "fleaflicker", league_id, season,
            ) or {}
            flea_user_id = staged.get("flea_user_id")
        my_team_id = resolve_fleaflicker_team_id(
            users, flea_user_id=flea_user_id,
        )
    return jsonify({"ok": True, "platform": "fleaflicker", "league_id": league_id,
                    "season": season, "name": league.get("name"), "teams": teams,
                    "my_team_id": my_team_id})


@link_bp.post("/api/link/fleaflicker/private/pending")
def link_fleaflicker_private_pending():
    data = request.get_json(silent=True) or {}
    league_id = str(data.get("league_id") or "").strip()
    try:
        season = int(data.get("season") or _default_season())
    except (TypeError, ValueError):
        return jsonify({"ok": False, "error": "Invalid private Fleaflicker request."}), 400
    if not league_id.isdigit():
        return jsonify({"ok": False, "error": "Invalid private Fleaflicker request."}), 400
    try:
        from dashboard_services.providers.registry import get_provider
        credentials = _flea_private_credentials(data)
        info = get_provider("fleaflicker").connect_league(
            league_id, season, token=credentials.get("token"),
        )
        from dashboard_services.accounts import stage_private_provider_connection
        token = stage_private_provider_connection(
            "fleaflicker", league_id, season,
            info.get("name") or f"Fleaflicker League {league_id}", credentials,
        )
        session["pending_provider_connection_token"] = token
        return jsonify({
            "ok": True, "provider": "fleaflicker", "connection_method": "private",
            "league_id": league_id, "season": season, "name": info.get("name"),
            "auth_url": "/auth/google?intent=onboarding&next=/",
        })
    except Exception as exc:
        error, status = _provider_connect_error(exc, "fleaflicker", "private")
        return jsonify(error), status


@link_bp.post("/api/link/fleaflicker/private/guest")
def link_fleaflicker_private_guest():
    data = request.get_json(silent=True) or {}
    league_id = str(data.get("league_id") or "").strip()
    try:
        season = int(data.get("season") or _default_season())
    except (TypeError, ValueError):
        return jsonify({"ok": False, "error": "Invalid request."}), 400
    token = session.get("pending_provider_connection_token")
    from dashboard_services.accounts import peek_private_provider_connection
    if not peek_private_provider_connection(token, "fleaflicker", league_id, season):
        return jsonify({"ok": False, "error": "Private Fleaflicker session expired. Reconnect."}), 403
    _apply_fleaflicker_guest_viewer(token, league_id, season)
    return jsonify({
        "ok": True, "redirect_url": f"/fleaflicker/{season}/{league_id}/dashboard",
    })


@link_bp.post("/api/link/fleaflicker/private/saved")
def link_fleaflicker_private_saved():
    account_id = session.get("account_id")
    if not account_id:
        return jsonify({"ok": False, "error": "Sign in to open a saved private league."}), 401
    data = request.get_json(silent=True) or {}
    league_id = str(data.get("league_id") or "").strip()
    try:
        season = int(data.get("season") or _default_season())
    except (TypeError, ValueError):
        return jsonify({"ok": False, "error": "Invalid request."}), 400
    from dashboard_services.accounts import get_provider_league_credentials, mark_provider_connection_status
    credentials = get_provider_league_credentials(account_id, "fleaflicker", league_id, season) or {}
    if not credentials.get("token"):
        return jsonify({
            "ok": False,
            "error": "Sign in to Fleaflicker or paste a login token to connect this private league.",
        }), 400
    try:
        from dashboard_services.providers.registry import get_provider
        info = get_provider("fleaflicker").connect_league(
            league_id, season, token=credentials.get("token"),
        )
    except Exception as exc:
        error, status = _provider_connect_error(exc, "fleaflicker", "private")
        try:
            mark_provider_connection_status(
                account_id, "fleaflicker", league_id, season, "reauth_required",
                "fleaflicker_auth_rejected",
            )
        except Exception:
            pass
        return jsonify(error), status
    team_id = None
    try:
        from dashboard_services.providers.registry import get_provider
        from dashboard_services.providers.fleaflicker_api import resolve_fleaflicker_team_id
        provider = get_provider("fleaflicker")
        users = provider.get_users(league_id, season, token=credentials.get("token"))
        flea_uid = str(credentials.get("flea_user_id") or "").strip()
        if flea_uid:
            team_id = resolve_fleaflicker_team_id(users, flea_user_id=flea_uid)
        _persist_fleaflicker_viewer(league_id, season, team_id, token=credentials.get("token"))
    except Exception:
        logger.warning("[link/fleaflicker/saved] viewer resolution failed", exc_info=True)
    return jsonify({
        "ok": True, "platform": "fleaflicker", "connection_method": "private",
        "league_id": league_id, "season": season, "name": info.get("name"),
        "redirect_url": f"/fleaflicker/{season}/{league_id}/dashboard",
    })


@link_bp.post("/api/link/fleaflicker/reconnect")
def link_fleaflicker_reconnect():
    account_id = session.get("account_id")
    if not account_id:
        return jsonify({"ok": False, "error": "Sign in to reconnect."}), 401
    data = request.get_json(silent=True) or {}
    allowed = {"league_id", "season", "token", "authorization", "email", "password"}
    if not isinstance(data, dict) or set(data) - allowed:
        return jsonify({"ok": False, "error": "Unexpected fields for this connection method."}), 400
    league_id = str(data.get("league_id") or "").strip()
    try:
        season = int(data.get("season") or _default_season())
    except (TypeError, ValueError):
        return jsonify({"ok": False, "error": "Invalid request."}), 400
    from dashboard_services.accounts import owns_user_league, replace_provider_credentials
    if not owns_user_league(account_id, "fleaflicker", league_id, season):
        return jsonify({"ok": False, "error": "League not found on this account."}), 404
    try:
        credentials = _flea_private_credentials(data)
        from dashboard_services.providers.registry import get_provider
        get_provider("fleaflicker").connect_league(league_id, season, token=credentials.get("token"))
        if not replace_provider_credentials(account_id, "fleaflicker", league_id, season, credentials):
            return jsonify({"ok": False, "error": "Could not update credentials."}), 400
        flea_uid = str(credentials.get("flea_user_id") or "").strip()
        if flea_uid:
            from dashboard_services.accounts import link_platform_identity
            link_platform_identity(account_id, "fleaflicker", flea_uid)
        team_id = _resolve_fleaflicker_team(
            get_provider("fleaflicker"), league_id, season,
            token=credentials.get("token"), flea_user_id=flea_uid or None,
        )
        _persist_fleaflicker_viewer(league_id, season, team_id, token=credentials.get("token"))
    except Exception as exc:
        error, status = _provider_connect_error(exc, "fleaflicker", "private")
        return jsonify(error), status
    return jsonify({"ok": True, "redirect_url": f"/fleaflicker/{season}/{league_id}/dashboard"})
