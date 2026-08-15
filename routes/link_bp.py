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
from datetime import datetime

from flask import Blueprint, jsonify, request, session

link_bp = Blueprint("link", __name__)
logger = logging.getLogger(__name__)

_ESPN_PUBLIC_FIELDS = {"league_id", "season"}
_ESPN_PRIVATE_FIELDS = {"league_id", "season", "swid", "espn_s2"}


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
    if provider not in ("espn", "sleeper", "yahoo") or method not in ("public", "private"):
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
    if name == "ESPNInvalidLeague" or "404" in msg:
        return "No ESPN league was found for that ID and season.", 404
    if name == "ESPNAccessDenied" or "401" in msg or "403" in msg:
        if method == "public":
            return ("This ESPN league could not be accessed publicly. If it is a private "
                    "league, connect using the Private League option."), 403
        return "ESPN rejected these credentials or the session has expired.", 403
    if "429" in msg or "rate" in msg:
        return "ESPN is rate limiting requests. Please wait a moment and try again.", 429
    if "timeout" in msg or "500" in msg or "502" in msg or "503" in msg:
        return "ESPN is temporarily unavailable. Please try again later.", 503
    return "ESPN returned an unexpected response. Please verify the details and try again.", 502


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
    swid, espn_s2 = str(data.get("swid") or "").strip(), str(data.get("espn_s2") or "").strip()
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
        "auth_url": "/auth/google?intent=onboarding&next=/",
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
    swid, espn_s2 = str(data.get("swid") or "").strip(), str(data.get("espn_s2") or "").strip()
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


@link_bp.route("/api/link/yahoo/preview")
def link_yahoo_preview():
    from dashboard_services.providers.yahoo_api import yahoo_enabled
    if not yahoo_enabled():
        return jsonify({"ok": False, "error": "Yahoo connections are temporarily unavailable."}), 503
    league_id = (request.args.get("league_id") or "").strip()
    if not league_id:
        return jsonify({"ok": False, "error": "Yahoo league ID required."}), 400
    access_token = session.get("yahoo_access_token") or ""
    if not access_token:
        # The modal sends the user through /auth/yahoo and returns here.
        return jsonify({"ok": False, "needs_oauth": True, "auth_url": "/auth/yahoo?next=/portfolio"}), 401
    season = int(request.args.get("season") or _default_season())
    try:
        from dashboard_services.providers.yahoo_api import get_league, get_users, resolve_league_key
        # Resolve the real, season-specific league key first. Yahoo's "nfl" game
        # code only reaches the current season, so a prior-season league 403s even
        # for a member without this. "absent" => account isn't in that league;
        # "unknown" => couldn't list, so fall through to a direct fetch.
        resolved = resolve_league_key(access_token, league_id)
        if resolved.get("status") == "absent":
            return jsonify({
                "ok": False, "needs_oauth": True, "auth_url": "/auth/yahoo?reauth=1&next=/portfolio",
                "error": ("That Yahoo account isn't in any league with ID " + league_id +
                          ". Check the ID, or reconnect with the account that's in it."),
            }), 401
        if resolved.get("season"):
            season = int(resolved["season"])
        league = get_league(season, league_id, access_token)
        users = get_users(season, league_id, access_token)
    except Exception as exc:
        msg = str(exc)
        logger.warning("[link/yahoo] preview failed: %s", msg)
        # 403 = valid token but the authorized account can't see this league.
        # Drop the stale token and re-offer OAuth so they can reconnect with the
        # right Yahoo account instead of dead-ending on the error.
        if "403" in msg or "Forbidden" in msg:
            session.pop("yahoo_access_token", None)
            session.pop("yahoo_guid", None)
            return jsonify({
                "ok": False, "needs_oauth": True, "auth_url": "/auth/yahoo?reauth=1&next=/portfolio",
                "error": ("That Yahoo account can't access league " + league_id +
                          ". Reconnect with the account that's in this league."),
            }), 401
        return jsonify({"ok": False, "error": "Could not load that Yahoo league (check the ID)."}), 400
    my_guid = session.get("yahoo_guid") or ""
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


@link_bp.route("/api/link/pending", methods=["POST"])
def link_pending():
    """Stash a league the user selected *before* signing in, then send them to
    Google. The callback attaches it to the new account and drops them into it —
    the select-league-then-login onboarding order."""
    data = request.get_json(force=True) or {}
    platform = (data.get("platform") or "").strip().lower()
    league_id = str(data.get("league_id") or "").strip()
    if platform not in ("espn", "yahoo", "sleeper") or not league_id:
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
    if platform not in ("espn", "yahoo", "sleeper") or not league_id:
        return jsonify({"ok": False, "error": "Missing platform or league_id."}), 400
    season = data.get("season")
    try:
        season = int(season) if season else None
    except (TypeError, ValueError):
        season = None
    team_id = (str(data.get("team_id")).strip() or None) if data.get("team_id") else None
    name = (str(data.get("name")).strip() or None) if data.get("name") else None
    try:
        from dashboard_services.accounts import add_user_league
        add_user_league(account_id, platform, league_id, season=season, team_id=team_id, name=name)
    except Exception as exc:
        logger.warning("[link/add] failed: %s", exc)
        return jsonify({"ok": False, "error": "Could not save that league."}), 500
    return jsonify({"ok": True})
