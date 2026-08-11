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


def _default_season() -> int:
    try:
        from dashboard_services.api import get_nfl_state
        return int((get_nfl_state() or {}).get("season") or datetime.now().year)
    except Exception:
        return datetime.now().year


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
    league_id = (request.args.get("league_id") or "").strip()
    if not league_id:
        return jsonify({"ok": False, "error": "Yahoo league ID required."}), 400
    access_token = session.get("yahoo_access_token") or ""
    if not access_token:
        # The modal sends the user through /auth/yahoo and returns here.
        return jsonify({"ok": False, "needs_oauth": True, "auth_url": "/auth/yahoo?next=/portfolio"}), 401
    season = int(request.args.get("season") or _default_season())
    try:
        from dashboard_services.providers.yahoo_api import get_league, get_users
        league = get_league(season, league_id, access_token)
        users = get_users(season, league_id, access_token)
    except Exception as exc:
        logger.warning("[link/yahoo] preview failed: %s", exc)
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
