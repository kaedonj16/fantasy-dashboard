"""Extracted from app.py -- league_meta_bp (see route list below)."""
from __future__ import annotations
import logging
import os
import hmac
from datetime import datetime
import pandas as pd
from flask import Blueprint, jsonify, request, session
from utils.api_params import api_int

from dashboard_services.api import (
    get_sleeper_user_by_username,
    get_sleeper_user_leagues,
)

logger = logging.getLogger(__name__)

league_meta_bp = Blueprint("league_meta_bp", __name__)


# ── Lazy shims to app.py internals (resolved at request time) ──
def format_sleeper_league_option(*a, **k):
    # Lives in utils.league_payload (a pure module), not app.py -- import it
    # directly so the Sleeper league-option formatting doesn't depend on app
    # re-exporting the name (which it no longer does).
    from utils.league_payload import format_sleeper_league_option as _fn
    return _fn(*a, **k)

def get_available_history_seasons(*a, **k):
    from app import get_available_history_seasons as _fn
    return _fn(*a, **k)

def get_league_ctx_from_cache(*a, **k):
    from app import get_league_ctx_from_cache as _fn
    return _fn(*a, **k)

def get_nfl_state(*a, **k):
    from app import get_nfl_state as _fn
    return _fn(*a, **k)

def resolve_league_id_for_season(*a, **k):
    from app import resolve_league_id_for_season as _fn
    return _fn(*a, **k)


@league_meta_bp.route("/api/sleeper-user-leagues")
def api_sleeper_user_leagues():
    username = (request.args.get("username") or "").strip()

    # If no username provided, try to get from session
    if not username:
        username = session.get("viewer_username")

    if not username:
        return jsonify({"ok": False, "error": "Missing username"}), 400

    season = api_int("season", (get_nfl_state() or {}).get("season"))

    try:
        user = get_sleeper_user_by_username(username)
        if not user:
            return jsonify({"ok": False, "error": "Sleeper username not found"}), 404

        leagues = get_sleeper_user_leagues(user["user_id"], season)

        # Optional: filter out leagues without usable ids
        leagues = [lg for lg in leagues if lg.get("league_id")]

        # Optional: sort nicer
        leagues.sort(
            key=lambda lg: (
                str(lg.get("status") or "") != "in_season",
                -(int(lg.get("total_rosters") or 0)),
                str(lg.get("name") or "").lower(),
            )
        )

        return jsonify({
            "ok": True,
            "user": {
                "user_id": user.get("user_id"),
                "username": user.get("username"),
                "display_name": user.get("display_name"),
                "avatar": user.get("avatar"),
            },
            "leagues": [format_sleeper_league_option(lg) for lg in leagues],
        })
    except Exception as e:
        logger.exception("[api/sleeper-user-leagues] unexpected error")
        return jsonify({"ok": False, "error": "Internal error"}), 500


@league_meta_bp.route("/api/my-leagues")
def api_my_leagues():
    """The signed-in user's leagues for the league switcher.

    Sourced identically to the My Leagues page (/portfolio) via the shared
    resolve_my_leagues() builder. Every league saved to the Google account is
    returned regardless of platform, then linked Sleeper identities may provide
    live metadata enrichment. Each entry carries its own platform
    so the switcher can navigate cross-platform."""
    out = []
    try:
        _cur_season = int((get_nfl_state() or {}).get("season") or 0) or None
    except Exception:
        _cur_season = None
    try:
        from dashboard_services.accounts import resolve_my_leagues
        import inspect
        _params = inspect.signature(resolve_my_leagues).parameters
        _args = (
            session.get("viewer_user_id"), session.get("account_id"), _cur_season,
        )
        if "enrich_live" in _params:
            leagues, _season = resolve_my_leagues(
                *_args, enrich_live=not bool(session.get("account_id")),
            )
        else:  # compatibility for injected/legacy resolvers
            leagues, _season = resolve_my_leagues(*_args)
        from utils.league_chrome import fields_from_provider_league, format_label
        for m in leagues:
            plat = m.get("platform") or "sleeper"
            season = m.get("season")
            name = m.get("name") or f"{plat.title()} League"
            label = f"{name} · {season}" if season else name
            live = fields_from_provider_league(m)
            row = {
                "platform": plat,
                "league_id": m.get("league_id"),
                "season": season,
                "name": name,
                "label": label,
                "connection_status": m.get("connection_status") or "connected",
                "last_synced_at": m.get("last_synced_at"),
                "last_successful_sync_at": m.get("last_successful_sync_at"),
                "needs_reconnect": m.get("connection_status") == "reauth_required",
                "is_favorite": bool(m.get("is_favorite")),
            }
            if live.get("has_format"):
                row["sf"] = bool(live.get("is_sf"))
                if live.get("size") and int(live["size"]) >= 2:
                    row["size"] = int(live["size"])
                row["format"] = format_label(live.get("size") or 0, bool(live.get("is_sf")))
            out.append(row)
        # Fire-and-forget membership reconciliation runs only after the list is
        # built, in its own guard: a scheduler hiccup (e.g. thread exhaustion)
        # must never discard the resolved leagues and silently downgrade
        # clients to a fallback league source.
        if session.get("account_id"):
            try:
                from dashboard_services.accounts import schedule_account_league_reconciliation
                schedule_account_league_reconciliation(session.get("account_id"), _season)
            except Exception:
                logger.debug("[my-leagues] reconcile schedule failed", exc_info=True)
    except Exception as exc:
        logger.warning("[my-leagues] resolve failed: %s", exc)

    resp = jsonify({"ok": True, "leagues": out})
    # Account-scoped list must never be cached -- leagues added on another device
    # need to appear on the next refresh/tab focus here.
    resp.headers["Cache-Control"] = "no-store"
    return resp


@league_meta_bp.route("/api/my-leagues/favorite", methods=["POST"])
def api_favorite_my_league():
    """Set an account-owned league's durable favorite state."""
    account_id = session.get("account_id")
    if not account_id:
        return jsonify({"ok": False, "error": "Sign in to save favorites"}), 401
    data = request.get_json(silent=True) or {}
    platform = str(data.get("platform") or "").strip().lower()
    league_id = str(data.get("league_id") or "").strip()
    favorite = data.get("favorite")
    if not platform or not league_id or not isinstance(favorite, bool):
        return jsonify({"ok": False, "error": "Invalid favorite request"}), 400
    from dashboard_services.accounts import set_user_league_favorite
    if not set_user_league_favorite(account_id, platform, league_id, favorite):
        return jsonify({"ok": False, "error": "League not found"}), 404
    resp = jsonify({"ok": True, "is_favorite": favorite})
    resp.headers["Cache-Control"] = "no-store"
    return resp


@league_meta_bp.route("/api/weekly-trends")
def api_weekly_trends():
    """Bulk per-player usage trend map (last-6-week series + recent-vs-season
    delta) for the leaderboard trend column and waivers usage risers."""
    nfl_state = get_nfl_state() or {}
    season_str = (request.args.get("season") or "").strip()
    season = int(season_str) if season_str.isdigit() else int(
        nfl_state.get("season") or datetime.now().year
    )
    try:
        from data_building.weekly_metrics import get_usage_trends
        trends = get_usage_trends(season)
    except Exception as exc:
        logger.warning("[weekly-trends] failed: %s", exc)
        trends = {}
    return jsonify({"season": season, "players": trends})


@league_meta_bp.route("/api/rivalry/<platform>/<int:season>/<league_id>")
def api_rivalry(platform: str, season: int, league_id: str):
    """All-time head-to-head record between two managers (by user_id).

    Scans every available historical season plus the current one, pairs
    finalized matchup rows by (week, matchup_id), and returns the full
    meeting list so the frontend can render record, streaks, and blowouts.
    """
    a = (request.args.get("a") or "").strip()
    b = (request.args.get("b") or "").strip()
    if not a or not b or a == b:
        return jsonify({"error": "Pick two different managers."}), 400

    try:
        available = get_available_history_seasons(platform, league_id, season) or []
    except Exception:
        available = []
    seasons_to_scan = sorted(set(int(s) for s in available) | {int(season)})

    games = []
    for hist_s in seasons_to_scan:
        try:
            rid = resolve_league_id_for_season(platform, league_id, season, hist_s)
            ctx = get_league_ctx_from_cache(platform, rid, hist_s)
        except Exception:
            continue

        df = ctx.get("df_weekly", pd.DataFrame())
        if df.empty or "matchup_id" not in df.columns or "roster_id" not in df.columns:
            continue

        roster_to_uid = {
            str(r.get("roster_id")): str(r.get("owner_id") or "")
            for r in (ctx.get("rosters") or [])
        }

        d = df.copy()
        if "finalized" in d.columns:
            d = d[d["finalized"] == True]  # noqa: E712
        d["__uid"] = d["roster_id"].astype(str).map(roster_to_uid)
        sub = d[d["__uid"].isin([a, b])]
        if sub.empty:
            continue

        for (wk, mid), grp in sub.groupby(["week", "matchup_id"]):
            if pd.isna(mid) or len(grp) != 2:
                continue
            if set(grp["__uid"]) != {a, b}:
                continue
            row_a = grp[grp["__uid"] == a].iloc[0]
            row_b = grp[grp["__uid"] == b].iloc[0]
            pts_a = float(row_a.get("points") or 0)
            pts_b = float(row_b.get("points") or 0)
            if pts_a == 0 and pts_b == 0:
                continue  # unplayed/empty matchup
            games.append({
                "season": int(hist_s),
                "week": int(wk),
                "a_pts": round(pts_a, 2),
                "b_pts": round(pts_b, 2),
            })

    games.sort(key=lambda g: (g["season"], g["week"]))
    wins_a = sum(1 for g in games if g["a_pts"] > g["b_pts"])
    wins_b = sum(1 for g in games if g["b_pts"] > g["a_pts"])
    return jsonify({
        "games": games,
        "wins_a": wins_a,
        "wins_b": wins_b,
        "ties": len(games) - wins_a - wins_b,
        "pts_a": round(sum(g["a_pts"] for g in games), 2),
        "pts_b": round(sum(g["b_pts"] for g in games), 2),
    })


@league_meta_bp.route("/api/espn-validate-league")
def api_espn_validate_league():
    """
    Validate an ESPN league ID and return basic league info.
    Used by the landing page ESPN flow.
    """
    league_id = (request.args.get("league_id") or "").strip()
    if not league_id or not league_id.isdigit():
        return jsonify({"ok": False, "error": "Invalid ESPN league ID. Must be a number."}), 400

    nfl_state = get_nfl_state() or {}
    season = api_int("season", nfl_state.get("season") or datetime.now().year)

    try:
        # Public validation must never fall back to server/account cookies.
        from dashboard_services.providers.espn_api import connect_league
        info = connect_league(season, league_id)
        # Teams unlock the home/link "your team" picker so ESPN username/team
        # name is passed into the viewer session (needed for Scout and peers).
        teams = [
            {"team_id": str(t.get("id") or ""), "name": str(t.get("name") or "").strip() or f"Team {t.get('id')}"}
            for t in (info.get("teams") or [])
            if t.get("id") is not None
        ]
        return jsonify({
            "ok": True,
            "league": {
                "league_id": info.get("league_id"),
                "name": info.get("name") or f"ESPN League {league_id}",
                "season": info.get("season"),
            },
            "teams": teams,
        })
    except Exception as e:
        # Map the espn_api library's typed errors to clear, actionable messages.
        # ESPNAccessDenied = 401 (private league + missing/wrong/expired cookies);
        # ESPNInvalidLeague = 404 (league id or season not found).
        from dashboard_services.providers.espn_api import espn_diagnostics
        diag = espn_diagnostics()
        name = type(e).__name__

        if name == "ESPNAccessDenied":
            err = ("This ESPN league could not be accessed publicly. If it is a "
                   "private league, connect using the Private League option.")
            return jsonify({"ok": False, "error": err, "diagnostics": diag}), 403

        if name == "ESPNInvalidLeague":
            err = (f"ESPN couldn't find league {league_id} for {season}. Check the "
                   f"league ID, and that the {season} season exists for this league "
                   f"(it may not have rolled over yet).")
            return jsonify({"ok": False, "error": err, "diagnostics": diag}), 404

        if name == "ESPNRateLimited":
            return jsonify({
                "ok": False,
                "error": "ESPN is rate limiting requests. Please wait a moment and try again.",
            }), 429

        if name == "ESPNUnavailable":
            return jsonify({
                "ok": False,
                "error": "ESPN is temporarily unavailable. Please try again later.",
            }), 503

        if name == "ESPNMalformedResponse":
            return jsonify({
                "ok": False,
                "error": ("ESPN returned incomplete league data. Check that the league "
                          "and current season are available, then try again."),
            }), 422

        return jsonify({
            "ok": False,
            "error": "Could not load that ESPN league. Please try again.",
        }), 500


@league_meta_bp.route("/api/espn-debug")
def api_espn_debug():
    """Read-only diagnostics for a 'my private ESPN league won't load' report.

    Reports whether the server can see ESPN_S2 / ESPN_SWID (presence + length
    only -- never the values) and, if a league_id is given, the exact result of
    trying to load it. Hit /api/espn-debug?league_id=<id>[&season=<yr>].

    Auth: CRON_SECRET gate, same as /api/debug-values. Pass the secret as
    ``?secret=``, an ``X-Cron-Secret`` header, or JSON ``{"secret": ...}``.
    Fails closed with 403 when CRON_SECRET is unset or does not match.
    """
    # CRON_SECRET check, identical to /api/debug-values: require the secret to
    # be set AND match. Fails closed when CRON_SECRET is unset.
    secret = os.environ.get("CRON_SECRET", "")
    provided = str(
        request.args.get("secret")
        or request.headers.get("X-Cron-Secret")
        or (request.get_json(silent=True) or {}).get("secret")
        or ""
    )
    if not secret or not provided or not hmac.compare_digest(provided, secret):
        return jsonify({"error": "unauthorized"}), 403

    from dashboard_services.providers.espn_api import espn_diagnostics
    out = {"diagnostics": espn_diagnostics()}

    league_id = (request.args.get("league_id") or "").strip()
    if league_id.isdigit():
        nfl_state = get_nfl_state() or {}
        season = api_int("season", nfl_state.get("season") or datetime.now().year)
        try:
            from dashboard_services.providers.espn_api import get_league as espn_get_league
            info = espn_get_league(season, league_id)
            out["league_load"] = {"ok": True, "season": season, "name": info.get("name")}
        except Exception as e:
            logger.exception("[api/espn-debug] league load failed for %s/%s", season, league_id)
            out["league_load"] = {
                "ok": False,
                "season": season,
                "error_type": type(e).__name__,
                "error": "Internal error",
            }
    return jsonify(out)


@league_meta_bp.route("/api/lineup-lock-hint")
def api_lineup_lock_hint():
    """Thin hint for in-app lineup-lock toast (R06.3).

  Returns whether we're in the pre-kickoff window and the viewer's lineup has
  hard issues or a material bench upgrade. Best-effort; always safe to ignore.
    """
    from datetime import timezone

    league_id = (request.args.get("league_id") or session.get("last_league_id") or "").strip()
    platform = (request.args.get("platform") or session.get("last_platform") or "sleeper").strip().lower()
    if not league_id:
        return jsonify({"ok": False})

    nfl_state = get_nfl_state() or {}
    try:
        season = int(request.args.get("season") or session.get("last_season") or nfl_state.get("season") or datetime.now().year)
    except (TypeError, ValueError):
        season = int(nfl_state.get("season") or datetime.now().year)
    week = int(nfl_state.get("week") or 0)
    if not week or nfl_state.get("season_type") not in ("reg", "post"):
        return jsonify({"ok": False})

    try:
        from utils.utils import load_week_schedule
        games = load_week_schedule(season, week) or []
        # gameTime_epoch arrives as a string from the JSON schedule cache.
        epochs = []
        for g in games:
            try:
                epochs.append(float(g.get("gameTime_epoch")))
            except (TypeError, ValueError):
                continue
        if not epochs:
            return jsonify({"ok": False})
        # gameTime_epoch is seconds (nfl_game_data._iso_epoch); every other
        # consumer uses fromtimestamp() directly, so no /1000 here.
        kickoff = datetime.fromtimestamp(min(epochs), tz=timezone.utc)
        now = datetime.now(tz=timezone.utc)
        mins = (kickoff - now).total_seconds() / 60
        in_window = 40 <= mins <= 100
    except Exception:
        return jsonify({"ok": False})

    if not in_window:
        return jsonify({"ok": True, "in_window": False, "has_issues": False, "season": season, "week": week})

    viewer_roster_id = session.get("viewer_roster_id")
    if not viewer_roster_id:
        return jsonify({"ok": True, "in_window": True, "has_issues": False, "season": season, "week": week})

    try:
        from utils.lineup_issues import (
            find_lineup_issues,
            format_lineup_lock_swaps,
            projection_upgrades,
            summarize_issues,
        )
        from dashboard_services.api import get_nfl_players
        from dashboard_services.platform_api import get_league

        ctx = get_league_ctx_from_cache(platform, league_id, season) or {}
        rosters = ctx.get("rosters") or []
        roster = next(
            (r for r in rosters if str(r.get("roster_id")) == str(viewer_roster_id)),
            None,
        )
        if not roster:
            return jsonify({"ok": True, "in_window": True, "has_issues": False, "season": season, "week": week})

        starters = [str(p) for p in (roster.get("starters") or [])]
        nfl_players = get_nfl_players() or {}
        teams_playing = set()
        for g in games:
            for side in ("home", "away"):
                t = str(g.get(side) or "").upper()
                if t:
                    teams_playing.add(t)

        player_info = {}
        for pid in starters:
            pl = nfl_players.get(pid) or {}
            player_info[pid] = {
                "name": pl.get("full_name") or pl.get("last_name") or "",
                "team": pl.get("team") or "",
                "injury_status": pl.get("injury_status") or "",
            }
        issues = find_lineup_issues(starters, player_info, teams_playing)
        message = ""
        if issues:
            message = f"Week {week} kicks off soon. {summarize_issues(issues)}."

        league = get_league(platform, str(league_id), int(season)) or {}
        roster_positions = [str(s) for s in (league.get("roster_positions") or [])]
        proj_map_wk = {}
        try:
            from app import build_projections_by_week
            _bpw = build_projections_by_week(season, int(week), None) or {}
            proj_map_wk = {
                str(k): v
                for k, v in ((_bpw.get(int(week)) or {}).get("projections") or {}).items()
            }
        except Exception:
            proj_map_wk = {}
        if proj_map_wk and roster_positions:
            reserve_set = {str(p) for p in (roster.get("reserve") or [])}
            taxi_set = {str(p) for p in (roster.get("taxi") or [])}
            eligible = [
                str(p) for p in (roster.get("players") or [])
                if str(p) not in reserve_set and str(p) not in taxi_set
            ]
            pos_map = {
                pid: str((nfl_players.get(pid) or {}).get("position") or "")
                for pid in eligible
            }
            injury_status = {
                pid: str((nfl_players.get(pid) or {}).get("injury_status") or "")
                for pid in eligible
            }
            swaps = projection_upgrades(
                starters, eligible, proj_map_wk, pos_map,
                roster_positions, min_gain=2.0, max_swaps=2,
                injury_status=injury_status,
            )
            if swaps:
                names = {}
                for sw in swaps:
                    for pid in (sw.get("in"), sw.get("out")):
                        pl = nfl_players.get(str(pid or "")) or {}
                        names[str(pid)] = (
                            pl.get("full_name") or pl.get("last_name") or ""
                        )
                swap_line = format_lineup_lock_swaps(swaps, names)
                if message:
                    message = f"{message} {swap_line}."
                else:
                    message = f"Week {week} kicks off soon. {swap_line}."

        return jsonify({
            "ok": True,
            "in_window": True,
            "has_issues": bool(message),
            "message": message,
            "season": season,
            "week": week,
        })
    except Exception:
        logger.debug("lineup-lock-hint failed", exc_info=True)
        return jsonify({"ok": True, "in_window": True, "has_issues": False, "season": season, "week": week})
