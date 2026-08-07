"""Extracted from app.py — league_meta_bp (see route list below)."""
from __future__ import annotations
import logging
from datetime import datetime
import pandas as pd
from flask import Blueprint, jsonify, request, session
logger = logging.getLogger(__name__)

league_meta_bp = Blueprint("league_meta_bp", __name__)


# ── Lazy shims to app.py internals (resolved at request time) ──
def _api_err(*a, **k):
    from app import _api_err as _fn
    return _fn(*a, **k)

def _weighted_pos_strength(*a, **k):
    from app import _weighted_pos_strength as _fn
    return _fn(*a, **k)

def format_sleeper_league_option(*a, **k):
    from app import format_sleeper_league_option as _fn
    return _fn(*a, **k)

def get_available_history_seasons(*a, **k):
    from app import get_available_history_seasons as _fn
    return _fn(*a, **k)

def get_league(*a, **k):
    from app import get_league as _fn
    return _fn(*a, **k)

def get_league_ctx_from_cache(*a, **k):
    from app import get_league_ctx_from_cache as _fn
    return _fn(*a, **k)

def get_model_value_table_cached(*a, **k):
    from app import get_model_value_table_cached as _fn
    return _fn(*a, **k)

def get_nfl_state(*a, **k):
    from app import get_nfl_state as _fn
    return _fn(*a, **k)

def get_rosters(*a, **k):
    from app import get_rosters as _fn
    return _fn(*a, **k)

def get_sleeper_user_by_username(*a, **k):
    from app import get_sleeper_user_by_username as _fn
    return _fn(*a, **k)

def get_sleeper_user_leagues(*a, **k):
    from app import get_sleeper_user_leagues as _fn
    return _fn(*a, **k)

def resolve_league_id_for_season(*a, **k):
    from app import resolve_league_id_for_season as _fn
    return _fn(*a, **k)


@league_meta_bp.route("/api/draft-needs")
def api_draft_needs():
    """
    Returns positional needs for a team relative to league averages.
    Uses the same weighted positional strength + z-score approach as the Teams page.
    Need levels: -2 stacked, -1 depth, 0 neutral, 1 need, 2 major need.
    """
    try:
        from utils.utils import load_players_index, load_model_value_table
        league_id = request.args.get("league_id")
        platform  = request.args.get("platform", "sleeper")
        season    = int(request.args.get("season") or datetime.now().year)
        roster_id = request.args.get("roster_id") or ""

        # Fall back to the session viewer when roster_id is absent or the sentinel
        if not roster_id or roster_id == "viewer":
            roster_id = session.get("viewer_roster_id") or ""

        if not league_id or not roster_id:
            return jsonify({"error": "league_id and roster_id required"}), 400

        rosters = get_rosters(platform, league_id, season) or []
        league  = get_league(platform, league_id, season) or {}
        players_index = load_players_index() or {}
        value_table   = list(get_model_value_table_cached() or [])

        roster_positions = (league.get("roster_positions") or [])
        is_sf  = any(str(s).upper() in {"SUPER_FLEX", "SFLEX"} for s in roster_positions)
        vfield = "sf_value" if is_sf else "value"

        values_by_id = {str(r["id"]): r for r in value_table if isinstance(r, dict) and r.get("id")}

        CORE = ("QB", "RB", "WR", "TE")

        # Count roster slots for _weighted_pos_strength
        slot_counts: dict[str, int] = {}
        for rp in roster_positions:
            rp_str = str(rp).upper()
            slot_counts[rp_str] = slot_counts.get(rp_str, 0) + 1

        # Build per-roster positional value lists (same as teams page)
        roster_pos_vals: dict[str, dict[str, list]] = {}
        for r in rosters:
            rid = str(r.get("roster_id", ""))
            pv: dict[str, list] = {p: [] for p in CORE}
            for pid in (r.get("players") or []):
                meta = players_index.get(str(pid)) or {}
                pos  = str(meta.get("pos") or "").upper()
                if pos not in CORE:
                    continue
                vrow = values_by_id.get(str(pid)) or {}
                val  = float(vrow.get(vfield) or vrow.get("value") or 0)
                pv[pos].append(val)
            roster_pos_vals[rid] = pv

        if not roster_pos_vals:
            return jsonify({"needs": {}, "league_type": "sf" if is_sf else "1qb"})

        # Compute weighted positional strength per roster (mirrors _weighted_pos_strength)
        roster_strength: dict[str, dict[str, float]] = {}
        for rid, pv in roster_pos_vals.items():
            roster_strength[rid] = {
                pos: _weighted_pos_strength(pv[pos], pos, slot_counts)
                for pos in CORE
            }

        # League avg + std per position
        import math
        n = len(roster_strength)
        league_avg = {pos: sum(rv[pos] for rv in roster_strength.values()) / n for pos in CORE}
        league_std = {}
        for pos in CORE:
            variance = sum((rv[pos] - league_avg[pos]) ** 2 for rv in roster_strength.values()) / n
            league_std[pos] = math.sqrt(variance) if variance > 0 else 1.0

        viewer = roster_strength.get(str(roster_id), {p: 0.0 for p in CORE})

        needs: dict = {}
        for pos in CORE:
            mu    = league_avg[pos]
            sigma = league_std[pos]
            z     = (viewer[pos] - mu) / sigma if sigma > 0 else 0.0
            # Map z-score to need level (same thresholds as teams page z-score usage)
            if   z >= 1.0:  level = -2   # stacked
            elif z >= 0.35: level = -1   # depth
            elif z >= -0.35:level =  0   # neutral
            elif z >= -1.0: level =  1   # need
            else:           level =  2   # major need
            needs[pos]              = level
            needs[f"{pos}_count"]   = len(roster_pos_vals.get(str(roster_id), {}).get(pos, []))
            needs[f"{pos}_value"]   = round(viewer[pos], 1)
            needs[f"{pos}_avg"]     = round(league_avg[pos], 1)

        return jsonify({
            "needs": needs,
            "league_type": "sf" if is_sf else "1qb",
            "league_size": len(rosters),
        })

    except Exception as e:
        import traceback; traceback.print_exc()
        return _api_err("Request failed", e)


@league_meta_bp.route("/api/sleeper-user-leagues")
def api_sleeper_user_leagues():
    username = (request.args.get("username") or "").strip()

    # If no username provided, try to get from session
    if not username:
        username = session.get("viewer_username")

    if not username:
        return jsonify({"ok": False, "error": "Missing username"}), 400

    season = int(request.args.get("season") or get_nfl_state().get("season"))

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
        return jsonify({"ok": False, "error": str(e)}), 500


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
    season = int(request.args.get("season") or nfl_state.get("season") or datetime.now().year)

    try:
        from dashboard_services.providers.espn_api import get_league as espn_get_league
        info = espn_get_league(season, league_id)
        return jsonify({
            "ok": True,
            "league": {
                "league_id": info.get("league_id"),
                "name": info.get("name") or f"ESPN League {league_id}",
                "season": info.get("season"),
            },
        })
    except Exception as e:
        # Map the espn_api library's typed errors to clear, actionable messages.
        # ESPNAccessDenied = 401 (private league + missing/wrong/expired cookies);
        # ESPNInvalidLeague = 404 (league id or season not found).
        from dashboard_services.providers.espn_api import espn_diagnostics
        diag = espn_diagnostics()
        creds_present = diag["espn_s2_present"] and diag["espn_swid_present"]
        name = type(e).__name__

        if name == "ESPNAccessDenied":
            if not creds_present:
                err = ("This is a private ESPN league, but the server has no "
                       "ESPN_S2 / ESPN_SWID set. Add them as environment variables "
                       "and redeploy.")
            else:
                err = ("ESPN rejected the ESPN_S2 / ESPN_SWID cookies. They may be "
                       "for a different ESPN account, expired, or copied with extra "
                       "characters — re-copy both from espn.com and update the env "
                       "vars.")
            return jsonify({"ok": False, "error": err, "diagnostics": diag}), 403

        if name == "ESPNInvalidLeague":
            err = (f"ESPN couldn't find league {league_id} for {season}. Check the "
                   f"league ID, and that the {season} season exists for this league "
                   f"(it may not have rolled over yet).")
            return jsonify({"ok": False, "error": err, "diagnostics": diag}), 404

        return jsonify({
            "ok": False,
            "error": f"Could not load ESPN league: {e}",
            "diagnostics": diag,
        }), 500


@league_meta_bp.route("/api/espn-debug")
def api_espn_debug():
    """Read-only diagnostics for a 'my private ESPN league won't load' report.

    Reports whether the server can see ESPN_S2 / ESPN_SWID (presence + length
    only — never the values) and, if a league_id is given, the exact result of
    trying to load it. Hit /api/espn-debug?league_id=<id>[&season=<yr>].
    """
    from dashboard_services.providers.espn_api import espn_diagnostics
    out = {"diagnostics": espn_diagnostics()}

    league_id = (request.args.get("league_id") or "").strip()
    if league_id.isdigit():
        nfl_state = get_nfl_state() or {}
        season = int(request.args.get("season") or nfl_state.get("season") or datetime.now().year)
        try:
            from dashboard_services.providers.espn_api import get_league as espn_get_league
            info = espn_get_league(season, league_id)
            out["league_load"] = {"ok": True, "season": season, "name": info.get("name")}
        except Exception as e:
            out["league_load"] = {
                "ok": False,
                "season": season,
                "error_type": type(e).__name__,
                "error": str(e)[:300],
            }
    return jsonify(out)
