"""
Live-draft API endpoints: detect a league's drafts and poll live state.

Sleeper remains the original live board. ESPN uses the DraftSyncProvider
companion (observe picks only; never submit). Routes:

    /api/draft/detect   (api_draft_detect)
    /api/draft/live     (api_draft_live)

Extracted from app.py to reduce monolith size. The two draft-type/order helpers
live in app.py and are reached via the lazy shims below, so importing this
blueprint at start-up stays free of a circular import — they're resolved only
when a request is served. Everything else comes from dashboard_services.
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional, Tuple

from flask import Blueprint, jsonify, request, session

from dashboard_services.api import build_league_history_map
from dashboard_services.platform_api import (
    get_drafts, get_rosters, get_traded_picks, get_users,
)

logger = logging.getLogger(__name__)

draft_api_bp = Blueprint("draft_api", __name__)


# ── Lazy shims to app.py internals (resolved at request time) ─────────────────

def _live_draft_type(*args, **kwargs):
    from app import _live_draft_type as _fn
    return _fn(*args, **kwargs)


def _order_from_sleeper(*args, **kwargs):
    from app import _order_from_sleeper as _fn
    return _fn(*args, **kwargs)


def _viewer_ids():
    try:
        return (
            str(session.get("viewer_user_id") or "") or None,
            str(session.get("viewer_roster_id") or "") or None,
        )
    except RuntimeError:
        return None, None


def _espn_detect_sync(league_id: str, season: int):
    from dashboard_services.draft_sync import (
        DraftSyncAuthError,
        DraftSyncError,
        get_draft_sync_provider,
        snapshot_to_detect_record,
    )
    try:
        viewer_uid, viewer_rid = _viewer_ids()
        snapshot = get_draft_sync_provider("espn").get_snapshot(
            league_id, season,
            viewer_user_id=viewer_uid, viewer_roster_id=viewer_rid,
        )
    except DraftSyncAuthError:
        return jsonify({"drafts": [], "error": "auth_denied", "retry": False})
    except DraftSyncError as exc:
        logger.warning("[draft-detect] espn sync error_type=%s code=%s", type(exc).__name__, exc.code)
        return jsonify({"drafts": [], "error": exc.code, "retry": bool(exc.retry)})
    except Exception:
        logger.warning("[draft-detect] espn sync error_type=Exception")
        return jsonify({"drafts": [], "error": "fetch_failed"})
    return jsonify({"drafts": [snapshot_to_detect_record(snapshot)], "source": "espn"})


def _espn_live(draft_id: str, league_id: str, season: int):
    from dashboard_services.draft_sync import (
        DraftSyncAuthError,
        DraftSyncError,
        get_draft_sync_provider,
        parse_espn_draft_id,
        snapshot_to_live_payload,
    )
    from dashboard_services.espn_draft_relay import get_relay_snapshot, merge_live_with_relay

    parsed = parse_espn_draft_id(draft_id) if draft_id else None
    if parsed:
        league_id, season = parsed[0], parsed[1]
    if not league_id:
        return jsonify({"error": "league_required", "retry": False}), 400
    if not season:
        season = datetime.now().year
    try:
        viewer_uid, viewer_rid = _viewer_ids()
        snapshot = get_draft_sync_provider("espn").get_snapshot(
            league_id, int(season),
            viewer_user_id=viewer_uid, viewer_roster_id=viewer_rid,
        )
    except DraftSyncAuthError:
        return jsonify({"error": "auth_denied", "retry": False}), 403
    except DraftSyncError as exc:
        logger.warning("[draft-live] espn error_type=%s code=%s", type(exc).__name__, exc.code)
        status = 404 if exc.code == "not_found" else 502
        return jsonify({"error": exc.code, "retry": bool(exc.retry)}), status
    except Exception:
        logger.warning("[draft-live] espn error_type=Exception")
        return jsonify({"error": "fetch_failed", "retry": True}), 502
    payload = snapshot_to_live_payload(snapshot)
    try:
        payload = merge_live_with_relay(payload, get_relay_snapshot(str(league_id), int(season)))
    except Exception:
        logger.info("[draft-live] relay merge skipped error_type=Exception")
    # Never echo provider secrets even if a future field is added upstream.
    for secret_key in ("espn_s2", "swid", "SWID", "cookie", "cookies"):
        payload.pop(secret_key, None)
    return jsonify(payload)


def _espn_relay_normalize(body: dict):
    """Map extension-observed ESPN draft picks onto the live board pick shape."""
    from dashboard_services.draft_sync import normalize_espn_relay_payload
    from dashboard_services.providers import espn_api
    from dashboard_services.providers.espn_draft import _dst_from_espn_id, _player_lookup

    try:
        canon = espn_api._espn_to_canon_cached()
    except Exception:
        canon = {}
    return normalize_espn_relay_payload(
        body,
        espn_to_canon=canon,
        player_lookup=_player_lookup,
        dst_mapper=_dst_from_espn_id,
    )


def _relay_cors(resp):
    """Allow the ESPN draft origin to POST picks from the mobile bookmarklet."""
    origin = (request.headers.get("Origin") or "").strip()
    allowed = (
        origin.endswith(".espn.com")
        or origin == "https://fantasy.espn.com"
        or origin == "https://www.espn.com"
    )
    if allowed:
        resp.headers["Access-Control-Allow-Origin"] = origin
        resp.headers["Access-Control-Allow-Methods"] = "POST, GET, OPTIONS"
        resp.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
        resp.headers["Access-Control-Max-Age"] = "86400"
        resp.headers["Vary"] = "Origin"
    return resp


def _bearer_token() -> str:
    auth = (request.headers.get("Authorization") or "").strip()
    if auth.lower().startswith("bearer "):
        return auth[7:].strip()
    return (request.args.get("token") or "").strip()


def _relay_auth_ok(league_id: str, season: int) -> Tuple[bool, Optional[str]]:
    """Session (Draft Room / extension) or signed mobile token."""
    from dashboard_services.espn_draft_relay import verify_relay_token

    token = _bearer_token()
    if token:
        claims = verify_relay_token(token)
        if not claims:
            return False, "invalid_token"
        if str(claims["league_id"]) != str(league_id):
            return False, "league_mismatch"
        if int(claims["season"]) != int(season):
            return False, "season_mismatch"
        return True, None
    try:
        if session.get("account_id") or session.get("viewer_user_id") or session.get("viewer_roster_id"):
            return True, None
        # Same-origin Draft Room (guest boards) — browser sends Sec-Fetch-Site.
        if (request.headers.get("Sec-Fetch-Site") or "").lower() == "same-origin":
            return True, None
    except RuntimeError:
        pass
    return False, "auth_required"



@draft_api_bp.route("/api/draft/detect")
def api_draft_detect():
    """List drafts for a league so the user can connect a live draft (Sleeper)."""
    platform = (request.args.get("platform") or "sleeper").strip().lower()
    league_id = (request.args.get("league_id") or "").strip()
    season = int(request.args.get("season") or datetime.now().year)
    if not league_id:
        return jsonify({"drafts": [], "error": "league_required"})
    # ESPN: the dashboard countdown card only needs start_time + status and must
    # not hit mDraftDetail on every tick. Draft Room / cheat-sheet live connect
    # pass sync=1 to fetch the companion snapshot (inProgress, rounds, teams).
    if platform == "espn":
        want_sync = (request.args.get("sync") or "").strip().lower() in ("1", "true", "yes")
        if want_sync:
            return _espn_detect_sync(league_id, season)
        try:
            _drafts = get_drafts("espn", league_id, season) or []
        except Exception as exc:
            logger.warning("[draft-detect] espn error_type=%s", type(exc).__name__)
            _drafts = []
        return jsonify({"drafts": [{
            "draft_id": d.get("draft_id"),
            "status": d.get("status"),
            "type": d.get("type"),
            "season": d.get("season"),
            "start_time": d.get("start_time"),
        } for d in _drafts]})
    if platform == "fleaflicker":
        try:
            _drafts = get_drafts("fleaflicker", league_id, season) or []
        except Exception as exc:
            logger.warning("[draft-detect] fleaflicker error_type=%s", type(exc).__name__)
            _drafts = []
        return jsonify({"drafts": [{
            "draft_id": d.get("draft_id"),
            "status": d.get("status"),
            "type": d.get("type"),
            "season": d.get("season"),
            "start_time": d.get("start_time"),
        } for d in _drafts]})
    if platform != "sleeper":
        return jsonify({"drafts": [], "unsupported": True})
    # The draft-history page wants every season's draft; the dashboard countdown
    # card only needs this league's, so it stays a single call unless history=1.
    want_history = (request.args.get("history") or "").strip() in ("1", "true", "yes")
    out = []
    seen_ids: set = set()
    try:
        # A Sleeper league chains across seasons via previous_league_id, so walk
        # the whole history and collect every season's drafts. Newest league
        # first so the current season leads.
        hist = {}
        if want_history:
            try:
                hist = build_league_history_map(platform, league_id, season) or {}
            except Exception:
                hist = {}
        league_ids = list(dict.fromkeys(
            [league_id] + [lid for _yr, lid in sorted(hist.items(), reverse=True)]))
        for lid in league_ids:
            try:
                drafts = get_drafts(platform, lid, season) or []
            except Exception:
                continue
            for d in drafts:
                did = str(d.get("draft_id") or "")
                if not did or did in seen_ids:
                    continue
                seen_ids.add(did)
                settings = d.get("settings") or {}
                rounds_val = int(settings.get("rounds") or 15)
                draft_type = _live_draft_type(
                    rounds_val, platform, d.get("league_id") or lid,
                    int(d.get("season") or season), cache_key=did,
                )
                out.append({
                    "draft_id": did,
                    "status": d.get("status"),
                    "type": d.get("type"),
                    "draft_type": draft_type,
                    "season": d.get("season"),
                    "start_time": d.get("start_time"),   # scheduled start (epoch ms) for the imminent-draft banner
                    "teams": settings.get("teams"),
                    "rounds": settings.get("rounds"),
                    "order": _order_from_sleeper(d),
                })
    except Exception as exc:
        logger.warning("[draft-detect] error: %s", exc)
    return jsonify({"drafts": out})


@draft_api_bp.route("/api/draft/espn-relay", methods=["POST", "OPTIONS", "GET"])
def api_draft_espn_relay():
    """Normalize + store ESPN draft picks from the extension or mobile bookmarklet.

    Observe-only: does not talk to ESPN and never submits picks.
    """
    if request.method == "OPTIONS":
        return _relay_cors(jsonify({"ok": True}))

    from dashboard_services.espn_draft_relay import (
        get_relay_snapshot,
        put_relay_snapshot,
    )

    if request.method == "GET":
        league_id = (request.args.get("league_id") or request.args.get("leagueId") or "").strip()
        try:
            season = int(request.args.get("season") or request.args.get("seasonId") or 0)
        except (TypeError, ValueError):
            season = 0
        if not league_id or not season:
            return _relay_cors(jsonify({"error": "league_required"})), 400
        ok, err = _relay_auth_ok(league_id, season)
        if not ok:
            return _relay_cors(jsonify({"error": err or "auth_required"})), 401
        entry = get_relay_snapshot(league_id, season)
        if not entry:
            return _relay_cors(jsonify({"picks": [], "empty": True, "league_id": league_id, "season": season}))
        payload = dict(entry.get("payload") or {})
        payload["relay_updated_at"] = entry.get("updated_at")
        payload["relay_source"] = entry.get("source")
        for secret_key in ("espn_s2", "swid", "SWID", "cookie", "cookies", "token"):
            payload.pop(secret_key, None)
        return _relay_cors(jsonify(payload))

    body = request.get_json(silent=True) or {}
    if not isinstance(body, dict):
        return _relay_cors(jsonify({"error": "invalid_body"})), 400
    picks = body.get("picks")
    if not isinstance(picks, list):
        return _relay_cors(jsonify({"error": "picks_required"})), 400
    if len(picks) > 500:
        return _relay_cors(jsonify({"error": "too_many_picks"})), 400

    league_id = str(body.get("leagueId") or body.get("league_id") or "").strip()
    try:
        season = int(body.get("season") or body.get("seasonId") or 0)
    except (TypeError, ValueError):
        season = 0
    # Token may carry league/season when the bookmarklet URL lacked query params.
    token = _bearer_token()
    if token and (not league_id or not season):
        from dashboard_services.espn_draft_relay import verify_relay_token
        claims = verify_relay_token(token)
        if claims:
            league_id = league_id or str(claims["league_id"])
            season = season or int(claims["season"])
    if not league_id or not season:
        return _relay_cors(jsonify({"error": "league_required"})), 400

    ok, err = _relay_auth_ok(league_id, season)
    if not ok:
        return _relay_cors(jsonify({"error": err or "auth_required"})), 401

    try:
        payload = _espn_relay_normalize(body)
    except Exception:
        logger.warning("[draft-espn-relay] normalize failed error_type=Exception")
        return _relay_cors(jsonify({"error": "normalize_failed"})), 500

    payload["league_id"] = league_id
    payload["season"] = season
    source = str(body.get("source") or "relay")
    try:
        put_relay_snapshot(league_id, season, payload, source=source)
    except Exception:
        logger.info("[draft-espn-relay] store skipped error_type=Exception")

    for secret_key in ("espn_s2", "swid", "SWID", "cookie", "cookies", "token"):
        payload.pop(secret_key, None)
    return _relay_cors(jsonify(payload))


@draft_api_bp.route("/api/draft/espn-relay/token", methods=["POST"])
def api_draft_espn_relay_token():
    """Mint a short-lived token + bookmarklet for mobile ESPN draft sync."""
    from dashboard_services.espn_draft_relay import (
        build_bookmarklet,
        mint_relay_token,
        shortcut_javascript,
        site_origin,
    )

    body = request.get_json(silent=True) or {}
    league_id = str(
        body.get("league_id") or body.get("leagueId") or request.args.get("league_id") or ""
    ).strip()
    try:
        season = int(body.get("season") or body.get("seasonId") or request.args.get("season") or 0)
    except (TypeError, ValueError):
        season = 0
    if not league_id or not season:
        return jsonify({"error": "league_required"}), 400
    try:
        account_id = session.get("account_id")
    except RuntimeError:
        account_id = None
    try:
        minted = mint_relay_token(
            league_id=league_id,
            season=season,
            account_id=str(account_id) if account_id not in (None, "") else None,
        )
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    origin = site_origin(request.host_url)
    bookmarklet = build_bookmarklet(
        origin, minted["token"], league_id=league_id, season=season
    )
    return jsonify({
        **minted,
        "origin": origin,
        "bookmarklet": bookmarklet,
        "shortcut_js": shortcut_javascript(
            origin, minted["token"], league_id=league_id, season=season
        ),
        "espn_draft_url": (
            f"https://fantasy.espn.com/football/draft?leagueId={league_id}&seasonId={season}"
        ),
        "instructions": {
            "android": "Open ESPN draft → browser menu → Bookmarks → Add → edit URL to the bookmarklet.",
            "ios": (
                "Safari only. Live ESPN drafts often cannot be read by iOS Shortcuts "
                "(no React state / empty mDraftDetail). Prefer laptop + Chrome extension "
                "or manual tracking. If trying Shortcuts: Share Sheet → Safari web pages → "
                "Shortcut Input → paste shortcut_js."
            ),
            "desktop": "Drag the bookmarklet to your bookmarks bar, open the ESPN draft, click it after each pick (or when picks look behind).",
        },
    })


@draft_api_bp.route("/api/draft/live")
def api_draft_live():
    """Current state + picks for a live draft (polled by the live board).

    Sleeper is keyed by ``draft_id``. ESPN is keyed by ``espn_{league_id}_{season}``
    (or ``league_id`` + ``season``) and is observe-only.
    """
    platform = (request.args.get("platform") or "sleeper").strip().lower()
    draft_id = (request.args.get("draft_id") or "").strip()
    if platform == "espn":
        league_id = (request.args.get("league_id") or "").strip()
        try:
            season = int(request.args.get("season") or 0)
        except (TypeError, ValueError):
            season = 0
        return _espn_live(draft_id, league_id, season)
    if not draft_id or platform != "sleeper":
        return jsonify({"error": "unsupported"}), 400
    try:
        from dashboard_services.api import get_draft as _gd, get_draft_picks as _gdp
        draft = _gd(draft_id) or {}
        picks_raw = _gdp(draft_id) or []
    except Exception as exc:
        logger.warning("[draft-live] error: %s", exc)
        return jsonify({"error": "fetch_failed"}), 502
    settings = draft.get("settings") or {}
    picks = []
    for p in picks_raw:
        pid = str(p.get("player_id") or "").strip()
        if not pid or pid in ("0", "None"):
            # Predraft slot placeholders are not selections; skip so the board
            # stays empty until a player is actually picked.
            continue
        meta = p.get("metadata") or {}
        nm = (str(meta.get("first_name") or "") + " " + str(meta.get("last_name") or "")).strip()
        picks.append({
            "pick_no": p.get("pick_no"),
            "round": p.get("round"),
            "draft_slot": p.get("draft_slot"),
            "picked_by": p.get("picked_by"),   # user_id of who made the pick (for ownership)
            "player_id": pid,
            "name": nm or (meta.get("player_name") or "Unknown"),
            "position": (meta.get("position") or "").upper(),
            "team": meta.get("team") or "",
            "picked_at": p.get("picked_at"),   # ms epoch when Sleeper recorded the pick
        })

    rounds_val = int(settings.get("rounds") or 15)
    # Rookie drafts are short (1-5 rounds); a long draft is a dynasty startup only
    # for true dynasty leagues. Redraft and keeper leagues run a full draft each
    # year, so they resolve to 'redraft' (redraft ADP + values), not dynasty.
    draft_type = _live_draft_type(
        rounds_val, platform, draft.get("league_id"),
        int(draft.get("season") or datetime.now().year), cache_key=draft_id,
    )
    pick_timer = int(settings.get("pick_timer") or 0)

    # Light poll: the board hits this every few seconds and only needs the things
    # that actually change - status + picks. Skip the 3 extra league API calls
    # (users, traded picks, rosters) that drive slot names / future-pick ownership;
    # those barely change and the board fetches them on connect + a periodic full
    # refresh. This keeps the hot poll path to ~2 upstream calls.
    roster_positions = settings.get("roster_positions") or []
    if request.args.get("light"):
        return jsonify({
            "status": draft.get("status"),
            "type": draft.get("type"),
            "draft_type": draft_type,
            "season": draft.get("season"),   # draft's season, so a historical view grades vs that year's ADP
            "pick_timer": pick_timer,
            "start_time": draft.get("start_time"),   # scheduled start (epoch ms), pre-draft countdown
            "teams": settings.get("teams"),
            "rounds": settings.get("rounds"),
            "order": _order_from_sleeper(draft),
            "draft_order": draft.get("draft_order") or {},
            "roster_positions": roster_positions,
            "picks": picks,
        })

    # Map draft slot -> team/owner display name (so the board shows real names).
    slot_names = {}
    draft_order = draft.get("draft_order") or {}     # {user_id: slot}
    traded_picks_out = []
    user_roster_map = {}  # user_id -> roster_id (for traded-pick ownership resolution)
    try:
        _lid = draft.get("league_id")
        _season = int(draft.get("season") or datetime.now().year)
        if _lid and draft_order:
            _users = get_users(platform, _lid, _season) or []
            _by_uid = {str(u.get("user_id")): u for u in _users}
            for _uid, _slot in draft_order.items():
                _u = _by_uid.get(str(_uid)) or {}
                _name = ((_u.get("metadata") or {}).get("team_name")
                         or _u.get("display_name") or ("Team " + str(_slot)))
                slot_names[str(_slot)] = _name
        if _lid:
            # Traded picks - filter to this draft's season so the frontend can
            # resolve future pick ownership correctly (trades change who owns what).
            _all_traded = get_traded_picks(platform, _lid, _season) or []
            _season_str = str(_season)
            traded_picks_out = [
                {"season": tp.get("season"), "round": tp.get("round"),
                 "roster_id": tp.get("roster_id"),   # original owner's roster_id
                 "owner_id": tp.get("owner_id")}      # current owner's roster_id
                for tp in _all_traded
                if str(tp.get("season") or "") == _season_str
            ]
            # Build user_id -> roster_id map so the frontend can identify the
            # viewer's roster_id and match it against traded_picks.owner_id.
            _rosters = get_rosters(platform, _lid, _season) or []
            for _r in _rosters:
                _uid = str(_r.get("owner_id") or "")
                _rid = _r.get("roster_id")
                if _uid and _rid is not None:
                    user_roster_map[_uid] = _rid
    except Exception as _e_sn:
        logger.info("[draft-live] slot names / traded picks skipped: %s", _e_sn)

    return jsonify({
        "status": draft.get("status"),
        "type": draft.get("type"),
        "draft_type": draft_type,
        "season": draft.get("season"),   # draft's season, so a historical view grades vs that year's ADP
        "pick_timer": pick_timer,
        "start_time": draft.get("start_time"),   # scheduled start (epoch ms), pre-draft countdown
        "teams": settings.get("teams"),
        "rounds": settings.get("rounds"),
        "order": _order_from_sleeper(draft),
        "draft_order": draft_order,
        "slot_names": slot_names,
        "roster_positions": roster_positions,
        "picks": picks,
        "traded_picks": traded_picks_out,
        "user_roster_map": user_roster_map,
    })


# Draft "roster slot" names -> the slot strings simulate_playoff_odds expects.
_DRAFT_SLOT_TO_ENGINE = {"SF": "SUPER_FLEX"}


def _draft_roster_positions(roster: dict) -> list:
    """Flatten the draft's roster-slot counts into the ordered position list the
    playoff simulator reads (starting slots first, bench last). SF -> SUPER_FLEX
    so the engine treats it as a QB-eligible flex."""
    out: list = []
    for slot in ("QB", "SF", "RB", "WR", "TE", "FLEX", "K", "DEF"):
        n = int(roster.get(slot) or 0)
        eng = _DRAFT_SLOT_TO_ENGINE.get(slot, slot)
        out.extend([eng] * n)
    out.extend(["BN"] * int(roster.get("BN") or 0))
    return out


@draft_api_bp.route("/api/draft-playoff-odds", methods=["POST"])
def api_draft_playoff_odds():
    """Projected playoff odds for a COMPLETED draft's teams.

    Runs the same Monte Carlo the standings page uses (``simulate_playoff_odds``
    in its preseason mode: project each roster's strength from Sleeper/FP player
    projections, simulate a full season with skew-normal weekly scoring over a
    balanced round-robin schedule). The draft room posts the drafted rosters and
    renders the returned ``playoff_pct`` per team; it falls back to a light
    client-side estimate if this call fails.
    """
    data = request.get_json(silent=True) or {}
    teams_in = data.get("teams") or []
    if not isinstance(teams_in, list) or len(teams_in) < 2:
        return jsonify({"error": "need_two_teams"}), 400
    if len(teams_in) > 32:
        return jsonify({"error": "too_many_teams"}), 400

    roster = data.get("roster") or {}
    season = int(data.get("season") or 0) or datetime.now().year
    try:
        rec_pts = float(data.get("ppr"))
    except (TypeError, ValueError):
        rec_pts = 1.0

    roster_positions = _draft_roster_positions(roster)
    rosters = []
    roster_map = {}
    for t in teams_in:
        try:
            rid = int(t.get("slot"))
        except (TypeError, ValueError):
            continue
        pids = [str(x) for x in (t.get("players") or []) if x not in (None, "")]
        rosters.append({"roster_id": rid, "players": pids})
        roster_map[str(rid)] = str(t.get("name") or ("Team " + str(rid)))
    if len(rosters) < 2:
        return jsonify({"error": "need_two_teams"}), 400

    try:
        playoff_teams = int(data.get("playoff_teams") or 0)
    except (TypeError, ValueError):
        playoff_teams = 0
    if playoff_teams <= 0:
        playoff_teams = 4 if len(rosters) <= 8 else 6
    playoff_teams = max(1, min(playoff_teams, len(rosters) - 1))

    ctx = {
        "season": season,
        "current_week": 0,          # preseason path (no games yet)
        "league_id": "",            # no real league -> round-robin fallback schedule
        "scoring_settings": {"rec": rec_pts},
        "raw_scoring_settings": {},
        "roster_positions": roster_positions,
        "rosters": rosters,
        "roster_map": roster_map,
        "team_stats": None,
        "league_settings": {
            "playoff_teams": playoff_teams,
            "playoff_week_start": 15,
            "divisions": 0,
        },
    }
    try:
        from data_building.simulate_playoff_odds import simulate_playoff_odds
        # Deterministic per-board seed so odds don't drift on re-open for the same
        # rosters. n_sims trimmed from the 10k default for snappier response.
        res = simulate_playoff_odds(ctx, platform="sleeper", n_sims=5000, seed=1234)
    except Exception as exc:
        logger.warning("[draft-playoff-odds] sim failed: %s", exc)
        return jsonify({"error": "sim_failed"}), 502

    odds = [
        {
            "slot": r.get("roster_id"),
            "playoff_pct": round(float(r.get("playoff_pct") or 0), 1),
            "bye_pct": round(float(r.get("bye_pct") or 0), 1),
            "first_seed_pct": round(float(r.get("first_seed_pct") or 0), 1),
            "avg_final_wins": round(float(r.get("avg_final_wins") or 0), 1),
            "avg_final_losses": round(float(r.get("avg_final_losses") or 0), 1),
        }
        for r in (res or [])
    ]
    return jsonify({"odds": odds, "playoff_teams": playoff_teams})
