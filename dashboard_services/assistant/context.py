"""Account and league context resolution for the assistant.

Account key matches _user_key() in routes/user_pages_bp.py: "acct:<account_id>"
from the Flask session, None for guests. League context is reached through
the same lazy-shim pattern as routes/waiver_bp.py (import from app at call
time, never at module import).
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


# -- lazy shims to app.py internals (resolved at call time) -------------------

def get_league_ctx_from_cache(*args, **kwargs):
    from app import get_league_ctx_from_cache as _fn
    return _fn(*args, **kwargs)


def get_viewer_session_for_league(*args, **kwargs):
    from app import get_viewer_session_for_league as _fn
    return _fn(*args, **kwargs)


def get_players_index_global(*args, **kwargs):
    from app import get_players_index_global as _fn
    return _fn(*args, **kwargs)


def get_model_value_table_cached(*args, **kwargs):
    from app import get_model_value_table_cached as _fn
    return _fn(*args, **kwargs)


def resolve_account_key():
    try:
        from flask import session
        account_id = session.get("account_id")
    except Exception:
        return None
    if account_id not in (None, ""):
        return "acct:" + str(account_id).strip()
    return None


def resolve_account_id():
    key = resolve_account_key()
    if not key:
        return None
    try:
        return int(key.split(":", 1)[1])
    except Exception:
        return None


def current_week(ctx: dict | None = None) -> int | None:
    try:
        from dashboard_services.api import get_nfl_state
        state = get_nfl_state() or {}
        wk = state.get("week") or state.get("display_week")
        if wk:
            return int(wk)
    except Exception:
        pass
    if ctx:
        for k in ("week", "current_week", "scoring_week"):
            if ctx.get(k):
                try:
                    return int(ctx[k])
                except Exception:
                    pass
        league = ctx.get("league") or {}
        if league.get("week"):
            try:
                return int(league["week"])
            except Exception:
                pass
    return None


def _viewer_roster(ctx: dict) -> tuple[dict | None, dict]:
    """Find the viewer's roster inside a league ctx.

    ctx["viewer"] is populated by get_league_ctx_from_cache via
    get_viewer_session_for_league; fall back to calling that shim directly,
    then to matching rosters by roster_id / owner_id fields present.
    """
    viewer = ctx.get("viewer") if isinstance(ctx.get("viewer"), dict) else {}
    if not viewer:
        try:
            viewer = get_viewer_session_for_league(
                ctx.get("users") or [], ctx.get("rosters") or [],
                ctx.get("platform"), ctx.get("league_id") or ((ctx.get("league") or {}).get("league_id")),
                ctx.get("season"),
            ) or {}
        except Exception:
            viewer = {}
    rosters = ctx.get("rosters") or []
    rid = viewer.get("roster_id") or viewer.get("viewer_roster_id")
    owner = viewer.get("user_id") or viewer.get("viewer_user_id") or viewer.get("owner_id")
    roster = None
    for r in rosters:
        if not isinstance(r, dict):
            continue
        if rid is not None and str(r.get("roster_id")) == str(rid):
            roster = r
            break
        if owner and str(r.get("owner_id") or "") == str(owner):
            roster = r
            break
    if roster is None and rid is not None:
        roster = {"roster_id": rid, "players": viewer.get("players") or [], "starters": viewer.get("starters") or []}
    return roster, viewer


def _session_current_league(leagues: list) -> dict | None:
    """The league the user is currently in, per the Flask session.

    render_page records session["last_league_id"] (plus platform/season)
    on every page render, and league-less pages (My Leagues, home) render
    their nav chrome from that same value, so it is the league the header
    pill shows. It only counts when it names a league the account owns.
    Outside a request context (background callers) there is no current
    league and this returns None.
    """
    try:
        from flask import has_request_context, session
        if not has_request_context():
            return None
        lid = session.get("last_league_id")
        plat = str(session.get("last_platform") or "").lower()
        seas = session.get("last_season")
    except Exception:
        return None
    if not lid:
        return None
    fallback = None
    for lg in leagues or []:
        if str(lg.get("league_id") or "") != str(lid):
            continue
        if plat and str(lg.get("platform") or "").lower() != plat:
            continue
        if seas not in (None, "") and lg.get("season") not in (None, "") \
                and str(lg.get("season")) != str(seas):
            if fallback is None:
                fallback = lg
            continue
        return lg
    return fallback


def resolve_league(account_key: str | None, *, platform=None, league_id=None, season=None) -> dict:
    """Return a league bundle dict (never raises).

    Keys: platform, league_id, season, league_name, team_name, week, ctx,
    roster, viewer, roster_player_ids, assumed_league_uncertain.
    """
    out = {
        "platform": (platform or "sleeper"), "league_id": league_id, "season": season,
        "league_name": None, "team_name": None, "week": None, "ctx": None,
        "roster": None, "viewer": {}, "roster_player_ids": set(),
        "assumed_league_uncertain": False,
    }
    explicit = bool(league_id)
    chosen = None
    if not explicit and account_key:
        account_id = resolve_account_id()
        if account_id:
            try:
                from dashboard_services.accounts import list_user_leagues
                leagues = list_user_leagues(account_id) or []
            except Exception:
                leagues = []
            if leagues:
                # The user's current league (the one the header pill
                # shows) wins over the favorite-first default: picking
                # leagues[0] here put the assistant in a different
                # league than the page the user was looking at. The
                # current league is not an assumption, so it does not
                # raise the uncertain flag.
                current = _session_current_league(leagues)
                if current is not None:
                    chosen = current
                    out["assumed_league_uncertain"] = False
                else:
                    chosen = leagues[0]  # favorite-first ordering in list_user_leagues
                    out["assumed_league_uncertain"] = len(leagues) > 1
                out["platform"] = chosen.get("platform") or out["platform"]
                out["league_id"] = chosen.get("league_id")
                if season is None:
                    out["season"] = chosen.get("season")
                out["league_name"] = chosen.get("name")
    if not out["league_id"]:
        out["week"] = current_week(None)
        return out
    if out["season"] is None:
        try:
            from dashboard_services.api import get_nfl_state
            state = get_nfl_state() or {}
            out["season"] = int(state.get("season") or 0) or None
        except Exception:
            out["season"] = None
    ctx = None
    try:
        ctx = get_league_ctx_from_cache(out["platform"], str(out["league_id"]), out["season"]) or {}
    except Exception:
        logger.debug("[assistant] league ctx lookup failed", exc_info=True)
        ctx = {}
    out["ctx"] = ctx or None
    if ctx:
        league = ctx.get("league") or {}
        out["league_name"] = out["league_name"] or league.get("name") or ctx.get("league_name")
        if out["season"] is None:
            out["season"] = ctx.get("season") or league.get("season")
        roster, viewer = _viewer_roster(ctx)
        out["roster"] = roster
        out["viewer"] = viewer or {}
        out["team_name"] = (
            viewer.get("team_name") or viewer.get("viewer_team_name")
            or (roster or {}).get("team_name")
        )
        if roster:
            out["roster_player_ids"] = {str(p) for p in (roster.get("players") or [])}
    out["week"] = current_week(ctx)
    return out
