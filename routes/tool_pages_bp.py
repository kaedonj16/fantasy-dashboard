"""
Standalone league tool pages.

Routes:
    /draft[, /<platform>/<season>/<league_id>/draft]           (Draft Room)
    /keeper[, /<platform>/<season>/<league_id>/keeper]         (Keeper Assistant)
    /draft/history[, /<platform>/<season>/<league_id>/draft/history]

Extracted from app.py to reduce monolith size. Each renders via app.render_page
and delegates its body to a dashboard_services.pages.* builder.

App.py internals (render_page, get_league_ctx_from_cache, _nav_show_keeper) are
reached through the lazy shims below rather than a top-level ``from app import``
so importing this module during app start-up does not trigger a circular import
— the real functions are only fetched when a request is actually served.
"""
from __future__ import annotations

import logging
from datetime import datetime

from flask import Blueprint, request, session

logger = logging.getLogger(__name__)

tool_pages_bp = Blueprint("tool_pages", __name__)


# ── Lazy shims to app.py internals (resolved at request time) ─────────────────

def render_page(*args, **kwargs):
    from app import render_page as _fn
    return _fn(*args, **kwargs)


def get_league_ctx_from_cache(*args, **kwargs):
    from app import get_league_ctx_from_cache as _fn
    return _fn(*args, **kwargs)


def _nav_show_keeper(*args, **kwargs):
    from app import _nav_show_keeper as _fn
    return _fn(*args, **kwargs)


@tool_pages_bp.route("/draft")
@tool_pages_bp.route("/<platform>/<int:season>/<league_id>/draft")
def page_draft_room(platform: str = None, season: int = None, league_id: str = None):
    """Standalone Draft Room / draft board."""
    from dashboard_services.pages.draft_room_page import build_draft_room_body
    is_guest = not league_id
    num_teams = None
    is_sf = False
    roster_positions = None
    if league_id:
        try:
            ctx = get_league_ctx_from_cache(platform, league_id, season)
            num_teams = ctx.get("total_rosters") or None
            _rp = ctx.get("roster_positions") or []
            if hasattr(_rp, "tolist"):
                _rp = _rp.tolist()
            roster_positions = [str(s) for s in _rp] if _rp else None
            is_sf = any(str(s).upper() in {"SUPER_FLEX", "SFLEX"} for s in _rp)
            league_id = ctx.get("league_id") or league_id
            season = int(ctx.get("season") or season or datetime.now().year)
        except Exception as _e:
            logger.info("[draft-room] league ctx load failed: %s", _e)
    num_rounds_rookie = None
    num_rounds_startup = None
    if league_id and roster_positions:
        try:
            _ls = get_league_ctx_from_cache(platform, league_id, season).get("league_settings") or {}
            _rr = int(_ls.get("draft_rounds") or 0)
            if _rr:
                num_rounds_rookie = _rr
            _draftable = [p for p in roster_positions if str(p).upper() not in ("TAXI", "IR")]
            if _draftable:
                num_rounds_startup = len(_draftable)
        except Exception:
            logger.debug("suppressed exception", exc_info=True)
    # Hide the Keeper draft type + keeper options + the keeper banner entirely
    # for dynasty and plain redraft (non-keeper) leagues. A dynasty league can
    # still carry a max_keepers value, which is why this gates on _nav_show_keeper
    # (type 2 is never a keeper league) rather than the raw keeper limit. Guests
    # (no league) keep it available.
    show_keeper = True
    if league_id:
        try:
            show_keeper = _nav_show_keeper(platform, league_id, season)
        except Exception:
            show_keeper = True
    # League keepers: surface them in the draft room either when the league is a
    # real keeper league, or when the user explicitly came from the keeper tool
    # (?keepers=1). Never for dynasty / plain redraft leagues, so the board is
    # unchanged there.
    keepers_payload = None
    if league_id and show_keeper:
        try:
            from dashboard_services.pages.keeper_page import compute_league_keepers, league_keeper_limit
            _ctx = get_league_ctx_from_cache(platform, league_id, season)
            if request.args.get("keepers") or league_keeper_limit(_ctx) > 0:
                # The keeper tool hands off the limit and cost rules the user is
                # playing by so rival projections use the same ones, instead of
                # the server defaults (which price every undrafted player at the
                # last round).
                def _karg(name):
                    try:
                        raw = request.args.get(name)
                        return int(raw) if raw not in (None, "") else None
                    except (TypeError, ValueError):
                        return None
                _klimit = _karg("klimit") or None
                _krules = {k: v for k, v in (
                    ("undrafted_round", _karg("kundr")),
                    ("round_offset",    _karg("koff")),
                    ("escalation",      _karg("kesc")),
                ) if v is not None}
                # One-pick-per-round flag (kopr=0/1); only apply when explicitly
                # sent so rival projections match the rule the user is playing by.
                _kopr = request.args.get("kopr")
                if _kopr in ("0", "1"):
                    _krules["one_per_round"] = (_kopr == "1")
                keepers_payload = compute_league_keepers(
                    _ctx, platform=platform, league_id=league_id,
                    viewer_roster_id=session.get("viewer_roster_id"),
                    limit_override=_klimit, rules_override=_krules or None,
                )
        except Exception:
            logger.debug("[draft-room] keeper compute skipped", exc_info=True)
    body = build_draft_room_body(
        league_id, season, platform,
        is_guest=is_guest, num_teams=num_teams, is_superflex=is_sf,
        roster_positions=roster_positions,
        viewer_user_id=session.get("viewer_user_id"),
        num_rounds_rookie=num_rounds_rookie,
        num_rounds_startup=num_rounds_startup,
        keepers=keepers_payload,
        show_keeper=show_keeper,
    )
    return render_page(
        "Draft Room | BR Fantasy", league_id, "draft", body, platform, season,
        description=(
            "Fantasy football draft assistant and draft board with best-available, "
            "ADP, and snake / linear / third-round-reversal support for Sleeper, ESPN, and Yahoo."
        ),
    )


@tool_pages_bp.route("/keeper")
@tool_pages_bp.route("/<platform>/<int:season>/<league_id>/keeper")
def page_keeper(platform: str = None, season: int = None, league_id: str = None):
    """Keeper Assistant: decide who to keep next season."""
    from dashboard_services.pages.keeper_page import build_keeper_body
    if not league_id:
        body = (
            "<div class='card central' style='text-align:center;padding:44px 16px;'>"
            "<h2>Keeper Assistant</h2>"
            "<p style='color:var(--text-muted);max-width:54ch;margin:10px auto 0;line-height:1.6;'>"
            "Open one of your leagues to see keeper recommendations: who returns the most "
            "draft-capital value at their keeper cost, and the optimal set to keep under your "
            "league limit.</p></div>"
        )
        return render_page(
            "Keeper Assistant | BR Fantasy", None, "keeper", body,
            description="Fantasy football keeper tool: decide who to keep with surplus-value scoring and an optimal-set optimizer.",
        )
    ctx = {}
    try:
        ctx = get_league_ctx_from_cache(platform, league_id, season)
        league_id = ctx.get("league_id") or league_id
        season = int(ctx.get("season") or season or datetime.now().year)
    except Exception as _e:
        logger.info("[keeper] league ctx load failed: %s", _e)
    viewer_roster_id = session.get("viewer_roster_id") or None
    from dashboard_services.adp_service import ADP_SOURCE_LABELS as _ADP_LABELS
    _kadp_src = (request.args.get("adp_source") or "consensus").strip().lower()
    if _kadp_src not in _ADP_LABELS:
        _kadp_src = "consensus"
    body = build_keeper_body(
        ctx or {}, viewer_roster_id=viewer_roster_id,
        platform=(platform or "sleeper"), league_id=league_id,
        adp_source=_kadp_src, season=season,
        force=bool(request.args.get("show")),
    )
    return render_page(
        "Keeper Assistant | BR Fantasy", league_id, "keeper", body, platform, season,
        description=("Keeper league tool: rank your roster by keeper surplus and pick the optimal "
                     "set to keep under your league's limit, from redraft values and market ADP."),
    )


@tool_pages_bp.route("/draft/history")
@tool_pages_bp.route("/<platform>/<int:season>/<league_id>/draft/history")
def page_draft_history(platform: str = None, season: int = None, league_id: str = None):
    """Draft history: the league's real drafts (from Sleeper), openable to review."""
    from dashboard_services.pages.draft_room_page import build_draft_history_body
    body = build_draft_history_body(league_id, season, platform)
    return render_page(
        "Draft History | BR Fantasy", league_id, "draft", body, platform, season,
        description="Review your league's past and live fantasy football drafts pick-by-pick.",
    )
