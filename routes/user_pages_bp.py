"""
User account hub pages.

Routes:
    /portfolio   (My Leagues cross-league hub)
    /watchlist   (saved-player watchlist)

Extracted from app.py to reduce monolith size. Both are signed-in user pages
that render via app.render_page.

App.py internals used by the handlers are resolved through the lazy shims below
rather than a top-level from app import ... so importing this module during
app start-up does not trigger a circular import -- the real functions are only
fetched when a request is actually served.
"""
from __future__ import annotations

import copy
import html
import logging
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

from flask import Blueprint, g, jsonify, redirect, request, session, url_for

from dashboard_services.portfolio_summary import get_cached_summary

import json
from extensions import limiter

logger = logging.getLogger(__name__)

user_pages_bp = Blueprint("user_pages", __name__)


def portfolio_signed_in_label(sess=None) -> str:
    """Who My Leagues should say you're signed in as.

    ``viewer_username`` is league-scoped: opening an ESPN / Fleaflicker / Yahoo
    league overwrites it with that team's owner or team name. Prefer the Google
    account label (same as the home-page greeting) so the hub doesn't claim
    you're a different manager.
    """
    data = sess if sess is not None else session
    for key in ("account_first_name", "account_email", "viewer_username"):
        value = str(data.get(key) or "").strip()
        if value:
            return value
    return "your account"


def sleeper_owner_id_for_account(account_id, viewer_user_id, platform="sleeper"):
    """Session viewer_user_id is only a Sleeper owner when linked to this account.

    Visiting another platform overwrites session ``viewer_user_id`` with that
    league's owner id. Using it as a Sleeper user id would attach the wrong
    roster (and the wrong Rank / Rec on the card).
    """
    if str(platform or "sleeper").lower() != "sleeper":
        return None
    uid = str(viewer_user_id or "").strip()
    if not uid:
        return None
    if not account_id:
        return uid
    from dashboard_services.accounts import list_account_platform_ids
    linked = {str(x) for x in list_account_platform_ids(account_id, "sleeper")}
    return uid if uid in linked else None


# ── Lazy shims to app.py internals (resolved at request time) ─────────────────

def render_page(*args, **kwargs):
    from app import render_page as _fn
    return _fn(*args, **kwargs)

def get_league_ctx_from_cache(*args, **kwargs):
    from app import get_league_ctx_from_cache as _fn
    return _fn(*args, **kwargs)

def ensure_weekly_bits(*args, **kwargs):
    from app import ensure_weekly_bits as _fn
    return _fn(*args, **kwargs)

def get_model_value_table_cached(*args, **kwargs):
    from app import get_model_value_table_cached as _fn
    return _fn(*args, **kwargs)

def get_nfl_state(*args, **kwargs):
    from app import get_nfl_state as _fn
    return _fn(*args, **kwargs)

def count_roster_positions(*args, **kwargs):
    from app import count_roster_positions as _fn
    return _fn(*args, **kwargs)


def build_portfolio_body(*args, **kwargs):
    from app import build_portfolio_body as _fn
    return _fn(*args, **kwargs)

def build_projections_by_week(*args, **kwargs):
    from app import build_projections_by_week as _fn
    return _fn(*args, **kwargs)

def _games_scheduled_today(*args, **kwargs):
    from app import _games_scheduled_today as _fn
    return _fn(*args, **kwargs)

def _team_bye_map(*args, **kwargs):
    from app import _team_bye_map as _fn
    return _fn(*args, **kwargs)


# ── Portfolio summary stale-while-revalidate ─────────────────────────────────
# /portfolio used to recompute every league's summary inline on each page load
# (value lookups, positional rankings, streaks). On NFL game days that pinned
# workers and the page "took forever to load". Summaries are now served from
# the Postgres portfolio_summary_cache (populated by /api/portfolio/summary
# and /api/portfolio/card via build_league_summary). A fresh entry renders
# as-is; a STALE entry also renders (marked stale) while a bounded background
# refresh revalidates it -- serving last-good is the point of SWR, and the
# previous stale->shell behavior blanked every card at once each time the
# TTL expired, since all leagues' summaries are generated together. Only a
# true cache miss renders a hydratable shell.

_PORTFOLIO_SWR_GAME_DAY_TTL_S = 15 * 60
_PORTFOLIO_SWR_TTL_S = 60 * 60

_refresh_pool = None
_refresh_pool_lock = threading.Lock()

# Per-league single-flight for summary refresh kicks: a key is present from
# the moment a rebuild is submitted to the bounded pool until its build
# finishes, so page renders, card hydrations and explicit refreshes for the
# same league coalesce onto one build instead of stacking duplicates.
_SUMMARY_REFRESH_INFLIGHT: set = set()
_SUMMARY_REFRESH_INFLIGHT_LOCK = threading.Lock()


def _portfolio_refresh_pool():
    global _refresh_pool
    with _refresh_pool_lock:
        if _refresh_pool is None:
            # 2, not 3: each summary build walks a full league context, and
            # this pool runs alongside the request threads on a 2GB box
            # (the 2026-10-25/26 My Leagues OOMs).
            _refresh_pool = ThreadPoolExecutor(
                max_workers=2, thread_name_prefix="pf-refresh")
        return _refresh_pool


def _kick_summary_refresh(account_id, membership, _attempt=0):
    """Rebuild one league's cached portfolio summary without blocking the page.

    Fire-and-forget: failures are swallowed so a broken provider can never
    take down /portfolio. A cold league context kicks its own background warm
    inside get_league_ctx_from_cache and build_league_summary raises
    ContextPending; the refresh re-kicks itself on a timer until the warm
    lands (up to ~30s) instead of dropping the rebuild on the floor, which
    previously left the summary cache stale until a client happened to drive
    a successful /api/portfolio/card for that league.
    """
    lid = str((membership or {}).get("league_id") or "")
    _key = (str(account_id),
            str((membership or {}).get("platform") or "sleeper").lower(),
            lid, str((membership or {}).get("season") or ""))
    with _SUMMARY_REFRESH_INFLIGHT_LOCK:
        if _key in _SUMMARY_REFRESH_INFLIGHT:
            return
        _SUMMARY_REFRESH_INFLIGHT.add(_key)
    try:
        pool = _portfolio_refresh_pool()
    except Exception:
        with _SUMMARY_REFRESH_INFLIGHT_LOCK:
            _SUMMARY_REFRESH_INFLIGHT.discard(_key)
        logger.debug("portfolio refresh pool unavailable", exc_info=True)
        return

    def _run():
        try:
            from dashboard_services.portfolio_summary import (
                ContextPending, build_league_summary,
            )
            build_league_summary(
                account_id, dict(membership or {}),
                lambda p, i, s: get_league_ctx_from_cache(p, i, s, allow_build=False),
            )
        except ContextPending:
            if _attempt < 10:
                timer = threading.Timer(
                    3.0, _kick_summary_refresh,
                    args=(account_id, membership),
                    kwargs={"_attempt": _attempt + 1},
                )
                timer.daemon = True
                timer.start()
        except Exception:
            logger.debug("portfolio background refresh failed league=%s", lid,
                         exc_info=True)
        finally:
            # Cleared on every outcome (the #2820 lesson): a failed build
            # must never leak the mark and block later refreshes. A
            # ContextPending retry re-kicks after this clear, so it
            # re-acquires the mark when its timer fires.
            with _SUMMARY_REFRESH_INFLIGHT_LOCK:
                _SUMMARY_REFRESH_INFLIGHT.discard(_key)

    try:
        pool.submit(_run)
    except Exception:
        with _SUMMARY_REFRESH_INFLIGHT_LOCK:
            _SUMMARY_REFRESH_INFLIGHT.discard(_key)
        logger.debug("portfolio refresh submit failed league=%s", lid, exc_info=True)


def _cached_summary_fresh(cached, ttl_s):
    """True when the cached summary was generated within the TTL window."""
    if not isinstance(cached, dict):
        return False
    try:
        gen = datetime.fromisoformat(str(cached.get("generated_at") or ""))
    except (TypeError, ValueError):
        return False
    if gen.tzinfo is None:
        gen = gen.replace(tzinfo=timezone.utc)
    age_s = (datetime.now(timezone.utc) - gen).total_seconds()
    return 0 <= age_s <= ttl_s


def _portfolio_swr_ttl_s(season, week):
    """SWR freshness window for portfolio summaries: 15 min on game days."""
    try:
        game_day = bool(_games_scheduled_today(season, week))
    except Exception:
        logger.debug("portfolio game-day check failed", exc_info=True)
        game_day = False
    return _PORTFOLIO_SWR_GAME_DAY_TTL_S if game_day else _PORTFOLIO_SWR_TTL_S


@user_pages_bp.route("/portfolio")
def page_portfolio():
    viewer_username = session.get("viewer_username")
    viewer_user_id = session.get("viewer_user_id")
    # Allow account users through even without a Sleeper viewer identity: a user
    # who linked only ESPN/Yahoo leagues via a Google account has an account_id but
    # no viewer_username/viewer_user_id, and their leagues are merged in below from
    # the account. Without this they'd be bounced home from their own My Leagues.
    if (not viewer_username or not viewer_user_id) and not session.get("account_id"):
        return redirect(url_for("index"))
    # Use league nav context from query param, falling back to last visited league
    from_league = request.args.get("from_league", "").strip() or session.get("last_league_id") or None
    from_platform = request.args.get("platform", "").strip() or session.get("last_platform") or "sleeper"
    from_season_raw = request.args.get("season", "")
    try:
        from_season = int(from_season_raw) if from_season_raw else int(session.get("last_season") or 0) or None
    except ValueError:
        from_season = None
    nfl_state = get_nfl_state() or {}
    season = int(nfl_state.get("season") or datetime.now().year)
    # Current NFL week gates the portfolio "Last Week" stat: it is only
    # meaningful after week 1, so build_portfolio_body omits it until then.
    try:
        current_week = int(nfl_state.get("week") or 0)
    except (TypeError, ValueError):
        current_week = 0
    # Every league durably linked to the Google account, regardless of platform,
    # plus live Sleeper enrichment from linked identities. The shared builder also
    # backs /api/my-leagues so the portfolio and switcher never diverge.
    from dashboard_services.accounts import resolve_my_leagues
    # Read the account id off the request thread once. The per-league summaries
    # below run in a worker pool where Flask's request-local ``session`` is not
    # bound, so every session value they need is captured here first.
    account_id = session.get("account_id")
    import inspect
    _resolver_params = inspect.signature(resolve_my_leagues).parameters
    if "enrich_live" in _resolver_params:
        league_inputs, season = resolve_my_leagues(
            viewer_user_id, account_id, season, enrich_live=not bool(account_id)
        )
    else:  # compatibility for injected/legacy resolvers
        league_inputs, season = resolve_my_leagues(viewer_user_id, account_id, season)
    # Share this resolution with the insights/bye preambles (the shared
    # portfolio snapshot): the page just paid for it, so the endpoints
    # the page is about to fire must not pay it again.
    try:
        _pf_resolution_put(viewer_user_id, account_id,
                           int(nfl_state.get("season") or season),
                           league_inputs, season)
    except Exception:
        logger.debug("portfolio resolution share failed", exc_info=True)
    # Keep the durable membership reconciliation asynchronous, but build the
    # portfolio from league contexts below for account users too.  Returning a
    # membership-only shell here meant that only Record/Standing were ever
    # hydrated: the server never computed player holdings, team exposure, or
    # cross-league positional rankings, leaving the entire insights section
    # absent for every signed-in account.
    if account_id:
        from dashboard_services.accounts import schedule_account_league_reconciliation
        schedule_account_league_reconciliation(account_id, season)
    def _league_summary(lg):
        lid = str(lg.get("league_id") or "")
        if not lid:
            return None
        lg_platform = (lg.get("platform") or "sleeper").lower()
        lg_season = int(lg.get("season") or season)
        try:
            lctx = get_league_ctx_from_cache(lg_platform, lid, lg_season, allow_build=False)
        except Exception:
            return {"league_id": lid, "name": lg.get("name", "Unknown"),
                    "platform": lg_platform, "error": True,
                    "is_favorite": bool(lg.get("is_favorite"))}
        if not lctx:
            # Cold cache: a background warm was kicked. Render a hydratable shell
            # now rather than block the entire page on this one league's cold
            # build -- the old behavior built every league here, so a multi-league
            # cold load took 80s+ and pinned a worker, which starved the very
            # summary/refresh XHRs the page fires. The client summary + matchup
            # loaders fill the shell, and its cross-league insights fold in on a
            # later load once the context is warm.
            return {"league_id": lid, "name": lg.get("name") or "Unknown",
                    "platform": lg_platform, "season": lg_season,
                    "loading": True,
                    "is_favorite": bool(lg.get("is_favorite"))}
        rosters = lctx.get("rosters") or []
        players_index = lctx.get("players_index") or {}
        league_obj = lctx.get("league") or {}
        latest_draft = lctx.get("latest_draft") if isinstance(lctx.get("latest_draft"), dict) else {}
        from utils.league_payload import draft_start_ms, startup_draft_phase
        draft_phase = startup_draft_phase(league_obj, latest_draft, rosters)
        if draft_phase != "drafted":
            # Roster shells exist before a startup/redraft, so a linked team used
            # to look "drafted" and get fake positional ranks. Treat thin rosters
            # as pending and show the draft countdown instead.
            return {
                "league_id": lid,
                "name": league_obj.get("name") or lg.get("name") or "Unknown",
                "platform": lg_platform,
                "season": lg_season,
                "pending": True,
                "predraft": True,
                "draft_phase": draft_phase,
                "draft_start_ms": draft_start_ms(league_obj, latest_draft),
                "reason": "Drafting now" if draft_phase == "drafting" else "Draft not started",
                "is_favorite": bool(lg.get("is_favorite")),
            }
        # Which roster is "yours": prefer the account's stored team (and ESPN SWID /
        # Sleeper identity), then the session viewer. A leftover ESPN owner id in
        # the session must not mark every Sleeper league as "Team not linked yet".
        viewer_roster = None
        _account_id = account_id
        if _account_id:
            try:
                from dashboard_services.accounts import resolve_account_viewer_for_league
                _av = resolve_account_viewer_for_league(
                    _account_id, lg_platform, lid, lg_season,
                    lctx.get("users") or [], rosters,
                )
                _rid = str((_av or {}).get("viewer_roster_id") or "")
                if _rid:
                    viewer_roster = next(
                        (r for r in rosters if str(r.get("roster_id") or "") == _rid),
                        None,
                    )
            except Exception:
                logger.debug("portfolio account viewer resolve failed", exc_info=True)
        if viewer_roster is None:
            from utils.scorezone_user import match_viewer_roster
            sleeper_owner = sleeper_owner_id_for_account(
                _account_id, viewer_user_id, lg_platform,
            )
            viewer_roster = match_viewer_roster(
                rosters,
                team_id=lg.get("team_id"),
                owner_id=sleeper_owner,
            )
        if not viewer_roster:
            # Startup/redraft-not-started already returned above. A full roster
            # league with no matching team is a link problem, not a draft one.
            return {
                "league_id": lid,
                "name": league_obj.get("name") or lg.get("name") or "Unknown",
                "platform": lg_platform,
                "season": lg_season,
                "pending": True,
                "predraft": False,
                "reason": "Team not linked yet",
                "is_favorite": bool(lg.get("is_favorite")),
            }
        rid = str(viewer_roster.get("roster_id"))
        from dashboard_services.ai.context_builders import (
            league_format_value_lookup, portfolio_record_and_rank,
        )
        from dashboard_services.display_names import team_label_from_user
        from utils.roster_strength import (
            rank_rosters_by_position, roster_pos_value_lists, strength_percentile,
        )
        values_by_id = league_format_value_lookup(lctx, _cache=_value_lookup_cache)
        wins, losses, ties, pf, rank = portfolio_record_and_rank(lctx, rid, viewer_roster)
        owner_id = str(viewer_roster.get("owner_id") or "")
        owner_user = next(
            (u for u in (lctx.get("users") or [])
             if str(u.get("user_id") or "") == owner_id),
            None,
        )
        team_name = team_label_from_user(owner_user, viewer_roster, fallback="")
        total_teams = int(lctx.get("total_rosters") or len(rosters) or 12)
        player_ids = [str(p) for p in (viewer_roster.get("players") or [])]
        all_players = {}
        total_value = 0.0
        for pid in player_ids:
            v = values_by_id.get(pid) or {}
            val = float(v.get("value") or 0)
            total_value += val
            meta = players_index.get(pid) or {}
            pos = (v.get("position") or meta.get("pos") or "").upper()
            nfl_team = (v.get("team") or meta.get("team") or "").upper()
            all_players[pid] = {
                "name": v.get("name") or meta.get("name") or f"Player {pid}",
                "position": pos,
                "value": val,
                "pos_rank": v.get("pos_rank_label") or "",
                "nfl_team": nfl_team,
            }

        def _median(xs):
            s = sorted(xs)
            n = len(s)
            if not n:
                return 0.0
            m = n // 2
            return s[m] if n % 2 else (s[m - 1] + s[m]) / 2

        # Same ranking helper as the Teams page (weighted_pos_strength +
        # league-type values) so a WR5 chip cannot read as WR6 in-league.
        #
        # The My Leagues summary card shows the in-league PERCENTILE of that
        # strength (then averages those percentiles across leagues). Percentiles
        # stay centered at 50th for a typical team; averaging signed % vs median
        # let one thin league drag a stacked position negative.
        roster_positions = lctx.get("roster_positions") or []
        try:
            slot_counts = count_roster_positions(roster_positions)
        except Exception:
            slot_counts = {"QB": 1, "RB": 2, "WR": 3, "TE": 1, "FLEX": 1}
        team_pos_values = roster_pos_value_lists(
            rosters, values_by_id, players_index=players_index, rid_cast=str,
        )
        pos_strengths, pos_ranks_map = rank_rosters_by_position(
            team_pos_values, slot_counts,
        )
        pos_user_rank = {}
        pos_user_vals = {}
        pos_user_pctile = {}
        pos_league_avgs = {}
        for pos in ["QB", "RB", "WR", "TE"]:
            all_strengths = [s.get(pos, 0.0) for s in pos_strengths.values()]
            user_strength = (pos_strengths.get(rid) or {}).get(pos, 0.0)
            # Median is still used for the per-league "WR-Spread" archetype
            # badge (ratio vs a typical team). The summary card uses percentile.
            pos_user_vals[pos] = user_strength
            pos_league_avgs[pos] = _median(all_strengths) or 1
            pos_user_pctile[pos] = strength_percentile(
                user_strength, all_strengths,
            )
            if len(all_strengths) > 1:
                pos_user_rank[pos] = (pos_ranks_map.get(pos) or {}).get(rid, "?")
            else:
                pos_user_rank[pos] = 1
        # Recent streak from df_weekly (last 3 finalized weeks for this roster)
        streak = []
        try:
            df_w = lctx.get("df_weekly")
            if df_w is not None and not df_w.empty and "finalized" in df_w.columns:
                my_rows = df_w[(df_w["roster_id"].astype(str) == rid) & (df_w["finalized"] == True)]
                if not my_rows.empty and "week" in my_rows.columns:
                    my_rows = my_rows.sort_values("week", ascending=False).head(3)
                    for _, row in my_rows.iterrows():
                        pts = row.get("points")
                        opp = row.get("points_against")
                        if pts is None or opp is None or pts != pts or opp != opp:
                            continue
                        pts, opp = float(pts), float(opp)
                        streak.append("W" if pts > opp else ("L" if pts < opp else "T"))
        except Exception:
            logger.debug("suppressed exception", exc_info=True)
        # Urgency score: lower = needs more attention
        urgency = wins - losses + (rank if isinstance(rank, int) else 0) * -0.1
        summary = {
            "league_id": lid,
            "name": league_obj.get("name") or lg.get("name") or "Unknown",
            "platform": lg_platform,
            "season": lg_season,
            "wins": wins, "losses": losses, "ties": ties,
            "record": f"{wins}-{losses}" + (f"-{ties}" if ties else ""),
            "rank": rank, "total_teams": total_teams,
            "pf": round(pf, 1),
            "total_value": round(total_value, 1),
            "all_players": all_players,
            "streak": streak,
            "urgency": urgency,
            "pos_user_vals": pos_user_vals,
            "pos_league_avgs": pos_league_avgs,
            "pos_user_pctile": pos_user_pctile,
            "pos_user_rank": pos_user_rank,
            "offseason": lctx.get("offseason_mode", False),
            "team_name": team_name,
            "is_favorite": bool(lg.get("is_favorite")),
        }
        # Playoff odds + posture (v25 cards/board), cache-only like
        # build_league_summary: never run the sim inline here.
        try:
            from dashboard_services.portfolio_board import (
                odds_for_ctx, posture_for, window_posture_for,
            )
            _odds = odds_for_ctx(lctx, lg_platform, rid)
            _win = window_posture_for(
                lctx, rosters, values_by_id, rid, odds_pct=_odds.get("pct"))
            if _win:
                _posture_label, _posture_cls = _win["label"], _win["cls"]
                _posture_color, _posture_shape = _win["color"], _win["shape"]
            else:
                from dashboard_services.ai.context_builders import (
                    ctx_scoring_type as _ctx_scoring_type,
                )
                _posture_label, _posture_cls = posture_for(
                    _odds.get("pct"), wins, losses,
                    is_redraft=_ctx_scoring_type(lctx) == "redraft")
                _posture_color, _posture_shape = "", ""
            summary.update(
                playoff_odds_pct=_odds.get("pct"),
                playoff_odds_delta=_odds.get("delta"),
                posture=_posture_label,
                posture_cls=_posture_cls,
                posture_color=_posture_color,
                posture_shape=_posture_shape,
            )
        except Exception:
            logger.debug("portfolio inline odds attach failed", exc_info=True)
        return summary

    # Stale-while-revalidate: serve each league's card from the Postgres
    # portfolio summary cache -- 15 minutes fresh on game days (scores move
    # fast), 1 hour otherwise. A stale full card still renders (marked stale)
    # and kicks a bounded background refresh; only a miss renders a hydratable
    # shell, so the page never blocks on summary computation and a TTL expiry
    # never blanks the whole portfolio at once.
    # Sleeper-only sessions have no account_id (and therefore no cache); they
    # keep the inline _league_summary path.
    week = int(nfl_state.get("week") or 0)
    _swr_ttl_s = _portfolio_swr_ttl_s(season, week)

    def _league_card(lg):
        if not account_id:
            return _league_summary(lg)
        lid = str(lg.get("league_id") or "")
        if not lid:
            return None
        lg_platform = (lg.get("platform") or "sleeper").lower()
        lg_season = int(lg.get("season") or season)
        is_fav = bool(lg.get("is_favorite"))

        def _shell():
            return {"league_id": lid, "name": lg.get("name") or "Unknown",
                    "platform": lg_platform, "season": lg_season,
                    "loading": True, "is_favorite": is_fav}

        try:
            cached = get_cached_summary(account_id, lg_platform, lid, lg_season)
        except Exception:
            logger.debug("portfolio summary cache read failed league=%s", lid,
                         exc_info=True)
            cached = None
        if isinstance(cached, dict) and cached:
            state = cached.get("state")
            fresh = _cached_summary_fresh(cached, _swr_ttl_s)
            base = {"league_id": lid,
                    "name": cached.get("name") or lg.get("name") or "Unknown",
                    "platform": lg_platform, "season": lg_season,
                    "is_favorite": is_fav}
            if state == "team_not_linked":
                if fresh:
                    return {**base, "pending": True, "predraft": False,
                            "reason": "Team not linked yet"}
            elif state == "reconnect_required":
                if fresh:
                    return {**base, "error": True}
            else:
                # "ready" / "partial" (or legacy entries without a state):
                # full card, fresh OR stale. A stale card is marked so the
                # client still hydrates it, but the user sees last-good
                # record/standing/players instantly instead of a loading
                # shell -- and the cross-league aggregates, which only count
                # non-loading cards, stay populated. pos_user_vals /
                # pos_league_avgs only feed the archetype badge, which
                # degrades to "Balanced" when absent.
                card = dict(cached)
                card.update(base)
                card.setdefault("pos_user_vals", {})
                card.setdefault("pos_league_avgs", {})
                if fresh:
                    # Legacy entries cached before build_league_summary
                    # carried pos_user_vals serve the card but can only ever
                    # badge "Balanced"; rebuild one in the background so the
                    # next load badges from real strengths.
                    if not card.get("pos_user_vals"):
                        _kick_summary_refresh(account_id, lg)
                    return card
                card["stale"] = True
                _kick_summary_refresh(account_id, lg)
                return card
        # Cache miss (or a stale non-card state): shell now, refresh in the
        # background.
        _kick_summary_refresh(account_id, lg)
        return _shell()

    # Per-request memoization for league_format_value_lookup, used by the
    # Sleeper-only _league_summary fallback below: the value table walk
    # depends only on league type settings, not the viewer.
    _value_lookup_cache: dict = {}
    leagues_data = [_league_card(lg) for lg in (league_inputs or [])]
    leagues_data = [r for r in leagues_data if r]
    leagues_data.sort(key=lambda x: (not x.get("is_favorite", False), x.get("name", "")))

    valid_leagues = [lg for lg in leagues_data
                     if not lg.get("error") and not lg.get("not_in_league")
                     and not lg.get("pending") and not lg.get("loading")]
    num_leagues = len(leagues_data)
    total_wins = sum(lg.get("wins", 0) for lg in valid_leagues)
    total_losses = sum(lg.get("losses", 0) for lg in valid_leagues)
    total_ties = sum(lg.get("ties", 0) for lg in valid_leagues)

    # Cross-league positional strength: mean in-league percentile per position.
    # Empty portfolio omits the card rather than rendering fake 50ths.
    from utils.roster_strength import average_league_percentiles
    cross_pos = (
        average_league_percentiles(lg.get("pos_user_pctile") or {} for lg in valid_leagues)
        if valid_leagues else {}
    )

    # Sort valid leagues by urgency: losing records and low standings first
    valid_leagues.sort(key=lambda lg: (
        not lg.get("is_favorite", False),
        lg.get("wins", 0) - lg.get("losses", 0),
        -(lg.get("rank") if isinstance(lg.get("rank"), int) else 999),
    ))

    # NFL team concentration: group all player holdings by their NFL team
    nfl_team_data: dict = {}  # team -> {player_list: [...], league_ids: set}
    pid_meta: dict = {}
    pid_leagues: dict = {}  # pid -> [league_name_abbrev]
    for lg in valid_leagues:
        lg_abbrev = (lg.get("name") or "?")[:14]
        for pid, p in (lg.get("all_players") or {}).items():
            if not p.get("value"):
                continue
            nfl = p.get("nfl_team") or ""
            if nfl and nfl not in ("", "FA", "N/A"):
                if nfl not in nfl_team_data:
                    nfl_team_data[nfl] = {"player_list": [], "league_ids": set()}
                nfl_team_data[nfl]["player_list"].append({
                    "pid": pid,
                    "name": p.get("name", ""),
                    "position": p.get("position", ""),
                    "value": p.get("value", 0),
                    "pos_rank": p.get("pos_rank", ""),
                    "league": lg_abbrev,
                })
                nfl_team_data[nfl]["league_ids"].add(lg.get("league_id", ""))
            if pid not in pid_meta or p.get("value", 0) > pid_meta[pid].get("value", 0):
                pid_meta[pid] = p
            pid_leagues.setdefault(pid, []).append(lg_abbrev)

    # Top NFL teams by player count - deduplicate players, sort by value
    nfl_exposure = []
    for t, d in nfl_team_data.items():
        seen_pids: set = set()
        unique_players = []
        for pl in sorted(d["player_list"], key=lambda x: -x["value"]):
            if pl["pid"] not in seen_pids:
                seen_pids.add(pl["pid"])
                unique_players.append(pl)
        nfl_exposure.append({
            "team": t,
            "count": len(unique_players),
            "leagues": len(d["league_ids"]),
            "players": unique_players,
        })
    nfl_exposure.sort(key=lambda x: (-x["count"], -x["leagues"]))
    nfl_exposure = nfl_exposure[:12]

    # Player holdings across leagues
    holdings = []
    for pid, p in pid_meta.items():
        if p.get("value", 0) > 0:
            holdings.append({
                **p, "pid": pid,
                "shares": len(pid_leagues.get(pid, [])),
                "in_leagues": pid_leagues.get(pid, []),
            })
    # v25: holdings sort strictly by value desc (shares break ties only).
    holdings.sort(key=lambda x: (-x["value"], -x["shares"]))

    # Join 7-day value-rank movement from the in-memory value table (no DB round
    # trip) so the holdings trend column works even when the per-league player
    # blobs don't carry it. Only fill when missing so a real per-league value wins.
    try:
        _pf_rc = {str(r.get("id")): r.get("rank_change_7d")
                  for r in (get_model_value_table_cached() or [])
                  if isinstance(r, dict) and r.get("id")}
        for _h in holdings:
            if _h.get("rank_change_7d") in (None, 0):
                _rc = _pf_rc.get(str(_h.get("pid")))
                if _rc is not None:
                    _h["rank_change_7d"] = _rc
    except Exception:
        logger.debug("suppressed exception", exc_info=True)

    # Bye week + injury status for the holdings columns (v25), from the
    # shared NFL players index. Best-effort: missing data renders a dash.
    # The raw Sleeper rows carry no bye_week (see _pf_players_with_byes),
    # so attach byes from the team map first or the column is all dashes.
    try:
        from dashboard_services.api import get_nfl_players as _gnp_hold
        _nfl_hold = _pf_players_with_byes(_gnp_hold() or {}, season)
        for _h in holdings:
            _pl = _nfl_hold.get(str(_h.get("pid"))) or {}
            if _pl.get("bye_week") is not None:
                _h["bye_week"] = _pl.get("bye_week")
            _st = str(_pl.get("injury_status") or "").strip()
            if _st and _st.lower() not in ("active", "act"):
                _h["status"] = _st
    except Exception:
        logger.debug("holdings bye/status enrich failed", exc_info=True)

    # ── Insights first paint, embedded ──────────────────────────────────
    # This pass already paid for the league resolution (shared into the
    # snapshot above). The Bye Watch payload and, while the slate is
    # still ahead, the pregame Rooting Guide payload (no live calls)
    # are computed from the LIGHT resolver (provider-layer rosters +
    # shared players index/model table by reference -- no league
    # contexts) and embedded in the page, so the two cards paint with
    # zero first-load API calls. The endpoints stay as the
    # refresh/fallback path (live points, post-kickoff byes rollover).
    _embed_bye = _embed_rooting = None
    try:
        if current_week and (nfl_state.get("season_type") in ("reg", "post")):
            _emb_base = _pf_light_base(
                identity={"viewer_username": viewer_username,
                          "viewer_user_id": viewer_user_id,
                          "account_id": account_id},
                league_inputs=league_inputs or [],
                season=season, week=current_week)
            _embed_bye, _embed_rooting = _pf_embed_payloads(_emb_base)
    except Exception:
        logger.debug("portfolio insights embed failed", exc_info=True)
        _embed_bye = _embed_rooting = None

    body = build_portfolio_body(
        portfolio_signed_in_label(),
        valid_leagues, leagues_data, season,
        holdings, num_leagues, nfl_exposure, cross_pos,
        total_wins, total_losses, total_ties,
        current_week=current_week,
    )
    if _embed_bye is not None:
        body += ("<script type=\"application/json\" id=\"pf-bye-embedded\">"
                 + json.dumps(_embed_bye).replace("</", "<\\/")
                 + "</script>")
    if _embed_rooting is not None:
        body += ("<script type=\"application/json\" id=\"pf-rooting-embedded\">"
                 + json.dumps(_embed_rooting).replace("</", "<\\/")
                 + "</script>")

    # Always render with a league nav context - fall back to first valid league
    nav_league_id = from_league
    nav_platform = from_platform
    nav_season = from_season or season
    if not nav_league_id and valid_leagues:
        first = valid_leagues[0]
        nav_league_id = first.get("league_id")
        nav_platform = first.get("platform") or "sleeper"
        nav_season = first.get("season") or season
    return render_page("My Leagues – BR Fantasy", nav_league_id, "portfolio", body, nav_platform, nav_season)


def build_watchlist_page_body() -> str:
    """Shell for the full watchlist page. Client-driven: static/app.js's
    initWatchlistPage reads the (synced/local) watchlist, fetches per-player
    value/mover/injury data, and renders the sortable table + value-vs-age chart."""
    return """
    <div class="page-layout" data-page="watchlist">
      <main class="page-main">
        <div class="wl-page">
          <header class="wl-page-head">
            <div class="wl-head-title">
              <h1 class="wl-page-title">Watchlist</h1>
              <span id="wlPageCount" class="wl-count-pill" hidden></span>
            </div>
            <label class="wl-page-sort">Sort
              <select id="wlPageSort" class="search">
                <option value="value">Value (high to low)</option>
                <option value="mover">Biggest 7-day move</option>
                <option value="age">Age (young to old)</option>
                <option value="name">Name (A to Z)</option>
                <option value="added">Recently added</option>
              </select>
            </label>
          </header>
          <div id="wlPageSyncNote" class="wl-sync-note"></div>

          <div id="wlPageStats" class="wl-stats"></div>
          <div id="wlPageAlerts" class="wl-alerts-wrap"></div>

          <div class="card wl-page-card">
            <div class="wl-card-head">
              <h3>Value vs Age</h3>
              <span class="wl-card-hint">Younger &amp; higher is better</span>
            </div>
            <div class="card-body"><div id="wlPageScatter" class="wl-page-scatter"></div></div>
          </div>
          <div class="card wl-page-card">
            <div class="wl-card-head"><h3>Watched Players</h3></div>
            <div class="card-body"><div id="wlPageTable" class="wl-page-table"></div></div>
          </div>
        </div>
      </main>
    </div>
    """


@user_pages_bp.route("/api/portfolio-actions")
def api_portfolio_actions():
    """Cross-league action digest for My Leagues (roadmap R04).

    PRO-gated Front Office-style bullets across linked leagues. Best-effort:
    scans up to 12 linked leagues for lineup issues and injury hints.
    """
    from flask import jsonify, session
    from dashboard_services.subscriptions import has_premium_for_viewer
    from utils.cross_league_actions import (
        calendar_action,
        injury_stash_action,
        lineup_actions_from_issues,
        make_action,
        rank_cross_league_actions,
        roster_slot_action,
        select_waiver_add,
        waiver_pickup_action,
    )
    from utils.lineup_issues import find_lineup_issues, projection_upgrades
    from utils.scorezone_user import match_viewer_roster

    # Scan accounting: the client needs to tell "checked everything, nothing
    # to do" apart from "checked nothing / only some leagues". scan_total is
    # set later to the full linked-league count; every ATTEMPTED league lands
    # in exactly one bucket below (scanned / skipped_cold / skipped_no_roster),
    # so scanned + skipped_* = attempted <= total, and incomplete holds
    # whenever scanned < total (including leagues past the attempt slice).
    # A league covered by a fresh per-league fragment (scanned by a pass
    # inside _ACTIONS_FRAGMENT_TTL) counts as scanned without an attempt.
    scan_total = 0
    scan_scanned = 0
    scan_skipped_cold = 0
    scan_skipped_no_roster = 0

    def _scan_payload(force_incomplete=False):
        return {
            "total": scan_total,
            "scanned": scan_scanned,
            "skipped_cold": scan_skipped_cold,
            "skipped_no_roster": scan_skipped_no_roster,
            "incomplete": bool(force_incomplete) or scan_scanned < scan_total,
        }

    viewer_username = session.get("viewer_username")
    viewer_user_id = session.get("viewer_user_id")
    account_id = session.get("account_id")
    if (not viewer_username or not viewer_user_id) and not account_id:
        # Nothing was scanned at all, so an empty list here is not "all
        # caught up" - report it as incomplete.
        return jsonify({"actions": [], "scan": _scan_payload(force_incomplete=True)})

    # Cheap cache probe BEFORE the heavy preamble (resolve_my_leagues live
    # enrichment, the PRO sweep, player/value loads): a fresh complete scan
    # is served without paying any of it. The key needs only the session
    # identity and the NFL season/week, mirroring the dispatch key below.
    try:
        _probe_state = get_nfl_state() or {}
        _probe_season = int(_probe_state.get("season") or datetime.now().year)
        _probe_week = int(_probe_state.get("week") or 0)
    except Exception:
        _probe_season = _probe_week = 0
    if _probe_season and _probe_week:
        _probe_key = (_user_key() or f"viewer:{viewer_user_id}",
                      _probe_season, _probe_week)
        with _ACTIONS_LOCK:
            _probe_hit = _ACTIONS_CACHE.get(_probe_key)
            if _probe_hit and time.time() - _probe_hit[0] < _ACTIONS_TTL:
                return jsonify(_reattach_done_states(
                    _probe_hit[1], _user_key(), _probe_season, _probe_week))

    nfl_state = get_nfl_state() or {}
    season = int(nfl_state.get("season") or datetime.now().year)
    try:
        from dashboard_services.accounts import resolve_my_leagues
        league_inputs, season = resolve_my_leagues(
            viewer_user_id, session.get("account_id"), season
        )
    except Exception:
        return jsonify({"actions": [], "scan": _scan_payload(force_incomplete=True)})

    def _portfolio_viewer_has_pro() -> bool:
        if has_premium_for_viewer(viewer_username, viewer_user_id, None, "sleeper", season):
            return True
        for lg in (league_inputs or [])[:8]:
            lid = str(lg.get("league_id") or "")
            plat = str(lg.get("platform") or "sleeper").lower()
            sea = int(lg.get("season") or season)
            if lid and has_premium_for_viewer(
                viewer_username, viewer_user_id, lid, plat, sea,
            ):
                return True
        return False

    if not _portfolio_viewer_has_pro():
        return jsonify({"paywall": True, "error": "Premium required", "actions": []}), 403

    actions: list = []
    board_meta: dict = {}
    # Memoized league value lookups for board-header window posture, so
    # same-format leagues share one value-table walk across the scan.
    _board_value_cache: dict = {}
    week = 0
    in_season = False
    week_games: list = []
    # Shared fallback projection map for the swap scan (Part C): built at most
    # once per request, only for leagues whose ctx carries no proj_by_week.
    shared_proj_map = None
    shared_proj_tried = False
    try:
        from dashboard_services.api import get_nfl_players
        from utils.utils import load_week_schedule
        nfl_players = get_nfl_players() or {}
        week = int(nfl_state.get("week") or 0)
        in_season = bool(week and nfl_state.get("season_type") in ("reg", "post"))
        teams_playing = set()
        if in_season:
            week_games = list(load_week_schedule(season, week) or [])
            for g in week_games:
                for side in ("home", "away"):
                    t = str(g.get(side) or "").upper()
                    if t:
                        teams_playing.add(t)
    except Exception:
        nfl_players = {}
        teams_playing = set()
        week_games = []

    # Model value table (cached) powers the format-aware waiver pickup. Load it
    # once for all leagues rather than per-league.
    try:
        model_value_table = list(get_model_value_table_cached() or [])
    except Exception:
        model_value_table = []
    try:
        from utils.waiver_score import WEIGHTS as _WAIVER_WEIGHTS
        _waiver_base_min = float(_WAIVER_WEIGHTS.min_value)
    except Exception:
        _waiver_base_min = 25.0

    # The scan total covers ALL linked leagues, not just the attempted slice:
    # a league the slice never reaches is still a league that was not checked,
    # and the client's missed count (total - scanned) must include it.
    scan_total = sum(
        1 for lg in (league_inputs or []) if str(lg.get("league_id") or "")
    )

    # Cache / single-flight key (see _ACTIONS_CACHE above). The dispatch
    # lives below _run_actions_compute: serve a fresh complete scan, or
    # start / defer to the in-flight background compute and answer with
    # the honest incomplete payload immediately (no request-thread wait).
    _actions_key = None
    if season and week:
        _actions_key = (_user_key() or f"viewer:{viewer_user_id}",
                        season, week)
    # Session-derived identity is captured here, on the request thread:
    # the scan itself may run on a background thread with no request
    # context (see the dispatch below _run_actions_compute).
    _captured_ukey = _user_key()

    def _deferred_payload():
        return {
            "actions": [],
            "scan": {"total": scan_total, "scanned": 0,
                     "skipped_cold": scan_total,
                     "skipped_no_roster": 0, "incomplete": True},
            "week": week, "season": season,
            "board": {"groups": [], "week": week},
            "grade_line": "",
        }

    def _fallback_payload():
        return {
            "actions": [],
            "scan": _scan_payload(force_incomplete=True),
            "week": week, "season": season,
            "board": {"groups": [], "week": week},
            "grade_line": "",
        }

    def _run_actions_compute():
        nonlocal scan_scanned, scan_skipped_cold, scan_skipped_no_roster
        nonlocal shared_proj_map, shared_proj_tried
        for lg in (league_inputs or [])[:12]:
            lid = str(lg.get("league_id") or "")
            plat = str(lg.get("platform") or "sleeper").lower()
            lg_season = int(lg.get("season") or season)
            if not lid:
                continue
            _frag_key = None
            _frag_actions_start = len(actions)
            _scanned_before = scan_scanned
            if _actions_key is not None:
                _frag_key = (_actions_key, plat, lid, lg_season)
                _frag = _get_action_fragment(_frag_key)
                if _frag is not None:
                    # A recent pass already scanned this league; reuse its
                    # fragment rather than rescanning a context the cache
                    # carousel may have evicted since (see _ACTIONS_FRAGMENTS).
                    actions.extend(copy.deepcopy(_frag["actions"]))
                    if _frag.get("board_meta") is not None:
                        board_meta[f"{plat}|{lid}"] = copy.deepcopy(
                            _frag["board_meta"])
                    scan_scanned += 1
                    continue
            try:
                lctx = get_league_ctx_from_cache(plat, lid, lg_season, allow_build=False) or {}
            except Exception:
                scan_skipped_cold += 1
                continue
            if not lctx:
                # Cold cache: a background warm was kicked. Skip this league for now so
                # the digest returns promptly instead of blocking on a cold build (the
                # old behavior serially built every league, taking 100s+ on a miss).
                scan_skipped_cold += 1
                continue
            rosters = lctx.get("rosters") or []
            league_obj = lctx.get("league") or {}
            league_name = league_obj.get("name") or lg.get("name") or lid
            viewer_roster = None
            if account_id:
                try:
                    from dashboard_services.accounts import resolve_account_viewer_for_league
                    av = resolve_account_viewer_for_league(
                        account_id, plat, lid, lg_season,
                        lctx.get("users") or [], rosters,
                    )
                    rid = str((av or {}).get("viewer_roster_id") or "")
                    if rid:
                        viewer_roster = next(
                            (r for r in rosters if str(r.get("roster_id") or "") == rid), None
                        )
                except Exception:
                    viewer_roster = None
            if viewer_roster is None:
                try:
                    viewer_roster = match_viewer_roster(
                        rosters,
                        team_id=lg.get("team_id"),
                        owner_id=sleeper_owner_id_for_account(
                            account_id, viewer_user_id, plat,
                        ),
                    )
                except Exception:
                    # One league's identity data must not abort the whole
                    # scan (it previously escaped as a 500 and leaked the
                    # single-flight mark): count it unresolved, move on.
                    logger.debug("[portfolio-actions] roster match failed",
                                 exc_info=True)
                    viewer_roster = None
            if not viewer_roster:
                scan_skipped_no_roster += 1
                continue
            # Roster matched: the scans below actually run for this league.
            # Mid-scan sub-scan failures are best-effort and do not un-count it.
            scan_scanned += 1
            # Board group header meta (v25): record line, cache-only playoff
            # odds, posture. Best-effort; the header degrades to the name.
            try:
                from dashboard_services import portfolio_board as _pb_meta
                _vs = viewer_roster.get("settings") or {}
                _mw = int(_vs.get("wins") or 0)
                _ml = int(_vs.get("losses") or 0)
                _mt = int(_vs.get("ties") or 0)
                _ranked = sorted(
                    rosters,
                    key=lambda r: (
                        -int((r.get("settings") or {}).get("wins") or 0),
                        -float((r.get("settings") or {}).get("fpts") or 0),
                    ),
                )
                _rk = next(
                    (i + 1 for i, r in enumerate(_ranked)
                     if str(r.get("roster_id")) == str(viewer_roster.get("roster_id"))),
                    None,
                )
                _odds = _pb_meta.odds_for_ctx(lctx, plat, viewer_roster.get("roster_id"))
                # Same Teams-parity window posture the league cards carry, so
                # a board group header can never disagree with its card.
                from dashboard_services.ai.context_builders import (
                    ctx_scoring_type as _ctx_scoring_type,
                    league_format_value_lookup as _lg_value_lookup,
                )
                _win = _pb_meta.window_posture_for(
                    lctx, rosters,
                    _lg_value_lookup(lctx, _cache=_board_value_cache),
                    viewer_roster.get("roster_id"), odds_pct=_odds.get("pct"))
                if _win:
                    _posture = (_win["label"], _win["cls"])
                else:
                    _posture = _pb_meta.posture_for(
                        _odds.get("pct"), _mw, _ml,
                        is_redraft=_ctx_scoring_type(lctx) == "redraft")
                _rec = f"{_mw}-{_ml}" + (f"-{_mt}" if _mt else "")
                if _rk:
                    _suf = ("th" if 10 <= _rk % 100 <= 20 else
                            {1: "st", 2: "nd", 3: "rd"}.get(_rk % 10, "th"))
                    _rec += f" · {_rk}{_suf} of {len(rosters)}"
                _bmeta = {
                    "league_name": league_name,
                    "record_line": _rec,
                    "odds_pct": _odds.get("pct"),
                    "odds_delta": _odds.get("delta"),
                    "posture": _posture[0],
                    "roster_id": str(viewer_roster.get("roster_id") or ""),
                    # Structured record facts beside the formatted
                    # record line, so the deep dive's Brief can rank
                    # leagues (flagship / deep end) without parsing
                    # display text back apart.
                    "wins": _mw,
                    "losses": _ml,
                    "ties": _mt,
                    "rank": _rk,
                    "team_count": len(rosters),
                }
                # Free riders from data already in hand in this loop:
                # the viewer's points for (the standing sort above
                # already reads fpts off every roster's settings) and
                # the league's calendar weeks (the calendar nudge
                # below reads the same settings). No new fetches.
                try:
                    _pf = float(_vs.get("fpts"))
                    _bmeta["points_for"] = _pf
                except (TypeError, ValueError):
                    pass
                _bset = (league_obj.get("settings")
                         or lctx.get("league_settings") or {})
                for _skey, _mkey in (
                        ("trade_deadline", "trade_deadline_week"),
                        ("playoff_week_start", "playoff_week_start")):
                    try:
                        _sv = int(_bset.get(_skey) or 0)
                    except (TypeError, ValueError):
                        _sv = 0
                    if 1 <= _sv <= 18:
                        _bmeta[_mkey] = _sv
                board_meta[f"{plat}|{lid}"] = _bmeta
            except Exception:
                logger.debug("[portfolio-actions] board meta failed", exc_info=True)
            starters = [str(p) for p in (viewer_roster.get("starters") or [])]
            if starters and nfl_players:
                info = {}
                for pid in starters:
                    pl = nfl_players.get(pid) or {}
                    info[pid] = {
                        "name": pl.get("full_name") or pl.get("last_name") or "",
                        "team": pl.get("team") or "",
                        "injury_status": pl.get("injury_status") or "",
                    }
                try:
                    issues = find_lineup_issues(
                        starters, info, teams_playing or None)
                    actions.extend(lineup_actions_from_issues(
                        issues, platform=plat, season=lg_season,
                        league_id=lid, league_name=league_name,
                    ))
                except Exception:
                    logger.debug("[portfolio-actions] lineup scan failed",
                                 exc_info=True)
            # Bench-swap parity (Part C): the Start-X-over-Y alerts the push
            # reminders and the league dashboard Next Steps produce. Same solver,
            # same thresholds (min_gain 2.0, max 2 per league), locked players
            # excluded from both sides. Best-effort like every other sub-scan.
            if in_season and starters:
                try:
                    proj_map = {}
                    pbw = lctx.get("proj_by_week") or {}
                    wk_proj = {}
                    if isinstance(pbw, dict):
                        wk_proj = pbw.get(week) or pbw.get(str(week)) or {}
                    if isinstance(wk_proj, dict):
                        proj_map = {
                            str(k): v for k, v in (wk_proj.get("projections") or {}).items()
                        }
                    if not proj_map:
                        if not shared_proj_tried:
                            shared_proj_tried = True
                            try:
                                _bpw = build_projections_by_week(season, week, None) or {}
                                _wk = {}
                                if isinstance(_bpw, dict):
                                    _wk = _bpw.get(week) or _bpw.get(str(week)) or {}
                                if isinstance(_wk, dict):
                                    shared_proj_map = {
                                        str(k): v
                                        for k, v in (_wk.get("projections") or {}).items()
                                    } or None
                            except Exception:
                                shared_proj_map = None
                        proj_map = dict(shared_proj_map or {})
                    if proj_map:
                        from utils.lineups import locked_teams_from_games
                        _reserve = {str(p) for p in (viewer_roster.get("reserve") or []) if p}
                        _taxi = {str(p) for p in (viewer_roster.get("taxi") or []) if p}
                        eligible = [
                            str(p) for p in (viewer_roster.get("players") or [])
                            if str(p) not in _reserve and str(p) not in _taxi
                        ]
                        _pidx = lctx.get("players_index") or {}
                        pos_map = {}
                        name_map = {}
                        inj_map_swap = {}
                        for _pid in eligible:
                            _pl = nfl_players.get(_pid) or {}
                            _meta = _pidx.get(_pid) or {}
                            pos_map[_pid] = str(
                                _pl.get("position") or _meta.get("position")
                                or _meta.get("pos") or ""
                            )
                            name_map[_pid] = (
                                _pl.get("full_name") or _meta.get("full_name")
                                or _meta.get("name") or _pl.get("last_name") or ""
                            )
                            inj_map_swap[_pid] = str(_pl.get("injury_status") or "")
                        for _pid in starters:
                            if _pid not in name_map:
                                _pl = nfl_players.get(_pid) or {}
                                name_map[_pid] = (
                                    _pl.get("full_name") or _pl.get("last_name") or ""
                                )
                        locked_teams = locked_teams_from_games(week_games)
                        locked_pids = {
                            _pid for _pid in eligible
                            if str((nfl_players.get(_pid) or {}).get("team") or "").upper()
                            in locked_teams
                        }
                        swaps = projection_upgrades(
                            starters, eligible, proj_map, pos_map,
                            lctx.get("roster_positions") or [],
                            min_gain=2.0, max_swaps=2,
                            injury_status=inj_map_swap,
                            locked_pids=locked_pids,
                        )
                        for _sw in swaps[:2]:
                            _in = str(_sw.get("in") or "")
                            _out = str(_sw.get("out") or "")
                            if not _in or not _out:
                                continue
                            _in_name = name_map.get(_in) or f"Player {_in}"
                            _out_name = name_map.get(_out) or f"Player {_out}"
                            try:
                                _in_proj = float(proj_map.get(_in) or 0)
                                _out_proj = float(proj_map.get(_out) or 0)
                            except (TypeError, ValueError):
                                _in_proj = _out_proj = 0.0
                            _gain = float(_sw.get("gain") or (_in_proj - _out_proj))
                            # Severity: a 2pt gain is a real but modest nudge
                            # (0.7); a 5pt+ gain is top priority (1.0).
                            _sev = min(1.0, 0.5 + _gain / 10.0)
                            _act = make_action(
                                kind="lineup",
                                platform=plat, season=lg_season, league_id=lid,
                                league_name=league_name,
                                title=f"Start {_in_name} over {_out_name}",
                                detail=(
                                    f"{_in_name} projects {_in_proj:.1f} "
                                    f"vs {_out_name}'s {_out_proj:.1f}"
                                ),
                                href=(
                                    f"/{plat}/{int(lg_season)}/{lid}"
                                    "/waivers?tab=startsit"
                                ),
                                severity=_sev,
                                dismiss_key=f"swap:{_out}>{_in}",
                            )
                            # Board (v25): Done verification reads the lineup
                            # back; evidence carries the projection rows.
                            _act["verify"] = {"type": "swap", "in": _in, "out": _out}
                            _act["evidence"] = [
                                {"k": "Start", "v": f"{_in_name} · proj {_in_proj:.1f}"},
                                {"k": "Bench", "v": f"{_out_name} · proj {_out_proj:.1f}"},
                                {"k": "Gain", "v": f"+{_gain:.1f} projected pts"},
                            ]
                            _teams = {
                                str((nfl_players.get(_p) or {}).get("team") or "").upper()
                                for _p in (_in, _out)
                            }
                            _lock = None
                            for _g in week_games:
                                if {_g.get("home"), _g.get("away")} & _teams:
                                    _ep = _g.get("gameTime_epoch")
                                    if _ep and (_lock is None or float(_ep) < _lock):
                                        _lock = float(_ep)
                            if _lock:
                                _act["lock_epoch"] = _lock
                            actions.append(_act)
                except Exception:
                    logger.debug("[portfolio-actions] swap scan failed", exc_info=True)
            # One injury stash/drop hint per league (active-roster candidates only;
            # players already in reserve/IR do not need a stash/move-to-IR tip).
            _league_drop_act = None
            try:
                from utils.injury_plan import injury_plan, ir_capacity
                from dashboard_services.injury_return import weeks_out_for_player
                values_by_id = {
                    str(r.get("id") or ""): r
                    for r in (lctx.get("model_value_table") or [])
                    if r.get("id")
                }
                reserve_set = {str(p) for p in (viewer_roster.get("reserve") or []) if p}
                capacity = ir_capacity(
                    lctx.get("roster_positions") or [], viewer_roster.get("reserve") or [],
                    reserve_slots=(lctx.get("league") or {}).get("reserve_slots"),
                )
                for pid in [str(p) for p in (viewer_roster.get("players") or [])][:40]:
                    pl = nfl_players.get(pid) or {}
                    st = str(pl.get("injury_status") or "").strip()
                    if not st or st.lower() in ("active", "act"):
                        continue
                    plan = injury_plan(
                        status=st,
                        espn_weeks=weeks_out_for_player(pid),
                        player_value=float((values_by_id.get(pid) or {}).get("value") or 0) or None,
                        has_open_ir_slot=capacity["has_open_ir_slot"],
                        already_on_ir=pid in reserve_set,
                    )
                    if not plan or plan.get("verdict") not in ("Move to IR", "Drop candidate"):
                        continue
                    act = injury_stash_action(
                        platform=plat, season=lg_season, league_id=lid,
                        league_name=league_name,
                        player_name=pl.get("full_name") or pl.get("last_name") or "Player",
                        verdict=plan["verdict"],
                        weeks_label=plan.get("weeks_label") or "",
                        already_on_ir=pid in reserve_set,
                        ir_used=capacity["ir_used"], ir_slots=capacity["ir_slots"],
                    )
                    if act:
                        # Board (v25): an IR move verifies against the reserve
                        # list; a drop verifies against the roster itself.
                        act["verify"] = {
                            "type": "ir" if plan["verdict"] == "Move to IR" else "drop",
                            "pid": pid,
                        }
                        act["evidence"] = [
                            {"k": "Status", "v": st},
                            {"k": "Time out", "v": plan.get("weeks_label") or ""},
                            {"k": "IR room",
                             "v": f"{capacity['ir_used']} of {capacity['ir_slots']} slots used"},
                        ]
                        actions.append(act)
                        if plan["verdict"] == "Drop candidate":
                            # The waiver block below pairs this drop with
                            # the league's add (the freed spot's use).
                            _league_drop_act = act
                        break
            except Exception:
                pass

            # Waiver add only when the player clears the quality bar (startable
            # rank / roster hole) -- not merely the highest leftover FA value.
            try:
                if model_value_table:
                    from app import (
                        _league_is_redraft,
                        _waiver_rank_label_key,
                        _waiver_value_keys,
                    )
                    from utils.lineup_slots import is_superflex_lineup
                    from utils.value_helpers import format_rank_key
                    is_rd = _league_is_redraft(lctx)
                    rp = lctx.get("roster_positions") or []
                    is_sf = is_superflex_lineup(rp)
                    vf, vfb = _waiver_value_keys(lctx)
                    rank_label_key = _waiver_rank_label_key(lctx)
                    rank_key = format_rank_key(is_redraft=is_rd, is_sf=is_sf)
                    rostered_ids = {
                        str(p) for r in rosters for p in (r.get("players") or [])
                    }
                    players_index = dict(lctx.get("players_index") or {})
                    for pid in (viewer_roster.get("players") or []):
                        spid = str(pid)
                        pl = nfl_players.get(spid) or {}
                        prev = players_index.get(spid) or {}
                        players_index[spid] = {
                            **prev,
                            "position": pl.get("position") or prev.get("position"),
                            "team": pl.get("team") or prev.get("team"),
                            "injury_status": pl.get("injury_status") or prev.get("injury_status") or "",
                            "name": pl.get("full_name") or prev.get("name"),
                        }
                    inj_by_pid = {
                        str(k): str((v or {}).get("injury_status") or "")
                        for k, v in (nfl_players or {}).items()
                        if isinstance(v, dict) and v.get("injury_status")
                    }
                    # Selector opportunity inputs (item 2 parity): the same
                    # per-week maps + weekly availability the Waivers page path
                    # builds, from the one shared construction, so calendar
                    # need and the lineup-gain classification finally run on
                    # this path too.
                    _opp = {}
                    try:
                        from utils.waivers import selector_opportunity_inputs
                        _pos_by_pid = {
                            str(_k): str((_v or {}).get("position")
                                         or (_v or {}).get("pos") or "").upper()
                            for _k, _v in players_index.items()
                        }
                        _cand_rows = []
                        for _row in model_value_table:
                            if not isinstance(_row, dict):
                                continue
                            _cpid = str(_row.get("id") or "")
                            _cpos = str(_row.get("position")
                                        or _row.get("pos") or "").upper()
                            if not _cpid or _cpid in rostered_ids:
                                continue
                            if _cpos not in {"QB", "RB", "WR", "TE"}:
                                continue
                            _pos_by_pid.setdefault(_cpid, _cpos)
                            _cand_rows.append(_row)

                        def _row_value(_r):
                            try:
                                return float(_r.get(vf) or _r.get(vfb) or 0.0)
                            except (TypeError, ValueError):
                                return 0.0

                        _cand_rows.sort(key=lambda r: -_row_value(r))
                        _opp = selector_opportunity_inputs(
                            season=lg_season, week=week,
                            roster_pids=[str(p) for p in
                                         (viewer_roster.get("players") or [])],
                            candidate_pids=[str(r.get("id") or "")
                                            for r in _cand_rows[:96]],
                            positions_by_pid=_pos_by_pid,
                            players_index=players_index,
                            raw_scoring_settings=lctx.get("raw_scoring_settings") or {},
                        )
                    except Exception:
                        logger.debug("[portfolio-actions] waiver opportunity inputs failed",
                                     exc_info=True)
                        _opp = {}
                    hit = select_waiver_add(
                        model_value_table, rostered_ids,
                        value_key=vf, fallback_key=vfb,
                        rank_key=rank_key, rank_label_key=rank_label_key,
                        is_redraft=is_rd, is_sf=is_sf,
                        n_teams=len(rosters) or 12,
                        roster_players=list(viewer_roster.get("players") or []),
                        roster_positions=rp,
                        pidx=players_index,
                        min_value=_waiver_base_min,
                        injured_status_by_pid=inj_by_pid,
                        weekly_availability=_opp.get("weekly_availability") or None,
                        lineup_weekly=_opp.get("weekly_maps"),
                        weekly_pts_by_pid=_opp.get("weekly_pts_by_pid"),
                        forward_pts_by_pid=_opp.get("forward_pts_by_pid") or None,
                    )
                    if hit:
                        _wact = waiver_pickup_action(
                            platform=plat, season=lg_season, league_id=lid,
                            league_name=league_name,
                            player_name=hit["name"],
                            position=hit.get("position") or "",
                            is_redraft=is_rd,
                            pos_rank_label=hit.get("pos_rank_label") or "",
                            value=float(hit.get("value") or 0),
                            reason=hit.get("reason") or "",
                            severity=float(hit.get("severity") or 0),
                            starter_gap=float(hit.get("starter_gap") or 0),
                            pos_rank=hit.get("pos_rank"),
                            is_sf=is_sf,
                            n_teams=len(rosters) or 12,
                            player_id=str(hit.get("player_id") or ""),
                        )
                        if _wact:
                            # Board (v25): a claim cannot verify at click time
                            # (claims process later), so Done lands as marked.
                            _wact["verify"] = {
                                "type": "claim",
                                "pid": str(hit.get("player_id") or ""),
                            }
                            _wact["evidence"] = [
                                {"k": "Add",
                                 "v": f"{hit['name']} · value {float(hit.get('value') or 0):,.0f}"},
                                {"k": "Why", "v": hit.get("reason") or ""},
                            ]
                            # Waiver pairing: the shared selector already
                            # named the drop this add requires (its
                            # add_drop outcome cleared the x1.05 bar
                            # against that actual player). Surface it as
                            # the detail's final segment + an evidence
                            # row; no drop exists for open-spot/stash
                            # outcomes, and none is invented there.
                            _drop = hit.get("drop") or {}
                            _drop_name = str(_drop.get("name") or "").strip()
                            if _drop_name:
                                _drop_pos = str(
                                    _drop.get("position") or "").strip()
                                _seg = (f"Drop {_drop_name} ({_drop_pos})"
                                        if _drop_pos else f"Drop {_drop_name}")
                                _wact["detail"] = (
                                    (_wact["detail"] + " · ")
                                    if _wact.get("detail") else ""
                                ) + _seg
                                _wact["evidence"].append({
                                    "k": "Drop",
                                    "v": (f"{_drop_name} · {_drop_pos}"
                                          if _drop_pos else _drop_name),
                                })
                            # Fact chips: the segments that compose the
                            # detail, so the client can render them as
                            # chips instead of one dotted sentence. The
                            # detail string itself is unchanged.
                            if _wact.get("detail"):
                                _wact["facts"] = [
                                    _s.strip()
                                    for _s in str(_wact["detail"]).split(" · ")
                                    if _s.strip()
                                ]
                            actions.append(_wact)
                            # Drop side of the pairing: this league's
                            # injury "Drop candidate" move names the add
                            # its freed spot enables (same pass, no new
                            # engine call).
                            if _league_drop_act is not None:
                                _add_pos = str(
                                    hit.get("position") or "").strip()
                                _pair = (
                                    f"Add {hit['name']} ({_add_pos}) "
                                    "with the freed spot"
                                    if _add_pos else
                                    f"Add {hit['name']} with the freed spot"
                                )
                                _league_drop_act["detail"] = (
                                    (_league_drop_act["detail"] + " · ")
                                    if _league_drop_act.get("detail") else ""
                                ) + _pair
                                _league_drop_act.setdefault(
                                    "evidence", []).append({
                                        "k": "Add",
                                        "v": (f"{hit['name']} · {_add_pos}"
                                              if _add_pos else hit["name"]),
                                    })
            except Exception:
                logger.debug("[portfolio-actions] waiver pickup failed", exc_info=True)

            # Wasted roster capacity: IR-eligible players in active spots, recovered
            # players stuck on IR, open taxi slots (Sleeper IR/taxi; taxi is
            # dynasty-only -- leftover taxi_slots on keeper/redraft are ignored).
            try:
                settings = (
                    league_obj.get("settings")
                    or lctx.get("league_settings")
                    or {}
                )
                from utils.roster_compliance import (
                    effective_taxi_slots,
                    roster_compliance_issues,
                )
                reserve_slots = int(settings.get("reserve_slots") or 0)
                taxi_slots = effective_taxi_slots(
                    settings, league=league_obj, platform=plat,
                )
                if reserve_slots > 0 or taxi_slots > 0:
                    r_players = [str(p) for p in (viewer_roster.get("players") or [])]
                    r_info = {}
                    for pid in r_players:
                        pl = nfl_players.get(pid) or {}
                        r_info[pid] = {
                            "name": pl.get("full_name") or pl.get("last_name") or "",
                            "injury_status": pl.get("injury_status") or "",
                            "years_exp": pl.get("years_exp"),
                        }
                    r_issues = roster_compliance_issues(
                        players=r_players,
                        starters=starters,
                        reserve=[str(p) for p in (viewer_roster.get("reserve") or [])],
                        taxi=[str(p) for p in (viewer_roster.get("taxi") or [])],
                        player_info=r_info,
                        reserve_slots=reserve_slots,
                        taxi_slots=taxi_slots,
                        current_week=week,
                    )
                    r_act = roster_slot_action(
                        r_issues, platform=plat, season=lg_season,
                        league_id=lid, league_name=league_name,
                    )
                    if r_act:
                        actions.append(r_act)
            except Exception:
                logger.debug("[portfolio-actions] roster slot check failed", exc_info=True)

            # Calendar nudge: trade deadline / playoff countdown (in-season only).
            if in_season:
                try:
                    settings = (
                        league_obj.get("settings")
                        or lctx.get("league_settings")
                        or {}
                    )
                    c_act = calendar_action(
                        platform=plat, season=lg_season, league_id=lid,
                        league_name=league_name, week=week,
                        trade_deadline=int(settings.get("trade_deadline") or 0),
                        playoff_week_start=int(settings.get("playoff_week_start") or 0),
                    )
                    if c_act:
                        actions.append(c_act)
                except Exception:
                    logger.debug("[portfolio-actions] calendar nudge failed", exc_info=True)

            # Iteration end: a league counted as scanned in this pass
            # leaves a fragment, so the next pass can reuse it even if
            # the league's context has been evicted from DASHBOARD_CACHE
            # by then (see _ACTIONS_FRAGMENTS). Cold and no-roster
            # outcomes are deliberately not fragmented: they re-attempt
            # on every pass.
            if _frag_key is not None and scan_scanned > _scanned_before:
                _store_action_fragment(
                    _frag_key, actions[_frag_actions_start:],
                    board_meta.get(f"{plat}|{lid}"))

        try:
            _ranked_actions = rank_cross_league_actions(actions, limit=8)
        except Exception:
            logger.debug("[portfolio-actions] action ranking failed",
                         exc_info=True)
            _ranked_actions = actions[:8]
        payload = {
            "actions": _ranked_actions,
            "scan": _scan_payload(),
        }
        # Per-league state for EVERY scanned league, not just the ones
        # holding board actions: board_meta is computed per scanned
        # league above and until now only reached the client as group
        # headers for leagues with moves (group_actions only creates a
        # group for a league holding a top-24 action). The deep dive's
        # Brief narrates the leagues themselves (the contending /
        # bubble / rebuilding split, flagship and deep-end records),
        # so the payload ships the meta the scan already computed.
        # Optional facts (odds, points for, calendar weeks) ride only
        # when the scan actually produced them; nothing is derived
        # here that the scan did not compute. Fragment-served leagues
        # carry the same entry (fragments store board_meta whole), so
        # a fragment-built payload keeps the full league layer.
        _payload_leagues = []
        for _mk, _meta in board_meta.items():
            _mp, _, _mlid = _mk.partition("|")
            _ent = {
                "key": _mk,
                "platform": _mp,
                "league_id": _mlid,
                "league_name": _meta.get("league_name") or _mlid,
                "record_line": _meta.get("record_line") or "",
                "posture": _meta.get("posture") or "",
                "wins": _meta.get("wins"),
                "losses": _meta.get("losses"),
                "ties": _meta.get("ties") or 0,
                "rank": _meta.get("rank"),
                "team_count": _meta.get("team_count"),
            }
            if _meta.get("odds_pct") is not None:
                _ent["odds_pct"] = _meta.get("odds_pct")
            for _opt in ("points_for", "trade_deadline_week",
                         "playoff_week_start"):
                if _meta.get(_opt) is not None:
                    _ent[_opt] = _meta[_opt]
            _payload_leagues.append(_ent)
        payload["leagues"] = _payload_leagues
        # ── Board (v25): the same scan, grouped by league with global ranks,
        # done states, lock timing, a served-board snapshot for self-grading,
        # and last week's grade line. Best-effort: the digest above stands
        # even if board assembly fails.
        try:
            from dashboard_services import portfolio_board as _pb
            _groups = _pb.group_actions(
                rank_cross_league_actions(actions, limit=24), board_meta)
            _now = time.time()
            _ukey = _captured_ukey
            _done = (_pb.load_done_states(_ukey, season, week)
                     if _ukey and week else {})
            # Conservative lock fallback for lineup moves without a player
            # kickoff: the week's first Sunday kickoff, game day only.
            _fall_lock = None
            try:
                from zoneinfo import ZoneInfo as _ZI
                for _g in week_games:
                    _ep = _g.get("gameTime_epoch")
                    if _ep and datetime.fromtimestamp(
                            float(_ep), _ZI("America/New_York")).weekday() == 6:
                        if _fall_lock is None or float(_ep) < _fall_lock:
                            _fall_lock = float(_ep)
            except Exception:
                _fall_lock = None
            for _grp in _groups:
                _dc = 0
                for _mv in _grp["moves"]:
                    if (not _mv.get("lock_epoch") and _fall_lock
                            and _mv.get("kind") == "lineup"):
                        _mv["lock_epoch"] = _fall_lock
                    _mv["timing"] = _pb.timing_for(_mv, _now)
                    _st = _done.get((_grp["platform"], _grp["league_id"],
                                     _mv["move_key"]))
                    _mv["done"] = _st
                    if _st:
                        _dc += 1
                _grp["meta"]["moves_count"] = len(_grp["moves"])
                _grp["meta"]["done_count"] = _dc
            payload["board"] = {"groups": _groups, "week": week}
            payload["week"] = week
            payload["season"] = season
            if _ukey and week and _groups:
                _pb.save_board_snapshot(_ukey, season, week, _groups)
            _grade_line = ""
            if _ukey and week and week > 1:
                try:
                    _pb.grade_board_week(_ukey, season, week - 1)
                except Exception:
                    logger.debug("[portfolio-actions] lazy grade failed",
                                 exc_info=True)
                _grade_line = _pb.grade_line(
                    _pb.load_board_grade(_ukey, season, week - 1))
            payload["grade_line"] = _grade_line
        except Exception:
            logger.debug("[portfolio-actions] board assembly failed", exc_info=True)
        if _actions_key is not None:
            with _ACTIONS_LOCK:
                if scan_scanned > 0:
                    # The latest finished pass is served on later cache
                    # misses (real counts + moves found so far) instead of
                    # the hardcoded-zero deferred payload; a pass that
                    # scanned nothing never replaces it.
                    _ACTIONS_LATEST[_actions_key] = (time.time(), payload)
                if not (payload.get("scan") or {}).get("incomplete"):
                    _now_s = time.time()
                    _ACTIONS_CACHE[_actions_key] = (_now_s, payload)
                    for _k in [k for k, (ts, _p) in _ACTIONS_CACHE.items()
                               if _now_s - ts >= _ACTIONS_TTL]:
                        _ACTIONS_CACHE.pop(_k, None)
        return payload


    # Dispatch: the scan runs at most once per viewer (single-flight) on
    # a background thread, so neither a client abort at 45s nor a
    # saturated request thread can kill or starve it, and an unexpected
    # failure answers with the honest incomplete payload instead of a
    # 500 the client renders as a dead end (or, worse, an empty board).
    # The in-flight mark is cleared in the worker's finally, so a failed
    # scan can never poison later requests into deferring to a compute
    # that is not running. While a scan runs (or before the first one
    # lands), a cache miss answers with the latest finished pass payload
    # when one is fresh -- real scanned counts and the moves found so
    # far, accumulating across passes via _ACTIONS_FRAGMENTS -- and only
    # falls back to the zero-count deferred payload when no pass has
    # scanned anything yet.
    if _actions_key is None:
        try:
            return jsonify(_run_actions_compute())
        except Exception:
            logger.exception("[portfolio-actions] scan failed")
            return jsonify(_fallback_payload())
    _holder = None
    _spawn = False
    with _ACTIONS_LOCK:
        _now_c = time.time()
        _hit = _ACTIONS_CACHE.get(_actions_key)
        if _hit and _now_c - _hit[0] < _ACTIONS_TTL:
            return jsonify(_reattach_done_states(
                _hit[1], _captured_ukey, season, week))
        _started = _ACTIONS_INFLIGHT.get(_actions_key)
        if _started and _now_c - _started < 120.0:
            _holder = _ACTIONS_PENDING.get(_actions_key)
            if _holder is None:
                return jsonify(_deferred_payload())
        else:
            _ACTIONS_INFLIGHT[_actions_key] = _now_c
            _holder = {"event": threading.Event(), "payload": None,
                       "started": _now_c}
            _ACTIONS_PENDING[_actions_key] = _holder
            _spawn = True
    if _spawn:
        def _worker():
            try:
                if _ACTIONS_COMPUTE_SEM.acquire(blocking=False):
                    try:
                        _holder["payload"] = _run_actions_compute()
                    except Exception:
                        logger.exception(
                            "[portfolio-actions] background scan failed")
                    finally:
                        _ACTIONS_COMPUTE_SEM.release()
            finally:
                _holder["event"].set()
                with _ACTIONS_LOCK:
                    if _ACTIONS_PENDING.get(_actions_key) is _holder:
                        _ACTIONS_PENDING.pop(_actions_key, None)
                    if (_ACTIONS_INFLIGHT.get(_actions_key)
                            == _holder.get("started")):
                        _ACTIONS_INFLIGHT.pop(_actions_key, None)

        try:
            threading.Thread(target=_worker, daemon=True).start()
        except Exception:
            with _ACTIONS_LOCK:
                if _ACTIONS_PENDING.get(_actions_key) is _holder:
                    _ACTIONS_PENDING.pop(_actions_key, None)
                if (_ACTIONS_INFLIGHT.get(_actions_key)
                        == _holder.get("started")):
                    _ACTIONS_INFLIGHT.pop(_actions_key, None)
            try:
                return jsonify(_run_actions_compute())
            except Exception:
                logger.exception("[portfolio-actions] scan failed")
                return jsonify(_fallback_payload())
    # No request-thread wait: a scan that is still running (just spawned
    # above or already in flight) answers immediately with the latest
    # finished pass payload when one is fresh (the deferred zero-count
    # payload otherwise), and the client's retry ladder lands on the
    # complete scan once the background worker caches it. A payload that
    # somehow already landed is still served.
    if _holder is not None and _holder.get("payload") is not None:
        return jsonify(_reattach_done_states(
            _holder["payload"], _captured_ukey, season, week))
    _partial = _latest_partial_payload(_actions_key)
    if _partial is not None:
        return jsonify(_reattach_done_states(
            _partial, _captured_ukey, season, week))
    return jsonify(_deferred_payload())

# ── Board Done states (v25) ──────────────────────────────────────────────────
# Same family as /api/action-dismissals: account users persist server-side
# per (platform, league, season, week, move_key); guests get synced: False
# and keep local-only behavior. A Done POST verifies against the league's
# cached roster read-back where the move carries a checkable spec.

def _move_done_scope(body):
    """Validate a move-done scope with the dismissal validator (which
    returns a tuple); normalize to a dict or None."""
    cleaned = _clean_dismissal_scope({
        "platform": (body or {}).get("platform"),
        "league_id": (body or {}).get("league_id"),
        "season": (body or {}).get("season"),
        "week": (body or {}).get("week"),
        "day": (body or {}).get("day") or "2026-01-01",
        "item_key": (body or {}).get("move_key"),
    })
    if cleaned is None or len(cleaned) != 6:
        return None
    platform, league_id, season, week, _day, item_key = cleaned
    return {"platform": platform, "league_id": league_id, "season": season,
            "week": week, "item_key": item_key}


def _done_time_label(iso_text):
    try:
        from zoneinfo import ZoneInfo
        dt = datetime.fromisoformat(str(iso_text))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(ZoneInfo("America/New_York")).strftime("%-I:%M %p")
    except Exception:
        return ""


@user_pages_bp.route("/api/portfolio-move-done", methods=["GET", "POST", "DELETE"])
@limiter.limit("120/minute")
def api_portfolio_move_done():
    from dashboard_services import portfolio_board as _pb
    user_key = _user_key()
    if request.method == "GET":
        if not user_key:
            return jsonify({"ok": True, "synced": False, "states": []})
        try:
            season = int(request.args.get("season") or 0)
            week = int(request.args.get("week") or 0)
        except (TypeError, ValueError):
            return jsonify({"ok": False, "error": "invalid_scope"}), 400
        states = _pb.load_done_states(user_key, season, week)
        out = []
        for (plat, lid, mk), st in states.items():
            out.append({
                "platform": plat, "league_id": lid, "move_key": mk,
                "verified": st.get("verified"), "label": st.get("label"),
                "done_at": st.get("done_at"),
                "time_label": _done_time_label(st.get("done_at")),
            })
        return jsonify({"ok": True, "synced": True, "states": out})
    if not user_key:
        # Guests keep local-only behavior, like the dismissal store.
        return jsonify({"ok": True, "synced": False})
    body = request.get_json(silent=True) or {}
    scope = _move_done_scope(body)
    if scope is None:
        return jsonify({"ok": False, "error": "invalid_scope"}), 400
    plat = scope["platform"]
    lid = scope["league_id"]
    season = scope["season"]
    week = scope["week"]
    move_key_v = scope["item_key"]
    if request.method == "DELETE":
        ok = _pb.clear_done(user_key, plat, lid, season, week, move_key_v)
        return jsonify({"ok": ok, "synced": True})
    # POST: verify the claim against the cached roster read-back, then store.
    action = body.get("action") or {}
    roster_view = None
    try:
        lctx = get_league_ctx_from_cache(plat, lid, season, allow_build=False) or {}
        rosters = lctx.get("rosters") or []
        viewer_roster = None
        account_id = session.get("account_id")
        if account_id:
            try:
                from dashboard_services.accounts import resolve_account_viewer_for_league
                av = resolve_account_viewer_for_league(
                    account_id, plat, lid, season,
                    lctx.get("users") or [], rosters,
                )
                rid = str((av or {}).get("viewer_roster_id") or "")
                if rid:
                    viewer_roster = next(
                        (r for r in rosters
                         if str(r.get("roster_id") or "") == rid), None)
            except Exception:
                viewer_roster = None
        if viewer_roster is None:
            from utils.scorezone_user import match_viewer_roster
            viewer_roster = match_viewer_roster(
                rosters,
                owner_id=sleeper_owner_id_for_account(
                    account_id, session.get("viewer_user_id"), plat),
            )
        if viewer_roster:
            roster_view = {
                "players": [str(p) for p in (viewer_roster.get("players") or []) if p],
                "starters": [str(p) for p in (viewer_roster.get("starters") or []) if p],
                "reserve": [str(p) for p in (viewer_roster.get("reserve") or []) if p],
            }
    except Exception:
        logger.debug("[portfolio-move-done] roster read-back failed", exc_info=True)
    result = _pb.verify_done(
        {"platform": plat, "verify": (action or {}).get("verify") or {}},
        roster_view,
    )
    ok = _pb.set_done(user_key, plat, lid, season, week, move_key_v,
                      bool(result["verified"]), str(result["label"]))
    return jsonify({
        "ok": ok, "synced": True,
        "verified": bool(result["verified"]),
        "label": str(result["label"]),
        "time_label": _done_time_label(datetime.now(timezone.utc).isoformat()),
    })


@user_pages_bp.route("/api/portfolio-deep-dive", methods=["POST"])
@limiter.limit("30/minute")
def api_portfolio_deep_dive():
    """The Front Office full report, composed as The Brief.

    A short prose brief over the same scan the board renders:
    deterministic templates only, every sentence conditional on its
    facts existing in the payload. The shape:

    - a kicker ("Week N · Portfolio Brief"; the week segment drops
      when the payload carries no week),
    - a standfirst splitting the scanned leagues by posture, over
      the payload's ``leagues`` array (one entry per scanned league,
      not just the leagues holding board moves),
    - a portfolio paragraph: the flagship league (best record), the
      deep end (worst record), the featured trade-deadline league,
      and how many leagues need nothing,
    - a moves paragraph: the biggest projected gain on the board,
      then the waiver claims as one clock sentence,
    - a clock paragraph: fills, anytime moves, locks and already
      locked moves as clauses, with the calendar facts folded in,
    - end matter: the done moves by short title, then the coverage
      line for the leagues that need nothing.

    Calendar nudges (kind="calendar": trade deadline, playoff
    countdown) are dates to be aware of, not moves to execute: they
    are never counted or numbered, their facts live in the portfolio
    and clock paragraphs, and a nudge already marked done drops out
    entirely. There is deliberately no cross-league "conflict"
    callout: separate leagues have independent wires, FAAB and
    rosters, so the same player claimed in two leagues (or claimed
    in one and dropped in another) is stated plainly, never as a
    collision to prioritize. The report text carries no em dashes
    (standing UI rule). Paragraphs return as span arrays
    ({text, strong}); strong marks player names and digit-bearing
    tokens only, which is the typography the client renders. A
    partial scan is refused with a retryable error instead of being
    narrated as the full picture, because the counts of a
    half-finished scan read as final."""
    resp = api_portfolio_actions()
    if isinstance(resp, tuple):
        resp = resp[0]
    if getattr(resp, "status_code", 200) != 200:
        return resp
    data = resp.get_json() or {}
    if (data.get("scan") or {}).get("incomplete"):
        return jsonify({"ok": False, "error": "scan_incomplete"})
    groups = ((data.get("board") or {}).get("groups")) or []
    if not groups:
        return jsonify({"ok": False, "error": "no_board"})
    try:
        week = int(data.get("week")
                   or (data.get("board") or {}).get("week") or 0)
    except (TypeError, ValueError):
        week = 0

    def _evidence_value(mv, key):
        for row in mv.get("evidence") or []:
            if isinstance(row, dict) and str(row.get("k") or "") == key:
                v = str(row.get("v") or "").strip()
                if v:
                    return v
        return ""

    # When a move carries no detail, its reason is the SINGLE strongest
    # evidence row, not every row stacked: the swap's gain, the claim's
    # why, the injury's time out, in that spirit.
    _EVIDENCE_PRIORITY = ("Gain", "Matchup", "Trend", "Why",
                          "Time out", "Status", "IR room", "Start",
                          "Bench", "Add", "Drop")

    def _norm_name(name):
        """Player-name key for cross-referencing moves: lowercase,
        punctuation stripped, whitespace collapsed. Used for the
        restatement check; a name that does not normalize to at
        least two tokens never matches anything."""
        return " ".join(
            re.sub(r"[^a-z0-9 ]", " ", str(name or "").lower()).split())

    def _title_player_names(mv):
        """The player names a title carries, parsed from the board's
        own title shapes only: 'Start A over B', and 'Add X (POS)' /
        'Stash X (POS)'. Anything else yields no names, so the
        restatement check stays silent rather than guessing."""
        title = str(mv.get("title") or "").strip()
        m = re.match(r"^Start\s+(.+?)\s+over\s+(.+)$", title)
        if m:
            return [m.group(1).strip(), m.group(2).strip()]
        m = re.match(r"^(?:Add|Stash)\s+(.+?)(?:\s*\([^)]*\))?$", title)
        if m:
            return [m.group(1).strip()]
        return []

    _NUM_RE = re.compile(r"\d+(?:\.\d+)?")

    def _restates_title(mv, text):
        """True when a candidate reason merely repeats the title's own
        claim. Explicit rule, no similarity scoring: the text names
        every player the title names AND every number in it already
        appears in one of the move's Start/Bench evidence rows (i.e.
        it is the raw projection line restated). A text carrying a
        number found nowhere in those rows, or naming no title player,
        adds a fact and is kept."""
        names = _title_player_names(mv)
        if not names:
            return False
        low = _norm_name(text)
        if not all(_norm_name(n) and _norm_name(n) in low
                   for n in names):
            return False
        proj_nums = set()
        for key in ("Start", "Bench"):
            proj_nums.update(
                _NUM_RE.findall(_evidence_value(mv, key)))
        nums = _NUM_RE.findall(text)
        return bool(nums) and all(n in proj_nums for n in nums)

    def _move_reason(mv):
        """The why for one move, from its own fields only, and never a
        restatement of the title: when the detail only repeats the
        title's claim (the swap shape: title 'Start A over B', detail
        'A projects X vs B's Y'), the reason compresses to the one
        added fact, the projected delta from the Gain row ('Worth
        +5.1 projected pts in the swap.'). Otherwise the detail stands
        (with the gain or the named drop folded in when it omits
        them), else the single strongest evidence row that is not
        itself a restatement. A move whose detail was dropped for
        restating the title falls back to the consequence the title
        states, so a bye or an out starter still carries its reason.
        No invented numbers."""
        detail = str(mv.get("detail") or "").strip()
        if detail and _restates_title(mv, detail):
            gain = _evidence_value(mv, "Gain")
            if gain:
                return f"Worth {gain} in the swap."
            detail = ""
        if detail:
            gain = _evidence_value(mv, "Gain")
            if (gain and "gain" not in detail.lower()
                    and gain.split()[0] not in detail):
                detail = f"{detail} ({gain})"
            if mv.get("kind") == "waiver":
                # A claim's step must name its drop even when the
                # detail forgot to (the drop used to ride a separate
                # roundup line; the step carries it now).
                drop = _evidence_value(mv, "Drop")
                if drop:
                    name = drop.split(" · ")[0].strip()
                    if name and name not in detail:
                        if " · " in drop:
                            nm, pos = drop.split(" · ", 1)
                            detail += f" · Drop {nm} ({pos})"
                        else:
                            detail += f" · Drop {drop}"
            return detail
        for key in _EVIDENCE_PRIORITY:
            v = _evidence_value(mv, key)
            if v and not _restates_title(mv, v):
                return v
        ev = [str(r.get("v") or "").strip()
              for r in (mv.get("evidence") or []) if isinstance(r, dict)]
        ev = [v for v in ev if v and not _restates_title(mv, v)]
        if ev:
            return ev[0]
        low = str(mv.get("title") or "").lower()
        if "on bye" in low:
            return ("The starter is on a bye this week, so the slot "
                    "needs a fill.")
        if " is out" in low:
            return ("The starter is out, so the slot needs a "
                    "replacement.")
        if low.startswith("empty starting slot"):
            return ("A starting slot is empty and scores nothing until "
                    "it is filled.")
        return ""

    # ── The Brief ────────────────────────────────────────────────────
    # Posture buckets over the Teams window / redraft labels the
    # cards and board headers carry: Contend and Rebuild name
    # themselves; everything else is the bubble.
    _CONTENDING = {"Contend", "Powerhouse", "Contender", "Win-Now",
                   "Contender Window"}
    _REBUILDING = {"Long Shot", "Retooling", "Rebuilding",
                   "Full Rebuild", "Rebuild"}

    def _posture_bucket(label):
        if label in _CONTENDING:
            return "Contend"
        if label in _REBUILDING:
            return "Rebuild"
        return None

    _WORDS = ["zero", "one", "two", "three", "four", "five", "six",
              "seven", "eight", "nine", "ten", "eleven", "twelve",
              "thirteen", "fourteen", "fifteen", "sixteen",
              "seventeen", "eighteen", "nineteen", "twenty"]

    def _word(n, cap=False):
        try:
            n = int(n)
        except (TypeError, ValueError):
            return str(n)
        w = _WORDS[n] if 0 <= n <= 20 else str(n)
        return w.capitalize() if cap else w

    def _ordinal(n):
        suf = ("th" if 10 <= n % 100 <= 20 else
               {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th"))
        return f"{n}{suf}"

    _SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}

    def _surname(name):
        toks = [t for t in re.split(r"\s+", str(name or "").strip())
                if t]
        if not toks:
            return ""
        last = toks[-1].strip(".,")
        if last.lower() in _SUFFIXES and len(toks) > 1:
            last = toks[-2].strip(".,")
        return last

    def _short_title(mv):
        """A done move's short title for the end matter: a swap
        compresses to surnames ('Bowers over Kincaid'); anything
        whose shape is not certainly a swap keeps its full title
        rather than a guessed compression."""
        names = _title_player_names(mv)
        title = str(mv.get("title") or "").strip()
        if len(names) == 2 and title.lower().startswith("start "):
            a, b = _surname(names[0]), _surname(names[1])
            if a and b:
                return f"{a} over {b}"
        return title

    def _entry_from_meta(key, name, meta):
        """One league's state, normalized from either the payload's
        leagues array or a board group's meta (same field names).
        Structured record fields ride the payload; when an older
        payload lacks them, the record is recovered from the record
        line's own segments instead of being invented."""
        meta = meta or {}
        ent = {
            "key": key, "league_name": name or "",
            "record_line": str(meta.get("record_line") or ""),
            "posture": str(meta.get("posture") or ""),
            "wins": meta.get("wins"), "losses": meta.get("losses"),
            "ties": meta.get("ties") or 0, "rank": meta.get("rank"),
            "team_count": meta.get("team_count"),
            "points_for": meta.get("points_for"),
        }
        if ent["wins"] is None and ent["record_line"]:
            m = re.match(
                r"^(\d+)-(\d+)(?:-(\d+))?"
                r"(?:\s*·\s*(\d+)(?:st|nd|rd|th) of (\d+))?",
                ent["record_line"])
            if m:
                ent["wins"] = int(m.group(1))
                ent["losses"] = int(m.group(2))
                ent["ties"] = int(m.group(3) or 0)
                if m.group(4):
                    ent["rank"] = int(m.group(4))
                if m.group(5):
                    ent["team_count"] = int(m.group(5))
        return ent

    entries = []
    raw_leagues = data.get("leagues")
    if isinstance(raw_leagues, list) and raw_leagues:
        for lg in raw_leagues:
            if isinstance(lg, dict):
                entries.append(_entry_from_meta(
                    str(lg.get("key") or ""), lg.get("league_name"),
                    lg))
    if not entries:
        # A payload assembled before the leagues array existed (an
        # in-flight cached scan): the league layer degrades to the
        # leagues holding board moves, whose group meta carries the
        # same fields.
        for g in groups:
            entries.append(_entry_from_meta(
                str(g.get("key") or ""), g.get("league_name"),
                g.get("meta")))
    by_key = {e["key"]: e for e in entries if e["key"]}

    def _record_str(ent):
        if ent["wins"] is not None and ent["losses"] is not None:
            rec = f"{ent['wins']}-{ent['losses']}"
            if ent["ties"]:
                rec += f"-{ent['ties']}"
            return rec
        if ent["record_line"]:
            return ent["record_line"].split(" · ")[0]
        return ""

    def _standing_str(ent):
        rk, tc = ent["rank"], ent["team_count"]
        if rk:
            if tc and tc > 1 and rk == tc:
                return "last"
            return _ordinal(rk)
        return ""

    ordered = []
    for g in groups:
        for mv in g.get("moves") or []:
            ordered.append((g, mv))
    ordered.sort(key=lambda gm: (gm[1].get("rank") or 99))
    open_moves, done_moves, reminders = [], [], []
    for g, mv in ordered:
        if mv.get("kind") == "calendar":
            if not mv.get("done"):
                reminders.append((g, mv))
            continue
        (done_moves if mv.get("done") else open_moves).append((g, mv))

    def _mv_key(g, mv):
        return (g.get("key"), mv.get("move_key") or mv.get("rank"))

    _open_by_key, _done_by_key = {}, {}
    for g, mv in open_moves:
        _open_by_key[g.get("key")] = _open_by_key.get(g.get("key"), 0) + 1
    for g, mv in done_moves:
        _done_by_key[g.get("key")] = _done_by_key.get(g.get("key"), 0) + 1

    def _moves_clause(ent):
        oc = _open_by_key.get(ent["key"], 0)
        dc = _done_by_key.get(ent["key"], 0)
        if oc:
            return (f", with {_word(oc)} move"
                    + ("s" if oc != 1 else "") + " open")
        if dc:
            return (f", and its {_word(dc)} move"
                    + ("s" if dc != 1 else "")
                    + (" is" if dc == 1 else " are") + " done")
        return ""

    def _deadline_of(mv):
        """(due_week, gap) for a trade-deadline calendar nudge, else
        None. due_week rides the action itself (calendar_action
        stamps it for the board's due pill); the gap derives from it
        and the payload's week, never from label text."""
        try:
            due = int(mv.get("due_week"))
        except (TypeError, ValueError):
            return None
        if not due:
            return None
        try:
            gap = int(mv.get("due_in_weeks"))
        except (TypeError, ValueError):
            gap = (due - week) if week else None
        return due, gap

    # Standfirst: the posture split over every league with a posture.
    postured = [e for e in entries if e["posture"]]
    n_contend = sum(1 for e in postured
                    if _posture_bucket(e["posture"]) == "Contend")
    n_rebuild = sum(1 for e in postured
                    if _posture_bucket(e["posture"]) == "Rebuild")
    n_bubble = len(postured) - n_contend - n_rebuild
    standfirst = ""
    if postured:
        total = len(postured)
        parts = []
        if n_contend:
            parts.append((n_contend, "contending"))
        if n_bubble:
            parts.append((n_bubble, "on the bubble"))
        if n_rebuild:
            parts.append((n_rebuild, "rebuilding"))
        lw = "league" if total == 1 else "leagues"
        if len(parts) == 1:
            standfirst = (f"{_word(total, cap=True)} {lw}, all "
                          f"{parts[0][1]}.")
        else:
            joined = ", ".join(f"{n} {lbl}" for n, lbl in parts[:-1])
            joined += f" and {parts[-1][0]} {parts[-1][1]}"
            standfirst = (f"{_word(total, cap=True)} {lw}, split "
                          f"{joined}.")

    # Coverage: leagues the scan actually scanned that produced no
    # board group (a group exists for every league holding a top-24
    # action or reminder) needed nothing. Skipped leagues were never
    # checked and are never counted as quiet; without a scanned
    # tally the number cannot be derived honestly and no line is
    # emitted.
    scan = data.get("scan") or {}
    n_quiet = None
    if scan.get("scanned") is not None:
        try:
            n_scanned = int(scan.get("scanned") or 0)
        except (TypeError, ValueError):
            n_scanned = 0
        covered_keys = {g.get("key") for g in groups}
        if n_scanned > len(covered_keys):
            n_quiet = n_scanned - len(covered_keys)

    # Paragraph 1: the portfolio itself.
    p1 = []
    elig = [e for e in entries
            if e["wins"] is not None and e["losses"] is not None]
    if elig:
        flag = sorted(elig, key=lambda e: (
            -e["wins"], e["losses"], e["points_for"] is None,
            -(e["points_for"] or 0), e["league_name"]))[0]
        deep = sorted(elig, key=lambda e: (
            e["wins"], -e["losses"], e["points_for"] is None,
            e["points_for"] or 0, e["league_name"]))[0]
        rec = _record_str(flag)
        s = f"{flag['league_name']} is the flagship"
        if rec:
            s += f": {rec}"
            st = _standing_str(flag)
            if st:
                s += f" and {st}"
        p1.append(s + _moves_clause(flag) + ".")
        if deep["key"] != flag["key"]:
            rec = _record_str(deep)
            s = f"{deep['league_name']} is the deep end"
            if rec:
                s += f" at {rec}"
                st = _standing_str(deep)
                if st:
                    s += f" and {st}"
            if (_posture_bucket(deep["posture"]) == "Rebuild"
                    and deep["rank"] and deep["team_count"]
                    and deep["rank"] == deep["team_count"]):
                s += ", already playing for next year"
            p1.append(s + _moves_clause(deep) + ".")
    deadlines = []
    for g, mv in reminders:
        dl = _deadline_of(mv)
        if dl:
            deadlines.append((g, mv, dl[0], dl[1]))
    if deadlines:
        deadlines.sort(key=lambda d: (d[3] if d[3] is not None else 99,
                                      d[2]))
        g, mv, due, gap = deadlines[0]
        ent = by_key.get(g.get("key"))
        lname = ((ent or {}).get("league_name")
                 or g.get("league_name") or "")
        gap_phrase = {0: "this week", 1: "a week out",
                      2: "two weeks out"}.get(
                          gap, f"{gap} weeks out" if gap is not None
                          else "")
        rec = _record_str(ent) if ent else ""
        if rec:
            s = f"{lname} is {rec}"
            if ent.get("posture"):
                s += f" and {ent['posture'].lower()}"
            if gap == 0:
                s += (f", and its trade deadline is this week, "
                      f"Week {due}")
            else:
                s += f", and its trade deadline lands in Week {due}"
                if gap_phrase:
                    s += f", {gap_phrase}"
            s += (", so it is the one league where the calendar "
                  "matters more than the wire.")
        elif gap == 0:
            s = (f"The trade deadline in {lname} is this week, "
                 f"Week {due}, so it is the one league where the "
                 "calendar matters more than the wire.")
        else:
            s = f"The trade deadline in {lname} lands in Week {due}"
            if gap_phrase:
                s += f", {gap_phrase}"
            s += (", so it is the one league where the calendar "
                  "matters more than the wire.")
        p1.append(s)
    if n_quiet:
        p1.append(f"{_word(n_quiet, cap=True)} league"
                  + ("s" if n_quiet != 1 else "")
                  + (" need" if n_quiet != 1 else " needs")
                  + " nothing this week.")

    # Paragraph 2: the moves, biggest gain first, claims as one
    # clock sentence.
    covered = set()
    p2 = []
    best_gain = None
    for g, mv in open_moves:
        if mv.get("kind") != "lineup":
            continue
        names = _title_player_names(mv)
        gain = _evidence_value(mv, "Gain")
        if len(names) != 2 or not gain:
            continue
        mnum = _NUM_RE.search(gain)
        if not mnum:
            continue
        try:
            gv = float(mnum.group(0))
        except ValueError:
            continue
        if best_gain is None or gv > best_gain[0]:
            best_gain = (gv, g, mv, gain, names)
    lead = None
    if best_gain:
        _, g, mv, gain, names = best_gain
        s = (f"The biggest gain on the board is the "
             f"{_surname(names[0])} swap in {g.get('league_name')}: "
             f"start {names[0]} over {names[1]} for {gain}")
        label = str((mv.get("timing") or {}).get("label") or "")
        if label.startswith("Locks "):
            s += f", and it locks at {label[len('Locks '):]}"
        p2.append(s + ".")
        covered.add(_mv_key(g, mv))
        lead = (g, mv)
    claims = [(g, mv) for g, mv in open_moves
              if mv.get("kind") == "waiver"]
    if not lead:
        for g, mv in open_moves:
            if mv.get("kind") == "waiver":
                continue
            s = (f"The first move on the board is in "
                 f"{g.get('league_name')}: {mv.get('title')}."
                 )
            reason = _move_reason(mv)
            if reason:
                s += " " + reason.rstrip(".") + "."
            p2.append(s)
            covered.add(_mv_key(g, mv))
            lead = (g, mv)
            break
    if claims:
        segs = []
        prev_player = None
        for g, mv in claims:
            names = _title_player_names(mv)
            player = names[0] if names else ""
            if not player:
                add = _evidence_value(mv, "Add")
                player = add.split(" · ")[0].strip() if add else ""
            drop = _evidence_value(mv, "Drop")
            drop_name = drop.split(" · ")[0].strip() if drop else ""
            if player and player == prev_player:
                seg = f"in {g.get('league_name')}"
            elif player:
                seg = f"add {player} in {g.get('league_name')}"
            else:
                seg = f"a claim in {g.get('league_name')}"
            if drop_name:
                seg += f" (dropping {drop_name})"
            segs.append(seg)
            if player:
                prev_player = player
            covered.add(_mv_key(g, mv))
        if len(segs) == 1:
            joined = segs[0]
        elif len(segs) == 2:
            joined = f"{segs[0]} and {segs[1]}"
        else:
            joined = ", ".join(segs[:-1]) + f" and {segs[-1]}"
        n = len(claims)
        tail = (", before waivers run." if n == 1
                else ", both before waivers run." if n == 2
                else ", all before waivers run.")
        p2.append(f"{_word(n, cap=True)} claim"
                  + ("s" if n != 1 else "")
                  + (" runs" if n == 1 else " run")
                  + f" against the waiver clock: {joined}{tail}")

    # Paragraph 3: the clock. Every remaining open move lands in
    # exactly one clause; calendar facts close the paragraph.
    def _move_noun(mv):
        """A short noun phrase for one move, from its title's own
        shape ('the Waddle swap', 'the Fairbairn fill'); unknown
        shapes fall back to the title itself, first letter lowered."""
        title = str(mv.get("title") or "").strip()
        names = _title_player_names(mv)
        if len(names) == 2:
            return f"the {_surname(names[0])} swap"
        m = re.match(r"^(.+?) on bye", title)
        if m:
            return f"the {_surname(m.group(1))} fill"
        m = re.match(r"^(.+?) is out", title)
        if m:
            return f"the {_surname(m.group(1))} replacement"
        m = re.match(r"^Move (.+?) to IR", title)
        if m:
            return f"the {_surname(m.group(1))} IR move"
        if names:
            return f"the {_surname(names[0])} claim"
        low = title[:1].lower() + title[1:] if title else ""
        return f"the {low}" if low else "the move"

    clauses, cal_clauses = [], []
    for g, mv in open_moves:
        if _mv_key(g, mv) in covered:
            continue
        league = str(g.get("league_name") or "")
        title = str(mv.get("title") or "").strip()
        timing = mv.get("timing") or {}
        label = str(timing.get("label") or "")
        if mv.get("kind") == "calendar":
            # Unreachable today (calendar moves partition into
            # reminders before this loop); kept so a future kind
            # change degrades into a clause instead of vanishing.
            low = title[:1].lower() + title[1:] if title else ""
            if low:
                cal_clauses.append(f"in {league}, {low}")
            continue
        if timing.get("locked"):
            clauses.append(
                f"{_move_noun(mv)} in {league} already locked")
            continue
        m = re.match(r"^(.+?) on bye", title)
        if m:
            clauses.append(
                f"the {_surname(m.group(1))} bye in {league} needs "
                "a fill before Sunday")
            continue
        m = re.match(r"^(.+?) is out", title)
        if m:
            clauses.append(
                f"the {_surname(m.group(1))} absence in {league} "
                "needs a fill before Sunday")
            continue
        if title.lower().startswith("empty starting slot"):
            clauses.append(
                f"an empty starting slot in {league} needs a fill "
                "before Sunday")
            continue
        m = re.match(r"^Move (.+?) to IR", title)
        if m:
            clauses.append(
                f"the {_surname(m.group(1))} IR move in {league} can "
                "happen anytime")
            continue
        if label.startswith("Locks "):
            clauses.append(
                f"{_move_noun(mv)} in {league} locks at "
                f"{label[len('Locks '):]}")
            continue
        if mv.get("kind") in ("injury", "roster", "trade") \
                or label == "Do now":
            clauses.append(
                f"{_move_noun(mv)} in {league} can happen anytime")
            continue
        clauses.append(
            f"{_move_noun(mv)} in {league} comes due before Sunday")
    for g, mv in reminders:
        league = str(g.get("league_name") or "")
        title = str(mv.get("title") or "").strip()
        dl = _deadline_of(mv)
        if dl:
            due, gap = dl
            if gap == 0:
                cal_clauses.append(
                    f"the {league} trade deadline lands this week, "
                    f"Week {due}, so deals there get lined up now")
            elif gap is not None:
                cal_clauses.append(
                    f"the {league} trade deadline lands in {gap} "
                    f"week" + ("s" if gap != 1 else "")
                    + f", Week {due}, so deals there get lined up "
                    "now")
            else:
                cal_clauses.append(
                    f"the {league} trade deadline lands in Week "
                    f"{due}, so deals there get lined up now")
        elif title.startswith("Playoffs "):
            rest = title[len("Playoffs "):]
            rest = rest[:1].lower() + rest[1:]
            cal_clauses.append(f"the playoffs in {league} {rest}")
        else:
            low = title[:1].lower() + title[1:] if title else ""
            if low:
                cal_clauses.append(f"in {league}, {low}")
    p3 = []
    all_clauses = clauses + cal_clauses
    if all_clauses:
        if len(all_clauses) == 1:
            body = all_clauses[0]
        else:
            body = (", ".join(all_clauses[:-1])
                    + f", and {all_clauses[-1]}")
        p3.append(body[:1].upper() + body[1:] + ".")

    # End matter: done moves by short title, then coverage.
    end_lines = []
    if done_moves:
        parts = [f"{_short_title(mv)} in {g.get('league_name')}"
                 for g, mv in done_moves]
        if len(parts) == 1:
            joined = parts[0]
        elif len(parts) == 2:
            joined = f"{parts[0]} and {parts[1]}"
        else:
            joined = ", ".join(parts[:-1]) + f" and {parts[-1]}"
        n = len(done_moves)
        end_lines.append(
            f"{_word(n, cap=True)} move" + ("s" if n != 1 else "")
            + (" are" if n != 1 else " is")
            + f" already done: {joined}.")
    if n_quiet:
        end_lines.append(
            f"The other {n_quiet} league"
            + ("s" if n_quiet != 1 else "")
            + (" need" if n_quiet != 1 else " needs")
            + " nothing this week.")

    # Span splitting for the client's typography: strong marks
    # player names (full names and surnames, collected from the
    # moves themselves) and digit-bearing tokens only. League
    # names, spelled-out numbers and posture words stay plain.
    _names = set()
    for g in groups:
        for mv in g.get("moves") or []:
            for nm in _title_player_names(mv):
                _names.add(nm)
            for key in ("Add", "Drop", "Start", "Bench"):
                v = _evidence_value(mv, key)
                if v:
                    _names.add(v.split(" · ")[0].strip())
            t = str(mv.get("title") or "")
            for pat in (r"^(.+?) on bye", r"^(.+?) is out",
                        r"^Move (.+?) to IR"):
                mm = re.match(pat, t)
                if mm:
                    _names.add(mm.group(1).strip())
    _name_list = [n for n in _names if n]
    for nm in list(_name_list):
        sn = _surname(nm)
        if sn and sn not in _name_list:
            _name_list.append(sn)

    def _spans(text):
        if not text:
            return []
        ranges = []
        for nm in sorted(_name_list, key=len, reverse=True):
            for m in re.finditer(
                    r"(?<![A-Za-z])" + re.escape(nm) + r"(?![A-Za-z])",
                    text):
                ranges.append((m.start(), m.end()))
        for m in re.finditer(r"\S*\d\S*", text):
            s0, e0 = m.start(), m.end()
            while s0 < e0 and text[s0] in "([":
                s0 += 1
            while e0 > s0 and text[e0 - 1] in ",.;":
                e0 -= 1
            if s0 < e0:
                ranges.append((s0, e0))
        ranges.sort()
        merged = []
        for s0, e0 in ranges:
            if merged and s0 < merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], e0)
            else:
                merged.append([s0, e0])
        out, pos = [], 0
        for s0, e0 in merged:
            if s0 > pos:
                out.append({"text": text[pos:s0], "strong": False})
            out.append({"text": text[s0:e0], "strong": True})
            pos = e0
        if pos < len(text):
            out.append({"text": text[pos:], "strong": False})
        return out

    paragraphs = []
    for sents in (p1, p2, p3):
        if sents:
            paragraphs.append(" ".join(sents))
    brief = {
        "kicker": (f"Week {week} · Portfolio Brief" if week
                   else "Portfolio Brief"),
        "standfirst": _spans(standfirst),
        "paragraphs": [_spans(p) for p in paragraphs],
        "end_matter": [_spans(line) for line in end_lines],
    }
    return jsonify({
        "ok": True, "brief": brief,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    })


# ── Portfolio insights: Rooting Guide + Bye Watch ────────────────────────────
# Two endpoints share one preamble (identity, NFL state, the viewer's league
# list, players + model values) and one per-league resolver (cache-only ctx +
# viewer roster):
#
# - /api/portfolio-bye serves Bye Watch alone. Bye needs NO provider matchup
#   data -- only the light resolver's provider-layer rosters, the viewer's
#   starters and bye weeks -- so it answers immediately instead of waiting
#   behind the rooting scan.
# - /api/portfolio-insights serves the Rooting Guide. Its per-league work
#   (a provider matchup fetch per league) runs on a bounded background
#   thread, never on the request thread, and the assembled response is
#   cached per viewer/season/week for a short TTL so repeat loads in the
#   window are free. The payload keeps an empty bye block for shape
#   compatibility; the client reads byes from the bye endpoint now.
_INSIGHTS_CACHE: dict = {}
_INSIGHTS_CACHE_LOCK = threading.Lock()
_INSIGHTS_TTL = 60.0  # seconds
# Single-flight for the insights compute, mirroring _ACTIONS_INFLIGHT:
# cache key -> monotonic start of the compute currently running for that
# viewer. Concurrent requests for the same viewer defer to it instead of
# piling a duplicate scan onto the box; a mark older than the stale window
# is treated as abandoned and a new compute may take over. The background
# worker clears its mark in a finally (see _pf_kick_background).
_INSIGHTS_INFLIGHT: dict = {}
_INSIGHTS_INFLIGHT_LOCK = threading.Lock()
_INSIGHTS_INFLIGHT_STALE_S = 120.0  # seconds

# /api/portfolio-actions response cache + single-flight (2026-10-26).
# The warm scan measured 12.9s for just the 4 Sleeper leagues locally;
# across all 11 linked leagues in production it overruns the client's
# fetch timeout, so the moves board showed "Could not load your moves"
# on every load and every retry rescanned from zero (no cache existed).
# Complete scans are cached per viewer for the insights TTL; while one
# request computes, concurrent requests for the same viewer answer with
# the honest incomplete payload immediately so the client's retry
# ladder lands on the finished cache instead of piling a duplicate
# scan onto the box. An in-flight mark older than 120s is treated as
# abandoned (its request died) and a new compute may take over.
_ACTIONS_CACHE: dict = {}
_ACTIONS_INFLIGHT: dict = {}
_ACTIONS_LOCK = threading.Lock()
_ACTIONS_TTL = 60.0  # seconds
# Pending background scans, keyed like _ACTIONS_CACHE: {"event", "payload",
# "started"}. The scan runs on a daemon thread (see api_portfolio_actions)
# so a client-side abort or a saturated request thread can no longer kill
# or starve it; the compute is single-flight per viewer and globally
# capped, and an over-cap kick is dropped (marks cleared) so the client's
# next retry re-kicks it.
_ACTIONS_PENDING: dict = {}
_ACTIONS_COMPUTE_SEM = threading.BoundedSemaphore(2)
# Legacy bounded grace, retained only so existing references keep working:
# the request thread no longer waits on the background scan at all (it
# answers with the deferred incomplete payload immediately and the
# client's retry ladder lands on the finished cache).
_ACTIONS_SYNC_WAIT = 8.0
try:
    _ACTIONS_SYNC_WAIT = float(
        os.environ.get("PORTFOLIO_ACTIONS_SYNC_WAIT_SECONDS") or 8.0)
except Exception:
    pass

# Per-league scan fragments + latest-partial payloads (2026-10-27).
#
# A scan pass can only count a league whose context is readable at that
# instant (allow_build=False), and the process-local DASHBOARD_CACHE
# holds at most DASHBOARD_CACHE_MAX (default 4) league contexts, so for
# a viewer with more leagues than that the warm carousel keeps evicting
# contexts other leagues' builds just wrote: no single pass could ever
# see every league, the complete-only cache never landed, and the board
# showed the deferred "Scanned 0 of N" payload forever. Fragments make
# progress accumulate across passes instead: each league a pass manages
# to scan leaves its raw actions + board meta here for
# _ACTIONS_FRAGMENT_TTL seconds, later passes reuse fresh fragments
# instead of rescanning, and a pass completes (and caches, as before)
# once every league is covered by a fresh scan or a fresh fragment.
# _ACTIONS_LATEST holds the most recent finished pass payload per
# viewer (complete or not) so cache misses answer with real counts and
# the moves found so far instead of the hardcoded-zero deferred payload;
# a failed pass never overwrites it (failures are still never cached).
_ACTIONS_FRAGMENTS: dict = {}
_ACTIONS_LATEST: dict = {}
try:
    _ACTIONS_FRAGMENT_TTL = float(
        os.environ.get("PORTFOLIO_ACTIONS_FRAGMENT_TTL_SECONDS") or 300.0)
except Exception:
    _ACTIONS_FRAGMENT_TTL = 300.0


def _get_action_fragment(key):
    if key is None:
        return None
    with _ACTIONS_LOCK:
        ent = _ACTIONS_FRAGMENTS.get(key)
        if ent and time.time() - ent[0] < _ACTIONS_FRAGMENT_TTL:
            return ent[1]
    return None


def _store_action_fragment(key, actions_slice, board_meta_entry):
    if key is None:
        return
    frag = {"actions": copy.deepcopy(list(actions_slice)),
            "board_meta": copy.deepcopy(board_meta_entry)}
    now = time.time()
    with _ACTIONS_LOCK:
        _ACTIONS_FRAGMENTS[key] = (now, frag)
        if len(_ACTIONS_FRAGMENTS) > 512:
            for _k in [k for k, (ts, _f) in _ACTIONS_FRAGMENTS.items()
                       if now - ts >= _ACTIONS_FRAGMENT_TTL]:
                _ACTIONS_FRAGMENTS.pop(_k, None)
            while len(_ACTIONS_FRAGMENTS) > 512:
                _oldest = min(_ACTIONS_FRAGMENTS.items(),
                              key=lambda kv: kv[1][0])[0]
                _ACTIONS_FRAGMENTS.pop(_oldest, None)


def _latest_partial_payload(actions_key):
    """Most recent finished pass payload for this viewer, while fresh."""
    if actions_key is None:
        return None
    with _ACTIONS_LOCK:
        ent = _ACTIONS_LATEST.get(actions_key)
        if ent and time.time() - ent[0] < _ACTIONS_FRAGMENT_TTL:
            return ent[1]
    return None


def _reattach_done_states(payload, user_key, season, week):
    """Refresh ``mv.done`` on a served board payload from the Done store.

    Cached payloads (_ACTIONS_CACHE, 60s) and latest-pass payloads
    (_ACTIONS_LATEST, 300s) are assembled before any Done clicks that
    land while they are fresh, and the move-done POST/DELETE never
    invalidates them, so serving one verbatim resurrects a just-Done
    move as not-done until the next recompute. One indexed
    load_done_states pass over the payload's groups at serve time keeps
    the durable store the source of truth. The cached object is updated
    in place (every serve path shares it), which is idempotent."""
    if not payload or not user_key or not season or not week:
        return payload
    groups = ((payload.get("board") or {}).get("groups")) or []
    if not groups:
        return payload
    try:
        from dashboard_services import portfolio_board as _pb
        states = _pb.load_done_states(user_key, season, week)
    except Exception:
        return payload
    for grp in groups:
        plat = str(grp.get("platform") or "")
        lid = str(grp.get("league_id") or "")
        done_count = 0
        for mv in grp.get("moves") or []:
            st = states.get((plat, lid, str(mv.get("move_key") or "")))
            mv["done"] = st
            if st:
                done_count += 1
        meta = grp.get("meta")
        if isinstance(meta, dict) and "moves_count" in meta:
            meta["done_count"] = done_count
    return payload


def _insights_cache_get(key, max_age=None):
    with _INSIGHTS_CACHE_LOCK:
        ent = _INSIGHTS_CACHE.get(key)
        if ent and time.monotonic() - ent[0] < (
                _INSIGHTS_TTL if max_age is None else max_age):
            return ent[1]
    return None


def _insights_cache_put(key, payload, age_s=0.0):
    # age_s backdates the entry: the disk tier re-seeds payloads whose
    # true compute time is in the past, and their freshness must be
    # judged from that time, never from the re-seed moment.
    with _INSIGHTS_CACHE_LOCK:
        _INSIGHTS_CACHE[key] = (time.monotonic() - max(0.0, age_s), payload)
        if len(_INSIGHTS_CACHE) > 64:
            oldest = min(_INSIGHTS_CACHE.items(), key=lambda kv: kv[1][0])[0]
            _INSIGHTS_CACHE.pop(oldest, None)


# Stale-while-revalidate for both insight endpoints (2026-10-27 speed
# pass). Past the fresh TTL, the last finished payload is served
# immediately -- its ``updated`` stays the true compute time -- while ONE
# bounded background refresh per viewer recomputes it. Memory bounds:
# every cache here holds only finished JSON payloads (never league
# contexts, never the players index), capped at 64 entries with the
# oldest evicted, and stale entries stop being served past the grace
# below. Background refreshes are single-flight per viewer (the
# _*_INFLIGHT marks, stale after 120s like the actions marks) AND
# globally capped by _INSIGHTS_COMPUTE_SEM (see _pf_kick_background):
# a kick that cannot take the semaphore is dropped and its mark
# cleared, so the next read re-kicks.
_INSIGHTS_STALE_S = 1800.0  # serve-stale grace for the Rooting Guide
_BYE_CACHE: dict = {}
_BYE_CACHE_LOCK = threading.Lock()
# Bye Watch is next week's outlook built from cache-only data, so a
# 300s fresh TTL is safe (the insights TTL is 60s because the Rooting
# Guide carries live scores); a hit pays no preamble at all.
_BYE_TTL = 300.0  # seconds
_BYE_STALE_S = 3600.0  # serve-stale grace for Bye Watch
_BYE_INFLIGHT: dict = {}
_BYE_INFLIGHT_LOCK = threading.Lock()
_BYE_INFLIGHT_STALE_S = 120.0  # seconds, mirrors _INSIGHTS_INFLIGHT_STALE_S


def _bye_cache_get(key, max_age=None):
    with _BYE_CACHE_LOCK:
        ent = _BYE_CACHE.get(key)
        if ent and time.monotonic() - ent[0] < (
                _BYE_TTL if max_age is None else max_age):
            return ent[1]
    return None


def _bye_cache_put(key, payload, age_s=0.0):
    # age_s: see _insights_cache_put (disk re-seeds keep true age).
    with _BYE_CACHE_LOCK:
        _BYE_CACHE[key] = (time.monotonic() - max(0.0, age_s), payload)
        if len(_BYE_CACHE) > 64:
            oldest = min(_BYE_CACHE.items(), key=lambda kv: kv[1][0])[0]
            _BYE_CACHE.pop(oldest, None)


# ── Disk tier for both insight payloads (worker-recycle survival) ──────
# Production runs ONE gunicorn worker that recycles every ~40-80 page
# loads (--max-requests), so the in-process dicts above are wiped
# constantly and the cold path was the COMMON path: every recycle meant
# a full background compute while the client's retry ladder burned out.
# Finished payloads therefore also persist to the shared disk cache
# (CACHE_DIR, the store the team-offense-ranks payload already uses:
# the container filesystem is shared, so it survives recycles). One
# file holds a bounded dict for BOTH kinds, keyed
# "kind:account:season:week"; entries carry their true epoch compute
# time and TTLs are enforced on read, so a disk hit never restamps
# ``updated``. Bounds: at most _PF_DISK_MAX small finished JSON
# payloads on disk, oldest evicted on write; nothing else is stored.
_PF_DISK_LOCK = threading.Lock()
_PF_DISK_MAX = 64
_PF_DISK_FILENAME = "pf_insights_cache.json"
_PF_DISK_PATH_OVERRIDE = None  # tests point this at a tmp file
_INSIGHTS_DISK_FRESH_S = 120.0  # disk payloads this young serve as fresh


def _pf_disk_path():
    if _PF_DISK_PATH_OVERRIDE:
        return _PF_DISK_PATH_OVERRIDE
    try:
        from utils.utils import CACHE_DIR
        return os.path.join(str(CACHE_DIR), _PF_DISK_FILENAME)
    except Exception:
        return None


def _pf_disk_store_key(kind, key):
    return "%s:%s:%s:%s" % (kind, key[0], key[1], key[2])


def _pf_disk_read_all():
    """The whole disk-tier dict, or {} on any failure. Never raises."""
    path = _pf_disk_path()
    if not path:
        return {}
    try:
        with open(path, encoding="utf-8") as handle:
            blob = json.load(handle)
        return blob if isinstance(blob, dict) else {}
    except Exception:
        return {}


def _pf_disk_get(kind, key, max_age_s):
    """(payload, age_seconds) for a disk entry within max_age_s, else
    None. Age is measured from the entry's true compute time."""
    ent = _pf_disk_read_all().get(_pf_disk_store_key(kind, key))
    if not isinstance(ent, dict):
        return None
    try:
        computed_at = float(ent.get("computed_at") or 0)
    except (TypeError, ValueError):
        return None
    payload = ent.get("payload")
    if not computed_at or not isinstance(payload, dict):
        return None
    age = time.time() - computed_at
    if age < 0 or age >= max_age_s:
        return None
    return payload, age


def _pf_disk_put(kind, key, payload):
    """Persist one finished payload to the disk tier, evicting the
    oldest entries past the cap. Never raises."""
    path = _pf_disk_path()
    if not path:
        return
    try:
        with _PF_DISK_LOCK:
            blob = _pf_disk_read_all()
            blob[_pf_disk_store_key(kind, key)] = {
                "computed_at": time.time(), "payload": payload}
            if len(blob) > _PF_DISK_MAX:
                oldest_first = sorted(
                    blob.items(),
                    key=lambda kv: float(
                        (kv[1] or {}).get("computed_at") or 0))
                for store_key, _ent in oldest_first[:len(blob) - _PF_DISK_MAX]:
                    blob.pop(store_key, None)
            from utils.utils import write_json
            write_json(path, blob)
    except Exception:
        logger.debug("[portfolio-%s] disk cache write failed",
                     kind, exc_info=True)


# Dedicated global cap for the two insight computes ONLY. They used to
# share _ACTIONS_COMPUTE_SEM(2) with the moves-board scans, which run
# far longer (11 leagues) and re-kick on every board load: with both
# slots busy, an insights/bye kick was silently dropped and its
# progress depended on a later client retry winning the same race
# again. One dedicated slot keeps the bound strict (at most ONE
# insights/bye compute at a time per worker, one daemon thread) while
# removing the starvation; no pool is raised anywhere.
_INSIGHTS_COMPUTE_SEM = threading.BoundedSemaphore(1)


def _pf_kick_background(kind, key, identity):
    """Kick the deferred compute for one insight endpoint on a daemon
    thread (the /api/portfolio-actions dispatch pattern).

    Single-flight per viewer via the endpoint's inflight mark; globally
    capped by _INSIGHTS_COMPUTE_SEM (acquired non-blocking: an over-cap
    kick is dropped, its mark cleared in the finally, and the next read
    re-kicks). Returns True when a worker was spawned or one is already
    in flight for this key. The worker captures no request state: the
    preamble runs off the snapshot identity, and only the finished JSON
    payload enters the cache.
    """
    if kind == "bye":
        inflight, inflight_lock = _BYE_INFLIGHT, _BYE_INFLIGHT_LOCK
        stale_s = _BYE_INFLIGHT_STALE_S
    else:
        inflight, inflight_lock = _INSIGHTS_INFLIGHT, _INSIGHTS_INFLIGHT_LOCK
        stale_s = _INSIGHTS_INFLIGHT_STALE_S
    with inflight_lock:
        started = inflight.get(key)
        if started is not None and time.monotonic() - started < stale_s:
            return True
        inflight[key] = time.monotonic()

    def _worker():
        owned = [key]
        try:
            if not _INSIGHTS_COMPUTE_SEM.acquire(blocking=False):
                return  # over the global cap: dropped, re-kicked next read
            try:
                base = _pf_light_base(identity=identity)
                if base is None:
                    return
                real_key = base["cache_key"]
                if real_key != key:
                    # The resolver moved the season: hold the mark and
                    # cache under the real key as well.
                    with inflight_lock:
                        inflight[real_key] = time.monotonic()
                    owned.append(real_key)
                if kind == "bye":
                    payload = _compute_portfolio_bye(base)
                    # A pass that scanned nothing never replaces the
                    # last good payload (the actions latest-partial rule).
                    if (payload.get("scan") or {}).get("scanned", 0) > 0:
                        _bye_cache_put(real_key, payload)
                        _pf_disk_put("bye", real_key, payload)
                else:
                    payload = _compute_portfolio_insights(base)
                    _insights_cache_put(real_key, payload)
                    if (payload.get("scan") or {}).get("scanned", 0) > 0:
                        _pf_disk_put("insights", real_key, payload)
            except Exception:
                logger.exception(
                    "[portfolio-%s] background compute failed", kind)
            finally:
                _INSIGHTS_COMPUTE_SEM.release()
        finally:
            with inflight_lock:
                for _k in owned:
                    inflight.pop(_k, None)

    try:
        threading.Thread(target=_worker, daemon=True,
                         name=f"pf-{kind}-bg").start()
    except Exception:
        with inflight_lock:
            inflight.pop(key, None)
        logger.debug("[portfolio-%s] background spawn failed",
                     kind, exc_info=True)
        return False
    return True


def _pf_players_with_byes(nfl_players, season):
    """Attach bye weeks to player rows that do not carry one.

    Sleeper's /players/nfl payload has no bye field at all (verified
    against the live shape), so any consumer reading ``bye_week`` off
    those rows silently gets None for every player. Derive the bye from
    the player's team through the schedule-derived team bye map (the
    same source the league-players endpoint attaches from), on copies,
    so the shared players cache is never mutated. Rows that already
    carry a bye keep it; when no map is available the rows pass through
    unchanged and callers see the honest no-data state.
    """
    try:
        byes = _team_bye_map(season) or {}
    except Exception:
        byes = {}
    if not byes or not isinstance(nfl_players, dict):
        return nfl_players
    from utils.nfl import norm_sched_team
    out = dict(nfl_players)
    for pid, pl in nfl_players.items():
        if not isinstance(pl, dict) or pl.get("bye_week") is not None:
            continue
        bw = byes.get(norm_sched_team(pl.get("team") or ""))
        if bw:
            out[pid] = {**pl, "bye_week": bw}
    return out


def _pf_identity_snapshot():
    """Session identity captured on the request thread.

    Background insight computes (see _pf_kick_background) run with no
    request context, so they receive this plain dict instead of touching
    ``session``; league resolution itself takes explicit ids and needs
    no session (the /api/portfolio-actions precedent).
    """
    return {
        "viewer_username": session.get("viewer_username"),
        "viewer_user_id": session.get("viewer_user_id"),
        "account_id": session.get("account_id"),
    }


# ── Shared portfolio snapshot: league resolution ────────────────────────
# One My Leagues load used to pay resolve_my_leagues (live Sleeper
# enrichment) separately in the page build, the bye endpoint, the
# insights endpoint and every background compute. The resolution result
# is now shared through this bounded cache: the page build stores its
# resolution on every load, and the preamble below reads it instead of
# resolving again. Only the small resolution result is held (the
# league_inputs list + resolved season) -- never league contexts or
# the players index, which stay in their own shared caches. The
# finished bye/insights payloads live in their own caches + disk tier
# alongside this, which together form the per-viewer snapshot.
# Bounds: _PF_RESOLUTION_MAX entries, _PF_RESOLUTION_TTL seconds.
# The page build itself never READS this cache (it always resolves
# fresh, so a just-linked league shows immediately); it only writes.
_PF_RESOLUTION_CACHE: dict = {}
_PF_RESOLUTION_LOCK = threading.Lock()
_PF_RESOLUTION_TTL = 120.0  # seconds
_PF_RESOLUTION_MAX = 64


def _pf_resolution_get(viewer_user_id, account_id, season):
    key = (account_id or f"viewer:{viewer_user_id}", int(season))
    with _PF_RESOLUTION_LOCK:
        ent = _PF_RESOLUTION_CACHE.get(key)
        if ent and time.monotonic() - ent[0] < _PF_RESOLUTION_TTL:
            return ent[1], ent[2]
    return None


def _pf_resolution_put(viewer_user_id, account_id, season,
                       league_inputs, resolved_season):
    key = (account_id or f"viewer:{viewer_user_id}", int(season))
    with _PF_RESOLUTION_LOCK:
        _PF_RESOLUTION_CACHE[key] = (
            time.monotonic(), list(league_inputs or []),
            int(resolved_season))
        if len(_PF_RESOLUTION_CACHE) > _PF_RESOLUTION_MAX:
            oldest = min(_PF_RESOLUTION_CACHE.items(),
                         key=lambda kv: kv[1][0])[0]
            _PF_RESOLUTION_CACHE.pop(oldest, None)


def _pf_viewer_leagues(identity=None):
    """Shared preamble for the insights/bye endpoints.

    Returns a context dict, or None when the caller should answer with its
    empty payload (signed out, no current week, league resolution failed).
    ``identity`` is a _pf_identity_snapshot dict when the caller is a
    background thread with no request context.
    """
    if identity is None:
        identity = _pf_identity_snapshot()
    viewer_username = identity.get("viewer_username")
    viewer_user_id = identity.get("viewer_user_id")
    account_id = identity.get("account_id")
    if (not viewer_username or not viewer_user_id) and not account_id:
        return None
    nfl_state = get_nfl_state() or {}
    season = int(nfl_state.get("season") or datetime.now().year)
    week = int(nfl_state.get("week") or 0)
    if not week or nfl_state.get("season_type") not in ("reg", "post"):
        return None
    _res_hit = _pf_resolution_get(viewer_user_id, account_id, season)
    if _res_hit is not None:
        league_inputs, season = _res_hit
    else:
        try:
            from dashboard_services.accounts import resolve_my_leagues
            import inspect
            _resolver_params = inspect.signature(resolve_my_leagues).parameters
            if "enrich_live" in _resolver_params:
                # Account users already carry their durable memberships: the
                # live Sleeper enrichment is redundant for them (the same
                # enrich_live=not bool(account_id) page_portfolio passes).
                league_inputs, season = resolve_my_leagues(
                    viewer_user_id, account_id, season,
                    enrich_live=not bool(account_id))
            else:  # compatibility for injected/legacy resolvers
                league_inputs, season = resolve_my_leagues(
                    viewer_user_id, account_id, season)
        except Exception:
            return None
        _pf_resolution_put(viewer_user_id, account_id, season,
                           league_inputs, season)
    try:
        from dashboard_services.api import get_nfl_players
        nfl_players = get_nfl_players() or {}
    except Exception:
        nfl_players = {}
    nfl_players = _pf_players_with_byes(nfl_players, season)
    try:
        model_rows = list(get_model_value_table_cached() or [])
    except Exception:
        model_rows = []
    value_by_pid = {}
    for r in model_rows:
        if isinstance(r, dict) and r.get("id"):
            try:
                value_by_pid[str(r["id"])] = float(r.get("value") or 0)
            except (TypeError, ValueError):
                pass
    return {
        "viewer_user_id": viewer_user_id,
        "account_id": account_id,
        "season": season,
        "week": week,
        "league_inputs": league_inputs or [],
        "nfl_players": nfl_players,
        "model_rows": model_rows,
        "value_by_pid": value_by_pid,
        "cache_key": (account_id or f"viewer:{viewer_user_id}", season, week),
    }


def _pf_light_base(identity=None, league_inputs=None, season=None, week=None):
    """Light preamble for the insights surfaces (2026-10-28).

    Identity + the shared league resolution (same snapshot cache the
    old preamble used) + dashboard_services.portfolio_light.make_base.
    The light base holds the shared players index and the model value
    table BY REFERENCE and resolves each league from the cached
    provider layer, so a league no longer needs a warm full league
    context (the 17-week df_weekly build) to be scanned -- the
    dependency that made Rooting Guide / Bye Watch load only
    sometimes. Returns None in exactly the cases _pf_viewer_leagues
    does (signed out, no current week, off-season, resolution failed).

    With ``league_inputs`` given (the page build, which just paid for
    its own resolution), the guards and the resolution are skipped.
    """
    if identity is None:
        identity = _pf_identity_snapshot()
    viewer_user_id = identity.get("viewer_user_id")
    account_id = identity.get("account_id")
    if league_inputs is None:
        viewer_username = identity.get("viewer_username")
        if (not viewer_username or not viewer_user_id) and not account_id:
            return None
        nfl_state = get_nfl_state() or {}
        season = int(nfl_state.get("season") or datetime.now().year)
        week = int(nfl_state.get("week") or 0)
        if not week or nfl_state.get("season_type") not in ("reg", "post"):
            return None
        leg = int(nfl_state.get("leg") or week)
        _res_hit = _pf_resolution_get(viewer_user_id, account_id, season)
        if _res_hit is not None:
            league_inputs, season = _res_hit
        else:
            try:
                from dashboard_services.accounts import resolve_my_leagues
                import inspect
                _resolver_params = inspect.signature(resolve_my_leagues).parameters
                if "enrich_live" in _resolver_params:
                    league_inputs, season = resolve_my_leagues(
                        viewer_user_id, account_id, season,
                        enrich_live=not bool(account_id))
                else:  # compatibility for injected/legacy resolvers
                    league_inputs, season = resolve_my_leagues(
                        viewer_user_id, account_id, season)
            except Exception:
                return None
            _pf_resolution_put(viewer_user_id, account_id, season,
                               league_inputs, season)
    else:
        leg = week
    from dashboard_services import portfolio_light
    from dashboard_services.api import get_nfl_players
    try:
        players = get_nfl_players() or {}
    except Exception:
        players = {}
    try:
        model_rows = get_model_value_table_cached() or []
    except Exception:
        model_rows = []
    return portfolio_light.make_base(
        viewer_user_id=viewer_user_id, account_id=account_id,
        season=season, week=week, leg=leg,
        league_inputs=league_inputs or [],
        players=players, model_rows=model_rows,
        bye_map=_team_bye_map(season) or {})


def _pf_light_cache_key():
    """The insights cache key from session + NFL state alone.

    Mirrors the identity/season/week checks and the ``cache_key`` that
    _pf_viewer_leagues builds, but pays none of that preamble (no league
    resolution, no player or model-value loads), so a cache hit -- or a
    single-flight deferral -- can be answered before any of it runs.
    Returns None when the full preamble would also answer empty (signed
    out, no current week, or outside the regular/post season).
    """
    viewer_username = session.get("viewer_username")
    viewer_user_id = session.get("viewer_user_id")
    account_id = session.get("account_id")
    if (not viewer_username or not viewer_user_id) and not account_id:
        return None
    nfl_state = get_nfl_state() or {}
    season = int(nfl_state.get("season") or datetime.now().year)
    week = int(nfl_state.get("week") or 0)
    if not week or nfl_state.get("season_type") not in ("reg", "post"):
        return None
    return (account_id or f"viewer:{viewer_user_id}", season, week)


def _pf_scan_total(league_inputs):
    return sum(1 for lg in league_inputs if str(lg.get("league_id") or ""))


def _pf_resolve_league(lg, base):
    """Cache-only ctx + viewer roster for one league input.

    Returns a resolved-league dict, or None when the league is cold or the
    viewer's roster cannot be matched (the caller counts it unscanned).
    """
    from utils.scorezone_user import match_viewer_roster

    lid = str(lg.get("league_id") or "")
    if not lid:
        return None
    plat = str(lg.get("platform") or "sleeper").lower()
    lg_season = int(lg.get("season") or base["season"])
    try:
        lctx = get_league_ctx_from_cache(plat, lid, lg_season,
                                         allow_build=False) or {}
    except Exception:
        return None
    if not lctx:
        return None
    rosters = lctx.get("rosters") or []
    account_id = base["account_id"]
    viewer_roster = None
    if account_id:
        try:
            from dashboard_services.accounts import resolve_account_viewer_for_league
            av = resolve_account_viewer_for_league(
                account_id, plat, lid, lg_season,
                lctx.get("users") or [], rosters)
            rid = str((av or {}).get("viewer_roster_id") or "")
            if rid:
                viewer_roster = next(
                    (r for r in rosters
                     if str(r.get("roster_id") or "") == rid), None)
        except Exception:
            viewer_roster = None
    if viewer_roster is None:
        viewer_roster = match_viewer_roster(
            rosters, team_id=lg.get("team_id"),
            owner_id=sleeper_owner_id_for_account(
                account_id, base["viewer_user_id"], plat))
    if not viewer_roster:
        return None
    return {
        "plat": plat,
        "lid": lid,
        "lg_season": lg_season,
        "lctx": lctx,
        "rosters": rosters,
        "viewer_roster": viewer_roster,
        "league_name": (lctx.get("league") or {}).get("name")
                       or lg.get("name") or lid,
        "league_key": f"{plat}|{lid}",
        "starters": [str(p) for p in (viewer_roster.get("starters") or []) if p],
        "roster_positions": lctx.get("roster_positions") or [],
    }


def _pf_bye_row_for(res, next_week, base):
    """Bye Watch row for one resolved league: (row, out_count) or None.

    Moved verbatim from the old insights loop; a failure here never kills
    the league's scanned count -- the caller still counts it scanned.
    """
    from dashboard_services import portfolio_insights as _pins

    nfl_players = base["nfl_players"]
    try:
        out = []
        for i, pid in enumerate(res["starters"]):
            pl = nfl_players.get(pid) or {}
            bw = pl.get("bye_week")
            try:
                bw = int(bw) if bw is not None else None
            except (TypeError, ValueError):
                bw = None
            if bw == next_week:
                out.append({
                    "pid": pid,
                    "name": pl.get("full_name") or pl.get("last_name") or pid,
                    "pos": pl.get("position") or "",
                    "slot": res["roster_positions"][i]
                            if i < len(res["roster_positions"]) else "",
                })
        starter_set = set(res["starters"])
        reserve_set = {str(p) for p in (res["viewer_roster"].get("reserve") or []) if p}
        taxi_set = {str(p) for p in (res["viewer_roster"].get("taxi") or []) if p}
        bench = []
        for pid in (res["viewer_roster"].get("players") or []):
            spid = str(pid)
            if spid in starter_set or spid in reserve_set or spid in taxi_set:
                continue
            pl = nfl_players.get(spid) or {}
            bench.append({
                "pid": spid,
                "name": pl.get("full_name") or pl.get("last_name") or spid,
                "pos": pl.get("position") or "",
                "value": base["value_by_pid"].get(spid, 0.0),
            })
        row_in = {"name": res["league_name"], "platform": res["plat"],
                  "league_id": res["lid"], "season": res["lg_season"],
                  "out": out, "bench": bench, "top_fa": {}}
        row = _pins.bye_row(row_in)
        if row.get("hole_pos"):
            rostered = {str(p) for r in res["rosters"]
                        for p in (r.get("players") or [])}
            best = None
            for r in base["model_rows"]:
                if not isinstance(r, dict):
                    continue
                if str(r.get("position") or "").upper() != row["hole_pos"]:
                    continue
                rpid = str(r.get("id") or "")
                if not rpid or rpid in rostered:
                    continue
                try:
                    rv = float(r.get("value") or 0)
                except (TypeError, ValueError):
                    continue
                if best is None or rv > best["value"]:
                    pl = nfl_players.get(rpid) or {}
                    best = {"pid": rpid, "value": rv,
                            "name": pl.get("full_name") or r.get("name") or rpid}
            if best:
                row_in["top_fa"] = {row["hole_pos"]: best}
                row = _pins.bye_row(row_in)
        return row, row.get("out_count") or 0
    except Exception:
        logger.debug("[portfolio-bye] bye row failed", exc_info=True)
        return None


def _pf_rooting_for(res, base, include_live=True):
    """Rooting inputs for one resolved league: (starters, margin or None).

    ``include_live=False`` skips the provider matchup fetch entirely:
    starters then carry projections only (points stay None), which is
    the honest pregame shape -- the page build uses it for the embedded
    first paint, and live points arrive via the insights endpoint once
    games start.
    """
    nfl_players = base["nfl_players"]
    week = base["week"]
    starters = res["starters"]
    # Live points per starter from the provider matchup rows.
    points_by_pid = {}
    opp_points = None
    my_points = None
    try:
        rows = []
        if include_live:
            from dashboard_services.platform_api import get_matchups
            rows = get_matchups(res["plat"], res["lid"], week,
                                res["lg_season"]) or []
        mine = next((r for r in rows
                     if str(r.get("roster_id")) == str(res["viewer_roster"].get("roster_id"))),
                    None)
        if mine:
            points_by_pid = {str(k): v for k, v in
                             (mine.get("players_points") or {}).items()}
            my_points = sum(float(points_by_pid.get(p) or 0) for p in starters)
            mid = mine.get("matchup_id")
            opp = next((r for r in rows
                        if r is not mine and mid is not None
                        and r.get("matchup_id") == mid), None)
            if opp:
                opp_points = sum(
                    float(v or 0)
                    for v in (opp.get("players_points") or {}).values())
    except Exception:
        points_by_pid = {}
    proj_map = {}
    try:
        pbw = res["lctx"].get("proj_by_week") or {}
        wk = pbw.get(week) or pbw.get(str(week)) or {}
        if isinstance(wk, dict):
            proj_map = {str(k): v for k, v in (wk.get("projections") or {}).items()}
    except Exception:
        proj_map = {}
    starters_out = []
    for pid in starters:
        pl = nfl_players.get(pid) or {}
        starters_out.append({
            "pid": pid,
            "name": pl.get("full_name") or pl.get("last_name") or pid,
            "pos": pl.get("position") or "",
            "team": str(pl.get("team") or "").upper(),
            "league_key": res["league_key"],
            "league_name": res["league_name"],
            "points": points_by_pid.get(pid),
            "proj": proj_map.get(pid),
        })
    margin = None
    if my_points is not None and opp_points is not None:
        margin = {
            "league_name": res["league_name"],
            "my_score": my_points, "opp_score": opp_points,
            "remaining_names": [], "final": False,
        }
    return starters_out, margin


def _pf_bye_target_week(base):
    """The week Bye Watch must cover: the first week whose games have not
    kicked off yet.

    Sleeper advances state.week to the upcoming week on Tuesday, days
    before its first game, so a flat week + 1 skips the very week users
    are setting lineups (and running waivers) for: on the Tuesday before
    Week 8 it showed Week 9 and never warned about the Week 8 byes. When
    the current week's first kickoff is still ahead, the current week IS
    the target; once it has kicked off, the target rolls to the next
    week. With no schedule data to prove either way, keep the old
    week + 1 lookahead.
    """
    week = base["week"]
    try:
        from utils.utils import load_week_schedule
        games = load_week_schedule(base["season"], week) or []
    except Exception:
        games = []
    epochs = []
    for g in games:
        if not isinstance(g, dict):
            continue
        try:
            ep = float(g.get("gameTime_epoch"))
        except (TypeError, ValueError):
            continue
        epochs.append(ep)
    if epochs and min(epochs) > time.time():
        return week
    return week + 1


def _pf_slate_ahead(season, week):
    """True when the week's first kickoff is still in the future (the
    pregame case, where the Rooting Guide needs no live data at all).
    Any doubt (no schedule, no epochs) answers False."""
    try:
        from utils.utils import load_week_schedule
        epochs = []
        for g in (load_week_schedule(season, week) or []):
            try:
                epochs.append(float(g.get("gameTime_epoch")))
            except (TypeError, ValueError, AttributeError):
                continue
        return bool(epochs) and min(epochs) > time.time()
    except Exception:
        return False


def _pf_embed_payloads(base):
    """First-paint payloads for the page build's embedded JSON blocks.

    The page build already paid for the league resolution, and the
    light resolver holds everything Bye Watch and the pregame
    Rooting Guide need (provider-layer rosters, the shared players
    index and model values by reference, schedule + projections),
    so both payloads are computed or reused HERE, inside
    the pass that already paid for the data, and stored into the
    shared snapshot caches (+ disk tier) for the endpoints to serve.
    The page request never waits on a provider call for this: bye is
    cache-only by design, and rooting is embedded only while the whole
    slate is still ahead, in its no-live-calls shape.

    Returns (bye_payload_or_None, rooting_payload_or_None); a payload
    that scanned nothing is never embedded (the client then falls back
    to the endpoint, whose scan accounting stays honest).
    """
    key = base["cache_key"]
    bye_payload = _bye_cache_get(key)
    if bye_payload is None:
        disk = _pf_disk_get("bye", key, _BYE_TTL)
        if disk is not None:
            bye_payload, _age = disk
            _bye_cache_put(key, bye_payload, age_s=_age)
    if bye_payload is None:
        bye_payload = _compute_portfolio_bye(base)
        if (bye_payload.get("scan") or {}).get("scanned", 0) > 0:
            _bye_cache_put(key, bye_payload)
            _pf_disk_put("bye", key, bye_payload)
        else:
            bye_payload = None
    elif (bye_payload.get("scan") or {}).get("scanned", 0) == 0:
        bye_payload = None

    rooting_payload = None
    if _pf_slate_ahead(base["season"], base["week"]):
        rooting_payload = _insights_cache_get(key)
        if rooting_payload is None:
            disk = _pf_disk_get("insights", key, _INSIGHTS_DISK_FRESH_S)
            if disk is not None:
                rooting_payload, _age = disk
                _insights_cache_put(key, rooting_payload, age_s=_age)
        if rooting_payload is None:
            rooting_payload = _compute_portfolio_insights(
                base, include_live=False)
            _scan = rooting_payload.get("scan") or {}
            if _scan.get("scanned", 0) > 0:
                _insights_cache_put(key, rooting_payload)
                if not _scan.get("incomplete"):
                    _pf_disk_put("insights", key, rooting_payload)
            else:
                rooting_payload = None
    return bye_payload, rooting_payload


def _compute_portfolio_bye(base):
    """Assemble the Bye Watch payload from a light-resolved preamble
    (dashboard_services.portfolio_light). Kept as the bp-level entry
    the endpoints, the background worker and the embed pass call."""
    from dashboard_services import portfolio_light
    return portfolio_light.bye_payload(base)


@user_pages_bp.route("/api/portfolio-bye")
def api_portfolio_bye():
    """Bye Watch data for My Leagues (v25 insights band, split out).

    Cache-only league contexts, never a provider matchup call: cold leagues
    are skipped and reported in the scan block. Not PRO-gated, matching the
    other insight cards on the page.

    Speed (2026-10-27): the finished payload is cached per viewer for
    _BYE_TTL seconds, decided from the light key BEFORE the
    _pf_viewer_leagues preamble (live league-resolution enrichment, the
    players index, the model value table), so a hit pays none of it.
    Past the TTL the stale payload is served immediately -- ``updated``
    stays the true compute time -- while one bounded background refresh
    recomputes it (see _pf_kick_background). A cold load still computes
    on the request thread: the work is cache-only and bounded (at most
    12 league resolves, no provider calls), and the client has no retry
    ladder for this endpoint to land a deferred answer on."""
    empty = {"ok": True, "bye": {"week": 0, "rows": [], "total_out": 0},
             "scan": {"total": 0, "scanned": 0, "incomplete": True}}
    light_key = _pf_light_cache_key()
    if light_key is None:
        # Signed out / no current week / off-season: the preamble would
        # answer empty too, without paying for league resolution.
        return jsonify(empty)
    cached = _bye_cache_get(light_key)
    if cached is not None:
        return jsonify(cached)
    stale = _bye_cache_get(light_key, max_age=_BYE_STALE_S)
    if stale is not None:
        _pf_kick_background("bye", light_key, _pf_identity_snapshot())
        return jsonify(stale)
    # Disk tier: the payload survives worker recycles on the shared
    # disk cache, so a post-recycle load serves it (stale, with one
    # background refresh when past the fresh TTL) instead of paying
    # the full preamble synchronously on the request thread.
    disk = _pf_disk_get("bye", light_key, _BYE_STALE_S)
    if disk is not None:
        payload, age = disk
        _bye_cache_put(light_key, payload, age_s=age)
        if age >= _BYE_TTL:
            _pf_kick_background("bye", light_key, _pf_identity_snapshot())
        return jsonify(payload)
    base = _pf_light_base()
    if base is None:
        return jsonify(empty)
    payload = _compute_portfolio_bye(base)
    if payload["scan"]["scanned"] > 0:
        _bye_cache_put(base["cache_key"], payload)
        _pf_disk_put("bye", base["cache_key"], payload)
    return jsonify(payload)


def _compute_portfolio_insights(base, include_live=True):
    """Assemble the Rooting Guide payload from a light-resolved
    preamble (dashboard_services.portfolio_light).

    ``include_live=False`` produces the pregame shape with no provider
    calls at all (no matchup fetches, no ESPN overlay): schedule +
    starters + projections only, for the page build's embedded paint.
    Kept as the bp-level entry the background worker and the embed
    pass call."""
    from dashboard_services import portfolio_light
    return portfolio_light.rooting_payload(base, include_live=include_live)


@user_pages_bp.route("/api/portfolio-insights")
def api_portfolio_insights():
    """Rooting Guide data for My Leagues (v25 insights band).

    Cache-only like the actions scan: cold leagues are skipped and reported
    in the scan block, never built inline. Not PRO-gated, matching the
    other insight cards on the page. Responses are cached per viewer for
    _INSIGHTS_TTL seconds; the payload's ``updated`` stays the time the
    data was actually computed."""
    empty = {"ok": True, "rooting": {"windows": []},
             "bye": {"week": 0, "rows": [], "total_out": 0},
             "scan": {"total": 0, "scanned": 0, "incomplete": True}}
    # Cache and single-flight are decided from the light key BEFORE the
    # _pf_viewer_leagues preamble (resolve_my_leagues live enrichment,
    # get_nfl_players, the model value table): a hit pays none of it, and
    # no request thread ever runs the scan itself (2026-10-27 speed pass:
    # the owning request used to pay the full per-league provider sweep
    # synchronously while only concurrent requests deferred). The compute
    # now runs on one bounded background thread per viewer (see
    # _pf_kick_background); a fresh cache hit answers as-is, a stale hit
    # answers stale while the refresh runs, and a cold load answers with
    # the honest incomplete payload above, which the client's retry
    # ladder (loadInsights) lands on the finished cache -- the same
    # contract /api/portfolio-actions has shipped since 2026-10-26.
    light_key = _pf_light_cache_key()
    if light_key is None:
        # Signed out / no current week / off-season: the preamble would
        # answer empty too, without paying for league resolution.
        return jsonify(empty)
    cached = _insights_cache_get(light_key)
    if cached is not None:
        if (cached.get("scan") or {}).get("incomplete"):
            # Fresh but partial (the last compute landed while league
            # contexts were still cold): serve what exists, but keep a
            # refresh in flight instead of letting the fresh TTL
            # suppress re-kicks for its whole window.
            _pf_kick_background("insights", light_key,
                                _pf_identity_snapshot())
        return jsonify(cached)
    stale = _insights_cache_get(light_key, max_age=_INSIGHTS_STALE_S)
    if stale is not None:
        _pf_kick_background("insights", light_key, _pf_identity_snapshot())
        return jsonify(stale)
    # Disk tier: after a worker recycle both dicts above are empty, but
    # the last finished payload survives on the shared disk cache. A
    # young disk payload serves as-is (its ``updated`` is the true
    # compute time); an older one serves stale while a refresh runs.
    disk = _pf_disk_get("insights", light_key, _INSIGHTS_STALE_S)
    if disk is not None:
        payload, age = disk
        _insights_cache_put(light_key, payload, age_s=age)
        if age >= _INSIGHTS_DISK_FRESH_S or (
                (payload.get("scan") or {}).get("incomplete")):
            _pf_kick_background("insights", light_key,
                                _pf_identity_snapshot())
        return jsonify(payload)
    _pf_kick_background("insights", light_key, _pf_identity_snapshot())
    return jsonify(empty)


# ── Live matchup score for a My Leagues card ──────────────────────────────────
# The card's live-score slot fetches this per league, client-side and lazily, so
# a slow provider on one league never blocks the page or the other cards. The
# expensive part (a live matchup fetch + game statuses) is cached per
# league/week for a short window so a card refresh -- or several cards on the
# same league -- does not hammer the provider APIs.
_LIVE_MATCHUP_CACHE: dict = {}
_LIVE_MATCHUP_TTL = 30.0  # seconds
# Past the TTL, an entry may still be served while another thread is already
# rebuilding it (up to this grace), so concurrent card requests coalesce onto
# one provider build instead of every request re-paying the build in parallel
# -- the game-day pattern that pinned workers and left card previews stuck on
# their loading skeleton past the client's abort.
_LIVE_MATCHUP_STALE_GRACE = 300.0  # seconds
_LIVE_MATCHUP_LOCKS: dict = {}
_LIVE_MATCHUP_LOCKS_GUARD = threading.Lock()
# Keys with a background stale-while-revalidate rebuild in flight, so a burst
# of card polls schedules at most one provider rebuild per league/week.
_LIVE_MATCHUP_REBUILDING: set = set()


def _live_matchup_lock(key):
    with _LIVE_MATCHUP_LOCKS_GUARD:
        lock = _LIVE_MATCHUP_LOCKS.get(key)
        if lock is None:
            lock = threading.Lock()
            _LIVE_MATCHUP_LOCKS[key] = lock
        return lock


def _matchup_status_label(status_by_pid: dict, pids: list) -> str:
    """pre / in / final for the two teams' starters combined."""
    from dashboard_services.matchups import (
        STATUS_FINAL, STATUS_IN_PROGRESS, STATUS_NOT_STARTED,
    )
    seen = [
        status_by_pid.get(p) or status_by_pid.get(str(p)) or STATUS_NOT_STARTED
        for p in pids if p is not None
    ]
    if not seen:
        return "pre"
    if all(s == STATUS_FINAL for s in seen):
        return "final"
    if any(s in (STATUS_IN_PROGRESS, STATUS_FINAL) for s in seen):
        return "in"
    return "pre"


def _finalized_fantasy_week(ctx: dict, roster_id: str, requested_week: int,
                            current_leg: int | None = None):
    """Return the applicable finalized fantasy week, if the provider has one.

    NFL player statuses are useful for live projections, but they are not the
    authority for a fantasy matchup result (stat corrections and Monday-night
    games make that especially visible).  The normalized weekly table is the
    provider-backed fantasy-week authority and also lets Tuesday rollover show
    the matchup that just completed rather than an empty next-week preview.

    The table's ``finalized`` column is stamped when the league context was
    built (``week < leg`` at build time, in dashboard_services/service.py),
    and a context can live in cache for 12 hours -- so a context built during
    the Monday-night game still marks the just-finished week unfinalized on
    Tuesday morning, and the card above falls through to a next-week preview.
    When the caller supplies the CURRENT leg, re-apply that same rule at read
    time: any week earlier than the current leg counts as finalized, whether
    or not the build-time stamp has caught up. The roster must still have a
    row for the week (the provider actually has that week's matchup), and
    the one-week fallback guard below is unchanged.
    """
    df_weekly = (ctx or {}).get("df_weekly")
    if df_weekly is None or getattr(df_weekly, "empty", True):
        return None
    try:
        rows = df_weekly[df_weekly["roster_id"].astype(str) == str(roster_id)]
        if "week" not in rows.columns:
            return None
        if "finalized" in rows.columns:
            finalized = rows["finalized"] == True
        elif current_leg:
            import pandas as _pd
            finalized = _pd.Series(False, index=rows.index)
        else:
            return None
        if current_leg:
            finalized = finalized | (rows["week"].astype(int) < int(current_leg))
        rows = rows[finalized]
        weeks = [int(value) for value in rows["week"].tolist()
                 if int(value) <= int(requested_week)]
        if not weeks:
            return None
        latest = max(weeks)
        # Only fall back one week. Older finals belong in matchup history, not
        # in the compact current-week Portfolio card.
        return latest if latest >= int(requested_week) - 1 else None
    except Exception:
        return None


def _compute_live_matchups(key, platform, resolved_league_id, season, week, ctx):
    """Compute the live pieces for one league/week and store them in the cache.

    The caller must hold the per-key lock (the synchronous build path) or be
    the designated background rebuilder for the key (which takes the lock
    itself first). ``ctx`` dicts are read, never mutated, so sharing the same
    ctx object with a background thread matches the existing cross-thread use
    under the lock."""
    import time as _time

    from dashboard_services.matchups import build_matchup_preview
    from utils.utils import build_status_for_week

    matchups = build_matchup_preview(
        league_id=resolved_league_id,
        week=week,
        roster_map=ctx.get("roster_map") or {},
        players_map=ctx.get("players") or {},
        season=season,
        platform=platform,
    ) or []
    status_by_pid = build_status_for_week(
        season, week, ctx.get("players_index") or {}, ctx.get("teams_index") or {},
    ) or {}
    proj_bundle = build_projections_by_week(
        season, week, ctx.get("raw_scoring_settings"),
    ) or {}
    proj_map = (proj_bundle.get(week) or {}).get("projections") or {}

    result = (matchups, status_by_pid, proj_map)
    _LIVE_MATCHUP_CACHE[key] = (_time.time(), result)
    # Bound the cache: it is keyed per league/week, so it stays small, but prune
    # stale entries opportunistically so it never grows without limit.
    if len(_LIVE_MATCHUP_CACHE) > 200:
        cutoff = _time.time() - _LIVE_MATCHUP_TTL
        for k in [k for k, v in _LIVE_MATCHUP_CACHE.items() if v[0] < cutoff]:
            _LIVE_MATCHUP_CACHE.pop(k, None)
            _lk = _LIVE_MATCHUP_LOCKS.get(k)
            if _lk is not None and not _lk.locked():
                _LIVE_MATCHUP_LOCKS.pop(k, None)
    return result


def _schedule_live_matchup_rebuild(key, platform, resolved_league_id, season, week, ctx):
    """Rebuild one league/week's live entry in a daemon thread (at most one
    in flight per key). The requesting thread never waits on this: callers
    serve the within-grace stale entry and let the rebuild land for the next
    poll. The thread takes the per-key lock and re-checks freshness first --
    a sibling may have just stored a fresh entry."""
    import time as _time

    if _live_matchup_lock(key).locked():
        # A synchronous builder or an earlier rebuilder already holds the
        # per-key lock and will store a fresh entry; queueing behind it
        # would add nothing.
        return
    with _LIVE_MATCHUP_LOCKS_GUARD:
        if key in _LIVE_MATCHUP_REBUILDING:
            return
        _LIVE_MATCHUP_REBUILDING.add(key)

    def _run():
        lock = _live_matchup_lock(key)
        try:
            with lock:
                hit = _LIVE_MATCHUP_CACHE.get(key)
                if hit and (_time.time() - hit[0]) < _LIVE_MATCHUP_TTL:
                    return
                _compute_live_matchups(
                    key, platform, resolved_league_id, season, week, ctx,
                )
        except Exception:
            logger.debug("[portfolio] live matchup background rebuild failed",
                         exc_info=True)
        finally:
            with _LIVE_MATCHUP_LOCKS_GUARD:
                _LIVE_MATCHUP_REBUILDING.discard(key)

    try:
        threading.Thread(
            target=_run, name=f"live-matchup-{key[1]}-{key[3]}", daemon=True,
        ).start()
    except Exception:
        with _LIVE_MATCHUP_LOCKS_GUARD:
            _LIVE_MATCHUP_REBUILDING.discard(key)
        logger.debug("[portfolio] live matchup rebuild spawn failed", exc_info=True)


def _build_live_matchups(platform, resolved_league_id, season, week, ctx):
    """(matchups, status_by_pid, proj_map) for one league/week, TTL-cached.

    Only the live pieces are cached -- the viewer's side is picked per request.
    Stale-while-revalidate with NO synchronous build on the request thread:
    a fresh entry returns as-is; any other cached entry (inside the stale
    grace or beyond it) is served stale while a background rebuild refreshes
    it; and with nothing cached a background rebuild is kicked and None is
    returned, which callers translate into their pending shape so the
    client polls again instead of waiting on a provider build that can
    outlast its abort. The bounded-wait fallback this replaces (an
    unbounded lock wait behind an in-flight builder) could pin one
    of only four request threads for the full provider build.
    """
    import time as _time
    key = (str(platform), str(resolved_league_id), int(season), int(week))
    hit = _LIVE_MATCHUP_CACHE.get(key)
    if hit and (_time.time() - hit[0]) < _LIVE_MATCHUP_TTL:
        return hit[1]
    # Stale (any age) or missing: kick at most one background rebuild per
    # key and serve / defer immediately -- never build, never wait here.
    _schedule_live_matchup_rebuild(
        key, platform, resolved_league_id, season, week, ctx,
    )
    if hit:
        return hit[1]
    return None


@user_pages_bp.route("/api/portfolio/summary")
def api_portfolio_summary():
    """Authenticated summary hydration, independent of live matchup state."""
    account_id = session.get("account_id")
    if not account_id:
        return jsonify({"ok": False, "state": "unavailable", "message": "Sign in again"}), 401
    platform = str(request.args.get("platform") or "").strip().lower()
    league_id = str(request.args.get("league_id") or "").strip()
    try:
        season = int(request.args.get("season") or 0)
    except (TypeError, ValueError):
        season = 0
    auth_started = time.monotonic()
    from dashboard_services.accounts import resolve_account_leagues
    allowed = resolve_account_leagues(account_id, current_season=season)
    membership = next((lg for lg in allowed
                       if str(lg.get("platform") or "").lower() == platform
                       and str(lg.get("league_id") or "") == league_id
                       and int(lg.get("season") or 0) == season), None)
    if not membership:
        return jsonify({"ok": False, "state": "unavailable", "message": "League is no longer linked"}), 403
    auth_ms = (time.monotonic() - auth_started) * 1000
    from dashboard_services.portfolio_summary import (
        build_league_summary, classify_failure, get_cached_summary, ContextPending,
    )
    stale = get_cached_summary(account_id, platform, league_id, season)
    try:
        # Portfolio hydration is a passive reader.  A cold provider context is
        # warmed by get_league_ctx_from_cache, never built on this web worker.
        def cache_only_loader(p, lid, yr):
            return get_league_ctx_from_cache(p, lid, yr, allow_build=False)
        # Do not perform summary/provider work on a cold cache.
        ctx = cache_only_loader(platform, league_id, season)
        if not ctx:
            body = {"ok": True, "state": "pending", "pending": True,
                    "retry_after_ms": 3000}
            if stale:
                body.update(summary=stale, **stale, stale=True, _cache_stale=True)
            return jsonify(body)
        # Reuse the cache result and prevent a second context lookup.
        result = build_league_summary(
            account_id, membership, lambda *_args, **_kwargs: ctx)
        return jsonify({"ok": True, **result})
    except Exception as exc:
        if isinstance(exc, ContextPending):
            body = {"ok": True, "state": "pending", "pending": True,
                    "retry_after_ms": 3000}
            if stale:
                body.update(summary=stale, **stale, stale=True, _cache_stale=True)
            return jsonify(body)
        from dashboard_services.providers.base import ProviderUnavailableError
        category = classify_failure(exc)
        if isinstance(exc, ProviderUnavailableError):
            # Expected external outage (the provider host is down / returning 5xx
            # or a block page). A full traceback here reads like an app crash
            # during a provider outage; the retryable state + stale fallback below
            # already handle it, so log it compactly.
            logger.warning("portfolio summary unavailable for %s:%s: %s", platform, league_id, exc)
        else:
            logger.warning("portfolio summary failed for %s:%s", platform, league_id, exc_info=True)
        if stale:
            return jsonify({"ok": True, **stale, "stale": True, "_cache_stale": True,
                            "refresh_failure_category": category})
        return jsonify({"ok": False, "state": "unavailable",
                        "failure_category": category,
                        "retryable": category in {"transient_timeout", "rate_limited", "provider_5xx", "unknown"},
                        "message": "Summary temporarily unavailable."}), 503


@user_pages_bp.route("/api/weekly/optimal")
def api_weekly_optimal():
    """Authenticated Lineup-only fragment; never renders the Weekly Hub shell."""
    account_id = session.get("account_id")
    if not account_id:
        return jsonify({"ok": False, "message": "Sign in again"}), 401
    platform = str(request.args.get("platform") or "").lower().strip()
    league_id = str(request.args.get("league_id") or "").strip()
    view, period = request.args.get("view", "user"), request.args.get("period", "weekly")
    try:
        season = int(request.args.get("season") or 0)
        week = int(request.args.get("week") or 0)
    except (TypeError, ValueError):
        return jsonify({"ok": False, "message": "Invalid parameters"}), 400
    if view not in {"user", "league"} or period not in {"weekly", "season"}:
        return jsonify({"ok": False, "message": "Invalid parameters"}), 400
    from dashboard_services.accounts import resolve_account_leagues
    allowed = resolve_account_leagues(account_id, current_season=season)
    if not any(str(x.get("platform") or "").lower() == platform
               and str(x.get("league_id") or "") == league_id for x in allowed):
        return jsonify({"ok": False, "message": "League is no longer linked"}), 403
    ctx = get_league_ctx_from_cache(platform, league_id, season)
    # Mirror the Weekly Hub page route: the Lineup builder reads the ctx's
    # weekly bits (matchups / projections), and a cached ctx that has not
    # been through the hub yet may not carry them, which silently changed
    # (or dropped) the This Week view the fragment returned.
    if ctx:
        ensure_weekly_bits(ctx)
    from dashboard_services.pages.optimal_page import build_optimal_body
    html_fragment = build_optimal_body(ctx)
    query = f"tab=optimal&view={view}&period={period}"
    if period == "weekly" and week > 0:
        query += f"&week={week}"
    return jsonify({"ok": True, "html": html_fragment,
                    "canonical_url": f"/{platform}/{season}/{league_id}/weekly?{query}"})


@user_pages_bp.route("/api/portfolio/refresh", methods=["POST"])
def api_portfolio_refresh():
    """Kick a background refresh for at most 4 visible Portfolio cards.

    Kick-and-return (2026-10-27): busts each league and kicks its
    background summary refresh, then answers accepted/pending at once.
    The client polls /api/portfolio/card until the rebuilds land."""
    account_id = session.get("account_id")
    if not account_id:
        return jsonify({"ok": False, "failure_category": "auth_required"}), 401
    payload = request.get_json(silent=True) or {}
    if "account_id" in payload or not isinstance(payload.get("leagues"), list):
        return jsonify({"ok": False, "failure_category": "invalid_request"}), 400
    submitted = payload["leagues"]
    if not submitted or len(submitted) > 4:
        return jsonify({"ok": False, "failure_category": "invalid_request"}), 400
    from dashboard_services.accounts import resolve_account_leagues
    allowed = resolve_account_leagues(account_id, current_season=datetime.now().year)
    allowed_by_key = {
        (str(x.get("platform") or "").lower(), str(x.get("league_id") or ""), int(x.get("season") or 0)): x
        for x in allowed
    }
    requested = []
    for raw in submitted:
        try:
            key = (str(raw.get("platform") or "").lower(), str(raw.get("league_id") or ""), int(raw.get("season") or 0))
        except (AttributeError, TypeError, ValueError):
            return jsonify({"ok": False, "failure_category": "invalid_request"}), 400
        if key not in allowed_by_key:
            return jsonify({"ok": False, "failure_category": "unlinked_league"}), 403
        requested.append((key, allowed_by_key[key]))

    # Kick-and-return: bust each requested league and kick its rebuild on
    # the existing background machinery, then answer immediately. The
    # synchronous rebuild this replaces busted the cache and rebuilt on
    # the request thread (14-120s per batch, un-cancellable once the
    # client gave up, and a re-tap piled a second build onto the same
    # semaphores). _kick_summary_refresh submits to the bounded
    # _portfolio_refresh_pool (2 workers) with per-league single-flight,
    # and its build loads the league context cache-only, which kicks the
    # deduped league warm when the context is cold -- so the ctx and the
    # summary both rebuild in the background, and the client polls
    # /api/portfolio/card for completion instead of reading a summary
    # out of this response.
    from app import DASHBOARD_CACHE, _cache_key, _touch_league_bust
    results = []
    for key, membership in requested:
        platform, league_id, season = key
        try:
            cache_key = _cache_key(platform, season, league_id)
            if cache_key in DASHBOARD_CACHE:
                DASHBOARD_CACHE[cache_key]["ts"] = 0
        except Exception:
            logger.debug("portfolio refresh cache bust failed league=%s",
                         league_id, exc_info=True)
        try:
            _touch_league_bust(platform, season, league_id)
        except Exception:
            logger.debug("portfolio refresh bust touch failed league=%s",
                         league_id, exc_info=True)
        _kick_summary_refresh(account_id, membership)
        results.append({"ok": True, "pending": True,
                        "platform": platform, "league_id": league_id,
                        "season": season})
    logger.info("portfolio_refresh account=%s visible_card_count=%d accepted=%d pending=%d",
                account_id, len(requested), len(results), len(results))
    return jsonify({"ok": True, "accepted": True, "partial": False,
                    "results": results, "refreshed_at": None})


@user_pages_bp.route("/api/portfolio/card")
def api_portfolio_card():
    """Hydrate one portfolio card from a single cache-only context read."""
    started = time.monotonic()
    account_id = session.get("account_id")
    if not account_id:
        return jsonify({"ok": False, "state": "unavailable", "message": "Sign in again"}), 401
    platform = str(request.args.get("platform") or "").strip().lower()
    league_id = str(request.args.get("league_id") or "").strip()
    try:
        season = int(request.args.get("season") or 0)
    except (TypeError, ValueError):
        season = 0

    clock = time.monotonic
    auth_started = clock()
    from dashboard_services.accounts import resolve_account_leagues
    membership = next((lg for lg in resolve_account_leagues(account_id, current_season=season)
                       if str(lg.get("platform") or "").lower() == platform
                       and str(lg.get("league_id") or "") == league_id
                       and int(lg.get("season") or 0) == season), None)
    auth_ms = round((clock() - auth_started) * 1000)
    if not membership:
        logger.info("portfolio_card account=%s platform=%s league=%s season=%s auth_ms=%s unauthorized=true",
                    account_id, platform, league_id, season, auth_ms)
        return jsonify({"ok": False, "state": "unavailable", "message": "League is no longer linked"}), 403

    from dashboard_services.portfolio_summary import get_cached_summary
    stale_summary = get_cached_summary(account_id, platform, league_id, season)
    # Fresh-ready is the only summary state served as complete: it IS what
    # build_league_summary would re-derive from this context. Anything else
    # (stale, partial, or missing) is served stale-or-pending below while
    # the existing background refresh rebuilds it -- the build never runs
    # on this request thread.
    _summary_fresh = False
    if isinstance(stale_summary, dict) and stale_summary.get("state") == "ready":
        try:
            _nfl = get_nfl_state() or {}
            _ttl = _portfolio_swr_ttl_s(
                int(_nfl.get("season") or season), int(_nfl.get("week") or 0))
            _summary_fresh = _cached_summary_fresh(stale_summary, _ttl)
        except Exception:
            logger.debug("portfolio card freshness check failed", exc_info=True)
            _summary_fresh = False

    def _kick_card_refresh():
        try:
            _kick_summary_refresh(account_id, membership)
        except Exception:
            logger.debug("portfolio card refresh kick failed league=%s",
                         league_id, exc_info=True)
    ctx_started = clock()
    try:
        ctx = get_league_ctx_from_cache(platform, league_id, season, allow_build=False)
    except Exception:
        logger.warning("portfolio card cache read failed for %s:%s", platform, league_id, exc_info=True)
        ctx = {}
    context_ms = round((clock() - ctx_started) * 1000)
    cache_state = "miss"
    if ctx:
        cache_state = "stale" if ctx.get("_cache_stale") else "hit"

    summary_ms = matchup_ms = 0
    if not ctx:
        summary = stale_summary
        matchup = {"live": False, "state": "pending", "pending": True, "retry_after_ms": 3000}
        state, pending, warm_scheduled = "pending", True, True
        if not _summary_fresh:
            _kick_card_refresh()
    else:
        summary_started = clock()
        # Serve the persistent summary exactly as cached: fresh-ready
        # completes the card; stale/partial is served as last-good and
        # missing serves as the pending state, both with the background
        # summary refresh kicked so the client's poll (pending below,
        # retry_after_ms 3000) lands on the rebuilt entry.
        summary = stale_summary
        if not _summary_fresh:
            _kick_card_refresh()
        summary_ms = round((clock() - summary_started) * 1000)
        matchup_started = clock()
        # Reuse the existing matchup calculation without another membership or
        # context lookup.  Flask's g is request-local and never crosses users.
        g.portfolio_membership_authorized = True
        g.portfolio_card_ctx = ctx
        matchup_response = api_portfolio_matchup()
        if isinstance(matchup_response, tuple):
            matchup_response = matchup_response[0]
        matchup = matchup_response.get_json() if hasattr(matchup_response, "get_json") else matchup_response
        matchup_ms = round((clock() - matchup_started) * 1000)
        pending = bool((matchup or {}).get("pending")) or not _summary_fresh
        state = "pending" if pending else "ready"
        warm_scheduled = not _summary_fresh

    total_ms = round((clock() - started) * 1000)
    logger.info(
        "portfolio_card account=%s platform=%s league=%s season=%s auth_ms=%s context_state=%s "
        "context_ms=%s summary_ms=%s matchup_ms=%s total_ms=%s warm_scheduled=%s",
        account_id, platform, league_id, season, auth_ms, cache_state, context_ms,
        summary_ms, matchup_ms, total_ms, warm_scheduled,
    )
    return jsonify({"ok": True, "state": state, "pending": pending,
                    "retry_after_ms": 3000 if pending else None,
                    "summary": summary, "matchup": matchup})


@user_pages_bp.route("/api/portfolio/matchup")
def api_portfolio_matchup():
    """Live matchup score (your total vs opponent) for one My Leagues card.

    Returns {"live": false} for any state where a live score is meaningless
    (offseason, pre-draft, no linked team, bye week, or the fetch failing) so
    the card silently falls back to its record. When live, returns each side's
    current score and projected final total plus a pre/in/final status."""
    from flask import jsonify

    viewer_username = session.get("viewer_username")
    viewer_user_id = session.get("viewer_user_id")
    if (not viewer_username or not viewer_user_id) and not session.get("account_id"):
        return jsonify({"live": False})

    platform = (request.args.get("platform") or "sleeper").strip().lower()
    league_id = (request.args.get("league_id") or "").strip()
    if not league_id:
        return jsonify({"live": False})

    # Cache/build keys are league-scoped, but authorization is viewer-scoped and
    # must be rechecked on every request (including cache hits).
    account_id = session.get("account_id")
    if account_id and not getattr(g, "portfolio_membership_authorized", False):
        try:
            requested_season = int(request.args.get("season") or 0)
        except (TypeError, ValueError):
            requested_season = 0
        from dashboard_services.accounts import resolve_account_leagues
        allowed = resolve_account_leagues(account_id, current_season=requested_season)
        if not any(str(lg.get("league_id") or "") == league_id
                   and str(lg.get("platform") or "sleeper").lower() == platform
                   and int(lg.get("season") or 0) == requested_season for lg in allowed):
            logger.info("[portfolio-matchup] dropped unauthorized account=%s platform=%s league=%s", account_id, platform, league_id)
            return jsonify({"live": False}), 403

    nfl_state = get_nfl_state() or {}
    # Live scoring only makes sense in the regular season or playoffs. get_nfl_state
    # normalizes the phase to "reg"/"post" (VALID_PHASES = off/pre/reg/post), so
    # checking only "regular" here never matched in-season and hid every matchup
    # card. Accept both the normalized "reg" and the raw "regular" defensively.
    if str(nfl_state.get("season_type") or "").lower() not in ("reg", "regular", "post"):
        return jsonify({"live": False, "applicable": False, "reason": "season_type"})
    try:
        default_season = int(nfl_state.get("season") or datetime.now().year)
        season = int(request.args.get("season") or default_season)
        week = int(nfl_state.get("week") or nfl_state.get("leg") or 0)
    except (TypeError, ValueError):
        return jsonify({"live": False})
    if week < 1:
        return jsonify({"live": False, "applicable": False, "reason": "week"})

    try:
        ctx = getattr(g, "portfolio_card_ctx", None)
        if ctx is None:
            ctx = get_league_ctx_from_cache(platform, league_id, season, allow_build=False)
    except Exception:
        logger.debug("[portfolio-matchup] ctx load failed", exc_info=True)
        return jsonify({"live": False, "state": "error", "message": "Matchup temporarily unavailable"})
    if not ctx:
        # Cold cache: a background warm was kicked. Do not block the request on a
        # cold build (the card's fetch aborts long before it finishes). Tell the
        # client to poll again shortly, once the warm has populated the cache.
        return jsonify({"live": False, "state": "pending", "pending": True,
                        "retry_after_ms": 3000})
    if ctx.get("offseason_mode"):
        return jsonify({"live": False, "applicable": False, "reason": "offseason"})

    # Which roster is "yours". For a Google-account viewer the team is linked via
    # the account, not a Sleeper session identity, so the session-based
    # ctx["viewer"] is empty and every card would read live:false. Resolve through
    # the account first (as the portfolio summary / page / actions all do), then
    # fall back to the session viewer for Sleeper-session users.
    viewer_rid = ""
    if account_id:
        try:
            from dashboard_services.accounts import resolve_account_viewer_for_league
            _av = resolve_account_viewer_for_league(
                int(account_id), platform, league_id, season,
                ctx.get("users") or [], ctx.get("rosters") or [],
            )
            viewer_rid = str((_av or {}).get("viewer_roster_id") or "")
        except Exception:
            logger.debug("[portfolio-matchup] account viewer resolve failed", exc_info=True)
    if not viewer_rid:
        viewer_rid = str((ctx.get("viewer") or {}).get("viewer_roster_id") or "")
    if not viewer_rid:
        return jsonify({"live": False, "applicable": False, "reason": "no_viewer"})

    from zoneinfo import ZoneInfo
    today = datetime.now(ZoneInfo("America/New_York")).date()
    # Pass the CURRENT leg so finalized is re-derived at read time: the ctx's
    # build-time stamp can predate the week rollover by up to the 12h cache
    # TTL (a Monday-night build still marks the finished week unfinalized on
    # Tuesday morning, which stranded the card on a next-week 0.0 preview).
    try:
        current_leg = int(nfl_state.get("leg") or week)
    except (TypeError, ValueError):
        current_leg = week
    finalized_week = _finalized_fantasy_week(
        ctx, viewer_rid, week, current_leg=current_leg)
    # Some feeds advance nfl_state before the just-finished fantasy result has
    # cleared on Tuesday. Use that final only during the rollover window; later
    # in the week the card must move on to the scheduled/current matchup.
    if finalized_week == week - 1 and today.weekday() == 1:
        week = finalized_week

    # Show the matchup preview all week, not just from ~90 min before kickoff:
    # pre-kickoff the card renders a "pre" preview (projected totals + win
    # probability from the same model), matching the single-league dashboard,
    # which already previews the upcoming matchup. Live scores still take over at
    # kickoff and the final result stays up afterward -- the status label the
    # client renders ("Wk N" / "Live" / "Final") is driven by the players' game
    # state below, so no game has to have started for the preview to be useful.

    resolved_league_id = ctx.get("resolved_league_id") or league_id
    try:
        _live = _build_live_matchups(
            platform, resolved_league_id, season, week, ctx,
        )
    except Exception:
        logger.debug("[portfolio-matchup] live build failed", exc_info=True)
        return jsonify({"live": False, "state": "error", "message": "Matchup temporarily unavailable", "reason": "build_failed"})
    if _live is None:
        # No usable live entry yet and a background rebuild is running:
        # the pending shape the card client already polls on.
        return jsonify({"live": False, "state": "pending", "pending": True,
                        "retry_after_ms": 3000})
    matchups, status_by_pid, proj_map = _live

    # Find the viewer's matchup and orient "you" / "opp".
    you = opp = None
    for m in matchups:
        left = m.get("left") or {}
        right = m.get("right") or {}
        if str(left.get("roster_id") or "") == viewer_rid:
            you, opp = left, right
            break
        if str(right.get("roster_id") or "") == viewer_rid:
            you, opp = right, left
            break
    if not you:
        return jsonify({
            "live": False, "applicable": False, "reason": "no_matchup",
            "debug": {"week": week, "viewer_rid": viewer_rid, "matchups": len(matchups)},
        })

    from dashboard_services.matchups import (
        _proj_value_for_pid, compute_win_prob, team_live_totals, make_frac_lookup,
    )

    # Live game progress so in-progress starters project their finish here too,
    # matching the Matchups tab instead of freezing at their current points.
    try:
        from datetime import date as _date
        from dashboard_services.api import (
            get_nfl_scores_for_date, build_team_game_lookup,
        )
        _scores_body = get_nfl_scores_for_date(_date.today().strftime("%Y%m%d"), timeout=5)
        _team_game_lookup = build_team_game_lookup(_scores_body) if _scores_body else {}
    except Exception:
        logger.debug("[portfolio-matchup] live scores load failed", exc_info=True)
        _team_game_lookup = {}
    _frac_lookup = make_frac_lookup(_team_game_lookup)

    # The precomputed bundle can miss starters (ESPN/Yahoo roster pids, or a
    # first paint racing projection hydration), and those players then project
    # as 0.0 here -- understating a side's total and skewing the win bar, so the
    # card disagreed with the Matchups tab. The full matchup slide backfills such
    # misses from the raw weekly file (league-scored); do the same before scoring
    # so both team_live_totals and compute_win_prob see complete projections.
    scoring_settings = ctx.get("raw_scoring_settings") or {}
    try:
        from utils.utils import load_week_projection
        raw_week_map = load_week_projection(int(season), int(week)) or {}
    except Exception:
        raw_week_map = {}
    if raw_week_map:
        proj_map = dict(proj_map)  # never mutate the TTL-cached bundle
        for _team in (you, opp):
            for _p in ((_team or {}).get("starters") or []):
                if not _p:
                    continue  # empty slot placeholder
                _pid = _p.get("pid")
                if _pid is None or str(_pid) in proj_map or _pid in proj_map:
                    continue
                _val = _proj_value_for_pid(
                    proj_map, _pid, raw_week_map=raw_week_map,
                    scoring_settings=scoring_settings, pos=_p.get("pos") or "",
                )
                if _val:
                    proj_map[str(_pid)] = _val

    fantasy_final = finalized_week == week

    def _side(team):
        actual, proj = team_live_totals(
            team, status_by_pid, proj_map, frac_lookup=_frac_lookup,
        )
        # Provider matchup totals are authoritative once the fantasy matchup is
        # finalized. Recomputing from NFL statuses can omit stat corrections or
        # provider-specific scoring and must never replace the final score.
        if fantasy_final and isinstance(team.get("pts_total"), (int, float)):
            actual = float(team["pts_total"])
        return {
            "name": team.get("name") or "",
            "score": round(float(actual or 0.0), 2 if fantasy_final else 1),
            "proj": round(float(proj or 0.0), 1),
        }

    you_side = _side(you)
    has_opp = bool(opp and opp.get("roster_id"))
    opp_side = _side(opp) if has_opp else None

    # Your win probability from the same model the matchup slides' win bar uses:
    # locked scores plus projected remaining points as normal distributions.
    win_prob = None
    if has_opp:
        try:
            win_prob = round(compute_win_prob(
                you, opp, status_by_pid, proj_map, frac_lookup=_frac_lookup,
            ) * 100.0, 1)
        except Exception:
            logger.debug("[portfolio-matchup] win prob failed", exc_info=True)

    pids = [p.get("pid") for p in (you.get("starters") or []) if p]
    if has_opp:
        pids += [p.get("pid") for p in (opp.get("starters") or []) if p]
    status = "final" if fantasy_final else _matchup_status_label(status_by_pid, pids)

    matchup_result, margin = None, None
    if has_opp and status == "final":
        my_score = you_side["score"]
        opp_score = opp_side["score"]
        if my_score > opp_score:
            matchup_result = "W"
        elif my_score < opp_score:
            matchup_result = "L"
        else:
            matchup_result = "T"
        margin = round(abs(my_score - opp_score), 2)

    return jsonify({
        "live": True,
        "week": week,
        "status": status,
        "you": you_side,
        "opp": opp_side,
        "win_prob": win_prob,
        "result": matchup_result,
        "margin": margin,
    })


@user_pages_bp.route("/api/matchup/league-scores")
def api_matchup_league_scores():
    """League Scores: all matchups in the league for the current week.

    Returns every matchup with live scores, projected totals, win probability,
    and game status. The viewer's own matchup is flagged with is_you=True.
    Powers the "League Scores" tab on the matchup page.
    """
    from flask import jsonify

    viewer_username = session.get("viewer_username")
    viewer_user_id = session.get("viewer_user_id")
    if (not viewer_username or not viewer_user_id) and not session.get("account_id"):
        return jsonify({"matchups": [], "week": 0, "state": "error", "message": "Sign in to see league scores."}), 401

    platform = (request.args.get("platform") or "sleeper").strip().lower()
    league_id = (request.args.get("league_id") or "").strip()
    if not league_id:
        return jsonify({"matchups": [], "week": 0})

    account_id = session.get("account_id")
    if account_id and not getattr(g, "portfolio_membership_authorized", False):
        try:
            requested_season = int(request.args.get("season") or 0)
        except (TypeError, ValueError):
            requested_season = 0
        from dashboard_services.accounts import resolve_account_leagues
        allowed = resolve_account_leagues(account_id, current_season=requested_season)
        if not any(str(lg.get("league_id") or "") == league_id
                   and str(lg.get("platform") or "sleeper").lower() == platform
                   and int(lg.get("season") or 0) == requested_season for lg in allowed):
            logger.info("[league-scores] dropped unauthorized account=%s platform=%s league=%s", account_id, platform, league_id)
            return jsonify({"matchups": [], "week": 0}), 403

    nfl_state = get_nfl_state() or {}
    if str(nfl_state.get("season_type") or "").lower() not in ("reg", "regular", "post"):
        return jsonify({"matchups": [], "week": 0, "applicable": False, "reason": "season_type"})
    try:
        default_season = int(nfl_state.get("season") or datetime.now().year)
        season = int(request.args.get("season") or default_season)
        week = int(request.args.get("week") or nfl_state.get("week") or nfl_state.get("leg") or 0)
    except (TypeError, ValueError):
        return jsonify({"matchups": [], "week": 0})
    if week < 1:
        return jsonify({"matchups": [], "week": 0, "applicable": False, "reason": "week"})

    try:
        ctx = getattr(g, "portfolio_card_ctx", None)
        if ctx is None:
            ctx = get_league_ctx_from_cache(platform, league_id, season, allow_build=False)
    except Exception:
        logger.debug("[league-scores] ctx load failed", exc_info=True)
        return jsonify({"matchups": [], "week": week, "state": "error", "message": "League scores temporarily unavailable"})
    if not ctx:
        return jsonify({"matchups": [], "week": week, "state": "pending", "pending": True, "retry_after_ms": 3000})
    if ctx.get("offseason_mode"):
        return jsonify({"matchups": [], "week": week, "applicable": False, "reason": "offseason"})

    # Resolve the viewer's roster so we can flag their matchup.
    viewer_rid = ""
    if account_id:
        try:
            from dashboard_services.accounts import resolve_account_viewer_for_league
            _av = resolve_account_viewer_for_league(
                int(account_id), platform, league_id, season,
                ctx.get("users") or [], ctx.get("rosters") or [],
            )
            viewer_rid = str((_av or {}).get("viewer_roster_id") or "")
        except Exception:
            logger.debug("[league-scores] account viewer resolve failed", exc_info=True)
    if not viewer_rid:
        viewer_rid = str((ctx.get("viewer") or {}).get("viewer_roster_id") or "")

    resolved_league_id = ctx.get("resolved_league_id") or league_id
    try:
        _live = _build_live_matchups(
            platform, resolved_league_id, season, week, ctx,
        )
    except Exception:
        logger.debug("[league-scores] live build failed", exc_info=True)
        return jsonify({"matchups": [], "week": week, "state": "error", "message": "League scores temporarily unavailable", "reason": "build_failed"})
    if _live is None:
        return jsonify({"matchups": [], "week": week, "state": "pending",
                        "pending": True, "retry_after_ms": 3000})
    matchups, status_by_pid, proj_map = _live

    from dashboard_services.matchups import (
        _proj_value_for_pid, compute_win_prob, team_live_totals, make_frac_lookup,
    )
    try:
        from datetime import date as _date
        from dashboard_services.api import (
            get_nfl_scores_for_date, build_team_game_lookup,
        )
        _scores_body = get_nfl_scores_for_date(_date.today().strftime("%Y%m%d"), timeout=5)
        _team_game_lookup = build_team_game_lookup(_scores_body) if _scores_body else {}
    except Exception:
        logger.debug("[league-scores] live scores load failed", exc_info=True)
        _team_game_lookup = {}
    _frac_lookup = make_frac_lookup(_team_game_lookup)

    scoring_settings = ctx.get("raw_scoring_settings") or {}
    try:
        from utils.utils import load_week_projection
        raw_week_map = load_week_projection(int(season), int(week)) or {}
    except Exception:
        raw_week_map = {}
    if raw_week_map:
        proj_map = dict(proj_map)
        for m in matchups:
            for _team in ((m.get("left") or {}), (m.get("right") or {})):
                for _p in ((_team or {}).get("starters") or []):
                    if not _p:
                        continue  # empty slot placeholder
                    _pid = _p.get("pid")
                    if _pid is None or str(_pid) in proj_map or _pid in proj_map:
                        continue
                    _val = _proj_value_for_pid(
                        proj_map, _pid, raw_week_map=raw_week_map,
                        scoring_settings=scoring_settings, pos=_p.get("pos") or "",
                    )
                    if _val:
                        proj_map[str(_pid)] = _val

    def _side(team):
        actual, proj = team_live_totals(
            team, status_by_pid, proj_map, frac_lookup=_frac_lookup,
        )
        return {
            "name": team.get("name") or "",
            "avatar": team.get("avatar") or "",
            "roster_id": str(team.get("roster_id") or ""),
            "score": round(float(actual or 0.0), 1),
            "proj": round(float(proj or 0.0), 1),
        }

    out = []
    for m in matchups:
        left = m.get("left") or {}
        right = m.get("right") or {}
        left_side = _side(left)
        right_side = _side(right) if right.get("roster_id") else None
        win_prob = None
        if right_side:
            try:
                win_prob = round(compute_win_prob(
                    left, right, status_by_pid, proj_map, frac_lookup=_frac_lookup,
                ) * 100.0, 1)
            except Exception:
                logger.debug("[league-scores] win prob failed", exc_info=True)
        pids = [p.get("pid") for p in (left.get("starters") or []) if p]
        if right_side:
            pids += [p.get("pid") for p in (right.get("starters") or []) if p]
        status = _matchup_status_label(status_by_pid, pids)
        is_you = (str(left.get("roster_id") or "") == viewer_rid
                  or str(right.get("roster_id") or "") == viewer_rid)
        out.append({
            "left": left_side,
            "right": right_side,
            "win_prob": win_prob,
            "status": status,
            "is_you": is_you,
        })

    # Sort the viewer's matchup first, then by total points descending.
    out.sort(key=lambda x: (
        0 if x["is_you"] else 1,
        -((x["left"]["score"] or 0) + ((x["right"] or {}).get("score") or 0)),
    ))

    return jsonify({"matchups": out, "week": week, "count": len(out)})


def _moment_fantasy_pts(stat, scoring):
    """Fantasy points for a single ScoreZone play using the league's scoring.

    Maps the play's stat_line keys onto the league's normalized scoring
    settings (pass_int, fum_lost, etc.), falling back to standard values
    when a key is absent.
    """
    def num(v):
        try:
            return float(v or 0)
        except (TypeError, ValueError):
            return 0.0
    def rate(key, default):
        try:
            return float(scoring.get(key, default))
        except (TypeError, ValueError):
            return default
    pts = 0.0
    pts += rate("pass_td", 4.0) * num(stat.get("pass_td"))
    pts += rate("rush_td", 6.0) * num(stat.get("rush_td"))
    pts += rate("rec_td", 6.0) * num(stat.get("rec_td"))
    pts += rate("pass_yd", 0.04) * num(stat.get("pass_yds"))
    pts += rate("rush_yd", 0.1) * num(stat.get("rush_yds"))
    pts += rate("rec_yd", 0.1) * num(stat.get("rec_yds"))
    pts += rate("rec", 1.0) * (num(stat.get("receptions")) or num(stat.get("rec")))
    pts += rate("pass_int", -2.0) * num(stat.get("int"))
    fum = num(stat.get("fumble_lost")) or num(stat.get("fumbles_lost"))
    pts += rate("fum_lost", -2.0) * fum
    return round(pts * 10) / 10


@user_pages_bp.route("/api/redzone/moments")
@user_pages_bp.route("/api/scorezone/moments")
def api_scorezone_moments():
    """ScoreZone Moments: big plays from the viewer's current fantasy matchup.

    Returns TDs, 40+ yard gains, and turnovers involving STARTING players
    on either roster in the viewer's matchup (starters only; bench players'
    plays are not surfaced), sourced from the ScoreZone play store.
    Powers the "ScoreZone Moments" row under the matchup win probability bar.

    A cold league context (this worker has no cached ctx and none stale
    enough to serve) answers ``pending: true`` with an empty plays list so
    clients can tell "still warming" apart from "genuinely no moments" and
    retry; the success body carries ``status``/``live`` so clients know
    whether more moments can still arrive.
    """
    from flask import jsonify, request, session, g

    viewer_username = session.get("viewer_username")
    viewer_user_id = session.get("viewer_user_id")
    if (not viewer_username or not viewer_user_id) and not session.get("account_id"):
        return jsonify({"plays": [], "teams": {}})

    platform = (request.args.get("platform") or "sleeper").strip().lower()
    league_id = (request.args.get("league_id") or "").strip()
    if not league_id:
        return jsonify({"plays": [], "teams": {}})

    account_id = session.get("account_id")
    if account_id and not getattr(g, "portfolio_membership_authorized", False):
        try:
            requested_season = int(request.args.get("season") or 0)
        except (TypeError, ValueError):
            requested_season = 0
        from dashboard_services.accounts import resolve_account_leagues
        allowed = resolve_account_leagues(account_id, current_season=requested_season)
        if not any(str(lg.get("league_id") or "") == league_id
                   and str(lg.get("platform") or "sleeper").lower() == platform
                   and int(lg.get("season") or 0) == requested_season for lg in allowed):
            return jsonify({"plays": [], "teams": {}}), 403

    nfl_state = get_nfl_state() or {}
    if str(nfl_state.get("season_type") or "").lower() not in ("reg", "regular", "post"):
        return jsonify({"plays": [], "teams": {}})
    try:
        from datetime import datetime
        default_season = int(nfl_state.get("season") or datetime.now().year)
        season = int(request.args.get("season") or default_season)
        week = int(request.args.get("week") or nfl_state.get("week") or nfl_state.get("leg") or 0)
    except (TypeError, ValueError):
        return jsonify({"plays": [], "teams": {}})
    if week < 1:
        return jsonify({"plays": [], "teams": {}})

    try:
        ctx = getattr(g, "portfolio_card_ctx", None)
        if ctx is None:
            # allow_build=False serves the last-known-good (stale) ctx when
            # this worker has one and warms a fresh build in the background;
            # it only yields nothing when the worker has no entry at all.
            ctx = get_league_ctx_from_cache(platform, league_id, season, allow_build=False)
    except Exception:
        logger.debug("[scorezone-moments] ctx load failed", exc_info=True)
        return jsonify({"plays": [], "teams": {}, "pending": True})
    if not ctx:
        # Cold miss, not "no moments": the background warm kicked off by the
        # cache layer makes a retry succeed shortly. Flag it so the client
        # keeps retrying instead of treating the empty body as final.
        return jsonify({"plays": [], "teams": {}, "pending": True})
    if ctx.get("offseason_mode"):
        return jsonify({"plays": [], "teams": {}})

    viewer_rid = ""
    if account_id:
        try:
            from dashboard_services.accounts import resolve_account_viewer_for_league
            _av = resolve_account_viewer_for_league(
                int(account_id), platform, league_id, season,
                ctx.get("users") or [], ctx.get("rosters") or [],
            )
            viewer_rid = str((_av or {}).get("viewer_roster_id") or "")
        except Exception:
            logger.debug("[scorezone-moments] account viewer resolve failed", exc_info=True)
    if not viewer_rid:
        viewer_rid = str((ctx.get("viewer") or {}).get("viewer_roster_id") or "")
    if not viewer_rid:
        return jsonify({"plays": [], "teams": {}})

    resolved_league_id = ctx.get("resolved_league_id") or league_id
    try:
        _live = _build_live_matchups(
            platform, resolved_league_id, season, week, ctx,
        )
    except Exception:
        logger.debug("[scorezone-moments] live build failed", exc_info=True)
        return jsonify({"plays": [], "teams": {}})
    if _live is None:
        return jsonify({"plays": [], "teams": {}, "pending": True})
    matchups, status_by_pid, proj_map = _live

    you = opp = None
    for m in matchups:
        left = m.get("left") or {}
        right = m.get("right") or {}
        if str(left.get("roster_id") or "") == viewer_rid:
            you, opp = left, right
            break
        if str(right.get("roster_id") or "") == viewer_rid:
            you, opp = right, left
            break
    if not you:
        return jsonify({"plays": [], "teams": {}})

    # Collect starter PIDs for both teams, tagged by side.
    pid_to_side = {}
    for p in (you.get("starters") or []):
        if not p:
            continue  # empty slot placeholder
        pid = p.get("pid")
        if pid:
            pid_to_side[str(pid)] = "you"
    if opp:
        for p in (opp.get("starters") or []):
            if not p:
                continue  # empty slot placeholder
            pid = p.get("pid")
            if pid:
                pid_to_side[str(pid)] = "opp"
    if not pid_to_side:
        return jsonify({"plays": [], "teams": {}})

    from utils.scorezone_store import get_plays_for_pids
    # Week-scoped: a future/unplayed week must not surface the previous
    # week's plays (the store's recency window would otherwise do exactly
    # that, e.g. Week 3 plays showing under a Week 4 matchup on Tuesday).
    raw_plays = get_plays_for_pids(season, list(pid_to_side.keys()), week=week)

    # Player name lookup from the matchup data.
    pid_to_name = {}
    pid_to_pos = {}
    for team in (you, opp):
        if not team:
            continue
        for p in (team.get("starters") or []):
            if not p:
                continue  # empty slot placeholder
            pid = str(p.get("pid") or "")
            if pid:
                pid_to_name[pid] = p.get("name") or p.get("full_name") or ""
                pid_to_pos[pid] = p.get("pos") or ""

    moments = []
    scoring_settings = ctx.get("raw_scoring_settings") or {}
    for play in raw_plays:
        pid = str(play.get("pid") or "")
        if not pid or pid not in pid_to_side:
            continue
        stat = play.get("stat_line") or {}
        is_td = bool(play.get("is_td"))
        if not is_td:
            # Fallback: the store's is_td flag is set by the PBP attribution
            # pass and can miss when play-level TD detection fails. The
            # per-player stat line is ground truth: a TD in the stat line
            # means this player scored on this play.
            try:
                is_td = any(
                    float(stat.get(k) or 0) > 0
                    for k in ("rec_td", "rush_td", "pass_td", "def_td")
                )
            except (TypeError, ValueError):
                is_td = False
        yards = max(
            int(stat.get("pass_yds") or 0),
            int(stat.get("rush_yds") or 0),
            int(stat.get("rec_yds") or 0),
        )
        text_low = str(play.get("play_text") or "").lower()
        is_turnover = (
            int(stat.get("int") or 0) > 0
            or int(stat.get("def_int") or 0) > 0
            or "intercepted" in text_low
            or "fumble" in text_low
        )
        is_big_gain = yards >= 40
        if not (is_td or is_big_gain or is_turnover):
            continue
        kind = "td" if is_td else ("turnover" if is_turnover else "big_gain")
        moments.append({
            "play_id": play.get("play_id") or "",
            "game_id": play.get("game_id") or "",
            "kind": kind,
            "is_td": is_td,
            "pid": pid,
            "name": pid_to_name.get(pid) or play.get("name") or "",
            "pos": pid_to_pos.get(pid) or "",
            "team": play.get("team") or "",
            "side": pid_to_side.get(pid) or "",
            "play_text": play.get("play_text") or "",
            "yards": yards,
            "stat_line": stat,
            "fantasy_pts": _moment_fantasy_pts(stat, scoring_settings),
            "quarter": play.get("quarter") or "",
            "clock": play.get("clock") or "",
            "down": play.get("down") or "",
            "distance": play.get("distance") or "",
            "yard_line": play.get("yard_line") or "",
            "seq": play.get("seq") or 0,
            "observed_ts": play.get("_observed_ts") or 0,
        })

    # Sort: TDs first, then big gains, then turnovers; newest first within kind.
    kind_order = {"td": 0, "big_gain": 1, "turnover": 2}
    moments.sort(key=lambda m: (kind_order.get(m["kind"], 3), -(m.get("observed_ts") or 0)))

    td_count = sum(1 for m in moments if m["kind"] == "td")
    # Matchup liveness for the client's watch loop: "pre" / "in" / "final"
    # across both teams' starters. live == False means no new moments can
    # arrive (beyond final plays still landing in the store).
    status_label = _matchup_status_label(status_by_pid, list(pid_to_side.keys()))
    # Poller health surface: while the matchup is live, a store whose newest
    # play is minutes old means collection is broken (cron down), not that
    # nothing happened. Say so on the payload and kick one bounded fallback
    # collect (throttled, leader-locked in the store module) so a dead cron
    # cannot silently freeze Moments until someone reads Render logs.
    store_stale = False
    if status_label == "in":
        try:
            from utils.scorezone_store import maybe_fallback_collect, store_is_stale

            store_stale = store_is_stale(season, week)
            if store_stale:
                maybe_fallback_collect(season, week)
        except Exception:
            logger.debug("[scorezone-moments] stale check failed", exc_info=True)
    return jsonify({
        "plays": moments,
        "td_count": td_count,
        "teams": {
            "you": you.get("name") or "You",
            "opp": (opp or {}).get("name") or "Opp",
        },
        "status": status_label,
        "live": status_label == "in",
        "store_stale": store_stale,
    })


@user_pages_bp.route("/watchlist")
def page_watchlist():
    body = build_watchlist_page_body()
    nav_lid = session.get("last_league_id")
    nav_platform = session.get("last_platform")
    try:
        nav_season = int(session.get("last_season")) if session.get("last_season") else None
    except (TypeError, ValueError):
        nav_season = None
    return render_page(
        "Watchlist – BR Fantasy", nav_lid, "watchlist", body, nav_platform, nav_season,
        description="Your dynasty player watchlist with values, 7-day movers and injuries.",
    )


# ────────────────────────────────────────────────────────────────────────────
# Merged from routes/ui_prefs_bp.py: Account-scoped UI preferences + lightweight product events.
# ────────────────────────────────────────────────────────────────────────────

_TABLE_READY = False
_ALLOWED_PREF_KEYS = frozenset({
    "site_tour_done",
    "sub_welcome_done",
    # Push re-prompt state (signed-in users; guests use localStorage mirrors):
    # push_reprompt_dismissed = permanent "Don't show again"
    # push_reprompt_count     = total times the re-prompt has been shown
    # push_reprompt_last      = YYYY-MM-DD of the last show (once-per-day cap)
    "push_reprompt_dismissed",
    "push_reprompt_count",
    "push_reprompt_last",
})
_ALLOWED_EVENTS = frozenset({
    "site_tour_start",
    "site_tour_step",
    "site_tour_complete",
    "site_tour_skip",
    "site_tour_later",
    "sub_welcome_show",
    "sub_welcome_dismiss",
    "sub_welcome_cta",
    "sub_welcome_start_tour",
    "home_league_selected",
    "home_create_account_nudge",
})


def _user_key():
    account_id = session.get("account_id")
    if account_id not in (None, ""):
        return "acct:" + str(account_id).strip()
    return None


def _init_table():
    global _TABLE_READY
    if _TABLE_READY:
        return
    try:
        from dashboard_services.db import get_conn
        with get_conn() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS user_ui_prefs (
                    user_key    TEXT PRIMARY KEY,
                    prefs       JSONB NOT NULL DEFAULT '{}'::jsonb,
                    updated_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
                """
            )
            conn.commit()
        _TABLE_READY = True
    except Exception as exc:
        logger.warning("[ui-prefs] table init failed: %s", exc)


def _load_prefs(user_key: str) -> dict:
    from dashboard_services.db import get_conn
    with get_conn() as conn:
        row = conn.execute(
            "SELECT prefs FROM user_ui_prefs WHERE user_key = %s",
            (user_key,),
        ).fetchone()
    if not row:
        return {}
    prefs = row["prefs"] if isinstance(row, dict) else row[0]
    if isinstance(prefs, str):
        try:
            prefs = json.loads(prefs)
        except Exception:
            return {}
    return prefs if isinstance(prefs, dict) else {}


def _save_prefs(user_key: str, prefs: dict) -> dict:
    from dashboard_services.db import get_conn
    payload = json.dumps(prefs)
    with get_conn() as conn:
        conn.execute(
            """
            INSERT INTO user_ui_prefs (user_key, prefs, updated_at)
            VALUES (%s, %s::jsonb, NOW())
            ON CONFLICT (user_key) DO UPDATE
              SET prefs = EXCLUDED.prefs, updated_at = NOW()
            """,
            (user_key, payload),
        )
        conn.commit()
    return prefs


def _sanitize_prefs(raw) -> dict:
    if not isinstance(raw, dict):
        return {}
    import re as _re
    _date_re = _re.compile(r"^\d{4}-\d{2}-\d{2}$")
    out = {}
    for key, value in raw.items():
        if key not in _ALLOWED_PREF_KEYS:
            continue
        if key == "push_reprompt_count":
            # Small non-negative int; clamp absurd values.
            try:
                n = int(value)
            except (TypeError, ValueError):
                continue
            out[key] = max(0, min(n, 1000))
        elif key == "push_reprompt_last":
            if isinstance(value, str) and _date_re.match(value):
                out[key] = value
        elif isinstance(value, bool):
            out[key] = value
        elif value in (0, 1, "0", "1", "true", "false"):
            out[key] = value in (1, "1", "true", True)
    return out


@user_pages_bp.route("/api/ui-prefs", methods=["GET", "PUT"])
@limiter.limit("60/minute")
def api_ui_prefs():
    _init_table()
    user_key = _user_key()
    if not user_key:
        return jsonify({"ok": True, "synced": False, "prefs": {}})

    try:
        if request.method == "GET":
            return jsonify({"ok": True, "synced": True, "prefs": _load_prefs(user_key)})

        body = request.get_json(silent=True) or {}
        incoming = _sanitize_prefs(body.get("prefs") if "prefs" in body else body)
        if not incoming:
            return jsonify({"ok": False, "error": "No valid prefs."}), 400
        merged = _load_prefs(user_key)
        merged.update(incoming)
        _save_prefs(user_key, merged)
        return jsonify({"ok": True, "synced": True, "prefs": merged})
    except Exception as exc:
        logger.warning("[ui-prefs] error: %s", exc)
        return jsonify({"ok": False, "synced": False, "prefs": {}}), 500


# ─────────────────────────────────────────────────────────────────────────────
# Account-scoped action dismissals (dashboard Next steps + portfolio moves).
#
# Dismissals used to live only in localStorage ("br-actions-dismissed"), which
# is per-browser: dismissing an action on your phone left it sitting on your
# laptop, and the portfolio page never saw dashboard dismissals at all. This
# store is the server-side source of truth, keyed by account (the same
# _user_key() as ui-prefs: the Google-linked account id, never a Sleeper
# identity, which Google-only managers do not have). One row per dismissed
# item, scoped to (league, season, week, day) so the existing semantics hold:
# a dismissal hides the item for the rest of the day and it returns tomorrow.
#
# item_key is an opaque client key. Shared keys make the same action on the
# portfolio page and the league dashboard hide each other:
#   "wv:<player_id>"   the waiver add for that player (both surfaces)
#   "cat:lineup"       the portfolio lineup digest card; the dashboard hides
#                      its lineup-kind queue items for the league that day
#   "cat:roster"       same, for roster-kind items
#   "card:<type>"      a whole dashboard action card (nextsteps, bench-check)
#   anything else      a surface-local key (dashboard md5 action id, or a
#                      portfolio "pf:<kind>:<hash>" key)
# ─────────────────────────────────────────────────────────────────────────────

_DISMISS_TABLE_READY = False
_DISMISS_DAY_RE = None  # compiled lazily to keep module import light


def _init_dismissals_table():
    global _DISMISS_TABLE_READY
    if _DISMISS_TABLE_READY:
        return
    try:
        from dashboard_services.db import get_conn
        with get_conn() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS user_action_dismissals (
                    user_key    TEXT NOT NULL,
                    platform    TEXT NOT NULL,
                    league_id   TEXT NOT NULL,
                    season      INTEGER NOT NULL,
                    week        INTEGER NOT NULL,
                    day         TEXT NOT NULL,
                    item_key    TEXT NOT NULL,
                    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    PRIMARY KEY (user_key, platform, league_id, season, week, day, item_key)
                )
                """
            )
            conn.commit()
        _DISMISS_TABLE_READY = True
    except Exception as exc:
        logger.warning("[action-dismissals] table init failed: %s", exc)


def _clean_dismissal_scope(raw) -> tuple | None:
    """Validate one dismissal scope dict; return a clean tuple or None.

    Accepts (platform, league_id, season, week, day[, item_key]). Values are
    coerced conservatively; anything malformed rejects the whole request so a
    bad client can never write or read another scope shape.
    """
    import re as _re

    global _DISMISS_DAY_RE
    if _DISMISS_DAY_RE is None:
        _DISMISS_DAY_RE = _re.compile(r"^\d{4}-\d{2}-\d{2}$")
    if not isinstance(raw, dict):
        return None
    platform = str(raw.get("platform") or "").strip().lower()
    if not _re.fullmatch(r"[a-z0-9_-]{1,20}", platform):
        return None
    league_id = str(raw.get("league_id") or "").strip()
    if not _re.fullmatch(r"[A-Za-z0-9._-]{1,80}", league_id):
        return None
    try:
        season = int(raw.get("season"))
        week = int(raw.get("week"))
    except (TypeError, ValueError):
        return None
    if not (2000 <= season <= 2100 and 0 <= week <= 25):
        return None
    day = str(raw.get("day") or "").strip()
    if not _DISMISS_DAY_RE.match(day):
        return None
    scope = (platform, league_id, season, week, day)
    if "item_key" in raw:
        item_key = str(raw.get("item_key") or "").strip()
        if not _re.fullmatch(r"[A-Za-z0-9:._+\-]{1,96}", item_key):
            return None
        scope = scope + (item_key,)
    return scope


def _load_dismissals(user_key: str, scope: tuple) -> list:
    from dashboard_services.db import get_conn
    platform, league_id, season, week, day = scope
    with get_conn() as conn:
        rows = conn.execute(
            """
            SELECT item_key FROM user_action_dismissals
            WHERE user_key = %s AND platform = %s AND league_id = %s
              AND season = %s AND week = %s AND day = %s
            ORDER BY created_at
            """,
            (user_key, platform, league_id, season, week, day),
        ).fetchall()
    out = []
    for row in rows or []:
        out.append(row["item_key"] if isinstance(row, dict) else row[0])
    return out


@user_pages_bp.route("/api/action-dismissals", methods=["GET", "POST", "DELETE"])
@limiter.limit("120/minute")
def api_action_dismissals():
    """Read / record / undo account-scoped action dismissals.

    GET    ?platform&league_id&season&week&day -> {"dismissed": [item_key, ...]}
    POST   JSON scope + item_key              -> dismiss (idempotent)
    DELETE JSON scope + item_key              -> undo one dismissal
    """
    _init_dismissals_table()
    user_key = _user_key()
    if not user_key:
        # Guests keep the localStorage behavior; the server store is
        # account-only so a dismissal can follow the signed-in user.
        return jsonify({"ok": True, "synced": False, "dismissed": []})

    try:
        if request.method == "GET":
            scope = _clean_dismissal_scope(request.args.to_dict())
            if scope is None:
                return jsonify({"ok": False, "error": "Bad scope."}), 400
            return jsonify({
                "ok": True, "synced": True,
                "dismissed": _load_dismissals(user_key, scope),
            })

        body = request.get_json(silent=True) or {}
        scope = _clean_dismissal_scope(body)
        if scope is None or len(scope) != 6:
            return jsonify({"ok": False, "error": "Bad dismissal."}), 400
        platform, league_id, season, week, day, item_key = scope
        from dashboard_services.db import get_conn
        with get_conn() as conn:
            if request.method == "POST":
                conn.execute(
                    """
                    INSERT INTO user_action_dismissals
                        (user_key, platform, league_id, season, week, day, item_key)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT DO NOTHING
                    """,
                    (user_key, platform, league_id, season, week, day, item_key),
                )
            else:  # DELETE = undo
                conn.execute(
                    """
                    DELETE FROM user_action_dismissals
                    WHERE user_key = %s AND platform = %s AND league_id = %s
                      AND season = %s AND week = %s AND day = %s AND item_key = %s
                    """,
                    (user_key, platform, league_id, season, week, day, item_key),
                )
            conn.commit()
        return jsonify({
            "ok": True, "synced": True,
            "dismissed": _load_dismissals(user_key, scope[:5]),
        })
    except Exception as exc:
        logger.warning("[action-dismissals] error: %s", exc)
        return jsonify({"ok": False, "synced": False, "dismissed": []}), 500


@user_pages_bp.route("/api/events", methods=["POST"])
@limiter.limit("120/minute")
def api_events():
    """Record a product analytics event (structured log only)."""
    body = request.get_json(silent=True) or {}
    event = str(body.get("event") or "").strip()
    if event not in _ALLOWED_EVENTS:
        return jsonify({"ok": False, "error": "Unknown event."}), 400
    props = body.get("props") if isinstance(body.get("props"), dict) else {}
    # Keep props small and non-PII: coerce values to short strings/numbers/bools.
    clean = {}
    for key, value in list(props.items())[:12]:
        k = str(key)[:40]
        if isinstance(value, bool) or value is None:
            clean[k] = value
        elif isinstance(value, (int, float)):
            clean[k] = value
        else:
            clean[k] = str(value)[:80]
    account = bool(session.get("account_id"))
    logger.info(
        "[event] name=%s account=%s props=%s",
        event, account, json.dumps(clean, separators=(",", ":")),
    )
    return jsonify({"ok": True})
