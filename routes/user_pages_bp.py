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

import html
import logging
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


def _portfolio_refresh_pool():
    global _refresh_pool
    with _refresh_pool_lock:
        if _refresh_pool is None:
            _refresh_pool = ThreadPoolExecutor(
                max_workers=3, thread_name_prefix="pf-refresh")
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
    try:
        pool = _portfolio_refresh_pool()
    except Exception:
        logger.debug("portfolio refresh pool unavailable", exc_info=True)
        return
    lid = str((membership or {}).get("league_id") or "")

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

    try:
        pool.submit(_run)
    except Exception:
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
        return {
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
    holdings.sort(key=lambda x: (-x["shares"], -x["value"]))

    # Join 7-day value-rank movement from the in-memory value table (no DB round
    # trip) so the Portfolio Movers digest works even when the per-league player
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

    body = build_portfolio_body(
        portfolio_signed_in_label(),
        valid_leagues, leagues_data, season,
        holdings, num_leagues, nfl_exposure, cross_pos,
        total_wins, total_losses, total_ties,
        current_week=current_week,
    )

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
    if (not viewer_username or not viewer_user_id) and not session.get("account_id"):
        # Nothing was scanned at all, so an empty list here is not "all
        # caught up" - report it as incomplete.
        return jsonify({"actions": [], "scan": _scan_payload(force_incomplete=True)})

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

    for lg in (league_inputs or [])[:12]:
        lid = str(lg.get("league_id") or "")
        plat = str(lg.get("platform") or "sleeper").lower()
        lg_season = int(lg.get("season") or season)
        if not lid:
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
        account_id = session.get("account_id")
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
            viewer_roster = match_viewer_roster(
                rosters,
                team_id=lg.get("team_id"),
                owner_id=sleeper_owner_id_for_account(
                    account_id, viewer_user_id, plat,
                ),
            )
        if not viewer_roster:
            scan_skipped_no_roster += 1
            continue
        # Roster matched: the scans below actually run for this league.
        # Mid-scan sub-scan failures are best-effort and do not un-count it.
        scan_scanned += 1
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
            issues = find_lineup_issues(starters, info, teams_playing or None)
            actions.extend(lineup_actions_from_issues(
                issues, platform=plat, season=lg_season,
                league_id=lid, league_name=league_name,
            ))
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
                        actions.append(make_action(
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
                        ))
            except Exception:
                logger.debug("[portfolio-actions] swap scan failed", exc_info=True)
        # One injury stash/drop hint per league (active-roster candidates only;
        # players already in reserve/IR do not need a stash/move-to-IR tip).
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
                    actions.append(act)
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
                    actions.append(waiver_pickup_action(
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
                    ))
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

    return jsonify({
        "actions": rank_cross_league_actions(actions, limit=8),
        "scan": _scan_payload(),
    })


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


def _finalized_fantasy_week(ctx: dict, roster_id: str, requested_week: int):
    """Return the applicable finalized fantasy week, if the provider has one.

    NFL player statuses are useful for live projections, but they are not the
    authority for a fantasy matchup result (stat corrections and Monday-night
    games make that especially visible).  The normalized weekly table is the
    provider-backed fantasy-week authority and also lets Tuesday rollover show
    the matchup that just completed rather than an empty next-week preview.
    """
    df_weekly = (ctx or {}).get("df_weekly")
    if df_weekly is None or getattr(df_weekly, "empty", True):
        return None
    try:
        rows = df_weekly[df_weekly["roster_id"].astype(str) == str(roster_id)]
        if "finalized" not in rows.columns or "week" not in rows.columns:
            return None
        rows = rows[rows["finalized"] == True]
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
    Stale-while-revalidate: a within-grace stale entry is served immediately
    and refreshed by a background rebuild, so after the first build no card
    request blocks on a provider rebuild. Only a missing or beyond-grace
    entry pays the synchronous build below."""
    import time as _time
    key = (str(platform), str(resolved_league_id), int(season), int(week))
    hit = _LIVE_MATCHUP_CACHE.get(key)
    if hit and (_time.time() - hit[0]) < _LIVE_MATCHUP_TTL:
        return hit[1]
    if hit and (_time.time() - hit[0]) < _LIVE_MATCHUP_TTL + _LIVE_MATCHUP_STALE_GRACE:
        _schedule_live_matchup_rebuild(
            key, platform, resolved_league_id, season, week, ctx,
        )
        return hit[1]

    lock = _live_matchup_lock(key)
    if not lock.acquire(blocking=False):
        # Another thread is already building this league/week (synchronous
        # path or the background rebuilder). Serve the last-good entry inside
        # the grace window instead of queueing behind (or duplicating) a
        # provider build that can outlast the card's client abort; with
        # nothing usable cached, wait for the builder.
        if hit and (_time.time() - hit[0]) < _LIVE_MATCHUP_TTL + _LIVE_MATCHUP_STALE_GRACE:
            return hit[1]
        lock.acquire()
        try:
            hit = _LIVE_MATCHUP_CACHE.get(key)
            if hit:
                return hit[1]
        finally:
            lock.release()
        # The builder finished without storing (it raised): take the lock
        # and build on this thread below.
        lock.acquire()
    try:
        # Re-check under the lock: the previous builder may have just stored.
        hit = _LIVE_MATCHUP_CACHE.get(key)
        if hit and (_time.time() - hit[0]) < _LIVE_MATCHUP_TTL:
            return hit[1]
        return _compute_live_matchups(
            key, platform, resolved_league_id, season, week, ctx,
        )
    finally:
        lock.release()


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
    from dashboard_services.pages.optimal_page import build_optimal_body
    html_fragment = build_optimal_body(ctx)
    query = f"tab=optimal&view={view}&period={period}"
    if period == "weekly" and week > 0:
        query += f"&week={week}"
    return jsonify({"ok": True, "html": html_fragment,
                    "canonical_url": f"/{platform}/{season}/{league_id}/weekly?{query}"})


@user_pages_bp.route("/api/portfolio/refresh", methods=["POST"])
def api_portfolio_refresh():
    """Explicitly refresh at most the visible Portfolio cards, two at a time."""
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

    from dashboard_services.portfolio_summary import build_league_summary, classify_failure, get_cached_summary
    from app import DASHBOARD_CACHE, _cache_key, _touch_league_bust
    def refresh_one(item):
        key, membership = item
        platform, league_id, season = key
        stale = get_cached_summary(account_id, platform, league_id, season)
        try:
            cache_key = _cache_key(platform, season, league_id)
            if cache_key in DASHBOARD_CACHE:
                DASHBOARD_CACHE[cache_key]["ts"] = 0
            _touch_league_bust(platform, season, league_id)
            return {"ok": True, "summary": build_league_summary(
                account_id, membership, get_league_ctx_from_cache)}
        except Exception as exc:
            result = {"ok": False, "failure_category": classify_failure(exc)}
            if stale:
                result["last_good"] = stale
            return result

    # Browser cancellation cannot terminate an accepted WSGI request; this
    # bounded pool and league single-flight are therefore the authoritative cap.
    with ThreadPoolExecutor(max_workers=min(2, len(requested))) as pool:
        results = list(pool.map(refresh_one, requested))
    successes = sum(1 for value in results if value.get("ok"))
    logger.info("portfolio_refresh account=%s visible_card_count=%d concurrency=%d success=%d",
                account_id, len(requested), min(2, len(requested)), successes)
    successful_syncs = [r.get("summary", {}).get("last_successful_sync_at")
                        for r in results if r.get("ok") and r.get("summary", {}).get("last_successful_sync_at")]
    return jsonify({"ok": successes == len(results), "partial": 0 < successes < len(results),
                    "results": results,
                    # Batch generation is not data freshness. This value is
                    # intentionally unknown when no authoritative sync exists.
                    "refreshed_at": max(successful_syncs) if successful_syncs else None}), (200 if successes else 503)


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

    from dashboard_services.portfolio_summary import build_league_summary, get_cached_summary
    stale_summary = get_cached_summary(account_id, platform, league_id, season)
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
    else:
        summary_started = clock()
        # Fast path: a fresh cached "ready" summary IS what
        # build_league_summary would re-derive from this context (value
        # lookup over every roster + positional rankings). The client
        # hydrates every card on every page load through a 2-wide queue, so
        # re-deriving it per card made the last leagues wait on redundant
        # compute and pinned request threads. Serve the fresh cache and
        # spend the request on the matchup, which is not cached.
        summary = None
        if isinstance(stale_summary, dict) and stale_summary.get("state") == "ready":
            try:
                _nfl = get_nfl_state() or {}
                _ttl = _portfolio_swr_ttl_s(
                    int(_nfl.get("season") or season), int(_nfl.get("week") or 0))
                if _cached_summary_fresh(stale_summary, _ttl):
                    summary = stale_summary
            except Exception:
                logger.debug("portfolio card freshness check failed", exc_info=True)
                summary = None
        if summary is None:
            try:
                summary = build_league_summary(account_id, membership, lambda *_a, **_k: ctx)
            except Exception:
                logger.warning("portfolio card summary failed for %s:%s", platform, league_id, exc_info=True)
                summary = stale_summary
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
        pending = bool((matchup or {}).get("pending"))
        state = "pending" if pending else "ready"
        warm_scheduled = False

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
    finalized_week = _finalized_fantasy_week(ctx, viewer_rid, week)
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
        matchups, status_by_pid, proj_map = _build_live_matchups(
            platform, resolved_league_id, season, week, ctx,
        )
    except Exception:
        logger.debug("[portfolio-matchup] live build failed", exc_info=True)
        return jsonify({"live": False, "state": "error", "message": "Matchup temporarily unavailable", "reason": "build_failed"})

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
        matchups, status_by_pid, proj_map = _build_live_matchups(
            platform, resolved_league_id, season, week, ctx,
        )
    except Exception:
        logger.debug("[league-scores] live build failed", exc_info=True)
        return jsonify({"matchups": [], "week": week, "state": "error", "message": "League scores temporarily unavailable", "reason": "build_failed"})

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
        matchups, status_by_pid, proj_map = _build_live_matchups(
            platform, resolved_league_id, season, week, ctx,
        )
    except Exception:
        logger.debug("[scorezone-moments] live build failed", exc_info=True)
        return jsonify({"plays": [], "teams": {}})

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
    return jsonify({
        "plays": moments,
        "td_count": td_count,
        "teams": {
            "you": you.get("name") or "You",
            "opp": (opp or {}).get("name") or "Opp",
        },
        "status": status_label,
        "live": status_label == "in",
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
