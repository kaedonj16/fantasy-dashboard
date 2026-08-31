"""
User account hub pages.

Routes:
    /portfolio   (My Leagues cross-league hub)
    /watchlist   (saved-player watchlist)

Extracted from app.py to reduce monolith size. Both are signed-in user pages
that render via app.render_page.

App.py internals used by the handlers are resolved through the lazy shims below
rather than a top-level from app import ... so importing this module during
app start-up does not trigger a circular import — the real functions are only
fetched when a request is actually served.
"""
from __future__ import annotations

import html
import logging
from datetime import datetime

from flask import Blueprint, redirect, request, session, url_for

logger = logging.getLogger(__name__)

user_pages_bp = Blueprint("user_pages", __name__)


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

def _weighted_pos_strength(*args, **kwargs):
    from app import _weighted_pos_strength as _fn
    return _fn(*args, **kwargs)

def build_portfolio_body(*args, **kwargs):
    from app import build_portfolio_body as _fn
    return _fn(*args, **kwargs)


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
    # Every league durably linked to the Google account, regardless of platform,
    # plus live Sleeper enrichment from linked identities. The shared builder also
    # backs /api/my-leagues so the portfolio and switcher never diverge.
    from dashboard_services.accounts import resolve_my_leagues
    league_inputs, season = resolve_my_leagues(
        viewer_user_id, session.get("account_id"), season
    )
    def _league_summary(lg):
        lid = str(lg.get("league_id") or "")
        if not lid:
            return None
        lg_platform = (lg.get("platform") or "sleeper").lower()
        lg_season = int(lg.get("season") or season)
        try:
            lctx = get_league_ctx_from_cache(lg_platform, lid, lg_season)
        except Exception:
            return {"league_id": lid, "name": lg.get("name", "Unknown"),
                    "platform": lg_platform, "error": True}
        rosters = lctx.get("rosters") or []
        roster_map = lctx.get("roster_map") or {}
        standings_map = lctx.get("standings_map") or {}
        model_value_table = lctx.get("model_value_table") or []
        players_index = lctx.get("players_index") or {}
        values_by_id = {str(r.get("id") or ""): r for r in model_value_table if r.get("id")}
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
            }
        # Which roster is "yours": prefer the account's stored team (and ESPN SWID /
        # Sleeper identity), then the session viewer. A leftover ESPN owner id in
        # the session must not mark every Sleeper league as "Team not linked yet".
        viewer_roster = None
        _account_id = session.get("account_id")
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
            from utils.redzone_user import match_viewer_roster
            viewer_roster = match_viewer_roster(
                rosters,
                team_id=lg.get("team_id"),
                owner_id=viewer_user_id if lg_platform == "sleeper" else None,
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
            }
        rid = str(viewer_roster.get("roster_id"))
        from dashboard_services.ai.context_builders import portfolio_record_and_rank
        wins, losses, ties, pf, rank = portfolio_record_and_rank(lctx, rid, viewer_roster)
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
        # Resolve a player's position from the value table, falling back to the
        # players index — so the user side and the league side bucket the same
        # players (the user side already uses this same fallback above).
        def _pos_of(p):
            return ((values_by_id.get(p) or {}).get("position")
                    or (players_index.get(p) or {}).get("pos") or "").upper()

        def _median(xs):
            s = sorted(xs)
            n = len(s)
            if not n:
                return 0.0
            m = n // 2
            return s[m] if n % 2 else (s[m - 1] + s[m]) / 2

        # Positional strength uses the SAME starter-weighted strength as the
        # league-card ranks, so a #2 WR chip and a high WR percentile never
        # disagree. weighted_pos_strength emphasizes startable top-end talent
        # and only lightly credits bench depth.
        #
        # The My Leagues summary card shows the in-league PERCENTILE of that
        # strength (then averages those percentiles across leagues). Percentiles
        # stay centered at 50th for a typical team; averaging signed % vs median
        # let one thin league drag a stacked position negative.
        from utils.roster_strength import strength_percentile
        roster_positions = lctx.get("roster_positions") or []
        try:
            slot_counts = count_roster_positions(roster_positions)
        except Exception:
            slot_counts = {"QB": 1, "RB": 2, "WR": 3, "TE": 1, "FLEX": 1}
        pos_user_rank = {}
        pos_user_vals = {}
        pos_user_pctile = {}
        pos_league_avgs = {}
        for pos in ["QB", "RB", "WR", "TE"]:
            all_strengths = []
            user_strength = 0.0
            for r in rosters:
                r_pids = [str(p) for p in (r.get("players") or [])]
                r_vals = sorted(
                    [float((values_by_id.get(p) or {}).get("value") or 0)
                     for p in r_pids if _pos_of(p) == pos],
                    reverse=True,
                )
                strength = _weighted_pos_strength(r_vals, pos, slot_counts)
                all_strengths.append((str(r.get("roster_id")), strength))
                if str(r.get("roster_id")) == rid:
                    user_strength = strength
            # Median is still used for the per-league "WR-Spread" archetype
            # badge (ratio vs a typical team). The summary card uses percentile.
            pos_user_vals[pos] = user_strength
            pos_league_avgs[pos] = _median([s for _, s in all_strengths]) or 1
            pos_user_pctile[pos] = strength_percentile(
                user_strength, [s for _, s in all_strengths],
            )
            if len(all_strengths) > 1:
                ranked = sorted(all_strengths, key=lambda x: -x[1])
                pos_user_rank[pos] = next((i + 1 for i, (r_id, _) in enumerate(ranked) if r_id == rid), "?")
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
                        pts = float(row.get("pts") or row.get("PF") or 0)
                        opp = float(row.get("opp_pts") or row.get("PA") or 0)
                        streak.append("W" if pts > opp else "L")
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
        }

    leagues_data = []
    for _lg in league_inputs:
        _result = _league_summary(_lg)
        if _result:
            leagues_data.append(_result)
    leagues_data.sort(key=lambda x: x.get("name", ""))

    valid_leagues = [lg for lg in leagues_data
                     if not lg.get("error") and not lg.get("not_in_league") and not lg.get("pending")]
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
        viewer_username
        or session.get("account_email")
        or session.get("account_first_name")
        or "your account",
        valid_leagues, leagues_data, season,
        holdings, num_leagues, nfl_exposure, cross_pos,
        total_wins, total_losses, total_ties,
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
    scans up to 8 linked leagues for lineup issues and injury hints.
    """
    from flask import jsonify, session
    from dashboard_services.subscriptions import has_premium_for_viewer
    from utils.cross_league_actions import (
        injury_stash_action,
        lineup_actions_from_issues,
        rank_cross_league_actions,
    )
    from utils.lineup_issues import find_lineup_issues
    from utils.redzone_user import match_viewer_roster

    viewer_username = session.get("viewer_username")
    viewer_user_id = session.get("viewer_user_id")
    if (not viewer_username or not viewer_user_id) and not session.get("account_id"):
        return jsonify({"actions": []})

    nfl_state = get_nfl_state() or {}
    season = int(nfl_state.get("season") or datetime.now().year)
    try:
        from dashboard_services.accounts import resolve_my_leagues
        league_inputs, season = resolve_my_leagues(
            viewer_user_id, session.get("account_id"), season
        )
    except Exception:
        return jsonify({"actions": []})

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
    try:
        from dashboard_services.api import get_nfl_players
        from utils.utils import load_week_schedule
        nfl_players = get_nfl_players() or {}
        week = int(nfl_state.get("week") or 0)
        teams_playing = set()
        if week and nfl_state.get("season_type") in ("reg", "post"):
            for g in (load_week_schedule(season, week) or []):
                for side in ("home", "away"):
                    t = str(g.get(side) or "").upper()
                    if t:
                        teams_playing.add(t)
    except Exception:
        nfl_players = {}
        teams_playing = set()

    for lg in (league_inputs or [])[:8]:
        lid = str(lg.get("league_id") or "")
        plat = str(lg.get("platform") or "sleeper").lower()
        lg_season = int(lg.get("season") or season)
        if not lid:
            continue
        try:
            lctx = get_league_ctx_from_cache(plat, lid, lg_season) or {}
        except Exception:
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
                owner_id=viewer_user_id if plat == "sleeper" else None,
            )
        if not viewer_roster:
            continue
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
        # One injury stash/drop hint per league (bench/IR candidates).
        try:
            from utils.injury_plan import injury_plan
            from dashboard_services.injury_return import weeks_out_for_player
            values_by_id = {
                str(r.get("id") or ""): r
                for r in (lctx.get("model_value_table") or [])
                if r.get("id")
            }
            for pid in [str(p) for p in (viewer_roster.get("players") or [])][:40]:
                pl = nfl_players.get(pid) or {}
                st = str(pl.get("injury_status") or "").strip()
                if not st or st.lower() in ("active", "act"):
                    continue
                plan = injury_plan(
                    status=st,
                    espn_weeks=weeks_out_for_player(pid),
                    player_value=float((values_by_id.get(pid) or {}).get("value") or 0) or None,
                )
                if not plan or plan.get("verdict") not in ("IR", "Drop candidate", "Stash"):
                    continue
                if plan.get("verdict") == "Stash" and (plan.get("weeks_out") or 0) < 3:
                    continue
                act = injury_stash_action(
                    platform=plat, season=lg_season, league_id=lid,
                    league_name=league_name,
                    player_name=pl.get("full_name") or pl.get("last_name") or "Player",
                    verdict=plan["verdict"],
                    weeks_label=plan.get("weeks_label") or "",
                )
                if act:
                    actions.append(act)
                    break
        except Exception:
            pass

    return jsonify({"actions": rank_cross_league_actions(actions, limit=8)})


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
