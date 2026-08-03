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

def get_sleeper_user_leagues(*args, **kwargs):
    from app import get_sleeper_user_leagues as _fn
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
    if not viewer_username or not viewer_user_id:
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
    try:
        raw_leagues = get_sleeper_user_leagues(viewer_user_id, season) or []
    except Exception:
        raw_leagues = []
    # If no leagues found for the NFL-reported season, try the previous year
    if not raw_leagues:
        try:
            raw_leagues = get_sleeper_user_leagues(viewer_user_id, season - 1) or []
            if raw_leagues:
                season = season - 1
        except Exception:
            raw_leagues = []
    def _league_summary(lg):
        lid = str(lg.get("league_id") or "")
        if not lid:
            return None
        lg_season = int(lg.get("season") or season)
        try:
            lctx = get_league_ctx_from_cache("sleeper", lid, lg_season)
        except Exception:
            return {"league_id": lid, "name": lg.get("name", "Unknown"), "error": True}
        rosters = lctx.get("rosters") or []
        roster_map = lctx.get("roster_map") or {}
        standings_map = lctx.get("standings_map") or {}
        model_value_table = lctx.get("model_value_table") or []
        players_index = lctx.get("players_index") or {}
        values_by_id = {str(r.get("id") or ""): r for r in model_value_table if r.get("id")}
        viewer_roster = next(
            (r for r in rosters if str(r.get("owner_id")) == str(viewer_user_id)), None
        )
        if not viewer_roster:
            return {"league_id": lid, "name": lg.get("name", "Unknown"), "not_in_league": True}
        rid = str(viewer_roster.get("roster_id"))
        std = standings_map.get(rid) or {}
        wins = int(std.get("wins") or 0)
        losses = int(std.get("losses") or 0)
        ties = int(std.get("ties") or 0)
        pf = float(std.get("pf") or 0)
        all_std = sorted(standings_map.items(), key=lambda x: (-int(x[1].get("wins") or 0), -float(x[1].get("pf") or 0)))
        rank = next((i + 1 for i, (k, _) in enumerate(all_std) if k == rid), "?")
        total_teams = int(lctx.get("total_rosters") or len(rosters) or 12)
        player_ids = [str(p) for p in (viewer_roster.get("players") or [])]
        all_players = {}
        total_value = 0.0
        _pos_buckets = {"QB": [], "RB": [], "WR": [], "TE": []}
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
            if val > 0 and pos in _pos_buckets:
                _pos_buckets[pos].append(val)
        _top_n = {"QB": 1, "RB": 2, "WR": 3, "TE": 1}
        pos_user_vals = {p: sum(sorted(_pos_buckets[p], reverse=True)[:n]) for p, n in _top_n.items()}
        pos_league_avgs = {}
        for pos, top_n in _top_n.items():
            r_sums = []
            for r in rosters:
                r_pids = [str(p) for p in (r.get("players") or [])]
                r_pos_vals = sorted(
                    [float((values_by_id.get(p) or {}).get("value") or 0)
                     for p in r_pids if (values_by_id.get(p) or {}).get("position", "").upper() == pos],
                    reverse=True,
                )
                r_sums.append(sum(r_pos_vals[:top_n]))
            pos_league_avgs[pos] = (sum(r_sums) / len(r_sums)) if r_sums else 1
        # Positional rank within league using same weighted-strength + z-score as teams page
        roster_positions = lctx.get("roster_positions") or []
        try:
            slot_counts = count_roster_positions(roster_positions)
        except Exception:
            slot_counts = {"QB": 1, "RB": 2, "WR": 3, "TE": 1, "FLEX": 1}
        pos_user_rank = {}
        for pos in ["QB", "RB", "WR", "TE"]:
            all_strengths = []
            user_strength = 0.0
            for r in rosters:
                r_pids = [str(p) for p in (r.get("players") or [])]
                r_vals = sorted(
                    [float((values_by_id.get(p) or {}).get("value") or 0)
                     for p in r_pids if (values_by_id.get(p) or {}).get("position", "").upper() == pos],
                    reverse=True,
                )
                strength = _weighted_pos_strength(r_vals, pos, slot_counts)
                all_strengths.append((str(r.get("roster_id")), strength))
                if str(r.get("roster_id")) == rid:
                    user_strength = strength
            if len(all_strengths) > 1:
                vals_only = [s for _, s in all_strengths]
                mu = sum(vals_only) / len(vals_only)
                sigma = (sum((v - mu) ** 2 for v in vals_only) / len(vals_only)) ** 0.5
                user_z = (user_strength - mu) / sigma if sigma > 0 else 0.0
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
        league_obj = lctx.get("league") or {}
        # Urgency score: lower = needs more attention
        urgency = wins - losses + (rank if isinstance(rank, int) else 0) * -0.1
        return {
            "league_id": lid,
            "name": league_obj.get("name") or lg.get("name") or "Unknown",
            "platform": "sleeper",
            "season": season,
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
            "pos_user_rank": pos_user_rank,
            "offseason": lctx.get("offseason_mode", False),
        }

    leagues_data = []
    for _lg in raw_leagues:
        _result = _league_summary(_lg)
        if _result:
            leagues_data.append(_result)
    leagues_data.sort(key=lambda x: x.get("name", ""))

    valid_leagues = [lg for lg in leagues_data if not lg.get("error") and not lg.get("not_in_league")]
    num_leagues = len(leagues_data)
    total_wins = sum(lg.get("wins", 0) for lg in valid_leagues)
    total_losses = sum(lg.get("losses", 0) for lg in valid_leagues)
    total_ties = sum(lg.get("ties", 0) for lg in valid_leagues)

    # Cross-league avg positional value vs league average
    cross_pos = {}
    for pos in ["QB", "RB", "WR", "TE"]:
        ratios = []
        for lg in valid_leagues:
            u = (lg.get("pos_user_vals") or {}).get(pos, 0)
            a = (lg.get("pos_league_avgs") or {}).get(pos) or 1
            ratios.append(u / a)
        cross_pos[pos] = round((sum(ratios) / len(ratios)) if ratios else 1.0, 2)

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
        viewer_username, valid_leagues, leagues_data, season,
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
