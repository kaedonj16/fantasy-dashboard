"""Lightweight portfolio resolver: one slim data pass for My Leagues.

Why this exists (2026-10-28): Rooting Guide and Bye Watch loaded only
sometimes in production even after the embed/disk-tier consolidation
(#2847). The intermittency is structural. The old preamble
(``_pf_viewer_leagues`` in routes/user_pages_bp.py) resolves a league
through ``get_league_ctx_from_cache(allow_build=False)``, i.e. it can
only read a league whose FULL league context another page happened to
build and keep warm. That context is the heaviest object on the site:
``dashboard_services.service.build_tables`` fetches matchups for
weeks 1..17 per league, builds a season-long pandas ``df_weekly`` for
every roster, and runs ``finalize_team_stats`` over it; the context
then reattaches the shared players map/index, a per-league copy of the
model value table, rookie rankings and an 18-week projections bundle.
The in-process dashboard cache holds at most a handful of those
contexts, so for a viewer with more leagues than that, no scan pass
could ever see every league warm at once -- the scan landed or not
depending on which pages were visited most recently, and the heavy
builds themselves are what get starved/killed on the 1-worker,
2GB production box.

This module is the lightweight function Kaedon asked for: everything
the portfolio page's insights surfaces need, and nothing else.

Data sources, cheapest first, all shared/cached layers:
- league meta / rosters / users from the cached provider fetch layer
  (``dashboard_services.platform_api``), matchups for the CURRENT week
  only (plus, for the card streak, the last <=3 finalized weeks);
- the process-shared NFL players index BY REFERENCE (never copied,
  never per-league); bye weeks are derived per player on demand from
  the schedule team-bye map instead of copying the index to attach
  them (the old ``_pf_players_with_byes`` rebuilt ~11k row dicts on
  every preamble);
- the model value table BY REFERENCE; raw values are distilled for
  rostered player ids only (one pass, wanted-set filter), and the
  format-adjusted lookup reuses
  ``dashboard_services.ai.context_builders.league_format_value_lookup``
  on a minimal settings dict -- a pure function over the shared table,
  memoized per base by scoring signature;
- current-week projections via the process-memoized
  ``build_projections_by_week`` (only weeks 1..current are built, and
  a week's flattened map is identical to the 18-week bundle's entry
  for that week by construction: each week flattens independently
  from its own projections file).

FORBIDDEN here (the zero-heavy-calls test pins this): building or
reading a league context (``get_league_ctx`` /
``get_league_ctx_from_cache``), season-long ``df_weekly``,
``finalize_team_stats``, power-rankings inputs, per-league players
index copies, and any playoff simulation.

Memory discipline: leagues are processed SEQUENTIALLY. Each league's
raw provider payloads are distilled into the small rows the surfaces
render, then released before the next league is fetched; only the
distilled output is retained. This module keeps NO module-level
caches of its own (the finished-payload caches + disk tier live in
routes/user_pages_bp.py, capped at 64 entries each; the per-base
format-value memo dies with the base), so there is no new unbounded
state to bound.
"""
from __future__ import annotations

import logging
import time
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

# The insights surfaces scan at most this many leagues per pass (the
# same cap the old ctx-based computes used).
MAX_LEAGUES = 12


# ── Provider seams (lazy, so tests can stub the owning modules) ──────

def _model_rows():
    from app import get_model_value_table_cached
    return get_model_value_table_cached() or []


def _team_bye_map(season):
    from app import _team_bye_map as _fn
    return _fn(season) or {}


def _projections_by_week(season, weeks, scoring_settings):
    from app import build_projections_by_week
    return build_projections_by_week(season, weeks, scoring_settings)


def _players_index():
    from dashboard_services.api import get_nfl_players
    return get_nfl_players() or {}


# ── Base ─────────────────────────────────────────────────────────────

def make_base(*, viewer_user_id, account_id, season, week, league_inputs,
              players=None, model_rows=None, bye_map=None, leg=0):
    """Assemble the light preamble shared by every light surface.

    ``players`` is the process-shared NFL players index, held BY
    REFERENCE (callers must not mutate it; nothing here does).
    ``model_rows`` is the shared model value table, also by reference:
    values are distilled per rostered id on demand, never copied
    wholesale into a per-viewer dict the way the old preamble did.
    """
    if players is None:
        players = _players_index()
    if model_rows is None:
        model_rows = _model_rows()
    if bye_map is None:
        bye_map = _team_bye_map(season)
    return {
        "viewer_user_id": viewer_user_id,
        "account_id": account_id,
        "season": int(season),
        "week": int(week),
        "leg": int(leg or week),
        "league_inputs": list(league_inputs or []),
        "players": players,
        "model_rows": model_rows,
        "bye_map": bye_map or {},
        "cache_key": (account_id or f"viewer:{viewer_user_id}",
                      int(season), int(week)),
        "_fmt_cache": {},
    }


def scan_total(league_inputs):
    return sum(1 for lg in league_inputs if str(lg.get("league_id") or ""))


def _player(base, pid):
    """One shared-index row by reference ({} when unknown)."""
    return (base["players"] or {}).get(str(pid)) or {}


def _bye_week(base, pl):
    """A player row's bye week: its own when the feed carries one,
    else derived from its team through the schedule bye map -- the
    exact semantics _pf_players_with_byes produced by copying the
    index, computed here per player with no copy."""
    bw = pl.get("bye_week")
    if bw is not None:
        return bw
    team = str(pl.get("team") or "")
    if not team:
        return None
    from utils.nfl import norm_sched_team
    return (base["bye_map"] or {}).get(norm_sched_team(team))


def raw_values_for(base, pids):
    """Raw model ``value`` for the wanted pid set only.

    One pass over the shared table keeping just the requested ids
    (floats; rows whose value does not parse are absent, matching the
    old preamble's value_by_pid construction). Nothing is retained
    beyond the returned small dict.
    """
    want = {str(p) for p in pids if p}
    out = {}
    if not want:
        return out
    for r in base["model_rows"] or []:
        if not isinstance(r, dict):
            continue
        rid = str(r.get("id") or "")
        if rid in want:
            try:
                out[rid] = float(r.get("value") or 0)
            except (TypeError, ValueError):
                pass
    return out


def _sleeper_owner_id(account_id, viewer_user_id, platform):
    """Mirror of routes.user_pages_bp.sleeper_owner_id_for_account:
    the session viewer id is only a Sleeper owner when linked to this
    account (duplicated here so this module never imports the routes
    package; the rule reads the same accounts source)."""
    if str(platform or "sleeper").lower() != "sleeper":
        return None
    uid = str(viewer_user_id or "").strip()
    if not uid:
        return None
    if not account_id:
        return uid
    try:
        from dashboard_services.accounts import list_account_platform_ids
        linked = {str(x) for x in list_account_platform_ids(account_id, "sleeper")}
    except Exception:
        return None
    return uid if uid in linked else None


# ── Per-league resolution ────────────────────────────────────────────

def resolve_league(lg, base):
    """Distill one league input into the small dict the surfaces use.

    Fetches league meta + rosters (+ users for account viewers) from
    the cached provider layer, matches the viewer's roster with the
    same rules the ctx path used, and returns None when the league
    cannot be read or the roster cannot be matched -- the caller
    counts it unscanned (incomplete), never fabricated. The raw
    payloads are released by the caller after distillation of the
    surfaces that need them; nothing here retains them beyond ``res``
    itself, whose roster rows are the provider's own small dicts.
    """
    from dashboard_services import platform_api

    lid = str(lg.get("league_id") or "")
    if not lid:
        return None
    plat = str(lg.get("platform") or "sleeper").lower()
    lg_season = int(lg.get("season") or base["season"])
    try:
        league_obj = platform_api.get_league(plat, lid, lg_season) or {}
        rosters = platform_api.get_rosters(plat, lid, lg_season) or []
    except Exception:
        logger.debug("[portfolio-light] league fetch failed league=%s",
                     lid, exc_info=True)
        return None
    if not rosters:
        return None
    account_id = base["account_id"]
    users = []
    if account_id:
        try:
            users = platform_api.get_users(plat, lid, lg_season) or []
        except Exception:
            users = []
    viewer_roster = None
    if account_id:
        try:
            from dashboard_services.accounts import (
                resolve_account_viewer_for_league,
            )
            av = resolve_account_viewer_for_league(
                account_id, plat, lid, lg_season, users, rosters)
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
            rosters, team_id=lg.get("team_id"),
            owner_id=_sleeper_owner_id(
                account_id, base["viewer_user_id"], plat))
    if not viewer_roster:
        return None
    return {
        "plat": plat,
        "lid": lid,
        "lg_season": lg_season,
        "league_obj": league_obj,
        "rosters": rosters,
        "users": users,
        "viewer_roster": viewer_roster,
        "league_name": league_obj.get("name") or lg.get("name") or lid,
        "league_key": f"{plat}|{lid}",
        "starters": [str(p) for p in (viewer_roster.get("starters") or []) if p],
        "roster_positions": league_obj.get("roster_positions") or [],
        "scoring_settings": league_obj.get("scoring_settings") or {},
    }


# ── Bye Watch ────────────────────────────────────────────────────────

def bye_target_week(base):
    """The week Bye Watch must cover: the first week whose games have
    not kicked off yet. (Same rule as the old preamble's: Sleeper
    advances state.week days before the first game, so a flat week+1
    skips the very week users are setting lineups for; with no
    schedule data to prove either way, keep the week+1 lookahead.)"""
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
            epochs.append(float(g.get("gameTime_epoch")))
        except (TypeError, ValueError):
            continue
    if epochs and min(epochs) > time.time():
        return week
    return week + 1


def bye_row_for(res, next_week, base):
    """Bye Watch row for one resolved league: (row, out_count) or
    None. Field-for-field the old _pf_bye_row_for; only the data
    source changed (light res + on-demand bye derivation + rostered-
    only raw values)."""
    from dashboard_services import portfolio_insights as _pins

    try:
        out = []
        for i, pid in enumerate(res["starters"]):
            pl = _player(base, pid)
            bw = _bye_week(base, pl)
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
        bench_pids = []
        for pid in (res["viewer_roster"].get("players") or []):
            spid = str(pid)
            if spid in starter_set or spid in reserve_set or spid in taxi_set:
                continue
            bench_pids.append(spid)
        bench_values = raw_values_for(base, bench_pids)
        bench = []
        for spid in bench_pids:
            pl = _player(base, spid)
            bench.append({
                "pid": spid,
                "name": pl.get("full_name") or pl.get("last_name") or spid,
                "pos": pl.get("position") or "",
                "value": bench_values.get(spid, 0.0),
            })
        row_in = {"name": res["league_name"], "platform": res["plat"],
                  "league_id": res["lid"], "season": res["lg_season"],
                  "out": out, "bench": bench, "top_fa": {}}
        row = _pins.bye_row(row_in)
        if row.get("hole_pos"):
            rostered = {str(p) for r in res["rosters"]
                        for p in (r.get("players") or [])}
            best = None
            for r in base["model_rows"] or []:
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
                    pl = _player(base, rpid)
                    best = {"pid": rpid, "value": rv,
                            "name": pl.get("full_name") or r.get("name") or rpid}
            if best:
                row_in["top_fa"] = {row["hole_pos"]: best}
                row = _pins.bye_row(row_in)
        return row, row.get("out_count") or 0
    except Exception:
        logger.debug("[portfolio-light] bye row failed", exc_info=True)
        return None


def bye_payload(base):
    """Assemble the Bye Watch payload (same shape as the old
    _compute_portfolio_bye). Leagues are distilled sequentially and
    their raw payloads released before the next fetch."""
    next_week = bye_target_week(base)
    total = scan_total(base["league_inputs"])
    bye_rows = []
    total_out = 0
    scanned = 0
    for lg in base["league_inputs"][:MAX_LEAGUES]:
        res = resolve_league(lg, base)
        if res is None:
            continue
        scanned += 1
        got = bye_row_for(res, next_week, base)
        if got:
            bye_rows.append(got[0])
            total_out += got[1]
        res = None  # release this league's raw payloads before the next
    return {
        "ok": True,
        "week": next_week,
        "season": base["season"],
        "bye": {"week": next_week, "rows": bye_rows, "total_out": total_out},
        "scan": {"total": total, "scanned": scanned,
                 "incomplete": scanned < total},
        "updated": datetime.now(timezone.utc).isoformat(),
    }


# ── Rooting Guide ────────────────────────────────────────────────────

def week_projections(base, res):
    """Current-week projections map for one league, flattened with
    the league's own scoring settings -- the same values the ctx's
    18-week proj_by_week bundle carries for this week (each week
    flattens independently from its own projections file)."""
    try:
        bundles = _projections_by_week(
            base["season"], base["week"], res["scoring_settings"])
        wk = bundles.get(base["week"]) or bundles.get(str(base["week"])) or {}
        if isinstance(wk, dict):
            return {str(k): v for k, v in (wk.get("projections") or {}).items()}
    except Exception:
        pass
    return {}


def rooting_for(res, base, include_live=True):
    """Rooting inputs for one resolved league: (starters, margin or
    None). Field-for-field the old _pf_rooting_for; live points come
    from the CURRENT week's provider matchups only, and with
    ``include_live=False`` no provider call happens at all (starters
    carry projections only, points honestly None -- the embedded
    pregame shape)."""
    week = base["week"]
    starters = res["starters"]
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
    proj_map = week_projections(base, res)
    starters_out = []
    for pid in starters:
        pl = _player(base, pid)
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


def rooting_payload(base, include_live=True):
    """Assemble the Rooting Guide payload (same shape as the old
    _compute_portfolio_insights, including the legacy empty bye block
    older clients were built against). Per-league work is SEQUENTIAL:
    one league's matchup rows are distilled and released before the
    next league is fetched, so transient memory stays at one league's
    rows instead of a pool's worth."""
    from dashboard_services import portfolio_insights as _pins
    from utils.utils import load_week_schedule

    season = base["season"]
    week = base["week"]
    games_norm = []
    try:
        sched = list(load_week_schedule(season, week) or [])
    except Exception:
        sched = []
    team_game = {}
    # The live overlay can only add scores/status, which cannot exist
    # before the week's first kickoff (and the scoreboard answers
    # future dates with the wrong slate, at real cost). Skip it while
    # the whole slate is ahead; once any game has kicked off, overlay
    # with a short timeout (the ScoreZone precedent).
    _epochs = []
    for g in sched:
        try:
            _epochs.append(float(g.get("gameTime_epoch")))
        except (TypeError, ValueError, AttributeError):
            continue
    _slate_ahead = bool(_epochs) and min(_epochs) > time.time()
    if not _slate_ahead and include_live:
        try:
            from dashboard_services.api import build_team_game_lookup, get_nfl_scores_for_date
            from zoneinfo import ZoneInfo as _ZI
            dates = set()
            for g in sched:
                ep = g.get("gameTime_epoch")
                if ep:
                    dates.add(datetime.fromtimestamp(
                        float(ep), _ZI("America/New_York")).strftime("%Y%m%d"))
            for d in dates:
                try:
                    body = get_nfl_scores_for_date(d, timeout=5) or {}
                    team_game.update(build_team_game_lookup(body))
                except Exception:
                    continue
        except Exception:
            team_game = {}
    for g in sched:
        home = str(g.get("home") or "")
        away = str(g.get("away") or "")
        gd = None
        for key in (home, away):
            if team_game.get(key):
                gd = team_game[key]
                break
        if gd:
            games_norm.append(_pins.normalize_game(gd))
        else:
            ep = g.get("gameTime_epoch")
            games_norm.append({
                "home": home, "away": away, "home_pts": None, "away_pts": None,
                "status": "pre", "game_status": "", "game_quarter": "",
                "game_clock": "",
                "epoch": float(ep) if ep else None,
            })

    total = scan_total(base["league_inputs"])
    all_starters = []
    margins = {}
    scanned = 0
    for lg in base["league_inputs"][:MAX_LEAGUES]:
        res = resolve_league(lg, base)
        if res is None:
            continue
        scanned += 1
        starters_out, margin = rooting_for(res, base, include_live=include_live)
        all_starters.extend(starters_out)
        if margin is not None:
            margins[res["league_key"]] = margin
        res = None  # release this league's raw payloads before the next

    try:
        from app import _NFL_COLORS as _team_colors
    except Exception:
        _team_colors = {}
    windows = _pins.group_rooting(games_norm, all_starters,
                                  team_colors=_team_colors)
    if windows:
        final_window = windows[-1]
        final_teams = set()
        any_live_left = False
        for card in final_window.get("games") or []:
            final_teams.add(card.get("away"))
            final_teams.add(card.get("home"))
            if card.get("status") != "final":
                any_live_left = True
        for s in all_starters:
            if s.get("team") in final_teams and s.get("league_key") in margins:
                margins[s["league_key"]]["remaining_names"].append(s.get("name"))
        # Need lines only while the final window can still change outcomes.
        final_window["need_lines"] = (
            _pins.need_lines(margins) if any_live_left else [])
    return {
        "ok": True,
        "week": week,
        "season": season,
        "rooting": {"windows": windows},
        # Bye Watch lives on /api/portfolio-bye now; the empty block keeps
        # the response shape older clients were built against.
        "bye": {"week": week + 1, "rows": [], "total_out": 0},
        "scan": {"total": total, "scanned": scanned,
                 "incomplete": scanned < total},
        "updated": datetime.now(timezone.utc).isoformat(),
    }


# ── League card summaries + Player Holdings ──────────────────────────

def format_values(res, base):
    """pid -> format-adjusted value row for one league, via the same
    league_format_value_lookup the card builder uses, run on a
    minimal settings dict (platform + league meta + roster positions
    + the shared model table by reference). Memoized per base by
    scoring signature through the lookup's own cache, so same-type
    leagues in one pass share one table walk."""
    from dashboard_services.ai.context_builders import league_format_value_lookup
    pseudo_ctx = {
        "platform": res["plat"],
        "model_value_table": base["model_rows"],
        "roster_positions": res["roster_positions"],
        "league": res["league_obj"],
        "scoring_settings": res["scoring_settings"],
    }
    return league_format_value_lookup(pseudo_ctx, _cache=base["_fmt_cache"])


def _record_and_rank(res):
    """(wins, losses, ties, pf, rank) from roster settings + a
    wins-then-PF sort of the league's rosters -- the same reads
    portfolio_record_and_rank makes from settings and its roster
    fallback. (The ctx path prefers the playoff-seed standings_map
    when one exists; seeds need division records, which need the
    season frame this module exists to avoid, so division leagues'
    seed-vs-sort order is the one documented divergence.)"""
    settings = (res["viewer_roster"] or {}).get("settings") or {}
    wins = int(settings.get("wins") or 0)
    losses = int(settings.get("losses") or 0)
    ties = int(settings.get("ties") or 0)
    pf = (float(settings.get("fpts") or 0)
          + float(settings.get("fpts_decimal") or 0) / 100.0)

    def _key(r):
        s = (r or {}).get("settings") or {}
        return (-int(s.get("wins") or 0),
                -(float(s.get("fpts") or 0)
                  + float(s.get("fpts_decimal") or 0) / 100.0))

    ordered = sorted(res["rosters"] or [], key=_key)
    rid = str(res["viewer_roster"].get("roster_id"))
    rank = next((i + 1 for i, r in enumerate(ordered)
                 if str(r.get("roster_id")) == rid), "?")
    return wins, losses, ties, pf, rank


def recent_streak(res, base):
    """Last <=3 finalized results (W/L/T, chronological) for the
    viewer's roster -- the same output portfolio_summary.recent_streak
    reads off the season df_weekly, reproduced from the provider
    matchups of just the finalized weeks it can touch (weeks before
    the current leg), never the whole season frame."""
    leg = int(base.get("leg") or base["week"])
    rid = str(res["viewer_roster"].get("roster_id"))
    out = []
    try:
        from dashboard_services.platform_api import get_matchups
    except Exception:
        return out
    for w in range(max(1, leg - 3), leg):
        try:
            rows = get_matchups(res["plat"], res["lid"], w,
                                res["lg_season"]) or []
        except Exception:
            continue
        mine = next((r for r in rows if str(r.get("roster_id")) == rid), None)
        if not mine:
            continue
        mid = mine.get("matchup_id")
        opp = next((r for r in rows
                    if r is not mine and mid is not None
                    and r.get("matchup_id") == mid), None)
        if not opp:
            continue
        try:
            pts = float(mine.get("points") or 0)
            against = float(opp.get("points") or 0)
        except (TypeError, ValueError):
            continue
        out.append("W" if pts > against else ("L" if pts < against else "T"))
    return out[-3:]


def card_summary(res, base):
    """The distilled league-card summary the portfolio page renders:
    record, standing, streak, total value, the per-player value map
    the cross-league aggregates fold over, and the raw inputs the
    posture tag classifies from. Mirrors the field set of the inline
    _league_summary builder (minus the playoff-odds/posture attach,
    which stays cache-only in the board layer by design)."""
    wins, losses, ties, pf, rank = _record_and_rank(res)
    values_by_id = format_values(res, base)
    player_ids = [str(p) for p in (res["viewer_roster"].get("players") or [])]
    all_players = {}
    total_value = 0.0
    for pid in player_ids:
        v = values_by_id.get(pid) or {}
        val = float(v.get("value") or 0)
        total_value += val
        meta = _player(base, pid)
        all_players[pid] = {
            "name": v.get("name") or meta.get("full_name") or f"Player {pid}",
            "position": (v.get("position") or meta.get("position") or "").upper(),
            "value": val,
            "pos_rank": v.get("pos_rank_label") or "",
            "nfl_team": (v.get("team") or meta.get("team") or "").upper(),
        }
    return {
        "league_id": res["lid"],
        "name": res["league_name"],
        "platform": res["plat"],
        "season": res["lg_season"],
        "wins": wins, "losses": losses, "ties": ties,
        "record": f"{wins}-{losses}" + (f"-{ties}" if ties else ""),
        "rank": rank,
        "total_teams": len(res["rosters"] or []),
        "pf": round(pf, 1),
        "total_value": round(total_value, 1),
        "all_players": all_players,
        "streak": recent_streak(res, base),
        "posture_inputs": {
            "wins": wins, "losses": losses, "rank": rank,
            "total_teams": len(res["rosters"] or []), "pf": round(pf, 1),
        },
    }


def resolve_cards(base):
    """(cards, scanned) for every resolvable league, sequentially."""
    cards = []
    scanned = 0
    for lg in base["league_inputs"][:MAX_LEAGUES]:
        res = resolve_league(lg, base)
        if res is None:
            continue
        scanned += 1
        cards.append(card_summary(res, base))
        res = None
    return cards, scanned


def holdings_rows(cards, base):
    """Player Holdings rows across leagues, aggregated exactly the
    way page_portfolio folds the card summaries: per pid keep the
    highest-value league entry, count shares, sort value desc
    (shares break ties), then join 7-day rank movement from the model
    table and bye/injury from the shared players index."""
    pid_meta = {}
    pid_leagues = {}
    for card in cards:
        lg_abbrev = (card.get("name") or "?")[:14]
        for pid, p in (card.get("all_players") or {}).items():
            if not p.get("value"):
                continue
            if pid not in pid_meta or p.get("value", 0) > pid_meta[pid].get("value", 0):
                pid_meta[pid] = p
            pid_leagues.setdefault(pid, []).append(lg_abbrev)
    holdings = []
    for pid, p in pid_meta.items():
        if p.get("value", 0) > 0:
            holdings.append({
                **p, "pid": pid,
                "shares": len(pid_leagues.get(pid, [])),
                "in_leagues": pid_leagues.get(pid, []),
            })
    holdings.sort(key=lambda x: (-x["value"], -x["shares"]))
    rc_by_pid = {}
    for r in base["model_rows"] or []:
        if isinstance(r, dict) and r.get("id"):
            rc_by_pid[str(r.get("id"))] = r.get("rank_change_7d")
    for h in holdings:
        if h.get("rank_change_7d") in (None, 0):
            rc = rc_by_pid.get(str(h.get("pid")))
            if rc is not None:
                h["rank_change_7d"] = rc
        pl = _player(base, h.get("pid"))
        bw = _bye_week(base, pl)
        if bw is not None:
            h["bye_week"] = bw
        st = str(pl.get("injury_status") or "").strip()
        if st and st.lower() not in ("active", "act"):
            h["status"] = st
    return holdings
