"""Lightweight portfolio resolver (dashboard_services/portfolio_light):

- the bye/rooting computes make ZERO calls into the heavy league-ctx
  machinery (no get_league_ctx(_from_cache), no build_tables, no
  finalize_team_stats) -- the dependency that made Rooting Guide and
  Bye Watch load only when another page had warmed a full context;
- old-vs-new payload parity on the same fixture inputs: the light
  per-league builders must produce exactly what the ctx-based
  builders produced;
- a league whose provider fetch fails is reported unscanned /
  incomplete, never fabricated;
- card summaries match portfolio_record_and_rank's settings reads,
  and holdings fold the way the page folds card summaries.
"""
import pathlib

import pytest

pytest.importorskip("flask")

ROOT = pathlib.Path(__file__).resolve().parents[1]

# Same real-shape fixtures as the endpoint suite: Sleeper player rows
# carry NO bye_week; byes derive from the team bye map.
_PLAYERS = {
    "s1": {"full_name": "Starter One", "position": "RB", "team": "BUF"},
    "s2": {"full_name": "Starter Two", "position": "WR", "team": "BUF"},
    "s3": {"full_name": "Texan Starter", "position": "WR", "team": "HOU"},
    "b1": {"full_name": "Bench One", "position": "RB", "team": "MIA"},
    "fa1": {"full_name": "Free Agent", "position": "RB", "team": "NE"},
}
_BYE_MAP = {"BUF": 8, "MIA": 5, "NE": 5, "HOU": 8, "WAS": 9}
_MODEL_ROWS = [{"id": "fa1", "position": "RB", "value": 500.0,
                "name": "Free Agent", "rank_change_7d": 3}]
_LEAGUES = [
    {"league_id": "L1", "platform": "sleeper", "season": 2026,
     "name": "Alpha"},
    {"league_id": "L2", "platform": "sleeper", "season": 2026,
     "name": "Beta"},
]


def _ctx(starter, league_name):
    return {
        "rosters": [{
            "roster_id": 1, "owner_id": "owner-1", "starters": [starter],
            "players": [starter, "b1"], "reserve": [], "taxi": [],
        }],
        "league": {"name": league_name, "settings": {}},
        "users": [],
        "roster_positions": ["RB", "BN"],
        "proj_by_week": {7: {"projections": {starter: 12.0}}},
    }


def _wire(monkeypatch, ctx_by_lid):
    """Stub BOTH seams over the same fixture data: the legacy ctx seam
    (for the old builders) and the provider layer (for the light
    resolver)."""
    import dashboard_services.api as ds_api
    import dashboard_services.platform_api as platform_api
    import dashboard_services.portfolio_light as pf_light
    import routes.user_pages_bp as bp_mod
    import utils.utils as uu

    monkeypatch.setattr(
        bp_mod, "get_league_ctx_from_cache",
        lambda plat, lid, season, allow_build=False: ctx_by_lid.get(lid, {}))
    monkeypatch.setattr(bp_mod, "_team_bye_map",
                        lambda season: dict(_BYE_MAP))
    monkeypatch.setattr(ds_api, "get_nfl_players",
                        lambda *a, **k: dict(_PLAYERS))
    monkeypatch.setattr(ds_api, "get_nfl_scores_for_date", lambda d, **k: {})
    monkeypatch.setattr(ds_api, "build_team_game_lookup", lambda body: {})
    monkeypatch.setattr(uu, "load_week_schedule", lambda season, week: [])

    def _stub_league(plat, lid, season):
        ctx = ctx_by_lid.get(lid) or {}
        league = dict(ctx.get("league") or {})
        if ctx:
            league.setdefault("roster_positions",
                              ctx.get("roster_positions") or [])
            league.setdefault("scoring_settings", {})
        return league

    monkeypatch.setattr(platform_api, "get_league", _stub_league)
    monkeypatch.setattr(
        platform_api, "get_rosters",
        lambda plat, lid, season: list(
            (ctx_by_lid.get(lid) or {}).get("rosters") or []))
    monkeypatch.setattr(
        platform_api, "get_users",
        lambda plat, lid, season: list(
            (ctx_by_lid.get(lid) or {}).get("users") or []))
    _proj = {}
    for _c in ctx_by_lid.values():
        for _w, _bundle in (_c.get("proj_by_week") or {}).items():
            _entry = _proj.setdefault(_w, {"projections": {}})
            _entry["projections"].update(
                (_bundle or {}).get("projections") or {})
    monkeypatch.setattr(
        pf_light, "_projections_by_week",
        lambda season, weeks, scoring: dict(_proj))
    return bp_mod, pf_light, platform_api


def _old_base(bp_mod):
    players = bp_mod._pf_players_with_byes(dict(_PLAYERS), 2026)
    values = {}
    for r in _MODEL_ROWS:
        values[str(r["id"])] = float(r.get("value") or 0)
    return {
        "viewer_user_id": "owner-1", "account_id": None,
        "season": 2026, "week": 7, "league_inputs": list(_LEAGUES),
        "nfl_players": players, "model_rows": list(_MODEL_ROWS),
        "value_by_pid": values,
        "cache_key": ("viewer:owner-1", 2026, 7),
    }


def _light_base(pf_light):
    return pf_light.make_base(
        viewer_user_id="owner-1", account_id=None, season=2026, week=7,
        league_inputs=list(_LEAGUES), players=dict(_PLAYERS),
        model_rows=list(_MODEL_ROWS), bye_map=dict(_BYE_MAP))


def _matchup_rows(plat, lid, week, season):
    return [
        {"roster_id": 1, "matchup_id": 1,
         "players_points": {"s1": 10.0, "s2": 9.0, "s3": 4.0}},
        {"roster_id": 2, "matchup_id": 1, "players_points": {"o1": 7.0}},
    ]


# ── Zero heavy calls ─────────────────────────────────────────────────

def test_light_computes_make_zero_heavy_calls(monkeypatch):
    """The whole point of the resolver: bye + rooting computes run
    end to end with the heavy machinery rigged to explode -- no
    league ctx reads or builds, no season frame, no finalize."""
    ctxs = {"L1": _ctx("s1", "Alpha"), "L2": _ctx("s2", "Beta")}
    bp_mod, pf_light, platform_api = _wire(monkeypatch, ctxs)
    monkeypatch.setattr(platform_api, "get_matchups", _matchup_rows)

    import dashboard_services.service as svc

    calls = {"ctx": 0, "finalize": 0, "tables": 0}

    def _boom_ctx(*a, **k):
        calls["ctx"] += 1
        raise AssertionError("light path must not read a league ctx")

    def _boom_fin(*a, **k):
        calls["finalize"] += 1
        raise AssertionError("light path must not finalize team stats")

    def _boom_tables(*a, **k):
        calls["tables"] += 1
        raise AssertionError("light path must not build season tables")

    monkeypatch.setattr(bp_mod, "get_league_ctx_from_cache", _boom_ctx)
    monkeypatch.setattr(svc, "finalize_team_stats", _boom_fin)
    monkeypatch.setattr(svc, "build_tables", _boom_tables)

    base = _light_base(pf_light)
    bye = pf_light.bye_payload(base)
    assert bye["scan"] == {"total": 2, "scanned": 2, "incomplete": False}
    rooting = pf_light.rooting_payload(base)
    assert rooting["scan"] == {"total": 2, "scanned": 2,
                               "incomplete": False}
    assert calls == {"ctx": 0, "finalize": 0, "tables": 0}


# ── Old-vs-new parity ────────────────────────────────────────────────

def test_bye_payload_parity_old_vs_light(monkeypatch):
    """Same fixture through the legacy ctx builders (assembled the
    way the old _compute_portfolio_bye assembled them) and through
    the light resolver: byte-identical payload apart from the
    ``updated`` stamp."""
    ctxs = {"L1": _ctx("s1", "Alpha"), "L2": _ctx("s2", "Beta")}
    bp_mod, pf_light, _pa = _wire(monkeypatch, ctxs)

    old_base = _old_base(bp_mod)
    next_week = bp_mod._pf_bye_target_week(old_base)
    rows, total_out, scanned = [], 0, 0
    for lg in old_base["league_inputs"]:
        res = bp_mod._pf_resolve_league(lg, old_base)
        assert res is not None
        scanned += 1
        got = bp_mod._pf_bye_row_for(res, next_week, old_base)
        if got:
            rows.append(got[0])
            total_out += got[1]
    old_payload = {
        "ok": True, "week": next_week, "season": 2026,
        "bye": {"week": next_week, "rows": rows, "total_out": total_out},
        "scan": {"total": 2, "scanned": scanned, "incomplete": False},
    }
    new_payload = pf_light.bye_payload(_light_base(pf_light))
    new_payload = {k: v for k, v in new_payload.items() if k != "updated"}
    assert new_payload == old_payload


def test_rooting_per_league_parity_old_vs_light(monkeypatch):
    """Per-league rooting (starters + margin) identical between the
    ctx builder and the light builder, live and pregame shapes."""
    ctxs = {"L1": _ctx("s1", "Alpha"), "L2": _ctx("s2", "Beta")}
    bp_mod, pf_light, platform_api = _wire(monkeypatch, ctxs)
    monkeypatch.setattr(platform_api, "get_matchups", _matchup_rows)

    old_base = _old_base(bp_mod)
    light_base = _light_base(pf_light)
    for lg in _LEAGUES:
        old_res = bp_mod._pf_resolve_league(lg, old_base)
        light_res = pf_light.resolve_league(lg, light_base)
        assert light_res is not None and old_res is not None
        for include_live in (True, False):
            old_out = bp_mod._pf_rooting_for(
                old_res, old_base, include_live=include_live)
            new_out = pf_light.rooting_for(
                light_res, light_base, include_live=include_live)
            assert new_out == old_out


def test_rooting_full_payload_parity_old_vs_light(monkeypatch):
    """The assembled Rooting Guide payload (windows, margins-driven
    need lines, legacy bye block, scan) matches the old assembly on
    the same schedule/matchups."""
    import time as _time
    from datetime import datetime
    from zoneinfo import ZoneInfo

    import utils.utils as uu

    future = _time.time() + 2 * 86400
    sched = [{"home": "BUF", "away": "NYJ",
              "gameTime_epoch": future}]
    ctxs = {"L1": _ctx("s1", "Alpha"), "L2": _ctx("s2", "Beta")}
    bp_mod, pf_light, platform_api = _wire(monkeypatch, ctxs)
    monkeypatch.setattr(uu, "load_week_schedule",
                        lambda season, week: list(sched))
    monkeypatch.setattr(platform_api, "get_matchups", _matchup_rows)

    # Old assembly: the pre-consolidation _compute_portfolio_insights
    # body, replayed from its own pieces (schedule games + per-league
    # _pf_rooting_for + group_rooting), in league order.
    from dashboard_services import portfolio_insights as _pins

    old_base = _old_base(bp_mod)
    games_norm = [{
        "home": "BUF", "away": "NYJ", "home_pts": None, "away_pts": None,
        "status": "pre", "game_status": "", "game_quarter": "",
        "game_clock": "", "epoch": float(future),
    }]
    all_starters, margins, scanned = [], {}, 0
    for lg in old_base["league_inputs"]:
        res = bp_mod._pf_resolve_league(lg, old_base)
        assert res is not None
        scanned += 1
        starters_out, margin = bp_mod._pf_rooting_for(res, old_base)
        all_starters.extend(starters_out)
        if margin is not None:
            margins[res["league_key"]] = margin
    windows = _pins.group_rooting(games_norm, all_starters,
                                  team_colors={})
    if windows:
        any_live = any(g.get("status") != "final"
                       for g in windows[-1].get("games") or [])
        windows[-1]["need_lines"] = (
            _pins.need_lines(margins) if any_live else [])
    old_payload = {
        "ok": True, "week": 7, "season": 2026,
        "rooting": {"windows": windows},
        "bye": {"week": 8, "rows": [], "total_out": 0},
        "scan": {"total": 2, "scanned": scanned, "incomplete": False},
    }
    new_payload = pf_light.rooting_payload(_light_base(pf_light))
    new_payload = {k: v for k, v in new_payload.items() if k != "updated"}
    assert new_payload == old_payload
    # Sanity on the shared expectation itself (a Sunday 1pm ET epoch
    # was used above via `future`; the window groups both starters).
    names = [s["name"] for s in windows[0]["games"][0]["sides"]["home"]]
    assert names == ["Starter One", "Starter Two"]


# ── Provider failure is honest ───────────────────────────────────────

def test_provider_failure_league_is_incomplete_never_fabricated(monkeypatch):
    ctxs = {"L1": _ctx("s1", "Alpha"), "L2": _ctx("s2", "Beta")}
    bp_mod, pf_light, platform_api = _wire(monkeypatch, ctxs)

    orig_rosters = platform_api.get_rosters

    def _flaky(plat, lid, season):
        if lid == "L2":
            raise RuntimeError("provider down")
        return orig_rosters(plat, lid, season)

    monkeypatch.setattr(platform_api, "get_rosters", _flaky)
    base = _light_base(pf_light)
    bye = pf_light.bye_payload(base)
    assert bye["scan"] == {"total": 2, "scanned": 1, "incomplete": True}
    assert [r["name"] for r in bye["bye"]["rows"]] == ["Alpha"]
    rooting = pf_light.rooting_payload(base)
    assert rooting["scan"] == {"total": 2, "scanned": 1,
                               "incomplete": True}


# ── Cards + holdings ─────────────────────────────────────────────────

def _card_rosters():
    return [
        {"roster_id": 1, "owner_id": "owner-1",
         "players": ["s1", "b1"], "starters": ["s1"],
         "reserve": [], "taxi": [],
         "settings": {"wins": 5, "losses": 2, "ties": 0,
                      "fpts": 700, "fpts_decimal": 50}},
        {"roster_id": 2, "owner_id": "owner-2",
         "players": ["s2"], "starters": ["s2"],
         "reserve": [], "taxi": [],
         "settings": {"wins": 3, "losses": 4, "ties": 0,
                      "fpts": 610, "fpts_decimal": 0}},
        {"roster_id": 3, "owner_id": "owner-3",
         "players": ["s3"], "starters": ["s3"],
         "reserve": [], "taxi": [],
         "settings": {"wins": 6, "losses": 1, "ties": 0,
                      "fpts": 640, "fpts_decimal": 25}},
    ]


def test_card_summary_matches_record_and_rank_settings_reads(monkeypatch):
    """portfolio_record_and_rank over a settings-only ctx (no
    team_stats, no standings_map: its documented roster fallback)
    vs the light card summary: same record, PF and rank."""
    import dashboard_services.portfolio_light as pf_light
    from dashboard_services.ai.context_builders import (
        portfolio_record_and_rank,
    )

    rosters = _card_rosters()
    viewer = rosters[0]
    ctx = {"standings_map": {}, "rosters": rosters, "roster_map": {}}
    wins, losses, ties, pf, rank = portfolio_record_and_rank(
        ctx, "1", viewer)

    league_obj = {"name": "Alpha", "roster_positions": ["RB", "BN"],
                  "scoring_settings": {}, "settings": {"type": 2}}
    base = pf_light.make_base(
        viewer_user_id="owner-1", account_id=None, season=2026, week=7,
        league_inputs=[], players=dict(_PLAYERS),
        model_rows=[], bye_map=dict(_BYE_MAP))
    res = {
        "plat": "sleeper", "lid": "L1", "lg_season": 2026,
        "league_obj": league_obj, "rosters": rosters, "users": [],
        "viewer_roster": viewer, "league_name": "Alpha",
        "league_key": "sleeper|L1", "starters": ["s1"],
        "roster_positions": ["RB", "BN"], "scoring_settings": {},
    }
    card = pf_light.card_summary(res, base)
    assert (card["wins"], card["losses"], card["ties"]) == (wins, losses, ties)
    assert card["pf"] == round(pf, 1)
    assert card["rank"] == rank
    assert card["record"] == "5-2"
    assert card["posture_inputs"]["rank"] == rank


def test_card_streak_from_finalized_week_matchups(monkeypatch):
    """Streak reads only the last finalized weeks' matchups: W (wk5),
    L (wk6), W (wk7) at leg 8, chronological, capped at 3."""
    import dashboard_services.platform_api as platform_api
    import dashboard_services.portfolio_light as pf_light

    results = {5: (100.0, 90.0), 6: (80.0, 95.0), 7: (120.0, 110.0)}

    def _rows(plat, lid, week, season):
        mine, opp = results[week]
        return [
            {"roster_id": 1, "matchup_id": 1, "points": mine},
            {"roster_id": 2, "matchup_id": 1, "points": opp},
        ]

    monkeypatch.setattr(platform_api, "get_matchups", _rows)
    base = pf_light.make_base(
        viewer_user_id="owner-1", account_id=None, season=2026, week=8,
        league_inputs=[], players={}, model_rows=[], bye_map={}, leg=8)
    res = {"plat": "sleeper", "lid": "L1", "lg_season": 2026,
           "viewer_roster": {"roster_id": 1}}
    assert pf_light.recent_streak(res, base) == ["W", "L", "W"]


def test_holdings_fold_matches_page_aggregation(monkeypatch):
    """holdings_rows over light cards: highest-value entry per pid,
    shares counted across leagues, value-desc order, rank movement
    and bye/status joined -- the page's fold, on the same inputs."""
    import dashboard_services.portfolio_light as pf_light

    cards = [
        {"name": "Alpha League", "all_players": {
            "s1": {"name": "Starter One", "position": "RB",
                   "value": 300.0, "pos_rank": "", "nfl_team": "BUF"},
            "b1": {"name": "Bench One", "position": "RB",
                   "value": 100.0, "pos_rank": "", "nfl_team": "MIA"},
        }},
        {"name": "Beta League", "all_players": {
            "s1": {"name": "Starter One", "position": "RB",
                   "value": 450.0, "pos_rank": "", "nfl_team": "BUF"},
            "fa1": {"name": "Free Agent", "position": "RB",
                    "value": 500.0, "pos_rank": "", "nfl_team": "NE"},
        }},
    ]
    base = pf_light.make_base(
        viewer_user_id="owner-1", account_id=None, season=2026, week=7,
        league_inputs=[], players=dict(_PLAYERS),
        model_rows=list(_MODEL_ROWS), bye_map=dict(_BYE_MAP))
    rows = pf_light.holdings_rows(cards, base)
    assert [r["pid"] for r in rows] == ["fa1", "s1", "b1"]
    by_pid = {r["pid"]: r for r in rows}
    assert by_pid["s1"]["value"] == 450.0  # highest-value league wins
    assert by_pid["s1"]["shares"] == 2
    assert by_pid["s1"]["in_leagues"] == ["Alpha League", "Beta League"]
    assert by_pid["fa1"]["rank_change_7d"] == 3  # joined from the table
    assert by_pid["s1"]["bye_week"] == 8  # BUF bye, derived on demand
    assert by_pid["b1"]["bye_week"] == 5
