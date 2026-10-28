"""Recap + Standings duplicate-work removal (sitewide audit, 2026-10-27).

Three findings, one contract each: identical displayed output, less work.

1. Recap live view: build_standings_as_of_week ran finalize_team_stats for
   the current and prior week even though nothing on the live path reads the
   capped team_stats (standings rows come from the capped df_weekly; power
   comes from build_power_rankings_context, which never reads team_stats).
   The capped builds now skip the finalize pass on the live view only;
   historical views keep it. Proven by call counts + byte-identical HTML
   against the forced-full-build render.
2. Standings power rankings: build_power_rankings_context runs on the base
   ctx and on week-capped copies that share the same rosters / model table /
   players objects. The df-independent profile (value lookup, totals,
   capital, per-roster grade profile) is computed once per league snapshot
   via _power_static_bundle. Proven by call counts + identical outputs.
   render_share_rankings reads the ctx model table's raw value columns
   instead of issuing its own player_values SELECT.
3. Season efficiency (Lineup Efficiency's engine): its result cache was
   in-process only with a 0s TTL for incomplete results, so an incomplete
   league re-paid the uncached provider refetch on every render. The cache
   now has a shared disk tier and an honest 60s TTL for incomplete results,
   bounded with oldest-first eviction on both tiers.
"""
import importlib

import pytest

pd = pytest.importorskip("pandas")
pytest.importorskip("flask")


def _prod_module(name):
    """The module object production's ``from <name> import ...`` resolves.

    ``import dashboard_services.service as svc`` binds through the parent
    package's attribute, which tests/test_matchup_cards_degrade.py's fixture
    leaves pointing at a stub-born re-import (it restores ``sys.modules``
    via monkeypatch but not the package attribute). app.py and recap_page
    resolve their call-time from-imports through ``sys.modules``, so patch
    the ``sys.modules`` entry — the object production actually calls —
    via importlib, which returns that entry even when the package
    attribute is stale.
    """
    return importlib.import_module(name)


# ── Finding 1: capped standings builder ─────────────────────────────────────

def _weekly_df():
    rows = []
    for wk in (1, 2, 3):
        for rid in (1, 2):
            rows.append({"week": wk, "roster_id": rid, "owner": f"Team {rid}",
                         "points": 100.0 + rid + wk,
                         "points_against": 90.0,
                         "matchup_id": 1, "finalized": True, "avatar": ""})
    return pd.DataFrame(rows)


def test_capped_standings_skip_finalize_keeps_identical_df_cap(monkeypatch):
    import app as appmod

    svc = _prod_module("dashboard_services.service")

    calls = {"n": 0}
    sentinel = object()

    def fake_finalize(*a, **k):
        calls["n"] += 1
        return sentinel

    monkeypatch.setattr(svc, "finalize_team_stats", fake_finalize)
    ctx = {"df_weekly": _weekly_df(), "team_stats": "ORIGINAL",
           "matchups_by_week": {}, "users": [], "league_settings": {}}

    capped = appmod.build_standings_as_of_week(ctx, 2, include_team_stats=False)
    assert calls["n"] == 0
    assert capped["team_stats"] == "ORIGINAL"
    assert sorted(capped["df_weekly"]["week"].unique().tolist()) == [1, 2]

    full = appmod.build_standings_as_of_week(ctx, 2)
    assert calls["n"] == 1
    assert full["team_stats"] is sentinel
    assert capped["df_weekly"].equals(full["df_weekly"])


def _recap_ctx():
    svc = _prod_module("dashboard_services.service")

    scores = {1: [100, 90, 110, 95], 2: [105, 99, 101, 88], 3: [120, 80, 90, 100]}
    rows = []
    for wk in (1, 2, 3):
        for i, rid in enumerate((1, 2, 3, 4)):
            rows.append({"week": wk, "roster_id": rid, "owner": f"Team {rid}",
                         "points": float(scores[wk][i]),
                         "points_against": float(scores[wk][3 - i]),
                         "matchup_id": (i // 2) + 1, "finalized": True, "avatar": ""})
    df = pd.DataFrame(rows)

    def side(rid, pts):
        return {"roster_id": str(rid), "name": f"Team {rid}",
                "lineup_is_historical": True, "starters": [], "bench": [],
                "pts_total": float(pts), "avatar": ""}

    mbw = {wk: [{"matchup_id": 1, "left": side(1, scores[wk][0]), "right": side(2, scores[wk][1])},
                {"matchup_id": 2, "left": side(3, scores[wk][2]), "right": side(4, scores[wk][3])}]
           for wk in (1, 2, 3)}
    return {
        "platform": "sleeper", "season": 2026, "league_id": "DEDUPE-L",
        "df_weekly": df,
        "roster_map": {str(i): f"Team {i}" for i in (1, 2, 3, 4)},
        "users": [], "league": {"settings": {"playoff_week_start": 14}},
        "league_settings": {"playoff_week_start": 14},
        "rosters": [{"roster_id": i, "owner_id": str(i), "players": [], "starters": [],
                     "settings": {"wins": 2, "losses": 1, "ties": 0,
                                  "fpts": 300, "fpts_decimal": 0}} for i in (1, 2, 3, 4)],
        "matchups_by_week": mbw,
        "roster_positions": ["QB", "RB", "WR", "FLEX"],
        "players_index": {}, "players_map": {},
        "model_value_table": [],
        "team_stats": svc.finalize_team_stats(df, {}, mbw, [], 3),
        "current_week": 4,
    }


def test_recap_live_view_skips_finalize_and_renders_identically(monkeypatch):
    import app as appmod
    from dashboard_services.pages.recap_page import build_recap_body

    eff = _prod_module("dashboard_services.season_efficiency")
    svc = _prod_module("dashboard_services.service")

    # Hermetic: no provider fetch, no disk cache tier.
    monkeypatch.setattr(
        eff, "compute_league_season_efficiency",
        lambda ctx: {"by_rid": {}, "state": "complete",
                     "completed_weeks": [1, 2, 3], "incomplete": {}})
    calls = {"n": 0}
    orig_finalize = svc.finalize_team_stats

    def counting(*a, **k):
        calls["n"] += 1
        return orig_finalize(*a, **k)

    monkeypatch.setattr(svc, "finalize_team_stats", counting)
    ctx = _recap_ctx()
    orig_bsw = appmod.build_standings_as_of_week

    with appmod.app.test_request_context("/"):
        calls["n"] = 0
        live_new = build_recap_body(dict(ctx), selected_week=3)
        assert calls["n"] == 0  # live view: both capped builds skip finalize

        monkeypatch.setattr(
            appmod, "build_standings_as_of_week",
            lambda c, w, **kw: orig_bsw(c, w, include_team_stats=True))
        calls["n"] = 0
        live_old = build_recap_body(dict(ctx), selected_week=3)
        assert calls["n"] == 2  # the pre-fix behavior paid both finalize passes
        assert live_new == live_old

        monkeypatch.setattr(appmod, "build_standings_as_of_week", orig_bsw)
        calls["n"] = 0
        hist_new = build_recap_body(dict(ctx), selected_week=2)
        assert calls["n"] == 2  # historical view still builds capped team_stats

        monkeypatch.setattr(
            appmod, "build_standings_as_of_week",
            lambda c, w, **kw: orig_bsw(c, w, include_team_stats=True))
        hist_old = build_recap_body(dict(ctx), selected_week=2)
        assert hist_new == hist_old


# ── Finding 3a: power-rankings static profile shared across passes ──────────

def _power_ctx(df):
    pos = ["QB", "RB", "RB", "WR", "WR", "TE", "RB", "WR"]
    table, pidx, pmap = [], {}, {}
    for rid in (1, 2, 3, 4):
        for j, p in enumerate(pos):
            pid = f"p{rid}{j}"
            base = (5 - rid) * 100 + (8 - j) * 37
            table.append({"id": pid, "position": p, "value": float(base),
                          "sf_value": float(base * 1.2), "model_value": float(base - 5),
                          "model_sf_value": float(base),
                          "redraft_value_1qb": float(base / 3),
                          "redraft_value_sf": float(base / 2),
                          "age": 24 + j, "name": f"Player {pid}", "team": "KC"})
            pidx[pid] = {"position": p, "age": 24 + j,
                         "injury_status": "Out" if (rid == 2 and j == 0) else ""}
            pmap[pid] = {"pos": p, "age": 24 + j}
    rosters = [{"roster_id": rid, "players": [f"p{rid}{j}" for j in range(8)],
                "settings": {"wins": 3, "losses": 1, "ties": 0,
                             "fpts": 400 + rid, "fpts_decimal": 0}}
               for rid in (1, 2, 3, 4)]
    return {"rosters": rosters,
            "roster_map": {str(r): f"Team {r}" for r in (1, 2, 3, 4)},
            "standings_map": {}, "model_value_table": table,
            "players_index": pidx, "players_map": pmap,
            "picks_by_roster": {"1": [{"display": "2027 1.01", "round": 1}],
                                "2": [], "3": [{"display": "2027 2.05", "round": 2}],
                                "4": []},
            "roster_positions": ["QB", "RB", "RB", "WR", "WR", "TE", "FLEX"],
            "league_settings": {"playoff_teams": 2}, "df_weekly": df,
            "current_week": 4, "current_season": 2026, "playoff_odds": [],
            "matchups_by_week": {}}


def _power_df():
    rows = []
    for wk in (1, 2, 3):
        for rid in (1, 2, 3, 4):
            rows.append({"week": wk, "roster_id": rid,
                         "points": 90.0 + rid * 3 + wk, "finalized": True,
                         "matchup_id": 1 if rid <= 2 else 2})
    return pd.DataFrame(rows)


def test_power_static_profile_runs_once_for_current_and_capped_pass(monkeypatch):
    import dashboard_services.ai.context_builders as cb

    counts = {"summarize": 0}
    orig_summarize = cb.summarize_roster_players

    def counting_summarize(*a, **k):
        counts["summarize"] += 1
        return orig_summarize(*a, **k)

    monkeypatch.setattr(cb, "summarize_roster_players", counting_summarize)
    df = _power_df()
    ctx_full = _power_ctx(df)
    ctx_cap = dict(ctx_full)
    ctx_cap["df_weekly"] = df[df["week"] <= 2].copy()

    cb._POWER_STATIC_CACHE["key"] = None
    full = cb.build_power_rankings_context(ctx_full)
    capped = cb.build_power_rankings_context(ctx_cap)
    # One profile build serves both passes (4 rosters), not one per pass (8).
    assert counts["summarize"] == 4

    # A cold-cache capped pass produces exactly the warm-cache output.
    cb._POWER_STATIC_CACHE["key"] = None
    capped_cold = cb.build_power_rankings_context(ctx_cap)
    assert capped == capped_cold
    assert full["teams"] != capped["teams"]  # the df cap still differentiates


# ── Finding 3b: share rankings read the ctx model table ─────────────────────

class _ValuesConn:
    def __init__(self, values):
        self._values = values

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, _sql):
        values = self._values

        class _Result:
            def fetchall(self):
                return [{"player_id": pid, "v": v} for pid, v in values.items()]

        return _Result()


def _shares_ctx(with_table):
    team_stats = pd.DataFrame([
        {"owner": "Bravo", "PF": 200.0},
        {"owner": "Alpha", "PF": 100.0},
        {"owner": "Charlie", "PF": 300.0},
    ])
    ctx = {
        "team_stats": team_stats,
        "rosters": [{"roster_id": 1, "players": ["a1"]},
                    {"roster_id": 2, "players": ["c1"]},
                    {"roster_id": 3, "players": ["b1"]}],
        "roster_map": {"1": "Alpha", "2": "Charlie", "3": "Bravo"},
        "df_weekly": None,
        "roster_positions": [],
    }
    if with_table:
        ctx["model_value_table"] = [
            {"id": "a1", "model_value": 300.0, "model_sf_value": 300.0},
            {"id": "c1", "model_value": 200.0, "model_sf_value": 200.0},
            {"id": "b1", "model_value": 100.0, "model_sf_value": 100.0},
        ]
    return ctx


def test_share_rankings_read_ctx_model_table_without_select(monkeypatch):
    import app as appmod
    import dashboard_services.db as db

    values = {"a1": 300.0, "c1": 200.0, "b1": 100.0}
    monkeypatch.setattr(db, "get_conn", lambda *a, **k: _ValuesConn(values))
    via_db = appmod.render_share_rankings(_shares_ctx(with_table=False))

    def forbidden_conn(*a, **k):
        raise AssertionError("render_share_rankings issued its own SELECT")

    monkeypatch.setattr(db, "get_conn", forbidden_conn)
    via_table = appmod.render_share_rankings(_shares_ctx(with_table=True))
    assert via_table == via_db


# ── Finding 2: season-efficiency cache tiers + honest incomplete TTL ────────

def _lineup_rows():
    return [{"week": 7, "roster_id": 1,
             "players": ["qb", "k", "DAL", "rb"],
             "starters": ["qb", "k", "DAL"],
             "players_points": {"qb": 10, "k": 0, "DAL": 6, "rb": 15},
             "points": 16}]


def _eff_ctx():
    df = pd.DataFrame([{"week": 7, "finalized": True}])
    return {"platform": "sleeper", "season": 2025, "league_id": "L",
            "df_weekly": df, "roster_positions": ["QB", "K", "DEF"],
            "rosters": [{"roster_id": 1}], "optimal_matchups_by_week": {},
            "efficiency_weeks": [7]}


def test_season_efficiency_repeat_calls_hit_cache_not_provider(monkeypatch, tmp_path):
    import app as appmod
    import dashboard_services.platform_api as api
    import dashboard_services.season_efficiency as service

    players = {"qb": {"position": "QB"}, "k": {"pos": "K"}, "rb": {"pos": "RB"}}
    monkeypatch.setattr(appmod, "get_players_index_global", lambda: players)
    fetches = {"n": 0}

    def counting_fetch(*a):
        fetches["n"] += 1
        return _lineup_rows()

    monkeypatch.setattr(api, "get_matchups", counting_fetch)
    service._CACHE.clear()
    monkeypatch.setattr(service, "_DISK_CACHE_FILE", tmp_path / "eff.json")

    first = service.compute_league_season_efficiency(_eff_ctx())
    assert first["state"] == "complete"
    assert fetches["n"] == 1

    second = service.compute_league_season_efficiency(_eff_ctx())
    assert second == first
    assert fetches["n"] == 1  # in-process tier

    service._CACHE.clear()  # simulate a worker recycle
    third = service.compute_league_season_efficiency(_eff_ctx())
    assert third == first
    assert fetches["n"] == 1  # disk tier


def test_season_efficiency_incomplete_result_not_refetched_within_ttl(monkeypatch, tmp_path):
    import json as _json

    import app as appmod
    import dashboard_services.platform_api as api
    import dashboard_services.season_efficiency as service

    players = {"qb": {"position": "QB"}, "k": {"pos": "K"}, "rb": {"pos": "RB"}}
    monkeypatch.setattr(appmod, "get_players_index_global", lambda: players)
    fetches = {"n": 0}

    def failing_fetch(*a):
        fetches["n"] += 1
        return None

    monkeypatch.setattr(api, "get_matchups", failing_fetch)
    service._CACHE.clear()
    disk_file = tmp_path / "eff.json"
    monkeypatch.setattr(service, "_DISK_CACHE_FILE", disk_file)

    first = service.compute_league_season_efficiency(_eff_ctx())
    assert first["state"] == "incomplete"
    assert fetches["n"] == 1

    # Previously TTL 0: every render refetched. Now the incomplete result is
    # served for its honest 60s TTL (and still reports itself incomplete).
    second = service.compute_league_season_efficiency(_eff_ctx())
    assert second["state"] == "incomplete"
    assert fetches["n"] == 1

    for entry in service._CACHE.values():
        entry["at"] -= service._INCOMPLETE_TTL + 1
    disk = _json.loads(disk_file.read_text())
    for entry in disk.values():
        entry["at"] -= service._INCOMPLETE_TTL + 1
    disk_file.write_text(_json.dumps(disk))

    third = service.compute_league_season_efficiency(_eff_ctx())
    assert third["state"] == "incomplete"
    assert fetches["n"] == 2  # TTL lapsed: provider retried
