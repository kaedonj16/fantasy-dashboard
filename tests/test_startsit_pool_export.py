"""Start/sit live pool export: app.build_start_sit_pool_export + the
CRON_SECRET-gated GET /api/start-sit-pool-export.

The export replaces the manual Render-shell paste that produced
startsit_live_pool_w{W}_ppr.json for the Thursday boards. The builder
loads the PPR bundles and scores them with the SAME inputs
start_score_pool_scores feeds compute_start_score (last-4 blend,
played-games shrinkage, _matchup_rank_table matchup ranks, live status
overlay + pool exclusion), so the fakes below mirror the real
signatures of load_start_score_bundles / variant_for_exact_scoring
(data_building.start_score_bundle) and compute_start_score
(utils.start_sit_score) exactly -- an extra or missing kwarg in the
builder makes the fake raise, the row is skipped, and the shaping
assertions fail. The parity section at the bottom runs the REAL
compute_start_score through both paths and asserts identical values.
"""
from __future__ import annotations

import datetime as _dt

import pytest

pytest.importorskip("flask")
pytest.importorskip("pandas")


FAKE_INDEX = {
    "1": {"full_name": "Alpha Back", "team": "DET"},
    "2": {"full_name": "Beta Back", "team": "MIA"},
    "3": {"full_name": "Gamma Back", "team": "BUF"},
    "5": {"full_name": "Zero Wideout", "team": "NYJ"},
    "6": {"full_name": "Boom Bust", "team": "DAL"},
    "7": {"full_name": "Kicker Guy", "team": "KC"},
    "9": {"full_name": "Top Passer", "team": "CIN"},
}


def _bundle(pos, proj, **over):
    b = {
        "pos": pos,
        "proj_pts": proj,
        "on_bye": False,
        "recent_ppg": 10.0,
        "season_ppg": 9.0,
        "last4_ppg": None,
        "played_games": None,
        "opponent": None,
        "usage_delta": None,
        "usage_season_avg": None,
        "injury_status": "",
        "implied_total": None,
        "bust_rate": None,
        "weather_kind": None,
        "oline_index": None,
        "expected_team_plays": None,
        "league_average_plays": None,
        "role_confidence": None,
        "snap_trend": None,
        "low_total_team": False,
    }
    b.update(over)
    return b


def _fake_compute_start_score(
    proj_pts,
    *,
    on_bye=False,
    recent_ppg=0.0,
    season_ppg=0.0,
    last4_ppg=None,
    def_rank=None,
    def_total=32,
    usage_delta=None,
    usage_season_avg=None,
    injury_status=None,
    played_games=None,
    implied_total=None,
    bust_rate=None,
    weather_kind=None,
    position=None,
    oline_index=None,
    expected_team_plays=None,
    league_average_plays=None,
    role_confidence=None,
    snap_trend=None,
    low_total_team=False,
):
    """Mirror of the builder's call surface: score = 2 x projection.

    proj == 13.37 is the poison row: the real compute raises on a bad
    bundle, and the builder must skip just that player. Every call's
    kwargs are recorded on ``.seen`` so tests can assert the unified
    inputs (last4_ppg / played_games / def_rank) actually flow through.
    """
    _fake_compute_start_score.seen.append({
        "proj_pts": proj_pts, "last4_ppg": last4_ppg,
        "played_games": played_games, "def_rank": def_rank,
        "def_total": def_total, "injury_status": injury_status,
    })
    if proj_pts == 13.37:
        raise ValueError("bad bundle")
    return (float(proj_pts) * 2.0, {}, None)


_fake_compute_start_score.seen = []


@pytest.fixture()
def export_env(monkeypatch):
    """Fake bundle layer + index + tier index around the real builder."""
    import app as appmod
    import data_building.start_score_bundle as bundle_mod
    import utils.start_sit_score as score_mod

    calls = {"load": [], "variant": []}
    state = {"bundles": {}}

    def fake_variant(scoring):
        calls["variant"].append(scoring)
        return "ppr"

    def fake_load(season, week, variant, player_ids):
        calls["load"].append((season, week, variant, list(player_ids)))
        return dict(state["bundles"])

    monkeypatch.setattr(bundle_mod, "variant_for_exact_scoring", fake_variant)
    monkeypatch.setattr(bundle_mod, "load_start_score_bundles", fake_load)
    monkeypatch.setattr(score_mod, "compute_start_score",
                        _fake_compute_start_score)
    _fake_compute_start_score.seen = []
    monkeypatch.setattr(appmod, "get_players_index_global",
                        lambda: dict(FAKE_INDEX))
    # Deterministic matchup ranks: DAL is the softest RB matchup here.
    monkeypatch.setattr(
        appmod, "_matchup_rank_table",
        lambda season, position, scoring_settings=None: (
            {"DAL": 32, "NE": 20, "SF": 4}, 32, {}, False))
    # Deterministic tier index: rank 1 -> 100, rank 2 -> 99, ...
    monkeypatch.setattr(appmod, "start_score_tier_index",
                        lambda pos, rank: 101 - rank)
    appmod._SS_EXPORT_CACHE.clear()
    appmod._SS_EXPORT_CACHE_TS.clear()
    yield calls, state
    appmod._SS_EXPORT_CACHE.clear()
    appmod._SS_EXPORT_CACHE_TS.clear()


# ── (a) Builder shaping / ranking / index ──────────────────────────────────

def test_export_ranks_ties_and_shape(export_env):
    import app as appmod

    calls, state = export_env
    state["bundles"] = {
        "1": _bundle("RB", 15.0),
        "2": _bundle("RB", 15.0),                    # tie with pid 1
        "3": _bundle("RB", 10.0, injury_status="Questionable"),
        "9": _bundle("QB", 20.0),
        "5": _bundle("WR", 0.0),                     # score 0 -> excluded
        "6": _bundle("WR", 13.37),                   # compute raises -> skipped
        "7": _bundle("K", 12.0),                     # not an export position
        "8": "not-a-bundle",                         # non-dict -> skipped
    }
    out = appmod.build_start_sit_pool_export(2026, 10)

    assert set(out.keys()) == {"QB", "RB", "WR", "TE"}
    rb = out["RB"]
    assert [r["pid"] for r in rb] == ["1", "2", "3"]
    # Competition ranking: the tie shares rank 1, the next rank is 3.
    assert [r["index"] for r in rb] == [100, 100, 98]
    assert rb[0] == {
        "pid": "1", "name": "Alpha Back", "team": "DET",
        "value": 30.0, "proj": 15.0, "injury": "", "index": 100,
    }
    assert rb[2]["injury"] == "Questionable"
    assert out["QB"] == [{
        "pid": "9", "name": "Top Passer", "team": "CIN",
        "value": 40.0, "proj": 20.0, "injury": "", "index": 100,
    }]
    assert out["WR"] == []
    assert out["TE"] == []

    # The load is the snippet's: PPR variant, every index pid requested.
    assert calls["variant"] == [{"rec": 1.0}]
    assert len(calls["load"]) == 1
    season, week, variant, pids = calls["load"][0]
    assert (season, week, variant) == (2026, 10, "ppr")
    assert set(pids) == set(FAKE_INDEX.keys())


def test_export_truncates_to_top_40(export_env):
    import app as appmod

    _calls, state = export_env
    bundles = {}
    for i in range(45):
        pid = str(100 + i)
        bundles[pid] = _bundle("WR", float(45 - i))  # proj 45 .. 1
    state["bundles"] = bundles
    out = appmod.build_start_sit_pool_export(2026, 10)
    # These pids are not in the fixture's players index, so name falls
    # back to the pid string; ranking/truncation is what is asserted.
    assert len(out["WR"]) == 40
    assert out["WR"][0]["value"] == 90.0
    assert out["WR"][-1]["value"] == 12.0  # proj 6 is the last one kept


def test_export_passes_unified_inputs(export_env):
    import app as appmod

    _calls, state = export_env
    state["bundles"] = {
        "1": _bundle("RB", 15.0, last4_ppg=17.5, played_games=8,
                     opponent="DAL"),
        "2": _bundle("RB", 12.0, last4_ppg=11.0, played_games=9,
                     opponent="SF"),
    }
    out = appmod.build_start_sit_pool_export(2026, 10)
    assert [r["pid"] for r in out["RB"]] == ["1", "2"]
    seen = {s["proj_pts"]: s for s in _fake_compute_start_score.seen}
    # The base blend + shrinkage + matchup inputs the page path passes
    # must reach the scorer on the export path too.
    assert seen[15.0]["last4_ppg"] == 17.5
    assert seen[15.0]["played_games"] == 8
    assert seen[15.0]["def_rank"] == 32       # DAL, softest in the fake map
    assert seen[15.0]["def_total"] == 32
    assert seen[12.0]["def_rank"] == 4        # SF
    assert seen[12.0]["last4_ppg"] == 11.0
    assert seen[12.0]["played_games"] == 9


def test_export_empty_bundles_returns_none(export_env):
    import app as appmod

    _calls, state = export_env
    state["bundles"] = {}
    assert appmod.build_start_sit_pool_export(2026, 10) is None


def test_export_caches_per_season_week(export_env):
    import app as appmod

    calls, state = export_env
    state["bundles"] = {"1": _bundle("RB", 15.0)}
    first = appmod.build_start_sit_pool_export(2026, 10)
    second = appmod.build_start_sit_pool_export(2026, 10)
    assert first == second
    assert len(calls["load"]) == 1
    appmod.build_start_sit_pool_export(2026, 11)
    assert len(calls["load"]) == 2


# ── (b/c) Route auth + pool availability ───────────────────────────────────

CANNED_EXPORT = {
    "QB": [{"pid": "9", "name": "Top Passer", "team": "CIN",
            "value": 40.0, "proj": 20.0, "injury": "", "index": 100}],
    "RB": [], "WR": [], "TE": [],
}


@pytest.fixture()
def client(monkeypatch):
    import app as appmod

    appmod.app.config["TESTING"] = True
    return appmod.app.test_client()


@pytest.fixture()
def build_calls(monkeypatch):
    import app as appmod

    calls = []

    def fake_build(season, week):
        calls.append((season, week))
        return CANNED_EXPORT

    monkeypatch.setattr(appmod, "build_start_sit_pool_export", fake_build)
    return calls


def test_route_fails_closed_when_secret_unset(monkeypatch, client, build_calls):
    monkeypatch.delenv("CRON_SECRET", raising=False)
    resp = client.get("/api/start-sit-pool-export",
                      query_string={"secret": "anything", "week": 10})
    assert resp.status_code == 403
    assert resp.get_json() == {"error": "unauthorized"}
    assert build_calls == []


def test_route_rejects_wrong_secret(monkeypatch, client, build_calls):
    monkeypatch.setenv("CRON_SECRET", "s3cret")
    resp = client.get("/api/start-sit-pool-export",
                      query_string={"secret": "wrong", "week": 10})
    assert resp.status_code == 403
    assert build_calls == []


def test_route_rejects_missing_secret(monkeypatch, client, build_calls):
    monkeypatch.setenv("CRON_SECRET", "s3cret")
    resp = client.get("/api/start-sit-pool-export",
                      query_string={"week": 10})
    assert resp.status_code == 403
    assert build_calls == []


def test_route_accepts_header_secret(monkeypatch, client, build_calls):
    monkeypatch.setenv("CRON_SECRET", "s3cret")
    resp = client.get("/api/start-sit-pool-export",
                      query_string={"week": 10, "season": 2026},
                      headers={"X-Cron-Secret": "s3cret"})
    assert resp.status_code == 200
    assert resp.get_json() == CANNED_EXPORT
    assert build_calls == [(2026, 10)]


def test_route_accepts_query_secret(monkeypatch, client, build_calls):
    monkeypatch.setenv("CRON_SECRET", "s3cret")
    resp = client.get("/api/start-sit-pool-export",
                      query_string={"secret": "s3cret", "week": 10,
                                    "season": 2026})
    assert resp.status_code == 200
    assert resp.get_json() == CANNED_EXPORT


def test_route_defaults_week_and_season(monkeypatch, client, build_calls):
    import app as appmod

    monkeypatch.setenv("CRON_SECRET", "s3cret")
    monkeypatch.setattr(appmod, "get_nfl_state", lambda: {"week": 11})
    resp = client.get("/api/start-sit-pool-export",
                      headers={"X-Cron-Secret": "s3cret"})
    assert resp.status_code == 200
    assert build_calls == [(_dt.datetime.now().year, 11)]


def test_route_invalid_week_is_400(monkeypatch, client, build_calls):
    monkeypatch.setenv("CRON_SECRET", "s3cret")
    resp = client.get("/api/start-sit-pool-export",
                      query_string={"secret": "s3cret", "week": "abc"})
    assert resp.status_code == 400
    assert build_calls == []


def test_route_503_when_no_pool(monkeypatch, client):
    import app as appmod

    monkeypatch.setenv("CRON_SECRET", "s3cret")
    monkeypatch.setattr(appmod, "build_start_sit_pool_export",
                        lambda season, week: None)
    resp = client.get("/api/start-sit-pool-export",
                      query_string={"secret": "s3cret", "week": 10})
    assert resp.status_code == 503
    assert "error" in resp.get_json()


# ── (d) Path parity: export vs start_score_pool_scores, REAL scorer ───────
# The 2026-11-13 unification: both bundle paths must produce the same
# value for the same player/week/scoring. These tests run the real
# compute_start_score through build_start_sit_pool_export and through
# start_score_pool_scores (the pool the Start/Sit page ranks by) over
# one shared bundle set and assert the values agree to the cent.

PARITY_INDEX = {
    "9": {"full_name": "Top Passer", "team": "CIN", "injury_status": ""},
    "1": {"full_name": "Alpha Back", "team": "SF", "injury_status": ""},
    "2": {"full_name": "Beta Back", "team": "DET", "injury_status": ""},
    "5": {"full_name": "Zero Wideout", "team": "NYJ", "injury_status": ""},
    "11": {"full_name": "Tight End", "team": "KC", "injury_status": ""},
    "3": {"full_name": "Hurt Back", "team": "MIA", "injury_status": ""},
    "6": {"full_name": "Back Wideout", "team": "DAL",
          "injury_status": "Questionable"},
}

PARITY_BUNDLES = {
    "9": _bundle("QB", 24.0, recent_ppg=21.0, season_ppg=22.5,
                 last4_ppg=23.5, played_games=8, opponent="DAL"),
    "1": _bundle("RB", 22.6, recent_ppg=25.1, season_ppg=20.8,
                 last4_ppg=23.2, played_games=8, usage_delta=2.0,
                 usage_season_avg=24.0, bust_rate=0.0, oline_index=85.3,
                 opponent="DAL"),
    # Gibbs-shaped: elite last-4, cold last-3 vs his own baseline.
    "2": _bundle("RB", 24.6, recent_ppg=18.5, season_ppg=26.2,
                 last4_ppg=23.3, played_games=8, usage_delta=-0.1,
                 usage_season_avg=26.0, bust_rate=0.0, oline_index=38.1,
                 opponent="NE"),
    "5": _bundle("WR", 18.0, recent_ppg=12.0, season_ppg=15.5,
                 last4_ppg=14.0, played_games=9, usage_delta=-1.5,
                 usage_season_avg=9.0, opponent="SF"),
    "11": _bundle("TE", 12.0, recent_ppg=13.0, season_ppg=11.0,
                  last4_ppg=15.0, played_games=7, usage_delta=0.5,
                  usage_season_avg=7.0, opponent="NE"),
    # IR in the bundle and no live override: excluded on both paths.
    "3": _bundle("RB", 15.0, injury_status="IR"),
    # IR at batch time but Questionable live: the live status wins and
    # the player is scored on both paths.
    "6": _bundle("WR", 14.0, recent_ppg=13.0, season_ppg=12.0,
                 last4_ppg=13.5, played_games=8, opponent="DAL",
                 injury_status="IR"),
}


@pytest.fixture()
def parity_env(monkeypatch):
    """Both bundle paths over one shared fake bundle set + index.

    compute_start_score is deliberately NOT faked here.
    """
    import app as appmod
    import data_building.start_score_bundle as bundle_mod

    monkeypatch.setattr(bundle_mod, "variant_for_exact_scoring",
                        lambda scoring: "ppr")
    monkeypatch.setattr(
        bundle_mod, "load_start_score_bundles",
        lambda season, week, variant, pids: dict(PARITY_BUNDLES))
    monkeypatch.setattr(
        bundle_mod, "load_start_score_bundles_for_scoring",
        lambda season, week, scoring, pids: dict(PARITY_BUNDLES))
    monkeypatch.setattr(appmod, "get_players_index_global",
                        lambda: dict(PARITY_INDEX))
    monkeypatch.setattr(
        appmod, "_matchup_rank_table",
        lambda season, position, scoring_settings=None: (
            {"DAL": 32, "NE": 20, "SF": 4}, 32, {}, False))
    monkeypatch.setattr(appmod, "_SS_POOL_MIN_BUNDLES", 1)
    appmod._SS_EXPORT_CACHE.clear()
    appmod._SS_EXPORT_CACHE_TS.clear()
    appmod._SS_POOL_CACHE.clear()
    appmod._SS_POOL_CACHE_TS.clear()
    yield
    appmod._SS_EXPORT_CACHE.clear()
    appmod._SS_EXPORT_CACHE_TS.clear()
    appmod._SS_POOL_CACHE.clear()
    appmod._SS_POOL_CACHE_TS.clear()


def test_export_matches_page_pool_values(parity_env):
    import app as appmod

    export = appmod.build_start_sit_pool_export(2026, 10)
    pool = appmod.start_score_pool_scores(2026, 10, {"rec": 1.0})
    assert pool["kind"] == "score"

    exported = {}
    for pos_rows in export.values():
        for row in pos_rows:
            exported[row["pid"]] = row["value"]
    # Every scored player matches the page pool's value to the cent
    # (the export rounds to 2dp; the pool keeps the raw float).
    assert set(exported) == set(pool["by_pid"])
    for pid, value in exported.items():
        assert value == round(pool["by_pid"][pid], 2), pid
    # The IR bundle row is excluded on both paths; the live-Questionable
    # overlay row is scored on both.
    assert "3" not in exported and "3" not in pool["by_pid"]
    assert "6" in exported and "6" in pool["by_pid"]
