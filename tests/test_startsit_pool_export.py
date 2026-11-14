"""Start/sit live pool export: app.build_start_sit_pool_export + the
CRON_SECRET-gated GET /api/start-sit-pool-export.

The export replaces the manual Render-shell paste that produced
startsit_live_pool_w{W}_ppr.json for the Thursday boards. The builder is
the shell snippet verbatim (bundle load -> compute_start_score with the
snippet's exact kwargs -> competition rank -> tier index), so the fakes
below mirror the real signatures of load_start_score_bundles /
variant_for_exact_scoring (data_building.start_score_bundle) and
compute_start_score (utils.start_sit_score) exactly -- an extra or
missing kwarg in the builder makes the fake raise, the row is skipped,
and the shaping assertions fail.
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
    usage_delta=None,
    usage_season_avg=None,
    injury_status=None,
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
    """Mirror of the snippet's call surface: score = 2 x projection.

    proj == 13.37 is the poison row: the real compute raises on a bad
    bundle, and the builder must skip just that player.
    """
    if proj_pts == 13.37:
        raise ValueError("bad bundle")
    return (float(proj_pts) * 2.0, {}, None)


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
    monkeypatch.setattr(appmod, "get_players_index_global",
                        lambda: dict(FAKE_INDEX))
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
