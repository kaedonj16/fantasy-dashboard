"""Duplicate-call fixes: Advanced Metrics batch + enriched cache, the
league-players shared disk tier, the sparklines server cache, and the
mtime-guarded pick-value table.

Parity contract: every leaderboard value and player row is identical to the
old per-metric path; only the call/compute pattern changes. The PRO-gating
test is the hard one: a free viewer must never receive a PRO metric's
payload from a cache entry a PRO viewer warmed.
"""
import json
import os

import pytest

pytest.importorskip("flask")
pytest.importorskip("pandas")


# ── Advanced Metrics: batch pays the preamble once, cache serves repeats ────

def _counting_builder(calls):
    def _build(metric, **kwargs):
        calls.append(metric)
        return [{"player_id": "p1", "position": "RB", "value": 3.5,
                 "games": 5, "season": 2026}]
    return _build


@pytest.fixture
def am_env(offline_client, monkeypatch):
    import data_building.advanced_metrics as am
    import routes.advanced_metrics_bp as bp

    bp._LB_CACHE.clear()
    calls = []
    monkeypatch.setattr(am, "get_metric_leaderboard", _counting_builder(calls))
    return offline_client, bp, calls


def test_batch_preset_load_computes_each_metric_once(am_env):
    client, bp, calls = am_env
    resp = client.get(
        "/api/advanced-metrics/leaderboard"
        "?metrics=target_share,opportunity_share&season=2026"
        "&min_vol_map=target_share:3")
    assert resp.status_code == 200, resp.get_data(as_text=True)
    results = resp.get_json()["results"]
    assert set(results) == {"target_share", "opportunity_share"}
    assert results["target_share"]["players"][0]["value"] == 3.5
    assert sorted(calls) == ["opportunity_share", "target_share"]

    # Repeat batch: enriched cache serves every metric, no recompute.
    resp2 = client.get(
        "/api/advanced-metrics/leaderboard"
        "?metrics=target_share,opportunity_share&season=2026"
        "&min_vol_map=target_share:3")
    assert resp2.status_code == 200
    assert resp2.get_json() == resp.get_json()
    assert len(calls) == 2, "repeat batch recomputed instead of hitting cache"

    # The single-metric route shares the same cache: the batch's effective
    # params for target_share (season 2026, min_vol 3) hit without recompute.
    resp3 = client.get(
        "/api/advanced-metrics/leaderboard"
        "?metric=target_share&season=2026&min_vol=3")
    assert resp3.status_code == 200
    assert resp3.get_json()["players"] == results["target_share"]["players"]
    assert len(calls) == 2


def test_pro_metric_never_leaks_from_cache_to_free_viewer(am_env, monkeypatch):
    client, bp, calls = am_env
    import data_building.advanced_metrics as am

    pro_metric = "ppr_over_expected_per_game"
    assert pro_metric in am.PRO_METRICS
    assert pro_metric in am.LEADERBOARD_METRICS

    # PRO viewer warms the cache.
    monkeypatch.setattr(bp, "_request_has_premium", lambda *a, **k: True)
    resp = client.get(
        f"/api/advanced-metrics/leaderboard?metric={pro_metric}&season=2026")
    assert resp.status_code == 200, resp.get_data(as_text=True)
    warmed_calls = len(calls)
    assert warmed_calls >= 1

    # Free viewer: same params must 403 at the gate, cache or no cache,
    # on both the single route and the batch route.
    monkeypatch.setattr(bp, "_request_has_premium", lambda *a, **k: False)
    resp_free = client.get(
        f"/api/advanced-metrics/leaderboard?metric={pro_metric}&season=2026")
    assert resp_free.status_code == 403
    assert resp_free.get_json()["error"] == "pro_only"

    resp_batch = client.get(
        "/api/advanced-metrics/leaderboard"
        f"?metrics={pro_metric},target_share&season=2026")
    assert resp_batch.status_code == 200
    entry = resp_batch.get_json()["results"][pro_metric]
    assert entry["error"] == "pro_only"
    assert entry["status"] == 403
    assert "players" not in entry
    # The free metric in the same batch still serves normally.
    assert resp_batch.get_json()["results"]["target_share"]["players"]


def test_batch_payload_matches_single_route_payload(am_env):
    """Parity: batch mode returns the exact payload the single route returns
    for the same metric + params. Both entry points share one compute
    function, so this holds by construction; the assertion pins it."""
    client, bp, calls = am_env
    single = client.get(
        "/api/advanced-metrics/leaderboard?metric=opportunity_share&season=2026")
    assert single.status_code == 200, single.get_data(as_text=True)
    batch = client.get(
        "/api/advanced-metrics/leaderboard?metrics=opportunity_share&season=2026")
    assert batch.status_code == 200
    assert batch.get_json()["results"]["opportunity_share"] == single.get_json()


def test_batch_unknown_metric_is_in_band_not_wholesale_400(am_env):
    client, bp, calls = am_env
    resp = client.get(
        "/api/advanced-metrics/leaderboard"
        "?metrics=target_share,not_a_metric&season=2026")
    assert resp.status_code == 200
    results = resp.get_json()["results"]
    assert results["not_a_metric"]["error"] == "unknown metric"
    assert results["not_a_metric"]["status"] == 400
    assert results["target_share"]["players"]


def test_am_frontend_uses_batch_for_extras():
    from dashboard_services.pages.advanced_metrics_page import _AM_JS

    assert "function _extraBatchUrl(" in _AM_JS
    assert "metrics: keys.join(',')" in _AM_JS
    assert "min_vol_map" in _AM_JS
    assert "function _loadExtrasPerKey(" in _AM_JS  # per-key fallback kept
    load_extras = _AM_JS[_AM_JS.index("function _loadExtras("):]
    assert "_extraBatchUrl(open, curSeason)" in load_extras
    assert "res.results" in load_extras


# ── League players: shared disk tier survives a simulated recycle ───────────

def test_league_players_disk_tier_serves_after_recycle(monkeypatch, tmp_path):
    import app as appmod

    payload = {
        "players": [{"id": "1", "name": "A Player", "value": 10.0}],
        "tier_thresholds": {}, "adp_sources": {},
        "adp_source_options": {}, "adp_columns": [],
    }
    builds = []

    def _uncached(kdef=False):
        builds.append(kdef)
        return json.loads(json.dumps(payload))

    monkeypatch.setattr(appmod, "_build_league_players_payload_uncached", _uncached)
    monkeypatch.setattr(appmod, "get_model_value_table_cached", lambda: [])
    monkeypatch.setattr(appmod, "_MODEL_VALUE_CACHE_TS", 424242.0)
    import dashboard_services.adp_service as adp_service
    monkeypatch.setattr(
        adp_service, "global_adp_snapshot_signature", lambda: 7.0)
    disk_path = tmp_path / "lp_payload.json"
    monkeypatch.setattr(
        appmod, "_lp_payload_disk_path", lambda key, ts, sig: str(disk_path))

    appmod._LP_PAYLOAD_CACHE.clear()
    first = appmod._build_league_players_payload()
    assert first == payload
    assert len(builds) == 1
    assert disk_path.exists(), "fresh build was not persisted to the disk tier"

    # Simulated worker recycle: in-process memo wiped, disk tier remains.
    appmod._LP_PAYLOAD_CACHE.clear()
    second = appmod._build_league_players_payload()
    assert second == payload
    assert len(builds) == 1, "rebuild ran even though the disk tier had the signature"


# ── Sparklines: one DB query per model signature ────────────────────────────

def test_sparklines_second_call_served_from_server_cache(monkeypatch):
    import app as appmod
    import dashboard_services.db as dbmod

    rows = [
        {"player_id": "p1", "value": 10.0, "value_sf": 20.0, "value_8": None,
         "value_12": None, "value_14": None, "sf_value_8": None,
         "sf_value_12": None, "sf_value_14": None,
         "calibrated_value_1qb": 11.0, "calibrated_value_sf": 22.0},
        {"player_id": "p1", "value": 12.0, "value_sf": 24.0, "value_8": None,
         "value_12": None, "value_14": None, "sf_value_8": None,
         "sf_value_12": None, "sf_value_14": None,
         "calibrated_value_1qb": 11.0, "calibrated_value_sf": 22.0},
    ]
    queries = []

    class _Conn:
        def execute(self, sql, *a, **k):
            queries.append(sql)

            class _Res:
                def fetchall(self):
                    return rows
            return _Res()

    class _Ctx:
        def __enter__(self):
            return _Conn()

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(dbmod, "get_conn", lambda: _Ctx())
    monkeypatch.setattr(appmod, "get_model_value_table_cached", lambda: [])
    monkeypatch.setattr(appmod, "_MODEL_VALUE_CACHE_TS", 989898.0)
    appmod._SPARKLINES_CACHE.update({"key": None, "ts": 0.0, "result": None})

    with appmod.app.test_request_context("/api/sparklines"):
        r1 = appmod.api_sparklines()
        body1 = r1.get_json()
    assert body1 and "p1" in body1
    assert len(queries) == 1

    with appmod.app.test_request_context("/api/sparklines"):
        r2 = appmod.api_sparklines()
        assert r2.get_json() == body1
    assert len(queries) == 1, "sparklines re-ran its DB query on a cache hit"


# ── Pick table: parsed once per source mtime ────────────────────────────────

def test_pick_table_parsed_once_per_mtime(monkeypatch, tmp_path):
    import dashboard_services.picks as picks

    (tmp_path / "pick_values_wls_latest.json").write_text(
        json.dumps({"1qb": {}, "sf": {}}), encoding="utf-8")
    monkeypatch.setattr(picks, "DATA_DIR", tmp_path)
    picks._PICK_TABLE_CACHE.clear()

    builds = []

    def _fake_uncached(**kwargs):
        builds.append(kwargs)
        return {"2027_1_01": 100.0}

    monkeypatch.setattr(picks, "_load_pick_value_table_uncached", _fake_uncached)

    t1 = picks.load_pick_value_table()
    t2 = picks.load_pick_value_table()
    assert t1 == t2 == {"2027_1_01": 100.0}
    assert len(builds) == 1, "pick table rebuilt despite unchanged sources"

    # A rebuilt source file (new mtime) invalidates the cache.
    wls = tmp_path / "pick_values_wls_latest.json"
    wls.write_text(json.dumps({"1qb": {}, "sf": {}}), encoding="utf-8")
    st = wls.stat()
    os.utime(wls, (st.st_atime, st.st_mtime + 5))
    t3 = picks.load_pick_value_table()
    assert t3 == {"2027_1_01": 100.0}
    assert len(builds) == 2, "pick table cache ignored a source mtime change"


def test_teams_page_dead_pick_load_removed():
    from pathlib import Path

    src = Path("dashboard_services/pages/teams_page.py").read_text(encoding="utf-8")
    assert "_pk_value_tbl" not in src
    assert "_lpvt_teams" not in src
