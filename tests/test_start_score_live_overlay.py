"""Live projection overlay for start/sit bundles.

The daily batch freezes proj_pts at build time, but Sleeper revises its
lines intraday — most sharply when a player is ruled out and his volume
moves to his teammates (2026-11-07: Skattebo Out moved Singletary's line
8.7 -> 13.9 while the site kept scoring the frozen 8.7). The bundle loader
therefore swaps proj_pts for the live value at load time, mirroring the
live injury-status overlay the request paths already apply. These tests
pin: (a) a live projection replaces the bundle value, under the bundle
variant's scoring, while a player with no live line keeps the batch value;
(b) a failed fetch falls back to the batch values, silently; (c) the
fetch is cached inside its TTL, so repeated loads cost one fetch.
"""
import time
from contextlib import contextmanager

import pytest

from data_building import start_score_bundle as ssb


@pytest.fixture(autouse=True)
def _clear_overlay_cache():
    ssb._LIVE_PROJ_CACHE.clear()
    yield
    ssb._LIVE_PROJ_CACHE.clear()


class _FakeResult:
    def __init__(self, rows):
        self._rows = rows

    def fetchall(self):
        return self._rows


class _FakeConn:
    def __init__(self, rows):
        self._rows = rows

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, *a, **k):
        return _FakeResult(self._rows)


def _serve_bundles(monkeypatch, bundles):
    """Patch the bundle table: {pid: bundle} served as DB rows."""
    rows = [{"player_id": pid, "bundle": dict(b)} for pid, b in bundles.items()]

    @contextmanager
    def _cm():
        yield _FakeConn(rows)

    monkeypatch.setattr("dashboard_services.db.get_conn", lambda *a, **k: _cm())


_BATCH = {
    # The Singletary shape: frozen as the backup's number at batch time.
    "100": {"pos": "RB", "proj_pts": 8.7},
    # No live line below: must keep the batch value.
    "200": {"pos": "WR", "proj_pts": 12.0},
}


def test_live_projection_replaces_bundle_value(monkeypatch):
    _serve_bundles(monkeypatch, _BATCH)
    monkeypatch.setattr(
        ssb, "_fetch_live_projection_map",
        lambda season, week: {"100": {"ppr": 13.9, "half_ppr": 12.4}})

    out = ssb.load_start_score_bundles(2026, 9, "ppr", ["100", "200"])
    assert out["100"]["proj_pts"] == 13.9      # live value, variant scoring
    assert out["200"]["proj_pts"] == 12.0      # absent live -> batch kept
    assert _BATCH["100"]["proj_pts"] == 8.7    # batch dict never mutated

    # Same live map under the half-PPR variant selects the half-PPR line.
    out_half = ssb.load_start_score_bundles(2026, 9, "half_ppr", ["100", "200"])
    assert out_half["100"]["proj_pts"] == 12.4
    assert out_half["200"]["proj_pts"] == 12.0


def test_fetch_failure_falls_back_to_bundle_values(monkeypatch):
    _serve_bundles(monkeypatch, _BATCH)

    def _boom(season, week, raw_scoring_settings=None):
        raise RuntimeError("sleeper down")

    monkeypatch.setattr("utils.projections.fetch_week_projections", _boom)
    out = ssb.load_start_score_bundles(2026, 9, "ppr", ["100", "200"])
    assert out["100"]["proj_pts"] == 8.7
    assert out["200"]["proj_pts"] == 12.0


def test_live_map_cached_within_ttl(monkeypatch):
    _serve_bundles(monkeypatch, _BATCH)
    calls = []

    def _counting_fetch(season, week):
        calls.append((season, week))
        return {"100": {"ppr": 13.9}}

    monkeypatch.setattr(ssb, "_fetch_live_projection_map", _counting_fetch)

    ssb.load_start_score_bundles(2026, 9, "ppr", ["100", "200"])
    ssb.load_start_score_bundles(2026, 9, "ppr", ["100", "200"])
    ssb.live_projection_map(2026, 9)
    assert len(calls) == 1

    # Past the TTL the next load refetches (and picks up revised lines).
    fetched_at, cached = ssb._LIVE_PROJ_CACHE[(2026, 9)]
    ssb._LIVE_PROJ_CACHE[(2026, 9)] = (
        fetched_at - ssb._LIVE_PROJ_TTL_SEC - 1, cached)
    ssb.load_start_score_bundles(2026, 9, "ppr", ["100", "200"])
    assert len(calls) == 2


def test_pool_cache_expires_on_overlay_cadence(monkeypatch):
    pytest.importorskip("flask")
    pytest.importorskip("pandas")
    import app as appmod
    import data_building.start_score_bundle as bundle_mod

    state = {"top": 30.0}

    def _bundles(*a, **k):
        out = {}
        for i in range(45):
            out[f"rb{i}"] = {"pos": "RB",
                             "proj_pts": state["top"] - 0.5 * i}
        return out

    monkeypatch.setattr(bundle_mod, "variant_for_exact_scoring",
                        lambda *a, **k: "ppr")
    monkeypatch.setattr(bundle_mod, "load_start_score_bundles", _bundles)
    monkeypatch.setattr(appmod, "get_players_index_global",
                        lambda *a, **k: {f"rb{i}": {} for i in range(45)})
    appmod._SS_POOL_CACHE.clear()
    appmod._SS_POOL_CACHE_TS.clear()

    pool = appmod.start_score_pool_scores(2026, 6, {"rec": 1.0})
    assert pool["by_pos"]["RB"][0] == 30.0

    # A revised line inside the TTL does not rebuild the pool...
    state["top"] = 31.0
    pool = appmod.start_score_pool_scores(2026, 6, {"rec": 1.0})
    assert pool["by_pos"]["RB"][0] == 30.0

    # ...but once the pool is older than the overlay cadence it is rebuilt
    # from the re-overlaid inputs.
    key = (2026, 6, appmod._ss_scoring_sig({"rec": 1.0}))
    appmod._SS_POOL_CACHE_TS[key] -= appmod._SS_POOL_TTL_SEC + 1
    pool = appmod.start_score_pool_scores(2026, 6, {"rec": 1.0})
    assert pool["by_pos"]["RB"][0] == 31.0
    appmod._SS_POOL_CACHE.clear()
    appmod._SS_POOL_CACHE_TS.clear()
