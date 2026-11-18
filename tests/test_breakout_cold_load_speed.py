"""Breakout Engine cold-load speed fix.

Pins the contract of the cold-load work:
- board + track record payloads are served from the shared (cross-worker)
  store on an in-process miss, under the same data watermark, so a cold
  worker does not repay a build a sibling worker already paid;
- a watermark change never serves the shared entry (the watermark is part
  of the shared key);
- shared-store failures degrade to fresh compute, exactly as before;
- shared payloads are JSON-normalized (Decimal/date) so they render
  identically to freshly computed ones;
- publish-time warming computes both payloads and never raises;
- forecast series loads are scoped to the open calls' players, with
  identical forecasts, and degrade to the unscoped load for loaders with
  the historical two-argument shape;
- the page loads candidates and weeks in parallel and defers the sidebar
  until the board has rendered;
- the page shell's season / last-updated lookups are cached for an hour
  and a failed read is not cached.
"""
from __future__ import annotations

import time
from datetime import date
from decimal import Decimal
from pathlib import Path

import pytest

pytest.importorskip("flask")

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _clear_caches():
    import dashboard_services.breakout_api as api

    api._clear_breakout_payload_caches()
    yield
    api._clear_breakout_payload_caches()


# ---------------------------------------------------------------------------
# Shared store fake
# ---------------------------------------------------------------------------

class _FakeSharedStore:
    """In-memory stand-in for dashboard_services.player_details_store."""

    def __init__(self):
        self.data = {}
        self.fail_get = False
        self.fail_set = False

    def get_json(self, namespace, key_parts, ttl_seconds):
        if self.fail_get:
            raise RuntimeError("redis down")
        entry = self.data.get((namespace, repr(key_parts)))
        if entry is None:
            return None
        return {"ts": time.time(), "payload": entry}

    def set_json(self, namespace, key_parts, payload, ts, ttl_seconds):
        if self.fail_set:
            raise RuntimeError("redis down")
        self.data[(namespace, repr(key_parts))] = payload
        return True


class _Watermark:
    def __init__(self, value=("run-1", "grades-1", 5, "opp-1")):
        self.value = value

    def __call__(self, season):
        return self.value


def _patch_board(monkeypatch, watermark, store):
    import dashboard_services.breakout_api as api
    from dashboard_services import player_details_store as pd_store

    monkeypatch.setattr(api, "_resolve_bo_season", lambda season: season)
    monkeypatch.setattr(api, "_breakout_data_watermark", watermark)
    monkeypatch.setattr(pd_store, "get_json", store.get_json)
    monkeypatch.setattr(pd_store, "set_json", store.set_json)
    calls = []

    def candidates(season, min_score=0.0, limit=None, week=None):
        calls.append((season, min_score, limit, week))
        return {
            "season": season,
            "candidates": [{
                "player_id": "1",
                "breakout_score": Decimal("55.0"),
                "as_of_date": date(2026, 9, 1),
            }],
            "count": 1,
            "data_available": True,
        }

    monkeypatch.setattr(api, "get_breakout_candidates", candidates)
    return api, calls


# ---------------------------------------------------------------------------
# Board: shared layer
# ---------------------------------------------------------------------------

def test_board_shared_hit_skips_recompute_on_cold_worker(monkeypatch):
    store = _FakeSharedStore()
    api, calls = _patch_board(monkeypatch, _Watermark(), store)

    first = api.get_breakout_board_candidates(2026)
    assert len(calls) == 1
    assert store.data, "fresh compute must write through to the shared store"

    # A cold worker: in-process cache empty, shared entry serves.
    api._clear_breakout_payload_caches()
    second = api.get_breakout_board_candidates(2026)
    assert len(calls) == 1
    # The shared copy is JSON-normalized: Decimal -> float, date -> ISO str.
    assert second == api._breakout_payload_jsonable(first)
    assert second["candidates"][0]["breakout_score"] == 55.0
    assert second["candidates"][0]["as_of_date"] == "2026-09-01"

    # The shared hit backfills the in-process cache.
    third = api.get_breakout_board_candidates(2026)
    assert third == second
    assert len(calls) == 1


def test_board_shared_entry_not_served_under_new_watermark(monkeypatch):
    watermark = _Watermark()
    store = _FakeSharedStore()
    api, calls = _patch_board(monkeypatch, watermark, store)

    api.get_breakout_board_candidates(2026)
    assert len(calls) == 1

    watermark.value = ("run-2", "grades-1", 5, "opp-1")
    api._clear_breakout_payload_caches()
    api.get_breakout_board_candidates(2026)
    assert len(calls) == 2


def test_board_shared_failure_degrades_to_compute(monkeypatch):
    store = _FakeSharedStore()
    store.fail_get = True
    store.fail_set = True
    api, calls = _patch_board(monkeypatch, _Watermark(), store)

    first = api.get_breakout_board_candidates(2026)
    second = api.get_breakout_board_candidates(2026)

    assert first["count"] == 1 and second["count"] == 1
    # In-process cache still serves the repeat; the shared failures changed
    # nothing about the result.
    assert len(calls) == 1


# ---------------------------------------------------------------------------
# Track record: shared layer
# ---------------------------------------------------------------------------

def test_track_record_shared_hit_skips_recompute(monkeypatch):
    import dashboard_services.breakout_api as api
    from dashboard_services import player_details_store as pd_store

    store = _FakeSharedStore()
    monkeypatch.setattr(api, "_resolve_bo_season", lambda season: season)
    monkeypatch.setattr(api, "_breakout_data_watermark", _Watermark())
    monkeypatch.setattr(pd_store, "get_json", store.get_json)
    monkeypatch.setattr(pd_store, "set_json", store.set_json)
    calls = []

    def compute(requested_season=None):
        calls.append(requested_season)
        return {"season": 2026, "weekly": {"available": True}, "hits": []}

    monkeypatch.setattr(api, "_compute_breakout_track_record", compute)

    first = api.get_breakout_track_record(2026)
    assert len(calls) == 1
    api._clear_breakout_payload_caches()
    second = api.get_breakout_track_record(2026)
    assert second == first
    assert len(calls) == 1


# ---------------------------------------------------------------------------
# Publish-time warming
# ---------------------------------------------------------------------------

def test_warm_computes_board_and_track_record(monkeypatch):
    import dashboard_services.breakout_api as api

    seen = []
    monkeypatch.setattr(api, "get_breakout_board_candidates",
                        lambda season=None: seen.append(("board", season)))
    monkeypatch.setattr(api, "get_breakout_track_record",
                        lambda season=None: seen.append(("track", season)))
    api.warm_breakout_payload_caches(2026)
    assert seen == [("board", 2026), ("track", 2026)]


def test_warm_never_raises(monkeypatch):
    import dashboard_services.breakout_api as api

    def boom(season=None):
        raise RuntimeError("db down")

    monkeypatch.setattr(api, "get_breakout_board_candidates", boom)
    monkeypatch.setattr(api, "get_breakout_track_record", boom)
    api.warm_breakout_payload_caches(2026)  # must not raise


# ---------------------------------------------------------------------------
# Forecast series scoping
# ---------------------------------------------------------------------------

def _wk(week, snap=60.0, tgt=6.0, car=8.0, ppr=12.0, snaps=60):
    return {"week": week, "snap_pct": snap, "targets": tgt, "carries": car,
            "ppr_pts": ppr, "snaps": snaps}


def _call(player_id="101", as_of_week=4):
    return {
        "player_id": player_id, "player_name": "Test Player", "season": 2026,
        "as_of_week": as_of_week, "scoring_version": "weekly-v6",
        "classification": "emerging_breakout", "breakout_score": 55.0,
        "baseline_source": "current_season", "baseline_weeks": [1, 2],
        "evidence": {
            "fantasy": {"baseline_ppg": 8.0, "recent_ppg": 14.0},
            "signals": {
                "snap_share": {"baseline": 40.0, "recent": 62.0},
                "carry_opportunity_pg": {"baseline": 10.0, "recent": 16.0},
            },
        },
    }


_HELD = [_wk(5, snap=65.0, tgt=9.0, car=11.0, ppr=18.0),
         _wk(6, snap=65.0, tgt=9.0, car=11.0, ppr=18.0)]


def test_forecasts_scope_series_load_to_open_call_players(monkeypatch):
    from data_building.breakout_engine import forecasts
    from data_building.breakout_engine import weekly_grading as wg

    seen = {}

    def scoped_loader(season, through, player_ids=None):
        seen["player_ids"] = player_ids
        # Extra players in the universe must not affect the forecast.
        return {"101": list(_HELD), "999": [_wk(5, ppr=30.0)]}

    monkeypatch.setattr(wg, "default_through_week", lambda season: 6)
    monkeypatch.setattr(wg, "load_season_series", scoped_loader)

    out = forecasts.weekly_forecasts_for_calls(2026, [_call()])

    assert seen["player_ids"] == ["101"]
    assert out["101"]["band"] == forecasts.BAND_TRACKING_HIT


def test_forecasts_unscoped_loader_still_works(monkeypatch):
    """A two-argument loader (the historical shape) gets the unscoped call
    and produces the identical forecast."""
    from data_building.breakout_engine import forecasts
    from data_building.breakout_engine import weekly_grading as wg

    monkeypatch.setattr(wg, "default_through_week", lambda season: 6)
    monkeypatch.setattr(wg, "load_season_series",
                        lambda season, through: {"101": list(_HELD)})

    out = forecasts.weekly_forecasts_for_calls(2026, [_call()])
    assert out["101"]["band"] == forecasts.BAND_TRACKING_HIT


def test_load_season_series_scoping_plumbs_player_ids(monkeypatch):
    import data_building.weekly_metrics as weekly_metrics
    from data_building.breakout_engine import weekly_grading as wg

    seen = {}

    def fake_series(season, through, player_ids=None):
        seen["args"] = (season, through, player_ids)
        return {"101": [_wk(5)]}

    monkeypatch.setattr(weekly_metrics, "get_weekly_series_by_player",
                        fake_series)
    out = wg.load_season_series(2026, 6, player_ids=["101"])
    assert out == {"101": [_wk(5)]}
    assert seen["args"] == (2026, 6, ["101"])


def test_weekly_series_query_scopes_to_player_ids(monkeypatch):
    import data_building.weekly_metrics as weekly_metrics

    recorded = {}

    class _Result:
        def fetchall(self):
            return []

    class _Conn:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def execute(self, sql, params):
            recorded["sql"] = sql
            recorded["params"] = params
            return _Result()

    monkeypatch.setattr(weekly_metrics, "init_weekly_metrics_db",
                        lambda: None)
    monkeypatch.setattr(weekly_metrics, "get_conn", lambda: _Conn())

    out = weekly_metrics.get_weekly_series_by_player(
        2026, 6, player_ids=["101", "202"])
    assert out == {}
    assert "player_id = ANY(%s)" in recorded["sql"]
    assert recorded["params"][-1] == ["101", "202"]
    assert weekly_metrics.get_weekly_series_by_player(2026, 6, player_ids=[]) == {}


# ---------------------------------------------------------------------------
# Page wiring: parallel load + deferred sidebar
# ---------------------------------------------------------------------------

def _page_src():
    return (ROOT / "app.py").read_text(encoding="utf-8")


def test_page_loads_candidates_in_parallel_with_weeks():
    src = _page_src()
    # The board's first load is kicked off before (not inside) the weeks
    # fetch resolution, and exactly once.
    assert src.count("loadBreakouts('latest');") == 1
    assert src.index("loadBreakouts('latest');") < src.index(
        "fetch('/api/breakout/weeks?season=")
    # The weeks handler refetches only when the in-flight board was
    # requested for a different week than the selector's default.
    assert "boRequestedWeek !== 'latest'" in src
    assert "loadBreakouts(latest);" in src


def test_page_defers_sidebar_until_board_renders():
    src = _page_src()
    assert "_boMaybeStartSidebar" in src
    assert "requestIdleCallback" in src
    # No bare top-level sidebar start remains in the page script.
    assert "\n      loadBreakoutSidebar();\n" not in src
    # The render path and the board error path both release the sidebar.
    assert src.count("_boMaybeStartSidebar();") >= 2


# ---------------------------------------------------------------------------
# Page shell meta cache
# ---------------------------------------------------------------------------

class _MetaCursor:
    def __init__(self, parent):
        self._parent = parent

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def execute(self, sql):
        self._parent.queries.append(sql)
        self._sql = sql

    def fetchone(self):
        if "breakout_opportunity_scores" in self._sql:
            return {"max": 2026}
        return {"m": date(2026, 10, 7)}


class _MetaConn:
    def __init__(self, parent):
        self._parent = parent

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def cursor(self):
        return _MetaCursor(self._parent)


class _MetaDb:
    def __init__(self):
        self.queries = []

    def conn(self):
        return _MetaConn(self)


@pytest.fixture()
def appmod():
    pytest.importorskip("pandas")
    import app as _appmod

    _appmod._bo_page_meta_cache.update(
        {"ts": 0.0, "season": None, "last_updated": ""})
    yield _appmod
    _appmod._bo_page_meta_cache.update(
        {"ts": 0.0, "season": None, "last_updated": ""})


def test_page_meta_cached_across_loads(appmod, monkeypatch):
    import dashboard_services.breakout_api as api

    db = _MetaDb()
    monkeypatch.setattr(api, "get_conn", db.conn)

    season1, updated1 = appmod._bo_page_meta(2025)
    season2, updated2 = appmod._bo_page_meta(2025)

    assert season1 == season2 == 2026
    assert "Oct 07, 2026" in updated1 and updated2 == updated1
    # Two shell loads, one pair of DB queries.
    assert len(db.queries) == 2


def test_page_meta_failure_is_not_cached(appmod, monkeypatch):
    import dashboard_services.breakout_api as api

    def boom():
        raise RuntimeError("db down")

    monkeypatch.setattr(api, "get_conn", boom)
    season, updated = appmod._bo_page_meta(2025)
    assert season == 2025 or season == date.today().year
    assert updated == ""
    assert appmod._bo_page_meta_cache["season"] is None
