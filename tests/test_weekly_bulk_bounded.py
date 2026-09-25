"""weekly-bulk must always be scoped (season + week range) and paginated."""
from __future__ import annotations

import pytest

pytest.importorskip("flask")

import data_building.advanced_metrics as am


@pytest.fixture
def stub_bulk(monkeypatch):
    seen = {}

    def _fake(season=None, week_start=None, week_end=None, position=None,
              limit=5000, offset=0):
        seen.update(season=season, week_start=week_start, week_end=week_end,
                    position=position, limit=limit, offset=offset)
        return {"byId": {"1": {"name": "Test Player"}}, "keys": []}

    monkeypatch.setattr(am, "get_all_weekly_metrics_bulk", _fake)
    # get_nfl_state is TTL-cached process-wide: an earlier test (or a real
    # network call) can leave real NFL state cached, which would bypass the
    # offline_client fetch_json stub these tests rely on for the default
    # season/week. Clear it so the stub is authoritative.
    from dashboard_services import api as _api
    _api.get_nfl_state.clear_cache()
    return seen


def test_weekly_bulk_defaults_scope_to_current_season(offline_client, stub_bulk):
    resp = offline_client.get("/api/advanced-metrics/weekly-bulk")
    assert resp.status_code == 200
    # offline_client stubs NFL state to season 2026, week 1
    assert stub_bulk["season"] == 2026
    assert stub_bulk["week_start"] == 1
    assert stub_bulk["week_end"] == 1
    assert stub_bulk["limit"] == 2000
    assert stub_bulk["offset"] == 0
    body = resp.get_json()
    assert body["season"] == 2026
    assert body["returned"] == 1


def test_weekly_bulk_explicit_params_pass_through(offline_client, stub_bulk):
    resp = offline_client.get(
        "/api/advanced-metrics/weekly-bulk?season=2025&week_start=3&week_end=5"
        "&position=rb&limit=100&offset=50"
    )
    assert resp.status_code == 200
    assert stub_bulk["season"] == 2025
    assert (stub_bulk["week_start"], stub_bulk["week_end"]) == (3, 5)
    assert stub_bulk["position"] == "RB"
    assert (stub_bulk["limit"], stub_bulk["offset"]) == (100, 50)


def test_weekly_bulk_limit_hard_cap(offline_client, stub_bulk):
    resp = offline_client.get("/api/advanced-metrics/weekly-bulk?limit=99999")
    assert resp.status_code == 200
    assert stub_bulk["limit"] == 5000
    assert resp.get_json()["limit"] == 5000


def test_weekly_bulk_bad_param_is_400(offline_client):
    resp = offline_client.get("/api/advanced-metrics/weekly-bulk?season=abc")
    assert resp.status_code == 400
    assert resp.get_json()["code"] == "bad_request"


def test_weekly_bulk_bad_week_range_is_400(offline_client):
    resp = offline_client.get(
        "/api/advanced-metrics/weekly-bulk?week_start=9&week_end=2"
    )
    assert resp.status_code == 400


class _FakeCursor:
    def __init__(self):
        self.sql = None
        self.params = None

    def execute(self, sql, params):
        self.sql = sql
        self.params = params
        return self

    def fetchall(self):
        return []


class _FakeConn:
    def __init__(self):
        self.cur = _FakeCursor()

    def __enter__(self):
        return self.cur

    def __exit__(self, *a):
        return False


def _run_bulk_sql(monkeypatch, **kwargs):
    conn = _FakeConn()
    monkeypatch.setattr(am, "get_conn", lambda: conn)
    import utils.utils as uu
    monkeypatch.setattr(uu, "load_players_index", lambda: {})
    am.get_all_weekly_metrics_bulk(**kwargs)
    return conn.cur


def test_bulk_sql_always_has_limit_offset(monkeypatch):
    cur = _run_bulk_sql(monkeypatch)
    assert "LIMIT %s OFFSET %s" in cur.sql
    assert "ORDER BY player_id" in cur.sql
    assert cur.params[-2:] == (5000, 0)


def test_bulk_sql_limit_clamped_at_function_level(monkeypatch):
    cur = _run_bulk_sql(monkeypatch, season=2026, limit=10**9, offset=-5)
    assert cur.params[-2:] == (5000, 0)


def test_bulk_sql_scoped_where_clause(monkeypatch):
    cur = _run_bulk_sql(monkeypatch, season=2026, week_start=1, week_end=4,
                        position="WR", limit=100, offset=20)
    assert "season = %s" in cur.sql
    assert "week >= %s" in cur.sql
    assert "week <= %s" in cur.sql
    assert "position = %s" in cur.sql
    assert cur.params == (2026, 1, 4, "WR", 100, 20)
