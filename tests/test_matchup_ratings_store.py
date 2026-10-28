"""Unit tests for dashboard_services.matchup_ratings_store (no live database).

The store persists the Wednesday cron's matchup ratings so they survive
the cron's ephemeral container and reach the web service. All DB access
goes through a faked get_conn; psycopg is only needed for the Json adapter
on the save path, so skip when the driver is absent.
"""
import pytest

from contextlib import contextmanager
from datetime import datetime
from types import ModuleType
import sys

from dashboard_services import matchup_ratings_store as store


@pytest.fixture()
def stub_json_adapter(monkeypatch):
    class _FakeJson:
        def __init__(self, obj):
            self.obj = obj

    for name in ("psycopg", "psycopg.types", "psycopg.types.json"):
        mod = ModuleType(name)
        monkeypatch.setitem(sys.modules, name, mod)
    sys.modules["psycopg.types.json"].Json = _FakeJson
    sys.modules["psycopg"].types = sys.modules["psycopg.types"]
    sys.modules["psycopg.types"].json = sys.modules["psycopg.types.json"]
    return _FakeJson


@pytest.fixture()
def reset_tables_ready():
    store._TABLES_READY = False
    yield
    store._TABLES_READY = False


class _FakeConn:
    def __init__(self, fetchone_result=None):
        self.writes = []
        self._fetchone_result = fetchone_result

    def execute(self, sql, args=()):
        self.writes.append((sql, args))
        return self

    def fetchone(self):
        return self._fetchone_result

    def commit(self):
        pass


def _patch_conn(monkeypatch, conn):
    @contextmanager
    def fake_get_conn(*a, **k):
        yield conn

    monkeypatch.setattr("dashboard_services.db.get_conn", fake_get_conn)


def _payload():
    return {
        "season": 2026,
        "through_week": 7,
        "generated_at": "2026-10-28T23:20:31+00:00",
        "scoring_profile": "e644f15a2c198fe2",
        "ratings": {"BUF": {"RB": {"adjusted_multiplier": 1.1}}},
    }


def test_save_upserts_row_keyed_by_season_and_profile(monkeypatch, reset_tables_ready, stub_json_adapter):
    conn = _FakeConn()
    _patch_conn(monkeypatch, conn)

    assert store.save_matchup_ratings(2026, _payload()) is True

    upserts = [w for w in conn.writes if "INSERT INTO matchup_ratings" in w[0]]
    assert len(upserts) == 1
    sql, args = upserts[0]
    assert "ON CONFLICT (season, scoring_profile) DO UPDATE" in sql
    assert args[0] == 2026
    assert args[1] == "e644f15a2c198fe2"
    assert args[2] == 7
    assert isinstance(args[3], datetime)
    assert args[4].obj["ratings"] == {"BUF": {"RB": {"adjusted_multiplier": 1.1}}}


def test_save_refuses_empty_ratings(monkeypatch, reset_tables_ready):
    conn = _FakeConn()
    _patch_conn(monkeypatch, conn)

    assert store.save_matchup_ratings(2026, {"ratings": {}}) is False
    assert conn.writes == []


def test_save_never_raises_when_db_down(monkeypatch, reset_tables_ready):
    @contextmanager
    def boom(*a, **k):
        raise RuntimeError("db unreachable")
        yield  # pragma: no cover

    monkeypatch.setattr("dashboard_services.db.get_conn", boom)
    assert store.save_matchup_ratings(2026, _payload()) is False


def test_load_returns_payload(monkeypatch, reset_tables_ready):
    conn = _FakeConn(fetchone_result=(_payload(),))
    _patch_conn(monkeypatch, conn)

    row = store.load_matchup_ratings(2026, "e644f15a2c198fe2")
    assert row is not None
    assert row["through_week"] == 7
    assert row["ratings"]["BUF"]["RB"]["adjusted_multiplier"] == 1.1


def test_load_returns_none_when_no_row(monkeypatch, reset_tables_ready):
    conn = _FakeConn(fetchone_result=None)
    _patch_conn(monkeypatch, conn)
    assert store.load_matchup_ratings(2026, "e644f15a2c198fe2") is None


def test_load_returns_none_when_db_down(monkeypatch, reset_tables_ready):
    @contextmanager
    def boom(*a, **k):
        raise RuntimeError("db unreachable")
        yield  # pragma: no cover

    monkeypatch.setattr("dashboard_services.db.get_conn", boom)
    assert store.load_matchup_ratings(2026) is None
