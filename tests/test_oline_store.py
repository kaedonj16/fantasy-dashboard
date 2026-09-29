"""Unit tests for dashboard_services.oline_store (no live database).

The store persists the weekly O-line ratings so they survive the cron's
ephemeral container. All DB access goes through a faked get_conn; psycopg is
only needed for the Json adapter on the save path, so skip when the driver is
absent (the CI lint shard installs neither pandas nor psycopg).
"""
import pytest

from contextlib import contextmanager
from datetime import datetime, timezone
from types import ModuleType
import sys

from dashboard_services import oline_store


@pytest.fixture()
def stub_json_adapter(monkeypatch):
    """Stand-in for psycopg.types.json.Json when the driver is absent."""
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
    oline_store._TABLES_READY = False
    yield
    oline_store._TABLES_READY = False


class _FakeConn:
    """Minimal stand-in for a psycopg connection."""

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


def _ratings_payload():
    return {
        "season": 2026,
        "through_week": 3,
        "generated_at": "2026-09-29T17:38:00+00:00",
        "ratings": {"KC": {"composite": 90.0}, "BUF": {"composite": 80.0}},
    }


def test_save_upserts_row(monkeypatch, reset_tables_ready, stub_json_adapter):
    conn = _FakeConn()
    _patch_conn(monkeypatch, conn)

    assert oline_store.save_oline_ratings(2026, _ratings_payload()) is True

    upserts = [w for w in conn.writes if "INSERT INTO oline_ratings" in w[0]]
    assert len(upserts) == 1
    sql, args = upserts[0]
    assert "ON CONFLICT (season) DO UPDATE" in sql
    assert args[0] == 2026
    assert args[1] == 3
    assert isinstance(args[2], datetime)
    assert args[2].year == 2026
    # psycopg Json adapter wraps the ratings dict
    assert args[3].obj == {"KC": {"composite": 90.0},
                           "BUF": {"composite": 80.0}}


def test_save_refuses_empty_ratings(monkeypatch, reset_tables_ready):
    conn = _FakeConn()
    _patch_conn(monkeypatch, conn)

    assert oline_store.save_oline_ratings(2026, {"ratings": {}}) is False
    assert conn.writes == []


def test_save_never_raises_when_db_down(monkeypatch, reset_tables_ready):
    @contextmanager
    def boom(*a, **k):
        raise RuntimeError("db unreachable")
        yield  # pragma: no cover

    monkeypatch.setattr("dashboard_services.db.get_conn", boom)
    assert oline_store.save_oline_ratings(2026, _ratings_payload()) is False


def test_load_round_trip_shape(monkeypatch, reset_tables_ready):
    conn = _FakeConn(fetchone_result=(
        3,
        datetime(2026, 9, 29, 17, 38, tzinfo=timezone.utc),
        {"KC": {"composite": 90.0}},
    ))
    _patch_conn(monkeypatch, conn)

    row = oline_store.load_oline_ratings(2026)
    assert row is not None
    assert row["through_week"] == 3
    assert row["generated_at"].startswith("2026-09-29")
    assert row["ratings"] == {"KC": {"composite": 90.0}}


def test_load_returns_none_when_no_row(monkeypatch, reset_tables_ready):
    conn = _FakeConn(fetchone_result=None)
    _patch_conn(monkeypatch, conn)
    assert oline_store.load_oline_ratings(2026) is None


def test_load_returns_none_when_db_down(monkeypatch, reset_tables_ready):
    @contextmanager
    def boom(*a, **k):
        raise RuntimeError("db unreachable")
        yield  # pragma: no cover

    monkeypatch.setattr("dashboard_services.db.get_conn", boom)
    assert oline_store.load_oline_ratings(2026) is None


def test_newest_season(monkeypatch, reset_tables_ready):
    conn = _FakeConn(fetchone_result=(2026,))
    _patch_conn(monkeypatch, conn)
    assert oline_store.newest_oline_season() == 2026


def test_newest_season_none_when_db_down(monkeypatch, reset_tables_ready):
    @contextmanager
    def boom(*a, **k):
        raise RuntimeError("db unreachable")
        yield  # pragma: no cover

    monkeypatch.setattr("dashboard_services.db.get_conn", boom)
    assert oline_store.newest_oline_season() is None


def test_init_tables_runs_once(monkeypatch, reset_tables_ready):
    conn = _FakeConn()
    _patch_conn(monkeypatch, conn)

    oline_store.init_oline_tables()
    oline_store.init_oline_tables()
    creates = [w for w in conn.writes if "CREATE TABLE IF NOT EXISTS" in w[0]]
    assert len(creates) == 1
