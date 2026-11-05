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
        "scoring_profile": "3ce7b75fe4f40427",
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
    assert args[1] == "3ce7b75fe4f40427"
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
    # Production connections use psycopg's dict_row factory
    # (dashboard_services.db.get_conn), so fetchone() returns a dict keyed
    # by column name, never a tuple. The fake must use that shape: a tuple
    # row let row[0] pass in tests while every production load raised
    # KeyError and silently missed.
    conn = _FakeConn(fetchone_result={"payload": _payload()})
    _patch_conn(monkeypatch, conn)

    row = store.load_matchup_ratings(2026, "3ce7b75fe4f40427")
    assert row is not None
    assert row["through_week"] == 7
    assert row["ratings"]["BUF"]["RB"]["adjusted_multiplier"] == 1.1


def test_load_returns_none_when_no_row(monkeypatch, reset_tables_ready):
    conn = _FakeConn(fetchone_result=None)
    _patch_conn(monkeypatch, conn)
    assert store.load_matchup_ratings(2026, "3ce7b75fe4f40427") is None


def test_load_returns_none_when_db_down(monkeypatch, reset_tables_ready):
    @contextmanager
    def boom(*a, **k):
        raise RuntimeError("db unreachable")
        yield  # pragma: no cover

    monkeypatch.setattr("dashboard_services.db.get_conn", boom)
    assert store.load_matchup_ratings(2026) is None


class _RoundTripConn:
    """In-memory stand-in for the matchup_ratings table: INSERTs land in a
    dict keyed by (season, scoring_profile) and SELECTs answer from it, so a
    save/load round trip exercises the real keying on both sides."""

    def __init__(self):
        self.rows = {}
        self._pending = None

    def execute(self, sql, args=()):
        if "INSERT INTO matchup_ratings" in sql:
            season, profile, _week, _generated, payload = args
            self.rows[(season, profile)] = payload.obj
        elif "SELECT payload FROM matchup_ratings" in sql and len(args) == 2:
            stored = self.rows.get((args[0], args[1]))
            # dict_row shape, matching production connections.
            self._pending = {"payload": stored} if stored is not None else None
        return self

    def fetchone(self):
        return self._pending

    def commit(self):
        pass


def test_default_build_round_trips_under_app_default_lookup_keys(
        monkeypatch, reset_tables_ready, stub_json_adapter):
    from utils.nfl import scoring_profile_hash

    conn = _RoundTripConn()
    _patch_conn(monkeypatch, conn)

    # Stamped exactly as build_matchup_ratings stamps the default profile
    # (scoring_settings None -> {"rec": 1}).
    payload = _payload()
    payload["scoring_profile"] = scoring_profile_hash({"rec": 1})
    assert store.save_matchup_ratings(2026, payload) is True

    # app.py _load_matchup_ratings tries these keys, in this order, for a
    # default request; before the int/float hash normalization the first key
    # never matched the builder's stamp and the read fell through to the
    # stale flat file.
    hit = None
    for key in [scoring_profile_hash({"rec": 1.0}), "standard-ppr"]:
        hit = store.load_matchup_ratings(2026, key)
        if hit and (hit.get("ratings") or {}):
            break
    assert hit is not None
    assert hit["ratings"]["BUF"]["RB"]["adjusted_multiplier"] == 1.1
