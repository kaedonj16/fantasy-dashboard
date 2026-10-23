"""
Persistent CFBD response store (ingestion, opt-in CFBD_PERSIST_STORE=1).

The TTL-3600 in-memory cache meant every retry of the derive /
reference-population build / backtest re-bought the same immutable
historical responses until the monthly CFBD allowance was exhausted.
The persistent store keeps successful, non-empty responses in
Postgres (cfbd_response_cache) so each endpoint + params is fetched
exactly once, ever.

Pinned here:
- flag ON: first call fetches + stores; a second identical call (with
  the in-memory cache cleared, as in a new process) is a store hit
  with ZERO additional network calls;
- empty and failed responses are NEVER stored (no poisoning);
- flag OFF: no DB contact at all and behavior is unchanged (retry
  ladder intact, counters untouched);
- CFBD_CALL_BUDGET: the call that would exceed it raises
  CfbdBudgetExhausted BEFORE fetching;
- a 429 under the flag stops after at most 2 attempts (the live
  path's long back-off ladder does not apply);
- flag ON without DATABASE_URL degrades to flag-off behavior;
- a store save failure still returns the fetched data.

No network, no live DB: requests.get is faked and _persist_connect
returns an in-memory fake connection.
"""

import json
import os
import sys

import pytest
import requests

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)

from data_building.rookie_pipeline import ingestion as ing  # noqa: E402


# ─────────────────────────────────────────────────────────────────────
# Fakes
# ─────────────────────────────────────────────────────────────────────

class _FakeCursor:
    def __init__(self, store):
        self.store = store
        self._row = None

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def execute(self, sql, params=None):
        norm = " ".join(sql.split()).upper()
        if norm.startswith("CREATE TABLE"):
            return
        if norm.startswith("SELECT RESPONSE FROM CFBD_RESPONSE_CACHE"):
            key = params[0]
            self._row = (self.store[key],) if key in self.store else None
            return
        if norm.startswith("INSERT INTO CFBD_RESPONSE_CACHE"):
            if self.store.get("__fail_insert__"):
                raise RuntimeError("insert blew up")
            self.store[params[0]] = json.loads(params[1])
            return
        raise AssertionError(f"unexpected SQL: {sql}")

    def fetchone(self):
        return self._row


class _FakeConn:
    def __init__(self, store):
        self.store = store

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def cursor(self):
        return _FakeCursor(self.store)


class _Resp:
    def __init__(self, payload=None, status=200):
        self._payload = payload
        self.status_code = status
        self.reason = "OK" if status == 200 else f"status {status}"
        self.headers = {}

    def raise_for_status(self):
        if self.status_code >= 400:
            err = requests.HTTPError(f"HTTP {self.status_code}")
            err.response = self
            raise err

    def json(self):
        return self._payload


class _Transport:
    """Fake requests.get: serves queued responses, logs every call."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def get(self, url, headers=None, params=None, timeout=None):
        self.calls.append((url, params))
        if not self.responses:
            raise AssertionError("transport called with no queued response")
        return self.responses.pop(0)


def _boom_connect():
    raise AssertionError("_persist_connect must not be called here")


@pytest.fixture()
def cfbd(monkeypatch):
    """Clean CFBD state: empty in-memory cache, zeroed counters, no
    sleeps, and _persist_connect wired to an in-memory store dict."""
    ing._CFBD_CACHE.clear()
    monkeypatch.setattr(ing, "CFBD_NETWORK_CALLS", 0)
    monkeypatch.setattr(ing, "CFBD_STORE_HITS", 0)
    monkeypatch.setattr(ing, "_PERSIST_TABLE_READY", False)
    monkeypatch.setattr(ing, "_PERSIST_NO_DB_WARNED", False)
    monkeypatch.setattr(ing.time, "sleep", lambda _s: None)
    store: dict = {}
    monkeypatch.setattr(ing, "_persist_connect", lambda: _FakeConn(store))
    return store


@pytest.fixture()
def store_on(cfbd, monkeypatch):
    monkeypatch.setenv("CFBD_PERSIST_STORE", "1")
    monkeypatch.setenv("DATABASE_URL", "postgres://fake")
    monkeypatch.delenv("CFBD_CALL_BUDGET", raising=False)
    return cfbd


# ─────────────────────────────────────────────────────────────────────
# Flag ON: fetch once, store, hit forever
# ─────────────────────────────────────────────────────────────────────

class TestPersistentStore:
    def test_first_call_fetches_and_stores_second_is_store_hit(
            self, store_on, monkeypatch):
        payload = [{"player": "A", "yards": 10}]
        transport = _Transport([_Resp(payload)])
        monkeypatch.setattr(ing.requests, "get", transport.get)

        data = ing._cfbd_get("/stats/season", {"year": 2016})
        assert data == payload
        assert len(transport.calls) == 1
        assert len(store_on) == 1  # exactly one stored response

        # A new process has an empty in-memory cache; the store must
        # serve the response with zero additional network calls.
        ing._CFBD_CACHE.clear()
        again = ing._cfbd_get("/stats/season", {"year": 2016})
        assert again == payload
        assert len(transport.calls) == 1
        assert ing.CFBD_STORE_HITS == 1
        assert ing.CFBD_NETWORK_CALLS == 1

    def test_param_order_does_not_matter(self, store_on, monkeypatch):
        transport = _Transport([_Resp([{"x": 1}])])
        monkeypatch.setattr(ing.requests, "get", transport.get)
        ing._cfbd_get("/stats/season", {"year": 2016, "seasonType": "both"})
        ing._CFBD_CACHE.clear()
        data = ing._cfbd_get("/stats/season",
                             {"seasonType": "both", "year": 2016})
        assert data == [{"x": 1}]
        assert len(transport.calls) == 1

    def test_empty_response_never_stored(self, store_on, monkeypatch):
        transport = _Transport([_Resp([])])
        monkeypatch.setattr(ing.requests, "get", transport.get)
        assert ing._cfbd_get("/stats/season", {"year": 2016}) == []
        assert store_on == {}

    def test_failed_response_never_stored(self, store_on, monkeypatch):
        transport = _Transport([_Resp(status=500), _Resp(status=500)])
        monkeypatch.setattr(ing.requests, "get", transport.get)
        assert ing._cfbd_get("/stats/season", {"year": 2016}) is None
        assert store_on == {}
        # Under the flag the retry ladder collapses to 2 attempts.
        assert len(transport.calls) == 2

    def test_save_failure_still_returns_data(self, store_on, monkeypatch):
        store_on["__fail_insert__"] = True
        transport = _Transport([_Resp([{"x": 1}])])
        monkeypatch.setattr(ing.requests, "get", transport.get)
        assert ing._cfbd_get("/stats/season", {"year": 2016}) == [{"x": 1}]
        assert len(transport.calls) == 1


# ─────────────────────────────────────────────────────────────────────
# Budget: stop BEFORE the call that would exceed it
# ─────────────────────────────────────────────────────────────────────

class TestCallBudget:
    def test_budget_exhaustion_raises_before_fetching(
            self, store_on, monkeypatch):
        monkeypatch.setenv("CFBD_CALL_BUDGET", "1")
        transport = _Transport([_Resp([{"x": 1}]), _Resp([{"y": 2}])])
        monkeypatch.setattr(ing.requests, "get", transport.get)

        assert ing._cfbd_get("/stats/season", {"year": 2016}) == [{"x": 1}]
        with pytest.raises(ing.CfbdBudgetExhausted):
            ing._cfbd_get("/stats/season", {"year": 2017})
        # The second request never reached the network.
        assert len(transport.calls) == 1
        assert ing.CFBD_NETWORK_CALLS == 1

    def test_store_hits_do_not_consume_budget(self, store_on, monkeypatch):
        monkeypatch.setenv("CFBD_CALL_BUDGET", "1")
        transport = _Transport([_Resp([{"x": 1}])])
        monkeypatch.setattr(ing.requests, "get", transport.get)
        ing._cfbd_get("/stats/season", {"year": 2016})
        ing._CFBD_CACHE.clear()
        # Store hit: free, even with the budget fully spent.
        assert ing._cfbd_get("/stats/season", {"year": 2016}) == [{"x": 1}]
        assert ing.CFBD_NETWORK_CALLS == 1


# ─────────────────────────────────────────────────────────────────────
# 429 discipline under the flag
# ─────────────────────────────────────────────────────────────────────

class TestRateLimitDiscipline:
    def test_429_stops_after_two_attempts(self, store_on, monkeypatch):
        transport = _Transport([_Resp(status=429)] * 6)
        monkeypatch.setattr(ing.requests, "get", transport.get)
        assert ing._cfbd_get("/stats/season", {"year": 2016}) is None
        assert len(transport.calls) == 2
        assert store_on == {}

    def test_429_ladder_unchanged_with_flag_off(self, cfbd, monkeypatch):
        monkeypatch.delenv("CFBD_PERSIST_STORE", raising=False)
        monkeypatch.setattr(ing, "_persist_connect", _boom_connect)
        transport = _Transport([_Resp(status=429)] * 6)
        monkeypatch.setattr(ing.requests, "get", transport.get)
        assert ing._cfbd_get("/stats/season", {"year": 2016},
                             retries=3) is None
        assert len(transport.calls) == 3


# ─────────────────────────────────────────────────────────────────────
# Flag OFF / degraded: the live path is untouched
# ─────────────────────────────────────────────────────────────────────

class TestFlagOff:
    def test_flag_off_never_touches_the_store(self, cfbd, monkeypatch):
        monkeypatch.delenv("CFBD_PERSIST_STORE", raising=False)
        monkeypatch.setenv("DATABASE_URL", "postgres://fake")
        monkeypatch.setattr(ing, "_persist_connect", _boom_connect)
        transport = _Transport([_Resp([{"x": 1}])])
        monkeypatch.setattr(ing.requests, "get", transport.get)
        assert ing._cfbd_get("/stats/season", {"year": 2016}) == [{"x": 1}]
        assert len(transport.calls) == 1
        # Counters belong to the opt-in path; the live path is unchanged.
        assert ing.CFBD_NETWORK_CALLS == 0
        assert ing.CFBD_STORE_HITS == 0

    def test_flag_on_without_database_url_degrades_to_live(
            self, cfbd, monkeypatch):
        monkeypatch.setenv("CFBD_PERSIST_STORE", "1")
        monkeypatch.delenv("DATABASE_URL", raising=False)
        monkeypatch.setattr(ing, "_persist_connect", _boom_connect)
        transport = _Transport([_Resp([{"x": 1}])])
        monkeypatch.setattr(ing.requests, "get", transport.get)
        assert ing._cfbd_get("/stats/season", {"year": 2016}) == [{"x": 1}]
        assert len(transport.calls) == 1
