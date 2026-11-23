"""Parallel Draft DB backfill (2026-11 rework).

The corpus outgrew the original single-threaded, silent backfill
(~134k drafts / ~124k leagues, 4 Sleeper calls per league). These tests
fake _get and the DB connection - no real network, no real DB - and pin:

  * leagues fan out in parallel, payloads fetched ONCE per league
  * recent-first work order (SQL ordering + league grouping order)
  * resume selection (metadata: NULL columns only; grades: missing
    grade_summary only, unless --recompute-all)
  * progress logging is emitted
  * one dead league never stops the others
  * CLI flags parse and wire through, result dict shape unchanged
  * the crawler's HTTP session is thread-local (it backs _get, which
    both the crawl and this backfill call from thread pools)
"""
from __future__ import annotations

import logging
import threading
from collections import Counter

import pytest

from data_building.trade_intel import draft_adp_crawler as crawler
from data_building.trade_intel import draft_db_backfill as bf


# ---------------------------------------------------------------------------
# Fakes (mirroring the psycopg3 surface the backfill uses: conn.execute()
# returning a cursor-like with fetchall(), conn as a context manager)
# ---------------------------------------------------------------------------

class _Result:
    def __init__(self, rows):
        self._rows = rows

    def fetchall(self):
        return list(self._rows)

    def fetchone(self):
        return self._rows[0] if self._rows else None


class FakeStore:
    def __init__(self, metadata_rows=(), grade_rows=()):
        self.metadata_rows = list(metadata_rows)
        self.grade_rows = list(grade_rows)
        self.queries = []   # (sql, params) in call order
        self.updates = []   # params of UPDATE statements
        self._lock = threading.Lock()

    def execute(self, sql, params=()):
        with self._lock:
            self.queries.append((sql, tuple(params or ())))
        if sql.startswith("UPDATE"):
            with self._lock:
                self.updates.append(tuple(params))
            return _Result([])
        if "status = 'complete'" in sql:
            return _Result(self.grade_rows)
        return _Result(self.metadata_rows)


class FakeConn:
    def __init__(self, store):
        self.store = store

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, sql, params=()):
        return self.store.execute(sql, params)


def _rows_for(leagues):
    return [{"draft_id": d, "league_id": lid}
            for lid, drafts in leagues.items() for d in drafts]


def make_fake_get(leagues, fail_leagues=()):
    """Sleeper-shaped payloads keyed off the /league/{id}... paths."""
    calls = []
    lock = threading.Lock()

    def fake_get(path):
        with lock:
            calls.append(path)
        parts = path.strip("/").split("/")
        if parts[0] == "league":
            lid = parts[1]
            if lid in fail_leagues:
                raise RuntimeError("league fetch exploded")
            if len(parts) == 2:
                return {"name": f"League {lid}",
                        "settings": {"type": 2,
                                     "scoring_settings": {"rec": 1.0}}}
            kind = parts[2]
            if kind == "users":
                return [{"user_id": "u1", "display_name": "Owner One",
                         "metadata": {}}]
            if kind == "rosters":
                return [{"roster_id": 1, "owner_id": "u1"}]
            if kind == "drafts":
                return [{"draft_id": d, "type": "snake",
                         "slot_to_roster_id": {"1": 1}, "metadata": {}}
                        for d in leagues[lid]]
        if parts[0] == "draft":
            return {}
        return None

    return fake_get, calls


@pytest.fixture()
def store(monkeypatch):
    s = FakeStore()
    monkeypatch.setattr(bf, "get_conn", lambda: FakeConn(s))
    return s


# ---------------------------------------------------------------------------
# Metadata phase
# ---------------------------------------------------------------------------

def test_parallel_metadata_fills_all_leagues_payloads_fetched_once(
        store, monkeypatch):
    leagues = {"L1": ["d1", "d2"], "L2": ["d3"], "L3": ["d4", "d5", "d6"]}
    store.metadata_rows = _rows_for(leagues)
    fake_get, calls = make_fake_get(leagues)
    monkeypatch.setattr(bf, "_get", fake_get)

    result = bf.backfill_metadata(workers=4)

    assert result == {"considered": 6, "filled": 6, "failed": 0}
    # One UPDATE per draft, each naming its draft_id.
    assert sorted(p[-1] for p in store.updates) == [
        "d1", "d2", "d3", "d4", "d5", "d6"]
    # Payloads fetched exactly once per league (4 calls), never per draft,
    # and the /draft/{id} fallback never fired (draft list covered all).
    counts = Counter(calls)
    for lid in leagues:
        assert counts[f"/league/{lid}"] == 1
        assert counts[f"/league/{lid}/users"] == 1
        assert counts[f"/league/{lid}/rosters"] == 1
        assert counts[f"/league/{lid}/drafts"] == 1
    assert not any(c.startswith("/draft/") for c in calls)


def test_recent_first_sql_and_league_work_order(store, monkeypatch):
    # Rows as the recent-first SQL returns them: L2026's drafts bracket
    # L2025's, so L2026 (newest) must enter the work queue first.
    store.metadata_rows = [
        {"draft_id": "d1", "league_id": "L2026"},
        {"draft_id": "d2", "league_id": "L2025"},
        {"draft_id": "d3", "league_id": "L2026"},
    ]
    leagues = {"L2026": ["d1", "d3"], "L2025": ["d2"]}
    fake_get, calls = make_fake_get(leagues)
    monkeypatch.setattr(bf, "_get", fake_get)

    result = bf.backfill_metadata(workers=1)

    assert result["filled"] == 3
    league_fetches = [c for c in calls
                      if c.startswith("/league/") and c.count("/") == 2]
    assert league_fetches == ["/league/L2026", "/league/L2025"]
    sql = store.queries[0][0]
    assert "league_name IS NULL" in sql          # resume: pending only
    assert "season DESC" in sql                  # recent first
    assert "draft_started_at DESC" in sql


def test_metadata_limit_applies_to_drafts(store, monkeypatch):
    store.metadata_rows = _rows_for({"L1": ["d1"]})
    fake_get, _ = make_fake_get({"L1": ["d1"]})
    monkeypatch.setattr(bf, "_get", fake_get)

    bf.backfill_metadata(limit=200, workers=1)

    sql, params = store.queries[0]
    assert "LIMIT %s" in sql
    assert params[-1] == 200


def test_failed_league_does_not_stop_others(store, monkeypatch):
    leagues = {"OK": ["d1", "d2"], "BAD": ["d3", "d4"]}
    store.metadata_rows = _rows_for(leagues)
    fake_get, _ = make_fake_get(leagues, fail_leagues={"BAD"})
    monkeypatch.setattr(bf, "_get", fake_get)

    result = bf.backfill_metadata(workers=2)

    assert result == {"considered": 4, "filled": 2, "failed": 2}
    assert sorted(p[-1] for p in store.updates) == ["d1", "d2"]


def test_progress_logging_emitted(store, monkeypatch, caplog):
    leagues = {"L1": ["d1"], "L2": ["d2"]}
    store.metadata_rows = _rows_for(leagues)
    fake_get, _ = make_fake_get(leagues)
    monkeypatch.setattr(bf, "_get", fake_get)
    monkeypatch.setattr(bf, "_PROGRESS_EVERY_LEAGUES", 1)

    with caplog.at_level(logging.INFO, logger=bf.__name__):
        bf.backfill_metadata(workers=2)

    messages = [r.getMessage() for r in caplog.records]
    assert any("metadata progress" in m for m in messages)
    assert any("metadata done" in m for m in messages)


# ---------------------------------------------------------------------------
# Grades phase
# ---------------------------------------------------------------------------

def _grade_recorder(monkeypatch):
    graded = []
    lock = threading.Lock()

    def fake_compute(conn, draft_id):
        with lock:
            graded.append(draft_id)
        return {"summary": True}

    monkeypatch.setattr(bf, "compute_and_store_grade_summary", fake_compute)
    return graded


def test_grades_parallel_and_missing_only_by_default(store, monkeypatch):
    store.grade_rows = [{"draft_id": f"g{i}"} for i in range(5)]
    graded = _grade_recorder(monkeypatch)

    result = bf.backfill_grades(workers=3)

    assert result == {"considered": 5, "graded": 5}
    assert sorted(graded) == [f"g{i}" for i in range(5)]
    sql = store.queries[0][0]
    assert "grade_summary IS NULL" in sql        # resume: missing only
    assert "season DESC" in sql                  # recent first


def test_grades_recompute_all_restores_full_selection(store, monkeypatch):
    store.grade_rows = [{"draft_id": "g1"}]
    _grade_recorder(monkeypatch)

    bf.backfill_grades(workers=1, recompute_all=True)

    sql = store.queries[0][0]
    assert "status = 'complete'" in sql
    assert "grade_summary IS NULL" not in sql


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def test_cli_flags_parse_and_wire(monkeypatch):
    seen = {}

    def fake_metadata(limit, league_id, workers):
        seen["metadata"] = (limit, league_id, workers)
        return {"considered": 0, "filled": 0, "failed": 0}

    def fake_grades(limit, league_id, workers, recompute_all):
        seen["grades"] = (limit, league_id, workers, recompute_all)
        return {"considered": 0, "graded": 0}

    monkeypatch.setattr(bf, "backfill_metadata", fake_metadata)
    monkeypatch.setattr(bf, "backfill_grades", fake_grades)
    monkeypatch.setattr(
        "sys.argv",
        ["draft_db_backfill", "--limit", "200", "--league-id", "L9",
         "--workers", "3", "--grade-workers", "2", "--recompute-all"])

    result = bf.main()

    assert seen["metadata"] == (200, "L9", 3)
    assert seen["grades"] == (200, "L9", 2, True)
    assert result == {"metadata": {"considered": 0, "filled": 0,
                                   "failed": 0},
                      "grades": {"considered": 0, "graded": 0}}


def test_cli_metadata_only_skips_grades(monkeypatch):
    monkeypatch.setattr(bf, "backfill_metadata",
                        lambda *a, **k: {"considered": 0, "filled": 0,
                                         "failed": 0})
    monkeypatch.setattr(bf, "backfill_grades",
                        lambda *a, **k: pytest.fail("grades must not run"))
    monkeypatch.setattr("sys.argv", ["draft_db_backfill", "--metadata-only"])

    result = bf.main()

    assert set(result) == {"metadata"}


# ---------------------------------------------------------------------------
# Crawler session thread-safety (backs _get for both thread pools)
# ---------------------------------------------------------------------------

def test_crawler_session_is_thread_local():
    pytest.importorskip("requests")
    main_session = crawler._session()
    assert crawler._session() is main_session

    seen = []

    def grab():
        seen.append(crawler._session())
        seen.append(crawler._session())

    t = threading.Thread(target=grab)
    t.start()
    t.join()

    assert seen[0] is seen[1]            # stable within the worker thread
    assert seen[0] is not main_session   # but not the main thread's session
