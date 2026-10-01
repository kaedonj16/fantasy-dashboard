"""Historical weekly snapshots: completed weeks under older scoring
versions stay listed and servable, while the default latest view keeps
requiring the current SCORING_VERSION.

Regression context: the week selector filtered completed runs by the
current scoring version, so when the scorer bumped weekly-v5 -> weekly-v6
every earlier week's snapshot vanished from the selector (only the
current week and Preseason remained). Historical weeks are now served
verbatim under the version their own run recorded.
"""
from __future__ import annotations

from datetime import date, datetime

import pytest

from data_building.breakout_engine import weekly_store
from data_building.breakout_engine.weekly_breakout import SCORING_VERSION

OLD_VERSION = "weekly-v5"


class _Result:
    def __init__(self, rows=None, one=None):
        self._rows = rows or []
        self._one = one

    def fetchall(self):
        return list(self._rows)

    def fetchone(self):
        return self._one


class _ListConn:
    """Conn for list_completed_weeks: returns rows as Postgres would after
    the DISTINCT ON dedupe (one row per week, serving run's version)."""

    def __init__(self, rows):
        self._rows = rows
        self.captured = {}

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def execute(self, query, params):
        self.captured["query"] = query
        self.captured["params"] = params
        return _Result(rows=self._rows)


def _no_init(monkeypatch):
    monkeypatch.setattr(weekly_store, "init_weekly_breakout_db", lambda: None)


def test_list_completed_weeks_includes_older_scoring_versions(monkeypatch):
    _no_init(monkeypatch)
    conn = _ListConn([
        {"w": 1, "d": date(2026, 9, 13), "v": OLD_VERSION},
        {"w": 2, "d": date(2026, 9, 20), "v": OLD_VERSION},
        {"w": 3, "d": date(2026, 9, 27), "v": OLD_VERSION},
        {"w": 4, "d": date(2026, 10, 4), "v": SCORING_VERSION},
    ])
    monkeypatch.setattr(weekly_store, "get_conn", lambda: conn)

    weeks = weekly_store.list_completed_weeks(2026)

    assert [w["as_of_week"] for w in weeks] == [1, 2, 3, 4]
    assert [w["scoring_version"] for w in weeks] == [
        OLD_VERSION, OLD_VERSION, OLD_VERSION, SCORING_VERSION]
    # The bug: the listing query gated runs on the current version.
    assert "r.scoring_version = %s" not in conn.captured["query"]
    assert "DISTINCT ON" in conn.captured["query"]
    assert conn.captured["params"] == (2026,)


def _run_row(run_id, week, version, completed_at):
    return {
        "id": run_id, "season": 2026, "as_of_week": week,
        "scoring_version": version, "status": "completed",
        "completed_at": completed_at, "as_of_date": date(2026, 9, 20),
        "expected_row_count": 3, "inserted_row_count": 3, "detail": {},
    }


class _Cursor:
    def __init__(self, conn):
        self._conn = conn

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def execute(self, query, params):
        self._conn.cursor_queries.append((query, params))

    def fetchall(self):
        return self._conn.score_rows


class _StoreConn:
    """Dispatch fake for load_weekly_candidates / get_serving_run."""

    def __init__(self, *, serving_run=None, latest_week=None,
                 score_rows=None, latest_run=None, completed_run=None):
        self.serving_run = serving_run
        self.latest_week = latest_week
        self.score_rows = score_rows or []
        self.latest_run = latest_run
        self.completed_run = completed_run
        self.queries = []
        self.cursor_queries = []

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def cursor(self):
        return _Cursor(self)

    def execute(self, query, params):
        self.queries.append((query, params))
        if "MAX(r.as_of_week)" in query:
            return _Result(one={"w": self.latest_week})
        if "r.completed_at DESC" in query:
            return _Result(one=self.serving_run)
        if "calculated_at DESC" in query:
            return _Result(one=self.latest_run)
        if "scoring_version=%s" in query:
            return _Result(one=self.completed_run)
        return _Result()


def _score_row(pid, run_id, week, version):
    return {
        "player_id": pid, "player_name": f"Player {pid}", "run_id": run_id,
        "season": 2026, "as_of_week": week, "scoring_version": version,
        "breakout_score": 50.0, "confidence": 70.0,
        "classification": "watchlist", "evidence": {},
        "as_of_date": date(2026, 9, 20),
    }


def test_explicit_old_version_week_serves_its_stored_snapshot(monkeypatch):
    _no_init(monkeypatch)
    run = _run_row(7, 2, OLD_VERSION, datetime(2026, 9, 20, 12, 0))
    rows = [_score_row("p1", 7, 2, OLD_VERSION)]
    conn = _StoreConn(serving_run=run, score_rows=rows,
                      latest_run={"as_of_week": 4, "status": "completed"})
    monkeypatch.setattr(weekly_store, "get_conn", lambda: conn)

    payload = weekly_store.load_weekly_candidates(2026, as_of_week=2)

    assert payload["data_available"] is True
    assert [c["player_id"] for c in payload["candidates"]] == ["p1"]
    # Labeled with the version the snapshot was actually scored under.
    assert payload["scoring_version"] == OLD_VERSION
    # Rows come from exactly the serving run.
    query, params = conn.cursor_queries[0]
    assert "s.run_id = %s" in query
    assert params[0] == 7


def test_two_version_week_serves_most_recently_completed_run(monkeypatch):
    _no_init(monkeypatch)
    # Postgres applies the ORDER BY (originals first, then completed_at
    # DESC, id DESC); the fake hands back the run that ordering selects:
    # the newer v6 re-run, both runs being originals.
    run = _run_row(11, 2, SCORING_VERSION, datetime(2026, 10, 5, 9, 0))
    conn = _StoreConn(serving_run=run)
    monkeypatch.setattr(weekly_store, "get_conn", lambda: conn)

    serving = weekly_store.get_serving_run(2026, 2)

    assert serving["id"] == 11
    query = conn.queries[0][0]
    assert "reconstructed" in query
    assert "r.completed_at DESC, r.id DESC" in query
    # The original-over-reconstruction preference sorts BEFORE recency.
    assert query.index("reconstructed") < query.index("r.completed_at DESC")
    assert "r.scoring_version = %s" not in query


def test_serving_run_prefers_original_over_newer_reconstruction(monkeypatch):
    _no_init(monkeypatch)
    # Week 2 has an original v5 run (completed Sep 20) and a NEWER
    # reconstructed v6 run (completed Oct 5). Under the serving ORDER BY
    # the original wins, so Postgres hands back the v5 run: the selector
    # keeps serving the board as it was actually published that week.
    original = _run_row(7, 2, OLD_VERSION, datetime(2026, 9, 20, 12, 0))
    conn = _StoreConn(serving_run=original)
    monkeypatch.setattr(weekly_store, "get_conn", lambda: conn)

    serving = weekly_store.get_serving_run(2026, 2)

    assert serving["id"] == 7
    assert serving["scoring_version"] == OLD_VERSION


def test_serving_run_falls_back_to_reconstruction_when_no_original(monkeypatch):
    _no_init(monkeypatch)
    recon = _run_row(13, 1, SCORING_VERSION, datetime(2026, 10, 5, 9, 0))
    recon["detail"] = {"reconstructed": True}
    conn = _StoreConn(serving_run=recon)
    monkeypatch.setattr(weekly_store, "get_conn", lambda: conn)

    serving = weekly_store.get_serving_run(2026, 1)

    assert serving["id"] == 13


class _LatestConn:
    def __init__(self, week):
        self._week = week
        self.captured = {}

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def execute(self, query, params):
        self.captured["query"] = query
        self.captured["params"] = params
        return _Result(one={"w": self._week})


def test_latest_scored_week_excludes_reconstructed_runs(monkeypatch):
    _no_init(monkeypatch)
    conn = _LatestConn(4)
    monkeypatch.setattr(weekly_store, "get_conn", lambda: conn)

    assert weekly_store.latest_scored_week(2026) == 4
    # A reconstructed run for a later week must not become the default
    # latest view: the MAX query filters reconstructions out.
    assert "reconstructed" in conn.captured["query"]
    assert conn.captured["params"] == (2026, SCORING_VERSION)


def test_list_completed_weeks_prefers_original_run_per_week(monkeypatch):
    _no_init(monkeypatch)
    conn = _ListConn([
        {"w": 1, "d": date(2026, 9, 13), "v": SCORING_VERSION},
        {"w": 2, "d": date(2026, 9, 20), "v": OLD_VERSION},
    ])
    monkeypatch.setattr(weekly_store, "get_conn", lambda: conn)

    weeks = weekly_store.list_completed_weeks(2026)

    assert [w["as_of_week"] for w in weeks] == [1, 2]
    query = conn.captured["query"]
    # The advertised version/date come from the serving run, so the
    # DISTINCT ON ordering applies the same original-first preference.
    assert "reconstructed" in query
    assert query.index("reconstructed") < query.index("r.completed_at DESC")


def test_explicit_week_without_completed_run_is_unavailable(monkeypatch):
    _no_init(monkeypatch)
    conn = _StoreConn(serving_run=None)
    monkeypatch.setattr(weekly_store, "get_conn", lambda: conn)

    payload = weekly_store.load_weekly_candidates(2026, as_of_week=3)

    assert payload["data_available"] is False
    assert payload["candidates"] == []
    assert payload["as_of_week"] == 3


def test_default_latest_view_still_requires_current_version(monkeypatch):
    _no_init(monkeypatch)
    # Only old-version history exists: no current-version week at all.
    conn = _StoreConn(latest_week=None)
    monkeypatch.setattr(weekly_store, "get_conn", lambda: conn)

    payload = weekly_store.load_weekly_candidates(2026)

    assert payload["data_available"] is False
    assert payload["candidates"] == []


def test_default_latest_view_serves_current_version_snapshot(monkeypatch):
    _no_init(monkeypatch)
    rows = [_score_row("p9", 12, 4, SCORING_VERSION)]
    completed = _run_row(12, 4, SCORING_VERSION, datetime(2026, 10, 4, 12, 0))
    conn = _StoreConn(latest_week=4, score_rows=rows, completed_run=completed,
                      latest_run={"as_of_week": 4, "status": "completed"})
    monkeypatch.setattr(weekly_store, "get_conn", lambda: conn)

    payload = weekly_store.load_weekly_candidates(2026)

    assert payload["data_available"] is True
    assert payload["scoring_version"] == SCORING_VERSION
    query, params = conn.cursor_queries[0]
    # The default path keeps its version gate.
    assert "s.scoring_version = %s" in query
    assert SCORING_VERSION in params


# ---------------------------------------------------------------------------
# breakout_api week list: old weeks selectable, default stays current-version
# ---------------------------------------------------------------------------

def _patch_weeks(monkeypatch, completed):
    import dashboard_services.breakout_api as api
    import data_building.breakout_engine.weekly_store as store

    monkeypatch.setattr(api, "_resolve_bo_season", lambda season: 2026)
    monkeypatch.setattr(store, "list_completed_weeks", lambda season: completed)
    monkeypatch.setattr(api, "opportunity_data_ready", lambda season: True)
    return api


def test_week_list_offers_old_version_weeks_but_opens_on_current(monkeypatch):
    api = _patch_weeks(monkeypatch, [
        {"as_of_week": 1, "as_of_date": "2026-09-13", "scoring_version": OLD_VERSION},
        {"as_of_week": 2, "as_of_date": "2026-09-20", "scoring_version": OLD_VERSION},
        {"as_of_week": 4, "as_of_date": "2026-10-04", "scoring_version": SCORING_VERSION},
    ])

    payload = api.list_breakout_weeks(2026)

    assert [w["value"] for w in payload["weeks"]] == ["preseason", 1, 2, 4]
    assert payload["weeks"][1]["scoring_version"] == OLD_VERSION
    assert payload["latest_week"] == 4


def test_week_list_with_only_old_version_weeks_opens_on_preseason(monkeypatch):
    api = _patch_weeks(monkeypatch, [
        {"as_of_week": 1, "as_of_date": "2026-09-13", "scoring_version": OLD_VERSION},
        {"as_of_week": 2, "as_of_date": "2026-09-20", "scoring_version": OLD_VERSION},
    ])

    payload = api.list_breakout_weeks(2026)

    assert [w["value"] for w in payload["weeks"]] == ["preseason", 1, 2]
    # Never open the default view on a stale-version snapshot.
    assert payload["latest_week"] == "preseason"
