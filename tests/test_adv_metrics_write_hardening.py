"""Hardening for the Advanced Metrics write paths.

Regression coverage for the class of bug where a preset's games-played (or
sample counts) silently lag behind other presets because a writer failed to
feed a column family:

* ``upsert_season`` / ``upsert_weekly_season`` must feed the xFP columns even
  when the caller forgets the xFP map (the Key Metrics GP freeze: the daily
  cron's season step called ``upsert_season`` without ``xfp_by_pid``).
* ``save_metrics_snapshot`` must not let a partial build shrink the
  season-cumulative counters (games, volume totals) that every snapshot
  preset renders in its G / carries / targets / receptions / attempts
  columns.
"""

import pytest

import scripts.sync_nflverse_metrics as sync


# --------------------------------------------------------------------------- #
# Fake DB plumbing
# --------------------------------------------------------------------------- #
class _FakeConn:
    """Stands in for a psycopg connection: records INSERT params, serves
    canned rows for SELECTs."""

    def __init__(self, select_rows=()):
        self.executes = []  # (sql, params)
        self._select_rows = list(select_rows)

    def execute(self, sql, params=None):
        self.executes.append((sql, params))
        return self

    def fetchall(self):
        return list(self._select_rows)

    def fetchone(self):
        return self._select_rows[0] if self._select_rows else None

    @property
    def rowcount(self):
        return 0

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def _patch_sync_db(monkeypatch, conn):
    monkeypatch.setattr(sync, "get_conn", lambda: conn)
    monkeypatch.setattr(sync, "init_advanced_metrics_db", lambda: None)
    monkeypatch.setattr(sync, "init_weekly_advanced_metrics_db", lambda: None)


# --------------------------------------------------------------------------- #
# upsert_season: xFP fallback
# --------------------------------------------------------------------------- #
def test_upsert_season_builds_xfp_when_caller_omits_it(monkeypatch):
    # The daily cron called upsert_season without xfp_by_pid: the snapshot xFP
    # columns froze at the last manual sync. The writer must fill the gap.
    conn = _FakeConn()
    _patch_sync_db(monkeypatch, conn)
    monkeypatch.setattr(
        sync, "build_nflverse_metrics_for_season",
        lambda season: {"1": {"rushing_epa_per_att": 0.05}},
    )
    built = {}

    def fake_build(season):
        built["called"] = True
        return ({"1": {"expected_ppr": 30.0, "ppr_over_expected": 2.0}}, {})

    monkeypatch.setattr(sync, "build_expected_points_both", fake_build)

    n = sync.upsert_season(2026, {}, purge_pff=False)
    assert built.get("called") is True
    assert n == 1
    inserts = [p for sql, p in conn.executes if "INSERT INTO player_advanced_metrics" in sql]
    assert len(inserts) == 1
    assert 30.0 in inserts[0]  # the xFP value rode the same upsert


def test_upsert_season_respects_explicit_xfp_map(monkeypatch):
    conn = _FakeConn()
    _patch_sync_db(monkeypatch, conn)
    monkeypatch.setattr(
        sync, "build_nflverse_metrics_for_season",
        lambda season: {"1": {"rushing_epa_per_att": 0.05}},
    )

    def fake_build(season):  # pragma: no cover - must not run
        raise AssertionError("internal xFP build must not run when a map is passed")

    monkeypatch.setattr(sync, "build_expected_points_both", fake_build)
    n = sync.upsert_season(
        2026, {}, purge_pff=False, xfp_by_pid={"1": {"expected_ppr": 9.5}})
    assert n == 1
    inserts = [p for sql, p in conn.executes if "INSERT INTO player_advanced_metrics" in sql]
    assert 9.5 in inserts[0]


def test_upsert_season_survives_xfp_build_failure(monkeypatch):
    conn = _FakeConn()
    _patch_sync_db(monkeypatch, conn)
    monkeypatch.setattr(
        sync, "build_nflverse_metrics_for_season",
        lambda season: {"1": {"rushing_epa_per_att": 0.05}},
    )

    def fake_build(season):
        raise RuntimeError("nfl_data_py exploded")

    monkeypatch.setattr(sync, "build_expected_points_both", fake_build)
    # Must not raise, and must still write the nflverse row.
    assert sync.upsert_season(2026, {}, purge_pff=False) == 1


# --------------------------------------------------------------------------- #
# upsert_weekly_season: xFP fallback
# --------------------------------------------------------------------------- #
def test_upsert_weekly_season_builds_xfp_when_caller_omits_it(monkeypatch):
    conn = _FakeConn()
    _patch_sync_db(monkeypatch, conn)
    monkeypatch.setattr(
        sync, "build_nflverse_weekly_metrics_for_season",
        lambda season: {("1", 3): {"w_carries": 12}},
    )
    built = {}

    def fake_build(season):
        built["called"] = True
        return ({}, {("1", 3): {"expected_ppr": 14.5, "ppr_over_expected": 2.0}})

    monkeypatch.setattr(sync, "build_expected_points_both", fake_build)

    n = sync.upsert_weekly_season(2026, {})
    assert built.get("called") is True
    assert n == 1
    inserts = [p for sql, p in conn.executes
               if "INSERT INTO player_weekly_advanced_metrics" in sql]
    assert len(inserts) == 1
    assert 14.5 in inserts[0]


def test_upsert_weekly_season_respects_explicit_empty_map(monkeypatch):
    # {} means "the xFP build already failed upstream": do not rebuild, and
    # the COALESCE merge must preserve whatever xFP is already stored.
    conn = _FakeConn()
    _patch_sync_db(monkeypatch, conn)
    monkeypatch.setattr(
        sync, "build_nflverse_weekly_metrics_for_season",
        lambda season: {("1", 3): {"w_carries": 12}},
    )

    def fake_build(season):  # pragma: no cover - must not run
        raise AssertionError("internal xFP build must not run for explicit {}")

    monkeypatch.setattr(sync, "build_expected_points_both", fake_build)
    assert sync.upsert_weekly_season(2026, {}, xfp_by_pw={}) == 1


# --------------------------------------------------------------------------- #
# save_metrics_snapshot: monotonic counters
# --------------------------------------------------------------------------- #
def _snapshot_conn(prev_rows):
    """Fake conn: serves prev_rows for the guard query, records INSERTs."""

    class _Conn(_FakeConn):
        def execute(self, sql, params=None):
            if "DISTINCT ON (player_id)" in sql:
                self.executes.append((sql, params))
                return self
            if sql.strip().startswith("SELECT 1"):
                self.executes.append((sql, params))
                return _FakeConn([])  # not existed -> insert path
            self.executes.append((sql, params))
            return _FakeConn([])

    return _Conn(prev_rows)


def _insert_params(conn):
    return [p for sql, p in conn.executes
            if "INSERT INTO player_advanced_metrics" in sql]


def test_snapshot_clamps_shrunken_games_and_totals(monkeypatch):
    import data_building.advanced_metrics as am

    prev = [{
        "player_id": "1", "games": 3, "total_targets": 20,
        "total_receptions": 14, "total_carries": 45, "total_touches": 59,
        "total_pass_att": None,
    }]
    conn = _snapshot_conn(prev)
    monkeypatch.setattr(am, "get_conn", lambda: conn)
    monkeypatch.setattr(am, "init_advanced_metrics_db", lambda: None)

    # Partial build: only 2 games made it into this run's usage map.
    am.save_metrics_snapshot([{
        "player_id": "1", "position": "RB", "games": 2,
        "total_targets": 12, "total_receptions": 8,
        "total_carries": 30, "total_touches": 38, "total_pass_att": None,
    }], "2026-09-28", season=2026)

    params = _insert_params(conn)
    assert len(params) == 1
    vals = params[0]
    # games is the 28th value, totals follow (see INSERT column order).
    assert vals[27] == 3          # games clamped to previous 3
    assert vals[28] == 20         # total_targets clamped
    assert vals[29] == 14         # total_receptions clamped
    assert vals[30] == 45         # total_carries clamped
    assert vals[31] == 59         # total_touches clamped
    assert vals[32] is None       # None/None stays None


def test_snapshot_keeps_growth_and_first_write(monkeypatch):
    import data_building.advanced_metrics as am

    prev = [{
        "player_id": "1", "games": 2, "total_targets": 12,
        "total_receptions": 8, "total_carries": 30, "total_touches": 38,
        "total_pass_att": None,
    }]
    conn = _snapshot_conn(prev)
    monkeypatch.setattr(am, "get_conn", lambda: conn)
    monkeypatch.setattr(am, "init_advanced_metrics_db", lambda: None)

    am.save_metrics_snapshot([{
        # Full build with a new week: larger values must pass through, and a
        # brand-new player (no prev row) writes as-is.
        "player_id": "1", "position": "RB", "games": 3,
        "total_targets": 20, "total_receptions": 14,
        "total_carries": 45, "total_touches": 59, "total_pass_att": None,
    }, {
        "player_id": "2", "position": "WR", "games": 1,
        "total_targets": 5, "total_receptions": 3,
        "total_carries": 0, "total_touches": 3, "total_pass_att": None,
    }], "2026-09-28", season=2026)

    params = _insert_params(conn)
    assert len(params) == 2
    assert params[0][27] == 3 and params[0][30] == 45
    assert params[1][27] == 1 and params[1][28] == 5


def test_clamp_helper_none_semantics():
    from data_building.advanced_metrics import _clamp_monotonic_counters

    # No prev row: untouched.
    m = {"games": 2}
    _clamp_monotonic_counters(m, None)
    assert m == {"games": 2}

    # New None keeps the old value instead of NULLing it out.
    m = {"games": None, "total_carries": 10}
    _clamp_monotonic_counters(
        m, {"games": 3, "total_carries": 4, "total_targets": None})
    assert m["games"] == 3
    assert m["total_carries"] == 10
    assert m.get("total_targets") is None
