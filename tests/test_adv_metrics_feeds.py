"""Unit tests for the nflverse-backed advanced-metrics feeds.

Covers the pure aggregation math (no pandas / nfl_data_py / network):
- _aggregate_air_yards_weekly: per-player and per-team air-yard/target totals
- _air_yards_rows_from_aggregates: per-game, share, and WOPR derivation
- _average_snap_pct: mean offense_pct per PFR id
- _merge_nflverse_snap_share: week/season filters, 0-1 fraction scale,
  never clobbers an existing snap value, degrades when nfl_data_py is missing
- import_air_yards: nflverse-first, legacy CSV fallback
"""

import sys
import types

import pytest

from data_building import advanced_metrics as am


# --- _aggregate_air_yards_weekly ---------------------------------------------

def _w(pid, team, week, targets, ay, pos="WR", stype="REG"):
    return {
        "player_id": pid, "recent_team": team, "week": week,
        "targets": targets, "receiving_air_yards": ay,
        "position": pos, "season_type": stype,
    }


def test_aggregate_air_yards_basic():
    rows = [
        _w("g1", "KC", 1, 8, 90),
        _w("g1", "KC", 2, 10, 110),
        _w("g2", "KC", 1, 5, 40),
        _w("g2", "KC", 2, 0, 0),  # no receiving involvement: skipped
        _w(None, "KC", 1, 3, 20),  # no player id: skipped
    ]
    players, teams = am._aggregate_air_yards_weekly(rows)
    assert players["g1"] == {"ay": 200.0, "targets": 18.0, "games": 2, "team": "KC"}
    assert players["g2"] == {"ay": 40.0, "targets": 5.0, "games": 1, "team": "KC"}
    assert teams["KC"] == {"ay": 240.0, "targets": 23.0}


def test_aggregate_air_yards_latest_team_wins():
    rows = [_w("g1", "KC", 1, 5, 50), _w("g1", "BUF", 2, 6, 60)]
    players, _ = am._aggregate_air_yards_weekly(rows)
    assert players["g1"]["team"] == "BUF"
    assert players["g1"]["games"] == 2


def test_aggregate_air_yards_empty():
    assert am._aggregate_air_yards_weekly([]) == ({}, {})
    assert am._aggregate_air_yards_weekly(None) == ({}, {})


# --- _air_yards_rows_from_aggregates ------------------------------------------

def test_air_yards_rows_math():
    players = {
        "g1": {"ay": 200.0, "targets": 20.0, "games": 2, "team": "KC"},
        "g9": {"ay": 50.0, "targets": 5.0, "games": 2, "team": "KC"},  # not in crosswalk
    }
    teams = {"KC": {"ay": 1000.0, "targets": 100.0}}
    rows = am._air_yards_rows_from_aggregates(players, teams, {"g1": "1234"})
    assert set(rows) == {"1234"}
    r = rows["1234"]
    assert r["air_yards_per_game"] == 100.0
    assert r["air_yards_share"] == 20.0
    # WOPR = 1.5 * target_share + 0.7 * air_yards_share (fraction form)
    assert r["wopr"] == pytest.approx(1.5 * 0.2 + 0.7 * 0.2)


def test_air_yards_rows_zero_team_totals_no_crash():
    players = {"g1": {"ay": 30.0, "targets": 4.0, "games": 1, "team": "KC"}}
    rows = am._air_yards_rows_from_aggregates(players, {}, {"g1": "7"})
    assert rows["7"]["air_yards_share"] == 0.0
    assert rows["7"]["wopr"] == 0.0
    assert rows["7"]["air_yards_per_game"] == 30.0


# --- _average_snap_pct ---------------------------------------------------------

def test_average_snap_pct():
    rows = [
        {"pfr_player_id": "p1", "offense_pct": 80.0},
        {"pfr_player_id": "p1", "offense_pct": 60.0},
        {"pfr_player_id": "p2", "offense_pct": 100.0},
        {"pfr_player_id": "", "offense_pct": 50.0},
        {"pfr_player_id": None, "offense_pct": 50.0},
    ]
    assert am._average_snap_pct(rows) == {"p1": 70.0, "p2": 100.0}


# --- _merge_nflverse_snap_share -------------------------------------------------

class _FakeSnaps:
    def __init__(self, rows):
        self._rows = rows
        self.empty = not rows

    def to_dict(self, orient):
        assert orient == "records"
        return list(self._rows)


def _fake_nfl(monkeypatch, snaps_rows=None, rosters_rows=None):
    mod = types.ModuleType("nfl_data_py")
    mod.import_snap_counts = lambda seasons: _FakeSnaps(snaps_rows or [])

    class _FakeRosters:
        columns = ["pfr_id", "sleeper_id"]

        def iterrows(self):
            for i, r in enumerate(rosters_rows or []):
                yield i, r

    mod.import_rosters = lambda seasons: _FakeRosters()
    mod.import_ids = lambda: _FakeRosters()  # not reached when rosters hit
    monkeypatch.setitem(sys.modules, "nfl_data_py", mod)
    return mod


def test_merge_snap_share_fills_fraction(monkeypatch):
    snaps = [
        {"pfr_player_id": "p1", "season_type": "REG", "week": 1, "offense_pct": 80.0},
        {"pfr_player_id": "p1", "season_type": "REG", "week": 2, "offense_pct": 60.0},
    ]
    rosters = [{"pfr_id": "p1", "sleeper_id": 1234.0}]
    _fake_nfl(monkeypatch, snaps, rosters)
    usage_map = {"1234": {"games": 2}}
    merged = am._merge_nflverse_snap_share(usage_map, 2026, 2)
    assert merged == 1
    # 0-1 fraction scale, matching the pct_frac snap_share spec
    assert usage_map["1234"]["avg_off_snap_pct"] == pytest.approx(0.7)


def test_merge_snap_share_filters_and_never_clobbers(monkeypatch):
    snaps = [
        {"pfr_player_id": "p1", "season_type": "REG", "week": 3, "offense_pct": 100.0},  # after cutoff
        {"pfr_player_id": "p1", "season_type": "PRE", "week": 1, "offense_pct": 100.0},  # preseason
        {"pfr_player_id": "p2", "season_type": "REG", "week": 1, "offense_pct": 50.0},  # unknown pfr id
    ]
    rosters = [{"pfr_id": "p1", "sleeper_id": "1234"}]
    _fake_nfl(monkeypatch, snaps, rosters)
    usage_map = {"1234": {"games": 2, "avg_off_snap_pct": 0.9}}  # existing real value
    merged = am._merge_nflverse_snap_share(usage_map, 2026, 2)
    assert merged == 0
    assert usage_map["1234"]["avg_off_snap_pct"] == 0.9


def test_merge_snap_share_no_nfl_data_py(monkeypatch):
    monkeypatch.setitem(sys.modules, "nfl_data_py", None)  # import -> ImportError
    usage_map = {"1234": {"games": 2}}
    assert am._merge_nflverse_snap_share(usage_map, 2026, 2) == 0
    assert "avg_off_snap_pct" not in usage_map["1234"]


def test_merge_snap_share_empty_map():
    assert am._merge_nflverse_snap_share({}, 2026, 2) == 0


# --- import_air_yards ------------------------------------------------------------

def test_import_air_yards_prefers_nflverse(monkeypatch):
    seen = {}

    def fake_fetch(season, completed_week=None):
        seen["fetch"] = (season, completed_week)
        return {"1234": {"air_yards_per_game": 50.0, "air_yards_share": 20.0, "wopr": 0.44}}

    def fake_upsert(season, rows):
        seen["upsert"] = (season, rows)
        return 1

    def fake_csv(season):
        raise AssertionError("CSV fallback must not run when nflverse succeeds")

    monkeypatch.setattr(am, "fetch_nflverse_air_yards", fake_fetch)
    monkeypatch.setattr(am, "_upsert_air_yards_wopr", fake_upsert)
    monkeypatch.setattr(am, "import_air_yards_from_stats_csv", fake_csv)
    assert am.import_air_yards(2026, 2) == 1
    assert seen["fetch"] == (2026, 2)
    assert seen["upsert"][0] == 2026


def test_import_air_yards_falls_back_to_csv(monkeypatch):
    calls = []

    def fake_csv(season):
        calls.append(season)
        return 7

    monkeypatch.setattr(am, "fetch_nflverse_air_yards", lambda s, cw=None: {})
    monkeypatch.setattr(am, "import_air_yards_from_stats_csv", fake_csv)
    assert am.import_air_yards(2026, 2) == 7
    assert calls == [2026]
