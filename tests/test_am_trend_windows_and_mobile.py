"""Regression tests for the 2026-09-30 Advanced Metrics mobile/trend report.

Kaedon reported three symptoms on her phone:

1. The moved + Metric button and compare bar were not visible. Kaedon
   corrected the platform: the miss was on desktop, where both controls
   still lived inside the hidden filter sheet. They now render
   server-side in the positions-row hosts at every width (the + Metric
   wrap at the right end of the row, the compare bar directly under it),
   and the old JS relocation is gone entirely.
2. Usage Trend (opportunity_trend) and xFP Trend showed 0.00% for every
   player. Two stacked causes: through week 3 the last-3 window IS the
   whole sample, so the old helpers returned None and the leaderboard's
   DISTINCT ON latest-non-null pick resurrected pre-fix rows storing a
   forced 0.0. The helpers now compare the latest week against the prior
   weeks early season, and trend leaderboards only read each player's
   newest snapshot row.
3. The Usage L3W column was a sparkline plus a dash, with every real
   number hidden in a hover-only title tooltip (invisible on touch). The
   usage-trends payload now carries baseline_avg and the cell renders the
   recent vs baseline numbers directly.
"""

import pytest

import data_building.advanced_metrics as am
import data_building.weekly_metrics as wm
from dashboard_services.pages.advanced_metrics_page import _AM_JS

# --------------------------------------------------------------------------- #
# Usage-trends payload: early-season delta + baseline
# --------------------------------------------------------------------------- #

class _TrendsConn:
    def __init__(self, rows):
        self._rows = rows

    def execute(self, sql, params=None):
        return self

    def fetchall(self):
        return list(self._rows)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def _weekly_row(pid, week, pos, snap, targets, touches):
    return {
        "player_id": pid, "week": week, "position": pos,
        "snap_pct": snap, "targets": targets, "touches": touches,
        "target_share": None,
    }


@pytest.fixture()
def trends(monkeypatch):
    rows = [
        # RB, 3 weeks: touches 10, 12, 20; snaps 50, 60, 80.
        _weekly_row("rb1", 1, "RB", 50.0, 2, 10),
        _weekly_row("rb1", 2, "RB", 60.0, 3, 12),
        _weekly_row("rb1", 3, "RB", 80.0, 4, 20),
        # WR, 5 weeks: targets 5, 5, 5, 5, 15.
        _weekly_row("wr1", 1, "WR", 70.0, 5, 5),
        _weekly_row("wr1", 2, "WR", 72.0, 5, 5),
        _weekly_row("wr1", 3, "WR", 71.0, 5, 5),
        _weekly_row("wr1", 4, "WR", 73.0, 5, 5),
        _weekly_row("wr1", 5, "WR", 90.0, 15, 15),
        # One-week player: excluded entirely (no comparison exists).
        _weekly_row("te1", 3, "TE", 40.0, 6, 6),
    ]
    conn = _TrendsConn(rows)
    monkeypatch.setattr(wm, "get_conn", lambda: conn)
    monkeypatch.setattr(wm, "init_weekly_metrics_db", lambda: None)
    return wm._compute_usage_trends(2026)


def test_usage_trends_early_window_compares_last_week_vs_prior(trends):
    rb = trends["rb1"]
    # touches: last week 20 vs prior avg 11 -> +9.0 (was None under the
    # degenerate-window rule, which rendered as a bare dash).
    assert rb["delta"] == 9.0
    assert rb["recent_avg"] == 20.0
    assert rb["baseline_avg"] == 11.0
    assert rb["season_avg"] == 14.0
    # snap %: last week 80 vs prior avg 55 -> +25.0 (was a forced 0.0).
    assert rb["snap_delta"] == 25.0
    assert rb["series"] == [10.0, 12.0, 20.0]


def test_usage_trends_full_window_uses_last3_vs_season(trends):
    wr = trends["wr1"]
    # targets: last-3 avg 8.3 vs season avg 7.0 -> +1.3.
    assert wr["delta"] == 1.3
    assert wr["recent_avg"] == 8.3
    assert wr["baseline_avg"] == 7.0


def test_usage_trends_single_week_player_excluded(trends):
    assert "te1" not in trends


def test_recent_momentum_early_window(monkeypatch):
    series = [
        {"snap_pct": 50.0}, {"snap_pct": 60.0}, {"snap_pct": 80.0},
    ]
    monkeypatch.setattr(wm, "get_player_weekly_series", lambda pid, season: series)
    # 3 weeks: last week 80 vs prior avg 55 -> +25.0 (was a forced 0.0).
    assert wm.get_recent_momentum("rb1", 2026) == 25.0
    monkeypatch.setattr(wm, "get_player_weekly_series", lambda pid, season: series[:1])
    assert wm.get_recent_momentum("rb1", 2026) is None


# --------------------------------------------------------------------------- #
# Trend leaderboards read only the newest snapshot row
# --------------------------------------------------------------------------- #

class _CaptureConn:
    """Serves the leaderboard's pre-check queries and captures the main
    DISTINCT ON query instead of executing it."""

    def __init__(self):
        self.main_queries = []

    def execute(self, sql, params=None):
        self._sql = sql
        if "information_schema" in sql:
            self._rows = [{"column_name": c} for c in (
                "player_id", "position", "season", "as_of_date", "games",
                "snap_share", "xfp_trend", "opportunity_trend",
                "total_targets", "total_receptions", "total_carries",
                "total_pass_att",
            )]
        elif "DISTINCT ON" in sql:
            self.main_queries.append(sql)
            self._rows = []
        else:
            self._rows = [{"present": 1}]
        return self

    def fetchall(self):
        return list(self._rows)

    def fetchone(self):
        return self._rows[0] if self._rows else None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


@pytest.fixture()
def capture_conn(monkeypatch):
    conn = _CaptureConn()
    monkeypatch.setattr(am, "get_conn", lambda: conn)
    am._METRIC_LEADERBOARD_CACHE.clear()
    yield conn
    am._METRIC_LEADERBOARD_CACHE.clear()


@pytest.mark.parametrize("metric", ["xfp_trend", "opportunity_trend"])
def test_trend_leaderboard_restricted_to_newest_snapshot_row(capture_conn, metric):
    am.get_metric_leaderboard(metric, position="RB", season=2026)
    assert len(capture_conn.main_queries) == 1
    sql = capture_conn.main_queries[0]
    # The stale-zero resurrection guard: candidates must sit on the
    # player's max as_of_date for the season.
    assert "MAX(cx.as_of_date)" in sql
    assert "cx.player_id = m.player_id" in sql


def test_non_trend_leaderboard_has_no_newest_row_restriction(capture_conn):
    am.get_metric_leaderboard("catch_rate", position="WR", season=2026)
    assert len(capture_conn.main_queries) == 1
    assert "MAX(cx.as_of_date)" not in capture_conn.main_queries[0]


# --------------------------------------------------------------------------- #
# Page JS: visible trend numbers + metric-control placement
# --------------------------------------------------------------------------- #

def test_trend_cell_renders_numbers_not_hover_only():
    # The recent vs baseline averages render in the cell body, and the
    # payload field they read is baseline_avg.
    assert "am-trend-nums" in _AM_JS
    assert "t.baseline_avg" in _AM_JS
    assert "last wk" in _AM_JS
    assert "no trend yet" in _AM_JS
    # The small-delta flat case shows its signed value instead of a dash.
    assert "d.toFixed(1)" in _AM_JS


def test_metric_controls_need_no_relocation():
    # The wrap and bar are rendered into the positions-row hosts
    # server-side, so there is no relocation pass and no sheet Metrics
    # section left to hide or strand the controls in.
    assert "amRelocateMetricControls" not in _AM_JS
    assert "amMetricsSec" not in _AM_JS
    # updateCompareBar drives the host + desktop grid compare row off the
    # same visibility state as the bar itself.
    assert "am-compare-on" in _AM_JS
    assert "am-has-compare" in _AM_JS
