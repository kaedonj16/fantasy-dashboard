"""Season breakout in-season role trajectory: weekly data path regressions.

The trajectory component used to read player_advanced_metrics snapshots
through a helper that queried a nonexistent `date` column; the error was
silently swallowed, so every in-season trajectory scored the neutral 50
no matter how a player's role was actually trending. These tests pin the
fixed behavior:

- trajectory windows come from per-week usage rows (snap share,
  opportunity share, red-zone share), not season snapshots;
- snapshot role_score still feeds the role sub-signal when a pair exists;
- a genuinely empty data set still returns the explicit neutral score;
- snapshot query errors surface instead of being swallowed.
"""
from datetime import date
from pathlib import Path

import pytest

components = pytest.importorskip("data_building.breakout_engine.components")
db_helpers = pytest.importorskip("data_building.breakout_engine.db_helpers")
weekly_metrics = pytest.importorskip("data_building.weekly_metrics")
team_history = pytest.importorskip(
    "data_building.external_data.player_team_history")

RISING = "1001"
TEAMMATE = "2002"
AS_OF = date(2026, 10, 15)


def _week(week, *, snap_pct, snaps, targets, carries, rz_targets, rz_carries):
    return {
        "week": week, "position": "RB", "snap_pct": snap_pct, "snaps": snaps,
        "team_snaps": 65, "targets": targets, "receptions": 0,
        "carries": carries, "touches": targets + carries,
        "target_share": 0.0, "ppr_pts": 0.0, "rec_yards": 0, "rush_yards": 0,
        "pass_att": 0, "rec_tds": 0, "rush_tds": 0, "pass_tds": 0,
        "rz_targets": rz_targets, "rz_carries": rz_carries,
    }


def _rising_series():
    """Six weeks of clearly rising usage for RISING; a flat teammate keeps
    the team totals honest so shares stay below 1."""
    player = [
        _week(1, snap_pct=40.0, snaps=26, targets=2, carries=4,
              rz_targets=0, rz_carries=0),
        _week(2, snap_pct=45.0, snaps=29, targets=3, carries=5,
              rz_targets=0, rz_carries=1),
        _week(3, snap_pct=55.0, snaps=36, targets=4, carries=8,
              rz_targets=1, rz_carries=0),
        _week(4, snap_pct=60.0, snaps=39, targets=5, carries=10,
              rz_targets=1, rz_carries=1),
        _week(5, snap_pct=70.0, snaps=46, targets=6, carries=14,
              rz_targets=1, rz_carries=2),
        _week(6, snap_pct=78.0, snaps=51, targets=8, carries=16,
              rz_targets=2, rz_carries=2),
    ]
    teammate = [
        _week(w, snap_pct=60.0, snaps=39, targets=6, carries=14,
              rz_targets=2, rz_carries=2)
        for w in range(1, 7)
    ]
    return {RISING: player, TEAMMATE: teammate}


def _clear_series_cache():
    clear = getattr(db_helpers, "clear_weekly_trajectory_cache", None)
    if clear is not None:
        clear()


@pytest.fixture(autouse=True)
def _clean_cache():
    _clear_series_cache()
    yield
    _clear_series_cache()


@pytest.fixture
def install_weekly(monkeypatch):
    """Patch the weekly data path at its source modules plus the snapshot
    helper and momentum reader the component consults."""
    def _install(series, snapshots=None):
        monkeypatch.setattr(
            weekly_metrics, "get_weekly_series_by_player",
            lambda season, through_week: series)
        monkeypatch.setattr(
            team_history, "team_for_week",
            lambda pid, season, week: "KC")
        if snapshots is None:
            monkeypatch.setattr(
                components, "get_player_advanced_metrics",
                lambda *args, **kwargs: None)
        else:
            monkeypatch.setattr(
                components, "get_player_advanced_metrics", snapshots)
        monkeypatch.setattr(
            weekly_metrics, "get_recent_momentum",
            lambda pid, season: None)
        _clear_series_cache()
    return _install


def test_rising_weekly_usage_scores_above_neutral(install_weekly):
    install_weekly(_rising_series())

    score, details = components._inseason_role_trajectory_score(
        RISING, AS_OF, 14)

    assert "note" not in details
    assert score != 50.0
    assert score > 60.0
    assert details["snap_delta"] > 0.1
    assert details["opp_delta"] > 0.1
    assert details["rz_delta"] > 0.1
    assert details["curr_snap_share"] == pytest.approx(0.74)
    assert details["prev_snap_share"] == pytest.approx(0.575)
    # No snapshot pair in this fixture: the role sub-signal sits at its
    # no-change point and says so, instead of zeroing the component.
    assert details["role_score_available"] is False
    assert details["role_delta"] == 0.0


def test_weekly_windows_use_team_totals_and_fraction_scales(install_weekly):
    install_weekly(_rising_series())

    windows = db_helpers.get_player_weekly_windows(RISING, 2026, 14)

    assert windows is not None
    current, previous = windows["current"], windows["previous"]
    # Weeks 5-6 vs weeks 3-4 (14 days -> 2-week windows).
    assert current["weeks"] == 2
    assert previous["weeks"] == 2
    # Opportunity share: (6+14 + 8+16) opps over team totals 40 and 44.
    assert current["opportunity_share"] == pytest.approx((20 / 40 + 24 / 44) / 2)
    assert previous["opportunity_share"] == pytest.approx((12 / 32 + 15 / 35) / 2)
    assert current["snap_share"] == pytest.approx(0.74)
    assert current["sample_size"] == 44
    assert previous["sample_size"] == 27


def test_role_score_delta_uses_snapshot_pair_when_present(install_weekly):
    def _snapshots(player_id, as_of_date, lookback_days):
        if as_of_date == AS_OF:
            return {"role_score": 60.0}
        return {"role_score": 40.0}

    install_weekly(_rising_series(), snapshots=_snapshots)

    score, details = components._inseason_role_trajectory_score(
        RISING, AS_OF, 14)

    assert details["role_score_available"] is True
    assert details["role_delta"] == 20.0
    assert details["curr_role_score_metric"] == 60.0
    assert details["role_score_component"] > 10.0


def test_no_weekly_data_returns_neutral_with_note(install_weekly):
    install_weekly({})

    score, details = components._inseason_role_trajectory_score(
        RISING, AS_OF, 14)

    assert score == 50.0
    assert details["note"] == "Insufficient data, neutral score"


def test_single_week_of_data_is_not_a_trajectory(install_weekly):
    install_weekly({RISING: _rising_series()[RISING][:1]})

    score, details = components._inseason_role_trajectory_score(
        RISING, AS_OF, 14)

    assert score == 50.0
    assert details["note"] == "Insufficient data, neutral score"


def test_snapshot_helper_queries_as_of_date_column():
    src = Path(db_helpers.__file__).read_text()
    assert "as_of_date >= %s" in src
    assert "as_of_date <= %s" in src
    assert "as_of_date = %s" in src
    assert "AND date >=" not in src
    assert "AND date =" not in src


def test_snapshot_query_error_is_not_swallowed(monkeypatch):
    def _boom():
        raise RuntimeError("db exploded")

    monkeypatch.setattr(db_helpers, "get_conn", _boom)

    with pytest.raises(RuntimeError):
        db_helpers.get_player_advanced_metrics(RISING, AS_OF, 14)
    with pytest.raises(RuntimeError):
        db_helpers.get_player_advanced_metrics(RISING, AS_OF, 0)
