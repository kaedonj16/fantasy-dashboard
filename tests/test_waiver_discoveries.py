"""Tests for the pure, DB-free transforms in dashboard_services.waiver_discoveries
(row -> GameContext, historical baseline). The DB I/O paths are exercised in
production; here we lock the logic that turns raw weekly rows into detector input."""
import pytest

from dashboard_services.waiver_discoveries import (
    game_context_from_rows,
    historical_baseline,
)
from utils.waiver_big_game import assess_big_game


def test_historical_baseline_trailing_and_season():
    # <4 games -> season_avg over what exists.
    b = historical_baseline([{"ppr_pts": 6.0}, {"ppr_pts": 10.0}])
    assert b["games"] == 2
    assert b["source"] == "season_avg"
    assert b["ppg"] == pytest.approx(8.0)
    # >=4 games -> trailing4 window.
    b4 = historical_baseline([{"ppr_pts": 2}, {"ppr_pts": 4}, {"ppr_pts": 6},
                              {"ppr_pts": 8}, {"ppr_pts": 20}])
    assert b4["source"] == "trailing4"
    assert b4["ppg"] == pytest.approx((4 + 6 + 8 + 20) / 4)
    # No history.
    assert historical_baseline([])["ppg"] is None


def test_game_context_converts_percent_usage_to_fraction():
    week_row = {"week": 6, "snap_pct": 82.0, "targets": 9, "target_share": 26.0,
                "touches": 9, "carries": 0, "ppr_pts": 18.0,
                "rec_yards": 95, "rush_yards": 0}
    prior = [{"week": 4, "snap_pct": 40.0, "targets": 3, "target_share": 10.0,
              "touches": 3, "carries": 0, "ppr_pts": 6.0},
             {"week": 5, "snap_pct": 50.0, "targets": 4, "target_share": 12.0,
              "touches": 4, "carries": 0, "ppr_pts": 7.0}]
    g = game_context_from_rows("p", "WR", 2025, 6, week_row, prior)
    # 0-100 stored -> 0-1 in the context.
    assert g.snap_share == pytest.approx(0.82)
    assert g.snap_share_prev == pytest.approx(0.45)      # mean(40,50)/100
    assert g.target_share == pytest.approx(0.26)
    assert g.targets == 9
    assert g.targets_prev == pytest.approx(3.5)
    assert g.actual_points == 18.0
    assert g.baseline_source == "season_avg"
    assert g.total_yards == 95.0


def test_game_context_prefers_pregame_snapshot():
    week_row = {"week": 5, "snap_pct": 70, "targets": 8, "ppr_pts": 22.0,
                "touches": 8, "target_share": 22, "rec_yards": 80, "rush_yards": 0}
    g = game_context_from_rows("p", "WR", 2025, 5, week_row, [],
                               pregame={"pts": 9.0, "source": "sleeper",
                                        "saved_at": "2025-10-01T12:00:00Z"})
    assert g.pregame_projection == 9.0
    assert g.projection_saved_at == "2025-10-01T12:00:00Z"
    a = assess_big_game(g)
    assert a.expectation_basis == "pregame_projection"
    assert a.expectation == 9.0


def test_league_points_override_used_for_custom_scoring():
    week_row = {"week": 5, "ppr_pts": 22.0, "snap_pct": 70, "targets": 8,
                "touches": 8, "target_share": 22, "rec_yards": 80, "rush_yards": 0}
    g = game_context_from_rows("p", "WR", 2025, 5, week_row, [], league_points=27.5)
    assert g.actual_points == 27.5     # league-scored total wins over PPR proxy


def test_thin_history_flagged_limited():
    week_row = {"week": 2, "ppr_pts": 20, "snap_pct": 60, "targets": 6, "touches": 6}
    g = game_context_from_rows("rook", "WR", 2025, 2, week_row,
                               [{"week": 1, "ppr_pts": 5}])
    assert g.limited_history is True    # only 1 prior game


def test_extra_features_enable_td_caution():
    week_row = {"week": 7, "ppr_pts": 20, "snap_pct": 30, "targets": 0,
                "touches": 4, "carries": 4, "target_share": 0,
                "rec_yards": 0, "rush_yards": 90}
    g = game_context_from_rows("p", "RB", 2025, 7, week_row, [],
                               extra_features={"touchdowns": 2, "longest_play_yards": 70})
    a = assess_big_game(g)
    assert "td_dependent" in a.cautions
