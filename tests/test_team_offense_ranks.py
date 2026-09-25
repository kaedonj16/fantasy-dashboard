"""Tests for utils/team_offense_ranks.py: honest per-team offense table.

Covers the audit fixes: real PPG (never the TDs*6 + yards/20 proxy), real
per-game denominators (never season totals mislabeled), competition ranking
with ties, zeroes ranked, and N/A only for truly-missing values.
"""

from __future__ import annotations

import csv
import os

import pytest

from utils.team_offense_ranks import (
    PROJECTION_DIVISOR,
    aggregate_completed_games,
    aggregate_sleeper_team_weeks,
    aggregate_sleeper_team_weeks_for_teams,
    canon_team,
    competition_ranks,
    compute_team_offense,
    fully_completed_weeks,
    rank_offense_table,
    read_csv_team_totals,
)


def _game(season, week, home, away, home_score, away_score):
    return {
        "season": season,
        "week": week,
        "home_team": home,
        "away_team": away,
        "home_score": home_score,
        "away_score": away_score,
    }


def _two_week_rows():
    # KC beats BUF 30-20 in week 1; both idle... no, keep it simple: two
    # completed weeks, KC plays both, BUF has a week-2 bye (fewer games).
    return [
        _game(2026, 1, "KC", "BUF", "30", "20"),
        _game(2026, 1, "PHI", "DAL", "24", "21"),
        _game(2026, 2, "KC", "PHI", "27", "24"),
        _game(2026, 2, "DAL", "NYG", "17", "14"),
    ]


def _week_teams(week):
    rows = {
        1: {
            "KC": {"pass_yds": 300, "pass_att": 35, "rush_yds": 120, "rush_att": 25,
                   "pass_tds": 3, "rush_tds": 1},
            "BUF": {"pass_yds": 250, "pass_att": 30, "rush_yds": 100, "rush_att": 20,
                    "pass_tds": 2, "rush_tds": 0},
            "PHI": {"pass_yds": 280, "pass_att": 32, "rush_yds": 110, "rush_att": 28,
                    "pass_tds": 2, "rush_tds": 1},
            "DAL": {"pass_yds": 200, "pass_att": 28, "rush_yds": 90, "rush_att": 22,
                    "pass_tds": 1, "rush_tds": 1},
        },
        2: {
            "KC": {"pass_yds": 320, "pass_att": 38, "rush_yds": 100, "rush_att": 22,
                   "pass_tds": 2, "rush_tds": 1},
            "PHI": {"pass_yds": 260, "pass_att": 30, "rush_yds": 130, "rush_att": 25,
                    "pass_tds": 3, "rush_tds": 0},
            "DAL": {"pass_yds": 220, "pass_att": 30, "rush_yds": 80, "rush_att": 20,
                    "pass_tds": 1, "rush_tds": 0},
            "NYG": {"pass_yds": 180, "pass_att": 25, "rush_yds": 70, "rush_att": 18,
                    "pass_tds": 1, "rush_tds": 0},
        },
        3: {
            # Thursday final: only KC and LV have played week 3.
            "KC": {"pass_yds": 340, "pass_att": 40, "rush_yds": 90, "rush_att": 20,
                   "pass_tds": 3, "rush_tds": 0},
            "LV": {"pass_yds": 210, "pass_att": 30, "rush_yds": 80, "rush_att": 18,
                   "pass_tds": 1, "rush_tds": 0},
        },
    }
    return rows.get(week, {})


def test_canon_team_aliases():
    assert canon_team("WSH") == "WAS"
    assert canon_team("JAC") == "JAX"
    assert canon_team("LA") == "LAR"
    assert canon_team("kc") == "KC"


def test_aggregate_completed_games_only_counts_scored_reg_games():
    rows = _two_week_rows() + [
        _game(2026, 3, "KC", "LV", "", ""),          # future, no scores
        _game(2026, 19, "KC", "BUF", "30", "20"),    # postseason, excluded
        _game(2025, 1, "KC", "BUF", "99", "99"),     # wrong season
    ]
    out = aggregate_completed_games(2026, rows)
    assert out["KC"]["points"] == 57.0
    assert out["KC"]["games"] == 2
    assert out["BUF"]["points"] == 20.0
    assert out["BUF"]["games"] == 1
    assert "LV" not in out


def test_fully_completed_weeks_excludes_partial_week():
    rows = _two_week_rows() + [_game(2026, 3, "KC", "LV", "", "")]
    assert fully_completed_weeks(2026, rows) == [1, 2]


def test_competition_ranks_ties_share_rank_and_skip():
    ranks = competition_ranks({"A": 30.0, "B": 27.0, "C": 27.0, "D": 20.0})
    assert ranks["A"]["rank"] == 1
    assert ranks["B"]["rank"] == 2
    assert ranks["C"]["rank"] == 2
    assert ranks["D"]["rank"] == 4
    assert all(r["total"] == 4 for r in ranks.values())


def test_competition_ranks_zeroes_ranked_none_unranked():
    ranks = competition_ranks({"A": 5.0, "B": 0.0, "C": 0.0, "D": None})
    assert ranks["B"]["rank"] == 2
    assert ranks["C"]["rank"] == 2
    assert ranks["D"] is None
    assert ranks["A"]["total"] == 3  # unranked team excluded from total


def test_aggregate_sleeper_team_weeks_sums_team_rows():
    totals = aggregate_sleeper_team_weeks(_week_teams, [1, 2])
    assert totals["KC"]["pass_yds"] == 620.0
    assert totals["KC"]["rush_tds"] == 2.0
    assert totals["BUF"]["pass_yds"] == 250.0  # week 2 bye: only week 1 counts


def test_aggregate_sleeper_team_weeks_accepts_singular_stat_names():
    """Real Sleeper weekly TEAM rows use pass_yd / rush_td, not pass_yds."""
    def get_week(week):
        return {"KC": {"pass_yd": 300, "pass_att": 35, "rush_yd": 120,
                       "rush_att": 25, "pass_td": 3, "rush_td": 1}}
    totals = aggregate_sleeper_team_weeks(get_week, [1])
    assert totals["KC"]["pass_yds"] == 300.0
    assert totals["KC"]["rush_tds"] == 1.0


def test_read_csv_team_totals(tmp_path):
    path = os.path.join(str(tmp_path), "stats_player_reg_2025.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["recent_team", "passing_yards", "attempts", "rushing_yards",
                    "carries", "passing_tds", "rushing_tds"])
        w.writerow(["KC", "4000", "550", "1800", "400", "30", "15"])
        w.writerow(["KC", "500", "60", "200", "40", "4", "2"])
        w.writerow(["BUF", "3800", "520", "1700", "380", "28", "12"])
    totals = read_csv_team_totals(path)
    assert totals["KC"]["pass_yds"] == 4500.0
    assert totals["KC"]["rush_tds"] == 17.0
    assert totals["BUF"]["pass_att"] == 520.0
    assert read_csv_team_totals("/nonexistent.csv") == {}


def test_compute_actual_uses_real_ppg_not_proxy():
    """The old proxy was TDs*6 + yards/20 on season totals. Real PPG is
    points / actual games played."""
    table = compute_team_offense(
        2026,
        games_rows=_two_week_rows(),
        get_week_teams=_week_teams,
        plays_pg_map={"KC": 64.5, "BUF": 61.0, "PHI": 63.0, "DAL": 62.0, "NYG": 60.0},
    )
    assert table["data_mode"] == "actual"
    kc = table["teams"]["KC"]
    assert kc["games"] == 2
    assert kc["points_pg"] == 28.5  # (30 + 27) / 2, real NFL points
    # The proxy would have given (5 pass + 2 rush TDs)*6 + 840 yds/20 = 84.
    assert kc["points_pg"] != 84.0
    # Per-game denominators use each team's ACTUAL games (BUF had a bye).
    assert kc["pass_yds_pg"] == 310.0  # 620 / 2
    assert table["teams"]["BUF"]["pass_yds_pg"] == 250.0  # 250 / 1, not / 2
    # plays_pg comes from the volume service, not attempts relabeled.
    assert kc["plays_pg"] == 64.5
    assert kc["pass_rate"] == 73 / (73 + 47)


def test_compute_actual_midweek_includes_thursday_final_per_team():
    rows = _two_week_rows() + [
        _game(2026, 3, "KC", "LV", "35", "10"),   # Thursday final...
        _game(2026, 3, "BUF", "MIA", "", ""),     # ...rest of week pending
    ]
    table = compute_team_offense(2026, games_rows=rows, get_week_teams=_week_teams)
    assert table["completed_weeks"] == [1, 2]
    assert table["in_progress_weeks"] == [3]
    kc = table["teams"]["KC"]
    assert kc["games"] == 3
    assert kc["points_pg"] == pytest.approx((30 + 27 + 35) / 3)
    # Thursday stats attributed to the two teams that played.
    assert kc["pass_yds_pg"] == pytest.approx((300 + 320 + 340) / 3)
    lv = table["teams"]["LV"]
    assert lv["games"] == 1
    assert lv["points_pg"] == 10.0
    assert lv["pass_yds_pg"] == 210.0
    # Teams that have not played week 3 are untouched.
    assert table["teams"]["BUF"]["games"] == 1
    assert "MIA" not in table["teams"]


def test_aggregate_sleeper_team_weeks_for_teams_only_final_teams():
    completed = {
        "KC": {"points": 92.0, "games": 3, "weeks": [1, 2, 3]},
        "LV": {"points": 10.0, "games": 1, "weeks": [3]},
        "BUF": {"points": 20.0, "games": 1, "weeks": [1]},
        "DAL": {"points": 41.0, "games": 2, "weeks": [1, 2]},
    }
    totals = aggregate_sleeper_team_weeks_for_teams(_week_teams, completed)
    assert totals["KC"]["pass_yds"] == 960.0   # 300 + 320 + 340
    assert totals["LV"]["pass_yds"] == 210.0   # week 3 only
    assert totals["BUF"]["pass_yds"] == 250.0  # week 1 only
    assert totals["DAL"]["pass_yds"] == 420.0  # 200 + 220
    assert "PHI" not in totals


def test_season_label_week_in_progress():
    pytest.importorskip("pandas")
    pytest.importorskip("flask")
    from app import _nfl_teams_season_label
    assert _nfl_teams_season_label(2026, "actual", [1, 2], [3]) == \
        "2026 actuals, Week 3 in progress"
    assert _nfl_teams_season_label(2026, "actual", [1, 2], []) == \
        "2026 actuals, through Week 2"
    assert _nfl_teams_season_label(2026, "actual", [1, 2]) == \
        "2026 actuals, through Week 2"
    assert _nfl_teams_season_label(2026, "projection", [], []) == \
        "2026 projections"
    assert _nfl_teams_season_label(2026, "actual", [], []) == \
        "2026 actuals"


def test_compute_projection_divides_by_17_and_omits_scoring():
    table = compute_team_offense(
        2026,
        games_rows=[_game(2026, 1, "KC", "BUF", "", "")],  # nothing completed
        projected_totals={"KC": {"pass_yds": 4760, "pass_att": 600, "rush_yds": 2040,
                                 "rush_att": 440, "pass_tds": 34, "rush_tds": 17}},
    )
    assert table["data_mode"] == "projection"
    kc = table["teams"]["KC"]
    assert kc["games"] == 0
    assert kc["points_pg"] is None  # never fabricate projected NFL points
    assert kc["pass_yds_pg"] == 4760 / PROJECTION_DIVISOR
    assert kc["plays_pg"] == (600 + 440) / PROJECTION_DIVISOR


def test_rank_offense_table_shape_and_missing_values():
    table = compute_team_offense(
        2026,
        games_rows=_two_week_rows(),
        get_week_teams=_week_teams,
    )
    ranks = rank_offense_table(table)
    pts = ranks["points_pg"]
    assert pts["KC"]["rank"] == 1
    assert pts["KC"]["value"] == 28.5
    # No play-volume map passed: plays_pg unranked (N/A), not zero-filled.
    assert ranks["plays_pg"]["KC"] is None
    # Zero rush TDs still earn a rank.
    assert ranks["rush_tds_pg"]["BUF"]["rank"] is not None
