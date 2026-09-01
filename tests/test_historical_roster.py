"""Preseason roster-spot analog (ADP rank among teammates)."""
from dashboard_services.historical.roster import (
    assign_observation_roster_spots,
    normalize_roster_spot,
    rank_roster_spots,
    roster_spot_label,
    stamp_roster_spots_on_queries,
)
from dashboard_services.historical.board import (
    attach_historical_signals,
    format_hist_trend_title,
    query_for_board_player,
)
from dashboard_services.historical.cohorts import evaluate_cohort, reset_cohort_cache
from dashboard_services.historical.filters import canonical_filter_key, extract_trend_features
from dashboard_services.historical.offense import apply_team_offense_overlay


def test_roster_spot_labels_have_no_underscore():
    assert roster_spot_label("RB", 1) == "RB1"
    assert roster_spot_label("RB", 2) == "RB2"
    assert roster_spot_label("RB", 3) == "RB3+"
    assert roster_spot_label("RB", 5) == "RB3+"
    assert roster_spot_label("WR", "3+") == "WR3+"
    assert "_" not in roster_spot_label("TE", 3)
    assert normalize_roster_spot(None) is None
    assert normalize_roster_spot(0) is None
    assert normalize_roster_spot("2") == 2


def test_rank_roster_spots_lowest_adp_is_starter():
    members = [
        {"pid": "a", "adp": 12.0, "name": "Lead"},
        {"pid": "b", "adp": 40.0, "name": "Two"},
        {"pid": "c", "adp": 90.0, "name": "Three"},
        {"pid": "d", "adp": 140.0, "name": "Four"},
        {"pid": "e", "name": "NoAdp"},
    ]
    spots = rank_roster_spots(members)
    assert spots == {"a": 1, "b": 2, "c": 3, "d": 3}
    assert "e" not in spots


def test_assign_observation_roster_spots_groups_by_season_team_pos():
    obs = [
        {"pid": "a", "season": 2024, "pos": "RB", "adp": 12.0, "name": "A",
         "feats": {"position": "RB", "team": "KC"}},
        {"pid": "b", "season": 2024, "pos": "RB", "adp": 40.0, "name": "B",
         "feats": {"position": "RB", "team": "KC"}},
        {"pid": "c", "season": 2024, "pos": "RB", "adp": 90.0, "name": "C",
         "feats": {"position": "RB", "team": "KC"}},
        {"pid": "d", "season": 2024, "pos": "RB", "adp": 15.0, "name": "D",
         "feats": {"position": "RB", "team": "DET"}},
        {"pid": "e", "season": 2024, "pos": "WR", "adp": 10.0, "name": "E",
         "feats": {"position": "WR", "team": "KC"}},
        {"pid": "f", "season": 2023, "pos": "RB", "adp": 8.0, "name": "F",
         "feats": {"position": "RB", "team": "KC"}},
        {"pid": "g", "season": 2024, "pos": "RB", "name": "G",
         "feats": {"position": "RB", "team": "KC"}},
    ]
    n = assign_observation_roster_spots(obs)
    assert n == 6
    by = {row["pid"]: row["feats"].get("roster_spot") for row in obs}
    assert by["a"] == 1
    assert by["b"] == 2
    assert by["c"] == 3
    assert by["d"] == 1
    assert by["e"] == 1
    assert by["f"] == 1
    assert by["g"] is None


def test_overlay_stamps_roster_spot_from_adp():
    data = {
        "cohort_index": {
            "observations": [
                {"pid": "1", "season": 2025, "pos": "RB", "adp": 18.0,
                 "feats": {"position": "RB", "team": "ARI"}},
                {"pid": "2", "season": 2025, "pos": "RB", "adp": 55.0,
                 "feats": {"position": "RB", "team": "ARI"}},
                {"pid": "3", "season": 2025, "pos": "RB", "adp": 110.0,
                 "feats": {"position": "RB", "team": "ARI"}},
            ]
        }
    }
    apply_team_offense_overlay(data, {"ranks_by_season": {}, "projected_ranks_by_season": {}})
    by = {row["pid"]: row["feats"]["roster_spot"] for row in data["cohort_index"]["observations"]}
    assert by == {"1": 1, "2": 2, "3": 3}


def test_canonical_filter_key_distinguishes_nested_all_intersections():
    top10 = {"group": "projected_offense", "field": "projected_offense_rank", "between": [1, 10]}
    rb1 = {"group": "roster_spot", "field": "roster_spot", "eq": 1}
    rb3 = {"group": "roster_spot", "field": "roster_spot", "eq": 3}
    a = {"group": "offense_roster", "all": [top10, rb1]}
    b = {"group": "offense_roster", "all": [top10, rb3]}
    assert canonical_filter_key([a]) != canonical_filter_key([b])
    assert canonical_filter_key([a, b]) == canonical_filter_key([b, a])


def test_extract_trend_features_passes_roster_spot():
    feats = extract_trend_features({
        "position": "RB",
        "years_experience": 0,
        "roster_spot": 3,
        "team": "ARI",
    })
    assert feats["roster_spot"] == 3
    assert feats["team"] == "ARI"


def test_live_board_ranks_teammates_by_adp():
    queries = [
        {"sleeper_id": "love", "position": "RB", "team": "ARI", "adp": 18.0},
        {"sleeper_id": "conner", "position": "RB", "team": "ARI", "adp": 48.0},
        {"sleeper_id": "benson", "position": "RB", "team": "ARI", "adp": 96.0},
        {"sleeper_id": "other", "position": "RB", "team": "DET", "adp": 22.0},
    ]
    stamp_roster_spots_on_queries(queries)
    by = {row["sleeper_id"]: row["roster_spot"] for row in queries}
    assert by["love"] == 1
    assert by["conner"] == 2
    assert by["benson"] == 3
    assert by["other"] == 1


def test_attach_historical_signals_stamps_live_roster_spot():
    profiles = {
        "love": {"position": "RB", "team": "ARI", "years_experience": 0},
        "conner": {"position": "RB", "team": "ARI", "years_experience": 8},
        "benson": {"position": "RB", "team": "ARI", "years_experience": 1},
    }
    aggs = {"preseason_profiles": {"by_player": profiles, "upcoming_season": 2026}}
    board = [
        {"id": "love", "position": "RB", "team": "ARI", "redraft_avg_pick": 18.0},
        {"id": "conner", "position": "RB", "team": "ARI", "redraft_avg_pick": 48.0},
        {"id": "benson", "position": "RB", "team": "ARI", "redraft_avg_pick": 96.0},
    ]
    attach_historical_signals(board, aggs)
    assert board[0]["historical"]["trend_feats"]["roster_spot"] == 1
    assert board[1]["historical"]["trend_feats"]["roster_spot"] == 2
    assert board[2]["historical"]["trend_feats"]["roster_spot"] == 3


def test_query_for_board_player_keeps_extra_roster_spot():
    query = query_for_board_player(
        {"id": "love", "position": "RB", "team": "ARI", "roster_spot": 1, "redraft_avg_pick": 18},
        {},
    )
    assert query["roster_spot"] == 1


def test_offense_roster_titles_name_spot_and_year():
    assert format_hist_trend_title(
        kind="offense_roster", label="Offense", bucket="Top 10, RB1"
    ) == "Top-10 projected offense, RB1"
    assert format_hist_trend_title(
        kind="offense_roster_1", label="Offense", bucket="Top 10, RB3+"
    ) == "Top-10 projected offense, RB3+, year 1"
    assert format_hist_trend_title(
        kind="offense_roster", label="Offense", bucket="21-32, RB2"
    ) == "21-32 projected offense, RB2"
    assert "_" not in format_hist_trend_title(
        kind="offense_roster", label="Offense", bucket="Top 10, RB3+"
    )


def test_rb1_on_top10_is_counted_intersection_not_a_product():
    reset_cohort_cache()
    obs = []
    for i in range(10):
        obs.append({
            "pid": f"s{i}",
            "pos": "RB",
            "season": 2020 + (i % 5),
            "finish": 5 if i < 8 else 40,
            "feats": {"position": "RB", "projected_offense_rank": 3, "roster_spot": 1},
        })
    for i in range(10):
        obs.append({
            "pid": f"d{i}",
            "pos": "RB",
            "season": 2020 + (i % 5),
            "finish": 5 if i < 1 else 40,
            "feats": {"position": "RB", "projected_offense_rank": 3, "roster_spot": 3},
        })
    aggs = {
        "cohort_index": {"observations": obs},
        "age_curves": {"RB": {"baseline": {"display_pct": 8, "sample_size": 100}}},
    }
    top10 = {"group": "projected_offense", "field": "projected_offense_rank", "between": [1, 10]}
    rb1 = {"group": "roster_spot", "field": "roster_spot", "eq": 1}
    rb3 = {"group": "roster_spot", "field": "roster_spot", "eq": 3}
    starter = evaluate_cohort(aggs, position="RB", filters=[top10, rb1], data_version="rs")
    depth = evaluate_cohort(aggs, position="RB", filters=[top10, rb3], data_version="rs")
    any_rb = evaluate_cohort(aggs, position="RB", filters=[top10], data_version="rs")
    assert starter["sample_size"] == 10
    assert depth["sample_size"] == 10
    assert starter["successes"] == 8
    assert depth["successes"] == 1
    assert abs(starter["raw_rate"] - 0.8) < 1e-9
    assert abs(depth["raw_rate"] - 0.1) < 1e-9
    product = any_rb["raw_rate"] * depth["raw_rate"]
    assert abs(any_rb["raw_rate"] - 0.45) < 1e-9
    assert abs(depth["raw_rate"] - product) > 0.05
