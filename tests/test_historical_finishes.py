"""Positional finishes, usage-row canonicalization, and no-leakage prior features.

Pure Python — runs in the slim CI job (no pandas / Flask).
"""
from copy import deepcopy

from dashboard_services.historical.finishes import (
    OUTCOME_COLUMNS,
    assign_season_finishes,
    attach_prior_career_features,
    competition_ranks,
    prior_career_features_for_player,
)
from dashboard_services.historical.seasons import canonicalize_usage_row, row_appeared


def test_competition_ranks_ties_share_and_skip():
    assert competition_ranks([30, 20, 20, 10]) == [1, 2, 2, 4]
    assert competition_ranks([None, 5, None]) == [None, 1, None]
    assert competition_ranks([None, None]) == [None, None]


def test_positional_finish_uses_total_points_not_ppg():
    rows = [
        {"sleeper_id": "a", "position": "RB", "ppr_points": 200, "ppr_ppg": 20.0, "games": 10},
        {"sleeper_id": "b", "position": "RB", "ppr_points": 210, "ppr_ppg": 12.0, "games": 17},
        {"sleeper_id": "c", "position": "WR", "ppr_points": 250, "ppr_ppg": 15.0, "games": 17},
    ]
    out = assign_season_finishes(rows, scoring="ppr")
    by_id = {r["sleeper_id"]: r for r in out}
    # Total points: C 250 overall 1, B 210 overall 2, A 200 overall 3.
    assert by_id["c"]["ppr_overall_finish"] == 1
    assert by_id["b"]["ppr_overall_finish"] == 2
    assert by_id["a"]["ppr_overall_finish"] == 3
    # Within RB, B outranks A on totals even though A has the higher PPG.
    assert by_id["b"]["ppr_positional_finish"] == 1
    assert by_id["a"]["ppr_positional_finish"] == 2
    assert by_id["c"]["ppr_positional_finish"] == 1
    assert by_id["b"]["ppr_tier"] == "RB1"
    assert by_id["b"]["ppr_top_12"] is True
    assert by_id["c"]["ppr_top_3"] is True


def test_ppg_finish_is_opt_in_and_does_not_replace_totals():
    rows = [
        {"sleeper_id": "a", "position": "RB", "ppr_points": 200, "ppr_ppg": 20.0},
        {"sleeper_id": "b", "position": "RB", "ppr_points": 210, "ppr_ppg": 12.0},
    ]
    out = assign_season_finishes(rows, scoring="ppr", by_ppg=True)
    by_id = {r["sleeper_id"]: r for r in out}
    assert by_id["a"]["ppr_ppg_positional_finish"] == 1
    assert by_id["b"]["ppr_ppg_positional_finish"] == 2
    assert "ppr_positional_finish" not in by_id["a"]


def test_unranked_when_points_missing_not_a_zero():
    rows = [
        {"sleeper_id": "a", "position": "WR", "ppr_points": 100},
        {"sleeper_id": "b", "position": "WR", "ppr_points": None},
    ]
    out = assign_season_finishes(rows, scoring="ppr")
    by_id = {r["sleeper_id"]: r for r in out}
    assert by_id["a"]["ppr_positional_finish"] == 1
    assert by_id["b"]["ppr_positional_finish"] is None
    assert by_id["b"]["ppr_top_12"] is False


def test_canonicalize_legacy_totals_schema():
    raw = {
        "id": "167",
        "position": "QB",
        "name": "T.Brady",
        "usage": {
            "gsis_id": "00-0019596",
            "team": "NE",
            "games": 16,
            "targets": 1,
            "carries": 23,
            "receptions": 1,
            "rec_yards": 6,
            "rec_tds": 0.0,
            "rush_yards": 35,
            "rush_tds": 2.0,
            "pass_attempts": 570,
            "pass_yards": 4355,
            "pass_tds": 29.0,
            "interceptions": 11.0,
            "ppr_ppg": 17.58,
            "ppr_total": 281.3,
            "snap_share": 0.0,
            "avg_off_snap_pct": 0.0,
            "target_share": 0.0019,
        },
    }
    identity = {
        "name": "Tom Brady",
        "birth_date": "8/3/1977",
        "draft_year": 2000,
        "nfl_draft_round": 6,
        "nfl_draft_pick": 199,
    }
    row = canonicalize_usage_row(raw, 2018, identity)
    assert row["sleeper_id"] == "167"
    assert row["gsis_id"] == "00-0019596"
    assert row["ppr_points"] == 281.3
    assert row["passing_attempts"] == 570
    assert row["standard_points"] == 281.3 - 1  # PPR minus receptions
    assert row["half_ppr_points"] == 281.3 - 0.5
    assert row["snap_pct"] is None  # 0% with 500+ pass attempts is missing, not a real 0
    assert row["age"] == 41.0  # 1977-08-03 → 2018-09-01
    assert row["years_experience"] == 18
    assert row["draft_capital_bucket"] == "day_3"
    assert row["source_schema"] == "legacy_totals"


def test_canonicalize_sleeper_averages_schema_null_not_zero():
    raw = {
        "id": "9488",
        "name": "Jaxon Smith-Njigba",
        "team": "SEA",
        "position": "WR",
        "usage": {
            "games": 17,
            "avg_off_snap_pct": 0.0,
            "avg_off_snaps": 55.7,
            "avg_targets": 8.0,
            "avg_receptions": 6.0,
            "avg_rec_yards": 66.0,
            "avg_rec_tds": 0.35,
            "avg_carries": 0.3,
            "avg_rush_yards": 3.5,
            "avg_rush_tds": 0.0,
            "ppr_ppg": 14.8,
            "half_ppr_ppg": 11.9,
            "std_scoring_ppg": 9.0,
            "std_ppg": 0.0,
            "rec_rz_tgt_pg": 0.76,
            "target_share": 0.24,
        },
    }
    identity = {"birth_date": "2/14/2002", "draft_year": 2023, "nfl_draft_round": 1, "nfl_draft_pick": 20}
    row = canonicalize_usage_row(raw, 2024, identity)
    assert row["targets"] == 8.0 * 17
    assert row["avg_targets"] == 8.0
    assert row["ppr_points"] == 14.8 * 17
    assert row["standard_ppg"] == 9.0  # not the leftover std_ppg=0.0
    assert row["air_yards"] is None
    assert row["adot"] is None
    assert row["starts"] is None
    assert row["snap_pct"] is None  # 0% with real snaps/targets → missing
    assert row["snaps"] == 55.7 * 17
    assert row["age"] == 22.5
    assert row["years_experience"] == 1
    assert row["draft_capital_bucket"] == "round_1"
    assert "projected_points" not in row
    assert "adp" not in row


def test_empty_usage_does_not_count_as_an_appeared_season():
    empty = canonicalize_usage_row({"id": "1", "position": "WR", "name": "X", "usage": {}}, 2022, {})
    assert empty is not None
    assert row_appeared(empty) is False
    played = canonicalize_usage_row(
        {"id": "1", "position": "RB", "usage": {"games": 8, "ppr_ppg": 5, "avg_carries": 10}},
        2022, {},
    )
    assert row_appeared(played) is True
    empty = {"id": "1", "position": "WR", "name": "X", "usage": {}}
    row = canonicalize_usage_row(empty, 2022, {"birth_date": "1/1/1998"})
    assert row["ppr_points"] is None
    assert row["games"] is None
    assert row["targets"] is None
    kicker = {"id": "2", "position": "PK", "name": "K", "usage": {"games": 17, "ppr_ppg": 10}}
    assert canonicalize_usage_row(kicker, 2024, {}) is None
    assert canonicalize_usage_row({"id": None, "position": "RB"}, 2024, {}) is None


def test_target_share_zero_with_targets_is_missing():
    raw = {
        "id": "9",
        "position": "WR",
        "usage": {"games": 16, "avg_targets": 5, "target_share": 0.0, "ppr_ppg": 8},
    }
    row = canonicalize_usage_row(raw, 2021, {})
    assert row["targets"] == 80
    assert row["target_share"] is None


def _career():
    return [
        {
            "sleeper_id": "rb1",
            "season": 2021,
            "position": "RB",
            "ppr_points": 250,
            "ppr_ppg": 15.0,
            "games": 16,
            "ppr_positional_finish": 8,
            "targets": 50,
        },
        {
            "sleeper_id": "rb1",
            "season": 2022,
            "position": "RB",
            "ppr_points": 300,
            "ppr_ppg": 18.0,
            "games": 17,
            "ppr_positional_finish": 3,
            "targets": 70,
        },
        {
            "sleeper_id": "rb1",
            "season": 2023,
            "position": "RB",
            "ppr_points": 120,
            "ppr_ppg": 10.0,
            "games": 12,
            "ppr_positional_finish": 22,
            "targets": 40,
        },
    ]


def test_prior_career_features_are_preseason_only():
    rows = prior_career_features_for_player(_career())
    by_year = {r["season"]: r for r in rows}
    rookie = by_year[2021]
    assert rookie["career_seasons_before_current"] == 0
    assert rookie["previous_season_finish"] is None
    assert rookie["previous_season_ppg"] is None
    assert rookie["career_best_finish_before_season"] is None
    assert rookie["prior_top12_count"] == 0
    assert rookie["previously_top12"] is False
    assert rookie["first_time_top12_candidate"] is True

    y2022 = by_year[2022]
    assert y2022["career_seasons_before_current"] == 1
    assert y2022["previous_season_finish"] == 8
    assert y2022["previous_season_ppg"] == 15.0
    assert y2022["previous_season_games"] == 16
    assert y2022["career_best_finish_before_season"] == 8
    assert y2022["career_best_ppg_before_season"] == 15.0
    assert y2022["prior_top12_count"] == 1
    assert y2022["previously_top12"] is True
    assert y2022["previously_top5"] is False
    assert y2022["first_time_top12_candidate"] is False

    y2023 = by_year[2023]
    assert y2023["career_seasons_before_current"] == 2
    assert y2023["previous_season_finish"] == 3
    assert y2023["career_best_finish_before_season"] == 3
    assert y2023["prior_top3_count"] == 1
    assert y2023["prior_top5_count"] == 1
    assert y2023["prior_top12_count"] == 2
    assert y2023["previously_top5"] is True


def test_2022_feature_row_does_not_change_when_2022_actuals_change():
    career = _career()
    original = {r["season"]: r for r in prior_career_features_for_player(career)}[2022]
    mutated = deepcopy(career)
    mutated[1]["ppr_points"] = 1.0
    mutated[1]["ppr_ppg"] = 0.1
    mutated[1]["games"] = 1
    mutated[1]["ppr_positional_finish"] = 80
    mutated[1]["targets"] = 0
    after = {r["season"]: r for r in prior_career_features_for_player(mutated)}[2022]
    for key, value in original.items():
        if key in OUTCOME_COLUMNS:
            continue
        assert after[key] == value, f"feature {key} leaked 2022 actuals: {value!r} vs {after[key]!r}"
    # Outcome columns on the 2022 row *are* allowed to change.
    assert after["ppr_points"] == 1.0
    assert after["ppr_positional_finish"] == 80


def test_2023_features_do_see_updated_2022_outcomes():
    """Changing 2022 actuals must move 2023's *prior* features — that is not leakage."""
    career = _career()
    mutated = deepcopy(career)
    mutated[1]["ppr_positional_finish"] = 80
    mutated[1]["ppr_ppg"] = 0.1
    after = {r["season"]: r for r in prior_career_features_for_player(mutated)}[2023]
    assert after["previous_season_finish"] == 80
    assert after["prior_top12_count"] == 1  # only 2021 remains top-12
    assert after["career_best_ppg_before_season"] == 15.0  # 2021, not the mutated 2022


def test_finishes_rank_within_season_only():
    rows = [
        {"sleeper_id": "a", "season": 2021, "position": "RB", "ppr_points": 300},
        {"sleeper_id": "b", "season": 2022, "position": "RB", "ppr_points": 100},
    ]
    out = assign_season_finishes(rows, scoring="ppr")
    by = {(r["sleeper_id"], r["season"]): r for r in out}
    assert by[("a", 2021)]["ppr_positional_finish"] == 1
    assert by[("b", 2022)]["ppr_positional_finish"] == 1
    assert by[("a", 2021)]["ppr_overall_finish"] == 1
    assert by[("b", 2022)]["ppr_overall_finish"] == 1


def test_attach_groups_by_player_and_keeps_scoring_prefixes():
    rows = [
        {"sleeper_id": "a", "season": 2022, "position": "WR",
         "ppr_points": 200, "ppr_ppg": 12, "games": 16, "ppr_positional_finish": 10,
         "half_ppr_points": 160, "half_ppr_ppg": 10, "half_ppr_positional_finish": 11,
         "standard_points": 120, "standard_ppg": 8, "standard_positional_finish": 12},
        {"sleeper_id": "a", "season": 2023, "position": "WR",
         "ppr_points": 220, "ppr_ppg": 13, "games": 17, "ppr_positional_finish": 8,
         "half_ppr_points": 170, "half_ppr_ppg": 10, "half_ppr_positional_finish": 9,
         "standard_points": 130, "standard_ppg": 8, "standard_positional_finish": 10},
        {"sleeper_id": "b", "season": 2023, "position": "TE",
         "ppr_points": 90, "ppr_ppg": 6, "games": 15, "ppr_positional_finish": 18,
         "half_ppr_points": 80, "half_ppr_ppg": 5, "half_ppr_positional_finish": 18,
         "standard_points": 70, "standard_ppg": 4, "standard_positional_finish": 18},
    ]
    attached = attach_prior_career_features(rows)
    a_2023 = next(r for r in attached if r["sleeper_id"] == "a" and r["season"] == 2023)
    assert a_2023["previous_season_finish"] == 10
    assert a_2023["half_ppr_previous_season_finish"] == 11
    assert a_2023["standard_previous_season_finish"] == 12
    b = next(r for r in attached if r["sleeper_id"] == "b")
    assert b["career_seasons_before_current"] == 0
    assert b["first_time_top12_candidate"] is True
