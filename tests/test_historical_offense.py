"""Last-year team offense ranks (projected-offense analog)."""
from dashboard_services.historical.definitions import (
    TRENDS_OFFENSE_RANGES,
    normalize_team_abbr,
    offense_rank_bucket,
    trends_offense_range,
)
from dashboard_services.historical.offense import (
    apply_team_offense_overlay,
    lookup_team_prior_offense_rank,
    overlay_payload,
    prior_offense_rank_for,
    rank_teams,
    team_offense_lookup_from_rows,
    usage_team_and_offense,
)
from dashboard_services.historical.board import (
    _offense_window_title,
    format_hist_trend_title,
)
from dashboard_services.historical.filters import extract_trend_features


def test_offense_ranges_cover_1_to_32():
    covered = []
    for _key, _label, lo, hi in TRENDS_OFFENSE_RANGES:
        covered.extend(range(lo, hi + 1))
    assert covered == list(range(1, 33))
    assert trends_offense_range(1)[0] == "top_10"
    assert trends_offense_range(10)[1] == "Top 10"
    assert trends_offense_range(11)[0] == "11_20"
    assert trends_offense_range(20)[1] == "11-20"
    assert trends_offense_range(21)[0] == "21_32"
    assert trends_offense_range(32)[1] == "21-32"
    assert trends_offense_range(33) is None
    assert trends_offense_range(None) is None
    assert offense_rank_bucket(8) == "top_10"
    assert offense_rank_bucket(0) is None


def test_normalize_team_aliases():
    assert normalize_team_abbr("was") == "WSH"
    assert normalize_team_abbr("JAC") == "JAX"
    assert normalize_team_abbr("ARI") == "ARI"
    assert normalize_team_abbr("FA") is None
    assert normalize_team_abbr("") is None


def test_rank_teams_best_is_one():
    ranks = rank_teams({"KC": 9000, "ARI": 4000, "NYJ": 1000})
    assert ranks["KC"] == 1
    assert ranks["ARI"] == 2
    assert ranks["NYJ"] == 3


def test_usage_rows_rank_and_prior_lookup():
    rows = [
        {
            "id": "qb-kc",
            "season": 2023,
            "team": "KC",
            "passing_yards": 4000,
            "rush_yards": 200,
            "passing_tds": 30,
            "rush_tds": 2,
        },
        {
            "id": "rb-ari",
            "season": 2023,
            "team": "ARI",
            "passing_yards": 0,
            "rush_yards": 800,
            "passing_tds": 0,
            "rush_tds": 6,
        },
        {
            "id": "qb-kc",
            "season": 2024,
            "team": "KC",
            "passing_yards": 3500,
            "rush_yards": 100,
            "passing_tds": 20,
            "rush_tds": 1,
        },
        {
            "id": "rb-new",
            "season": 2024,
            "team": "KC",
            "passing_yards": 0,
            "rush_yards": 200,
            "passing_tds": 0,
            "rush_tds": 1,
        },
    ]
    ranks, teams = team_offense_lookup_from_rows(rows)
    assert ranks[2023]["KC"] == 1
    assert ranks[2023]["ARI"] == 2
    assert teams["rb-new"]["2024"] == "KC"
    assert prior_offense_rank_for(ranks, "KC", 2024) == 1
    assert prior_offense_rank_for(ranks, "ARI", 2024) == 2
    assert prior_offense_rank_for(ranks, "KC", 2023) is None

    team, score = usage_team_and_offense(
        {
            "team": "WAS",
            "usage": {
                "games": 2,
                "avg_pass_yds": 200.0,
                "avg_rush_yards": 50.0,
                "avg_pass_tds": 1.0,
                "avg_rush_tds": 0.5,
            },
        }
    )
    assert team == "WSH"
    assert score == 400 + 100 + 40 * (2.0 + 1.0)


def test_overlay_stamps_observations_and_live_profiles():
    ranks, teams = team_offense_lookup_from_rows(
        [
            {"id": "1", "season": 2024, "team": "KC", "passing_yards": 5000, "rush_yards": 0},
            {"id": "2", "season": 2024, "team": "ARI", "passing_yards": 1000, "rush_yards": 0},
            {"id": "1", "season": 2025, "team": "KC", "passing_yards": 4000, "rush_yards": 0},
        ]
    )
    overlay = overlay_payload(ranks, teams)
    data = {
        "preseason_profiles": {
            "upcoming_season": 2026,
            "by_player": {
                "love": {"position": "RB", "team": "KC", "years_experience": 0},
            },
        },
        "cohort_index": {
            "observations": [
                {"pid": "1", "season": 2025, "pos": "QB", "feats": {"position": "QB"}},
                {"pid": "2", "season": 2025, "pos": "RB", "feats": {"position": "RB"}},
            ]
        },
    }
    stamped = apply_team_offense_overlay(data, overlay)
    assert stamped >= 2
    assert data["cohort_index"]["observations"][0]["feats"]["prior_offense_rank"] == 1
    assert data["preseason_profiles"]["by_player"]["love"]["prior_offense_rank"] == 1
    assert lookup_team_prior_offense_rank(data, "KC") == 1
    feats = extract_trend_features(data["preseason_profiles"]["by_player"]["love"])
    assert feats["prior_offense_rank"] == 1
    assert feats["prior_offense_rank_bucket"] == "top_10"
    assert feats["team"] == "KC"


def test_offense_titles_name_the_year():
    assert _offense_window_title("Top 10", "last_year") == "Top-10 offense last year"
    assert _offense_window_title("Top 10", "year_1") == "Top-10 offense last year, year 1"
    assert _offense_window_title("11-20", "year_2") == "11-20 offense last year, year 2"
    assert format_hist_trend_title(
        kind="offense", label="Team offense", bucket="Top 10"
    ) == "Top-10 offense last year"
    assert format_hist_trend_title(
        kind="offense_year_1", label="Team offense", bucket="Top 10"
    ) == "Top-10 offense last year, year 1"
