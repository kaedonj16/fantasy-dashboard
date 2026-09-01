"""Season-long projected team offense ranks and last-year actual analog."""
from dashboard_services.historical.definitions import (
    TRENDS_OFFENSE_RANGES,
    normalize_team_abbr,
    offense_rank_bucket,
    trends_offense_range,
)
from dashboard_services.historical.offense import (
    apply_team_offense_overlay,
    extra_observations_from_player_seasons,
    implied_team_points,
    lookup_team_prior_offense_rank,
    lookup_team_projected_offense_rank,
    overlay_payload,
    prior_offense_rank_for,
    projected_ranks_from_games,
    rank_teams,
    season_offense_rank_for,
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
    assert normalize_team_abbr("LA") == "LAR"
    assert normalize_team_abbr("STL") == "LAR"
    assert normalize_team_abbr("SL") == "LAR"
    assert normalize_team_abbr("ARZ") == "ARI"
    assert normalize_team_abbr("BLT") == "BAL"
    assert normalize_team_abbr("CLV") == "CLE"
    assert normalize_team_abbr("HST") == "HOU"
    assert normalize_team_abbr("SD") == "LAC"
    assert normalize_team_abbr("OAK") == "LV"
    assert normalize_team_abbr("FA") is None
    assert normalize_team_abbr("") is None


def test_rank_teams_best_is_one():
    ranks = rank_teams({"KC": 9000, "ARI": 4000, "NYJ": 1000})
    assert ranks["KC"] == 1
    assert ranks["ARI"] == 2
    assert ranks["NYJ"] == 3


def test_implied_team_points_home_favored():
    # total 46.5, home -10.5 favorite in nflverse (spread_line +10.5)
    assert implied_team_points(46.5, 10.5, home=True) == 28.5
    assert implied_team_points(46.5, 10.5, home=False) == 18.0
    assert implied_team_points(None, 3) is None
    assert implied_team_points(44, None) is None


def test_projected_ranks_average_all_games_not_just_one_week():
    ranks = projected_ranks_from_games(
        [
            {"home": "LAC", "away": "ARI", "total_line": 46.5, "spread_line": 10.5},
            {"home_team": "LA", "away_team": "SF", "total_line": 48.5, "spread_line": 3.5},
            {"home": "DET", "away": "NO", "total_line": 49.5, "spread_line": 7.0},
        ]
    )
    assert ranks["LAC"] == 1
    assert ranks["DET"] == 2
    assert ranks["LAR"] == 3
    assert ranks["SF"] == 4
    assert ranks["NO"] == 5
    assert ranks["ARI"] == 6
    assert season_offense_rank_for({2026: ranks}, "LA", 2026) == 3
    assert season_offense_rank_for({2026: ranks}, "ARI", 2026) == 6

    # A second LAC game with a low implied total should pull LAC down.
    # Summing would still leave LAC first; averaging should not.
    averaged = projected_ranks_from_games(
        [
            {"home": "LAC", "away": "ARI", "total_line": 46.5, "spread_line": 10.5},
            {"home": "LAC", "away": "DET", "total_line": 40.0, "spread_line": -6.0},
        ]
    )
    # LAC: (28.5 + 17.0) / 2 = 22.75; DET: 23.0; ARI: 18.0
    assert averaged["DET"] == 1
    assert averaged["LAC"] == 2
    assert averaged["ARI"] == 3


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


def test_overlay_stamps_projected_prior_and_extra_years():
    ranks, teams = team_offense_lookup_from_rows(
        [
            {"id": "1", "season": 2024, "team": "KC", "passing_yards": 5000, "rush_yards": 0},
            {"id": "2", "season": 2024, "team": "ARI", "passing_yards": 1000, "rush_yards": 0},
            {"id": "1", "season": 2025, "team": "KC", "passing_yards": 4000, "rush_yards": 0},
            {"id": "2", "season": 2025, "team": "ARI", "passing_yards": 1200, "rush_yards": 0},
        ]
    )
    projected = {
        2025: {"KC": 2, "ARI": 7},
        2026: {"KC": 16, "ARI": 31},
    }
    extras = extra_observations_from_player_seasons(
        [
            {
                "sleeper_id": "old-rb",
                "season": 2016,
                "position": "RB",
                "team": "KC",
                "name": "Old Back",
                "years_experience": 0,
                "ppr_positional_finish": 8,
                "nfl_draft_pick": 10,
            }
        ],
        projected_ranks={2016: {"KC": 4}},
        actual_ranks={2015: {"KC": 1}},
    )
    overlay = overlay_payload(
        ranks,
        teams,
        projected_ranks_by_season=projected,
        extra_observations=extras,
        extra_seasons=[2016, 2017],
    )
    data = {
        "preseason_profiles": {
            "upcoming_season": 2026,
            "by_player": {
                "love": {"position": "RB", "team": "ARI", "years_experience": 0},
            },
        },
        "cohort_index": {
            "observations": [
                {"pid": "1", "season": 2025, "pos": "QB", "feats": {"position": "QB", "team": "KC"}},
                {"pid": "2", "season": 2025, "pos": "RB", "feats": {"position": "RB", "team": "ARI"}},
            ]
        },
    }
    stamped = apply_team_offense_overlay(data, overlay)
    assert stamped >= 4
    obs = data["cohort_index"]["observations"]
    assert len(obs) == 3
    by_pid = {row["pid"]: row for row in obs}
    assert by_pid["1"]["feats"]["prior_offense_rank"] == 1
    assert by_pid["1"]["feats"]["projected_offense_rank"] == 2
    assert by_pid["2"]["feats"]["projected_offense_rank"] == 7
    extra = by_pid["old-rb"]
    assert extra["season"] == 2016
    assert extra["finish"] == 8
    assert extra["feats"]["projected_offense_rank"] == 4
    assert extra["feats"]["prior_offense_rank"] == 1
    assert extra["feats"]["career_stage"] == "rookie"
    love = data["preseason_profiles"]["by_player"]["love"]
    assert love["projected_offense_rank"] == 31
    assert love["prior_offense_rank"] == 2
    assert lookup_team_projected_offense_rank(data, "ARI") == 31
    assert lookup_team_prior_offense_rank(data, "ARI") == 2
    feats = extract_trend_features(love)
    assert feats["projected_offense_rank"] == 31
    assert feats["projected_offense_rank_bucket"] == "21_32"
    assert feats["prior_offense_rank"] == 2
    assert feats["team"] == "ARI"
    assert data["team_offense"]["projected_source"] == "nflverse_season_implied_total"


def test_committed_overlay_has_projected_and_extra_years():
    import json
    from pathlib import Path

    path = Path("cache/player_history/team_offense_overlay.json")
    if not path.exists():
        return
    data = json.loads(path.read_text(encoding="utf-8"))
    projected = data.get("projected_ranks_by_season") or {}
    assert "2016" in projected and "2026" in projected
    assert len(projected["2024"]) == 32
    assert projected["2026"]["ARI"] >= 1
    extras = data.get("extra_observations") or []
    seasons = {row["season"] for row in extras}
    assert 2016 in seasons and 2017 in seasons
    assert any(row.get("pos") == "RB" and row.get("finish") for row in extras)
    assert "2015" in (data.get("ranks_by_season") or {})


def test_offense_titles_name_the_year_and_analog():
    assert _offense_window_title("Top 10", "last_year") == "Top-10 offense last year"
    assert _offense_window_title("Top 10", "year_1") == "Top-10 offense last year, year 1"
    assert _offense_window_title("11-20", "year_2") == "11-20 offense last year, year 2"
    assert _offense_window_title("Top 10", "any", analog="projected") == "Top-10 projected offense"
    assert _offense_window_title("Top 10", "year_1", analog="projected") == "Top-10 projected offense, year 1"
    assert format_hist_trend_title(
        kind="offense", label="Team offense", bucket="Top 10"
    ) == "Top-10 projected offense"
    assert format_hist_trend_title(
        kind="offense_year_1", label="Team offense", bucket="Top 10"
    ) == "Top-10 projected offense, year 1"
    assert format_hist_trend_title(
        kind="offense_last_year", label="Team offense", bucket="Top 10"
    ) == "Top-10 offense last year"
    assert format_hist_trend_title(
        kind="offense_last_year_1", label="Team offense", bucket="21-32"
    ) == "21-32 offense last year, year 1"
    assert format_hist_trend_title(
        kind="offense_roster", label="Team offense", bucket="Top 10, RB3+"
    ) == "Top-10 projected offense, RB3+"
    assert format_hist_trend_title(
        kind="capital_roster", label="NFL", bucket="Round 1, WR1"
    ) == "Drafted NFL Round 1, WR1"
    assert format_hist_trend_title(
        kind="offense_capital", label="Offense", bucket="21-32, Top 10"
    ) == "21-32 projected offense, NFL Top 10"
