"""True multi-factor historical cohorts (slim CI, no pandas)."""
from pathlib import Path
import json

from dashboard_services.historical.cohorts import (
    EDGE_RANK_PRIOR_N,
    build_cohort_index,
    closest_examples,
    edge_bundle,
    evaluate_cohort,
    examples_summary,
    reset_cohort_cache,
)
from dashboard_services.historical.definitions import (
    DEFAULT_BAYES_PRIOR_N,
    ranking_adjusted_rate,
    wilson_interval,
)
from dashboard_services.historical.filters import (
    extract_trend_features,
    matches_filter_groups,
    matches_trend_filter,
    trajectory_buckets,
)
from dashboard_services.historical.finish_rates import make_rate


ROOT = Path(__file__).resolve().parents[1]


def _obs_row(**kwargs):
    row = {
        "sleeper_id": kwargs.get("sleeper_id", "p"),
        "name": kwargs.get("name", "Player"),
        "season": kwargs.get("season", 2024),
        "position": kwargs.get("position", "WR"),
        "years_experience": kwargs.get("years_experience", 2),
        "age": kwargs.get("age", 24.0),
        "draft_capital_bucket": kwargs.get("draft_capital_bucket", "day_2"),
        "ppr_positional_finish": kwargs.get("ppr_positional_finish", 30),
        "previous_season_finish": kwargs.get("previous_season_finish", 20),
        "previous_season_target_share": kwargs.get("previous_season_target_share", 0.22),
        "previous_season_year": kwargs.get("previous_season_year", 2023),
        "games": 16,
    }
    row.update(kwargs)
    return row


def _baseline_aggs(index, *, top12=0.08, top5=0.03, top24=0.18):
    return {
        "cohort_index": index,
        "comps": {
            "by_position": {
                "WR": {
                    "baseline": {
                        "top_5": {"raw_rate": top5, "sample_size": 400},
                        "top_12": {"raw_rate": top12, "sample_size": 400},
                        "top_24": {"raw_rate": top24, "sample_size": 400},
                    }
                }
            }
        },
        "adp": {
            "by_position": {
                "WR": {
                    "by_overall_bucket": {
                        "round_1": {
                            "conditional": {"raw_rate": 0.20, "smoothed_rate": 0.20, "sample_size": 80},
                        },
                        "round_3": {
                            "conditional": {"raw_rate": 0.10, "smoothed_rate": 0.10, "sample_size": 80},
                        },
                    }
                }
            }
        },
    }


def _combo_warehouse():
    """Age x capital grid where the intersection is 10%, not 0.5*0.5=25%."""
    rows = []
    # Age 23-24 + Day 2: 5 of 50 hit (10%).
    for i in range(50):
        rows.append(_obs_row(
            sleeper_id=f"a{i}",
            name=f"Young Day2 {i}",
            age=23.4,
            draft_capital_bucket="day_2",
            ppr_positional_finish=8 if i < 5 else 40,
            previous_season_target_share=0.22 if i < 30 else 0.12,
            adp_overall=36.0 if i < 40 else None,
            adp_bucket="round_3" if i < 40 else None,
        ))
    # Age 23-24 + Round 1: 45 of 50 hit (90%) → age marginal 50%.
    for i in range(50):
        rows.append(_obs_row(
            sleeper_id=f"b{i}",
            name=f"Young R1 {i}",
            age=24.1,
            draft_capital_bucket="round_1",
            ppr_positional_finish=4 if i < 45 else 40,
            previous_season_target_share=0.22,
            adp_overall=8.0,
            adp_bucket="round_1",
        ))
    # Age 28-30 + Day 2: 45 of 50 hit (90%) → day-2 marginal 50%.
    for i in range(50):
        rows.append(_obs_row(
            sleeper_id=f"c{i}",
            name=f"Old Day2 {i}",
            age=29.0,
            draft_capital_bucket="day_2",
            ppr_positional_finish=6 if i < 45 else 40,
            previous_season_target_share=0.22,
            adp_overall=36.0,
            adp_bucket="round_3",
        ))
    # Age 28-30 + Round 1: 5 of 50 hit.
    for i in range(50):
        rows.append(_obs_row(
            sleeper_id=f"d{i}",
            name=f"Old R1 {i}",
            age=29.2,
            draft_capital_bucket="round_1",
            ppr_positional_finish=8 if i < 5 else 40,
            previous_season_target_share=0.12,
            adp_overall=8.0,
            adp_bucket="round_1",
        ))
    return rows


AGE_23 = {"group": "age_bucket", "field": "age_bucket", "eq": "23-24", "label": "Age 23-24"}
DAY_2 = {"group": "draft_capital", "field": "draft_capital", "eq": "day_2", "label": "Day 2"}
TGT_20 = {"group": "target_share", "field": "target_share", "eq": "20-25%", "label": "Target share 20-25%"}
TGT_25 = {"group": "target_share", "field": "target_share", "eq": "25%+", "label": "Target share 25%+"}
AGE_28 = {"group": "age_bucket", "field": "age_bucket", "eq": "28-30", "label": "Age 28-30"}


def test_or_within_group_and_across_groups():
    feats = extract_trend_features(_obs_row(age=23.4, draft_capital_bucket="day_2"))
    assert matches_filter_groups(feats, [AGE_23, AGE_28]) is True
    assert matches_filter_groups(feats, [AGE_23, DAY_2]) is True
    assert matches_filter_groups(feats, [AGE_28, DAY_2]) is False
    assert matches_filter_groups(feats, [AGE_23, {"group": "draft_capital", "field": "draft_capital", "eq": "round_1"}]) is False
    assert matches_filter_groups({}, [AGE_23]) is False
    assert matches_filter_groups(feats, []) is False


def test_extract_trend_features_uses_live_draft_round():
    from dashboard_services.historical.filters import live_board_trend_features

    feats = live_board_trend_features({
        "id": "13287",
        "name": "Jeremiyah Love",
        "position": "RB",
        "years_exp": 0,
        "draft_round": 1,
        "draft_pick": 4,
        "age": 21,
    })
    assert feats["position"] == "RB"
    assert feats["draft_capital"] == "round_1"
    assert feats["career_stage"] == "rookie"
    assert feats["prior_top12_count"] == 0
    assert feats["nfl_draft_pick"] == 4
    round_1 = {"group": "draft_capital", "field": "draft_capital", "eq": "round_1"}
    top_10 = {"group": "draft_capital", "field": "nfl_draft_pick", "between": [1, 10]}
    late_first = {"group": "draft_capital", "field": "nfl_draft_pick", "between": [26, 32]}
    assert matches_filter_groups(feats, [round_1]) is True
    assert matches_filter_groups(feats, [top_10]) is True
    assert matches_filter_groups(feats, [late_first]) is False
    assert matches_trend_filter({}, top_10) is False


def test_stamp_live_draft_class_adds_unprofiled_round_1_rookie():
    from dashboard_services.historical.filters import stamp_live_draft_class_profiles

    by_player = {
        "12527": {
            "position": "RB",
            "years_experience": 1,
            "draft_capital_bucket": "round_1",
        }
    }
    picks = [
        {"player_name": "Jeremiyah Love", "position": "RB", "pick": 3, "round": 1, "nfl_team": "ARI"},
        {"player_name": "Ashton Jeanty", "position": "RB", "pick": 6, "round": 1, "nfl_team": "LV"},
    ]
    index = {
        "13287": {"name": "Jeremiyah Love", "pos": "RB", "team": "ARI", "bDay": "5/31/2005"},
        "12527": {"name": "Ashton Jeanty", "pos": "RB", "team": "LV", "bDay": "12/2/2003"},
    }
    added = stamp_live_draft_class_profiles(by_player, picks, index, upcoming_season=2026)
    assert added == 1
    assert by_player["13287"]["draft_capital_bucket"] == "round_1"
    assert by_player["13287"]["nfl_draft_pick"] == 3
    assert by_player["13287"]["years_experience"] == 0
    assert by_player["12527"]["years_experience"] == 1
    feats = extract_trend_features(by_player["13287"])
    assert feats["draft_capital"] == "round_1"
    assert feats["nfl_draft_pick"] == 3
    assert feats["career_stage"] == "rookie"


def test_apply_nfl_draft_pick_overlay_stamps_feats_and_profiles():
    from dashboard_services.historical.filters import apply_nfl_draft_pick_overlay

    data = {
        "cohort_index": {
            "observations": [
                {"pid": "4034", "pos": "RB", "feats": {"position": "RB", "draft_capital": "round_1"}},
                {"pid": "x", "pos": "WR", "feats": {"position": "WR", "nfl_draft_pick": 22}},
            ]
        },
        "preseason_profiles": {
            "by_player": {
                "4034": {"position": "RB", "draft_capital_bucket": "round_1"},
                "x": {"position": "WR", "nfl_draft_pick": 22},
            }
        },
    }
    n = apply_nfl_draft_pick_overlay(data, {"picks": {"4034": 8, "x": 99}})
    assert n == 2
    assert data["cohort_index"]["observations"][0]["feats"]["nfl_draft_pick"] == 8
    assert data["cohort_index"]["observations"][1]["feats"]["nfl_draft_pick"] == 22
    assert data["preseason_profiles"]["by_player"]["4034"]["nfl_draft_pick"] == 8
    assert data["preseason_profiles"]["by_player"]["x"]["nfl_draft_pick"] == 22


def test_canonical_filter_key_sorts_mixed_eq_types():
    from dashboard_services.historical.filters import canonical_filter_key

    round_1 = {"group": "draft_capital", "field": "draft_capital", "eq": "round_1", "label": "NFL Round 1"}
    never = {"group": "never_elite", "field": "prior_top12_count", "eq": 0, "label": "Never top-12"}
    two_plus = {"group": "prior_top12", "field": "prior_top12_count", "gte": 2}
    key = canonical_filter_key([round_1, never])
    assert key == canonical_filter_key([never, round_1])
    assert key != canonical_filter_key([round_1, two_plus])
    top_10 = {"group": "draft_capital", "field": "nfl_draft_pick", "between": [1, 10], "label": "Top 10"}
    mixed = canonical_filter_key([top_10, never])
    assert mixed == canonical_filter_key([never, top_10])
    reset_cohort_cache()
    index = build_cohort_index(_combo_warehouse())
    out = evaluate_cohort(
        _baseline_aggs(index),
        position="WR",
        filters=[round_1, never],
        data_version="mix",
    )
    assert out["unknown_reason"] != "error"
    assert "sample_size" in out


def test_combined_rate_uses_matched_rows_not_multiplied_marginals():
    reset_cohort_cache()
    index = build_cohort_index(_combo_warehouse())
    aggs = _baseline_aggs(index, top12=0.08)
    out = evaluate_cohort(aggs, position="WR", filters=[AGE_23, DAY_2], data_version="t1")
    assert out["available"] is True
    assert out["descriptive_only"] is True
    assert out["not_in_ranking"] is True
    assert out["not_in_pick_score"] is True
    assert out["kind"] == "player_season"
    assert out["sample_size"] == 50
    assert out["n_players"] == 50
    assert out["successes"] == 5
    assert abs(out["raw_rate"] - 0.10) < 1e-9
    assert out["display_pct"] == 10
    # Age-only and capital-only are each 50%. Product would be 25%.
    age_only = evaluate_cohort(aggs, position="WR", filters=[AGE_23], data_version="t1")
    cap_only = evaluate_cohort(aggs, position="WR", filters=[DAY_2], data_version="t1")
    assert abs(age_only["raw_rate"] - 0.50) < 1e-9
    assert abs(cap_only["raw_rate"] - 0.50) < 1e-9
    product = age_only["raw_rate"] * cap_only["raw_rate"]
    assert abs(product - 0.25) < 1e-9
    assert abs(out["raw_rate"] - product) > 0.10
    # Top-5 / Top-24 also come from the same 50 matched rows.
    assert out["rates"]["top_5"]["sample_size"] == 50
    assert out["rates"]["top_24"]["sample_size"] == 50


def test_same_group_or_combines_target_share_buckets():
    reset_cohort_cache()
    rows = []
    for i in range(40):
        rows.append(_obs_row(
            sleeper_id=f"t{i}",
            age=23.2,
            draft_capital_bucket="day_2",
            previous_season_target_share=0.22 if i < 20 else 0.28,
            ppr_positional_finish=10 if i < 12 else 40,
        ))
    for i in range(20):
        rows.append(_obs_row(
            sleeper_id=f"u{i}",
            age=23.2,
            draft_capital_bucket="day_2",
            previous_season_target_share=0.12,
            ppr_positional_finish=10,
        ))
    index = build_cohort_index(rows)
    aggs = _baseline_aggs(index)
    out = evaluate_cohort(
        aggs,
        position="WR",
        filters=[AGE_23, DAY_2, TGT_20, TGT_25],
        data_version="t2",
    )
    assert out["sample_size"] == 40
    assert out["successes"] == 12
    low = evaluate_cohort(
        aggs,
        position="WR",
        filters=[AGE_23, DAY_2, {"group": "target_share", "field": "target_share", "eq": "10-15%"}],
        data_version="t2",
    )
    assert low["sample_size"] == 20


def test_shrinkage_ranks_large_sample_over_noisy_raw_lift():
    baseline = 0.08
    noisy = make_rate(5, 12, prior_rate=baseline)  # ~43% raw, +35-ish pts
    solid = make_rate(109, 420, prior_rate=baseline)  # ~26% raw, +18 pts
    noisy_b = edge_bundle(noisy, baseline)
    solid_b = edge_bundle(solid, baseline)
    assert noisy_b["raw_edge"] > solid_b["raw_edge"]
    assert solid_b["adjusted_edge"] > noisy_b["adjusted_edge"]
    # Documented n=84 / 31% vs 8% maps onto about +17 pts adjusted.
    documented = ranking_adjusted_rate(26, 84, 0.08)
    assert abs((documented - 0.08) * 100 - 17) < 1
    assert EDGE_RANK_PRIOR_N == 30
    assert DEFAULT_BAYES_PRIOR_N == 10


def test_wilson_interval_bounds_and_empty_n():
    lo, hi = wilson_interval(None, None)
    assert lo is None and hi is None
    lo, hi = wilson_interval(0, 0)
    assert lo is None and hi is None
    lo, hi = wilson_interval(50, 100)
    assert 0.0 <= lo < 0.50 < hi <= 1.0
    lo, hi = wilson_interval(0, 40)
    assert lo == 0.0
    assert hi < 0.12


def test_market_adjusted_averages_adp_bucket_probabilities():
    reset_cohort_cache()
    rows = []
    for i in range(20):
        rows.append(_obs_row(
            sleeper_id=f"m{i}",
            age=23.1,
            draft_capital_bucket="day_2",
            ppr_positional_finish=8 if i < 8 else 40,
            adp_overall=36.0,
            adp_bucket="round_3",
        ))
    for i in range(20):
        rows.append(_obs_row(
            sleeper_id=f"n{i}",
            age=23.1,
            draft_capital_bucket="day_2",
            ppr_positional_finish=8 if i < 8 else 40,
            adp_overall=8.0,
            adp_bucket="round_1",
        ))
    index = build_cohort_index(rows)
    aggs = _baseline_aggs(index)
    out = evaluate_cohort(aggs, position="WR", filters=[AGE_23, DAY_2], data_version="mkt")
    mkt = out["market"]
    assert mkt["unknown_reason"] is None
    assert mkt["n_with_adp"] == 40
    assert abs(mkt["expected_market_rate"] - 0.15) < 1e-9
    assert abs(mkt["observed_rate"] - 0.40) < 1e-9
    assert abs(mkt["market_adjusted_edge"] - 0.25) < 1e-9
    top5 = evaluate_cohort(
        aggs, position="WR", filters=[AGE_23, DAY_2], tier="top_5", data_version="mkt"
    )
    assert top5["market"]["unknown_reason"] == "market_rates_are_top_12_only"


def test_market_adjusted_omitted_when_adp_coverage_is_thin():
    reset_cohort_cache()
    rows = [
        _obs_row(
            sleeper_id=f"thin{i}",
            age=23.1,
            draft_capital_bucket="day_2",
            ppr_positional_finish=8,
            adp_overall=36.0 if i < 5 else None,
            adp_bucket="round_3" if i < 5 else None,
        )
        for i in range(20)
    ]
    aggs = _baseline_aggs(build_cohort_index(rows))
    out = evaluate_cohort(aggs, position="WR", filters=[AGE_23, DAY_2], data_version="thin")
    assert out["market"]["unknown_reason"] == "insufficient_historical_adp"
    assert out["market"]["expected_market_rate"] is None


def test_scout_matching_parity_with_historical_matching():
    warehouse = _obs_row(
        sleeper_id="live",
        age=23.6,
        draft_capital_bucket="day_2",
        previous_season_target_share=0.21,
    )
    preseason = {
        "position": "WR",
        "age": 23.6,
        "years_experience": 2,
        "draft_capital_bucket": "day_2",
        "previous_season_target_share": 0.21,
        "previous_season_finish": 20,
        "previous_season_year": 2025,
    }
    live_feats = extract_trend_features(preseason)
    hist_feats = extract_trend_features(warehouse)
    filters = [AGE_23, DAY_2, TGT_20]
    assert matches_filter_groups(live_feats, filters) is True
    assert matches_filter_groups(hist_feats, filters) is True
    assert live_feats["age_bucket"] == hist_feats["age_bucket"]
    assert live_feats["draft_capital"] == hist_feats["draft_capital"]
    assert live_feats["target_share"] == hist_feats["target_share"]


def test_missing_feature_stays_unknown_and_does_not_match():
    feats = extract_trend_features(_obs_row(previous_season_target_share=None))
    assert "target_share" not in feats
    assert matches_trend_filter(feats, TGT_20) is False
    assert matches_trend_filter({"age": None}, {"field": "age", "gte": 23}) is False
    assert matches_trend_filter({"age": 22}, {"field": "age", "lte": 24}) is True
    assert matches_trend_filter({"age": 25}, {"field": "age", "lte": 24}) is False


def test_trajectory_uses_only_pre_outcome_seasons():
    prev = {
        "season": 2022,
        "position": "WR",
        "target_share": 0.10,
        "snap_pct": 0.50,
        "targets": 40,
    }
    last = {
        "season": 2023,
        "position": "WR",
        "target_share": 0.18,
        "snap_pct": 0.70,
        "targets": 80,
    }
    outcome = {
        "season": 2024,
        "position": "WR",
        "target_share": 0.40,
        "snap_pct": 0.99,
        "targets": 200,
    }
    buckets = trajectory_buckets(prev, last, position="WR", current_season=2024)
    assert buckets["target_share_change"] == "+5 pts or more"
    assert buckets["snap_pct_change"] == "+15 pts or more"
    assert buckets["workload_change"] == "materially increased"
    # Outcome year as the "last" season is rejected when current_season is 2024.
    assert trajectory_buckets(last, outcome, position="WR", current_season=2024) == {}
    # Non-consecutive priors are omitted.
    gap = dict(prev)
    gap["season"] = 2021
    assert trajectory_buckets(gap, last, position="WR", current_season=2024) == {}
    # Missing values stay omitted, never 0.
    missing = dict(last)
    missing["target_share"] = None
    out = trajectory_buckets(prev, missing, position="WR", current_season=2024)
    assert "target_share_change" not in out


def test_player_season_denominator_is_not_unique_careers():
    reset_cohort_cache()
    rows = [
        _obs_row(sleeper_id="same", season=2023, age=23.1, ppr_positional_finish=8),
        _obs_row(sleeper_id="same", season=2024, age=24.1, ppr_positional_finish=40),
    ]
    out = evaluate_cohort(
        _baseline_aggs(build_cohort_index(rows)),
        position="WR",
        filters=[AGE_23, DAY_2],
        data_version="career",
    )
    assert out["sample_size"] == 2
    assert out["n_players"] == 1
    assert out["successes"] == 1
    assert out["kind"] == "player_season"


def test_closest_examples_are_a_subset_not_the_cohort():
    rows = [
        _obs_row(
            sleeper_id=f"ex{i}",
            name=f"Ex {i}",
            season=2020 + (i % 5),
            age=23.2,
            ppr_positional_finish=8 if i < 3 else 20,
            adp_overall=30.0,
        )
        for i in range(12)
    ]
    index = build_cohort_index(rows)
    matched = index["observations"]
    examples = closest_examples(matched, filters=[AGE_23, DAY_2], limit=8)
    assert 1 <= len(examples) <= 8
    for ex in examples:
        for trait in ex.get("traits") or []:
            assert "_" not in str(trait)
    summary = examples_summary(examples)
    assert summary["n"] == len(examples)
    assert summary["top_12"] == 3
    assert summary["top_5"] == 0
    assert "Top-5" in summary["label"]
    assert "Top-12" in summary["label"]
    assert "Top-24" in summary["label"]
    assert len(matched) == 12
    assert summary["n"] < len(matched)


def test_closest_example_traits_use_readable_bucket_labels():
    rows = [
        _obs_row(
            sleeper_id="saquon",
            name="Saquon Barkley",
            season=2023,
            years_experience=5,
            age=26.2,
            draft_capital_bucket="round_1",
            previous_season_finish=3,
        )
    ]
    examples = closest_examples(build_cohort_index(rows)["observations"])
    assert examples
    traits = examples[0]["traits"]
    assert "Year 6+" in traits
    assert "Round 1" in traits
    assert "Top 5" in traits
    assert all("_" not in str(t) for t in traits)
    assert "year_6_plus" not in traits
    assert "round_1" not in traits
    assert "top_5" not in traits
    day2 = closest_examples(build_cohort_index([
        _obs_row(
            sleeper_id="achane",
            name="De'Von Achane",
            years_experience=2,
            draft_capital_bucket="day_2",
            previous_season_finish=5,
        )
    ])["observations"])
    assert "Day 2" in day2[0]["traits"]
    assert "Year 3" in day2[0]["traits"]
    assert all("_" not in str(t) for t in day2[0]["traits"])


def test_closest_examples_mark_top5_top12_top24_hits():
    from dashboard_services.historical.cohorts import example_finish_hit

    assert example_finish_hit(1)["hit_tier"] == "top_5"
    assert example_finish_hit(5)["hit_label"] == "Top 5"
    assert example_finish_hit(9)["hit_tier"] == "top_12"
    assert example_finish_hit(12)["hits"]["top_12"] is True
    assert example_finish_hit(12)["hits"]["top_5"] is False
    assert example_finish_hit(20)["hit_label"] == "Top 24"
    assert example_finish_hit(35)["hit_tier"] == "miss"
    assert example_finish_hit(None)["hit_tier"] is None
    rows = [
        _obs_row(sleeper_id="a", ppr_positional_finish=1),
        _obs_row(sleeper_id="b", ppr_positional_finish=10),
        _obs_row(sleeper_id="c", ppr_positional_finish=22),
        _obs_row(sleeper_id="d", ppr_positional_finish=40),
    ]
    examples = closest_examples(build_cohort_index(rows)["observations"])
    by_id = {ex["sleeper_id"]: ex for ex in examples}
    assert by_id["a"]["hit_label"] == "Top 5"
    assert by_id["b"]["hit_label"] == "Top 12"
    assert by_id["c"]["hit_label"] == "Top 24"
    assert by_id["d"]["hit_label"] == "Outside top 24"
    summary = examples_summary(examples)
    assert summary["top_5"] == 1
    assert summary["top_12"] == 2
    assert summary["top_24"] == 3
    assert summary["label"] == "1/4 Top-5 · 2/4 Top-12 · 3/4 Top-24"


def test_index_trajectory_ignores_outcome_year_actuals():
    rows = [
        _obs_row(
            sleeper_id="traj",
            season=2022,
            target_share=0.10,
            snap_pct=0.50,
            targets=40,
            ppr_positional_finish=50,
            previous_season_year=2021,
        ),
        _obs_row(
            sleeper_id="traj",
            season=2023,
            target_share=0.18,
            snap_pct=0.70,
            targets=80,
            ppr_positional_finish=50,
            previous_season_year=2022,
        ),
        _obs_row(
            sleeper_id="traj",
            season=2024,
            target_share=0.99,
            snap_pct=0.99,
            targets=200,
            ppr_positional_finish=8,
            previous_season_year=2023,
            previous_season_target_share=0.18,
        ),
    ]
    obs_2024 = next(o for o in build_cohort_index(rows)["observations"] if o["season"] == 2024)
    assert obs_2024["feats"]["target_share_change"] == "+5 pts or more"
    assert obs_2024["feats"]["workload_change"] == "materially increased"
    obs_2023 = next(o for o in build_cohort_index(rows)["observations"] if o["season"] == 2023)
    assert "target_share_change" not in obs_2023["feats"]


def test_cohort_modules_stay_pure_and_off_ranking():
    hist = ROOT / "dashboard_services" / "historical"
    for name in ("cohorts.py", "filters.py"):
        text = (hist / name).read_text(encoding="utf-8")
        assert "import pandas" not in text
        assert "import nfl_data_py" not in text
        assert "import flask" not in text.lower()
        assert "read_parquet" not in text
        assert "load_player_history_df" not in text
        assert "static/pick_score" not in text
        assert "from utils.projection_resolver" not in text
    pick = (ROOT / "static" / "pick_score.js").read_text(encoding="utf-8")
    core = (ROOT / "static" / "draft_board_core.js").read_text(encoding="utf-8")
    grade = (ROOT / "utils" / "draft_grade.py").read_text(encoding="utf-8")
    assert "p_hit_pct" not in pick
    assert "historical" not in pick
    assert "p_hit_pct" not in core
    assert "/api/historical-cohort" not in core
    assert "p_hit_pct" not in grade
    assert "historical-cohort" not in grade
    bp = (ROOT / "routes" / "historical_api_bp.py").read_text(encoding="utf-8")
    assert "/api/historical-cohort" in bp
    assert "evaluate_cohort" in bp
    assert "read_parquet" not in bp


def test_overlay_stamps_live_trajectory_without_leaking_into_index_payload():
    from dashboard_services.historical.aggregates_store import _merge_cohort_index

    data = {
        "preseason_profiles": {
            "by_player": {"p1": {"position": "WR", "age": 24}},
        }
    }
    overlay = {
        "kind": "player_season",
        "observations": [{"pid": "h1", "pos": "WR", "feats": {"position": "WR"}}],
        "preseason_trajectory": {"p1": {"target_share_change": "+5 pts or more"}},
    }
    _merge_cohort_index(data, overlay)
    assert data["preseason_profiles"]["by_player"]["p1"]["target_share_change"] == "+5 pts or more"
    assert "preseason_trajectory" not in data["cohort_index"]
    assert data["cohort_index"]["observations"]


def test_live_overlay_combined_cohort_is_an_intersection():
    from dashboard_services.historical.aggregates_store import load_profile_aggregates

    reset_cohort_cache()
    aggs = load_profile_aggregates()
    assert ((aggs.get("cohort_index") or {}).get("observations")), "cohort overlay missing"
    age = evaluate_cohort(aggs, position="WR", filters=[AGE_23], data_version="live")
    cap = evaluate_cohort(aggs, position="WR", filters=[DAY_2], data_version="live")
    both = evaluate_cohort(aggs, position="WR", filters=[AGE_23, DAY_2], data_version="live")
    assert both["available"] is True
    assert both["sample_size"] > 0
    assert both["n_players"] <= both["sample_size"]
    assert both["sample_size"] <= age["sample_size"]
    assert both["sample_size"] <= cap["sample_size"]
    assert both["raw_rate"] == both["rates"]["top_12"]["raw_rate"]
    assert both["ci_low"] is not None
    feats = {}
    from dashboard_services.historical.board import build_player_feature_index
    feats = build_player_feature_index(aggs)
    assert any(f.get("target_share_change") or f.get("workload_change") for f in feats.values())
    assert "observations" not in json.dumps(feats)
    love = feats.get("13287") or {}
    assert love.get("draft_capital") == "round_1"
    assert love.get("career_stage") == "rookie"
    assert love.get("position") == "RB"
