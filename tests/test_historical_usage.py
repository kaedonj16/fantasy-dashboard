"""Previous-season usage overlay, leakage, and two-stat rates (slim CI)."""
from copy import deepcopy
from pathlib import Path

from dashboard_services.historical.definitions import SNAP_RELIABLE_FLOOR, TARGET_SHARE_BUCKETS
from dashboard_services.historical.finishes import OUTCOME_COLUMNS, prior_career_features_for_player
from dashboard_services.historical.usage import (
    apply_efficiency_overlay,
    prior_usage_features,
    prior_usage_window_pair,
    build_prior_usage_rates,
)


ROOT = Path(__file__).resolve().parents[1]


def test_overlay_fills_missing_and_does_not_overwrite():
    row = {
        "sleeper_id": "x",
        "season": 2024,
        "games": 17,
        "targets": 137,
        "snap_pct": None,
        "adot": None,
        "air_yards": None,
        "avg_off_snap_pct": 0.0,
    }
    out = apply_efficiency_overlay(row, {
        "snap_pct": 0.82,
        "snaps": 900,
        "ngs_avg_intended_air_yards": 11.4,
        "ngs_avg_separation": 3.1,
        "ngs_avg_cushion": 5.8,
    })
    assert out["snap_pct"] == 0.82
    assert out["adot"] == 11.4
    assert out["air_yards"] == round(11.4 * 137, 1)
    assert out["ngs_created_separation"] == round(3.1 - 5.8, 2)
    # Live-valuation avg_* column is not rewritten.
    assert out["avg_off_snap_pct"] == 0.0
    kept = apply_efficiency_overlay({**row, "adot": 9.0, "snap_pct": 0.55}, {
        "adot": 11.4,
        "snap_pct": 0.99,
    })
    assert kept["adot"] == 9.0
    assert kept["snap_pct"] == 0.55


def test_overlay_rejects_zero_snap_with_real_volume():
    row = {"games": 16, "targets": 80, "carries": 50, "snap_pct": None}
    out = apply_efficiency_overlay(row, {"snap_pct": 0.0, "snaps": 0})
    assert out["snap_pct"] is None
    assert out.get("snaps") is None


def test_prior_usage_is_none_not_zero_without_previous_season():
    feats = prior_usage_features(None)
    assert feats["previous_season_year"] is None
    assert feats["previous_season_target_share"] is None
    assert feats["previous_season_snap_pct"] is None
    assert feats["previous_season_touches"] is None
    assert feats["previous_season_games"] is None


def test_prior_usage_does_not_leak_same_season_actuals():
    career = [
        {
            "sleeper_id": "wr",
            "season": 2023,
            "position": "WR",
            "ppr_points": 200,
            "ppr_ppg": 12.0,
            "games": 16,
            "ppr_positional_finish": 20,
            "target_share": 0.18,
            "snap_pct": 0.70,
            "adot": 10.0,
        },
        {
            "sleeper_id": "wr",
            "season": 2024,
            "position": "WR",
            "ppr_points": 280,
            "ppr_ppg": 16.0,
            "games": 17,
            "ppr_positional_finish": 8,
            "target_share": 0.28,
            "snap_pct": 0.90,
            "adot": 12.0,
        },
    ]
    rows = prior_career_features_for_player(career)
    y2024 = {r["season"]: r for r in rows}[2024]
    assert y2024["previous_season_year"] == 2023
    assert y2024["previous_season_target_share"] == 0.18
    assert y2024["previous_season_snap_pct"] == 0.70
    assert y2024["target_share"] == 0.28  # outcome on this row
    y2024_original = dict(y2024)
    mutated = deepcopy(career)
    mutated[1]["target_share"] = 0.99
    mutated[1]["snap_pct"] = 0.10
    mutated[1]["ppr_positional_finish"] = 80
    after = {r["season"]: r for r in prior_career_features_for_player(mutated)}[2024]
    for key, value in y2024_original.items():
        if key in OUTCOME_COLUMNS:
            continue
        assert after[key] == value, f"feature leaked 2024 actuals: {key}"
    assert after["previous_season_target_share"] == 0.18
    # 2023 features must not see 2023 actuals as "previous".
    y2023 = {r["season"]: r for r in rows}[2023]
    assert y2023["previous_season_target_share"] is None


def test_usage_distribution_vs_conditional_are_different():
    rows = []
    # Previous 25%+ share: 8 hits / 10 seasons → 80% conditional.
    for i in range(8):
        rows.append({
            "sleeper_id": f"h-{i}",
            "season": 2024,
            "position": "WR",
            "previous_season_year": 2023,
            "previous_season_target_share": 0.28,
            "ppr_positional_finish": 4,
        })
    for i in range(2):
        rows.append({
            "sleeper_id": f"m-high-{i}",
            "season": 2024,
            "position": "WR",
            "previous_season_year": 2023,
            "previous_season_target_share": 0.30,
            "ppr_positional_finish": 40,
        })
    # Previous <10% share: 2 hits / 20 seasons. 2 of 10 WR1s come from this bucket.
    for i in range(2):
        rows.append({
            "sleeper_id": f"h-low-{i}",
            "season": 2024,
            "position": "WR",
            "previous_season_year": 2023,
            "previous_season_target_share": 0.05,
            "ppr_positional_finish": 6,
        })
    for i in range(18):
        rows.append({
            "sleeper_id": f"m-low-{i}",
            "season": 2024,
            "position": "WR",
            "previous_season_year": 2023,
            "previous_season_target_share": 0.04,
            "ppr_positional_finish": 50,
        })
    pair_high = prior_usage_window_pair(
        rows, "WR", "previous_season_target_share", "25%+", TARGET_SHARE_BUCKETS
    )
    pair_low = prior_usage_window_pair(
        rows, "WR", "previous_season_target_share", "<10%", TARGET_SHARE_BUCKETS
    )
    assert pair_high["conditional"]["successes"] == 8
    assert pair_high["conditional"]["sample_size"] == 10
    assert abs(pair_high["conditional"]["raw_rate"] - 0.8) < 1e-9
    assert pair_high["distribution"]["count"] == 8
    assert pair_high["distribution"]["total"] == 10
    assert abs(pair_high["distribution"]["share"] - 0.8) < 1e-9
    # Low bucket: conditional 2/20 = 10%, distribution 2/10 = 20%.
    assert abs(pair_low["conditional"]["raw_rate"] - 0.10) < 1e-9
    assert abs(pair_low["distribution"]["share"] - 0.20) < 1e-9
    assert pair_low["distribution"]["share"] != pair_low["conditional"]["raw_rate"]


def test_missing_prior_usage_is_not_a_zero_bucket():
    rows = [
        {
            "sleeper_id": "known",
            "season": 2024,
            "position": "RB",
            "previous_season_year": 2023,
            "previous_season_target_share": 0.22,
            "ppr_positional_finish": 3,
        },
        {
            "sleeper_id": "unknown",
            "season": 2024,
            "position": "RB",
            "previous_season_year": 2023,
            "previous_season_target_share": None,
            "ppr_positional_finish": 1,
        },
    ]
    rates = build_prior_usage_rates(rows)
    rb = rates["target_share"]["by_position"]["RB"]
    assert rb["n_known"] == 1
    assert rb["n_missing_excluded"] == 1
    assert rb["by_bucket"]["20-25%"]["conditional"]["sample_size"] == 1
    assert rb["by_bucket"]["<10%"]["conditional"]["sample_size"] == 0
    assert rb["by_bucket"]["<10%"]["conditional"]["raw_rate"] is None
    assert "QB" not in rates["target_share"]["by_position"]


def test_snap_rates_require_reliable_prior_season():
    rows = [
        {
            "sleeper_id": "old",
            "season": 2021,
            "position": "RB",
            "previous_season_year": 2020,
            "previous_season_snap_pct": 0.85,
            "ppr_positional_finish": 2,
        },
        {
            "sleeper_id": "new",
            "season": 2024,
            "position": "RB",
            "previous_season_year": 2023,
            "previous_season_snap_pct": 0.85,
            "ppr_positional_finish": 2,
        },
    ]
    rates = build_prior_usage_rates(rows)
    rb = rates["snap_pct"]["by_position"]["RB"]
    assert SNAP_RELIABLE_FLOOR == 2022
    assert rb["n_known"] == 1
    assert rb["by_bucket"]["80%+"]["conditional"]["sample_size"] == 1


def test_prior_usage_stamps_touches_from_carries_and_receptions():
    feats = prior_usage_features({
        "season": 2023,
        "carries": 320,
        "receptions": 90,
        "targets": 110,
        "games": 17,
    })
    assert feats["previous_season_carries"] == 320
    assert feats["previous_season_receptions"] == 90
    assert feats["previous_season_touches"] == 410
    assert feats["previous_season_games"] == 17
    empty = prior_usage_features({"season": 2023, "target_share": 0.2})
    assert empty["previous_season_touches"] is None


def test_touches_rates_skip_missing_and_flag_workhorse_cliff():
    rows = [
        {
            "sleeper_id": "work",
            "season": 2024,
            "position": "RB",
            "previous_season_year": 2023,
            "previous_season_carries": 320,
            "previous_season_receptions": 90,
            "ppr_positional_finish": 18,
        },
        {
            "sleeper_id": "committee",
            "season": 2024,
            "position": "RB",
            "previous_season_year": 2023,
            "previous_season_carries": 140,
            "previous_season_receptions": 20,
            "ppr_positional_finish": 8,
        },
        {
            "sleeper_id": "unknown",
            "season": 2024,
            "position": "RB",
            "previous_season_year": 2023,
            "ppr_positional_finish": 1,
        },
    ]
    rates = build_prior_usage_rates(rows)
    rb = rates["touches"]["by_position"]["RB"]
    assert rb["n_known"] == 2
    assert rb["n_missing_excluded"] == 1
    assert rb["by_bucket"]["400+"]["conditional"]["sample_size"] == 1
    assert rb["by_bucket"]["<200"]["conditional"]["sample_size"] == 1
    assert rb["by_bucket"]["200-299"]["conditional"]["sample_size"] == 0
    assert "WR" not in rates["touches"]["by_position"]
    rec = rates["receptions"]["by_position"]["WR"]
    assert rec["n_known"] == 0


def test_usage_volume_overlay_keeps_share_tables_intact():
    from dashboard_services.historical.aggregates_store import _merge_usage_volume_overlay
    from dashboard_services.historical.usage import build_usage_volume_overlay

    aggs = {
        "prior_usage": {
            "target_share": {"keep": True},
            "touches": {"stale": True},
        },
        "prior_usage_by_tier": {
            "top_12": {"target_share": {"keep": True}},
        },
        "preseason_profiles": {
            "by_player": {
                "1": {"position": "RB", "previous_season_target_share": 0.2},
            }
        },
    }
    overlay = build_usage_volume_overlay([
        {
            "sleeper_id": "1",
            "season": 2025,
            "position": "RB",
            "carries": 250,
            "receptions": 40,
            "games": 16,
            "previous_season_year": 2024,
            "previous_season_carries": 250,
            "previous_season_receptions": 40,
            "ppr_positional_finish": 10,
        }
    ])
    _merge_usage_volume_overlay(aggs, overlay)
    assert aggs["prior_usage"]["target_share"] == {"keep": True}
    assert aggs["prior_usage"]["touches"]["by_position"]["RB"]["n_known"] >= 1
    assert aggs["prior_usage_by_tier"]["top_12"]["target_share"] == {"keep": True}
    rec = aggs["preseason_profiles"]["by_player"]["1"]
    assert rec["previous_season_target_share"] == 0.2
    assert rec["previous_season_touches"] == 290
    assert rec["previous_season_carries"] == 250


def test_usage_modules_stay_pure_and_do_not_estimate_snaps():
    hist = ROOT / "dashboard_services" / "historical"
    for name in ("usage.py", "seasons.py", "finishes.py", "comps.py", "adp.py"):
        text = (hist / name).read_text(encoding="utf-8")
        assert "import pandas" not in text
        assert "import nfl_data_py" not in text
        assert "estimate_snap_share_from_usage" not in text
    io_text = (ROOT / "data_building" / "historical" / "build_usage_efficiency.py").read_text(
        encoding="utf-8"
    )
    assert "estimate_snap_share_from_usage" not in io_text
