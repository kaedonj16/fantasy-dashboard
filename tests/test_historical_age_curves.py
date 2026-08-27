"""Age-curve stats: distribution vs conditional stay distinct (slim CI)."""
from pathlib import Path

from dashboard_services.historical.age_curves import (
    MIN_PRIME_N,
    age_window_pair,
    build_age_curves,
    derive_prime_window,
    rows_with_known_age,
)
from dashboard_services.historical.finish_rates import filter_era, make_rate


def _rb(season, age, finish, pid=None, **extra):
    row = {
        "sleeper_id": pid or f"rb-{season}-{age}-{finish}",
        "season": season,
        "position": "RB",
        "age": age,
        "ppr_positional_finish": finish,
        "games": 16,
    }
    row.update(extra)
    return row


def test_distribution_and_conditional_are_different_stats():
    """40% of RB1s at 24 ≠ 20% of 24-year-old RBs finishing RB1."""
    rows = []
    # Age 24: 8 hits out of 40 seasons → conditional 20%.
    for i in range(8):
        rows.append(_rb(2020, 24.1, 3, pid=f"h24-{i}"))
    for i in range(32):
        rows.append(_rb(2020, 24.4, 30, pid=f"m24-{i}"))
    # Age 30: 12 hits out of 12 → conditional 100%.
    for i in range(12):
        rows.append(_rb(2021, 30.2, 2, pid=f"h30-{i}"))

    pair = age_window_pair(rows, "RB", 23, 27, tier="top_12")
    # Distribution: 8 of 20 RB1 seasons are in 23-27.
    assert pair["distribution"]["count"] == 8
    assert pair["distribution"]["total"] == 20
    assert abs(pair["distribution"]["share"] - 0.4) < 1e-9
    assert pair["distribution"]["kind"] == "distribution"
    # Conditional: 8 of 40 age-23-27 seasons hit RB1.
    assert pair["conditional"]["successes"] == 8
    assert pair["conditional"]["sample_size"] == 40
    assert abs(pair["conditional"]["raw_rate"] - 0.2) < 1e-9
    assert pair["distribution"]["share"] != pair["conditional"]["raw_rate"]


def test_missing_age_is_skipped_not_treated_as_zero():
    rows = [
        _rb(2020, 24.0, 1, pid="known-hit"),
        _rb(2020, 24.0, 40, pid="known-miss"),
        _rb(2020, None, 1, pid="missing-hit"),
        _rb(2020, None, 40, pid="missing-miss"),
    ]
    aged = rows_with_known_age(rows)
    assert {r["sleeper_id"] for r in aged} == {"known-hit", "known-miss"}
    pair = age_window_pair(rows, "RB", 23, 27)
    assert pair["n_known_age"] == 2
    assert pair["distribution"]["total"] == 1  # only known-age hit
    assert pair["conditional"]["sample_size"] == 2


def test_prime_window_is_data_derived_not_hardcoded_23_27():
    """High hit rate at 30-32 with n>=15 each; 23-27 is cold. Prime ≠ 23-27."""
    rows = []
    for age in (23, 24, 25, 26, 27):
        rows.append(_rb(2020, float(age), 1, pid=f"young-hit-{age}"))
        for i in range(14):
            rows.append(_rb(2020, float(age), 40, pid=f"young-miss-{age}-{i}"))
    for age in (30, 31, 32):
        for i in range(10):
            rows.append(_rb(2021, float(age), 2, pid=f"old-hit-{age}-{i}"))
        for i in range(5):
            rows.append(_rb(2021, float(age), 40, pid=f"old-miss-{age}-{i}"))

    curves = build_age_curves(rows, season_from=2016)
    rb = curves["RB"]
    prime = rb["prime_window"]
    assert prime is not None
    assert prime["age_start"] == 30
    assert prime["age_end"] == 32
    assert prime["ages"] == [30, 31, 32]
    assert 23 not in prime["ages"]
    # The pair on the derived window must expose both stats.
    pair = rb["prime_window_pair"]
    assert pair["age_lo"] == 30 and pair["age_hi"] == 32
    assert "distribution" in pair and "conditional" in pair
    assert pair["distribution"]["kind"] == "distribution"
    assert "kind" not in pair["conditional"]


def test_derive_prime_window_requires_moderate_n():
    by_age = {
        24: {
            "conditional": make_rate(8, 10, prior_rate=0.2),
        },
        25: {
            "conditional": make_rate(9, 10, prior_rate=0.2),
        },
    }
    # n=10 is "low" — not prime even though the rate beats baseline.
    assert derive_prime_window(by_age, 0.2, min_n=MIN_PRIME_N) is None
    assert derive_prime_window(by_age, 0.2, min_n=5)["ages"] == [24, 25]


def test_era_filter_drops_pre_floor_seasons():
    rows = [
        _rb(2015, 24.0, 1, pid="old"),
        _rb(2020, 24.0, 1, pid="new-hit"),
        _rb(2020, 24.0, 40, pid="new-miss"),
    ]
    era = filter_era(rows, season_from=2016)
    assert all(r["season"] >= 2016 for r in era)
    curves = build_age_curves(rows, season_from=2016)
    assert curves["RB"]["n_known_age"] == 2
    assert "2015" not in str(curves["RB"]["baseline"]["season_range"] or [])


def test_age_curve_modules_stay_pure():
    root = Path(__file__).resolve().parents[1]
    for name in ("age_curves.py", "finish_rates.py", "career_profiles.py", "definitions.py", "usage.py", "walkforward.py"):
        text = (root / "dashboard_services" / "historical" / name).read_text(encoding="utf-8")
        assert "import pandas" not in text
        assert "import flask" not in text.lower()
        if name in ("career_profiles.py", "walkforward.py"):
            assert "from data_building.breakout_engine" not in text
            assert "import data_building.breakout_engine" not in text
