"""Pure aggregation of weekly scoring-variant projections into season PPG."""
from data_building.fetch_projections import season_ppg_from_weekly, weekly_variant_values


def test_weekly_variant_values_skips_byes_and_falls_back_to_ppr():
    weeks = [
        {"1": {"ppr": 12.0, "half_ppr": 10.0, "6pt_ppr": 14.0}},
        {"1": {"ppr": 0.0}},  # bye
        {"1": {"ppr": 18.0}},  # missing half/6pt -> ppr fallback
    ]
    collected = weekly_variant_values(weeks)
    assert collected["1"]["ppr"] == [12.0, 18.0]
    assert collected["1"]["half_ppr"] == [10.0, 18.0]
    assert collected["1"]["6pt_ppr"] == [14.0, 18.0]


def test_season_ppg_is_median_per_variant():
    weeks = [
        {"wr": {"ppr": 20.0, "half_ppr": 16.0, "std": 12.0, "6pt_ppr": 20.0}},
        {"wr": {"ppr": 16.0, "half_ppr": 12.0, "std": 8.0, "6pt_ppr": 16.0}},
        {"qb": {"ppr": 22.0, "half_ppr": 22.0, "6pt_ppr": 26.0}},
        {"qb": {"ppr": 18.0, "half_ppr": 18.0, "6pt_ppr": 22.0}},
    ]
    out = season_ppg_from_weekly(weeks)
    assert out["wr"]["ppr"] == 18.0
    assert out["wr"]["half_ppr"] == 14.0
    assert out["wr"]["std"] == 10.0
    assert out["qb"]["ppr"] == 20.0
    assert out["qb"]["6pt_ppr"] == 24.0
