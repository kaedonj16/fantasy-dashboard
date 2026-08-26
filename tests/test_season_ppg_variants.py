"""Pure aggregation of weekly scoring-variant projections into season PPG."""
from data_building.fetch_projections import (
    fill_missing_from_season_totals,
    fill_missing_ppg_variants,
    season_ppg_from_weekly,
    weekly_variant_values,
)


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


def test_season_totals_fill_players_weekly_files_omitted():
    weekly = {
        "9224": {"pos": "RB", "season_pts": 14.9, "ppg": 14.9},
    }
    season = {
        "5859": {"pts_ppr": 247.2, "pts_half_ppr": 207.7, "pts_std": 168.2, "gp": 18.0},
        "9224": {"pts_ppr": 255.2, "gp": 18.0},
    }
    out = fill_missing_from_season_totals(
        weekly, season, scoring="ppr", players_index={"5859": {"pos": "WR"}},
    )
    assert out["9224"]["ppg"] == 14.9
    assert out["9224"]["season_pts"] == 14.9
    assert out["5859"]["pos"] == "WR"
    assert out["5859"]["season_pts"] == 247.2
    assert out["5859"]["ppg"] == round(247.2 / 17.0, 2)


def test_season_totals_do_not_overwrite_weekly_median():
    weekly = {"5859": {"pos": "WR", "season_pts": 200.0, "ppg": 16.8}}
    season = {"5859": {"pts_ppr": 247.2, "gp": 18.0}}
    out = fill_missing_from_season_totals(weekly, season)
    assert out["5859"]["ppg"] == 16.8
    assert out["5859"]["season_pts"] == 200.0


def test_season_variant_fill_skips_players_already_on_weekly():
    weekly = {"9224": {"ppr": 14.9}}
    season = {
        "9224": {"pts_ppr": 255.2, "pts_half_ppr": 220.0, "gp": 18.0},
        "5859": {"pts_ppr": 247.2, "pts_half_ppr": 207.7, "pts_std": 168.2, "gp": 18.0},
    }
    out = fill_missing_ppg_variants(weekly, season)
    assert out["9224"] == {"ppr": 14.9}
    assert out["5859"]["ppr"] == round(247.2 / 17.0, 2)
    assert out["5859"]["half_ppr"] == round(207.7 / 17.0, 2)
    assert out["5859"]["std"] == round(168.2 / 17.0, 2)
