"""Pure aggregation of weekly scoring-variant projections into season PPG."""
from data_building.fetch_projections import (
    _published_base_points,
    fill_missing_from_season_totals,
    fill_missing_ppg_variants,
    season_ppg_from_weekly,
    weekly_variant_values,
)


def test_published_base_points_reads_sleeper_totals():
    raw = {"rec": 6, "rec_yd": 80, "pts_ppr": 21.7, "pts_half_ppr": 18.7, "pts_std": 15.7}
    assert _published_base_points(raw) == {"ppr": 21.7, "half_ppr": 18.7, "std": 15.7}
    # No raw line / no precomputed totals -> nothing to prefer.
    assert _published_base_points(None) == {}
    assert _published_base_points({"rec": 6}) == {}


def test_weekly_variant_values_prefers_published_totals():
    # The recompute stored a lower ppr (15.76) than Sleeper's own published
    # total (17.15); trust Sleeper for the plain PPR/half/std variants while the
    # 6pt layer keeps the computed value.
    weeks = [
        {"6786": {
            "ppr": 15.76, "half_ppr": 13.07, "std": 10.38, "6pt_ppr": 15.76,
            "raw_stats": {"rec": 6, "pts_ppr": 17.15, "pts_half_ppr": 14.46, "pts_std": 11.77},
        }},
    ]
    collected = weekly_variant_values(weeks)
    assert collected["6786"]["ppr"] == [17.15]
    assert collected["6786"]["half_ppr"] == [14.46]
    assert collected["6786"]["std"] == [11.77]
    assert collected["6786"]["6pt_ppr"] == [15.76]  # no published equivalent


def test_weekly_variant_values_falls_back_without_raw_stats():
    # Legacy cache rows (no raw_stats) still use the stored computed variants.
    weeks = [{"6786": {"ppr": 15.76, "half_ppr": 13.07, "std": 10.38}}]
    collected = weekly_variant_values(weeks)
    assert collected["6786"]["ppr"] == [15.76]
    assert collected["6786"]["half_ppr"] == [13.07]


def test_season_ppg_uses_published_totals_when_present():
    weeks = [
        {"6786": {"ppr": 15.0, "raw_stats": {"pts_ppr": 17.15, "pts_half_ppr": 14.46, "pts_std": 11.77}}},
        {"6786": {"ppr": 15.0, "raw_stats": {"pts_ppr": 17.15, "pts_half_ppr": 14.46, "pts_std": 11.77}}},
    ]
    out = season_ppg_from_weekly(weeks)
    assert out["6786"]["ppr"] == 17.15
    assert out["6786"]["std"] == 11.77


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
