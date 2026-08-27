"""Warehouse rebuild from synthetic usage rows (pandas I/O — integration job)."""
import pytest

pd = pytest.importorskip("pandas")
ph = pytest.importorskip("data_building.external_data.player_history")

build_canonical_player_history_for_season = ph.build_canonical_player_history_for_season
build_prior_career_features = ph.build_prior_career_features


def _legacy(pid, pos, points, games=16, finish_targets=40):
    return {
        "id": pid,
        "position": pos,
        "name": f"Player {pid}",
        "usage": {
            "gsis_id": f"00-{pid.zfill(8)}",
            "games": games,
            "targets": finish_targets,
            "carries": 100 if pos == "RB" else 0,
            "receptions": finish_targets - 5,
            "rec_yards": 400,
            "rec_tds": 3,
            "rush_yards": 800 if pos == "RB" else 0,
            "rush_tds": 6 if pos == "RB" else 0,
            "ppr_total": points,
            "ppr_ppg": points / games,
        },
    }


def test_canonical_df_keeps_nulls_and_skips_non_skill():
    rows = [
        _legacy("1", "RB", 250),
        {"id": "2", "position": "PK", "name": "Kicker", "usage": {"games": 17, "ppr_ppg": 9}},
        {"id": "3", "position": "WR", "name": "Ghost", "usage": {}},
    ]
    identity = {
        "1": {"birth_date": "1/1/1998", "draft_year": 2020, "nfl_draft_round": 2, "nfl_draft_pick": 45},
        "3": {"birth_date": "5/5/1999"},
    }
    df = build_canonical_player_history_for_season(2022, rows, identity)
    assert set(df["sleeper_id"]) == {"1"}
    rb = df.set_index("sleeper_id").loc["1"]
    assert rb["ppr_points"] == 250
    assert rb["avg_targets"] == 40 / 16
    assert "avg_off_snap_pct" in df.columns
    assert rb["air_yards"] is None or (isinstance(rb["air_yards"], float) and pd.isna(rb["air_yards"]))
    assert "projected_points" not in df.columns


def test_prior_career_pandas_wrapper_matches_pure_function():
    records = []
    for season, points, finish in ((2021, 200, 10), (2022, 240, 6), (2023, 90, 30)):
        records.append({
            "sleeper_id": "x",
            "season": season,
            "position": "WR",
            "ppr_points": points,
            "ppr_ppg": points / 16,
            "games": 16,
            "ppr_positional_finish": finish,
            "half_ppr_points": points - 20,
            "half_ppr_ppg": (points - 20) / 16,
            "half_ppr_positional_finish": finish,
            "standard_points": points - 40,
            "standard_ppg": (points - 40) / 16,
            "standard_positional_finish": finish,
        })
    df = pd.DataFrame(records)
    out = build_prior_career_features(df)
    assert len(out) == 3
    row = out.set_index("season").loc[2022]
    assert int(row["previous_season_finish"]) == 10
    assert int(row["prior_top12_count"]) == 1
    # Live valuation collapse is a separate function and still one-row-per-player
    # on the legacy avg_* schema; this wrapper must not collapse.
    assert set(out["season"]) == {2021, 2022, 2023}


def test_rebuild_from_injected_maps_does_not_scan_live_apis(monkeypatch, tmp_path):
    from data_building.historical import build_player_seasons as B

    usage = {
        2021: [_legacy("10", "RB", 180), _legacy("11", "WR", 220)],
        2022: [_legacy("10", "RB", 260), _legacy("11", "WR", 90)],
    }
    monkeypatch.setattr(B, "load_usage_rows", lambda season: usage.get(season, []))
    monkeypatch.setattr(B, "build_identity_map", lambda: {
        "10": {"birth_date": "3/3/1997", "draft_year": 2019, "nfl_draft_round": 1, "nfl_draft_pick": 8, "name": "RB Ten", "position": "RB"},
        "11": {"birth_date": "4/4/1998", "draft_year": 2020, "nfl_draft_round": 2, "nfl_draft_pick": 40, "name": "WR Eleven", "position": "WR"},
    })
    coverage = B.rebuild_historical_warehouse(seasons=(2021, 2022), write=False)
    rows = coverage["rows"]
    assert coverage["combined_rows"] == 4
    y2022_rb = next(r for r in rows if r["sleeper_id"] == "10" and r["season"] == 2022)
    assert y2022_rb["previous_season_finish"] == 1  # only RB in 2021
    assert y2022_rb["ppr_positional_finish"] == 1
    assert y2022_rb["years_experience"] == 3
    assert y2022_rb["ppr_top_5"] is True
    assert y2022_rb["previously_top5"] is True  # 2021 was the only RB, finish 1
    # Current-season projections must not appear on historical rows.
    assert "projected_points" not in y2022_rb
    assert "projected_ppg" not in y2022_rb


def test_efficiency_overlay_fills_adot_without_touching_avg_snap(monkeypatch):
    from data_building.historical import build_player_seasons as B

    usage = {2022: [_legacy("10", "RB", 260)]}
    monkeypatch.setattr(B, "load_usage_rows", lambda season: usage.get(season, []))
    monkeypatch.setattr(B, "build_identity_map", lambda: {
        "10": {"birth_date": "3/3/1997", "draft_year": 2019, "nfl_draft_round": 1, "nfl_draft_pick": 8},
    })
    monkeypatch.setattr(B, "load_efficiency_overlay", lambda season: {
        "10": {"snap_pct": 0.77, "ngs_avg_intended_air_yards": 4.2, "ngs_avg_separation": 2.0, "ngs_avg_cushion": 6.0},
    })
    rows = B.rebuild_historical_warehouse(seasons=(2022,), write=False)["rows"]
    rb = rows[0]
    assert abs(rb["snap_pct"] - 0.77) < 1e-9
    assert rb["adot"] == 4.2
    assert rb["ngs_created_separation"] == round(2.0 - 6.0, 2)
    assert "projected_points" not in rb



def test_profiles_rebuild_from_warehouse_rows_write_false(monkeypatch):
    from data_building.historical.build_profiles import rebuild_historical_profiles
    from data_building.historical import build_player_seasons as B

    usage = {
        2021: [_legacy("10", "RB", 180), _legacy("11", "WR", 220)],
        2022: [_legacy("10", "RB", 260), _legacy("11", "WR", 90)],
    }
    monkeypatch.setattr(B, "load_usage_rows", lambda season: usage.get(season, []))
    monkeypatch.setattr(B, "build_identity_map", lambda: {
        "10": {"birth_date": "3/3/1997", "draft_year": 2019, "nfl_draft_round": 1, "nfl_draft_pick": 8, "name": "RB Ten", "position": "RB"},
        "11": {"birth_date": "4/4/1998", "draft_year": 2020, "nfl_draft_round": 2, "nfl_draft_pick": 40, "name": "WR Eleven", "position": "WR"},
    })
    rows = B.rebuild_historical_warehouse(seasons=(2021, 2022), write=False)["rows"]
    payload = rebuild_historical_profiles(rows, write=False)
    assert payload["written_path"] is None
    assert payload["n_player_seasons"] == 4
    assert "distribution" in payload["age_curves"]["RB"]["by_integer_age"]["24"]
    assert "conditional" in payload["age_curves"]["RB"]["by_integer_age"]["24"]
    assert payload["definitions"]["no_adp"] is False
    assert payload["phase"] == 9
    assert "prior_usage" in payload
    assert "comps" in payload
    assert "adp" in payload
    assert payload["signals"]["no_blended_score"] is True
    assert payload["board"]["not_in_ranking"] is True
    assert payload["walkforward"]["not_a_second_engine"] is True
    assert payload["walkforward"]["pick_score"]["in_live_ranking"] is False
    assert payload["definitions"]["pick_score_in_live_ranking"] is False
    assert "preseason_profiles" in payload
    assert payload["adp"]["sf_tep_historical"] is False
    assert payload["comps"]["walk_forward"] is False
    assert payload["comps"]["pooled_historical"] is True
    assert payload["definitions"]["adp_in_comps"] is False
    assert "adp" not in payload["draft_capital"]
