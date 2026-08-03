"""Preseason usage-table validation must accept the (correct) all-zero-games table.

In August the NFL season_type is "pre" and every player legitimately has 0 games
played. The validator used to treat only "off" as a no-games-expected state, so
in preseason it fell into the strict in-season branch and raised on the
100%-zero-games table. That stopped usage_table.json from being written, which in
turn made rewrite_value_table_with_model fail (FileNotFoundError) — freezing all
player values on the last good model_values.json until Week 1.
"""
import pytest

su = pytest.importorskip("data_building.external_data.sleeper_usage")


def _rows(n=500, games=0, ppg=0.0):
    return [{"id": str(i), "usage": {"games": games, "ppr_ppg": ppg}} for i in range(n)]


def _set_season_type(monkeypatch, season_type):
    # get_nfl_state is imported inside _validate_usage_table, so patch the source.
    import dashboard_services.api as api
    monkeypatch.setattr(api, "get_nfl_state", lambda: {"season_type": season_type})


def test_preseason_all_zero_games_is_accepted(monkeypatch):
    _set_season_type(monkeypatch, "pre")
    # 100% zero games is correct in preseason and must NOT raise.
    su._validate_usage_table(_rows(games=0, ppg=0.0), {}, 2026)


def test_offseason_all_zero_games_is_accepted(monkeypatch):
    _set_season_type(monkeypatch, "off")
    su._validate_usage_table(_rows(games=0, ppg=0.0), {}, 2026)


def test_in_season_all_zero_games_is_rejected(monkeypatch):
    # Mid-season, 100% zero games really does mean a broken fetch — keep raising.
    _set_season_type(monkeypatch, "regular")
    with pytest.raises(ValueError):
        su._validate_usage_table(_rows(games=0, ppg=0.0), {}, 2026)
