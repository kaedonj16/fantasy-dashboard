"""IR/PUP/NFI players with no Sleeper projection must not inherit a healthy PPG."""
from data_building.fetch_projections import unprojected_season_injury


def test_ir_with_no_sleeper_proj_is_unprojected():
    assert unprojected_season_injury("IR", 0) is True
    assert unprojected_season_injury("IR", None) is True
    assert unprojected_season_injury("ir", "") is True


def test_pup_and_nfi_without_sleeper_proj_are_unprojected():
    assert unprojected_season_injury("PUP", 0) is True
    assert unprojected_season_injury("NFI", 0) is True


def test_sleeper_still_projecting_an_injured_player_is_kept():
    # Alec Pierce-style PUP: Sleeper still has a weekly number.
    assert unprojected_season_injury("PUP", 8.83) is False
    assert unprojected_season_injury("IR", 12.1) is False


def test_healthy_or_week_to_week_injury_is_not_forced_to_zero():
    assert unprojected_season_injury("", 0) is False
    assert unprojected_season_injury("Questionable", 0) is False
    assert unprojected_season_injury("OUT", 0) is False
    assert unprojected_season_injury("Active", 9.8) is False
