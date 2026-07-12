"""Unit tests for utils.nfl_stadiums (static game-environment tags)."""
from utils.nfl_stadiums import STADIUMS, game_environment, normalize_team


def test_all_32_teams_present():
    # Keyed by team, so all 32 (LAR/LAC share SoFi and NYG/NYJ share MetLife,
    # but each team is still its own key).
    assert len(STADIUMS) == 32
    # Every team key is a 2-3 letter abbr and carries the required fields.
    for abbr, st in STADIUMS.items():
        assert 2 <= len(abbr) <= 3
        assert set(st) == {"name", "dome", "climate", "lat", "lon"}
        assert st["climate"] in {"dome", "cold", "mild", "warm"}
        assert (st["climate"] == "dome") == st["dome"]
        assert -90 <= st["lat"] <= 90 and -180 <= st["lon"] <= 0  # continental US


def test_dome_is_weatherproof_regardless_of_week():
    for wk in (1, 14, 18):
        env = game_environment("NO", week=wk)
        assert env["dome"] is True
        assert env["cold"] is False
        assert env["label"] == "Dome"


def test_cold_site_only_flags_late_season():
    assert game_environment("GB", week=3)["cold"] is False
    assert game_environment("GB", week=13)["cold"] is False
    assert game_environment("GB", week=14)["cold"] is True
    assert game_environment("GB", week=17)["cold"] is True


def test_warm_outdoor_never_cold():
    for wk in (1, 15, 18):
        env = game_environment("MIA", week=wk)
        assert env["dome"] is False
        assert env["cold"] is False


def test_missing_week_never_cold():
    # Without a week we can't know it's the cold stretch, so don't flag it.
    assert game_environment("CHI")["cold"] is False


def test_aliases_resolve():
    assert normalize_team("JAC") == "JAX"
    assert normalize_team("wsh") == "WAS"
    assert normalize_team("LA") == "LAR"
    assert game_environment("OAK", week=1)["stadium"] == "Allegiant Stadium"


def test_unknown_team_returns_none():
    assert game_environment("XXX", week=5) is None
    assert game_environment("", week=5) is None


def test_sofi_tenants_are_domes():
    for team in ("LAR", "LAC"):
        assert game_environment(team, week=16)["dome"] is True
