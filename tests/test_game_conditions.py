"""Unit tests for utils.game_conditions pure helpers (no network)."""
import utils.game_conditions as gc
from utils.game_conditions import (
    _day_offset,
    build_week_conditions,
    implied_team_total,
    parse_open_meteo_daily,
    parse_tank01_odds,
    total_tag,
    weather_tag,
)


# ---- implied team total ---------------------------------------------------

def test_implied_total_favorite_and_dog():
    # 44.5 total, home favored by 6 -> ~25.25 / ~19.25, rounded to one decimal.
    assert implied_team_total(44.5, -6) == 25.2
    assert implied_team_total(44.5, 6) == 19.2
    # favorite is always implied for more than the underdog
    assert implied_team_total(44.5, -6) > implied_team_total(44.5, 6)


def test_implied_total_pickem_is_half_the_total():
    assert implied_team_total(48, 0) == 24.0


def test_implied_total_bad_inputs():
    assert implied_team_total(0, -3) is None
    assert implied_team_total("x", -3) is None
    assert implied_team_total(45, None) is None


# ---- total tag ------------------------------------------------------------

def test_total_tag_bands():
    assert total_tag(27.0)["kind"] == "high"
    assert total_tag(22.0)["kind"] == "mid"
    assert total_tag(17.0)["kind"] == "low"
    assert total_tag(None) is None
    # label uses %g so it drops trailing .0
    assert total_tag(24.0)["label"] == "24 implied"
    assert total_tag(24.5)["label"] == "24.5 implied"


# ---- weather tag ----------------------------------------------------------

def test_weather_dome_and_benign_are_none():
    assert weather_tag(True, 10, 30, 90) is None          # dome: skip entirely
    assert weather_tag(False, 60, 5, 10) is None          # mild + calm + dry
    assert weather_tag(False, None, None, None) is None   # unknown


def test_weather_wind_takes_priority():
    t = weather_tag(False, 40, 22, 20)
    assert t["kind"] == "wind"
    assert "22 mph wind" in t["label"]


def test_weather_precip_and_cold():
    assert weather_tag(False, 50, 5, 80)["kind"] == "precip"
    cold = weather_tag(False, 20, 5, 10)
    assert cold["kind"] == "cold"
    assert "20°" in cold["label"]


def test_weather_combines_multiple():
    t = weather_tag(False, 18, 20, 70)
    # wind wins the kind, but all three notable parts show
    assert t["kind"] == "wind"
    assert "20 mph wind" in t["label"] and "rain/snow" in t["label"] and "18°" in t["label"]


# ---- Tank01 odds parsing --------------------------------------------------

def test_parse_tank01_top_level_odds():
    body = {
        "20240101_GB@CHI": {
            "homeTeam": "CHI", "awayTeam": "GB",
            "totalOver": "43.5", "homeTeamSpread": "3.0", "awayTeamSpread": "-3.0",
        }
    }
    got = parse_tank01_odds(body)
    assert got["CHI"]["implied"] == implied_team_total(43.5, 3.0)
    assert got["GB"]["implied"] == implied_team_total(43.5, -3.0)
    assert got["GB"]["implied"] > got["CHI"]["implied"]  # GB favored -> more implied


def test_parse_tank01_infers_missing_spread():
    body = {"g1": {"home": "KC", "away": "DEN", "total": "45", "homeTeamSpread": "-7"}}
    got = parse_tank01_odds(body)
    assert got["DEN"]["spread"] == 7.0  # inferred as the negation


def test_parse_tank01_sportsbook_sublist():
    body = {
        "g1": {
            "homeTeam": "BUF", "awayTeam": "MIA",
            "sportsBookOdds": [{"total": "49.5", "homeTeamSpread": "-9.5", "awayTeamSpread": "9.5"}],
        }
    }
    got = parse_tank01_odds(body)
    assert got["BUF"]["total"] == 49.5
    assert got["BUF"]["implied"] > got["MIA"]["implied"]


def test_parse_tank01_garbage_is_safe():
    assert parse_tank01_odds(None) == {}
    assert parse_tank01_odds({"g": {"homeTeam": "", "awayTeam": ""}}) == {}
    assert parse_tank01_odds({"g": "notadict"}) == {}


# ---- Open-Meteo parsing ---------------------------------------------------

def test_parse_open_meteo_blends_temp():
    payload = {"daily": {
        "temperature_2m_max": [50, 40], "temperature_2m_min": [30, 20],
        "wind_speed_10m_max": [12, 25], "precipitation_probability_max": [10, 80],
    }}
    d0 = parse_open_meteo_daily(payload, 0)
    assert d0["wind_mph"] == 12 and d0["precip_pct"] == 10
    assert d0["temp_f"] == round(0.65 * 50 + 0.35 * 30, 1)  # 43.0
    d1 = parse_open_meteo_daily(payload, 1)
    assert d1["wind_mph"] == 25 and d1["precip_pct"] == 80


def test_parse_open_meteo_out_of_range_and_garbage():
    payload = {"daily": {"temperature_2m_max": [50], "wind_speed_10m_max": [12]}}
    assert parse_open_meteo_daily(payload, 5) is None
    assert parse_open_meteo_daily(None, 0) is None
    assert parse_open_meteo_daily({}, 0) is None


# ---- orchestration (monkeypatched fetchers, no network) -------------------

def test_build_week_conditions_assigns_and_shares(monkeypatch):
    # GB @ CHI (both outdoor cold sites) and a game hosted in a dome (DET).
    week_games = [("CHI", "GB", "20241215"), ("DET", "MIN", "20241215")]

    def fake_odds(season, week, dates):
        return {
            "CHI": {"implied": 21.0}, "GB": {"implied": 24.0},
            "DET": {"implied": 27.0}, "MIN": {"implied": 20.0},
        }

    def fake_weather(lat, lon, gd, today=None):
        return {"temp_f": 18, "wind_mph": 22, "precip_pct": 10}

    monkeypatch.setattr(gc, "fetch_week_odds", fake_odds)
    monkeypatch.setattr(gc, "fetch_game_weather", fake_weather)

    out = build_week_conditions(2024, 15, week_games)
    # Vegas implied totals flow through for every team.
    assert out["GB"]["implied_total"] == 24.0 and out["CHI"]["implied_total"] == 21.0
    # Outdoor venue weather is computed once and shared by both teams in the game.
    assert out["CHI"]["weather"]["kind"] == "wind"
    assert out["GB"]["weather"] == out["CHI"]["weather"]
    # The dome game gets no weather even though the fetcher would return some.
    assert out["DET"]["weather"] is None and out["MIN"]["weather"] is None
    assert out["DET"]["implied_total"] == 27.0


def test_build_week_conditions_survives_fetch_errors(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("network down")

    monkeypatch.setattr(gc, "fetch_week_odds", boom)
    out = build_week_conditions(2024, 15, [("CHI", "GB", "20241215")])
    assert out == {}  # never raises; degrades to nothing


# ---- day offset -----------------------------------------------------------

def test_day_offset():
    assert _day_offset("20240108", "20240105") == 3
    assert _day_offset("20240105", "20240105") == 0
    assert _day_offset("20240101", "20240105") == -4
    assert _day_offset(None, "20240105") is None
    assert _day_offset("bad", "20240105") is None
