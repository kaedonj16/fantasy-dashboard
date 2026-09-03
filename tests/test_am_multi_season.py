"""Multi-season advanced metrics: parse, career subset, leaderboard, page UI."""
from pathlib import Path

import pytest

pytest.importorskip("flask")

ROOT = Path(__file__).resolve().parents[1]
_AM_PAGE = (ROOT / "dashboard_services" / "pages" / "advanced_metrics_page.py").read_text(
    encoding="utf-8"
)


def test_parse_season_list_newest_first_unique():
    from data_building.advanced_metrics import parse_season_list

    assert parse_season_list("2024, 2025, 2022, 2025") == [2025, 2024, 2022]
    assert parse_season_list(2024) == [2024]
    assert parse_season_list(["2022", 2025, "nope"]) == [2025, 2022]
    assert parse_season_list("") == []
    assert parse_season_list(None) == []


def test_career_metrics_filters_to_requested_seasons(monkeypatch):
    import contextlib
    import data_building.advanced_metrics as am

    rows = [
        {"player_id": "1", "position": "WR", "season": 2025, "as_of_date": "2025-02",
         "yards_per_target": 10.0, "catch_rate": 0.7},
        {"player_id": "1", "position": "WR", "season": 2024, "as_of_date": "2024-02",
         "yards_per_target": 8.0, "catch_rate": 0.6},
        {"player_id": "1", "position": "WR", "season": 2022, "as_of_date": "2022-02",
         "yards_per_target": 6.0, "catch_rate": 0.5},
    ]

    class _Cur:
        def fetchall(self):
            return rows

    class _Conn:
        def execute(self, *a, **k):
            return _Cur()

    @contextlib.contextmanager
    def _fake_conn(*a, **k):
        yield _Conn()

    monkeypatch.setattr(am, "get_conn", _fake_conn)
    subset = am.get_player_career_metrics("1", seasons=[2025, 2022])
    all_years = am.get_player_career_metrics("1")
    assert subset is not None
    # Newest selected year weight 1, next selected weight 0.5 (2024 skipped).
    assert subset["yards_per_target"] == pytest.approx((10.0 * 1 + 6.0 * 0.5) / 1.5)
    assert all_years["yards_per_target"] != subset["yards_per_target"]
    assert am.get_player_career_metrics("1", seasons=[2019]) is None


def test_multi_season_leaderboard_default_is_each_year(monkeypatch):
    import data_building.advanced_metrics as am

    seen_vol = []

    def _single(metric, position=None, limit=500, season=None, min_vol=None):
        if isinstance(season, list):
            raise AssertionError("inner call must be a single year")
        seen_vol.append(min_vol)
        if season == 2025:
            return [
                {"player_id": "1", "name": "A", "position": "RB",
                 "value": 5.0, "vol": 100, "games": 10},
            ]
        return [
            {"player_id": "1", "name": "A", "position": "RB",
             "value": 3.0, "vol": 50, "games": 5},
        ]

    monkeypatch.setattr(am, "get_metric_leaderboard", _single)
    rows = am._multi_season_leaderboard(
        "yards_per_carry", position=None, limit=50, seasons=[2025, 2022], min_vol=40,
    )
    assert seen_vol == [40, 40]
    assert [(r["player_id"], r["season"], r["value"]) for r in rows] == [
        ("1", 2025, 5.0),
        ("1", 2022, 3.0),
    ]


def test_multi_season_leaderboard_combines_by_player(monkeypatch):
    import data_building.advanced_metrics as am

    def _single(metric, position=None, limit=500, season=None, min_vol=None):
        if isinstance(season, list):
            raise AssertionError("inner call must be a single year")
        assert min_vol is None
        if season == 2025:
            return [
                {"player_id": "1", "name": "A", "position": "RB",
                 "value": 5.0, "vol": 100, "games": 10, "car": 100},
                {"player_id": "2", "name": "B", "position": "RB",
                 "value": 4.0, "vol": 80, "games": 8, "car": 80},
            ]
        return [
            {"player_id": "1", "name": "A", "position": "RB",
             "value": 3.0, "vol": 50, "games": 5, "car": 50},
        ]

    monkeypatch.setattr(am, "get_metric_leaderboard", _single)
    rows = am._multi_season_leaderboard(
        "yards_per_carry", position=None, limit=50, seasons=[2025, 2022],
        min_vol=None, combine=True,
    )
    by_id = {r["player_id"]: r for r in rows}
    assert set(by_id) == {"1", "2"}
    assert by_id["1"]["value"] == pytest.approx((5.0 * 100 + 3.0 * 50) / 150)
    assert by_id["1"]["vol"] == 150
    assert by_id["1"]["games"] == 15
    assert by_id["1"]["car"] == 150
    assert by_id["1"]["season"] is None
    assert by_id["2"]["value"] == pytest.approx(4.0)
    assert by_id["2"]["vol"] == 80


def test_multi_season_min_vol_applies_after_combine(monkeypatch):
    import data_building.advanced_metrics as am

    def _single(metric, position=None, limit=500, season=None, min_vol=None):
        if season == 2025:
            return [
                {"player_id": "1", "name": "A", "position": "RB", "value": 5.0, "vol": 100},
                {"player_id": "2", "name": "B", "position": "RB", "value": 4.0, "vol": 80},
            ]
        return [
            {"player_id": "1", "name": "A", "position": "RB", "value": 3.0, "vol": 50},
        ]

    monkeypatch.setattr(am, "get_metric_leaderboard", _single)
    rows = am._multi_season_leaderboard(
        "yards_per_carry", position=None, limit=50, seasons=[2025, 2022],
        min_vol=120, combine=True,
    )
    assert [r["player_id"] for r in rows] == ["1"]


def test_multi_season_sums_counting_metrics(monkeypatch):
    import data_building.advanced_metrics as am

    def _single(metric, position=None, limit=500, season=None, min_vol=None):
        return [{"player_id": "1", "name": "A", "position": "RB",
                 "value": 100 if season == 2025 else 50, "vol": 100 if season == 2025 else 50}]

    monkeypatch.setattr(am, "get_metric_leaderboard", _single)
    rows = am._multi_season_leaderboard(
        "total_carries", position=None, limit=50, seasons=[2025, 2022],
        min_vol=None, combine=True,
    )
    assert len(rows) == 1
    assert rows[0]["value"] == 150


def test_leaderboard_api_passes_season_list(offline_client, monkeypatch):
    import data_building.advanced_metrics as am

    seen = {}

    def _lb(metric, position=None, limit=500, season=None, min_vol=None, combine=False):
        seen["season"] = season
        seen["combine"] = combine
        return [{"player_id": "1", "name": "A", "position": "RB", "value": 4.5, "season": 2025}]

    monkeypatch.setattr(am, "get_metric_leaderboard", _lb)
    monkeypatch.setattr(am, "get_value_leaderboard", lambda *a, **k: [])
    resp = offline_client.get(
        "/api/advanced-metrics/leaderboard?metric=yards_per_carry&season=2025,2024,2022"
    )
    assert resp.status_code == 200, resp.get_data(as_text=True)
    data = resp.get_json()
    assert seen["season"] == [2025, 2024, 2022]
    assert seen["combine"] is False
    assert data["selected_seasons"] == [2025, 2024, 2022]
    assert data["combine"] is False


def test_leaderboard_api_combine_flag(offline_client, monkeypatch):
    import data_building.advanced_metrics as am

    seen = {}

    def _lb(metric, position=None, limit=500, season=None, min_vol=None, combine=False):
        seen["combine"] = combine
        return [{"player_id": "1", "name": "A", "position": "RB", "value": 4.5}]

    monkeypatch.setattr(am, "get_metric_leaderboard", _lb)
    monkeypatch.setattr(am, "get_value_leaderboard", lambda *a, **k: [])
    resp = offline_client.get(
        "/api/advanced-metrics/leaderboard?metric=yards_per_carry&season=2025,2022&combine=1"
    )
    assert resp.status_code == 200, resp.get_data(as_text=True)
    assert seen["combine"] is True
    assert resp.get_json()["combine"] is True


def test_get_metric_leaderboard_forwards_combine(monkeypatch):
    import data_building.advanced_metrics as am

    seen = {}

    def _ms(*a, **k):
        seen.update(k)
        return [{"player_id": "1", "value": 1}]

    monkeypatch.setattr(am, "_multi_season_leaderboard", _ms)
    am.get_metric_leaderboard("yards_per_carry", season="2025,2022")
    assert seen.get("combine") is False
    am.get_metric_leaderboard("yards_per_carry", season=[2025, 2022], combine=True)
    assert seen.get("combine") is True


def test_metrics_page_renders_combine_toggle(offline_client):
    resp = offline_client.get("/metrics")
    assert resp.status_code == 200
    html = resp.get_data(as_text=True)
    assert 'id="amCombineToggle"' in html
    assert "Each year" in html
    assert 'data-combine="1"' in html


def test_page_has_multi_season_picker():
    assert 'id="amSeasonMenu"' in _AM_PAGE
    assert 'id="amSeasonBtn"' in _AM_PAGE
    assert "function applySeasonSelection" in _AM_PAGE
    assert "function amIsMultiSeason" in _AM_PAGE
    assert "function amIsEachYear" in _AM_PAGE
    assert 'id="amSeasonColHdr"' in _AM_PAGE
    assert 'id="amCombineToggle"' in _AM_PAGE
    assert "Each year" in _AM_PAGE
    assert "Pick any years that have data" in _AM_PAGE
    assert "one row per season" in _AM_PAGE
    assert "amRowKey" in _AM_PAGE
