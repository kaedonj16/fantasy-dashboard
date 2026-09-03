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


def test_multi_season_leaderboard_stamps_each_year(monkeypatch):
    import data_building.advanced_metrics as am

    def _single(metric, position=None, limit=500, season=None, min_vol=None):
        if isinstance(season, list):
            raise AssertionError("inner call must be a single year")
        return [{"player_id": "1", "name": "A", "position": "RB", "value": float(season)}]

    # Bypass the public wrapper so we exercise the merge helper directly.
    monkeypatch.setattr(am, "get_metric_leaderboard", _single)
    rows = am._multi_season_leaderboard(
        "yards_per_carry", position=None, limit=50, seasons=[2025, 2022], min_vol=None,
    )
    assert [r["season"] for r in rows] == [2025, 2022]
    assert [r["value"] for r in rows] == [2025.0, 2022.0]


def test_leaderboard_api_passes_season_list(offline_client, monkeypatch):
    import data_building.advanced_metrics as am

    seen = {}

    def _lb(metric, position=None, limit=500, season=None, min_vol=None):
        seen["season"] = season
        return [{"player_id": "1", "name": "A", "position": "RB", "value": 4.5, "season": 2025}]

    monkeypatch.setattr(am, "get_metric_leaderboard", _lb)
    monkeypatch.setattr(am, "get_value_leaderboard", lambda *a, **k: [])
    resp = offline_client.get(
        "/api/advanced-metrics/leaderboard?metric=yards_per_carry&season=2025,2024,2022"
    )
    assert resp.status_code == 200, resp.get_data(as_text=True)
    data = resp.get_json()
    assert seen["season"] == [2025, 2024, 2022]
    assert data["selected_seasons"] == [2025, 2024, 2022]


def test_page_has_multi_season_picker():
    assert 'id="amSeasonMenu"' in _AM_PAGE
    assert 'id="amSeasonBtn"' in _AM_PAGE
    assert "function applySeasonSelection" in _AM_PAGE
    assert "function amIsMultiSeason" in _AM_PAGE
    assert 'id="amSeasonColHdr"' in _AM_PAGE
    assert "Pick any years that have data" in _AM_PAGE
    assert "am-season-col" in _AM_PAGE
    assert "concat(amIsMultiSeason() ? ['Year'] : [])" in _AM_PAGE
    assert "amRowKey(rx)" in _AM_PAGE
