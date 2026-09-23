"""Regression coverage for the shared Advanced Metrics week selector."""
from pathlib import Path
import contextlib

import pytest

ROOT = Path(__file__).resolve().parents[1]
PAGE = (ROOT / "dashboard_services/pages/advanced_metrics_page.py").read_text()
MODAL = (ROOT / "static/player_modal.js").read_text()
APP = (ROOT / "static/app.js").read_text()
WEEK_RANGE = (ROOT / "static/week_range.js").read_text()
SHELL = (ROOT / "app.py").read_text()


def test_week_selector_is_core_loaded_not_modal_lazy_loaded():
    assert 'src="/static/week_range.js?v={week_range_v}"' in SHELL
    assert SHELL.index("week_range.js?v=") < SHELL.index("{app_js_file}?v=")
    assert "function _wkBarBuild" not in MODAL
    assert "function _wkBarInit" not in MODAL
    assert "global._wkBarBuild" in WEEK_RANGE
    assert "global._wkBarInit" in WEEK_RANGE


def test_player_and_compare_selectors_still_use_shared_component():
    assert "_wkBarBuild('advWkBar'" in MODAL
    assert "_wkBarInit('advWkBar'" in MODAL
    assert "_wkBarBuild('cmpWkBar'" in APP
    assert "_wkBarInit('cmpWkBar'" in APP


def test_page_restores_and_syncs_week_url_without_duplicate_init_fetch():
    assert "_initParams.get('week_start')" in PAGE
    assert "_initParams.get('week_end')" in PAGE
    assert "p.set('week_start', String(urlRange.ws))" in PAGE
    assert "window.addEventListener('load', function() { _amBuildWkBar" not in PAGE
    assert "_amBuildWkBar(r.ws == null" in PAGE
    assert "state.page=0; syncURL(); fetchData();" in PAGE


def test_available_weeks_union_is_grouped_by_season(monkeypatch):
    import data_building.advanced_metrics as metrics

    rows = [
        {"season": 2026, "week": 1},
        *({"season": 2025, "week": week} for week in range(1, 19)),
    ]

    class Conn:
        def execute(self, sql):
            assert "player_weekly_metrics" in sql
            assert "player_weekly_advanced_metrics" in sql
            assert "UNION" in sql
            return self
        def fetchall(self):
            return rows

    @contextlib.contextmanager
    def conn():
        yield Conn()

    monkeypatch.setattr(metrics, "get_conn", conn)
    result = metrics.get_available_weeks_by_season()
    assert result["2026"] == [1]
    assert result["2025"] == list(range(1, 19))


def test_page_uses_season_bounds_and_safe_last_two():
    assert "const lo = weeks[0], hi = weeks[weeks.length - 1]" in PAGE
    assert "const picked = weeks.slice(-count)" in PAGE
    assert "availableWeeksBySeason" in PAGE
    assert "_wkBarBuild('amWkBar', minW, maxW" in PAGE


def test_clicking_only_available_week_is_not_mistaken_for_season():
    callback = PAGE[PAGE.index("_wkBarInit('amWkBar'"):PAGE.index("function _amRefreshWeekControls")]
    assert "state.weekRange = 'custom'" in callback
    assert "state.weekStart = ws; state.weekEnd = we" in callback
    assert "isFull" not in callback


@pytest.mark.parametrize(
    ("query", "expected"),
    [("week_start=12&week_end=3", (3, 12)), ("week_start=0&week_end=30", (1, 18))],
)
def test_api_normalizes_week_ranges(monkeypatch, query, expected):
    pytest.importorskip("flask")
    from flask import Flask
    import data_building.advanced_metrics as metrics
    import routes.advanced_metrics_bp as route

    monkeypatch.setattr(metrics, "get_weekly_range_leaderboard", lambda *a, **kw: [])
    monkeypatch.setattr(route, "get_players_global", lambda: {})
    app = Flask(__name__)
    app.register_blueprint(route.advanced_metrics_bp)
    with app.test_client() as client:
        payload = client.get(
            "/api/advanced-metrics/leaderboard?metric=snap_share&" + query
        ).get_json()
    assert payload["is_week_filtered"] is True
    assert (payload["week_start"], payload["week_end"]) == expected


def test_api_requires_complete_range_and_disables_nonweekly(monkeypatch):
    pytest.importorskip("flask")
    from flask import Flask
    import data_building.advanced_metrics as metrics
    import routes.advanced_metrics_bp as route

    monkeypatch.setattr(metrics, "get_metric_leaderboard", lambda *a, **kw: [])
    monkeypatch.setattr(route, "get_players_global", lambda: {})
    app = Flask(__name__)
    app.register_blueprint(route.advanced_metrics_bp)
    with app.test_client() as client:
        partial = client.get("/api/advanced-metrics/leaderboard?metric=snap_share&week_start=1").get_json()
        # Use a free nonweekly metric (role_score is PRO-gated since #1848).
        nonweekly = client.get("/api/advanced-metrics/leaderboard?metric=air_yards_per_game&week_start=1&week_end=2").get_json()
    assert partial["is_week_filtered"] is False
    assert partial["week_start"] is None and partial["week_end"] is None
    assert nonweekly["is_week_filtered"] is False
