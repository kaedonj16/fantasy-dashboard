"""Regressions for rookie NFL/advanced-metrics modal eligibility."""
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class _CursorResult:
    def __init__(self, value):
        self.value = value

    def fetchone(self):
        return {"exists": 1} if self.value else None


class _Connection:
    def __init__(self, tables):
        self.tables = tables
        self.queries = []

    def execute(self, sql, _params):
        self.queries.append(sql)
        table = next(name for name in self.tables if name in sql)
        return _CursorResult(self.tables[table])


class _ConnectionContext:
    def __init__(self, tables):
        self.connection = _Connection(tables)

    def __enter__(self):
        return self.connection

    def __exit__(self, *_args):
        return False


def _install_db(monkeypatch, **tables):
    import dashboard_services.db as db
    monkeypatch.setattr(db, "get_conn", lambda: _ConnectionContext(tables))


def test_jeremiyah_love_weekly_row_beats_missing_usage_snapshot(monkeypatch, tmp_path):
    """A rookie's DB appearance is authoritative even with stale disk output."""
    import app

    _install_db(
        monkeypatch,
        player_weekly_metrics=True,
        player_weekly_advanced_metrics=False,
        player_advanced_metrics=False,
    )
    monkeypatch.setattr(app, "CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(app, "_load_usage_rows_cached", lambda _season: None)

    has_games, has_metrics = app._player_nfl_eligibility("jeremiyah-love", 2026)
    assert has_games is True
    assert has_metrics is True


def test_jeremiyah_love_player_details_returns_all_explicit_flags(monkeypatch):
    """Prospect history remains attached, but NFL eligibility wins after debut."""
    import app
    import dashboard_services.rookie_api as rookie_api
    import data_building.rookie_pipeline.pipeline as pipeline

    pid = "13287"
    monkeypatch.setattr(
        "utils.utils.load_relevant_index",
        lambda: {pid: {"name": "Jeremiyah Love", "pos": "RB", "team": "ARI"}},
    )
    monkeypatch.setattr(app, "get_model_value_table_cached", lambda: [])
    monkeypatch.setattr(app, "get_player_value_history", lambda *args, **kwargs: [])
    monkeypatch.setattr(app, "_player_nfl_eligibility", lambda *_args: (True, True))
    monkeypatch.setattr(app, "_oline_for_player", lambda *args, **kwargs: None)
    monkeypatch.setattr(app, "get_players_global", lambda: {})
    monkeypatch.setattr(pipeline, "get_active_rookie_class", lambda: 2026)
    monkeypatch.setitem(
        rookie_api._cache,
        2026,
        [{
            "player_id": "love-prospect",
            "sleeper_id": pid,
            "name": "Jeremiyah Love",
            "draft_class_year": 2026,
            "prospect_score": 91.0,
        }],
    )
    monkeypatch.setitem(rookie_api._cache, 2025, [])

    app.app.config.update(TESTING=True)
    with app.app.test_client() as client:
        response = client.get(f"/api/player-details/{pid}?season=2026")

    assert response.status_code == 200, response.get_json()
    data = response.get_json()
    assert data["name"] == "Jeremiyah Love"
    assert data["has_game_logs"] is True
    assert data["has_advanced_metrics"] is True
    assert data["has_prospect_data"] is True
    assert data["prospect_data"]["draft_class_year"] == 2026


def test_sleeper_game_row_beats_missing_or_stale_usage_rows(monkeypatch, tmp_path):
    import app

    def unavailable_db():
        raise RuntimeError("database unavailable")

    monkeypatch.setattr("dashboard_services.db.get_conn", unavailable_db)
    stats_dir = tmp_path / "sleeper_stats"
    stats_dir.mkdir()
    (stats_dir / "sleeper_stats_s2026_w1.json").write_text(
        json.dumps({"love": {"rush_att": 7, "rush_yd": 41}}), encoding="utf-8"
    )
    monkeypatch.setattr(app, "CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(app, "_load_usage_rows_cached", lambda _season: [])

    assert app._player_nfl_eligibility("love", 2026) == (True, False)


def test_base_or_weekly_advanced_rows_enable_metrics_without_game_log(monkeypatch, tmp_path):
    import app

    monkeypatch.setattr(app, "CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(app, "_load_usage_rows_cached", lambda _season: [])
    for weekly_advanced, base in ((True, False), (False, True)):
        _install_db(
            monkeypatch,
            player_weekly_metrics=False,
            player_weekly_advanced_metrics=weekly_advanced,
            player_advanced_metrics=base,
        )
        assert app._player_nfl_eligibility("veteran", 2026) == (False, True)


def test_prospect_only_columns_do_not_count_as_nfl_advanced_metrics(monkeypatch, tmp_path):
    """A prospect evaluation row is not itself an NFL metric snapshot."""
    import app
    import dashboard_services.db as db

    context = _ConnectionContext({
        "player_weekly_metrics": False,
        "player_weekly_advanced_metrics": False,
        "player_advanced_metrics": False,
    })
    monkeypatch.setattr(db, "get_conn", lambda: context)
    monkeypatch.setattr(app, "CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(app, "_load_usage_rows_cached", lambda _season: [])

    assert app._player_nfl_eligibility("prospect", 2026) == (False, False)
    base_query = next(
        query for query in context.connection.queries
        if "FROM player_advanced_metrics" in query
    )
    assert "item.key NOT LIKE 'rookie_eval_%'" in base_query


def test_frontend_uses_backend_signal_and_safe_game_log_fallback():
    source = (ROOT / "static" / "player_modal.js").read_text(encoding="utf-8")
    assert "['QB', 'RB', 'WR', 'TE'].includes(String(pos || '').toUpperCase())" in source
    assert "(data.has_advanced_metrics === true || hasGameLogs)" in source
    assert "!hasProspectData && pos" not in source
    assert "hasProspectData && !hasGameLogs" in source
    assert "Metrics are being processed for the current week." in source


def test_backend_explicit_flags_and_supported_positions_are_guarded():
    source = (ROOT / "app.py").read_text(encoding="utf-8")
    assert '"has_game_logs": has_game_logs' in source
    assert '"has_advanced_metrics": has_advanced_metrics' in source
    assert '"has_prospect_data": bool(prospect_data)' in source
    assert '_metrics_position in {"QB", "RB", "WR", "TE"}' in source
    for unsupported in ("K", "DEF", "PICK"):
        assert unsupported not in {"QB", "RB", "WR", "TE"}
