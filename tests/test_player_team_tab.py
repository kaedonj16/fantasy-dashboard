"""Player modal Team tab: /api/player-team endpoint and UI wiring."""
from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _player_team_route_src() -> str:
    src = (ROOT / "app.py").read_text(encoding="utf-8")
    start = src.find("def api_player_team")
    end = src.find("clean_nan_for_json = _sanitize_for_json", start)
    assert start > 0 and end > start
    return src[start:end]


def test_player_team_route_exists_and_uses_real_sources():
    src = (ROOT / "app.py").read_text(encoding="utf-8")
    start = src.find("def api_player_team")
    assert start > 0
    body = src[start:start + 4000]
    assert "get_players_index_global()" in body
    assert "load_relevant_index()" in body
    assert "get_team_full_name" in body
    assert "load_teams_index()" in body
    assert "get_players_global()" in body
    assert "_compute_team_offense_ranks" in body
    helpers = src[src.find("# ── Player modal Team tab"):start + 500]
    assert "stats_player_reg_" in helpers
    assert "fetch_season_snap_counts" in helpers
    assert "normalize_name" in helpers
    assert "_TEAM_OFFENSE_RANKS_CACHE" in helpers
    assert "_canon_team_abbr" in helpers


def test_player_modal_team_tab_ui_wiring():
    js = (ROOT / "static" / "player_modal.js").read_text(encoding="utf-8")
    assert 'id="pmTabTeam"' in js
    assert 'id="pm-panel-team"' in js
    assert "_pmBuildTeamHTML" in js
    assert "/api/player-team/" in js
    assert "pmHasTeam" in js
    assert "'team'" in js
    assert "_pmTeamAdvOpen" in js
    assert "openPlayerModal(pid,pname,{force:true})" in js.replace(" ", "")
    assert "pmPickTeamSeason" in js
    assert "pm-team-season-pills" in js
    assert "available_seasons" in js
    assert "data_mode" in js


def test_player_modal_team_tab_css():
    css = (ROOT / "static" / "dashboard.css").read_text(encoding="utf-8")
    for cls in (
        ".pm-team-header", ".pm-crest", ".pm-hero-stat", ".pm-team-depth",
        ".player-badge-inj-q", ".pm-team-adv-toggle", ".pm-team-usage",
        ".pm-team-season-pills", ".pm-team-season-pill",
    ):
        assert cls in css, cls


@pytest.fixture
def flask_client():
    try:
        from app import app as flask_app
    except Exception as exc:
        pytest.skip(f"app not importable ({type(exc).__name__})")
    flask_app.config.update(TESTING=True)
    with flask_app.test_client() as client:
        yield client


def _mock_sleeper_players():
    return {
        "4046": {
            "full_name": "Patrick Mahomes",
            "team": "KC",
            "position": "QB",
            "depth_chart_order": 1,
            "injury_status": "",
        },
        "9991": {
            "full_name": "Backup QB",
            "team": "KC",
            "position": "QB",
            "depth_chart_order": 2,
            "injury_status": "Questionable",
        },
        "8881": {
            "full_name": "Isiah Pacheco",
            "team": "KC",
            "position": "RB",
            "depth_chart_order": 1,
            "injury_status": "IR",
        },
        "7771": {
            "full_name": "Rashee Rice",
            "team": "KC",
            "position": "WR",
            "depth_chart_order": 1,
            "depth_chart_position": "LWR",
            "injury_status": "",
        },
    }


def test_api_player_team_known_qb(flask_client, monkeypatch):
    monkeypatch.setattr("app.get_players_global", lambda: _mock_sleeper_players())
    monkeypatch.setattr("app._get_pfr_snap_counts_cached", lambda season: {})

    resp = flask_client.get("/api/player-team/4046?season=2025")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["available"] is True
    assert data["team"] == "KC"
    assert data["position"] == "QB"
    assert data["player_id"] == "4046"
    assert data["data_mode"] == "actual"
    assert data["stats_season"] == 2025
    assert 2025 in data["available_seasons"]

    ranks = data["ranks"]
    for key in ("points", "pass_yds", "pass_att", "rush_yds", "rush_att"):
        assert key in ranks, key
        entry = ranks[key]
        if entry is not None:
            assert set(entry.keys()) == {"rank", "value", "total"}
            assert entry["total"] <= 32

    qb_rows = data["depth_chart"]["QB"]
    focus = [r for r in qb_rows if r.get("is_focus")]
    assert len(focus) == 1
    assert focus[0]["id"] == "4046"

    inj_rows = [r for r in qb_rows if r.get("injury")]
    assert any(r["injury"] for r in inj_rows)

    for row in qb_rows:
        assert row.get("snap_pct_source") in ("pfr", "derived", None)
        if row["id"] == "4046" and row.get("snap_pct") is not None:
            assert row["snap_pct_source"] == "derived"


def test_api_player_team_projection_season(flask_client, monkeypatch):
    """Seasons without a stats CSV should use Sleeper season projections."""
    monkeypatch.setattr("app.get_players_global", lambda: _mock_sleeper_players())
    monkeypatch.setattr("app._get_pfr_snap_counts_cached", lambda season: {})
    monkeypatch.setattr("app._has_stats_reg_csv", lambda season: False)
    monkeypatch.setattr(
        "app._sleeper_season_proj_lines",
        lambda season: {
            "4046": {
                "raw_stats": {
                    "pass_yd": 4200, "pass_att": 580, "pass_td": 32,
                    "rush_yd": 350, "rush_att": 60, "rush_td": 2,
                },
                "pts_ppr": 360,
            },
            "9991": {
                "raw_stats": {
                    "pass_yd": 200, "pass_att": 40, "pass_td": 1,
                    "rush_yd": 20, "rush_att": 5, "rush_td": 0,
                },
                "pts_ppr": 20,
            },
            "8881": {
                "raw_stats": {"rush_yd": 900, "rush_att": 220, "rush_td": 8},
                "pts_ppr": 180,
            },
            "7771": {
                "raw_stats": {"rec_yd": 1100, "rec": 90, "rec_td": 9, "rush_yd": 40, "rush_att": 8},
                "pts_ppr": 220,
            },
        },
    )
    monkeypatch.setattr(
        "app._list_team_tab_seasons",
        lambda current: [int(current), int(current) - 1],
    )

    resp = flask_client.get("/api/player-team/4046?season=2026")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["available"] is True
    assert data["data_mode"] == "projection"
    assert data["stats_season"] == 2026
    assert data["season"] == 2026
    assert data["available_seasons"] == [2026, 2025]
    assert data["ranks"]["pass_yds"] is not None
    assert data["ranks"]["pass_yds"]["value"] >= 4200
    assert data["ranks_more"]["pass_rate"] is not None


def test_api_player_team_wsh_was_not_double_counted(flask_client, monkeypatch):
    monkeypatch.setattr("app.get_players_global", lambda: {})
    monkeypatch.setattr("app._get_pfr_snap_counts_cached", lambda season: {})

    from app import _compute_team_offense_ranks

    payload = _compute_team_offense_ranks(2025)
    ranks = payload["ranks"]["points"]
    assert ranks
    assert "WSH" not in ranks
    totals = {v["total"] for v in ranks.values()}
    assert max(totals) <= 32
    assert payload["data_mode"] == "actual"

def test_api_player_team_unavailable_without_team(flask_client, monkeypatch):
    monkeypatch.setattr(
        "app.get_players_index_global",
        lambda: {"99999": {"name": "Free Agent", "pos": "WR", "team": ""}},
    )
    resp = flask_client.get("/api/player-team/99999?season=2025")
    assert resp.status_code == 200
    assert resp.get_json()["available"] is False


def test_api_player_team_hidden_position_def(flask_client, monkeypatch):
    monkeypatch.setattr(
        "app.get_players_index_global",
        lambda: {"88888": {"name": "Chiefs DST", "pos": "DEF", "team": "KC"}},
    )
    resp = flask_client.get("/api/player-team/88888?season=2025")
    assert resp.status_code == 200
    assert resp.get_json()["available"] is False
