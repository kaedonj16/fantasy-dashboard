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
    body = src[start:start + 5500]
    assert "get_players_index_global()" in body
    assert "load_relevant_index()" in body
    assert "get_team_full_name" in body
    assert "load_teams_index()" in body
    assert "get_players_global()" in body
    assert "_compute_team_offense_ranks" in body
    assert "build_team_schedule" in body
    assert "resolve_team_for_season" in body
    assert "schedule" in body
    assert "def api_player_team_boxscore" in src
    assert "get_shaped_boxscore" in src
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
    # O-line lives in its own Team-tab section (not mixed into Offense Profile
    # volume ranks), with position-primary highlighting and short labels.
    assert "pm-oline-sec" in js
    assert "Offensive Line" in js
    assert "pm-tp-primary" in js
    assert "Pass Block Grade" not in js
    assert "Run Block Grade" not in js
    assert "mkOl('Pass Block'" in js
    assert "mkOl('Run Block'" in js
    assert "mkOl('Overall'" in js
    assert "oline_' + metric" in js
    # Must not append O-line rows into the Offense Profile volume block.
    assert "${profile}${olineRows}" not in js
    assert "${profile}" in js
    assert "${olineSec}" in js
    # Schedule accordion under the Team tab header.
    assert "_pmBuildScheduleHTML" in js
    assert "Schedule" in js
    assert "pm-team-schedule" in js
    assert "pm-schedule-toggle" in js
    assert "pm-boxscore" in js
    assert "/api/player-team-boxscore" in js
    assert "pmPickTeamSeason" in js
    assert "_pmCollapseAllSchedule" in js
    assert "${scheduleSec}" in js
    assert "Box score available once the game begins" in js


def test_player_modal_team_tab_css():
    css = (ROOT / "static" / "dashboard.css").read_text(encoding="utf-8")
    for cls in (
        ".pm-team-header", ".pm-crest", ".pm-hero-stat", ".pm-team-depth",
        ".player-badge-inj-q", ".pm-team-adv-toggle", ".pm-team-usage",
        ".pm-team-season-pills", ".pm-team-season-pill",
        ".pm-tp-primary", ".pm-tp-for", ".pm-oline-link",
        ".pm-team-schedule", ".pm-schedule-row", ".pm-schedule-toggle",
        ".pm-boxscore", ".pm-boxscore-table", ".pm-boxscore-focus",
        ".pm-boxscore-team-pill",
    ):
        assert cls in css, cls

    # Schedule section reuses Team-tab section chrome (padding + top border).
    assert ".pm-team-sec { padding: 14px 18px; border-top: 1px solid var(--border); }" in css
    assert "font-variant-numeric: tabular-nums" in css
    assert "prefers-reduced-motion" in css


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
    monkeypatch.setattr(
        "utils.player_team_schedule.build_team_schedule",
        lambda *a, **k: [
            {
                "week": 1, "week_label": "Week 1", "date": "20250905",
                "date_label": "Sep 5", "opponent": "LAC", "opponent_name": "Chargers",
                "opponent_logo": "", "is_home": False, "ha": "@", "status": "final",
                "result": "W", "team_pts": 27, "opp_pts": 21, "kickoff": "",
                "quarter": "", "clock": "", "game_id": "20250905_KC@LAC",
                "season": 2025, "season_type": "reg", "is_postseason": False,
                "bye": False, "expandable": True,
            },
            {
                "week": 10, "week_label": "Week 10", "bye": True, "expandable": False,
                "opponent": "BYE", "status": "bye", "season": 2025, "season_type": "reg",
                "is_postseason": False, "game_id": "",
            },
        ],
    )
    # Bust payload cache between monkeypatched runs.
    from app import _TEAM_PAYLOAD_CACHE
    _TEAM_PAYLOAD_CACHE.clear()

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
    assert isinstance(data.get("schedule"), list)
    assert data["schedule"][0]["opponent"] == "LAC"
    assert data["schedule"][0]["result"] == "W"
    assert any(g.get("bye") for g in data["schedule"])

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
        "utils.player_team_schedule.build_team_schedule",
        lambda *a, **k: [],
    )
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
    from app import _TEAM_PAYLOAD_CACHE
    _TEAM_PAYLOAD_CACHE.clear()

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
    assert data.get("schedule") == []


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


def test_api_player_team_uses_historical_team(flask_client, monkeypatch):
    monkeypatch.setattr("app.get_players_global", lambda: _mock_sleeper_players())
    monkeypatch.setattr("app._get_pfr_snap_counts_cached", lambda season: {})
    monkeypatch.setattr(
        "app.get_players_index_global",
        lambda: {"4046": {"name": "Patrick Mahomes", "pos": "QB", "team": "KC"}},
    )
    monkeypatch.setattr(
        "utils.player_team_schedule.resolve_team_for_season",
        lambda pid, season, fallback="": "BUF",
    )
    seen = {}

    def _fake_sched(team, season, **kwargs):
        seen["team"] = team
        return []

    monkeypatch.setattr("utils.player_team_schedule.build_team_schedule", _fake_sched)
    from app import _TEAM_PAYLOAD_CACHE
    _TEAM_PAYLOAD_CACHE.clear()

    resp = flask_client.get("/api/player-team/4046?season=2024")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["team"] == "BUF"
    assert seen.get("team") == "BUF"


def test_api_player_team_boxscore_future_and_final(flask_client, monkeypatch):
    from app import _RZ_BOX_CACHE
    _RZ_BOX_CACHE.clear()

    def _fake_fetch(gid):
        if "FUTURE" in gid:
            return {}
        return {
            "home": "KC",
            "away": "BAL",
            "homePts": "27",
            "awayPts": "20",
            "gameStatusCode": "2",
            "gameStatus": "Final",
            "playerStats": {
                "1": {
                    "longName": "Patrick Mahomes",
                    "teamAbv": "KC",
                    "Passing": {
                        "passCompletions": "0",
                        "passAttempts": "1",
                        "passYds": "0",
                        "passTD": "0",
                        "int": "0",
                    },
                    "Rushing": {"carries": "0", "rushYds": "0", "rushTD": "0"},
                },
                "2": {
                    "longName": "Rashee Rice",
                    "teamAbv": "KC",
                    "Receiving": {
                        "targets": "2",
                        "receptions": "1",
                        "recYds": "12",
                        "recTD": "1",
                    },
                    # Cross-position production: WR throw.
                    "Passing": {
                        "passCompletions": "1",
                        "passAttempts": "1",
                        "passYds": "5",
                        "passTD": "1",
                        "int": "0",
                    },
                },
            },
        }

    monkeypatch.setattr("app._redzone_boxscore", _fake_fetch)
    monkeypatch.setattr(
        "app.get_players_index_global",
        lambda: {
            "4046": {"name": "Patrick Mahomes", "team": "KC", "pos": "QB", "tankId": "1"},
            "7771": {"name": "Rashee Rice", "team": "KC", "pos": "WR", "tankId": "2"},
        },
    )

    future = flask_client.get(
        "/api/player-team-boxscore?game_id=20990101_FUTURE@KC&team=KC&focus_pid=4046"
    )
    assert future.status_code == 200
    fdata = future.get_json()
    assert fdata["started"] is False
    assert "Box score available once the game begins" in fdata["message"]

    final = flask_client.get(
        "/api/player-team-boxscore?game_id=20240905_BAL@KC&team=KC&focus_pid=4046"
    )
    assert final.status_code == 200
    data = final.get_json()
    assert data["started"] is True
    assert data["status"] == "final"
    assert data["home"]["pts"] == 27
    kc = data["teams"]["KC"]
    qb = next(g for g in kc["groups"] if g["pos"] == "QB")
    focus = qb["players"][0]
    assert focus["is_focus"] is True
    # Recorded zero stays 0 (not en-dash) once the Passing block exists.
    assert focus["cells"]["pass_yds"] == 0
    assert focus["cells"]["cmp_att"] == "0/1"
    wr = next(g for g in kc["groups"] if g["pos"] == "WR")
    rice = wr["players"][0]
    assert "pass_td" in rice["cells"]
    assert rice["cells"]["pass_td"] == 1
