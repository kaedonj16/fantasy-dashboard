"""Player modal Team tab payload additions (feat/player-modal-fixes).

/api/player-team depth rows gain the counting stats the summary strip
renders next to the shares: per-player targets + the team target total
("51 of 186"), the viewed player's receiving line, and a positional PPG
rank on the focus row from the same rank machinery /api/player-details
uses for its ppg_rank. The payload also carries focus_weekly (played-week
targets + snap share) for the Usage-by-week card.
"""
import pytest

pytest.importorskip("flask")
pytest.importorskip("pandas")

from array import array

import app


# ---------------------------------------------------------------------------
# _team_total_targets: the usage table's own team attribution
# ---------------------------------------------------------------------------

def test_team_total_targets_sums_only_that_team():
    table = {
        "1": {"total_targets": 51.0, "season_team": "DAL"},
        "2": {"total_targets": 41.0, "season_team": "DAL"},
        "3": {"season_targets": 21.0, "season_team": "DAL"},
        "4": {"total_targets": 999.0, "season_team": "NYG"},
        "5": {"total_targets": 12.0},  # no attribution: not counted
    }
    assert app._team_total_targets(table, "DAL") == 113


def test_team_total_targets_list_shape_and_none():
    rows = [
        {"id": "1", "usage": {"total_targets": 10.0, "season_team": "KC"}},
        {"id": "2", "total_targets": 5.0, "season_team": "KC"},
    ]
    assert app._team_total_targets(rows, "KC") == 15
    assert app._team_total_targets({}, "KC") is None
    assert app._team_total_targets(None, "KC") is None


# ---------------------------------------------------------------------------
# Route-level: depth rows + focus ppg_rank + focus_weekly
# ---------------------------------------------------------------------------

@pytest.fixture
def flask_client():
    try:
        from app import app as flask_app
    except Exception as exc:
        pytest.skip(f"app not importable ({type(exc).__name__})")
    flask_app.config.update(TESTING=True)
    with flask_app.test_client() as client:
        yield client


def _dal_players():
    def p(name, pos, order, slot=None):
        d = {"full_name": name, "team": "DAL", "position": pos,
             "depth_chart_order": order, "injury_status": ""}
        if slot:
            d["depth_chart_position"] = slot
        return d
    return {
        "6786": p("CeeDee Lamb", "WR", 1, "WR"),
        "9002": p("George Pickens", "WR", 2, "WR"),
        "9003": p("Ryan Flournoy", "WR", 3, "WR"),
        "9004": p("KaVontae Turpin", "WR", 4, "WR"),
        "9005": p("Jonathan Mingo", "WR", 5, "WR"),
        "9006": p("Jake Ferguson", "TE", 1),
        "9007": p("Javonte Williams", "RB", 1),
    }


def _dal_usage():
    def u(tgt, games=5, rec=0.0, yd=0.0, td=0.0, ppg=None, team="DAL"):
        return {"games": games, "total_targets": float(tgt),
                "target_share": tgt / 186.0, "avg_targets": tgt / games,
                "avg_receptions": rec, "avg_rec_yards": yd,
                "avg_rec_tds": td, "ppr_ppg": ppg,
                "avg_off_snap_pct": 0.761, "season_team": team}
    return {
        # Lamb: 51 targets, 39 / 507 / 4 receiving over 5 games.
        "6786": u(51, rec=7.8, yd=101.4, td=0.8, ppg=23.0),
        "9002": u(41, rec=6.0, yd=80.0, td=0.6),
        "9003": u(27), "9004": u(7), "9005": u(7),
        "9006": u(21), "9007": u(20),
        # A DAL player with targets but no depth-chart row still counts
        # toward the team total (12 more -> 186).
        "9999": u(12),
        # Another team's targets never count toward DAL's total.
        "5555": u(999, team="NYG"),
    }


def _patch_team_route(monkeypatch):
    monkeypatch.setattr("app.get_players_global", _dal_players)
    monkeypatch.setattr("app.get_players_index_global", lambda: {
        "6786": {"team": "DAL", "pos": "WR", "name": "CeeDee Lamb"}})
    # Keep the route's usage_index hermetic: no embedded usage fallbacks
    # from the real relevant index.
    monkeypatch.setattr("utils.utils.load_relevant_index", lambda: {})
    monkeypatch.setattr("app.get_usage_table_global", _dal_usage)
    monkeypatch.setattr("app._get_pfr_snap_counts_cached", lambda season: {})
    monkeypatch.setattr(
        "utils.player_team_schedule.build_team_schedule", lambda *a, **k: [])
    # PPG rank machinery: Lamb 23.02 PPG, one WR above him, one below.
    monkeypatch.setattr("app._load_season_weekly_points", lambda s, st, weeks=None: {
        "6786": [15.4, 35.3, 20.2, 41.3, 2.9],
        "1111": [10.0] * 5,
        "2222": [30.0] * 5,
    })
    monkeypatch.setattr("app._season_rank_rows", lambda s, st: {
        "pos_ppg": {"WR": array("d", [10.0, 23.02, 30.0])},
        "pos_total": {}, "all_ppg": array("d"), "all_total": array("d"),
    })
    monkeypatch.setattr(
        "dashboard_services.breakout_api._weekly_series_for_payload",
        lambda pid, season: [
            {"week": 1, "snap_share": 81.0, "targets": 8},
            {"week": 2, "snap_share": 74.5, "targets": 9},
        ])
    from app import _TEAM_PAYLOAD_CACHE, _TEAM_OFFENSE_RANKS_CACHE
    _TEAM_PAYLOAD_CACHE.clear()
    _TEAM_OFFENSE_RANKS_CACHE.clear()


def test_player_team_payload_counts_receiving_and_rank(flask_client, monkeypatch):
    _patch_team_route(monkeypatch)
    resp = flask_client.get("/api/player-team/6786?season=2025")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["available"] is True

    wr_room = data["depth_chart"]["WR"]
    focus = next(r for r in wr_room if r["is_focus"])
    assert focus["targets"] == 51
    assert focus["team_targets"] == 186
    assert focus["rec"] == 39
    assert focus["rec_yd"] == 507
    assert focus["rec_td"] == 4
    assert focus["ppg_rank"] == 2

    pickens = next(r for r in wr_room if r["id"] == "9002")
    assert pickens["targets"] == 41
    assert pickens["team_targets"] == 186
    assert "ppg_rank" not in pickens  # rank is the focus player's only

    assert data["focus_weekly"] == [
        {"week": 1, "snap_share": 81.0, "targets": 8},
        {"week": 2, "snap_share": 74.5, "targets": 9},
    ]


def test_player_team_payload_without_usage_stays_empty(flask_client, monkeypatch):
    _patch_team_route(monkeypatch)
    monkeypatch.setattr("app.get_usage_table_global", lambda: {})
    monkeypatch.setattr("app._load_season_weekly_points",
                        lambda s, st, weeks=None: {})
    from app import _TEAM_PAYLOAD_CACHE
    _TEAM_PAYLOAD_CACHE.clear()
    resp = flask_client.get("/api/player-team/6786?season=2025")
    assert resp.status_code == 200
    data = resp.get_json()
    focus = next(r for r in data["depth_chart"]["WR"] if r["is_focus"])
    # Nothing fabricated: every new field is absent/None without a source.
    assert focus["targets"] is None
    assert focus["team_targets"] is None
    assert focus["rec"] is None
    assert focus["rec_yd"] is None
    assert focus["rec_td"] is None
    assert focus["ppg_rank"] is None
