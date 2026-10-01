"""Lineup Lab live mode (payload side).

When the requested week is the current week and games have started, the
matchup entries' players_points plus the week's schedule statuses mark
players final (locked at actuals) or in progress (display only). Pre-game
and past-week payloads must carry no live keys at all.
"""
import pytest

import data_building.lineup_lab as lab_mod
import data_building.player_distributions as pd_mod
import utils.fantasy_scoring as fs_mod
import utils.utils as utils_mod
import dashboard_services.api as api_mod


PROJ = {
    "1": 22.5, "2": 18.0, "3": 16.9, "4": 14.2, "5": 9.0,
    "6": 12.0, "7": 3.0, "8": 8.0, "9": 20.0, "10": 15.0,
}
POS = {"1": "QB", "2": "RB", "3": "WR", "4": "WR", "5": "TE",
       "6": "RB", "7": "K", "8": "DEF", "9": "QB", "10": "RB"}
TEAM = {"1": "LAR", "2": "DET", "3": "LAR", "4": "CIN", "5": "KC",
        "6": "SF", "7": "DAL", "8": "BUF", "9": "BUF", "10": "MIA"}

# gameStatusCode: "2" final, "1" in progress, "0" scheduled (Tank01 codes,
# read by utils.normalize_game_status_from_tank01).
SCHED = [
    {"home": "LAR", "away": "SF", "gameStatusCode": "2"},
    {"home": "DET", "away": "KC", "gameStatusCode": "1"},
    {"home": "CIN", "away": "MIA", "gameStatusCode": "2"},
    {"home": "DAL", "away": "BUF", "gameStatusCode": "2"},
]

MINE_POINTS = {"1": 24.3, "2": 10.0, "3": 18.2, "4": 6.4,
               "5": 7.5, "6": 12.0, "7": 9.0, "8": 11.0}
OPP_POINTS = {"9": 21.0, "10": 13.5}


def _players_index():
    return {
        pid: {"position": POS[pid], "pos": POS[pid], "team": TEAM[pid],
              "full_name": f"Player {pid}", "injury_status": ""}
        for pid in PROJ
    }


def _ctx(current_week=4):
    return {
        "current_week": current_week,
        "rosters": [
            {"roster_id": 7, "owner_id": "u1",
             "players": ["1", "2", "3", "4", "5", "6", "7", "8"],
             "reserve": [], "taxi": []},
            {"roster_id": 3, "owner_id": "u2", "players": ["9", "10"],
             "reserve": [], "taxi": []},
        ],
        "users": [{"user_id": "u2", "display_name": "Pittsburgh Pilots"}],
        "players_index": _players_index(),
        "players": {},
        "roster_positions": ["QB", "RB", "WR", "TE", "FLEX", "K", "DEF", "BN", "BN"],
        "raw_scoring_settings": {},
        "scoring_settings": {},
    }


def _matchups(with_points):
    mine = {"roster_id": 7, "matchup_id": 1,
            "starters": ["1", "2", "3", "5", "6", "7", "8"],
            "players": ["1", "2", "3", "4", "5", "6", "7", "8"]}
    opp = {"roster_id": 3, "matchup_id": 1,
           "starters": ["9", "10"], "players": ["9", "10"]}
    if with_points:
        mine["players_points"] = dict(MINE_POINTS)
        opp["players_points"] = dict(OPP_POINTS)
    return [mine, opp]


def _profiles(requests, season, week):
    out = {}
    for req in requests:
        pid = str(req["player_id"])
        mean = float(req["mean"])
        out[pid] = {
            "player_id": pid, "pos": req["pos"], "mean": mean,
            "std": round(2.0 + 0.42 * mean, 2), "skew_alpha": 2.0,
            "dud_risk": 0.0, "n_games": 3.0, "factors": {},
        }
    return out


@pytest.fixture
def live_mocks(monkeypatch):
    state = {"with_points": True, "sched": SCHED}
    monkeypatch.setattr(api_mod, "get_matchups",
                        lambda league_id, week: _matchups(state["with_points"]))
    monkeypatch.setattr(pd_mod, "build_profiles", _profiles)
    monkeypatch.setattr(pd_mod, "correlation_pairs", lambda pids, season, ctx=None: {})
    monkeypatch.setattr(utils_mod, "load_week_projection", lambda season, week: {})
    monkeypatch.setattr(utils_mod, "load_week_sched",
                        lambda season, week: state["sched"])
    monkeypatch.setattr(fs_mod, "weekly_projection_points",
                        lambda raw, pid, scoring, pos="": PROJ.get(str(pid)))
    return state


def _build(ctx, week=4):
    return lab_mod.build_lineup_lab_payload(
        ctx=ctx, league_id="123", viewer_roster_id=7, season=2026, week=week)


def _all_entries(payload):
    entries = list(payload["you"]["lineup"])
    for e in payload["you"]["lineup"]:
        entries.extend(e.get("bench") or [])
    return entries


def test_final_and_live_players_carry_live_state(live_mocks):
    payload = _build(_ctx())
    by_pid = {e["player_id"]: e for e in payload["you"]["lineup"]}
    # LAR/SF final: locked at actuals.
    assert by_pid["1"]["live"] == {"status": "final", "points": 24.3}
    assert by_pid["3"]["live"] == {"status": "final", "points": 18.2}
    assert by_pid["6"]["live"] == {"status": "final", "points": 12.0}
    # DET/KC in progress: flagged live with points so far (display only).
    assert by_pid["2"]["live"] == {"status": "live", "points": 10.0}
    assert by_pid["5"]["live"] == {"status": "live", "points": 7.5}
    # DAL/BUF final, including the defense.
    assert by_pid["7"]["live"] == {"status": "final", "points": 9.0}
    assert by_pid["8"]["live"] == {"status": "final", "points": 11.0}
    # Bench entries carry it too (WR bench player 4, CIN/MIA final).
    bench4 = [b for e in payload["you"]["lineup"]
              for b in (e.get("bench") or []) if b["player_id"] == "4"]
    assert bench4 and bench4[0]["live"] == {"status": "final", "points": 6.4}
    assert payload["live"] is True


def test_opponent_locks_only_when_fully_final(live_mocks):
    payload = _build(_ctx())
    # Both opp starters (BUF 9, MIA 10) are final: 21.0 + 13.5.
    assert payload["opponent"]["live_points"] == 34.5

    # CIN/MIA still in progress: the opponent aggregate cannot lock.
    live_mocks["sched"] = [
        g if g["home"] != "CIN" else {**g, "gameStatusCode": "1"}
        for g in SCHED
    ]
    payload = _build(_ctx())
    assert "live_points" not in payload["opponent"]
    assert payload["live"] is True  # your side still has live state


def test_pre_game_payload_has_no_live_keys(live_mocks):
    live_mocks["with_points"] = False
    payload = _build(_ctx())
    assert "live" not in payload
    assert "live_points" not in payload["opponent"]
    for e in _all_entries(payload):
        assert "live" not in e


def test_non_current_week_has_no_live_keys(live_mocks):
    # Points exist and games are final, but the viewed week is not the
    # current week: no live state is attached.
    payload = _build(_ctx(current_week=5), week=4)
    assert "live" not in payload
    assert "live_points" not in payload["opponent"]
    for e in _all_entries(payload):
        assert "live" not in e


def test_scheduled_games_have_no_live_state(live_mocks):
    live_mocks["sched"] = [{**g, "gameStatusCode": "0"} for g in SCHED]
    payload = _build(_ctx())
    assert "live" not in payload
    for e in _all_entries(payload):
        assert "live" not in e
