"""Tests for the Lineup Lab payload builder and route wiring."""
import re
from pathlib import Path

import pytest

import data_building.lineup_lab as lab_mod
import data_building.player_distributions as pd_mod
import utils.fantasy_scoring as fs_mod
import utils.utils as utils_mod
import dashboard_services.api as api_mod


PROJ = {
    "1": 22.5,  # QB starter
    "2": 18.0,  # RB starter
    "3": 16.9,  # WR starter (Puka-like)
    "4": 14.2,  # WR bench (eligible swap)
    "5": 9.0,   # TE starter
    "6": 12.0,  # RB bench (eligible for FLEX)
    "7": 3.0,   # K starter
    "8": 8.0,   # DEF starter
    "9": 20.0,  # opp QB
    "10": 15.0,  # opp RB
}
POS = {"1": "QB", "2": "RB", "3": "WR", "4": "WR", "5": "TE",
       "6": "RB", "7": "K", "8": "DEF", "9": "QB", "10": "RB"}
TEAM = {"1": "LAR", "2": "DET", "3": "LAR", "4": "CIN", "5": "KC",
        "6": "SF", "7": "DAL", "8": "BUF", "9": "BUF", "10": "MIA"}


def _players_index(extra=None):
    idx = {
        pid: {"position": POS[pid], "pos": POS[pid], "team": TEAM[pid],
              "full_name": f"Player {pid}", "injury_status": ""}
        for pid in PROJ
    }
    for pid, upd in (extra or {}).items():
        idx[pid].update(upd)
    return idx


def _ctx(players_index=None):
    return {
        "current_week": 4,
        "rosters": [
            {"roster_id": 7, "owner_id": "u1",
             "players": ["1", "2", "3", "4", "5", "6", "7", "8"],
             "reserve": [], "taxi": []},
            {"roster_id": 3, "owner_id": "u2", "players": ["9", "10"],
             "reserve": [], "taxi": []},
        ],
        "users": [{"user_id": "u2", "display_name": "Pittsburgh Pilots"}],
        "players_index": players_index or _players_index(),
        "players": {},
        "roster_positions": ["QB", "RB", "WR", "TE", "FLEX", "K", "DEF", "BN", "BN"],
        "raw_scoring_settings": {},
        "scoring_settings": {},
    }


def _matchups(league_id, week):
    return [
        {"roster_id": 7, "matchup_id": 1,
         "starters": ["1", "2", "3", "5", "6", "7", "8"],
         "players": ["1", "2", "3", "4", "5", "6", "7", "8"]},
        {"roster_id": 3, "matchup_id": 1,
         "starters": ["9", "10"], "players": ["9", "10"]},
    ]


def _profiles(requests, season, week):
    out = {}
    for req in requests:
        pid = str(req["player_id"])
        mean = float(req["mean"])
        out[pid] = {
            "player_id": pid, "pos": req["pos"], "mean": mean,
            "std": round(2.0 + 0.42 * mean, 2), "skew_alpha": 2.0,
            "dud_risk": 0.0, "n_games": 3.0,
            "factors": {"unrealized_ay": True} if pid == "3" else {},
        }
    return out


@pytest.fixture
def lab_mocks(monkeypatch):
    monkeypatch.setattr(api_mod, "get_matchups", _matchups)
    monkeypatch.setattr(pd_mod, "build_profiles", _profiles)
    monkeypatch.setattr(pd_mod, "correlation_pairs", lambda pids, season, ctx=None: {})
    monkeypatch.setattr(utils_mod, "load_week_projection", lambda season, week: {})
    monkeypatch.setattr(utils_mod, "load_week_sched", lambda season, week: [
        {"home": "LAR", "away": "SF", "gameDate": "2026-09-27"},
        {"home": "DET", "away": "KC", "gameDate": "2026-09-27"},
        {"home": "CIN", "away": "MIA", "gameDate": "2026-09-27"},
        {"home": "DAL", "away": "BUF", "gameDate": "2026-09-28"},
    ])
    monkeypatch.setattr(fs_mod, "weekly_projection_points",
                        lambda raw, pid, scoring, pos="": PROJ.get(str(pid)))


def _build(ctx):
    return lab_mod.build_lineup_lab_payload(
        ctx=ctx, league_id="123", viewer_roster_id=7, season=2026, week=4)


def test_payload_shape(lab_mocks):
    data = _build(_ctx())
    assert data["week"] == 4
    assert data["n_sims"] == 2000

    lineup = data["you"]["lineup"]
    assert len(lineup) == 7
    by_pid = {e["player_id"]: e for e in lineup}

    wr = by_pid["3"]
    assert wr["name"] == "Player 3"
    assert wr["pos"] == "WR"
    assert wr["proj"] == 16.9
    assert wr["profile"]["mean"] == 16.9
    assert wr["profile"]["std"] > 0
    assert wr["matchup"] == "vs SF"
    assert wr["floor"] < wr["proj"] < wr["ceiling"]
    assert any(t["kind"] == "due" for t in wr["tags"])

    # Bench eligibility: WR slot gets the WR bench player, not the RB.
    bench_ids = [b["player_id"] for b in wr["bench"]]
    assert "4" in bench_ids
    assert "6" not in bench_ids
    # FLEX starter gets both WR and RB bench options.
    flex_bench = [b["player_id"] for b in by_pid["6"]["bench"]]
    assert "4" in flex_bench

    opp = data["opponent"]
    assert opp["name"] == "Pittsburgh Pilots"
    # Opponent mean is net of the expected in-game injury loss (the browser
    # applies injury draws to your side only; see injury_adj below).
    from data_building.injury_rates import expected_injury_loss_per_week
    adj = (expected_injury_loss_per_week(20.0, "QB")
           + expected_injury_loss_per_week(15.0, "RB"))
    assert opp["mean"] == pytest.approx(round(35.0 - round(adj, 1), 1))
    assert opp["injury_adj"] == pytest.approx(round(adj, 1))
    assert opp["std"] > 0
    assert data["corr"] == {}


def test_payload_ships_slot_eligibility_and_usage(lab_mocks, monkeypatch):
    import data_building.weekly_metrics as wm_mod
    monkeypatch.setattr(wm_mod, "get_usage_trends", lambda season: {
        "1": {"stat": "snap_pct", "season_avg": 99.0},
        "2": {"stat": "touches", "season_avg": 18.5},
    })
    data = _build(_ctx())
    by_pid = {e["player_id"]: e for e in data["you"]["lineup"]}
    # Slot eligibility rides along so the browser can re-seat a demoted
    # starter under exactly the slots that accept his position.
    assert by_pid["1"]["eligible"] == ["QB"]
    assert by_pid["3"]["eligible"] == ["WR"]
    assert by_pid["6"]["eligible"] == ["RB", "TE", "WR"]
    # Per-position usage context for the row meta line (QB snap %, RB
    # touches); players without usage data ship nulls, never a fake stat.
    assert by_pid["1"]["usage_stat"] == "snap_pct"
    assert by_pid["1"]["usage_avg"] == 99.0
    assert by_pid["2"]["usage_stat"] == "touches"
    assert by_pid["2"]["usage_avg"] == 18.5
    assert by_pid["3"]["usage_stat"] is None
    assert by_pid["3"]["usage_avg"] is None


def test_unknown_roster_raises(lab_mocks):
    with pytest.raises(LookupError):
        lab_mod.build_lineup_lab_payload(
            ctx=_ctx(), league_id="123", viewer_roster_id=999,
            season=2026, week=4)


def test_bench_excludes_hurt_players(lab_mocks):
    ctx = _ctx(_players_index({"4": {"injury_status": "IR"}}))
    data = _build(ctx)
    wr = next(e for e in data["you"]["lineup"] if e["player_id"] == "3")
    assert "4" not in [b["player_id"] for b in wr["bench"]]


def test_bench_excludes_bye_players(lab_mocks, monkeypatch):
    # CIN has no game in the mocked schedule -> player 4 is on bye.
    monkeypatch.setattr(utils_mod, "load_week_sched", lambda season, week: [
        {"home": "LAR", "away": "SF", "gameDate": "2026-09-27"},
        {"home": "DET", "away": "KC", "gameDate": "2026-09-27"},
        {"home": "DAL", "away": "BUF", "gameDate": "2026-09-28"},
    ])
    data = _build(_ctx())
    wr = next(e for e in data["you"]["lineup"] if e["player_id"] == "3")
    assert "4" not in [b["player_id"] for b in wr["bench"]]


def test_route_wired_in_app():
    src = Path("app.py").read_text()
    assert '@app.route("/api/lineup-lab")' in src
    fn = src[src.index("def api_lineup_lab"):]
    fn = fn[:fn.index("\n@app.route", 1)]
    assert "build_lineup_lab_payload" in fn
    assert "_session_signed_in" in fn
    assert "get_league_ctx_from_cache" in fn
    # Auth failure shapes mirror /api/start-sit-options.
    assert "sign_in_required" in fn
    assert "team_not_linked" in fn


def test_opponent_flagged_missing_when_matchups_fetch_fails(lab_mocks, monkeypatch):
    def _boom(league_id, week):
        raise RuntimeError("sleeper down")

    monkeypatch.setattr(api_mod, "get_matchups", _boom)
    data = _build(_ctx())
    opp = data["opponent"]
    assert opp["missing"] is True
    assert opp["mean"] == 0
    assert opp["name"] == "Opponent"
    # Her side still builds via the optimal-by-projection fallback.
    assert len(data["you"]["lineup"]) > 0


def test_opponent_flagged_missing_when_opp_entry_absent(lab_mocks, monkeypatch):
    mine_only = [m for m in _matchups("123", 4) if str(m.get("roster_id")) == "7"]
    monkeypatch.setattr(api_mod, "get_matchups", lambda lid, wk: mine_only)
    data = _build(_ctx())
    assert data["opponent"]["missing"] is True
    assert data["opponent"]["mean"] == 0


def test_opponent_not_missing_when_resolved(lab_mocks):
    data = _build(_ctx())
    opp = data["opponent"]
    assert opp["missing"] is False
    assert opp["name"] == "Pittsburgh Pilots"
    from data_building.injury_rates import expected_injury_loss_per_week
    adj = (expected_injury_loss_per_week(20.0, "QB")
           + expected_injury_loss_per_week(15.0, "RB"))
    assert opp["mean"] == pytest.approx(round(35.0 - round(adj, 1), 1))


def test_matchups_fetched_exactly_once(lab_mocks, monkeypatch):
    # Hardening: one fetch reused for both sides, not two separate fetches.
    calls = []

    def _counting(league_id, week):
        calls.append((league_id, week))
        return _matchups(league_id, week)

    monkeypatch.setattr(api_mod, "get_matchups", _counting)
    _build(_ctx())
    assert len(calls) == 1
