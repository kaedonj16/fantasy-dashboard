"""Lineup Lab must resolve the opponent for Fleaflicker leagues.

Regression: the Fleaflicker adapter builds matchup starters from the
per-game boxscore. Before kickoff (or when the boxscore fetch fails) the
boxscore slots are empty, so the adapter emits "0" placeholders and the
matchup rows carry no usable starters. build_lineup_lab_payload found the
matchup row but zero opponent starters and stamped opponent.missing, so
the Lab rendered "The opposing lineup could not be loaded" even though
the adapter's get_rosters (FetchRoster START groups) carries the
currently-set lineup for both teams.

The builder now falls back to each side's roster starters when the
matchup row exists but its starters are empty.
"""
import pytest

import data_building.lineup_lab as lab_mod
import data_building.player_distributions as pd_mod
import utils.fantasy_scoring as fs_mod
import utils.utils as utils_mod
import dashboard_services.platform_api as platform_api_mod

pytest.importorskip("flask")


PROJ = {
    "1": 22.5, "2": 18.0, "3": 16.9, "4": 14.2, "5": 9.0,
    "6": 12.0, "7": 3.0, "8": 8.0, "9": 20.0, "10": 15.0,
}
POS = {"1": "QB", "2": "RB", "3": "WR", "4": "WR", "5": "TE",
       "6": "RB", "7": "K", "8": "DEF", "9": "QB", "10": "RB"}
TEAM = {"1": "LAR", "2": "DET", "3": "LAR", "4": "CIN", "5": "KC",
        "6": "SF", "7": "DAL", "8": "BUF", "9": "BUF", "10": "MIA"}

# The currently-set lineups, as Fleaflicker get_rosters returns them from
# FetchRoster START groups (canonical Sleeper pids, adapter output).
ROSTER_STARTERS = {101: ["1", "2", "3", "5", "6", "7", "8"],
                   202: ["9", "10"]}
ROSTER_PLAYERS = {101: ["1", "2", "3", "4", "5", "6", "7", "8"],
                  202: ["9", "10"]}


def _ctx():
    # Fleaflicker ctx shape: rosters carry int team ids, canonical pids,
    # and the set starters from FetchRoster.
    return {
        "current_week": 4,
        "rosters": [
            {"roster_id": 101, "owner_id": "101", "league_id": "14153",
             "metadata": {"team_name": "My Flea Team"},
             "players": list(ROSTER_PLAYERS[101]),
             "starters": list(ROSTER_STARTERS[101]),
             "reserve": [], "taxi": []},
            {"roster_id": 202, "owner_id": "202", "league_id": "14153",
             "metadata": {"team_name": "Flea Rivals"},
             "players": list(ROSTER_PLAYERS[202]),
             "starters": list(ROSTER_STARTERS[202]),
             "reserve": [], "taxi": []},
        ],
        "users": [{"user_id": "202", "display_name": "Rival Manager",
                   "username": "Rival Manager"}],
        "players_index": {
            pid: {"position": POS[pid], "pos": POS[pid], "team": TEAM[pid],
                  "full_name": f"Player {pid}", "injury_status": ""}
            for pid in PROJ
        },
        "players": {},
        "roster_positions": ["QB", "RB", "WR", "TE", "FLEX", "K", "DEF",
                             "BN", "BN"],
        "raw_scoring_settings": {},
        "scoring_settings": {},
    }


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
def flea_mocks(monkeypatch):
    monkeypatch.setattr(pd_mod, "build_profiles", _profiles)
    monkeypatch.setattr(pd_mod, "correlation_pairs",
                        lambda pids, season, ctx=None: {})
    monkeypatch.setattr(utils_mod, "load_week_projection",
                        lambda season, week: {})
    monkeypatch.setattr(utils_mod, "load_week_sched", lambda season, week: [
        {"home": "LAR", "away": "SF", "gameDate": "2026-09-27"},
        {"home": "DET", "away": "KC", "gameDate": "2026-09-27"},
        {"home": "CIN", "away": "MIA", "gameDate": "2026-09-27"},
        {"home": "DAL", "away": "BUF", "gameDate": "2026-09-28"},
    ])
    monkeypatch.setattr(fs_mod, "weekly_projection_points",
                        lambda raw, pid, scoring, pos="": PROJ.get(str(pid)))


def _pregame_matchups(platform, league_id, week, season, **kw):
    """Fleaflicker adapter shape when the boxscore has no players yet:
    the matchup rows exist, but every slot is a "0" placeholder."""
    return [
        {"roster_id": 101, "matchup_id": 3, "week": 4,
         "starters": ["0"] * 7, "players": [], "players_points": {},
         "starters_points": [None] * 7},
        {"roster_id": 202, "matchup_id": 3, "week": 4,
         "starters": ["0"] * 2, "players": [], "players_points": {},
         "starters_points": [None] * 2},
    ]


def _build(**overrides):
    kwargs = dict(ctx=_ctx(), league_id="14153", viewer_roster_id="101",
                  season=2026, week=4, platform="fleaflicker")
    kwargs.update(overrides)
    return lab_mod.build_lineup_lab_payload(**kwargs)


def test_fleaflicker_opponent_falls_back_to_roster_starters(
        monkeypatch, flea_mocks):
    monkeypatch.setattr(platform_api_mod, "get_matchups", _pregame_matchups)
    data = _build()
    opp = data["opponent"]
    assert opp["missing"] is False
    assert opp["roster_id"] == 202
    assert opp["name"] == "Rival Manager"
    # The opponent's set lineup (from get_rosters) seeds the sim, not zeros.
    assert opp["mean"] > 0


def test_fleaflicker_viewer_falls_back_to_roster_starters(
        monkeypatch, flea_mocks):
    monkeypatch.setattr(platform_api_mod, "get_matchups", _pregame_matchups)
    data = _build()
    by_pid = {e["player_id"]: e for e in data["you"]["lineup"]}
    # Her actual set starters, not the optimal-by-projection fallback.
    assert set(by_pid) == set(ROSTER_STARTERS[101])


def test_fleaflicker_no_matchup_stays_explicitly_missing(
        monkeypatch, flea_mocks):
    # A genuine no-matchup week (Fleaflicker has not published pairings)
    # must still surface the explicit missing state, never a fake one.
    monkeypatch.setattr(platform_api_mod, "get_matchups",
                        lambda platform, league_id, week, season, **kw: [])
    data = _build()
    opp = data["opponent"]
    assert opp["missing"] is True
    assert opp["mean"] == 0
    assert opp["roster_id"] is None
    # Her side still builds via the optimal-by-projection fallback.
    assert len(data["you"]["lineup"]) > 0


# --- Best moves must respect league rules ---------------------------------
# Regression: a starter the builder cannot seat (unknown position, e.g. an
# unmapped D/ST like "BAL", or a K when the league's slot list has no K)
# got slot "" with eligible=[] and the ENTIRE bench as its pool, which the
# client treated as unrestricted. Best moves then suggested illegal swaps
# like "Malik Nabers in for Jake Bates" (WR for K).

UNSEATED_PROJ = dict(PROJ, b2="8.5", BAL2=6.0)
UNSEATED_POS = dict(POS, b2="K")


def _unseated_build(monkeypatch, starters, roster_positions, extra_pids):
    """Payload for a week whose boxscore seats an unmapped D/ST ("BAL2")."""
    def _matchups(platform, league_id, week, season, **kw):
        return [
            {"roster_id": 101, "matchup_id": 1, "week": 4,
             "starters": list(starters), "players": list(starters),
             "players_points": {}},
            {"roster_id": 202, "matchup_id": 1, "week": 4,
             "starters": ["9", "10"], "players": ["9", "10"],
             "players_points": {}},
        ]
    monkeypatch.setattr(platform_api_mod, "get_matchups", _matchups)
    monkeypatch.setattr(
        fs_mod, "weekly_projection_points",
        lambda raw, pid, scoring, pos="": UNSEATED_PROJ.get(str(pid), 5.0))
    ctx = _ctx()
    ctx["players_index"] = {
        pid: {"position": UNSEATED_POS[pid], "pos": UNSEATED_POS[pid],
              "team": {"b2": "DAL"}.get(pid, TEAM.get(pid, "X")),
              "full_name": {"7": "Jake Bates"}.get(pid, f"Player {pid}"),
              "injury_status": ""}
        for pid in UNSEATED_PROJ if pid != "BAL2"
    }
    ctx["rosters"][0]["players"] = list(starters) + list(extra_pids)
    ctx["rosters"][0]["starters"] = list(starters)
    ctx["roster_positions"] = list(roster_positions)
    return lab_mod.build_lineup_lab_payload(
        ctx=ctx, league_id="14153", viewer_roster_id="101",
        season=2026, week=4, platform="fleaflicker")


def _row_by_name(data, name):
    for e in data["you"]["lineup"]:
        if e["name"] == name:
            return e
    raise AssertionError(f"no lineup row named {name!r}")


def test_unmapped_starter_gets_empty_bench_pool(monkeypatch, flea_mocks):
    # "BAL2" is a D/ST the adapter could not map: unknown position, so no
    # slot's eligibility matches and it stays unseated.
    data = _unseated_build(
        monkeypatch,
        starters=["1", "2", "3", "5", "6", "7", "8", "BAL2"],
        roster_positions=["QB", "RB", "WR", "TE", "FLEX", "K", "DEF",
                          "BN", "BN"],
        extra_pids=["4", "9", "10"],
    )
    row = _row_by_name(data, "Player BAL2")
    assert row["slot"] == ""
    assert row["eligible"] == []
    assert row["bench"] == []
    # Seated starters keep their legal pools (WR bench for the WR slot).
    wr = _row_by_name(data, "Player 3")
    assert wr["eligible"] == ["WR"]
    assert [b["player_id"] for b in wr["bench"]] == ["4"]


def test_unseated_known_position_falls_back_to_own_position(
        monkeypatch, flea_mocks):
    # No K slot in the league's slot list: the K starter cannot be seated,
    # so it falls back to its own position. A WR must never be swappable
    # for the K, but another K on the bench still is.
    data = _unseated_build(
        monkeypatch,
        starters=["1", "2", "3", "5", "6", "8", "7"],
        roster_positions=["QB", "RB", "WR", "TE", "FLEX", "DEF",
                          "BN", "BN", "BN"],
        extra_pids=["4", "9", "10", "b2"],
    )
    row = _row_by_name(data, "Jake Bates")
    assert row["slot"] == ""
    assert row["eligible"] == ["K"]
    assert [b["player_id"] for b in row["bench"]] == ["b2"]
