"""Lineup Lab must resolve the opponent for Yahoo leagues.

Regression: build_lineup_lab_payload fetched matchups through the
Sleeper-only dashboard_services.api.get_matchups(league_id, week). For a
Yahoo league id the Sleeper API 404s, the helper returns [], and the Lab
rendered "the opposing lineup could not be loaded" even though the Yahoo
provider adapter publishes canonical Sleeper-shaped matchup rows through
dashboard_services.platform_api.get_matchups.
"""
import pytest

import data_building.lineup_lab as lab_mod
import data_building.player_distributions as pd_mod
import utils.fantasy_scoring as fs_mod
import utils.utils as utils_mod
import dashboard_services.platform_api as platform_api_mod

pytest.importorskip("flask")
yahoo_api = pytest.importorskip("dashboard_services.providers.yahoo_api")


PROJ = {
    "1": 22.5, "2": 18.0, "3": 16.9, "4": 14.2, "5": 9.0,
    "6": 12.0, "7": 3.0, "8": 8.0, "9": 20.0, "10": 15.0,
}
POS = {"1": "QB", "2": "RB", "3": "WR", "4": "WR", "5": "TE",
       "6": "RB", "7": "K", "8": "DEF", "9": "QB", "10": "RB"}
TEAM = {"1": "LAR", "2": "DET", "3": "LAR", "4": "CIN", "5": "KC",
        "6": "SF", "7": "DAL", "8": "BUF", "9": "BUF", "10": "MIA"}

# Canonical (Sleeper) pids per Yahoo team id, as the yahoo_id crosswalk in
# _hydrate_yahoo_matchup_lineups produces them.
STARTERS = {1: ["1", "2", "3", "5", "6", "7", "8"], 4: ["9", "10"]}
PLAYERS = {1: ["1", "2", "3", "4", "5", "6", "7", "8"], 4: ["9", "10"]}


def _team_entry(team_id, points):
    return {"team": [
        [{"team_key": f"461.l.99.t.{team_id}"},
         {"team_id": str(team_id)},
         {"name": f"Team {team_id}"}],
        {"team_points": {"coverage_type": "week", "week": "4",
                         "total": str(points)}},
    ]}


def _scoreboard_raw():
    matchups = {"count": 1, "0": {"matchup": {
        "week": "4",
        "teams": {"count": 2,
                  "0": _team_entry(1, "10.0"),
                  "1": _team_entry(4, "20.0")},
    }}}
    return {"fantasy_content": {"league": [
        {"league_key": "461.l.99", "current_week": "4"},
        {"scoreboard": {"week": "4", "0": {"matchups": matchups}}},
    ]}}


def _hydrate(access_token, league_key, week, rows):
    """Stand-in for _hydrate_yahoo_matchup_lineups' effect on the rows."""
    for row in rows:
        rid = row["roster_id"]
        row["starters"] = list(STARTERS[rid])
        row["players"] = list(PLAYERS[rid])
        row["players_points"] = {}
        row["starters_points"] = [None] * len(STARTERS[rid])


def _ctx():
    # Yahoo ctx shape: rosters carry Yahoo team ids (ints), owner guids,
    # metadata team names, and canonical Sleeper pids (adapter output).
    return {
        "current_week": 4,
        "rosters": [
            {"roster_id": 1, "owner_id": "GUID_A", "league_id": "99",
             "metadata": {"team_name": "My Yahoo Team"},
             "players": list(PLAYERS[1]), "reserve": [], "taxi": None},
            {"roster_id": 4, "owner_id": "GUID_B", "league_id": "99",
             "metadata": {"team_name": "Yahoo Rivals"},
             "players": list(PLAYERS[4]), "reserve": [], "taxi": None},
        ],
        "users": [{"user_id": "GUID_B", "display_name": "Rival Manager",
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
def yahoo_mocks(monkeypatch):
    """Builder deps + the real Yahoo provider stack (network stubbed)."""
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
    # Real chain: platform_api -> YahooProvider -> yahoo_api.get_matchups.
    monkeypatch.setattr(platform_api_mod, "_yahoo_token", lambda *a, **k: "tok")
    monkeypatch.setattr(yahoo_api, "_yahoo_get", lambda *a, **k: _scoreboard_raw())
    monkeypatch.setattr(yahoo_api, "_league_key_for_season",
                        lambda *a, **k: "461.l.99")
    monkeypatch.setattr(yahoo_api, "_hydrate_yahoo_matchup_lineups", _hydrate)


def _build(**overrides):
    kwargs = dict(ctx=_ctx(), league_id="99", viewer_roster_id=1,
                  season=2026, week=4, platform="yahoo")
    kwargs.update(overrides)
    return lab_mod.build_lineup_lab_payload(**kwargs)


def test_yahoo_opponent_resolves_through_provider_stack(yahoo_mocks):
    data = _build()
    opp = data["opponent"]
    assert opp["missing"] is False
    assert opp["roster_id"] == 4
    assert opp["name"] == "Rival Manager"
    from data_building.injury_rates import expected_injury_loss_per_week
    adj = (expected_injury_loss_per_week(20.0, "QB")
           + expected_injury_loss_per_week(15.0, "RB"))
    assert opp["mean"] == pytest.approx(round(35.0 - round(adj, 1), 1))
    assert opp["std"] > 0


def test_yahoo_payload_starters_bench_and_profiles(yahoo_mocks):
    data = _build()
    lineup = data["you"]["lineup"]
    assert len(lineup) == 7
    by_pid = {e["player_id"]: e for e in lineup}
    # Projections/profiles attach to the canonical pids Yahoo ctx carries.
    assert by_pid["3"]["proj"] == 16.9
    assert by_pid["3"]["profile"]["mean"] == 16.9
    assert by_pid["1"]["proj"] == 22.5
    # Bench eligibility still applies (WR bench for the WR starter).
    bench_ids = [b["player_id"] for b in by_pid["3"]["bench"]]
    assert "4" in bench_ids
    assert by_pid["3"]["bench"][0]["proj"] == 14.2


def test_builder_forwards_platform_to_canonical_fetch(monkeypatch, yahoo_mocks):
    calls = []

    def _recording(platform, league_id, week, season, **kw):
        calls.append((platform, league_id, week, season))
        return [
            {"roster_id": 1, "matchup_id": 1, "starters": STARTERS[1],
             "players": PLAYERS[1], "players_points": {}},
            {"roster_id": 4, "matchup_id": 1, "starters": STARTERS[4],
             "players": PLAYERS[4], "players_points": {}},
        ]

    monkeypatch.setattr(platform_api_mod, "get_matchups", _recording)
    data = _build()
    assert calls == [("yahoo", "99", 4, 2026)]
    assert data["opponent"]["missing"] is False


def test_yahoo_no_matchup_stays_explicitly_missing(monkeypatch, yahoo_mocks):
    # A genuine no-matchup week (e.g. Yahoo has not published pairings)
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
