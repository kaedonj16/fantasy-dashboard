"""GM Brief Playoff Path line (PR A).

The brief's ``playoff_path`` block reads the SAME payloads as the
Standings Playoff Path tab: the shared cached sim rows (never a private
sim run) and ``compute_scenarios`` assembled from ``build_sim_state``.
Its magic number is the scenarios ``wins_to_clinch`` — the race line's
``utils.race_path`` arithmetic is a different number and is never used
here. Fixtures use the REAL scenario enumerator over a crafted sim
state (only ``build_sim_state`` is stubbed): 10 teams, 6 playoff spots,
2 byes, Weeks 13-14 left. The viewer (Nunky, roster 4) is alive with
magic number 1.
"""
import sys
import types
from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

pytest.importorskip("pandas")
pytest.importorskip("numpy")

import data_building.simulate_playoff_odds as spo
from dashboard_services import gm_brief as gb

ET = ZoneInfo("America/New_York")
NOW = datetime(2026, 11, 18, 12, 0, tzinfo=ET).timestamp()  # Wed noon ET

TEAMS = [
    {"roster_id": 1, "name": "Couch",  "wins": 8, "losses": 4, "ties": 0, "pf": 1427.6},
    {"roster_id": 2, "name": "Pilots", "wins": 9, "losses": 3, "ties": 0, "pf": 1460.0},
    {"roster_id": 3, "name": "Fire",   "wins": 8, "losses": 4, "ties": 0, "pf": 1390.0},
    {"roster_id": 4, "name": "Nunky",  "wins": 7, "losses": 5, "ties": 0, "pf": 1330.0},
    {"roster_id": 5, "name": "Trust",  "wins": 6, "losses": 6, "ties": 0, "pf": 1440.0},
    {"roster_id": 6, "name": "Quon",   "wins": 6, "losses": 6, "ties": 0, "pf": 1435.0},
    {"roster_id": 7, "name": "Veiny",  "wins": 6, "losses": 6, "ties": 0, "pf": 1450.0},
    {"roster_id": 8, "name": "Never",  "wins": 5, "losses": 7, "ties": 0, "pf": 1180.0},
    {"roster_id": 9, "name": "Kam",    "wins": 4, "losses": 8, "ties": 0, "pf": 1100.0},
    {"roster_id": 10, "name": "Jerry", "wins": 3, "losses": 9, "ties": 0, "pf": 1020.0},
]

STATE = {
    "teams": TEAMS,
    "playoff_teams": 6,
    "matchups": {
        13: [(1, 2), (3, 4), (5, 9), (6, 10), (7, 8)],
        14: [(1, 3), (2, 9), (4, 10), (5, 7), (6, 8)],
    },
}

STATE_EARLY = {
    "teams": TEAMS,
    "playoff_teams": 6,
    "matchups": {w: [(1, 2), (3, 4), (5, 9), (6, 10), (7, 8)]
                 for w in range(8, 14)},
}

# A 4-team field has no byes (_n_byes(4) == 0): the bye note must vanish.
STATE_NO_BYES = dict(STATE, playoff_teams=4)

ROW = {
    "roster_id": 4, "team_name": "Nunky",
    "playoff_pct": 62.4, "bye_pct": 3.1, "first_seed_pct": 0.4,
    "impact_week": 13,
    "playoff_if_win": 99.1, "playoff_if_lose": 27.2,
    "bye_if_win": 8.0, "bye_if_lose": 0.5,
    "first_seed_if_win": 1.2, "first_seed_if_lose": 0.1,
    "swing": 71.9,
    "root_for": [
        {"a": 1, "b": 2, "playoff_if_a": 55.0, "playoff_if_b": 68.0,
         "bye_if_a": None, "bye_if_b": None},
        {"a": 5, "b": 9, "playoff_if_a": 60.1, "playoff_if_b": 63.3,
         "bye_if_a": None, "bye_if_b": None},
        {"a": 7, "b": 8, "playoff_if_a": 70.2, "playoff_if_b": 41.0,
         "bye_if_a": None, "bye_if_b": None},
    ],
}


def _ctx(**over):
    ctx = {
        "platform": "sleeper",
        "current_week": 13,
        "league_settings": {"playoff_teams": 6, "playoff_week_start": 15},
        "league_rules": {
            "source": "user_confirmed", "playoff_teams": 6,
            "playoff_week_start": 15, "waiver_type": "rolling",
        },
        "rosters": [{"roster_id": t["roster_id"], "players": [],
                     "settings": {}} for t in TEAMS],
        "roster_map": {t["roster_id"]: t["name"] for t in TEAMS},
        "gm_standings": [
            {"roster_id": t["roster_id"], "team_name": t["name"],
             "wins": t["wins"], "losses": t["losses"], "ties": 0,
             "pf": t["pf"]}
            for t in TEAMS
        ],
        "playoff_odds": [ROW],
    }
    ctx.update(over)
    return ctx


@pytest.fixture
def sim_state(monkeypatch):
    monkeypatch.setattr(
        spo, "build_sim_state", lambda ctx, platform="sleeper": STATE)


# ---------------------------------------------------------------------------
# build_path_line
# ---------------------------------------------------------------------------

def test_path_line_in_window(sim_state):
    line = gb.build_path_line(_ctx(), "4")
    assert line is not None
    # The magic number is the enumerator's wins_to_clinch for this exact
    # fixture (1), carried next to the scenario summary.
    assert line["magic_number"] == 1
    assert line["summary"] == "Win and you're in"
    assert line["root_for"]["root_name"] == "Veiny"
    assert line["root_for"]["delta"] == 29.2
    assert line["bye"] == {"pct": 3.1, "if_win": 8.0, "if_lose": 0.5}
    assert line["text"] == (
        "Magic number 1 · can clinch this week · "
        "Root for Veiny: your odds go 70.2% vs 41.0% · "
        "Bye 3.1%: 8.0% with a win, 0.5% with a loss"
    )


def test_path_line_out_of_window_returns_none(monkeypatch):
    monkeypatch.setattr(
        spo, "build_sim_state", lambda ctx, platform="sleeper": STATE_EARLY)
    assert gb.build_path_line(_ctx(), "4") is None


def test_path_line_no_byes_omits_bye_note(monkeypatch):
    monkeypatch.setattr(
        spo, "build_sim_state",
        lambda ctx, platform="sleeper": STATE_NO_BYES)
    line = gb.build_path_line(_ctx(), "4")
    assert line is not None
    assert line["bye"] is None
    assert "Bye" not in line["text"]


def test_path_line_reads_shared_cache_when_ctx_has_no_rows(sim_state,
                                                           monkeypatch):
    calls = []

    def _fake_cached(ctx, platform, block=True):
        calls.append(block)
        return [ROW]

    fake_app = types.ModuleType("app")
    fake_app._playoff_sim_cached = _fake_cached
    monkeypatch.setitem(sys.modules, "app", fake_app)
    ctx = _ctx()
    del ctx["playoff_odds"]
    line = gb.build_path_line(ctx, "4")
    assert line is not None
    # The shared cache is read non-blocking: the brief never waits on a
    # cold Monte Carlo and never runs one itself.
    assert calls == [False]


def test_path_line_cold_cache_returns_none(sim_state, monkeypatch):
    fake_app = types.ModuleType("app")
    fake_app._playoff_sim_cached = lambda ctx, platform, block=True: []
    monkeypatch.setitem(sys.modules, "app", fake_app)
    ctx = _ctx()
    del ctx["playoff_odds"]
    assert gb.build_path_line(ctx, "4") is None


# ---------------------------------------------------------------------------
# build_gm_brief wiring
# ---------------------------------------------------------------------------

def test_brief_carries_playoff_path_and_layer(sim_state):
    brief = gb.build_gm_brief(_ctx(), "4", [], now=NOW)
    assert brief["playoff_path"] is not None
    assert brief["playoff_path"]["magic_number"] == 1
    assert brief["layers"]["path"] is True
    # The race line is untouched by the path block.
    assert brief["race_line"] is not None


def test_brief_without_path_data_degrades_honestly(monkeypatch):
    monkeypatch.setattr(
        spo, "build_sim_state", lambda ctx, platform="sleeper": STATE_EARLY)
    brief = gb.build_gm_brief(_ctx(), "4", [], now=NOW)
    assert brief["playoff_path"] is None
    assert brief["layers"]["path"] is False
