"""Roster-scoped start/sit: slot questions that name no players.

The bug (live, 2026-10-15): "Who should I start at FLEX this week?" is one
of the assistant's own starter chips, but build_start_sit only evaluated
caller-named players, so the no-name case died as "no resolved players".
These tests pin the fix: the candidates come from the asking user's own
roster via the site's lineup solver (utils.lineups assign_optimal_lineup /
assign_fixed_lineup) over the same week projection map the named path
scores from, and both router paths carry the slot.
"""
import pytest

pytest.importorskip("openai")
pytestmark = pytest.mark.integration

from dashboard_services.assistant import compose, evidence, router

INDEX = {
    "qb1": {"name": "Josh Allen", "team": "BUF", "position": "QB"},
    "rb1": {"name": "Bijan Robinson", "team": "ATL", "position": "RB"},
    "rb2": {"name": "Woody Marks", "team": "HOU", "position": "RB"},
    "wr1": {"name": "Ja'Marr Chase", "team": "CIN", "position": "WR"},
    "wr2": {"name": "Rome Odunze", "team": "CHI", "position": "WR"},
    "te1": {"name": "Trey McBride", "team": "ARI", "position": "TE"},
    "fx1": {"name": "Jordan Addison", "team": "MIN", "position": "WR"},
    "btj": {"name": "Brian Thomas Jr.", "team": "JAX", "position": "WR"},
    "flw": {"name": "Zay Flowers", "team": "BAL", "position": "WR"},
    "trc": {"name": "Tyrone Tracy", "team": "NYG", "position": "RB"},
    "bkr": {"name": "Baker Mayfield", "team": "TB", "position": "QB"},
}

PROJ = {"qb1": 24.5, "rb1": 16.9, "rb2": 12.3, "wr1": 18.4, "wr2": 13.8,
        "te1": 11.6, "fx1": 11.4, "btj": 12.6, "flw": 11.7, "trc": 10.8,
        "bkr": 19.6}

STARTERS = ["qb1", "rb1", "rb2", "wr1", "wr2", "te1", "fx1"]
ROSTER_POSITIONS = ["QB", "RB", "RB", "WR", "WR", "TE", "FLEX"]


@pytest.fixture()
def base(monkeypatch):
    monkeypatch.setattr(evidence, "get_players_index_global",
                        lambda *a, **k: dict(INDEX))
    monkeypatch.setattr(evidence, "get_model_value_table_cached",
                        lambda *a, **k: [{"id": pid, "value": 3000}
                                         for pid in INDEX])
    monkeypatch.setattr(evidence, "_metrics_for",
                        lambda pid, season=None: ({"ppg": 10.0}, []))
    monkeypatch.setattr(evidence, "load_week_projection",
                        lambda season, week: {"map": 1})
    monkeypatch.setattr("utils.fantasy_scoring.weekly_projection_points",
                        lambda week_map, pid, scoring, pos: PROJ.get(str(pid)))
    monkeypatch.setattr(evidence, "load_week_sched", lambda season, week: [])
    monkeypatch.setattr(evidence, "load_sleeper_week_stats",
                        lambda season, week: {})
    return monkeypatch


def _bundle(**over):
    bundle = {"platform": "sleeper", "league_id": "L1", "season": 2026,
              "league_name": "Blackedraw", "team_name": "Couch", "week": 7,
              "ctx": {"raw_scoring_settings": {"rec": 0.5},
                      "rosters": [], "roster_positions": list(ROSTER_POSITIONS)},
              "roster": {"roster_id": 3, "players": list(INDEX),
                         "starters": list(STARTERS), "reserve": [], "taxi": []},
              "roster_player_ids": set(INDEX)}
    bundle.update(over)
    return bundle


def _by_id(pack):
    return {p["player_id"]: p for p in pack["players"]}


# -- builder: roster mode ---------------------------------------------------------

def test_flex_question_recommends_bench_challenger(base):
    pack, tools = evidence.build_start_sit(
        _bundle(), {"slot": "FLEX", "start_mode": "start"}, [])
    assert pack["available"] is True
    assert pack["roster_mode"] is True and pack["slot"] == "FLEX"
    assert "utils.lineups.assign_optimal_lineup" in tools
    # The bench WR who out-projects the FLEX occupant is the pick, and the
    # decision names the swap with its projection gain.
    assert pack["pick_player_id"] == "btj"
    assert pack["decisions"] == [{"slot": "FLEX", "in_player_id": "btj",
                                  "out_player_id": "fx1", "gain": 1.2}]
    players = _by_id(pack)
    assert set(players) == {"btj", "flw", "trc", "fx1"}
    assert players["fx1"]["slot_occupant"] is True
    assert players["fx1"]["current_starter"] is True
    assert players["btj"]["current_starter"] is False
    assert players["btj"]["projection"] == 12.6


def test_position_question_answers_the_marginal_slot(base):
    pack, _ = evidence.build_start_sit(
        _bundle(), {"slot": "RB", "start_mode": "start"}, [])
    assert pack["available"] is True and pack["slot"] == "RB"
    # Every rostered RB is a candidate; the solver keeps Marks in the
    # second RB slot over the bench challenger, so he is the pick.
    assert set(_by_id(pack)) == {"rb1", "rb2", "trc"}
    assert pack["pick_player_id"] == "rb2"
    assert pack["decisions"] == []


def test_sit_question_mirrors_the_swap(base):
    pack, _ = evidence.build_start_sit(
        _bundle(), {"slot": "FLEX", "start_mode": "sit"}, [])
    assert pack["available"] is True
    assert pack["question_mode"] == "sit"
    assert pack["sit_player_id"] == "fx1"
    assert pack["pick_player_id"] == "btj"


def test_no_slot_question_reports_the_lineups_decisions(base):
    pack, _ = evidence.build_start_sit(_bundle(), {}, [])
    assert pack["available"] is True and pack["roster_mode"] is True
    assert pack["pick_player_id"] == "btj"
    assert pack["decisions"][0]["slot"] == "FLEX"
    assert set(_by_id(pack)) == {"btj", "fx1"}


def test_numbered_slot_uses_that_ordinal(base):
    pack, _ = evidence.build_start_sit(
        _bundle(), {"slot": "RB2", "start_mode": "start"}, [])
    assert pack["slot_label"] == "RB2"
    assert pack["pick_player_id"] == "rb2"


def test_named_head_to_head_is_unchanged(base):
    pack, _ = evidence.build_start_sit(
        _bundle(), {"slot": "FLEX", "start_mode": "start"}, ["wr2", "fx1"])
    assert "roster_mode" not in pack
    assert [p["player_id"] for p in pack["players"]] == ["wr2", "fx1"]
    assert pack["pick_player_id"] == "wr2"  # 13.8 projects over 11.4


def test_no_linked_roster_still_declines_cleanly(base):
    pack, _ = evidence.build_start_sit(
        _bundle(roster={"players": [], "starters": []}),
        {"slot": "FLEX", "start_mode": "start"}, [])
    assert pack["available"] is False
    assert "no linked team" in pack["missing"]


# -- compose: required receipts pin the two projections ----------------------------

def test_roster_pack_required_receipts(base):
    pack, _ = evidence.build_start_sit(
        _bundle(), {"slot": "FLEX", "start_mode": "start"}, [])
    rows = compose._required_receipts("start_sit", pack)
    assert rows == [{"label": "Brian Thomas Jr. projection", "value": "12.6"},
                    {"label": "Jordan Addison projection", "value": "11.4"}]


# -- routing: the slot rides both paths --------------------------------------------

@pytest.fixture()
def keyword(monkeypatch):
    def _boom(*a, **k):
        raise RuntimeError("model down")
    monkeypatch.setattr(router, "call_model_json", _boom)
    return router.classify


def _route(classify, message):
    return classify(message, {"league_name": "Blackedraw", "week": 7,
                              "season": 2026}, None)


@pytest.mark.parametrize("message,slot,mode", [
    ("Who should I start at FLEX this week?", "FLEX", "start"),
    ("Who should I sit at FLEX this week?", "FLEX", "sit"),
    ("Who do I bench at flex?", "FLEX", "sit"),
    ("Who should I start at RB2?", "RB2", "start"),
    ("Who should I start at running back?", "RB", "start"),
    ("Start or sit: who goes at my superflex?", "SUPER_FLEX", "start"),
    ("Who should I start this week?", None, "start"),
])
def test_keyword_route_carries_slot(keyword, message, slot, mode):
    routed = _route(keyword, message)
    assert routed["intent"] == "start_sit"
    assert routed["entities"]["slot"] == slot
    assert routed["entities"]["start_mode"] == mode


def test_model_route_passes_slot_through(monkeypatch):
    def _stub(system, payload, schema, name):
        assert "slot" in schema["properties"]
        assert "start_mode" in schema["properties"]
        assert "roster slot question" in system
        return {"intent": "start_sit", "player_names": [], "slot": "FLEX",
                "start_mode": "start"}
    monkeypatch.setattr(router, "call_model_json", _stub)
    routed = _route(router.classify, "Who should I start at FLEX this week?")
    assert routed["intent"] == "start_sit"
    assert routed["entities"]["slot"] == "FLEX"
    assert routed["entities"]["start_mode"] == "start"
