"""Rank-a-named-set tests: the deterministic rank shape, its guards,
the rank_players tool, and the open layer's pre-planner fetch.

The defect (collapse audit, 2026-10-16): "Rank Garrett Wilson, Nico
Collins and Justin Jefferson by value" had no governed intent and no
tool that ranks a named set, so the open planner fetched one player's
numbers and the composer asserted a ranking of all three from that
subset, with the order wrong. The fixture world mirrors
test_assistant_open_layer.py: Alpha 5000, Beta 4200, Gamma 6100, so
the correct value order differs from the mention order (the most
valuable player is named last).
"""
import pytest

pytest.importorskip("openai")
pytestmark = pytest.mark.integration

from dashboard_services.assistant import context as ctx_mod
from dashboard_services.assistant import evidence as ev
from dashboard_services.assistant import open_layer, router
from dashboard_services.assistant import tools as tools_mod

INDEX = {
    "1": {"name": "Alpha Receiver", "team": "CIN", "position": "WR"},
    "2": {"name": "Beta Receiver", "team": "MIN", "position": "WR"},
    "3": {"name": "Gamma Back", "team": "IND", "position": "RB"},
    "4": {"name": "Delta Smith", "team": "NYJ", "position": "WR"},
    "5": {"name": "Delta Smith", "team": "MIA", "position": "RB"},
}

WEEK_STATS = {
    1: {"1": {"rec_yd": 100, "rec": 8, "rec_tgt": 12, "rec_td": 1},
        "2": {"rec_yd": 80, "rec": 6, "rec_tgt": 9, "rec_td": 0},
        "3": {"rush_yd": 90, "rush_att": 18, "rush_td": 1}},
    2: {"1": {"rec_yd": 120, "rec": 9, "rec_tgt": 13, "rec_td": 2},
        "2": {"rec_yd": 60, "rec": 5, "rec_tgt": 8, "rec_td": 1},
        "3": {"rush_yd": 70, "rush_att": 15, "rush_td": 0}},
}

VALUES = [{"id": "1", "value": 5000}, {"id": "2", "value": 4200},
          {"id": "3", "value": 6100}]


def _bundle(**over):
    bundle = {"platform": "sleeper", "league_id": "L1", "season": 2026,
              "league_name": "Blackedraw", "team_name": "Couch", "week": 3,
              "ctx": None, "roster": None, "roster_player_ids": set(),
              "has_pro": True}
    bundle.update(over)
    return bundle


@pytest.fixture()
def base(monkeypatch):
    for mod in (ev, ctx_mod):
        monkeypatch.setattr(mod, "get_players_index_global",
                            lambda *a, **k: dict(INDEX))
        monkeypatch.setattr(mod, "get_model_value_table_cached",
                            lambda *a, **k: list(VALUES))
    monkeypatch.setattr(ev, "load_sleeper_week_stats",
                        lambda season, week: dict(WEEK_STATS.get(week) or {}))
    return monkeypatch


@pytest.fixture()
def keyword(monkeypatch):
    def _boom(*a, **k):
        raise RuntimeError("model down")
    monkeypatch.setattr(router, "call_model_json", _boom)
    return router.classify


def _classify(classify, message):
    return classify(message, {"league_name": "Blackedraw", "week": 7,
                              "season": 2026}, None)


# -- shape grammar --------------------------------------------------------------

def test_shape_audit_question():
    shape = router.rank_shape(
        "Rank Garrett Wilson, Nico Collins and Justin Jefferson by value")
    assert shape == {"names": ["garrett wilson", "nico collins",
                               "justin jefferson"],
                     "key": "value", "unsupported_key": None}


def test_shape_verbs_and_keys():
    shape = router.rank_shape(
        "order Ja'Marr Chase, Justin Jefferson and Tyreek Hill by recent ppg")
    assert shape["key"] == "recent_ppg" and len(shape["names"]) == 3
    shape = router.rank_shape("sort Alpha Receiver and Beta Receiver by ppg")
    assert shape["key"] == "ppg"
    shape = router.rank_shape(
        "Rank Alpha Receiver, Beta Receiver and Gamma Back by dynasty value")
    assert shape["key"] == "value"
    shape = router.rank_shape("Rank Alpha Receiver and Beta Receiver")
    assert shape["key"] == "value"  # default key


def test_shape_worth_more_form():
    shape = router.rank_shape("who is worth more, Garrett Wilson or Nico Collins")
    assert shape["names"] == ["garrett wilson", "nico collins"]
    assert shape["key"] == "value"


def test_shape_unsupported_key_is_named_not_substituted():
    shape = router.rank_shape(
        "rank Garrett Wilson and Nico Collins by touchdowns")
    assert shape["key"] == "value"
    assert shape["unsupported_key"] == "touchdowns"


# -- shape guards: governed shapes keep their routes -----------------------------

def test_shape_rejects_league_wide_boards():
    assert router.rank_shape("rank the wrs by value") is None
    assert router.rank_shape("rank the top wrs by value") is None
    assert router.rank_shape("rank the top 10 wrs by value") is None
    assert router.rank_shape("top wrs by value") is None
    assert router.rank_shape("rank the defenses by points") is None
    assert router.rank_shape("rank my team") is None


def test_shape_rejects_anchored_and_single_name_asks():
    assert router.rank_shape("start Garrett Wilson or Nico Collins") is None
    assert router.rank_shape(
        "should I start Garrett Wilson or Nico Collins this week") is None
    assert router.rank_shape("rank Justin Jefferson") is None
    assert router.rank_shape("compare Garrett Wilson and Nico Collins") is None
    assert router.rank_shape("what rb can I trade for aj brown") is None
    assert router.rank_shape(
        "rank Garrett Wilson and Nico Collins for the draft") is None


def test_keyword_route_rank_goes_open_with_every_name(keyword):
    out = _classify(keyword, "Rank Alpha Receiver, Beta Receiver and "
                             "Gamma Back by value")
    assert out["intent"] == "open"
    assert out["entities"]["player_names"] == [
        "alpha receiver", "beta receiver", "gamma back"]
    assert out["entities"]["rank_key"] == "value"


def test_keyword_route_guards_keep_their_intents(keyword):
    assert _classify(keyword, "rank the top wrs by value")["intent"] \
        == "leaderboard"
    assert _classify(keyword, "start alpha receiver or beta receiver")[
        "intent"] == "start_sit"


def test_model_route_correction_forces_open(monkeypatch):
    def _fake(system, payload, schema, name):
        return {"intent": "leaderboard", "player_names": [],
                "board_kind": "value"}
    monkeypatch.setattr(router, "call_model_json", _fake)
    out = _classify(router.classify,
                    "Rank Alpha Receiver, Beta Receiver and Gamma Back "
                    "by points per game")
    assert out["intent"] == "open"
    assert out["entities"]["player_names"] == [
        "alpha receiver", "beta receiver", "gamma back"]
    assert out["entities"]["rank_key"] == "ppg"


# -- the rank tool -----------------------------------------------------------------

def test_tool_ranks_by_value_not_mention_order(base):
    entry = tools_mod.run_tool(
        "rank_players",
        {"players": ["Alpha Receiver", "Beta Receiver", "Gamma Back"],
         "key": "value"}, _bundle())
    assert entry["ok"] is True
    assert [(r["name"], r["key_value"], r["rank"]) for r in entry["ranking"]] \
        == [("Gamma Back", 6100, 1), ("Alpha Receiver", 5000, 2),
            ("Beta Receiver", 4200, 3)]
    assert entry["unresolved"] == []
    assert [r["label"] for r in entry["receipts"]] == [
        "Gamma Back value", "Alpha Receiver value", "Beta Receiver value"]


def test_tool_ranks_by_the_stated_key(base, monkeypatch):
    profiles = {
        "1": {"player_id": "1", "name": "Alpha Receiver", "team": "CIN",
              "position": "WR", "value": 5000, "ppg": 12.0,
              "recent_ppg": 20.0},
        "2": {"player_id": "2", "name": "Beta Receiver", "team": "MIN",
              "position": "WR", "value": 4200, "ppg": 18.0,
              "recent_ppg": 9.0},
        "3": {"player_id": "3", "name": "Gamma Back", "team": "IND",
              "position": "RB", "value": 6100, "ppg": 15.0,
              "recent_ppg": 11.0},
    }
    monkeypatch.setattr(tools_mod, "_profile_lite",
                        lambda bundle, pid: dict(profiles[str(pid)]))
    by_ppg = tools_mod.run_tool(
        "rank_players",
        {"players": ["Alpha Receiver", "Beta Receiver", "Gamma Back"],
         "key": "ppg"}, _bundle())
    # PPG order (Beta, Gamma, Alpha) differs from the value order.
    assert [r["name"] for r in by_ppg["ranking"]] == [
        "Beta Receiver", "Gamma Back", "Alpha Receiver"]
    assert by_ppg["receipts"][0] == {"label": "Beta Receiver points per game",
                                     "value": "18.0"}


def test_tool_unresolved_name_carried_by_name_with_nearest(base):
    entry = tools_mod.run_tool(
        "rank_players",
        {"players": ["Gamma Bak", "Alpha Receiver", "Beta Receiver"],
         "key": "value"}, _bundle())
    assert entry["ok"] is True
    assert [r["name"] for r in entry["ranking"]] == [
        "Alpha Receiver", "Beta Receiver"]
    assert entry["unresolved"] == [
        {"asked": "Gamma Bak", "reason": "not_found",
         "nearest": ["Gamma Back"]}]


def test_tool_ambiguous_name_is_unresolved_with_candidates(base):
    entry = tools_mod.run_tool(
        "rank_players",
        {"players": ["Delta Smith", "Alpha Receiver", "Beta Receiver"],
         "key": "value"}, _bundle())
    assert entry["ok"] is True
    un = entry["unresolved"]
    assert len(un) == 1 and un[0]["reason"] == "ambiguous"
    assert len(un[0]["candidates"]) == 2


def test_tool_fewer_than_two_resolved_is_not_a_ranking(base):
    entry = tools_mod.run_tool(
        "rank_players",
        {"players": ["Gamma Bak", "Alpha Receiver"], "key": "value"},
        _bundle())
    assert entry["ok"] is False
    assert entry["not_found"] == "Gamma Bak"
    assert entry["nearest"] == ["Gamma Back"]
    assert [r["name"] for r in entry["ranking"]] == ["Alpha Receiver"]


def test_tool_unsupported_key_rides_the_entry(base):
    entry = tools_mod.run_tool(
        "rank_players",
        {"players": ["Alpha Receiver", "Beta Receiver"],
         "key": "value", "unsupported_key": "touchdowns"}, _bundle())
    assert entry["unsupported_key"] == "touchdowns"
    assert entry["key"] == "value"


# -- open layer: deterministic fetch, planner never improvises the set -------------

def test_plan_fetches_rank_before_planner(base, monkeypatch):
    def _boom(*a, **k):
        raise AssertionError("planner must not run for a rank shape")
    monkeypatch.setattr(open_layer, "call_model_json", _boom)
    pack, tools_called = open_layer.plan_and_fetch(
        "Rank Alpha Receiver, Beta Receiver and Gamma Back by value",
        _bundle(), thread={})
    assert tools_called == ["rank_players"]
    entry = pack["evidence"][0]
    assert [r["name"] for r in entry["ranking"]] == [
        "Gamma Back", "Alpha Receiver", "Beta Receiver"]


def test_answer_open_ranks_the_full_set(base, monkeypatch):
    import json as _json
    from dashboard_services.assistant import compose as compose_mod

    def _draft(system, payload, schema, name):
        payload = _json.loads(payload)
        entry = payload["evidence_pack"]["evidence"][0]
        lines = [f"{r['rank']}. {r['name']}: {r['key_value']}"
                 for r in entry["ranking"]]
        return {"verdict": f"{entry['ranking'][0]['name']} tops this "
                           "group by value",
                "text": " ".join(lines) + ".", "followups": []}
    monkeypatch.setattr(compose_mod, "call_model_json", _draft)
    answer, pack, tools_called, names = open_layer.answer_open(
        "Rank Alpha Receiver, Beta Receiver and Gamma Back by value",
        _bundle(), thread={})
    assert tools_called == ["rank_players"]
    assert answer["verdict"].startswith("Gamma Back")
    text = answer["text"]
    assert text.index("Gamma Back") < text.index("Alpha Receiver") \
        < text.index("Beta Receiver")
    assert names == ["Gamma Back", "Alpha Receiver", "Beta Receiver"]


def test_answer_open_gap_is_named_over_the_resolved(base):
    pack, tools_called = open_layer.plan_and_fetch(
        "Rank Gamma Bak, Alpha Receiver and Beta Receiver by value",
        _bundle(), thread={})
    entry = pack["evidence"][0]
    assert entry["ok"] is True and len(entry["ranking"]) == 2
    assert entry["unresolved"][0]["asked"] == "gamma bak"
    assert entry["unresolved"][0]["nearest"] == ["Gamma Back"]


def test_answer_open_declines_by_name_when_under_two_resolve(base):
    answer, pack, tools_called, names = open_layer.answer_open(
        "Rank Gamma Bak and Alpha Receiver by value", _bundle(), thread={})
    assert tools_called == ["rank_players"]
    assert answer["verdict"] is None
    assert "could not find gamma bak" in answer["text"]
    assert "Gamma Back" in answer["text"]  # nearest match offered
    assert "Only Alpha Receiver resolved" in answer["text"]
    assert any("Gamma Back" in (f.get("ask") or "")
               for f in answer["followups"])


def test_plan_board_question_still_uses_planner(base, monkeypatch):
    # The guard control at the fetch layer: a league-wide board ask is
    # not a rank shape, so no deterministic rank entry is produced.
    assert open_layer._deterministic_rank_entry(
        _bundle(), "rank the top wrs by value") is None
    assert open_layer._deterministic_rank_entry(
        _bundle(), "start alpha receiver or beta receiver") is None
