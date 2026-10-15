"""Router tests: the new P1 intents and their disambiguation.

The keyword fallback is exercised by forcing the model call to fail (the
same degradation path production takes on a rate limit); the model path
is exercised with a stubbed classifier response to pin the entity
passthrough for the new fields.
"""
import pytest

pytest.importorskip("openai")
pytestmark = pytest.mark.integration

from dashboard_services.assistant import router


@pytest.fixture()
def keyword(monkeypatch):
    def _boom(*a, **k):
        raise RuntimeError("model down")
    monkeypatch.setattr(router, "call_model_json", _boom)
    return router.classify


def _intent(classify, message):
    return classify(message, {"league_name": "Blackedraw", "week": 7, "season": 2026}, None)


# -- breakout family disambiguation ----------------------------------------------

def test_keyword_breakout_board(keyword):
    out = _intent(keyword, "Who are this week's breakout calls?")
    assert out["intent"] == "breakout_board"
    assert out["router_model"] == "keyword-fallback"


def test_keyword_breakout_track_record(keyword):
    assert _intent(keyword, "How good is BR at breakout calls?")["intent"] == "breakout_track_record"
    assert _intent(keyword, "What is the breakout track record?")["intent"] == "breakout_track_record"


def test_keyword_breakout_history_vs_board(keyword):
    # A single player's call history is a different question from the board.
    assert _intent(keyword, "Was Rome Odunze a breakout call?")["intent"] == "breakout_history"
    assert _intent(keyword, "Did BR call anyone a breakout this week?")["intent"] == "breakout_history"
    assert _intent(keyword, "Show me the breakout board")["intent"] == "breakout_board"


# -- pick ownership vs trade eval ---------------------------------------------------

def test_keyword_pick_ownership(keyword):
    assert _intent(keyword, "Who owns my 2027 first?")["intent"] == "pick_ownership"
    assert _intent(keyword, "Whose pick is the 2027 1.03?")["intent"] == "pick_ownership"


def test_keyword_trade_with_picks_stays_trade_eval(keyword):
    out = _intent(keyword, "Should I trade my 2027 1st and Woody Marks for Zay Flowers?")
    assert out["intent"] == "trade_eval"


# -- teams ---------------------------------------------------------------------------

def test_keyword_teams_questions(keyword):
    out = _intent(keyword, "Whose roster is strongest?")
    assert out["intent"] == "teams"
    out = _intent(keyword, "Who is weakest at TE?")
    assert out["intent"] == "teams"
    assert out["entities"]["position_hint"] == "TE"
    out = _intent(keyword, "Grade every team's WR room")
    assert out["intent"] == "teams"
    assert out["entities"]["position_hint"] == "WR"


def test_keyword_teams_mode_this_season(keyword):
    out = _intent(keyword, "Grade my team's WR room this season")
    assert out["intent"] == "teams"
    assert out["entities"]["teams_mode"] == "this_season"


def test_keyword_teams_does_not_steal_digest_or_metric(keyword):
    assert _intent(keyword, "What should I do with my team this week?")["intent"] == "this_week"
    assert _intent(keyword, "What is target share?")["intent"] == "metric"
    assert _intent(keyword, "Start Odunze or Addison at flex?")["intent"] == "start_sit"


# -- model path: entity passthrough ----------------------------------------------------

def test_model_entities_carry_new_fields(monkeypatch):
    def _fake(system, payload, schema, name):
        return {"intent": "trade_eval", "player_names": [],
                "give_player_names": ["Woody Marks"], "receive_player_names": ["Zay Flowers"],
                "picks": [], "give_picks": ["my 2027 1st"], "receive_picks": [],
                "other_team_hint": None, "metric_hint": None, "position_hint": None,
                "teams_mode": None, "week": None, "preference_text": None}
    monkeypatch.setattr(router, "call_model_json", _fake)
    out = router.classify("Trade Marks and my 2027 1st for Flowers", {}, None)
    assert out["intent"] == "trade_eval"
    assert out["entities"]["give_picks"] == ["my 2027 1st"]
    assert out["entities"]["receive_picks"] == []


def test_model_teams_entities(monkeypatch):
    def _fake(system, payload, schema, name):
        return {"intent": "teams", "player_names": [], "give_player_names": [],
                "receive_player_names": [], "picks": [], "give_picks": [], "receive_picks": [],
                "other_team_hint": None, "metric_hint": None, "position_hint": "TE",
                "teams_mode": "this_season", "week": None, "preference_text": None}
    monkeypatch.setattr(router, "call_model_json", _fake)
    out = router.classify("Who is weakest at TE this season?", {}, None)
    assert out["intent"] == "teams"
    assert out["entities"]["position_hint"] == "TE"
    assert out["entities"]["teams_mode"] == "this_season"


def test_new_intents_registered_with_builders():
    from dashboard_services.assistant.evidence import BUILDERS
    for intent in ("breakout_board", "breakout_track_record", "breakout_history",
                   "teams", "pick_ownership"):
        assert intent in router.INTENTS
        assert intent in BUILDERS
