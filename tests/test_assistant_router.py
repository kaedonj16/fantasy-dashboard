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


# -- win/lose outcome questions are matchup, never the digest -------------------
# Kaedon's live transcript: "Will I win this week?" routed to this_week,
# whose pack has no win probability, so the answer honestly reported no
# read and padded with digest waiver actions. The matchup builder has
# the number (build_matchup_preview + compute_win_prob); routing was
# the failure, in the keyword fallback AND the model prompt's wording.

def test_keyword_win_outcome_questions_route_to_matchup(keyword):
    for q in ("Will I win this week?", "Will I win?", "Am I going to win this week?",
              "Do I beat Trustfund QBs this week?", "What are my chances this week?",
              "Am I projected to win?", "Will I lose this week?", "Will I win my matchup?"):
        assert _intent(keyword, q)["intent"] == "matchup", q


def test_keyword_plan_questions_stay_this_week(keyword):
    # The hard constraint: the digest ask that works today must not move.
    for q in ("What should I do with my team", "What should I do this week?",
              "What should I do with my team this week?"):
        assert _intent(keyword, q)["intent"] == "this_week", q


def test_keyword_outcome_rule_does_not_steal_neighbors(keyword):
    assert _intent(keyword, "What are my chances of making the playoffs?")["intent"] == "playoffs"
    assert _intent(keyword, "Who should I add from waivers to win this week?")["intent"] == "waivers"
    assert _intent(keyword, "Break down my matchup")["intent"] == "matchup"


# -- Class E battery: keyword-fallback gaps (deep-dive audit) -------------------
# Bare player questions carried no anchor and fell to out_of_scope, so the
# service declined before entity resolution ever ran. A question-shaped
# message with a name-like token now falls through to player, with the
# name extracted for the resolver.

def test_keyword_bare_player_questions_fall_through_to_player(keyword):
    out = _intent(keyword, "How is Puka Nacua doing this season?")
    assert out["intent"] == "player"
    assert out["entities"]["player_names"] == ["Puka Nacua"]
    out = _intent(keyword, "Tell me about Bijan Robinson")
    assert out["intent"] == "player"
    assert out["entities"]["player_names"] == ["Bijan Robinson"]
    out = _intent(keyword, "Is Josh Allen healthy?")
    assert out["intent"] == "player"
    assert out["entities"]["player_names"] == ["Josh Allen"]
    out = _intent(keyword, "How many points did Chase score last week?")
    assert out["intent"] == "player"
    assert out["entities"]["player_names"] == ["Chase"]


def test_keyword_player_fallthrough_does_not_claim_real_nfl_trivia(keyword):
    # "Who wins the game" is the user's phrasing for a real game, defined
    # as out_of_scope in the SYSTEM prompt; the team names in it are not
    # adjacent and sit in no name slot, so the fallthrough must not fire.
    assert _intent(keyword, "Who wins the game tonight, Chiefs or Bills?")["intent"] == "out_of_scope"
    # The user's own game stays a matchup question (#2626 outcome rule).
    assert _intent(keyword, "Who wins my game this week?")["intent"] == "matchup"


def test_keyword_start_sit_play_and_bench_anchors(keyword):
    assert _intent(keyword, "Who should I play at QB this week, Allen or Mahomes?")["intent"] == "start_sit"
    assert _intent(keyword, "Bench Chase or play Higgins on Sunday?")["intent"] == "start_sit"


def test_keyword_trade_verb_forms_and_accept_anchor(keyword):
    assert _intent(keyword, "Is trading Lamb for a 2027 first fair?")["intent"] == "trade_eval"
    assert _intent(keyword, "Should I accept Gibbs for Bijan and a second?")["intent"] == "trade_eval"


def test_keyword_trade_deadline_date_question_is_out_of_scope(keyword):
    assert _intent(keyword, "When is the trade deadline?")["intent"] == "out_of_scope"
    # A finding ask that merely mentions the deadline still finds a trade.
    assert _intent(keyword, "Who should I target before the trade deadline?")["intent"] == "trade_finding"


def test_keyword_standings_seed_and_points_leader(keyword):
    assert _intent(keyword, "What seed am I right now?")["intent"] == "standings"
    assert _intent(keyword, "Who has scored the most points in the league?")["intent"] == "standings"


def test_keyword_win_precedence_and_this_week_protections_hold(keyword):
    # The #2626 controls, re-run against the widened rules: outcome
    # questions stay matchup, plan questions stay this_week, and the
    # waiver add keeps its anchor over any win wording.
    assert _intent(keyword, "Will I win this week?")["intent"] == "matchup"
    assert _intent(keyword, "What should I do with my team this week?")["intent"] == "this_week"
    assert _intent(keyword, "What should I do with my team")["intent"] == "this_week"
    assert _intent(keyword, "Who should I add from waivers to win this week?")["intent"] == "waivers"
    assert _intent(keyword, "What are my chances of making the playoffs?")["intent"] == "playoffs"
    # Metric anchors still outrank the player fallthrough.
    assert _intent(keyword, "What is Puka Nacua's target share?")["intent"] == "metric"


def test_router_prompt_carries_the_outcome_precedence():
    # The model path gets the same rule in words: outcome questions are
    # matchup even when they say "this week"; this_week is plan/actions.
    assert "will I win" in router.SYSTEM
    assert "Outcome questions are matchup" in router.SYSTEM
    assert "NOT whether the user will win" in router.SYSTEM


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
