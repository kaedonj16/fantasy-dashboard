"""Parity 1: assistant directive answers from the GM brief.

The moves / next-steps phrasings route to the new gm_directive
intent; the bare "what should I do" forms stay with this_week per
the settled hard constraint in test_assistant_router, and specific
moves keep their own intents. The evidence pack assembles decided
calls from build_gm_brief: the race line, claim cards, coverage
holes and stash holds.
"""
import pytest

pytest.importorskip("pandas")

from dashboard_services import gm_brief
from dashboard_services.assistant import evidence, router


@pytest.mark.parametrize("msg", [
    "what are my next steps",
    "tell me what to do this week",
    "what moves should I make?",
    "what's my next move?",
    "what do I need to do?",
])
def test_directive_phrasings_route_to_gm_directive(msg):
    assert router._keyword_route(msg)["intent"] == "gm_directive"


@pytest.mark.parametrize("msg", [
    "What should I do with my team",
    "What should I do this week?",
    "What should I do with my team this week?",
])
def test_bare_what_should_i_do_stays_this_week(msg):
    # The settled hard constraint (test_assistant_router's
    # test_keyword_plan_questions_stay_this_week): the digest ask
    # that works today must not move. gm_directive owns only the
    # moves / next-steps phrasings.
    assert router._keyword_route(msg)["intent"] == "this_week"


@pytest.mark.parametrize("msg,expected", [
    ("who should I start at FLEX?", "start_sit"),
    ("should I trade Gibbs for Chase?", "trade_eval"),
    ("what does my front office report say?", "front_office"),
])
def test_specific_moves_keep_their_intents(msg, expected):
    assert router._keyword_route(msg)["intent"] == expected


def test_directive_intent_is_registered():
    assert "gm_directive" in router.INTENTS
    assert evidence.BUILDERS["gm_directive"] is evidence.build_gm_directive


def _brief():
    return {
        "race_line": {"text": "Win and the 8 seed is yours to lose",
                      "path": {"win_and_in": True,
                               "note": "Butter Boys lose the tiebreak"}},
        "claims": [{"action": "Add Sam Darnold",
                    "brief_reason": "No QB in Week 11",
                    "funding_text": "Drops a bench RB",
                    "culture_note": None, "why": None}],
        # The brief's coverage map rides under "qb_coverage" (the
        # build_coverage dict), with holes in build_coverage's shape.
        "qb_coverage": {"holes": [{"week": 12, "position": "TE",
                                   "count": 0, "required": 1,
                                   "causes": [], "fills_action": None}],
                        "weeks": [], "missing_weeks": [],
                        "through_week": 14, "label": None,
                        "roles_confirmed": True, "requirements": {}},
        "return_horizons": [{"name": "Keeper Stash", "return_week": 10,
                             "weeks_of_coverage": 5}],
        "league_culture": {"signals": []},
    }


def test_pack_assembles_decided_calls(monkeypatch):
    monkeypatch.setattr(gm_brief, "build_gm_brief",
                        lambda ctx, rid, actions=None: _brief())
    bundle = {"ctx": {"league": {}}, "roster": {"roster_id": "7"},
              "league_id": "L1", "platform": "fleaflicker",
              "season": 2026}
    pack, tools = evidence.build_gm_directive(bundle)
    kinds = [d["kind"] for d in pack["directives"]]
    assert kinds == ["race", "waiver", "coverage", "stash"]
    assert pack["directives"][0]["call"] == "Win this week and you are in"
    assert "No QB in Week 11" in pack["directives"][1]["why"]
    assert "Drops a bench RB" in pack["directives"][1]["why"]
    assert pack["directives"][3]["why"].startswith("Back Week 10")
    assert "gm_brief.build_gm_brief" in tools


def test_pack_without_roster_is_unavailable():
    pack, _tools = evidence.build_gm_directive(
        {"ctx": {}, "roster": {}})
    assert pack["available"] is False
