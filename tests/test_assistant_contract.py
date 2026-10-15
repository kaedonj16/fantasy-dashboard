"""UI contract + prose grounding tests for the assistant composer.

Contract: the answer dict compose_answer returns is exactly what
static/app.js initAssistant consumes:
  answer.verdict / answer.text / answer.caveat   (strings or None)
  answer.receipts[]                              {label, value} strings
  answer.followups[]                             strings
  answer.act_link                                {href, label} object or None
  answer.actions[]  (this_week)                  {tag, headline, detail,
                                                  href, cta_label, key_number}
  answer.candidates[] (clarify, service-level)   {name, position, team, player_id}

Grounding: numbers stated in prose (verdict, text, caveat, follow-ups) are
verified against the evidence pack, not just receipt numbers. A poisoned
draft is regenerated once; a still-poisoned retry is stripped, so an
invented number never reaches the response.
"""
import pytest

pytest.importorskip("openai")
pytestmark = pytest.mark.integration

from dashboard_services.assistant import compose as compose_mod
from dashboard_services.assistant.compose import compose_answer, no_data_answer

PACK = {
    "type": "start_sit", "available": True, "week": 7,
    "players": [
        {"player_id": "1", "name": "Rome Odunze", "team": "CHI", "position": "WR",
         "projection": 17.9, "start_score": 82.5},
        {"player_id": "2", "name": "Darnell Mooney", "team": "ATL", "position": "WR",
         "projection": 11.0, "start_score": 61.0},
    ],
    "margin": {"thin": False}, "as_of": {"week": 7},
}

CLEAN = {"verdict": "START Rome Odunze",
         "text": "Odunze projects 17.9 points against 11.0 for Mooney.",
         "receipts": [{"label": "Odunze projection", "value": "17.9"}],
         "caveat": None, "followups": ["What about next week?"], "act_link": None}


def _canned(monkeypatch, *outputs):
    calls = {"n": 0}

    def fake(system, user, schema, name):
        calls["n"] += 1
        return dict(outputs[min(calls["n"] - 1, len(outputs) - 1)])

    monkeypatch.setattr(compose_mod, "call_model_json", fake)
    return calls


def _assert_ui_shape(answer):
    assert answer["verdict"] is None or isinstance(answer["verdict"], str)
    assert isinstance(answer["text"], str)
    assert answer["caveat"] is None or isinstance(answer["caveat"], str)
    for r in answer["receipts"]:
        assert isinstance(r["label"], str) and isinstance(r["value"], str)
    for f in answer["followups"]:
        assert isinstance(f, str)
    link = answer["act_link"]
    assert link is None or (
        isinstance(link, dict)
        and isinstance(link["href"], str) and link["href"].startswith("/")
        and isinstance(link["label"], str) and link["label"]
    )
    for a in answer["actions"]:
        assert set(("tag", "headline", "detail", "href", "cta_label", "key_number")) <= set(a)
        assert isinstance(a["href"], str) and a["href"].startswith("/")
        assert isinstance(a["cta_label"], str) and a["cta_label"]


def test_act_link_is_object_with_deterministic_target(monkeypatch):
    _canned(monkeypatch, CLEAN)
    answer = compose_answer("start_sit", "Start Odunze or Mooney?", PACK)
    _assert_ui_shape(answer)
    # The model's own act_link (None here) never wins: the link is code-assigned.
    assert answer["act_link"] == {"href": "/waivers?tab=startsit", "label": "Open Start/Sit"}


def test_no_data_answer_matches_contract():
    answer = no_data_answer({"available": False, "missing": "routes not tracked",
                             "nearest": ["target_share"]})
    _assert_ui_shape(answer)
    assert answer["act_link"]["href"] == "/advanced-metrics"


def test_this_week_actions_carry_href_cta_and_key_number(monkeypatch):
    pack = {"type": "this_week", "available": True, "week": 7, "as_of": {},
            "actions": [
                {"tag": "lineup", "headline": "Start Odunze at FLEX", "detail": "Mooney is on bye",
                 "key_number": {"value": "+1.9", "label": "proj"}, "act_link": "/waivers?tab=startsit"},
                {"tag": "waiver", "headline": "Add Woody Marks", "detail": "Pierce is droppable",
                 "key_number": "85%", "act_link": "/waivers"},
                {"tag": "trade", "headline": "Offer Addison for Thomas", "detail": "",
                 "key_number": None, "act_link": "/trade-hub"},
            ]}
    _canned(monkeypatch, {"verdict": "Start Odunze at FLEX", "text": "Three moves.",
                          "receipts": [], "caveat": None, "followups": [], "act_link": None})
    answer = compose_answer("this_week", "What should I do this week?", pack)
    _assert_ui_shape(answer)
    lineup, waiver, trade = answer["actions"]
    assert lineup["href"] == "/waivers?tab=startsit"
    assert lineup["cta_label"] == "Set lineup"
    assert lineup["key_number"] == {"value": "+1.9", "label": "proj"}
    assert waiver["href"] == "/waivers" and waiver["cta_label"] == "View waivers"
    assert waiver["key_number"] == "85%"
    assert trade["href"] == "/trade-hub" and trade["cta_label"] == "Build offer"
    assert trade["key_number"] is None


def test_poisoned_prose_number_is_regenerated_then_clean(monkeypatch):
    poisoned = dict(CLEAN)
    poisoned["text"] = "Odunze projects 17.9 points and averages 31.4 yards per route."
    calls = _canned(monkeypatch, poisoned, CLEAN)
    answer = compose_answer("start_sit", "Start Odunze or Mooney?", PACK)
    assert calls["n"] == 2  # one strict regeneration
    assert "31.4" not in answer["text"]
    assert answer["text"] == CLEAN["text"]


def test_poisoned_prose_number_is_stripped_when_retry_still_poisoned(monkeypatch):
    poisoned = dict(CLEAN)
    poisoned["text"] = ("Odunze projects 17.9 points against 11.0 for Mooney. "
                         "He also averages 31.4 yards per route run.")
    poisoned["followups"] = ["Why is his 31.4 yards per route so high?"]
    calls = _canned(monkeypatch, poisoned, poisoned)
    answer = compose_answer("start_sit", "Start Odunze or Mooney?", PACK)
    assert calls["n"] == 2
    # The invented stat reaches no surface: prose sentence stripped,
    # follow-up carrying it dropped, receipts were already pack-clean.
    assert "31.4" not in answer["text"]
    assert "17.9" in answer["text"]
    assert all("31.4" not in f for f in answer["followups"])
    assert all("31.4" not in r["value"] for r in answer["receipts"])


def test_invented_receipt_still_dropped_and_percent_grounds(monkeypatch):
    draft = dict(CLEAN)
    draft["text"] = "Odunze owns a 17.9 target share."  # prose number lives in the pack (projection)
    draft["receipts"] = [{"label": "Odunze projection", "value": "17.9"},
                         {"label": "Odunze target share", "value": "17.9%"},
                         {"label": "Invented", "value": "9999"}]
    _canned(monkeypatch, draft)
    answer = compose_answer("start_sit", "Start Odunze or Mooney?", PACK)
    assert answer["receipts"] == [{"label": "Odunze projection", "value": "17.9"},
                                  {"label": "Odunze target share", "value": "17.9%"}]
