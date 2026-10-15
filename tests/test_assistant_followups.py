"""Follow-up chips are {label, ask} pairs (Kaedon, 2026-10-15).

The chip the user sees is the label; the message a tap sends is the ask.
Labels were phrased as offers TO the user ("Want the cheapest buy-in for
Zay Flowers?") and the chip sent its label verbatim, so the user appeared
to "say" the assistant's offer. Now every producer emits pairs, the ask
is a natural question in the user's own voice, an offer-style string with
no ask is rejected rather than reworded, and the eval scorer reads both
surfaces (an invented number in a label or an ask is still invented).
"""
import pytest

pytest.importorskip("openai")
pytestmark = pytest.mark.integration

from dashboard_services.assistant import compose as compose_mod
from dashboard_services.assistant.compose import (
    SCHEMA,
    SYSTEM,
    _clean_followups,
    compose_answer,
    gated_answer,
    no_data_answer,
)
from assistant_eval.scorer import answer_numbers

CHEAPEST_LABEL = "Want the cheapest buy-in for Zay Flowers?"
CHEAPEST_ASK = "What is the cheapest way to buy Zay Flowers?"
CHEAPEST_PAIR = {"label": CHEAPEST_LABEL, "ask": CHEAPEST_ASK}


def _canned(monkeypatch, *outputs):
    calls = {"n": 0}

    def fake(system, user, schema, name):
        calls["n"] += 1
        return dict(outputs[min(calls["n"] - 1, len(outputs) - 1)])

    monkeypatch.setattr(compose_mod, "call_model_json", fake)
    return calls


# -- Kaedon's exact example -----------------------------------------------------

def test_cheapest_buy_in_pair_survives_cleaning_verbatim():
    # The label stays the inviting offer; the ask is the user's question.
    assert _clean_followups("trade_finding", [dict(CHEAPEST_PAIR)]) == [CHEAPEST_PAIR]


def test_cheapest_buy_in_pair_reaches_the_answer(monkeypatch):
    pack = {"type": "trade_finding", "available": True, "week": 7,
            "findings": [], "margin": {"thin": False}, "as_of": {"week": 7}}
    draft = {"verdict": None, "text": "There is a path to Flowers this week.",
             "receipts": [], "caveat": None,
             "followups": [dict(CHEAPEST_PAIR)], "act_link": None}
    _canned(monkeypatch, draft)
    answer = compose_answer("trade_finding", "Find me a trade", pack)
    assert answer["followups"] == [CHEAPEST_PAIR]
    # The chip sends the ask, so the ask is what the user "says".
    assert answer["followups"][0]["ask"] == "What is the cheapest way to buy Zay Flowers?"
    assert answer["followups"][0]["label"] == "Want the cheapest buy-in for Zay Flowers?"


def test_offer_string_without_an_ask_is_rejected_not_reworded():
    # A bare offer string carries no user-voiced ask: it is dropped and
    # the deterministic trade-finding templates refill, never converted.
    out = _clean_followups("trade_finding", [CHEAPEST_LABEL])
    assert out == compose_mod._FOLLOWUP_TEMPLATES["trade_finding"]
    assert all(not f["ask"].lower().startswith("want") for f in out)


def test_pair_with_an_offer_for_an_ask_is_rejected():
    out = _clean_followups("trade_finding", [{"label": "Cheapest buy-in?",
                                              "ask": "Want me to find the cheapest buy-in?"}])
    assert out == compose_mod._FOLLOWUP_TEMPLATES["trade_finding"]


# -- normalization ----------------------------------------------------------------

def test_legacy_string_followup_becomes_a_pair():
    assert _clean_followups("start_sit", ["What about next week?"]) == [
        {"label": "What about next week?", "ask": "What about next week?"}]


def test_pair_with_system_ask_is_dropped():
    out = _clean_followups("start_sit", [{"label": "More data?", "ask": "Send your roster"}])
    assert out == compose_mod._FOLLOWUP_TEMPLATES["start_sit"]


def test_schema_and_prompt_carry_the_pair_contract():
    items = SCHEMA["properties"]["followups"]["items"]
    assert items["required"] == ["label", "ask"]
    assert items["additionalProperties"] is False
    # The prompt teaches the split with Kaedon's example, both halves.
    assert CHEAPEST_LABEL in SYSTEM
    assert CHEAPEST_ASK in SYSTEM


# -- template + producer coverage ---------------------------------------------------

def test_every_template_emits_both_fields():
    for intent, entries in compose_mod._FOLLOWUP_TEMPLATES.items():
        assert entries, intent
        for entry in entries:
            assert isinstance(entry, dict), (intent, entry)
            assert set(entry) == {"label", "ask"}, (intent, entry)
            assert entry["label"] and entry["ask"], (intent, entry)
            # The ask is the user's voice: never an offer, never a
            # system instruction, never internal language.
            assert not compose_mod._OFFER_RE.search(entry["ask"]), (intent, entry)
            assert not compose_mod._SYSTEM_ASK_RE.search(entry["ask"]), (intent, entry)
            assert not compose_mod._INTERNAL_RE.search(entry["ask"]), (intent, entry)
            # And each template survives the cleaner unchanged.
            assert _clean_followups(intent, [dict(entry)]) == [entry], (intent, entry)


def test_deterministic_producers_emit_pairs():
    answer = no_data_answer({"available": False, "missing": "routes not tracked",
                             "nearest": ["target share"]})
    assert answer["followups"] == [
        {"label": "What is target share?", "ask": "What is target share?"}]

    gated_tf = gated_answer({"type": "trade_finding", "league": {}})
    assert gated_tf["followups"]
    assert all(set(f) == {"label", "ask"} for f in gated_tf["followups"])

    gated_metric = gated_answer({"type": "leaderboard", "label": "WOPR"})
    assert gated_metric["followups"]
    assert all(set(f) == {"label", "ask"} for f in gated_metric["followups"])


# -- scorer coverage ------------------------------------------------------------------

def test_scorer_reads_numbers_from_label_and_ask():
    base = {"verdict": None, "text": "Fine.", "receipts": [], "caveat": None, "actions": []}
    in_ask = dict(base, followups=[{"label": "Flowers?", "ask": "Is he worth 31.4?"}])
    assert any("31.4" in n for n in answer_numbers(in_ask))
    in_label = dict(base, followups=[{"label": "Worth 9999?", "ask": "Is he good?"}])
    assert any("9999" in n for n in answer_numbers(in_label))
    legacy = dict(base, followups=["Is he worth 31.4?"])
    assert any("31.4" in n for n in answer_numbers(legacy))
