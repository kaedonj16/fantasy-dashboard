"""Front Office pack: PRO gate, distillation, routing, compose guarantees.

The gate mirrors /api/gm-memo exactly (subscriptions.has_premium_for_viewer
with the bundle viewer identity, before any report data is built), and the
pack carries the report's computed DATA only
(ai.front_office_report.build_front_office_data), never its written memo.
Both sources are monkeypatched at the evidence module's lazy shims.
"""
import pytest

pytest.importorskip("openai")
pytestmark = pytest.mark.integration

from dashboard_services.assistant import compose, evidence, router

FAKE_DATA = {
    "direction": "contender", "record": "5-2", "playoff_pct": 78.0,
    "weakest_positions": ["TE"],
    "top_moves": [{"line": "Add Kaleb Johnson and drop Tank Bigsby",
                   "type": "waiver", "why": "Top waiver target."}],
    "trade_targets": [{"partner": "Pittsburgh Pilots",
                       "gets": [{"id": "a5", "name": "Zay Flowers", "position": "WR",
                                 "value": 4680.0}],
                       "gives": [{"id": "b2", "name": "Woody Marks", "position": "RB",
                                  "value": 3400.0},
                                 {"id": "c1", "name": "Jordan Addison", "position": "WR",
                                  "value": 1350.0}],
                       "fairness_label": "Fair"}],
    "waiver_targets": [{"id": "w1", "name": "Kaleb Johnson", "position": "RB",
                        "value": 900.0}],
    "cut_candidates": [{"id": "x1", "name": "Tank Bigsby", "position": "RB",
                        "value": 640.0}],
    "ai_memo_prose": "THIS PROSE MUST NEVER REACH THE PACK",
}


def _bundle():
    return {"platform": "sleeper", "league_id": "blackedraw", "season": 2026,
            "league_name": "Blackedraw", "team_name": "Caleb's Casting Couch",
            "week": 7, "ctx": {"rosters": []}, "roster": {"roster_id": 2},
            "viewer": {"viewer_username": "hoodiekj1", "viewer_user_id": "u1"}}


def test_gate_blocks_before_any_data_is_built(monkeypatch):
    monkeypatch.setattr(evidence, "has_premium_for_viewer", lambda *a, **k: False)

    def _boom(*a, **k):
        raise AssertionError("report data must not be built for a gated caller")
    monkeypatch.setattr(evidence, "front_office_data", _boom)
    pack, _tools = evidence.build_front_office(_bundle(), {}, [])
    assert pack["gated"] is True
    assert "direction" not in pack and "trade_targets" not in pack


def test_builder_distills_data_only(monkeypatch):
    monkeypatch.setattr(evidence, "has_premium_for_viewer", lambda *a, **k: True)
    monkeypatch.setattr(evidence, "front_office_data", lambda ctx, rid: FAKE_DATA)
    pack, tools = evidence.build_front_office(_bundle(), {}, [])
    assert "ai.front_office_report.build_front_office_data" in tools
    assert pack["direction"] == "contender"
    assert pack["playoff_pct"] == 78.0
    target = pack["trade_targets"][0]
    assert target["partner"] == "Pittsburgh Pilots"
    assert target["get_total"] == 4680.0 and target["give_total"] == 4750.0
    assert [g["name"] for g in target["gets"]] == ["Zay Flowers"]
    assert pack["waiver_targets"][0]["name"] == "Kaleb Johnson"
    assert pack["cut_candidates"][0]["name"] == "Tank Bigsby"
    assert "ai_memo_prose" not in pack


def test_builder_no_data_is_no_data(monkeypatch):
    monkeypatch.setattr(evidence, "has_premium_for_viewer", lambda *a, **k: True)
    monkeypatch.setattr(evidence, "front_office_data", lambda ctx, rid: None)
    pack, _tools = evidence.build_front_office(_bundle(), {}, [])
    assert pack["available"] is False


def test_front_office_registered():
    assert evidence.BUILDERS["front_office"] is evidence.build_front_office
    assert "front_office" in router.INTENTS


# -- routing ---------------------------------------------------------------------

@pytest.fixture()
def keyword(monkeypatch):
    def _boom(*a, **k):
        raise RuntimeError("model down")
    monkeypatch.setattr(router, "call_model_json", _boom)
    return router.classify


def _intent(classify, message):
    return classify(message, {"league_name": "Blackedraw", "week": 7, "season": 2026}, None)


@pytest.mark.parametrize("message", [
    "What does my front office report say?",
    "Pull up my GM memo",
    "Front office top moves please",
])
def test_keyword_front_office(keyword, message):
    assert _intent(keyword, message)["intent"] == "front_office"


def test_keyword_front_office_does_not_steal_this_week(keyword):
    assert _intent(keyword, "What should I do with my team this week?")["intent"] == "this_week"


# -- compose guarantees ------------------------------------------------------------

def _golden_pack(qid):
    import json, pathlib
    data = json.loads(pathlib.Path("assistant_eval/questions.json").read_text())
    return next(q["evidence_pack"] for q in data["questions"] if q["id"] == qid)


def test_gated_answer_names_pro_and_leaks_nothing():
    ans = compose.gated_answer(_golden_pack("fo1"))
    assert "PRO" in ans["verdict"]
    assert ans["receipts"] == []
    assert ans["act_link"] == {"href": "/sleeper/2026/blackedraw/pricing",
                               "label": "See PRO plans"}


def test_front_office_required_receipts():
    rows = compose._required_receipts("front_office", _golden_pack("fo2"))
    assert rows == [
        {"label": "Playoff odds (%)", "value": "78"},
        {"label": "Pittsburgh Pilots trade: you receive (total value)", "value": "4680"},
        {"label": "Pittsburgh Pilots trade: you give (total value)", "value": "4750"}]


def test_front_office_required_mentions_and_injection():
    pack = _golden_pack("fo2")
    assert compose._required_mentions("front_office", pack) == [("Pittsburgh Pilots", "front_office")]
    data = {"verdict": "Contender", "text": "Your odds look good.",
            "caveat": None, "receipts": [], "followups": []}
    out = compose._ensure_required_mentions("front_office", pack, data, [])
    assert "Pittsburgh Pilots" in out["text"]


def test_front_office_act_link():
    assert compose._resolve_act_link("front_office", _golden_pack("fo2")) == {
        "href": "/sleeper/2026/blackedraw/dashboard", "label": "View Front Office"}
