"""League activity pack: builder, routing, and compose guarantees.

The activity frame is the Activity page's own data (app.ensure_activity_bits
over dashboard_services.service.build_week_activity). The builder test fakes
the frame as a DataFrame of row dicts in the exact shape build_week_activity
returns, and prices players through the evidence module's value-table shim.
"""
import pytest

pytest.importorskip("openai")
pytest.importorskip("pandas")
pytestmark = pytest.mark.integration

import pandas as pd

from dashboard_services.assistant import compose, evidence, router

VALUES = [
    {"id": "a5", "value": 4680.0},   # Zay Flowers
    {"id": "b2", "value": 3400.0},   # Woody Marks
    {"id": "x1", "value": 1350.0},   # Jordan Addison
    {"id": "w1", "value": 900.0},    # Kaleb Johnson
]


def _frame():
    rows = [
        {"kind": "trade", "week": 7, "ts": pd.Timestamp("2026-10-12", tz="UTC"),
         "data": {"teams": [
             {"rid": "1", "roster_id": 1, "name": "Pittsburgh Pilots",
              "gets": [{"pid": "a5", "name": "Zay Flowers", "pos": "WR"}],
              "sends": [{"pid": "b2", "name": "Woody Marks", "pos": "RB"},
                        {"pid": "x1", "name": "Jordan Addison", "pos": "WR"}]},
             {"rid": "2", "roster_id": 2, "name": "Caleb's Casting Couch",
              "gets": [{"pid": "b2", "name": "Woody Marks", "pos": "RB"},
                       {"pid": "x1", "name": "Jordan Addison", "pos": "WR"}],
              "sends": [{"pid": "a5", "name": "Zay Flowers", "pos": "WR"}]}],
             "draft_picks": []}},
        {"kind": "waiver", "week": 7, "ts": pd.Timestamp("2026-10-11", tz="UTC"),
         "data": {"rid": "3", "name": "Gridiron Goblins",
                  "adds": [{"pid": "w1", "name": "Kaleb Johnson", "pos": "RB"}]}},
        {"kind": "injury", "week": 7, "ts": pd.Timestamp("2026-10-10", tz="UTC"),
         "data": {"player": {"name": "Drake London", "pos": "WR", "team": "ATL", "pid": "i1"},
                  "status": "OUT", "body": "Ankle", "team_name": "Sacko Squad"}},
        {"kind": "waiver", "week": 6, "ts": pd.Timestamp("2026-10-04", tz="UTC"),
         "data": {"rid": "3", "name": "Gridiron Goblins",
                  "adds": [{"pid": "w9", "name": "Old Add", "pos": "RB"}]}},
    ]
    return pd.DataFrame(rows, columns=["kind", "week", "ts", "data"])


def _bundle(ctx):
    return {"platform": "sleeper", "league_id": "blackedraw", "season": 2026,
            "league_name": "Blackedraw", "team_name": "Caleb's Casting Couch",
            "week": 7, "ctx": ctx, "roster": {"roster_id": 2}}


@pytest.fixture()
def priced(monkeypatch):
    monkeypatch.setattr(evidence, "get_model_value_table_cached", lambda *a, **k: VALUES)
    monkeypatch.setattr(evidence, "ensure_activity_bits", lambda ctx: None)


def test_activity_builder_prices_latest_trade(priced):
    pack, tools = evidence.build_activity(_bundle({"activity_df": _frame()}), {}, [])
    assert pack["available"] is True
    assert "service.build_week_activity" in tools
    assert pack["counts"] == {"trades": 1, "trades_this_week": 1,
                              "waiver_adds": 2, "injuries": 1}
    latest = pack["latest_trade"]
    assert latest["week"] == 7
    assert latest["value_gap"] == 70.0
    assert latest["fairness"] == "fair"
    assert latest["ahead_team"] == "Caleb's Casting Couch"
    pilots, couch = latest["teams"]
    assert pilots["gets_value"] == 4680.0 and pilots["involves_you"] is False
    assert couch["gets_value"] == 4750.0 and couch["involves_you"] is True
    assert len(pack["recent_adds"]) == 2
    assert pack["recent_adds"][0]["team_name"] == "Gridiron Goblins"
    assert pack["recent_adds"][0]["adds"] == ["Kaleb Johnson"]
    assert pack["injuries"] == [{"player": "Drake London", "status": "OUT",
                                 "team_name": "Sacko Squad"}]


def test_activity_builder_no_ctx_is_no_data(priced):
    pack, _tools = evidence.build_activity(_bundle(None), {}, [])
    assert pack["available"] is False


def test_activity_builder_unbuilt_frame_is_no_data(priced):
    pack, _tools = evidence.build_activity(_bundle({}), {}, [])
    assert pack["available"] is False


def test_activity_registered():
    assert evidence.BUILDERS["activity"] is evidence.build_activity
    assert "activity" in router.INTENTS


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
    "What trades have gone down in my league recently?",
    "What happened in my league this week?",
    "Show me the league activity",
    "Any recent trades in my league?",
    "Who traded this week?",
])
def test_keyword_activity(keyword, message):
    assert _intent(keyword, message)["intent"] == "activity"


@pytest.mark.parametrize("message,intent", [
    ("Who should I add from waivers?", "waivers"),
    ("What waiver moves should I make?", "waivers"),
    ("Find me a trade", "trade_finding"),
    ("What should I do with my team this week?", "this_week"),
])
def test_keyword_activity_does_not_steal_advice(keyword, message, intent):
    assert _intent(keyword, message)["intent"] == intent


# -- compose guarantees ------------------------------------------------------------

def _golden_pack():
    import json, pathlib
    data = json.loads(pathlib.Path("assistant_eval/questions.json").read_text())
    return next(q["evidence_pack"] for q in data["questions"] if q["id"] == "ac1")


def test_activity_required_receipts_cite_latest_trade():
    rows = compose._required_receipts("activity", _golden_pack())
    labels = [r["label"] for r in rows]
    assert "Pittsburgh Pilots received (total value)" in labels
    assert "Caleb's Casting Couch received (total value)" in labels
    assert "Fairness gap" in labels
    assert {r["value"] for r in rows} == {"4680", "4750", "70"}


def test_activity_required_mentions_name_both_teams():
    assert compose._required_mentions("activity", _golden_pack()) == [
        ("Pittsburgh Pilots", "activity"), ("Caleb's Casting Couch", "activity")]


def test_activity_mentions_injected_when_draft_skips_them():
    data = {"verdict": "One deal this week", "text": "A receiver changed teams.",
            "caveat": None, "receipts": [], "followups": []}
    out = compose._ensure_required_mentions("activity", _golden_pack(), data, [])
    assert "Pittsburgh Pilots" in out["text"]
    assert "Caleb's Casting Couch" in out["text"]


def test_activity_act_link_is_the_league_activity_page():
    link = compose._resolve_act_link("activity", _golden_pack())
    assert link == {"href": "/sleeper/2026/blackedraw/activity", "label": "View Activity"}
