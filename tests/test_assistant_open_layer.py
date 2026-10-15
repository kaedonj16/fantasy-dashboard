"""Open answering layer tests: semantic aliases, the site-data tools,
the bounded planner loop, and the grounding parity gate over tool
evidence. Everything external is monkeypatched at the same seams the
evidence tests use (the evidence module's lazy shims) plus the two
model seams (open_layer.call_model_json for the planner,
compose.call_model_json for the draft).
"""
import json

import pytest

pytest.importorskip("openai")
pytestmark = pytest.mark.integration

from dashboard_services.assistant import aliases, compose as compose_mod
from dashboard_services.assistant import evidence as ev
from dashboard_services.assistant import open_layer, router, service
from dashboard_services.assistant import tools as tools_mod

INDEX = {
    "1": {"name": "Alpha Receiver", "team": "CIN", "position": "WR"},
    "2": {"name": "Beta Receiver", "team": "MIN", "position": "WR"},
    "3": {"name": "Gamma Back", "team": "IND", "position": "RB"},
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
    monkeypatch.setattr(ev, "get_players_index_global",
                        lambda *a, **k: dict(INDEX))
    monkeypatch.setattr(ev, "get_model_value_table_cached",
                        lambda *a, **k: list(VALUES))
    monkeypatch.setattr(ev, "load_sleeper_week_stats",
                        lambda season, week: dict(WEEK_STATS.get(week) or {}))
    return monkeypatch


# -- aliases -----------------------------------------------------------------

def test_alias_bare_receiving_family_defaults_to_yards():
    r = aliases.resolve_phrase("who is top 5 in the most receiving stats")
    assert r["kind"] == "stat" and r["key"] == "rec_yd"
    assert r["family"] == "receiving"


def test_alias_specific_stats():
    assert aliases.resolve_phrase("most catches this year")["key"] == "rec"
    assert aliases.resolve_phrase("who has the most carries")["key"] == "rush_att"
    assert aliases.resolve_phrase("passing touchdowns leaders")["key"] == "pass_td"
    assert aliases.resolve_phrase("receiving touchdowns")["key"] == "rec_td"


def test_alias_per_game_phrases_are_metrics():
    r = aliases.resolve_phrase("receiving yards per game leaders")
    assert r["kind"] == "metric" and r["key"] == "rec_yards_per_game"
    r = aliases.resolve_phrase("who has the most fantasy points")
    assert r["kind"] == "metric" and r["key"] == "ppr_pts"


def test_alias_best_receiver_is_fantasy_points_at_wr():
    r = aliases.resolve_phrase("who is the best receiver this year")
    assert r["kind"] == "metric" and r["key"] == "ppr_pts"
    assert r["position"] == "WR"


def test_alias_neighbors_are_semantic_never_value_metrics():
    n = aliases.neighbors_for("receiving touchdowns per route")
    assert n[0] == "Receiving yards"
    assert "vorp" not in n and "war" not in n
    assert aliases.neighbors_for("what is the weather in green bay") == []


# -- evidence: stat boards + semantic nearest ---------------------------------

def test_stat_board_rows_sum_completed_weeks(base):
    rows, completed = ev.stat_board_rows(_bundle(), "rec_yd")
    assert completed == 2
    assert rows[0]["name"] == "Alpha Receiver" and rows[0]["value"] == 220
    assert rows[0]["rec"] == 17 and rows[0]["rec_tgt"] == 25
    assert rows[0]["rec_td"] == 3 and rows[0]["games"] == 2


def test_leaderboard_builds_stat_board_for_receiving_phrase(base):
    pack, _tools = ev.build_leaderboard(
        _bundle(), {"board_kind": "board", "metric_hint": "receiving yards",
                    "top_n": 5}, [])
    assert pack["available"] is True and pack["stat_board"] is True
    assert pack["label"] == "Receiving yards"
    assert pack["board"][0]["name"] == "Alpha Receiver"
    assert pack["board"][0]["value"] == 220
    assert "Week 2" in pack["qualification_text"]


def test_metric_no_key_names_semantic_neighbors_not_vorp_war(base):
    pack, _tools = ev.build_metric(
        _bundle(), {"metric_hint": "receiving yards"}, ["1"])
    assert pack["available"] is False
    assert pack["nearest"][0] == "Receiving yards"
    assert "vorp" not in pack["nearest"] and "war" not in pack["nearest"]


# -- tools ----------------------------------------------------------------------

def test_tool_top_board_stat(base):
    entry = tools_mod.run_tool("top_board", {"stat": "receiving yards",
                                             "top_n": 5}, _bundle())
    assert entry["ok"] is True and entry["stat_board"] is True
    assert entry["board"][0]["name"] == "Alpha Receiver"
    assert entry["board"][0]["value"] == 220
    assert entry["receipts"][0]["value"] == "220"
    assert entry["basis"] == "Season totals through Week 2"


def test_tool_top_board_pro_gate_holds_inside_the_tool(base):
    entry = tools_mod.run_tool("top_board", {"stat": "wopr"},
                               _bundle(has_pro=False))
    assert entry["gated"] is True
    assert "board" not in entry and entry["receipts"] == []


def test_tool_top_board_pro_board_for_pro_caller(base, monkeypatch):
    monkeypatch.setattr(ev, "metric_leaderboard",
                        lambda metric, **k: [{"player_id": "1", "name": "Alpha Receiver",
                                              "team": "CIN", "position": "WR",
                                              "value": 0.41, "games": 2}])
    entry = tools_mod.run_tool("top_board", {"stat": "wopr"},
                               _bundle(has_pro=True))
    assert entry["ok"] is True and entry["board"][0]["value"] == 0.41


def test_tool_player_metrics_strips_pro_without_pro(base, monkeypatch):
    monkeypatch.setattr(tools_mod, "resolve_player",
                        lambda name, roster_player_ids=None:
                        {"status": "resolved", "player_id": "1"})
    monkeypatch.setattr(ev, "_metrics_for",
                        lambda pid, season=None: ({"ppg": 17.0}, []))
    import data_building.advanced_metrics as am
    monkeypatch.setattr(am, "get_player_metric_ranks", lambda pid, season=None: {
        "ranks": {"wopr": {"value": 0.41, "rank": 2},
                  "target_share": {"value": 0.29, "rank": 4}}})
    entry = tools_mod.run_tool("player_metrics", {"player": "Alpha Receiver"},
                               _bundle(has_pro=False))
    keys = {r["metric"] for r in entry["metric_ranks"]}
    assert keys == {"target_share"}
    assert entry["gated_metrics"] == ["WOPR"]
    assert "0.41" not in json.dumps(entry)


def test_tool_metric_definition_gated_board_has_no_leader(base):
    entry = tools_mod.run_tool("metric_definition", {"metric": "wopr"},
                               _bundle(has_pro=False))
    assert entry["ok"] is True and entry["definition"]
    assert entry["gated_board"] is True and "leader" not in entry


def test_tool_unknown_and_failure_degrade(base):
    assert tools_mod.run_tool("nope", {}, _bundle())["ok"] is False
    entry = tools_mod.run_tool("player_profile", {"player": "Nobody Real"},
                               _bundle())
    assert entry["ok"] is False


# -- probe ------------------------------------------------------------------------

def test_probe_anchors_on_stat_phrase():
    out = open_layer.probe("who is top 5 in the most receiving stats",
                           _bundle())
    assert out["anchors"] is True and out["stat"]["key"] == "rec_yd"


def test_probe_no_anchor_for_weather():
    out = open_layer.probe("What is the weather in Green Bay this weekend?",
                           _bundle())
    assert out["anchors"] is False


# -- planner loop + composer --------------------------------------------------------

class _FakeModel:
    """One fake behind both model seams: planner and draft."""

    def __init__(self, plans, draft):
        self.plans = list(plans)
        self.draft = draft
        self.plan_payloads = []
        self.draft_calls = 0

    def __call__(self, system, user, schema, name):
        if name == "assistant_open_planner":
            self.plan_payloads.append(json.loads(user))
            if self.plans:
                return self.plans.pop(0)
            return {"tool_calls": [], "decline": False}
        self.draft_calls += 1
        return dict(self.draft)


def _install_model(monkeypatch, fake):
    monkeypatch.setattr(open_layer, "call_model_json", fake)
    monkeypatch.setattr(compose_mod, "call_model_json", fake)


_GOOD_DRAFT = {"verdict": "Alpha Receiver leads Receiving yards",
               "text": "Alpha Receiver leads receiving yards at 220.",
               "receipts": [{"label": "Alpha Receiver receiving yards",
                             "value": "220"}],
               "caveat": None, "followups": [], "actions": []}


def test_open_turn_end_to_end_grounded(base, monkeypatch):
    fake = _FakeModel(
        plans=[{"tool_calls": [{"tool": "top_board", "stat": "receiving yards",
                                 "top_n": 5, "player": None, "player_a": None,
                                 "player_b": None, "metric": None,
                                 "position": None, "team": None, "kind": None}],
                "decline": False}],
        draft=_GOOD_DRAFT)
    _install_model(monkeypatch, fake)
    answer, pack, tools_called, names = open_layer.answer_open(
        "who is top 5 in the most receiving stats", _bundle())
    assert tools_called == ["top_board"]
    assert pack["evidence"][0]["board"][0]["value"] == 220
    assert answer["verdict"] == "Alpha Receiver leads Receiving yards"
    assert "220" in answer["text"]
    assert names == ["Alpha Receiver", "Beta Receiver"]


def test_open_turn_thread_context_reaches_planner(base, monkeypatch):
    fake = _FakeModel(plans=[{"tool_calls": [], "decline": True}],
                      draft=_GOOD_DRAFT)
    _install_model(monkeypatch, fake)
    open_layer.plan_and_fetch(
        "what about his handcuff?", _bundle(),
        thread={"context_players": ["Puka Nacua"],
                "last_intent": "player", "last_answer": "Puka Nacua update"})
    payload = fake.plan_payloads[0]
    assert payload["thread_context"]["players"] == ["Puka Nacua"]


def _board_plan(*stats):
    return {"tool_calls": [
        {"tool": "top_board", "stat": s, "top_n": 5, "player": None,
         "player_a": None, "player_b": None, "metric": None,
         "position": None, "team": None, "kind": None} for s in stats],
        "decline": False}


def test_open_planner_call_cap_is_four(base, monkeypatch):
    fake = _FakeModel(plans=[
        _board_plan("receiving yards", "receptions", "targets"),
        _board_plan("passing yards", "rushing yards", "carries")],
        draft=_GOOD_DRAFT)
    _install_model(monkeypatch, fake)
    _pack, tools_called = open_layer.plan_and_fetch("boards please", _bundle())
    assert len(tools_called) == open_layer.MAX_TOOL_CALLS == 4


def test_open_decline_spends_no_draft_call(base, monkeypatch):
    fake = _FakeModel(plans=[{"tool_calls": [], "decline": True}],
                      draft=_GOOD_DRAFT)
    _install_model(monkeypatch, fake)
    answer, pack, _tools, _names = open_layer.answer_open(
        "something the site cannot know", _bundle())
    assert pack["available"] is False
    assert answer["verdict"] is None
    assert "not available" in answer["text"]
    assert fake.draft_calls == 0


def test_open_poison_number_is_stripped(base, monkeypatch):
    poison = dict(_GOOD_DRAFT)
    poison["text"] = "Alpha Receiver leads receiving yards at 220. He also has 9999 catches."
    fake = _FakeModel(
        plans=[{"tool_calls": [{"tool": "top_board", "stat": "receiving yards",
                                 "top_n": 5, "player": None, "player_a": None,
                                 "player_b": None, "metric": None,
                                 "position": None, "team": None, "kind": None}],
                "decline": False}],
        draft=poison)
    _install_model(monkeypatch, fake)
    answer, _pack, _tools, _names = open_layer.answer_open(
        "who is top 5 in the most receiving stats", _bundle())
    surfaces = json.dumps(answer)
    assert "9999" not in surfaces
    assert "220" in answer["text"]


def test_open_gated_evidence_gets_gated_answer(base, monkeypatch):
    gated_entry = {"tool": "top_board", "ok": True, "gated": True,
                   "label": "WOPR", "metric_key": "wopr", "players": [],
                   "receipts": [], "link": None}
    fake = _FakeModel(
        plans=[{"tool_calls": [{"tool": "top_board", "stat": "wopr",
                                 "top_n": 5, "player": None, "player_a": None,
                                 "player_b": None, "metric": None,
                                 "position": None, "team": None, "kind": None}],
                "decline": False}],
        draft=_GOOD_DRAFT)
    _install_model(monkeypatch, fake)
    answer, _pack, _tools, _names = open_layer.answer_open(
        "Who leads in WOPR?", _bundle(),
        frozen_evidence=[gated_entry])
    assert answer["verdict"] is None and answer["receipts"] == []
    assert "PRO" in answer["text"] and "WOPR" in answer["text"]
    assert fake.draft_calls == 0


def test_compose_answer_open_seam_with_frozen_pack(base, monkeypatch):
    data = json.load(open("assistant_eval/questions.json"))
    q = next(x for x in data["questions"] if x["id"] == "ow2")
    fake = _FakeModel(plans=[], draft={
        "verdict": "Ja'Marr Chase leads Receiving yards",
        "text": "Ja'Marr Chase leads receiving yards at 612.",
        "receipts": [{"label": "Ja'Marr Chase receiving yards", "value": "612"}],
        "caveat": None, "followups": [], "actions": []})
    monkeypatch.setattr(compose_mod, "call_model_json", fake)
    answer = compose_mod.compose_answer("open", q["question"],
                                        q["evidence_pack"])
    assert answer["verdict"] == "Ja'Marr Chase leads Receiving yards"
    assert answer["mentions"][0]["name"] == "Ja'Marr Chase"
    assert answer["act_link"]["label"] == "View on Advanced Metrics"


# -- citation completion (live eval round 3: ow4/ow9/ow13/ow15/ow17) --------------

def _stats_pack(receipt_rows, **fields):
    entry = {"tool": "player_stats", "ok": True,
             "player": {"player_id": "p1", "name": "Alpha Receiver"},
             "players": [{"player_id": "p1", "name": "Alpha Receiver"}],
             "receipts": receipt_rows, "link": None}
    entry.update(fields)
    return {"type": "open", "available": True, "league": {}, "week": 7,
            "as_of": {"week": 7, "season": 2026}, "evidence": [entry],
            "tools_called": ["player_stats"]}


def test_open_citation_completion_backs_every_stated_number(monkeypatch):
    # The draft states three evidence numbers but its own receipt picks
    # carry only one; completion must source rows for all three.
    pack = _stats_pack(
        [{"label": "Fantasy points", "value": "123.8"},
         {"label": "Receiving yards", "value": "612"},
         {"label": "Receptions", "value": "59"}],
        totals={"fantasy_points": 123.8, "rec_yd": 612, "rec": 59})
    fake = _FakeModel(plans=[], draft={
        "verdict": "Alpha Receiver",
        "text": "Alpha Receiver has 612 receiving yards on 59 catches, "
                "good for 123.8 fantasy points.",
        "receipts": [{"label": "Receiving yards", "value": "612"}],
        "caveat": None, "followups": [], "actions": []})
    monkeypatch.setattr(compose_mod, "call_model_json", fake)
    answer = compose_mod.compose_answer(
        "open", "How is Alpha Receiver doing?", pack)
    assert {r["value"] for r in answer["receipts"]} == {"123.8", "612", "59"}
    labels = {r["value"]: r["label"] for r in answer["receipts"]}
    assert labels["123.8"] == "Fantasy points"
    assert labels["59"] == "Receptions"


def test_open_citation_completion_matches_percent_citation(monkeypatch):
    # A share stored as 0.29 in the evidence is stated as 29% in prose;
    # the receipt must still land, carrying the evidence's own value.
    pack = {"type": "open", "available": True, "league": {}, "week": 7,
            "as_of": {"week": 7, "season": 2026},
            "evidence": [{"tool": "metric_definition", "ok": True,
                          "label": "Target Share",
                          "leader": {"name": "Puka Nacua", "value": 0.29},
                          "players": [{"player_id": "p9", "name": "Puka Nacua"}],
                          "receipts": [{"label": "Puka Nacua Target Share",
                                        "value": "0.29"}],
                          "link": None}],
            "tools_called": ["metric_definition"]}
    fake = _FakeModel(plans=[], draft={
        "verdict": "Target Share",
        "text": "Target share is the slice of team targets a player earns. "
                "The leader sits at 29%.",
        "receipts": [], "caveat": None, "followups": [], "actions": []})
    monkeypatch.setattr(compose_mod, "call_model_json", fake)
    answer = compose_mod.compose_answer(
        "open", "What is target share?", pack)
    assert {"label": "Puka Nacua Target Share", "value": "0.29"} \
        in answer["receipts"]


def test_open_citation_completion_never_invents_a_receipt(monkeypatch):
    # A stated number with no evidence source: the grounding gate
    # strips it from the prose and completion adds no row for it.
    pack = _stats_pack(
        [{"label": "Receiving yards", "value": "612"}],
        totals={"rec_yd": 612})
    fake = _FakeModel(plans=[], draft={
        "verdict": "Alpha Receiver",
        "text": "Alpha Receiver has 612 receiving yards and 9999 targets.",
        "receipts": [], "caveat": None, "followups": [], "actions": []})
    monkeypatch.setattr(compose_mod, "call_model_json", fake)
    answer = compose_mod.compose_answer(
        "open", "How many yards does Alpha Receiver have?", pack)
    assert "9999" not in json.dumps(answer)
    assert {r["value"] for r in answer["receipts"]} == {"612"}


def test_open_citation_completion_cap_prioritizes_stated_numbers(monkeypatch):
    rows = [{"label": f"Stat {i}", "value": str(v)}
            for i, v in enumerate((10, 20, 30, 40, 50), start=1)]
    pack = _stats_pack(rows, totals={"a": 10, "b": 20, "c": 30, "d": 40,
                                     "e": 50, "f": 60})
    fake = _FakeModel(plans=[], draft={
        "verdict": "Alpha Receiver",
        "text": "The stated ones are 40 and 50.",
        "receipts": [{"label": "Other stat", "value": "60"}],
        "caveat": None, "followups": [], "actions": []})
    monkeypatch.setattr(compose_mod, "call_model_json", fake)
    answer = compose_mod.compose_answer("open", "Numbers?", pack)
    values = [r["value"] for r in answer["receipts"]]
    assert len(values) == 4
    assert values[:2] == ["40", "50"]  # stated numbers first
    assert "60" in values  # the model's own grounded receipt survives


def test_open_required_mentions_names_evidence_teams(monkeypatch):
    # Live eval round 3 ow12: the draft answered the record question
    # without naming the user's team or the comparison team in the
    # fetched standings evidence; both must be injected.
    data = json.load(open("assistant_eval/questions.json"))
    q = next(x for x in data["questions"] if x["id"] == "ow12")
    fake = _FakeModel(plans=[], draft={
        "verdict": "Seed 3 at 4-3",
        "text": "You are 4-3 with 745.2 points for so far.",
        "receipts": [{"label": "Your points for", "value": "745.2"}],
        "caveat": None, "followups": [], "actions": []})
    monkeypatch.setattr(compose_mod, "call_model_json", fake)
    answer = compose_mod.compose_answer("open", q["question"],
                                        q["evidence_pack"])
    assert "Caleb's Casting Couch" in answer["text"]
    assert "Pittsburgh Pilots" in answer["text"]
    assert "745.2" in {r["value"] for r in answer["receipts"]}


# -- router ------------------------------------------------------------------------

@pytest.fixture()
def keyword(monkeypatch):
    def _boom(*a, **k):
        raise RuntimeError("model down")
    monkeypatch.setattr(router, "call_model_json", _boom)
    return router.classify


def test_router_open_intent_registered():
    assert "open" in router.INTENTS
    assert "open" in router.SCHEMA["properties"]["intent"]["enum"]


def test_keyword_stat_board_receiving(keyword):
    out = keyword("who is top 5 in the most receiving stats",
                  {"league_name": "Blackedraw", "week": 7, "season": 2026}, None)
    assert out["intent"] == "leaderboard"
    assert out["entities"]["metric_hint"] == "receiving stats"


def test_keyword_stat_board_rushing_tds(keyword):
    out = keyword("Who leads in rushing touchdowns?",
                  {"league_name": "Blackedraw", "week": 7, "season": 2026}, None)
    assert out["intent"] == "leaderboard"
    assert out["entities"]["metric_hint"] == "rushing touchdowns"


def test_keyword_metric_question_unchanged(keyword):
    out = keyword("What is Puka Nacua's target share?",
                  {"league_name": "Blackedraw", "week": 7, "season": 2026}, None)
    assert out["intent"] == "metric"


# -- service wiring ------------------------------------------------------------------

def _patch_service(monkeypatch, routed, answer_open=None, probe=None):
    calls = {"logs": []}
    monkeypatch.setattr(service, "_active_plan", lambda ak: None)
    monkeypatch.setattr(service, "last_usage", lambda: {})
    monkeypatch.setattr(service.ctx_mod, "resolve_league",
                        lambda *a, **k: _bundle())
    monkeypatch.setattr(service, "classify", lambda *a, **k: routed)
    monkeypatch.setattr(service.store, "count_monthly_asked", lambda ak: 0)
    monkeypatch.setattr(service.store, "get_or_create_thread",
                        lambda *a, **k: {"thread_id": "t1", "turn_count": 0})
    monkeypatch.setattr(service.store, "get_last_log_for_thread",
                        lambda tid: None)
    monkeypatch.setattr(service.store, "list_preferences", lambda ak: [])
    monkeypatch.setattr(service.store, "write_log",
                        lambda **kw: calls["logs"].append(kw) or 7)
    monkeypatch.setattr(service.store, "update_thread", lambda *a, **k: None)
    if answer_open is not None:
        monkeypatch.setattr(open_layer, "answer_open", answer_open)
    if probe is not None:
        monkeypatch.setattr(open_layer, "probe", probe)
    return calls


def test_service_open_turn_logs_once_and_counts_once(monkeypatch):
    def fake_open(message, bundle, **kw):
        return ({"verdict": "Alpha Receiver leads Receiving yards",
                 "text": "Alpha Receiver leads receiving yards at 220.",
                 "receipts": [{"label": "Receiving yards", "value": "220"}],
                 "caveat": None, "followups": [], "act_link": None,
                 "actions": [], "mentions": []},
                {"type": "open", "evidence": []}, ["top_board"],
                ["Alpha Receiver"])
    calls = _patch_service(
        monkeypatch,
        {"intent": "open", "entities": {"player_names": []},
         "router_model": "fake"},
        answer_open=fake_open)
    out = service.ask("acct:1", "who is top 5 in the most receiving stats")
    assert out["intent"] == "open"
    assert len(calls["logs"]) == 1
    assert calls["logs"][0]["intent"] == "open"
    assert calls["logs"][0]["tools_called"] == ["top_board"]
    assert out["caps"]["used"] == 1


def test_service_out_of_scope_without_anchors_still_declines(monkeypatch):
    def _no_open(*a, **k):  # pragma: no cover - must never run
        raise AssertionError("open path ran without anchors")
    calls = _patch_service(
        monkeypatch,
        {"intent": "out_of_scope", "entities": {"player_names": []},
         "router_model": "fake"},
        answer_open=_no_open,
        probe=lambda *a, **k: {"anchors": False})
    out = service.ask("acct:1", "What is the weather in Green Bay?")
    assert out["intent"] == "out_of_scope"
    assert "outside what I can answer" in out["answer"]["text"]
    assert len(calls["logs"]) == 1
