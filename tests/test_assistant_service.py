"""Assistant service tests: no DB, no OpenAI.

store.get_conn is faked with an in-memory pattern-matching fake; the model is
faked by monkeypatching compose.call_model_json / service.compose_answer.
"""
import json
import re

import pytest

# The AI client imports `openai`, which the lint job doesn't install; skip at
# collection there (same pattern as tests/test_ai_empty_response_handling.py).
# These tests need the full stack, so they run in the integration shard.
pytest.importorskip("openai")
pytestmark = pytest.mark.integration

from dashboard_services.assistant import compose as compose_mod
from dashboard_services.assistant import service, store
from dashboard_services.assistant.resolve import resolve_player as _real_resolve


# -- in-memory fake DB ---------------------------------------------------------

class FakeCursor:
    def __init__(self, rows=None, row=None):
        self._rows = rows or []
        self._row = row

    def fetchone(self):
        if self._row is not None:
            return self._row
        return self._rows[0] if self._rows else None

    def fetchall(self):
        return list(self._rows)


class FakeConn:
    def __init__(self, state):
        self.state = state

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def commit(self):
        pass

    def _cols(self, sql):
        m = re.search(r"INSERT INTO \w+ \(([^)]+)\) VALUES", sql)
        return [c.strip() for c in m.group(1).split(",")] if m else []

    def execute(self, sql, params=()):
        s = " ".join(sql.split())
        st = self.state
        if s.startswith("CREATE"):
            return FakeCursor()
        if "INSERT INTO assistant_threads" in s:
            row = dict(zip(self._cols(s), params))
            st["threads"][row["thread_id"]] = row
            return FakeCursor()
        if "FROM assistant_threads" in s and s.startswith("SELECT"):
            row = st["threads"].get(params[0])
            return FakeCursor(row=dict(row) if row else None)
        if "UPDATE assistant_threads" in s:
            row = st["threads"].get(params[2])
            if row:
                row["last_entities"] = params[0]
                row["turn_count"] = params[1]
            return FakeCursor()
        if "INSERT INTO assistant_advice_log" in s:
            row = dict(zip(self._cols(s), params))
            st["log_seq"] += 1
            row["log_id"] = st["log_seq"]
            st["logs"].append(row)
            return FakeCursor(row={"log_id": row["log_id"]})
        if "SELECT COUNT" in s and "assistant_advice_log" in s:
            n = sum(1 for r in st["logs"] if r.get("account_key") == params[0] and not r.get("proactive"))
            return FakeCursor(row={"n": n})
        if "FROM assistant_advice_log" in s and "proactive = TRUE" in s:
            rows = [r for r in reversed(st["logs"])
                    if r.get("account_key") == params[0] and str(r.get("league_id")) == str(params[1])
                    and r.get("week") == params[2] and r.get("proactive") and r.get("trigger") == "digest"]
            return FakeCursor(row=dict(rows[0]) if rows else None)
        if "FROM assistant_advice_log" in s and "WHERE thread_id" in s:
            rows = [r for r in reversed(st["logs"]) if r.get("thread_id") == params[0]]
            return FakeCursor(row=dict(rows[0]) if rows else None)
        if "FROM assistant_advice_log" in s and "log_id" in s:
            rows = [r for r in st["logs"] if r.get("log_id") == params[0]
                    and (len(params) < 2 or r.get("account_key") == params[1])]
            return FakeCursor(row=dict(rows[0]) if rows else None)
        if "UPDATE assistant_advice_log SET thumb" in s:
            for r in st["logs"]:
                if r.get("log_id") == params[1] and r.get("account_key") == params[2]:
                    r["thumb"] = params[0]
            return FakeCursor()
        if "UPDATE assistant_advice_log SET correction_category" in s:
            for r in st["logs"]:
                if r.get("log_id") == params[4] and r.get("account_key") == params[5]:
                    (r["correction_category"], r["correction_text"],
                     r["corrected_claim"], r["reask_flag"]) = params[0], params[1], params[2], params[3]
            return FakeCursor()
        if "INSERT INTO assistant_preferences" in s:
            row = dict(zip(self._cols(s), params))
            st["pref_seq"] += 1
            row["id"] = st["pref_seq"]
            row["created_at"] = "2026-10-14T00:00:00"
            st["prefs"].append(row)
            return FakeCursor(row=dict(row))
        if "FROM assistant_preferences" in s and s.startswith("SELECT"):
            rows = [dict(r) for r in st["prefs"] if r.get("account_key") == params[0]]
            return FakeCursor(rows=rows)
        if "DELETE FROM assistant_preferences" in s:
            if "WHERE id" in s:
                st["prefs"] = [r for r in st["prefs"] if not (r.get("id") == params[0] and r.get("account_key") == params[1])]
            else:
                st["prefs"] = [r for r in st["prefs"] if r.get("account_key") != params[0]]
            return FakeCursor()
        raise AssertionError(f"fake DB cannot handle SQL: {s}")


INDEX = {
    "1": {"name": "Rome Odunze", "team": "CHI", "position": "WR"},
    "2": {"name": "Darnell Mooney", "team": "ATL", "position": "WR"},
    "3": {"name": "Chris Smith", "team": "NYG", "position": "WR"},
    "4": {"name": "Chris Smith", "team": "DAL", "position": "RB"},
}

BUNDLE = {
    "platform": "sleeper", "league_id": "L1", "season": 2026, "league_name": "Blackedraw",
    "team_name": "Couch", "week": 7, "ctx": {"rosters": []}, "roster": {"players": ["1"], "starters": []},
    "viewer": {}, "roster_player_ids": {"1"}, "assumed_league_uncertain": False,
}


@pytest.fixture()
def env(monkeypatch):
    state = {"threads": {}, "logs": [], "prefs": [], "log_seq": 0, "pref_seq": 0}
    conn = FakeConn(state)
    monkeypatch.setattr(store, "_TABLE_READY", False)
    monkeypatch.setattr(store, "get_conn", lambda *a, **k: conn)
    monkeypatch.setattr(service, "_active_plan", lambda key: None)
    service._clear_resolve_snapshot_cache()
    monkeypatch.setattr(service.ctx_mod, "resolve_league", lambda key, **kw: dict(BUNDLE))
    monkeypatch.setattr(service, "resolve_player",
                        lambda name, roster_player_ids=None: _real_resolve(name, index=INDEX, roster_player_ids=roster_player_ids))
    return state


def _route(intent, **entities):
    base = {"player_names": [], "give_player_names": [], "receive_player_names": [], "picks": [],
            "other_team_hint": None, "metric_hint": None, "week": None}
    base.update(entities)
    return lambda q, c, h: {"intent": intent, "entities": base, "preference_text": entities.get("preference_text"), "router_model": "fake"}


def _start_sit_pack(bundle, entities, player_ids):
    return ({"type": "start_sit", "available": True, "week": 7,
             "players": [{"player_id": "1", "name": "Rome Odunze", "value": 100},
                         {"player_id": "2", "name": "Darnell Mooney", "value": 80}],
             "margin": {"thin": False}, "as_of": {"week": 7}}, ["fake_tool"])


# -- tests ---------------------------------------------------------------------

def test_start_sit_happy_path_and_receipt_grounding(env, monkeypatch):
    monkeypatch.setattr(service, "classify", _route("start_sit", player_names=["Rome Odunze", "Darnell Mooney"]))
    monkeypatch.setitem(service.BUILDERS, "start_sit", _start_sit_pack)

    def canned(system, user, schema, name):
        return {"verdict": "START Rome Odunze", "text": "Start Odunze.",
                "receipts": [{"label": "Odunze value", "value": "100"},
                             {"label": "Invented", "value": "9999"}],
                "caveat": None, "followups": ["What about next week?"], "act_link": None}
    monkeypatch.setattr(compose_mod, "call_model_json", canned)

    out = service.ask("acct:42", "Start Odunze or Mooney?")
    assert out["intent"] == "start_sit"
    assert out["answer"]["verdict"] == "START Rome Odunze"
    # Invented receipt dropped by the grounding post-check.
    assert out["answer"]["receipts"] == [{"label": "Odunze value", "value": "100"}]
    assert out["log_id"] is not None
    row = env["logs"][-1]
    assert row["intent"] == "start_sit"
    assert json.loads(row["verdict"])["call"] == "START Rome Odunze"
    assert json.loads(row["numbers_shown"]) == {"Odunze value": "100"}
    assert out["caps"]["used"] == 1


def test_cap_reached(env, monkeypatch):
    monkeypatch.setenv("ASSISTANT_FREE_MONTHLY_CAP", "1")
    store.write_log(account_key="acct:42", proactive=False, intent="player")
    with pytest.raises(service.CapReached):
        service.ask("acct:42", "Start Odunze or Mooney?")


def test_caps_tier_ladder_by_plan(env, monkeypatch):
    # Free: no active plan row.
    caps = service.caps_for("acct:42")
    assert caps["tier"] == "free" and caps["limit"] == 25 and caps["plan_key"] is None
    assert caps["exempt"] is False
    assert [t["key"] for t in caps["tiers"]] == ["free", "starter", "all_pro", "hall_of_fame"]
    assert [t["limit"] for t in caps["tiers"]] == [25, 100, 250, 1000]
    assert [t["current"] for t in caps["tiers"]] == [True, False, False, False]

    cases = [
        ("starter", "starter", 100), ("single_league", "starter", 100),
        ("all_pro", "all_pro", 250), ("league", "all_pro", 250),
        ("hall_of_fame", "hall_of_fame", 1000), ("user", "hall_of_fame", 1000),
        ("combo", "hall_of_fame", 1000),
        ("some_future_plan", "starter", 100),  # unknown paid keys land on the starter cap
    ]
    for plan_key, tier, limit in cases:
        monkeypatch.setattr(service, "_active_plan", lambda key, pk=plan_key: {"plan_key": pk})
        caps = service.caps_for("acct:42")
        assert caps["tier"] == tier and caps["limit"] == limit, plan_key
        assert caps["plan_key"] == plan_key
        assert caps["tier_label"] == service.TIER_LABELS[tier]
        assert [t["current"] for t in caps["tiers"]] == [t["key"] == tier for t in caps["tiers"]]


def test_caps_env_overrides(env, monkeypatch):
    monkeypatch.setenv("ASSISTANT_FREE_MONTHLY_CAP", "5")
    monkeypatch.setenv("ASSISTANT_PAID_MONTHLY_CAP", "300")  # legacy knob: starter tier
    monkeypatch.setenv("ASSISTANT_ALL_PRO_MONTHLY_CAP", "2000")
    caps = service.caps_for("acct:42")
    assert caps["limit"] == 5
    assert [t["limit"] for t in caps["tiers"]] == [5, 300, 2000, 1000]


def test_owner_exempt_never_capped_but_usage_counts(env, monkeypatch):
    monkeypatch.setenv("ASSISTANT_FREE_MONTHLY_CAP", "1")
    monkeypatch.setattr(service, "_is_exempt", lambda: True)
    monkeypatch.setattr(service, "classify", _route("start_sit", player_names=["Rome Odunze", "Darnell Mooney"]))
    monkeypatch.setitem(service.BUILDERS, "start_sit", _start_sit_pack)
    monkeypatch.setattr(compose_mod, "call_model_json",
                        lambda system, user, schema, name: {
                            "verdict": "START Rome Odunze", "text": "Start Odunze.",
                            "receipts": [], "caveat": None, "followups": [], "act_link": None})
    # Two prior asks this month: already past the free cap of 1.
    store.write_log(account_key="acct:42", proactive=False, intent="player")
    store.write_log(account_key="acct:42", proactive=False, intent="player")
    out = service.ask("acct:42", "Start Odunze or Mooney?")
    assert out["caps"]["exempt"] is True
    assert out["caps"]["used"] == 3  # the exempt ask still counted
    # A non-owner at the same usage is capped (see also test_cap_reached).
    monkeypatch.setattr(service, "_is_exempt", lambda: False)
    with pytest.raises(service.CapReached):
        service.ask("acct:42", "Start Odunze or Mooney?")


def test_env_listed_account_exempt_without_admin_session(env, monkeypatch):
    # No admin session exists in the test environment (the _is_exempt admin
    # path is False outside a request context); the env list alone exempts.
    monkeypatch.setenv("ASSISTANT_EXEMPT_ACCOUNT_KEYS", "acct:42")
    assert service.caps_for("acct:42")["exempt"] is True
    assert service.caps_for("acct:7")["exempt"] is False
    # Bare ids, blanks, and messy spacing normalize the same way.
    monkeypatch.setenv("ASSISTANT_EXEMPT_ACCOUNT_KEYS", " 7 , acct:9 ,")
    assert service.caps_for("acct:42")["exempt"] is False
    assert service.caps_for("acct:7")["exempt"] is True
    assert service.caps_for("acct:9")["exempt"] is True


def test_env_listed_account_never_capped_but_usage_counts(env, monkeypatch):
    monkeypatch.setenv("ASSISTANT_FREE_MONTHLY_CAP", "1")
    monkeypatch.setenv("ASSISTANT_EXEMPT_ACCOUNT_KEYS", "42")
    monkeypatch.setattr(service, "classify", _route("start_sit", player_names=["Rome Odunze", "Darnell Mooney"]))
    monkeypatch.setitem(service.BUILDERS, "start_sit", _start_sit_pack)
    monkeypatch.setattr(compose_mod, "call_model_json",
                        lambda system, user, schema, name: {
                            "verdict": "START Rome Odunze", "text": "Start Odunze.",
                            "receipts": [], "caveat": None, "followups": [], "act_link": None})
    # Two prior asks this month for both accounts: past the free cap of 1.
    for key in ("acct:42", "acct:7"):
        store.write_log(account_key=key, proactive=False, intent="player")
        store.write_log(account_key=key, proactive=False, intent="player")
    out = service.ask("acct:42", "Start Odunze or Mooney?")
    assert out["caps"]["exempt"] is True
    assert out["caps"]["used"] == 3  # the exempt ask still counted
    # An account not on the list, at the same usage, is capped.
    with pytest.raises(service.CapReached):
        service.ask("acct:7", "Start Odunze or Mooney?")


def test_malformed_exempt_env_exempts_nobody(env, monkeypatch):
    for raw in ("", "   ", "banana", "acct:", "acct:abc", "acct:-1", ",", "acct:4.2"):
        monkeypatch.setenv("ASSISTANT_EXEMPT_ACCOUNT_KEYS", raw)
        assert service._exempt_account_ids() == frozenset(), raw
        assert service.caps_for("acct:42")["exempt"] is False, raw


def test_env_listed_email_exempt(env, monkeypatch):
    monkeypatch.setattr(service, "_account_email",
                        lambda aid: "owner@example.com" if aid == 42 else None)
    # Matching is case-insensitive against the stored account email.
    monkeypatch.setenv("ASSISTANT_EXEMPT_EMAILS", "Owner@Example.com")
    assert service.caps_for("acct:42")["exempt"] is True
    assert service.caps_for("acct:7")["exempt"] is False
    monkeypatch.setenv("ASSISTANT_EXEMPT_EMAILS", "someoneelse@example.com")
    assert service.caps_for("acct:42")["exempt"] is False


def test_clarify_on_ambiguous_name(env, monkeypatch):
    monkeypatch.setattr(service, "classify", _route("player", player_names=["Chris Smith"]))
    out = service.ask("acct:42", "What is going on with Chris Smith?")
    assert out["intent"] == "clarify"
    assert out["answer"]["verdict"] is None
    assert len(out["answer"]["candidates"]) == 2
    assert env["logs"][-1]["intent"] == "clarify"


def _canned_model(monkeypatch):
    monkeypatch.setattr(compose_mod, "call_model_json",
                        lambda system, user, schema, name: {
                            "verdict": "START Rome Odunze", "text": "Start Odunze.",
                            "receipts": [{"label": "Odunze value", "value": "100"}],
                            "caveat": None, "followups": [], "act_link": None})


def test_clarification_pick_resumes_original_intent(env, monkeypatch):
    # Turn 1: a genuine ambiguity (two Chris Smiths, both fantasy-pool
    # positions, no footprint to separate them) clarifies.
    routes = iter([
        _route("start_sit", player_names=["Chris Smith", "Rome Odunze"])("q", None, None),
        _route("player", player_names=["Chris Smith"])("q", None, None),
        _route("start_sit", player_names=["Chris Smith", "Rome Odunze"])("q", None, None),
    ])
    monkeypatch.setattr(service, "classify", lambda q, c, h: next(routes))
    seen = {}

    def pack(bundle, entities, player_ids):
        seen["player_ids"] = list(player_ids)
        return _start_sit_pack(bundle, entities, player_ids)
    monkeypatch.setitem(service.BUILDERS, "start_sit", pack)
    _canned_model(monkeypatch)

    out1 = service.ask("acct:42", "Start Chris Smith or Rome Odunze?")
    assert out1["intent"] == "clarify"
    assert {c["player_id"] for c in out1["answer"]["candidates"]} == {"3", "4"}

    # Turn 2: the candidate chip tap sends the bare name plus the player
    # id. The pick must resume the pending start/sit request (with the
    # OTHER resolved player intact), not degrade into a player profile.
    out2 = service.ask("acct:42", "Chris Smith", thread_id=out1["thread_id"],
                       player_id_hint="3")
    assert out2["intent"] == "start_sit"
    assert seen["player_ids"] == ["3", "1"]
    assert env["logs"][-1]["intent"] == "start_sit"
    assert json.loads(env["logs"][-1]["entities"])["player_ids"] == ["3", "1"]

    # Turn 3: re-asking the original question in the same thread never
    # re-clarifies; the settled name resolves from the thread memory.
    out3 = service.ask("acct:42", "Start Chris Smith or Rome Odunze?",
                       thread_id=out1["thread_id"])
    assert out3["intent"] == "start_sit"
    assert seen["player_ids"] == ["3", "1"]


def test_clarification_bare_typed_name_does_not_guess(env, monkeypatch):
    routes = iter([
        _route("start_sit", player_names=["Chris Smith", "Rome Odunze"])("q", None, None),
        _route("player", player_names=["Chris Smith"])("q", None, None),
    ])
    monkeypatch.setattr(service, "classify", lambda q, c, h: next(routes))
    monkeypatch.setitem(service.BUILDERS, "start_sit", _start_sit_pack)
    _canned_model(monkeypatch)

    out1 = service.ask("acct:42", "Start Chris Smith or Rome Odunze?")
    assert out1["intent"] == "clarify"
    # No hint, and both candidates are named "Chris Smith": the bare
    # typed name cannot pick between them, so the service asks again
    # rather than guessing.
    out2 = service.ask("acct:42", "Chris Smith", thread_id=out1["thread_id"])
    assert out2["intent"] == "clarify"


def test_clarification_unique_typed_name_resumes(env, monkeypatch):
    # Candidates with distinct names ("Robinson" -> Bijan Robinson /
    # Brian Robinson Jr.): typing one full name is a real pick.
    robinson_index = {
        "10": {"name": "Bijan Robinson", "team": "ATL", "position": "RB"},
        "11": {"name": "Brian Robinson Jr.", "team": "WAS", "position": "RB"},
        "1": {"name": "Rome Odunze", "team": "CHI", "position": "WR"},
    }
    monkeypatch.setattr(service, "resolve_player",
                        lambda name, roster_player_ids=None: _real_resolve(
                            name, index=robinson_index,
                            roster_player_ids=roster_player_ids))
    routes = iter([
        _route("start_sit", player_names=["Robinson", "Rome Odunze"])("q", None, None),
        _route("player", player_names=["Bijan Robinson"])("q", None, None),
    ])
    monkeypatch.setattr(service, "classify", lambda q, c, h: next(routes))
    seen = {}

    def pack(bundle, entities, player_ids):
        seen["player_ids"] = list(player_ids)
        return _start_sit_pack(bundle, entities, player_ids)
    monkeypatch.setitem(service.BUILDERS, "start_sit", pack)
    _canned_model(monkeypatch)

    out1 = service.ask("acct:42", "Start Robinson or Rome Odunze?")
    assert out1["intent"] == "clarify"
    assert {c["player_id"] for c in out1["answer"]["candidates"]} == {"10", "11"}
    out2 = service.ask("acct:42", "Bijan Robinson", thread_id=out1["thread_id"])
    assert out2["intent"] == "start_sit"
    assert seen["player_ids"] == ["10", "1"]


def test_open_clarification_pick_resumes_with_forced_pick(env, monkeypatch):
    from dashboard_services.assistant import open_layer
    monkeypatch.setattr(service, "classify",
                        _route("open", player_names=[]))
    calls = []

    def fake_open(question, bundle, thread=None, history=None,
                  preferences=None, assumed_context=None):
        calls.append({"question": question,
                      "forced": dict(bundle.get("forced_player_ids") or {})})
        if len(calls) == 1:
            return ({"verdict": None, "text": "Which player do you mean?",
                     "receipts": [], "caveat": None, "followups": [],
                     "act_link": None, "actions": [], "mentions": [],
                     "candidates": [
                         {"player_id": "6794", "name": "Justin Jefferson",
                          "team": "MIN", "position": "WR"},
                         {"player_id": "9001", "name": "Justin Jefferson",
                          "team": "BUF", "position": "WR"}]},
                    {"type": "open", "evidence": []}, ["compare_players"], [])
        return ({"verdict": "Ja'Marr Chase over Justin Jefferson",
                 "text": "Chase, on value.", "receipts": [], "caveat": None,
                 "followups": [], "act_link": None, "actions": [], "mentions": []},
                {"type": "open", "evidence": []}, ["compare_players"],
                ["Ja'Marr Chase", "Justin Jefferson"])
    monkeypatch.setattr(open_layer, "answer_open", fake_open)

    out1 = service.ask("acct:42", "Compare Ja'Marr Chase and Justin Jefferson")
    assert out1["intent"] == "open"
    assert len(out1["answer"]["candidates"]) == 2

    # The chip tap resumes the pending open request: the open turn runs
    # against the ORIGINAL question with the pick forced into the tools.
    monkeypatch.setattr(service, "classify",
                        _route("player", player_names=["Justin Jefferson"]))
    out2 = service.ask("acct:42", "Justin Jefferson",
                       thread_id=out1["thread_id"], player_id_hint="6794")
    assert out2["intent"] == "open"
    assert out2["answer"]["verdict"] == "Ja'Marr Chase over Justin Jefferson"
    assert calls[1]["question"] == "Compare Ja'Marr Chase and Justin Jefferson"
    assert calls[1]["forced"] == {"justin jefferson": "6794"}


def test_namesake_policy_flows_through_service_resolution(env, monkeypatch):
    # The Jefferson pair at service level: with the value footprint
    # naming only the WR, the name resolves directly, no clarify turn.
    jeff_index = {
        "6794": {"name": "Justin Jefferson", "team": "MIN", "position": "WR"},
        "13524": {"name": "Justin Jefferson", "team": "CLE", "position": "LB"},
    }
    monkeypatch.setattr(service, "resolve_player",
                        lambda name, roster_player_ids=None: _real_resolve(
                            name, index=jeff_index,
                            roster_player_ids=roster_player_ids,
                            footprint_ids={"6794"}))
    monkeypatch.setattr(service, "classify",
                        _route("player", player_names=["Justin Jefferson"]))
    seen = {}

    def pack(bundle, entities, player_ids):
        seen["player_ids"] = list(player_ids)
        return ({"type": "player", "available": True,
                 "player": {"player_id": "6794", "name": "Justin Jefferson"},
                 "as_of": {}}, ["fake_tool"])
    monkeypatch.setitem(service.BUILDERS, "player", pack)
    monkeypatch.setattr(service, "compose_answer",
                        lambda intent, q, p, *, preferences, assumed_context, history=None:
                        {"verdict": "Justin Jefferson update", "text": "ok",
                         "receipts": [], "caveat": None, "followups": [],
                         "act_link": None, "actions": []})
    out = service.ask("acct:42", "How is Justin Jefferson doing?")
    assert out["intent"] == "player"
    assert seen["player_ids"] == ["6794"]


def test_remember_stores_preference_and_next_answer_uses_it(env, monkeypatch):
    monkeypatch.setattr(service, "classify",
                        _route("remember", preference_text="I prefer upside at flex"))
    out = service.ask("acct:42", "Remember: I prefer upside at flex")
    assert out["intent"] == "remember"
    assert [p["text"] for p in store.list_preferences("acct:42")] == ["I prefer upside at flex"]

    seen = {}

    def fake_compose(intent, question, pack, *, preferences, assumed_context, history=None):
        seen["prefs"] = [p["text"] for p in preferences]
        return {"verdict": "START X", "text": "ok", "receipts": [], "caveat": None,
                "followups": [], "act_link": None, "actions": []}
    monkeypatch.setattr(service, "compose_answer", fake_compose)
    monkeypatch.setattr(service, "classify", _route("start_sit", player_names=["Rome Odunze"]))
    monkeypatch.setitem(service.BUILDERS, "start_sit", _start_sit_pack)
    service.ask("acct:42", "Start Odunze?")
    assert seen["prefs"] == ["I prefer upside at flex"]


def test_metric_no_data_pack_returns_no_data_shape(env, monkeypatch):
    monkeypatch.setattr(service, "classify", _route("metric", metric_hint="route rate"))
    monkeypatch.setitem(service.BUILDERS, "metric",
                        lambda b, e, p: ({"type": "metric", "available": False, "missing": "routes not tracked",
                                          "nearest": ["target_share"]}, ["advanced_metrics.config"]))
    out = service.ask("acct:42", "Who leads in route rate?")
    assert out["intent"] == "metric"
    assert "Routes not tracked" in out["answer"]["text"]
    assert out["answer"]["receipts"] == []
    assert env["logs"][-1]["intent"] == "metric"


def test_out_of_scope_is_logged(env, monkeypatch):
    monkeypatch.setattr(service, "classify", _route("out_of_scope"))
    out = service.ask("acct:42", "Who won the Super Bowl in 1999?")
    assert out["intent"] == "out_of_scope"
    assert env["logs"][-1]["intent"] == "out_of_scope"


def test_evidence_metric_builder_no_data_for_route_rate(monkeypatch):
    # Direct builder check against the real config source name handling.
    from dashboard_services.assistant import evidence
    pack, tools = evidence.build_metric(dict(BUNDLE), {"metric_hint": "route rate"})
    assert pack["available"] is False
    assert pack["missing"] == "routes not tracked"
    assert "target_share" in pack["nearest"]


# -- corrections (fix 5) --------------------------------------------------------

def _seed_start_sit_answer(env, monkeypatch):
    monkeypatch.setattr(service, "classify", _route("start_sit", player_names=["Rome Odunze", "Darnell Mooney"]))
    monkeypatch.setitem(service.BUILDERS, "start_sit", _start_sit_pack)
    monkeypatch.setattr(service, "compose_answer",
                        lambda intent, q, pack, *, preferences, assumed_context, history=None:
                        {"verdict": "START Rome Odunze", "text": "Start Odunze.",
                         "receipts": [{"label": "Odunze value", "value": "100"}],
                         "caveat": None, "followups": [], "act_link": None, "actions": []})
    out = service.ask("acct:42", "Start Odunze or Mooney?")
    return out["log_id"]


def test_correction_wrong_data_recheck_changed_posts_revision(env, monkeypatch):
    log_id = _seed_start_sit_answer(env, monkeypatch)
    # Fresh evidence flips the call: the re-check must post a revised answer.
    monkeypatch.setitem(service.BUILDERS, "start_sit",
                        lambda b, e, p: ({"type": "start_sit", "available": True, "as_of": {}}, ["fresh_tool"]))
    monkeypatch.setattr(service, "compose_answer",
                        lambda intent, q, pack, *, preferences, assumed_context, history=None:
                        {"verdict": "START Darnell Mooney", "text": "Mooney now.",
                         "receipts": [{"label": "Mooney value", "value": "120"}],
                         "caveat": None, "followups": [], "act_link": None, "actions": []})
    result = service.apply_correction(log_id, "acct:42", category="wrong_data",
                                      text="Odunze is actually hurt")
    assert result["recheck"] == "changed"
    assert result["reask_flag"] is True
    assert result["corrected_claim"] == "Odunze is actually hurt"
    assert result["revision"]["answer"]["verdict"] == "START Darnell Mooney"
    orig = next(r for r in env["logs"] if r["log_id"] == log_id)
    assert orig["correction_category"] == "wrong_data"
    assert orig["corrected_claim"] == "Odunze is actually hurt"
    assert orig["reask_flag"] is True
    revision_row = env["logs"][-1]
    assert revision_row["log_id"] == result["revision"]["log_id"]
    assert revision_row["revises_log_id"] == log_id
    assert revision_row["trigger"] == "correction_recheck"


def test_correction_wrong_data_recheck_unchanged(env, monkeypatch):
    log_id = _seed_start_sit_answer(env, monkeypatch)
    monkeypatch.setitem(service.BUILDERS, "start_sit",
                        lambda b, e, p: ({"type": "start_sit", "available": True, "as_of": {}}, ["fresh_tool"]))
    monkeypatch.setattr(service, "compose_answer",
                        lambda intent, q, pack, *, preferences, assumed_context, history=None:
                        {"verdict": "START Rome Odunze", "text": "Still Odunze.",
                         "receipts": [{"label": "Odunze value", "value": "100"}],
                         "caveat": None, "followups": [], "act_link": None, "actions": []})
    n_logs = len(env["logs"])
    result = service.apply_correction(log_id, "acct:42", category="wrong_data", text=None)
    assert result["recheck"] == "unchanged"
    assert result["revision"] is None
    # No free-text correction: the disputed claim falls back to the call itself.
    assert result["corrected_claim"] == "START Rome Odunze"
    assert len(env["logs"]) == n_logs  # no revision row written


def test_correction_bad_reasoning_no_recheck(env, monkeypatch):
    log_id = _seed_start_sit_answer(env, monkeypatch)
    result = service.apply_correction(log_id, "acct:42", category="bad_reasoning", text=None)
    assert result["recheck"] is None
    assert result["reask_flag"] is False
    assert result["corrected_claim"] == "START Rome Odunze"
    orig = next(r for r in env["logs"] if r["log_id"] == log_id)
    assert orig["correction_category"] == "bad_reasoning"
    assert orig["reask_flag"] is False


def test_correction_other_users_log_not_found(env, monkeypatch):
    log_id = _seed_start_sit_answer(env, monkeypatch)
    assert service.apply_correction(log_id, "acct:99", category="wrong_data") is None


# -- Per-conversation resolve snapshot ---------------------------------------

def _counting_ask_env(env, monkeypatch):
    counts = {"resolve": 0, "plan": 0, "prefs": 0, "used": 0}
    real_resolve = service.ctx_mod.resolve_league
    real_plan = service._active_plan
    real_prefs = store.list_preferences
    real_used = store.count_monthly_asked

    def resolve(key, **kw):
        counts["resolve"] += 1
        return real_resolve(key, **kw)

    def plan(key):
        counts["plan"] += 1
        return real_plan(key)

    def prefs(key):
        counts["prefs"] += 1
        return real_prefs(key)

    def used(key):
        counts["used"] += 1
        return real_used(key)

    monkeypatch.setattr(service.ctx_mod, "resolve_league", resolve)
    monkeypatch.setattr(service, "_active_plan", plan)
    monkeypatch.setattr(store, "list_preferences", prefs)
    monkeypatch.setattr(store, "count_monthly_asked", used)
    monkeypatch.setattr(service, "classify",
                        _route("start_sit", player_names=["Rome Odunze"]))
    monkeypatch.setitem(service.BUILDERS, "start_sit", _start_sit_pack)
    monkeypatch.setattr(service, "compose_answer",
                        lambda intent, q, pack, *, preferences, assumed_context, history=None:
                        {"verdict": "START Rome Odunze", "text": "Start Odunze.",
                         "receipts": [], "caveat": None, "followups": [],
                         "act_link": None, "actions": []})
    return counts


def test_resolve_snapshot_serves_second_turn_from_cache(env, monkeypatch):
    counts = _counting_ask_env(env, monkeypatch)
    service.ask("acct:42", "Start Odunze?", thread_id="t1")
    service.ask("acct:42", "Start Odunze?", thread_id="t1")
    # League resolution, plan lookup and preferences are paid once per
    # conversation; the monthly used count is re-read every message.
    assert counts == {"resolve": 1, "plan": 1, "prefs": 1, "used": 2}
    # A league switch mid-thread misses and re-resolves.
    service.ask("acct:42", "Start Odunze?", thread_id="t1", league_id="L2")
    assert counts["resolve"] == 2 and counts["plan"] == 2 and counts["prefs"] == 2
    assert counts["used"] == 3


def test_resolve_snapshot_invalidated_by_remember(env, monkeypatch):
    counts = _counting_ask_env(env, monkeypatch)
    service.ask("acct:42", "Start Odunze?", thread_id="t1")
    assert counts["prefs"] == 1
    monkeypatch.setattr(service, "classify",
                        _route("remember", preference_text="I prefer upside at flex"))
    service.ask("acct:42", "Remember: I prefer upside at flex", thread_id="t1")
    monkeypatch.setattr(service, "classify",
                        _route("start_sit", player_names=["Rome Odunze"]))
    service.ask("acct:42", "Start Odunze?", thread_id="t1")
    # The remember turn stored a preference and dropped the snapshot, so
    # the next turn re-reads preferences (and sees the new one).
    assert counts["prefs"] == 2
    assert [p["text"] for p in store.list_preferences("acct:42")] == [
        "I prefer upside at flex"]
