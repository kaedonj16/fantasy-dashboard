"""Assistant API tests: Flask test client, fake DB, canned model."""
import pytest

pytest.importorskip("flask")
pytest.importorskip("openai")
pytestmark = pytest.mark.integration

from flask import Flask

from dashboard_services.assistant import service, store
from routes.assistant_bp import assistant_bp
from test_assistant_service import BUNDLE, FakeConn, _route, _start_sit_pack


@pytest.fixture()
def state(monkeypatch):
    st = {"threads": {}, "logs": [], "prefs": [], "log_seq": 0, "pref_seq": 0}
    conn = FakeConn(st)
    monkeypatch.setattr(store, "_TABLE_READY", False)
    monkeypatch.setattr(store, "get_conn", lambda *a, **k: conn)
    monkeypatch.setattr(service, "_active_plan", lambda key: None)
    service._clear_resolve_snapshot_cache()
    # Digest builds run on a background thread in production; run them
    # inline here so context tests stay deterministic.
    monkeypatch.setattr(service, "_spawn_digest_thread", lambda fn: fn())
    monkeypatch.setattr(service.ctx_mod, "resolve_league", lambda key, **kw: dict(BUNDLE))
    return st


@pytest.fixture()
def app(state):
    app = Flask(__name__)
    app.secret_key = "test"
    app.register_blueprint(assistant_bp)
    return app


def _client(app, account_id=42):
    c = app.test_client()
    if account_id is not None:
        with c.session_transaction() as sess:
            sess["account_id"] = account_id
    return c


def test_guest_401_on_all_endpoints(app):
    c = _client(app, None)
    assert c.get("/api/assistant/context").status_code == 401
    assert c.post("/api/assistant/chat", json={"message": "hi"}).status_code == 401
    assert c.post("/api/assistant/feedback", json={"log_id": 1, "thumb": "up"}).status_code == 401
    assert c.post("/api/assistant/correction", json={"log_id": 1, "category": "other"}).status_code == 401
    assert c.get("/api/assistant/preferences").status_code == 401
    assert c.post("/api/assistant/preferences", json={"text": "x"}).status_code == 401
    assert c.delete("/api/assistant/preferences/1").status_code == 401
    assert c.delete("/api/assistant/preferences").status_code == 401
    assert c.get("/api/assistant/context").get_json() == {"error": "login_required"}


def test_context_happy_path_and_teaser_generated_once(app, state, monkeypatch):
    calls = {"n": 0}

    def fake_builder(bundle, entities, player_ids=None):
        return ({"type": "this_week", "available": True,
                 "actions": [{"tag": "lineup", "headline": "Start Odunze", "detail": "Mooney is on bye", "act_link": "/waivers"}],
                 "as_of": {}}, ["fake_digest"])

    def fake_compose(intent, question, pack, *, preferences, assumed_context, history=None):
        calls["n"] += 1
        return {"verdict": None, "text": "Do this.", "receipts": [], "caveat": None,
                "followups": [], "act_link": "/", "actions": pack.get("actions") or []}
    monkeypatch.setitem(service.BUILDERS, "this_week", fake_builder)
    monkeypatch.setattr(service, "compose_answer", fake_compose)

    c = _client(app)
    r1 = c.get("/api/assistant/context?platform=sleeper&league_id=L1&season=2026")
    assert r1.status_code == 200
    body = r1.get_json()
    assert body["ok"] is True
    assert body["league"]["league_name"] == "Blackedraw"
    caps = body["caps"]
    assert caps["used"] == 0 and caps["limit"] == 25 and caps["tier"] == "free"
    assert caps["tier_label"] == "Free" and caps["plan_key"] is None
    assert caps["exempt"] is False
    assert [(t["key"], t["limit"], t["current"]) for t in caps["tiers"]] == [
        ("free", 25, True), ("starter", 100, False),
        ("all_pro", 250, False), ("hall_of_fame", 1000, False),
    ]
    # The GET never composes inline: the first fetch only kicks the
    # (here, inline-run) background build, so it serves no teaser even
    # though the compose has now happened exactly once.
    assert body["teaser"] is None
    assert calls["n"] == 1

    r2 = c.get("/api/assistant/context")
    teaser = r2.get_json()["teaser"]
    assert teaser["kind"] == "digest"
    assert teaser["title"] == "Your Week 7 moves are ready"
    assert teaser["lead"] == "Start Odunze"
    assert calls["n"] == 1  # second fetch served from the stored proactive row


def test_context_get_never_composes_or_builds_inline(app, state, monkeypatch):
    # With the background spawn captured (not run), the GET must leave
    # builder and compose untouched and request exactly one kick.
    counts = {"build": 0, "compose": 0, "kick": 0}

    def fake_builder(bundle, entities, player_ids=None):
        counts["build"] += 1
        return ({"type": "this_week", "available": True, "actions": [], "as_of": {}}, [])

    def fake_compose(intent, question, pack, *, preferences, assumed_context, history=None):
        counts["compose"] += 1
        return {"verdict": None, "text": "x", "receipts": [], "caveat": None,
                "followups": [], "act_link": None, "actions": []}

    monkeypatch.setitem(service.BUILDERS, "this_week", fake_builder)
    monkeypatch.setattr(service, "compose_answer", fake_compose)
    monkeypatch.setattr(service, "_spawn_digest_thread",
                        lambda fn: counts.__setitem__("kick", counts["kick"] + 1))

    c = _client(app)
    body = c.get("/api/assistant/context?platform=sleeper&league_id=L1&season=2026").get_json()
    assert body["teaser"] is None
    assert counts == {"build": 0, "compose": 0, "kick": 1}
    # Single-flight: a second GET while the first build is still marked
    # in flight does not kick again.
    body2 = c.get("/api/assistant/context?platform=sleeper&league_id=L1&season=2026").get_json()
    assert body2["teaser"] is None
    assert counts["kick"] == 1


def test_context_caps_reflect_paid_tier(app, state, monkeypatch):
    monkeypatch.setattr(service, "_active_plan", lambda key: {"plan_key": "all_pro"})
    monkeypatch.setitem(service.BUILDERS, "this_week",
                        lambda bundle, entities, player_ids=None:
                        ({"type": "this_week", "available": True, "actions": [], "as_of": {}}, []))
    c = _client(app)
    body = c.get("/api/assistant/context").get_json()
    caps = body["caps"]
    assert caps["tier"] == "all_pro" and caps["tier_label"] == "All-Pro"
    assert caps["limit"] == 250 and caps["plan_key"] == "all_pro"
    assert caps["exempt"] is False
    assert [t["current"] for t in caps["tiers"]] == [False, False, True, False]
    assert body["teaser"] is None  # empty digest: marker row only, no teaser


def test_chat_happy_and_cap_429(app, state, monkeypatch):
    monkeypatch.setattr(service, "classify", _route("start_sit", player_names=["Rome Odunze"]))
    monkeypatch.setitem(service.BUILDERS, "start_sit", _start_sit_pack)
    monkeypatch.setattr(service, "compose_answer",
                        lambda intent, q, pack, *, preferences, assumed_context, history=None:
                        {"verdict": "START Rome Odunze", "text": "Start Odunze.", "receipts": [],
                         "caveat": None, "followups": [], "act_link": None, "actions": []})
    monkeypatch.setattr(service, "resolve_player",
                        lambda name, roster_player_ids=None: {"status": "resolved", "player_id": "1"})
    c = _client(app)
    r = c.post("/api/assistant/chat", json={"message": "Start Odunze?"})
    assert r.status_code == 200
    body = r.get_json()
    assert body["intent"] == "start_sit" and body["answer"]["verdict"] == "START Rome Odunze"
    assert body["thread_id"] and body["log_id"]

    monkeypatch.setenv("ASSISTANT_FREE_MONTHLY_CAP", "1")
    r2 = c.post("/api/assistant/chat", json={"message": "Start Odunze?"})
    assert r2.status_code == 429
    assert r2.get_json()["error"] == "cap_reached"
    # The 429 carries the full usage payload so the UI can open the plans panel.
    caps429 = r2.get_json()["caps"]
    assert caps429["used"] == 1 and caps429["limit"] == 1 and caps429["exempt"] is False
    assert [t["key"] for t in caps429["tiers"]] == ["free", "starter", "all_pro", "hall_of_fame"]


def test_chat_clarify_shape(app, state, monkeypatch):
    monkeypatch.setattr(service, "classify", _route("player", player_names=["Chris Smith"]))
    monkeypatch.setattr(service, "resolve_player",
                        lambda name, roster_player_ids=None: {"status": "ambiguous", "candidates": [
                            {"player_id": "3", "name": "Chris Smith", "team": "NYG", "position": "WR"},
                            {"player_id": "4", "name": "Chris Smith", "team": "DAL", "position": "RB"}]})
    c = _client(app)
    r = c.post("/api/assistant/chat", json={"message": "Chris Smith?"})
    assert r.status_code == 200
    body = r.get_json()
    assert body["intent"] == "clarify"
    assert body["answer"]["verdict"] is None
    assert len(body["answer"]["candidates"]) == 2


def test_feedback_and_correction_ownership_and_ack(app, state):
    log_id = store.write_log(account_key="acct:42", intent="start_sit", proactive=False)
    other_id = store.write_log(account_key="acct:99", intent="start_sit", proactive=False)
    c = _client(app, 42)
    assert c.post("/api/assistant/feedback", json={"log_id": log_id, "thumb": "up"}).status_code == 200
    assert state["logs"][0]["thumb"] == "up"
    assert c.post("/api/assistant/feedback", json={"log_id": other_id, "thumb": "up"}).status_code == 404
    r = c.post("/api/assistant/correction", json={"log_id": log_id, "category": "bad_reasoning", "text": "ignored the bye"})
    assert r.status_code == 200
    assert r.get_json()["ack"] == ("Logged under bad reasoning for that start/sit call. "
                                   "It goes into the weekly review of flagged answers.")
    assert r.get_json()["revision"] is None
    # The correction records the disputed claim and no re-ask for this category.
    assert state["logs"][0]["corrected_claim"] == "ignored the bye"
    assert state["logs"][0]["reask_flag"] is False
    assert c.post("/api/assistant/correction", json={"log_id": other_id, "category": "other"}).status_code == 404


def test_correction_wrong_data_sets_reask_and_claim(app, state):
    log_id = store.write_log(account_key="acct:42", intent="start_sit", proactive=False,
                             league_id="L1", platform="sleeper", season=2026, week=7,
                             verdict={"intent": "start_sit", "call": "START Rome Odunze"},
                             numbers_shown={"Odunze value": "100"})
    c = _client(app, 42)
    r = c.post("/api/assistant/correction", json={"log_id": log_id, "category": "wrong_data"})
    assert r.status_code == 200
    body = r.get_json()
    assert "wrong data" in body["ack"]
    row = state["logs"][0]
    assert row["reask_flag"] is True
    # No free text given: the disputed claim is the call on record.
    assert row["corrected_claim"] == "START Rome Odunze"


def test_preferences_crud_and_forget_all(app, state):
    c = _client(app)
    r = c.post("/api/assistant/preferences", json={"text": "Prefer upside at flex"})
    assert r.status_code == 200 and r.get_json()["preference"]["text"] == "Prefer upside at flex"
    pref_id = r.get_json()["preference"]["id"]
    assert len(c.get("/api/assistant/preferences").get_json()["preferences"]) == 1
    assert c.delete(f"/api/assistant/preferences/{pref_id}").status_code == 200
    assert c.get("/api/assistant/preferences").get_json()["preferences"] == []
    c.post("/api/assistant/preferences", json={"text": "Never suggest my keepers"})
    assert c.delete("/api/assistant/preferences").status_code == 200
    assert c.get("/api/assistant/preferences").get_json()["preferences"] == []
