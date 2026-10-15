"""Stage 2 inferred preferences: proposals from the advice log, the
confirm-before-save flow, and the confirmed risk profile in compose.

No real DB, no OpenAI: store.get_conn is faked in-memory (a local fake
that knows the preferences/proposals/inference SQL), and the model is
faked by monkeypatching compose.call_model_json.
"""
import json
import re

import pytest

pytest.importorskip("flask")
pytest.importorskip("openai")
pytestmark = pytest.mark.integration

from dashboard_services.assistant import compose as compose_mod
from dashboard_services.assistant import infer, service, store


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
        if s.startswith(("CREATE", "ALTER")):
            return FakeCursor()
        if "INSERT INTO assistant_preferences" in s:
            row = dict(zip(self._cols(s), params))
            st["pref_seq"] += 1
            row["id"] = st["pref_seq"]
            row["created_at"] = "2026-10-14T00:00:00"
            st["prefs"].append(row)
            return FakeCursor(row=dict(row))
        if "INSERT INTO assistant_preference_proposals" in s:
            row = dict(zip(self._cols(s), params))
            if "status" not in row:
                row["status"] = "pending"  # SQL literal in the real statement
            st["prop_seq"] += 1
            row["id"] = st["prop_seq"]
            row["created_at"] = "2026-10-14T00:00:00"
            row["updated_at"] = "2026-10-14T00:00:00"
            st["proposals"].append(row)
            return FakeCursor(row=dict(row))
        if "FROM assistant_preferences" in s and s.startswith("SELECT"):
            return FakeCursor(rows=[dict(r) for r in st["prefs"] if r.get("account_key") == params[0]])
        if "FROM assistant_preference_proposals" in s and s.startswith("SELECT"):
            if s.startswith("SELECT id FROM"):
                rows = [r for r in st["proposals"] if r.get("id") == params[0]
                        and r.get("account_key") == params[1] and r.get("status") == "pending"]
                return FakeCursor(row=dict(rows[0]) if rows else None)
            if "WHERE id" in s:
                rows = [r for r in st["proposals"] if r.get("id") == params[0]
                        and r.get("account_key") == params[1] and r.get("status") == "pending"]
                return FakeCursor(row=dict(rows[0]) if rows else None)
            rows = [dict(r) for r in st["proposals"] if r.get("account_key") == params[0]]
            if len(params) > 1:
                rows = [r for r in rows if r.get("status") == params[1]]
            return FakeCursor(rows=rows)
        if "UPDATE assistant_preference_proposals" in s:
            if "preference_id" in s and "SET status = 'dismissed'" in s:
                for r in st["proposals"]:
                    if r.get("preference_id") == params[0] and r.get("account_key") == params[1]:
                        r["status"] = "dismissed"
                return FakeCursor()
            if "SET status = 'confirmed'" in s:
                for r in st["proposals"]:
                    if r.get("id") == params[1] and r.get("account_key") == params[2]:
                        r["status"] = "confirmed"
                        r["preference_id"] = params[0]
                return FakeCursor()
            if "SET status = 'dismissed'" in s:
                for r in st["proposals"]:
                    if r.get("id") == params[0] and r.get("account_key") == params[1]:
                        r["status"] = "dismissed"
                return FakeCursor()
            if "SET text" in s:
                for r in st["proposals"]:
                    if r.get("id") == params[4] and r.get("account_key") == params[5]:
                        r["text"], r["evidence"] = params[0], params[1]
                return FakeCursor()
            raise AssertionError(f"fake DB cannot handle UPDATE: {s}")
        if "DELETE FROM assistant_preference_proposals" in s:
            if "AND status = 'pending'" in s:
                st["proposals"] = [r for r in st["proposals"]
                                    if not (r.get("id") == params[0] and r.get("account_key") == params[1])]
            else:
                st["proposals"] = [r for r in st["proposals"] if r.get("account_key") != params[0]]
            return FakeCursor()
        if "DELETE FROM assistant_preferences" in s:
            if "WHERE id" in s:
                st["prefs"] = [r for r in st["prefs"]
                               if not (r.get("id") == params[0] and r.get("account_key") == params[1])]
            else:
                st["prefs"] = [r for r in st["prefs"] if r.get("account_key") != params[0]]
            return FakeCursor()
        if "FROM assistant_advice_log" in s and s.startswith("SELECT"):
            rows = [dict(r) for r in st["logs"]
                    if r.get("account_key") == params[0] and not r.get("proactive")]
            rows.sort(key=lambda r: r["log_id"], reverse=True)
            return FakeCursor(rows=rows[:params[1]])
        raise AssertionError(f"fake DB cannot handle SQL: {s}")


@pytest.fixture()
def state(monkeypatch):
    st = {"logs": [], "prefs": [], "proposals": [],
          "log_seq": 0, "pref_seq": 0, "prop_seq": 0}
    conn = FakeConn(st)
    monkeypatch.setattr(store, "_TABLE_READY", False)
    monkeypatch.setattr(store, "get_conn", lambda *a, **k: conn)
    return st


def _add_log(state, account="acct:1", **fields):
    state["log_seq"] += 1
    row = {"log_id": state["log_seq"], "account_key": account, "proactive": False,
           "created_at": "2026-10-10T12:00:00", "intent": "player", "question_text": "",
           "thumb": None, "verdict": {}, "evidence_pack": {}}
    row.update(fields)
    state["logs"].append(row)
    return row


_BOOM = [{"week": w, "points": p} for w, p in ((3, 4.0), (4, 28.0), (5, 6.0), (6, 31.0))]
_STEADY = [{"week": w, "points": p} for w, p in ((3, 14.0), (4, 15.0), (5, 13.0), (6, 15.5))]


def _ss_row(state, account="acct:1", *, pick="boom", thumb="up", thin=True,
            question="Start A or B?"):
    pack = {"type": "start_sit", "pick_player_id": pick, "margin": {"thin": thin},
            "players": [{"player_id": "boom", "name": "Boom", "recent_form": list(_BOOM)},
                        {"player_id": "steady", "name": "Steady", "recent_form": list(_STEADY)}]}
    return _add_log(state, account, intent="start_sit", question_text=question,
                    thumb=thumb, evidence_pack=pack)


def _faab_row(state, account="acct:1", *, pct=22.0, thumb="up"):
    pack = {"type": "waivers", "faab": {"enabled": True},
            "candidates": [{"player_id": "x", "name": "X", "faab": {"pct_target": pct}}]}
    return _add_log(state, account, intent="waivers", question_text="Who should I add?",
                    thumb=thumb, evidence_pack=pack)


def _trade_row(state, account="acct:1", *, call="ACCEPT", thumb="down"):
    pack = {"type": "trade_eval", "trade_verdict": call}
    return _add_log(state, account, intent="trade_eval", question_text="Should I do this trade?",
                    thumb=thumb, evidence_pack=pack, verdict={"intent": "trade_eval", "call": call})


def _kinds(candidates):
    return {c["kind"]: c for c in candidates}


# -- inference: thresholds and kinds -------------------------------------------

def test_lineup_upside_proposed_from_backed_close_calls(state):
    for _ in range(4):
        _ss_row(state, pick="boom")
    cands = _kinds(infer.candidates_from_log(store.list_log_for_inference("acct:1")))
    assert "lineup_risk_upside" in cands
    ev = cands["lineup_risk_upside"]["evidence"]
    assert ev["supporting"] == 4
    assert len(ev["log_ids"]) == 4
    assert cands["lineup_risk_upside"]["dimension"] == "lineup_risk"
    assert cands["lineup_risk_upside"]["value"] == "upside"


def test_lineup_floor_proposed_from_backed_close_calls(state):
    for _ in range(3):
        _ss_row(state, pick="steady")
    cands = _kinds(infer.candidates_from_log(store.list_log_for_inference("acct:1")))
    assert "lineup_risk_floor" in cands


def test_no_proposal_below_threshold(state):
    for _ in range(2):
        _ss_row(state, pick="boom")
    cands = infer.candidates_from_log(store.list_log_for_inference("acct:1"))
    assert not any(c["kind"].startswith("lineup_risk") for c in cands)


def test_no_proposal_when_signals_split(state):
    for _ in range(3):
        _ss_row(state, pick="boom")
        _ss_row(state, pick="steady")
    cands = infer.candidates_from_log(store.list_log_for_inference("acct:1"))
    assert not any(c["kind"].startswith("lineup_risk") for c in cands)


def test_no_proposal_without_thumb_or_thin_margin(state):
    for _ in range(4):
        _ss_row(state, pick="boom", thumb=None)
        _ss_row(state, pick="boom", thin=False)
    cands = infer.candidates_from_log(store.list_log_for_inference("acct:1"))
    assert not any(c["kind"].startswith("lineup_risk") for c in cands)


def test_faab_aggressive_and_trade_patient_and_position_lean(state):
    for _ in range(3):
        _faab_row(state, pct=22.0)
    for _ in range(4):
        _trade_row(state, call="ACCEPT", thumb="down")
    for i in range(4):
        _add_log(state, intent="player", question_text=f"How is my tight end doing? {i}")
    _add_log(state, intent="player", question_text="How is my running back doing?")
    cands = _kinds(infer.candidates_from_log(store.list_log_for_inference("acct:1")))
    assert cands["faab_aggressive"]["dimension"] == "faab_posture"
    assert cands["trade_patient"]["dimension"] == "trade_posture"
    assert cands["position_lean_te"]["value"] == "TE"
    assert cands["position_lean_te"]["dimension"] is None


def test_faab_midrange_bids_make_no_proposal(state):
    for _ in range(4):
        _faab_row(state, pct=10.0)
    cands = infer.candidates_from_log(store.list_log_for_inference("acct:1"))
    assert not any(c["kind"].startswith("faab") for c in cands)


# -- panel, confirm, dismiss -----------------------------------------------------

def test_panel_proposes_then_confirm_saves_preference_and_profile(state):
    for _ in range(4):
        _ss_row(state)
    panel = service.preference_panel("acct:1")
    assert [p["kind"] for p in panel["proposals"]] == ["lineup_risk_upside"]
    assert panel["risk_profile"] == {}
    proposal_id = panel["proposals"][0]["id"]

    result = service.confirm_proposal("acct:1", proposal_id)
    assert result is not None
    pref = result["preference"]
    assert pref["source"] == "inferred" and pref["kind"] == "lineup_risk_upside"
    assert result["risk_profile"] == {"lineup_risk": "upside"}
    assert any("Lineup risk" in line for line in result["profile_lines"])
    assert result["proposals"] == []

    # Confirmed kinds are not proposed again while the preference stands.
    panel2 = service.preference_panel("acct:1")
    assert panel2["proposals"] == []
    assert [p["text"] for p in panel2["preferences"]] == [
        "You usually take the upside in close start/sit calls."]


def test_dismissed_proposal_stays_down_for_same_evidence(state):
    for _ in range(3):
        _ss_row(state)
    panel = service.preference_panel("acct:1")
    proposal_id = panel["proposals"][0]["id"]
    assert service.dismiss_proposal("acct:1", proposal_id) is True
    assert service.preference_panel("acct:1")["proposals"] == []
    # More of the same behavior: the evidence set grows, so the proposal
    # may return, now carrying the larger evidence set.
    for _ in range(2):
        _ss_row(state)
    panel2 = service.preference_panel("acct:1")
    assert [p["kind"] for p in panel2["proposals"]] == ["lineup_risk_upside"]
    assert panel2["proposals"][0]["evidence"]["supporting"] == 5


def test_deleting_confirmed_preference_does_not_repropose(state):
    for _ in range(4):
        _ss_row(state)
    panel = service.preference_panel("acct:1")
    result = service.confirm_proposal("acct:1", panel["proposals"][0]["id"])
    store.delete_preference("acct:1", result["preference"]["id"])
    assert service.preference_panel("acct:1")["proposals"] == []


def test_forget_all_clears_proposals(state):
    for _ in range(4):
        _ss_row(state)
    assert service.preference_panel("acct:1")["proposals"]
    store.delete_all_preferences("acct:1")
    assert state["proposals"] == []


# -- privacy ---------------------------------------------------------------------

def test_proposals_never_read_another_accounts_log(state):
    for _ in range(4):
        _ss_row(state, account="acct:1")
    mine = infer.candidates_from_log(store.list_log_for_inference("acct:1"))
    theirs = infer.candidates_from_log(store.list_log_for_inference("acct:2"))
    assert mine and theirs == []
    my_ids = {r["log_id"] for r in state["logs"] if r["account_key"] == "acct:1"}
    for cand in mine:
        assert set(cand["evidence"]["log_ids"]) <= my_ids
    panel = service.preference_panel("acct:1")
    proposal_id = panel["proposals"][0]["id"]
    assert service.confirm_proposal("acct:2", proposal_id) is None
    assert service.dismiss_proposal("acct:2", proposal_id) is False
    # Still pending for its owner after the other account's attempts.
    assert service.preference_panel("acct:1")["proposals"]


# -- compose: profile shapes framing only -----------------------------------------

_PACK = {
    "type": "start_sit", "available": True, "week": 7,
    "players": [
        {"player_id": "1", "name": "Rome Odunze", "team": "CHI", "position": "WR",
         "projection": 17.9, "start_score": 82.5},
        {"player_id": "2", "name": "Darnell Mooney", "team": "ATL", "position": "WR",
         "projection": 11.0, "start_score": 61.0},
    ],
    "margin": {"thin": True}, "as_of": {"week": 7},
}

_CANNED = {"verdict": "START Rome Odunze",
           "text": "Odunze projects 17.9 points against 11.0 for Mooney.",
           "receipts": [{"label": "Odunze projection", "value": "17.9"}],
           "caveat": None, "followups": ["What about next week?"], "act_link": None}


def test_compose_profile_injected_and_framing_only(monkeypatch):
    seen = {}

    def fake(system, user, schema, name):
        seen["system"] = system
        seen["payload"] = json.loads(user)
        return dict(_CANNED)

    monkeypatch.setattr(compose_mod, "call_model_json", fake)

    compose_mod.compose_answer("start_sit", "Start Odunze or Mooney?", _PACK)
    assert seen["system"] == compose_mod.SYSTEM
    assert seen["payload"]["risk_profile"] == {}

    answer = compose_mod.compose_answer(
        "start_sit", "Start Odunze or Mooney?", _PACK,
        risk_profile={"lineup_risk": "upside"})
    assert seen["payload"]["risk_profile"] == {"lineup_risk": "upside"}
    assert seen["system"].startswith(compose_mod.SYSTEM)
    assert "risk_profile" in seen["system"]
    assert "never overrides a clear margin" in seen["system"]
    # Numbers come from the pack either way; the profile adds no figures.
    assert {"label": "Odunze projection", "value": "17.9"} in answer["receipts"]
    for r in answer["receipts"]:
        assert "9999" not in str(r["value"])


# -- UI wiring (source contract) ---------------------------------------------------

def test_memory_view_renders_proposals_and_profile():
    from pathlib import Path

    js = (Path(__file__).resolve().parents[1] / "static" / "app.js").read_text(encoding="utf-8")
    start = js.index("function initAssistant()")
    body = js[start:js.index("function init()", start)]
    assert "Suggested from your chats" in body
    assert "Your risk profile" in body
    assert "/api/assistant/preference-proposals/" in body
    assert "'/confirm'" in body and "'/dismiss'" in body
    assert "br-asst-prop-row" in body
    # Light UI only: proposal rows, no big card classes in the memory view.
    memory = body[body.index("function loadPreferences()"):]
    assert "ns-item" not in memory


# -- API ---------------------------------------------------------------------------

@pytest.fixture()
def client(state):
    from flask import Flask

    from routes.assistant_bp import assistant_bp

    app = Flask(__name__)
    app.secret_key = "test"
    app.register_blueprint(assistant_bp)
    c = app.test_client()
    with c.session_transaction() as sess:
        sess["account_id"] = 1
    return c


def test_api_panel_confirm_dismiss_and_guest_401(state, client):
    guest = client.application.test_client()
    assert guest.post("/api/assistant/preference-proposals/1/confirm").status_code == 401
    assert guest.post("/api/assistant/preference-proposals/1/dismiss").status_code == 401

    for _ in range(4):
        _ss_row(state)
    body = client.get("/api/assistant/preferences").get_json()
    assert [p["kind"] for p in body["proposals"]] == ["lineup_risk_upside"]
    proposal_id = body["proposals"][0]["id"]

    r = client.post(f"/api/assistant/preference-proposals/{proposal_id}/confirm")
    assert r.status_code == 200
    payload = r.get_json()
    assert payload["preference"]["source"] == "inferred"
    assert payload["risk_profile"] == {"lineup_risk": "upside"}
    # Confirming again is a 404: the proposal is no longer pending.
    assert client.post(f"/api/assistant/preference-proposals/{proposal_id}/confirm").status_code == 404
    assert client.post("/api/assistant/preference-proposals/999/dismiss").status_code == 404
