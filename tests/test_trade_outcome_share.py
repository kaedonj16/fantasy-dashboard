"""Shareable trade-outcome links: POST /api/save-trade-outcome mints a short id,
GET /trade-outcome-card/<id> renders the frozen "who won the trade" card, and
/o/<id> redirects to it. Unknown ids 404.

The Postgres layer is faked with an in-memory dict (monkeypatched), so these
run without a database.
"""
from __future__ import annotations

import pytest

pytest.importorskip("flask")


def _payload(**over):
    p = {
        "team_a": "Caleb's Casting Couch",
        "team_b": "Pittsburgh Pilots",
        "trade_date": "2026-08-20",
        "verdict": "WIN",
        "net_delta_now": 142.3,
        "total_received_now": 812.4,
        "total_sent_now": 670.1,
        "then_estimated": False,
        "received": [
            {"id": "1", "name": "Ja'Marr Chase", "is_pick": False,
             "value_now": 512.4, "value_then": 480.0, "delta": 32.4},
            {"id": "2026_1_mid", "name": "2026 1st (Mid)", "is_pick": True,
             "value_now": 300.0, "value_then": None, "delta": None},
        ],
        "sent": [
            {"id": "2", "name": "Drake London", "is_pick": False,
             "value_now": 670.1, "value_then": 700.0, "delta": -29.9},
        ],
    }
    p.update(over)
    return p


@pytest.fixture
def fake_store(monkeypatch):
    import datetime
    store = {}

    def fake_create(params):
        sid = "oc" + str(len(store) + 1)
        store[sid] = {"params": params,
                      "created_at": datetime.datetime.now(datetime.timezone.utc)}
        return sid

    def fake_get(share_id):
        row = store.get((share_id or "").strip())
        if not row:
            return None
        return {"params": row["params"], "created_at": row["created_at"]}

    import app as appmod
    monkeypatch.setattr(appmod, "create_outcome_share", fake_create)
    monkeypatch.setattr(appmod, "get_outcome_share", fake_get)
    return store


@pytest.fixture
def client(fake_store):
    from app import app as flask_app
    flask_app.config.update(TESTING=True)
    return flask_app.test_client()


def test_save_and_card_render_win(client):
    res = client.post("/api/save-trade-outcome", json=_payload())
    assert res.status_code == 200, res.get_data(as_text=True)
    share_id = res.get_json()["share_id"]
    assert share_id

    card = client.get(f"/trade-outcome-card/{share_id}")
    assert card.status_code == 200
    html = card.get_data(as_text=True)
    assert "Caleb&#x27;s Casting Couch won the trade" in html
    assert "Pittsburgh Pilots" in html
    assert "Ja&#x27;Marr Chase" in html
    assert "Drake London" in html
    assert "2026 1st (Mid)" in html
    assert "+32.4" in html and "-29.9" in html
    assert "+142.3 value since the trade" in html
    assert "noindex" in html  # share links must not be indexed


def test_card_renders_loss_and_even(client):
    res = client.post("/api/save-trade-outcome",
                      json=_payload(verdict="LOSS", net_delta_now=-50.0))
    sid = res.get_json()["share_id"]
    html = client.get(f"/trade-outcome-card/{sid}").get_data(as_text=True)
    assert "Pittsburgh Pilots won the trade" in html

    res = client.post("/api/save-trade-outcome",
                      json=_payload(verdict="EVEN", net_delta_now=0.0))
    sid = res.get_json()["share_id"]
    html = client.get(f"/trade-outcome-card/{sid}").get_data(as_text=True)
    assert "Dead even" in html


def test_card_renders_then_estimated_badge(client):
    res = client.post("/api/save-trade-outcome",
                      json=_payload(then_estimated=True))
    sid = res.get_json()["share_id"]
    html = client.get(f"/trade-outcome-card/{sid}").get_data(as_text=True)
    assert "approximate" in html.lower()


def test_short_url_redirects_to_card(client):
    sid = client.post("/api/save-trade-outcome", json=_payload()).get_json()["share_id"]
    res = client.get(f"/o/{sid}", follow_redirects=False)
    assert res.status_code == 302
    assert res.headers["Location"].endswith(f"/trade-outcome-card/{sid}")


def test_save_rejects_empty_payload(client):
    res = client.post("/api/save-trade-outcome", json={"verdict": "WIN"})
    assert res.status_code == 400


def test_card_404_for_unknown_id(client):
    assert client.get("/trade-outcome-card/doesnotexist").status_code == 404


def test_short_url_404_for_unknown_id(client):
    assert client.get("/o/doesnotexist").status_code == 404


def test_sanitize_caps_and_normalizes():
    from dashboard_services.trade_outcome_shares import sanitize_outcome_params
    rows = [{"name": "P" + str(i), "value_now": i} for i in range(100)]
    p = sanitize_outcome_params({
        "team_a": "A" * 200, "team_b": "", "verdict": "BOGUS",
        "received": rows, "sent": [{"name": "X" * 200, "value_now": 5}],
    })
    assert len(p["a_rows"]) == 30          # per-side cap
    assert len(p["team_a"]) == 60          # team name cap
    assert p["verdict"] == "EVEN"          # unknown verdict normalizes
    assert p["team_b"] == "Team B"         # empty team name defaults
    assert len(p["b_rows"][0]["name"]) == 80
