"""The streamlined ESPN connect flow lets a member paste a whole cookie string
(or the two rows) into either field; the server pulls out SWID + espn_s2."""
import pytest

flask = pytest.importorskip("flask")

from routes.link_bp import link_bp, _extract_espn_credentials


@pytest.mark.parametrize("swid_in, s2_in, expect_swid, expect_s2", [
    # already-clean values pass straight through (the historical happy path)
    ("{ABC-123}", "AEBtok%2F", "{ABC-123}", "AEBtok%2F"),
    # the whole Cookie header pasted into the espn_s2 box
    ("", "SWID={ABC-123}; espn_s2=AEBlong%3D%3D; other=1", "{ABC-123}", "AEBlong%3D%3D"),
    # ...or into the SWID box
    ("SWID={ABC-123}; espn_s2=AEBlong; foo=bar", "", "{ABC-123}", "AEBlong"),
    # surrounding whitespace is trimmed
    ("  {ABC-123}  ", "  AEBtok  ", "{ABC-123}", "AEBtok"),
    # a quoted value in one field, a clean SWID in the other
    ("{ABC-123}", 'espn_s2="AEBquoted"', "{ABC-123}", "AEBquoted"),
])
def test_extract_espn_credentials(swid_in, s2_in, expect_swid, expect_s2):
    assert _extract_espn_credentials(swid_in, s2_in) == (expect_swid, expect_s2)


@pytest.fixture
def client(monkeypatch):
    app = flask.Flask(__name__)
    app.secret_key = "test"
    app.register_blueprint(link_bp)
    import dashboard_services.providers.espn_api as espn
    import dashboard_services.accounts as accounts
    seen = {}

    def fake_connect(season, league_id, swid=None, espn_s2=None):
        seen["swid"], seen["espn_s2"] = swid, espn_s2
        return {"name": "Test League"}

    monkeypatch.setattr(espn, "connect_league", fake_connect)
    monkeypatch.setattr(accounts, "add_espn_league_connection", lambda *a, **k: None)
    with app.test_client() as test_client:
        with test_client.session_transaction() as sess:
            sess["account_id"] = 7
        yield test_client, seen


def test_pasted_blob_connects_with_extracted_values(client):
    test_client, seen = client
    blob = "SWID={ABC-123}; espn_s2=AEBpasted%3D; s_ecid=noise"
    response = test_client.post(
        "/api/link/espn/private",
        json={"league_id": "123", "season": 2026, "swid": "", "espn_s2": blob},
    )
    assert response.status_code == 200
    # connect_league saw the two isolated values, not the raw blob
    assert seen["swid"] == "{ABC-123}"
    assert seen["espn_s2"] == "AEBpasted%3D"
