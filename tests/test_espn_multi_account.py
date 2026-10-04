"""Tests for ESPN multi-account support: SWID fingerprinting, smart 403 messages,
and per-account credential row reuse (no cross-account overwrite).
"""
import pytest

flask = pytest.importorskip("flask")


class TestSwidNormalization:
    def test_normalize_strips_braces(self):
        from dashboard_services.accounts import normalize_espn_swid
        assert normalize_espn_swid("{ABC-123}") == "ABC-123"

    def test_normalize_uppercases(self):
        from dashboard_services.accounts import normalize_espn_swid
        assert normalize_espn_swid("{abc-123}") == "ABC-123"

    def test_normalize_tolerates_missing_braces(self):
        from dashboard_services.accounts import normalize_espn_swid
        assert normalize_espn_swid("ABC-123") == "ABC-123"

    def test_normalize_empty(self):
        from dashboard_services.accounts import normalize_espn_swid
        assert normalize_espn_swid("") == ""
        assert normalize_espn_swid(None) == ""

    def test_fingerprint_stable_across_formats(self):
        from dashboard_services.accounts import espn_swid_fingerprint
        assert (espn_swid_fingerprint("{ABC-123}") ==
                espn_swid_fingerprint("abc-123") ==
                espn_swid_fingerprint("  {ABC-123}  "))

    def test_fingerprint_differs_across_accounts(self):
        from dashboard_services.accounts import espn_swid_fingerprint
        assert (espn_swid_fingerprint("{ABC-123}") !=
                espn_swid_fingerprint("{XYZ-789}"))

    def test_fingerprint_is_short_hex(self):
        from dashboard_services.accounts import espn_swid_fingerprint
        fp = espn_swid_fingerprint("{ABC-123}")
        assert len(fp) == 16
        int(fp, 16)  # valid hex


class TestSmartEspnError:
    """The 403 message distinguishes wrong-login from expired session."""

    @pytest.fixture
    def client(self, monkeypatch):
        from routes.link_bp import link_bp
        app = flask.Flask(__name__)
        app.secret_key = "test"
        app.register_blueprint(link_bp)
        import dashboard_services.providers.espn_api as espn
        import dashboard_services.accounts as accounts

        def denied(*a, **k):
            raise espn.ESPNAccessDenied("denied")

        monkeypatch.setattr(espn, "connect_league", denied)
        # Never touch the real DB in these tests.
        monkeypatch.setattr(accounts, "espn_swid_used_by_other_league",
                            lambda *a, **k: False)
        monkeypatch.setattr(accounts, "add_espn_league_connection",
                            lambda *a, **k: None)
        monkeypatch.setattr(accounts, "list_espn_credential_sets",
                            lambda *a, **k: [])
        with app.test_client() as test_client:
            with test_client.session_transaction() as sess:
                sess["account_id"] = 7
            yield test_client

    def _post(self, client, swid="{AAA}"):
        return client.post("/api/link/espn/private", json={
            "league_id": "123", "season": 2026,
            "swid": swid, "espn_s2": "secret",
        })

    def test_expired_session_message_when_no_other_league(self, client):
        resp = self._post(client)
        assert resp.status_code == 403
        assert "session has expired" in resp.json["error"]
        assert "different ESPN login" not in resp.json["error"]

    def test_wrong_login_message_when_swid_used_elsewhere(self, client, monkeypatch):
        import dashboard_services.accounts as accounts
        monkeypatch.setattr(accounts, "espn_swid_used_by_other_league",
                            lambda *a, **k: True)
        resp = self._post(client)
        assert resp.status_code == 403
        assert "different ESPN login" in resp.json["error"]
        assert "league 123" in resp.json["error"]

    def test_no_suggested_switch_when_no_other_credentials(self, client):
        resp = self._post(client)
        assert resp.status_code == 403
        assert "suggested_switch" not in resp.json

    def test_suggested_switch_when_other_login_works(self, client, monkeypatch):
        import dashboard_services.providers.espn_api as espn
        import dashboard_services.accounts as accounts

        working = {"swid": "{BBB}", "espn_s2": "other-secret"}

        def connect_or_deny(season, league_id, swid=None, espn_s2=None):
            if swid == "{BBB}":
                return {"name": "Other League"}
            raise espn.ESPNAccessDenied("denied")

        monkeypatch.setattr(espn, "connect_league", connect_or_deny)
        monkeypatch.setattr(
            accounts, "list_espn_credential_sets",
            lambda account_id: [{
                "connection_id": 42,
                "swid_fingerprint": "fp-bbb",
                "credentials": working,
                "updated_at": None,
            }],
        )
        resp = self._post(client, swid="{AAA}")
        assert resp.status_code == 403
        assert resp.json.get("suggested_switch") is True
        assert resp.json.get("suggested_connection_id") == 42

    def test_use_saved_endpoint_links_league(self, client, monkeypatch):
        import dashboard_services.providers.espn_api as espn
        import dashboard_services.accounts as accounts

        monkeypatch.setattr(
            espn, "connect_league", lambda *a, **k: {"name": "Saved League"})
        monkeypatch.setattr(
            accounts, "list_espn_credential_sets",
            lambda account_id: [{
                "connection_id": 42,
                "swid_fingerprint": "fp",
                "credentials": {"swid": "{BBB}", "espn_s2": "s"},
                "updated_at": None,
            }],
        )
        linked = {}
        monkeypatch.setattr(
            accounts, "link_espn_league_to_connection",
            lambda account_id, league_id, season, name, connection_id:
                linked.update(connection_id=connection_id, name=name),
        )
        resp = client.post("/api/link/espn/use-saved", json={
            "league_id": "123", "season": 2026, "connection_id": 42,
        })
        assert resp.status_code == 200
        assert resp.json["ok"] is True
        assert linked["connection_id"] == 42

    def test_use_saved_rejects_unknown_connection(self, client, monkeypatch):
        import dashboard_services.accounts as accounts
        monkeypatch.setattr(accounts, "list_espn_credential_sets",
                            lambda account_id: [])
        resp = client.post("/api/link/espn/use-saved", json={
            "league_id": "123", "season": 2026, "connection_id": 99,
        })
        assert resp.status_code == 404
