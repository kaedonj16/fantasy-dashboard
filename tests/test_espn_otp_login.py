"""ESPN email + one-time-code sign-in.

Broker-level tests need no flask and run everywhere; the HTTP tests use the
mock broker behind the feature flag and skip where flask isn't installed.
"""
import time
import types

import pytest

from dashboard_services.providers import espn_login as L


# ── broker contract (no flask) ────────────────────────────────────────────────
def test_mock_happy_path():
    b = L.MockEspnLoginBroker()
    login_id = b.start("t@example.com")
    assert login_id
    assert b.verify(login_id, "123456") == {"swid": "{MOCK-SWID-0000-0000}", "espn_s2": "MOCK_ESPN_S2_VALUE"}


def test_wrong_code_raises_invalid():
    b = L.MockEspnLoginBroker()
    login_id = b.start("t@example.com")
    with pytest.raises(L.EspnLoginInvalidCode):
        b.verify(login_id, "000000")


def test_unknown_login_id_is_expired():
    with pytest.raises(L.EspnLoginExpired):
        L.MockEspnLoginBroker().verify("nope", "123456")


def test_start_requires_a_real_email():
    with pytest.raises(L.EspnLoginError):
        L.MockEspnLoginBroker().start("not-an-email")


def test_start_is_rate_limited_per_email():
    b = L.MockEspnLoginBroker()
    for _ in range(L._START_LIMIT):
        b.start("spammy@example.com")
    with pytest.raises(L.EspnLoginRateLimited):
        b.start("spammy@example.com")


def test_attempt_cap_burns_the_session():
    b = L.MockEspnLoginBroker()
    login_id = b.start("cap@example.com")
    for _ in range(L._MAX_VERIFY_ATTEMPTS):
        with pytest.raises(L.EspnLoginInvalidCode):
            b.verify(login_id, "000000")
    with pytest.raises(L.EspnLoginTooManyAttempts):
        b.verify(login_id, "000000")


def test_session_expires_after_ttl(monkeypatch):
    b = L.MockEspnLoginBroker()
    login_id = b.start("exp@example.com")
    real = time.time
    monkeypatch.setattr(time, "time", lambda: real() + L._SESSION_TTL + 1)
    with pytest.raises(L.EspnLoginExpired):
        b.verify(login_id, "123456")


def test_enabled_defers_to_broker_then_override(monkeypatch):
    monkeypatch.delenv("ESPN_OTP_LOGIN_ENABLED", raising=False)
    monkeypatch.delenv("ESPN_OTP_BROKER", raising=False)
    monkeypatch.delenv("ESPN_ONEID_API_KEY", raising=False)
    L._reset_broker()
    assert L.otp_login_enabled() is False              # default OneID driver, no key ⇒ off
    monkeypatch.setenv("ESPN_ONEID_API_KEY", "APIKEY test-value")
    L._reset_broker()
    assert L.otp_login_enabled() is True               # key present ⇒ on, no flag needed
    monkeypatch.setenv("ESPN_OTP_LOGIN_ENABLED", "off")
    assert L.otp_login_enabled() is False              # explicit kill-switch wins
    monkeypatch.delenv("ESPN_ONEID_API_KEY", raising=False)
    monkeypatch.setenv("ESPN_OTP_LOGIN_ENABLED", "true")
    L._reset_broker()
    assert L.otp_login_enabled() is True               # explicit on wins even without a key
    L._reset_broker()


def test_default_broker_is_oneid(monkeypatch):
    monkeypatch.delenv("ESPN_OTP_BROKER", raising=False)
    L._reset_broker()
    assert isinstance(L.get_broker(), L.OneIdOtpBroker)
    L._reset_broker()


def test_playwright_stub_is_unavailable():
    b = L.PlaywrightEspnLoginBroker()
    assert b.available() is False
    with pytest.raises(L.EspnLoginUnavailable):
        b.start("t@example.com")


def test_oneid_broker_unavailable_without_api_key(monkeypatch):
    monkeypatch.delenv("ESPN_ONEID_API_KEY", raising=False)
    b = L.OneIdOtpBroker()
    assert b.available() is False
    with pytest.raises(L.EspnLoginUnavailable):
        b.start("t@example.com")
    monkeypatch.setenv("ESPN_ONEID_API_KEY", "APIKEY test-value")
    assert b.available() is True


def test_oneid_retries_first_transient_code_rejection(monkeypatch):
    responses = [
        (400, {}),
        (200, {"data": {"swid": "{SWID}", "recoveryToken": {"access_token": "token"}}}),
        (200, {"data": {"s2": "cookie", "token": {"swid": "{SWID}"}}}),
    ]
    calls = []

    class Response:
        def __init__(self, status_code, body):
            self.status_code = status_code
            self._body = body

        def json(self):
            return self._body

    class Client:
        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def post(self, path, **kwargs):
            calls.append(path)
            return Response(*responses.pop(0))

    sleeps = []
    fake_httpx = types.SimpleNamespace(Client=Client, HTTPError=OSError)
    monkeypatch.setitem(__import__("sys").modules, "httpx", fake_httpx)
    monkeypatch.setattr(time, "sleep", sleeps.append)

    creds = L.OneIdOtpBroker()._submit(
        {"conversation_id": "conversation", "api_key": "key", "session_id": "session"},
        "123456",
    )

    assert creds == {"swid": "{SWID}", "espn_s2": "cookie"}
    assert calls == ["/otp/redeem", "/otp/redeem", "/guest/login/recoveryToken"]
    assert sleeps == [L._ONEID_REDEEM_RETRY_DELAY]


# ── HTTP endpoints (flask; skipped where unavailable) ─────────────────────────
@pytest.fixture
def client(monkeypatch):
    flask = pytest.importorskip("flask")
    monkeypatch.setenv("ESPN_OTP_LOGIN_ENABLED", "1")
    monkeypatch.setenv("ESPN_OTP_BROKER", "mock")
    L._reset_broker()
    from routes.link_bp import link_bp
    import dashboard_services.providers.espn_api as espn
    import dashboard_services.accounts as accounts
    seen = {}

    def fake_connect(season, league_id, swid=None, espn_s2=None):
        seen["swid"], seen["espn_s2"] = swid, espn_s2
        return {"name": "Test League"}

    monkeypatch.setattr(espn, "connect_league", fake_connect)
    monkeypatch.setattr(accounts, "add_espn_league_connection", lambda *a, **k: None)
    app = flask.Flask(__name__)
    app.secret_key = "test"
    app.register_blueprint(link_bp)
    with app.test_client() as test_client:
        with test_client.session_transaction() as sess:
            sess["account_id"] = 7
        yield test_client, seen
    L._reset_broker()


def test_endpoint_happy_path_connects_with_otp_method(client):
    test_client, seen = client
    start = test_client.post("/api/link/espn/otp/start",
                             json={"email": "t@example.com", "league_id": "123", "season": 2026})
    assert start.status_code == 200
    login_id = start.json["login_id"]
    done = test_client.post("/api/link/espn/otp/verify",
                            json={"login_id": login_id, "code": "123456", "league_id": "123", "season": 2026})
    assert done.status_code == 200
    assert done.json["connection_method"] == "otp"
    assert seen["swid"] == "{MOCK-SWID-0000-0000}"


def test_endpoint_wrong_code_is_rejected(client):
    test_client, _ = client
    login_id = test_client.post("/api/link/espn/otp/start",
                                json={"email": "t@example.com", "league_id": "123", "season": 2026}).json["login_id"]
    bad = test_client.post("/api/link/espn/otp/verify",
                           json={"login_id": login_id, "code": "000000", "league_id": "123", "season": 2026})
    assert bad.status_code == 400


def test_endpoint_404_when_flag_off(client, monkeypatch):
    test_client, _ = client
    monkeypatch.setenv("ESPN_OTP_LOGIN_ENABLED", "off")  # explicit kill-switch
    r = test_client.post("/api/link/espn/otp/start",
                         json={"email": "t@example.com", "league_id": "123", "season": 2026})
    assert r.status_code == 404
