"""ESPN email + one-time-code sign-in.

Broker-level tests need no flask and run everywhere; the HTTP tests use the
mock broker behind the feature flag and skip where flask isn't installed.
"""
import time

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


def test_flag_is_off_by_default(monkeypatch):
    monkeypatch.delenv("ESPN_OTP_LOGIN_ENABLED", raising=False)
    assert L.otp_login_enabled() is False
    monkeypatch.setenv("ESPN_OTP_LOGIN_ENABLED", "true")
    assert L.otp_login_enabled() is True


def test_default_broker_is_stubbed_unavailable():
    with pytest.raises(L.EspnLoginUnavailable):
        L.PlaywrightEspnLoginBroker().start("t@example.com")


def test_oneid_broker_unavailable_without_api_key(monkeypatch):
    monkeypatch.delenv("ESPN_ONEID_API_KEY", raising=False)
    with pytest.raises(L.EspnLoginUnavailable):
        L.OneIdOtpBroker().start("t@example.com")


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
    monkeypatch.setenv("ESPN_OTP_LOGIN_ENABLED", "")
    r = test_client.post("/api/link/espn/otp/start",
                         json={"email": "t@example.com", "league_id": "123", "season": 2026})
    assert r.status_code == 404
