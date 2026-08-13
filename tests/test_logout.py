import pytest

flask = pytest.importorskip("flask")
Flask = flask.Flask

from routes.auth_bp import auth_bp


def _logout_app(cookie_domain=None):
    app = Flask(__name__)
    app.secret_key = "test-secret"
    app.config.update(
        SESSION_COOKIE_DOMAIN=cookie_domain,
        SESSION_COOKIE_SECURE=True,
        SESSION_COOKIE_HTTPONLY=True,
        SESSION_COOKIE_SAMESITE="Lax",
    )
    app.register_blueprint(auth_bp)
    return app


def test_logout_expires_domain_and_legacy_host_only_session_cookies():
    app = _logout_app("example.com")

    with app.test_client() as client:
        # Match the configured cookie domain while creating the session. Newer
        # Werkzeug clients correctly reject an example.com cookie created from
        # their default localhost origin.
        with client.session_transaction(base_url="https://example.com") as signed_in:
            signed_in["account_id"] = 42
            signed_in["account_email"] = "user@example.com"

        response = client.get("/logout", base_url="https://example.com")
        cookies = response.headers.getlist("Set-Cookie")

        assert response.status_code == 200
        assert any("Domain=example.com" in cookie and "Max-Age=0" in cookie for cookie in cookies)
        assert any("Domain=" not in cookie and "Max-Age=0" in cookie for cookie in cookies)

        with client.session_transaction(base_url="https://example.com") as signed_out:
            assert "account_id" not in signed_out
            assert "account_email" not in signed_out


def test_logout_does_not_emit_redundant_cookie_without_shared_domain():
    app = _logout_app()

    with app.test_client() as client:
        with client.session_transaction() as signed_in:
            signed_in["account_id"] = 42

        response = client.get("/logout", base_url="https://localhost")
        cookies = response.headers.getlist("Set-Cookie")

        assert len(cookies) == 1
        assert "Domain=" not in cookies[0]
        assert "Max-Age=0" in cookies[0]
