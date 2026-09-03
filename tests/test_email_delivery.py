"""Provider-independent email sender (Brevo primary, SMTP fallback)."""
from __future__ import annotations

import json
from unittest import mock
from urllib.error import HTTPError

from utils.email_delivery import (
    active_provider,
    html_to_text,
    is_configured,
    send_email,
)


class _FP:
    def __init__(self, data: bytes):
        self._data = data

    def read(self, *a, **k):
        return self._data

    def close(self):
        return None


def test_not_configured(monkeypatch):
    monkeypatch.delenv("BREVO_API_KEY", raising=False)
    monkeypatch.delenv("EMAIL_USER", raising=False)
    monkeypatch.delenv("EMAIL_PASSWORD", raising=False)
    assert is_configured() is False
    result = send_email("a@b.com", "Hi", "<p>x</p>")
    assert result.ok is False
    assert result.error_category == "not_configured"


def test_brevo_success_captures_message_id(monkeypatch):
    monkeypatch.setenv("BREVO_API_KEY", "xkeysib-test-not-real")
    monkeypatch.setenv("BREVO_SENDER_EMAIL", "noreply@example.com")

    class _Resp:
        status = 201

        def read(self):
            return b'{"messageId": "abc-123"}'

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    captured = {}

    def fake_urlopen(req, timeout=15):
        captured["url"] = req.full_url
        captured["headers"] = dict(req.header_items())
        captured["body"] = json.loads(req.data.decode())
        return _Resp()

    with mock.patch("urllib.request.urlopen", fake_urlopen):
        result = send_email(
            "user@example.com", "Subj", "<p>Hello</p>",
            unsubscribe_url="https://ex/unsub", tags=["weekly-digest", "dynasty"],
        )
    assert result.ok is True
    assert result.provider == "brevo"
    assert result.message_id == "abc-123"
    assert captured["url"].endswith("/v3/smtp/email")
    header_names = {k.lower() for k in captured["headers"]}
    assert "api-key" in header_names
    body = captured["body"]
    assert body["htmlContent"] == "<p>Hello</p>"
    assert body["subject"] == "Subj"
    assert "List-Unsubscribe" in body["headers"]
    assert "weekly-digest" in body["tags"]


def test_brevo_http_error(monkeypatch):
    monkeypatch.setenv("BREVO_API_KEY", "xkeysib-test-not-real")

    def fake_urlopen(req, timeout=15):
        raise HTTPError(
            "https://api.brevo.com/v3/smtp/email", 400, "Bad",
            hdrs=None, fp=_FP(b'{"message":"invalid"}'),
        )

    with mock.patch("urllib.request.urlopen", fake_urlopen):
        result = send_email("user@example.com", "S", "<p>x</p>")
    assert result.ok is False
    assert result.provider == "brevo"
    assert result.status_code == 400
    assert result.error_category == "provider"


def test_brevo_rate_limited(monkeypatch):
    monkeypatch.setenv("BREVO_API_KEY", "xkeysib-test-not-real")

    def fake_urlopen(req, timeout=15):
        raise HTTPError(
            "https://api.brevo.com/v3/smtp/email", 429, "Too Many",
            hdrs=None, fp=_FP(b'{"message":"rate"}'),
        )

    with mock.patch("urllib.request.urlopen", fake_urlopen):
        result = send_email("user@example.com", "S", "<p>x</p>")
    assert result.ok is False
    assert result.error_category == "rate_limited"
    assert result.status_code == 429


def test_invalid_recipient_short_circuits(monkeypatch):
    monkeypatch.setenv("BREVO_API_KEY", "xkeysib-test-not-real")
    with mock.patch("urllib.request.urlopen") as urlopen:
        result = send_email("not-an-email", "S", "<p>x</p>")
    urlopen.assert_not_called()
    assert result.error_category == "invalid_recipient"


def test_html_to_text_strips_tags():
    assert "Hello" in html_to_text("<p>Hello<br>world</p>")


def test_active_provider_prefers_brevo(monkeypatch):
    monkeypatch.setenv("BREVO_API_KEY", "xkeysib-test")
    monkeypatch.setenv("EMAIL_USER", "smtp@example.com")
    monkeypatch.setenv("EMAIL_PASSWORD", "pw")
    assert active_provider() == "brevo"
