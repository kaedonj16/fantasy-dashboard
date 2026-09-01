"""Weekly send loop: opt-out, dedupe, Brevo success/failure, no real sends."""
from __future__ import annotations

from unittest import mock

import utils.weekly_email as we
from utils.email_delivery import SendResult


def _recip(**kw):
    base = {
        "account_id": 1, "email": "user@example.com", "first_name": "Sam",
        "platform": "sleeper", "league_id": "L1", "season": 2026,
        "roster_id": "7", "email_opt_out": False,
    }
    base.update(kw)
    return base


class _Conn:
    def __init__(self, sent_week=None):
        self.sent_week = sent_week
        self.writes = []

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def execute(self, sql, params=None):
        self.writes.append((sql, params))
        class R:
            def __init__(self, week):
                self._week = week
            def fetchone(self):
                if self._week is None:
                    return None
                return {"value": self._week}
            def fetchall(self):
                return []
        return R(self.sent_week)

    def commit(self):
        pass


def test_opted_out_recipient_is_skipped(monkeypatch):
    monkeypatch.setenv("BREVO_API_KEY", "xkeysib-test")
    with mock.patch.object(we, "_recipients", return_value=[_recip()]), \
         mock.patch("utils.email_preferences.is_enabled", return_value=False), \
         mock.patch("utils.email_events.is_suppressed", return_value=False), \
         mock.patch("utils.email_delivery.send_email") as send, \
         mock.patch("utils.digest_context.DigestRunCache.load_shared", lambda self: None), \
         mock.patch("dashboard_services.db.get_conn", return_value=_Conn()):
        summary = we.send_weekly_digests()
    send.assert_not_called()
    assert summary["skipped_opted_out"] == 1
    assert summary["sent"] == 0


def test_weekly_dedupe_skips_already_sent(monkeypatch):
    monkeypatch.setenv("BREVO_API_KEY", "xkeysib-test")
    week = we._iso_week()
    with mock.patch.object(we, "_recipients", return_value=[_recip()]), \
         mock.patch("utils.email_preferences.is_enabled", return_value=True), \
         mock.patch("utils.email_events.is_suppressed", return_value=False), \
         mock.patch("utils.email_delivery.send_email") as send, \
         mock.patch("utils.digest_context.DigestRunCache.load_shared", lambda self: None), \
         mock.patch("dashboard_services.db.get_conn", return_value=_Conn(sent_week=week)):
        summary = we.send_weekly_digests()
    send.assert_not_called()
    assert summary["skipped_already_sent"] == 1


def test_successful_brevo_send_records_message_id(monkeypatch):
    monkeypatch.setenv("BREVO_API_KEY", "xkeysib-test")
    digest = {"subject": "Hello", "html": "<p>x {UNSUB}</p>", "tags": ["weekly-digest"]}
    recorded = {}

    def _record(**kw):
        recorded.update(kw)

    with mock.patch.object(we, "_recipients", return_value=[_recip()]), \
         mock.patch.object(we, "build_digest", return_value=digest), \
         mock.patch.object(we, "other_leagues_for_account", return_value=[]), \
         mock.patch.object(we, "multi_league_sections_html", return_value=""), \
         mock.patch("utils.email_preferences.is_enabled", return_value=True), \
         mock.patch("utils.email_events.is_suppressed", return_value=False), \
         mock.patch("utils.email_events.record_send", _record), \
         mock.patch("utils.email_delivery.send_email",
                    return_value=SendResult(ok=True, provider="brevo", message_id="mid-9")), \
         mock.patch("utils.digest_context.DigestRunCache.load_shared", lambda self: None), \
         mock.patch("dashboard_services.db.get_conn", return_value=_Conn()):
        summary = we.send_weekly_digests()
    assert summary["sent"] == 1
    assert summary["failed"] == 0
    assert recorded.get("provider_message_id") == "mid-9"
    assert recorded.get("status") == "sent"


def test_failed_brevo_does_not_mark_week_complete(monkeypatch):
    monkeypatch.setenv("BREVO_API_KEY", "xkeysib-test")
    digest = {"subject": "Hello", "html": "<p>x {UNSUB}</p>", "tags": ["weekly-digest"]}
    conn = _Conn()
    with mock.patch.object(we, "_recipients", return_value=[_recip()]), \
         mock.patch.object(we, "build_digest", return_value=digest), \
         mock.patch.object(we, "other_leagues_for_account", return_value=[]), \
         mock.patch.object(we, "multi_league_sections_html", return_value=""), \
         mock.patch("utils.email_preferences.is_enabled", return_value=True), \
         mock.patch("utils.email_events.is_suppressed", return_value=False), \
         mock.patch("utils.email_events.record_send"), \
         mock.patch("utils.email_delivery.send_email",
                    return_value=SendResult(ok=False, provider="brevo",
                                            error_category="provider", status_code=500)), \
         mock.patch("utils.digest_context.DigestRunCache.load_shared", lambda self: None), \
         mock.patch("dashboard_services.db.get_conn", return_value=conn):
        summary = we.send_weekly_digests()
    assert summary["failed"] == 1
    assert summary["sent"] == 0
    assert not any("weekly_email_sent" in str(s[0]) and "INSERT" in str(s[0]) for s in conn.writes)


def test_rate_limit_counted_and_not_marked_sent(monkeypatch):
    monkeypatch.setenv("BREVO_API_KEY", "xkeysib-test")
    digest = {"subject": "Hello", "html": "<p>x {UNSUB}</p>", "tags": ["weekly-digest"]}
    with mock.patch.object(we, "_recipients", return_value=[_recip()]), \
         mock.patch.object(we, "build_digest", return_value=digest), \
         mock.patch.object(we, "other_leagues_for_account", return_value=[]), \
         mock.patch.object(we, "multi_league_sections_html", return_value=""), \
         mock.patch("utils.email_preferences.is_enabled", return_value=True), \
         mock.patch("utils.email_events.is_suppressed", return_value=False), \
         mock.patch("utils.email_events.record_send"), \
         mock.patch("utils.email_delivery.sleep_briefly"), \
         mock.patch("utils.email_delivery.send_email",
                    return_value=SendResult(ok=False, provider="brevo",
                                            error_category="rate_limited", status_code=429)), \
         mock.patch("utils.digest_context.DigestRunCache.load_shared", lambda self: None), \
         mock.patch("dashboard_services.db.get_conn", return_value=_Conn()):
        summary = we.send_weekly_digests()
    assert summary["provider_rate_limited"] == 1
    assert summary["sent"] == 0


def test_dry_run_never_sends(monkeypatch):
    monkeypatch.setenv("BREVO_API_KEY", "xkeysib-test")
    digest = {"subject": "Hello", "html": "<p>x {UNSUB}</p>", "tags": ["weekly-digest"]}
    with mock.patch.object(we, "_recipients", return_value=[_recip()]), \
         mock.patch.object(we, "build_digest", return_value=digest), \
         mock.patch.object(we, "other_leagues_for_account", return_value=[]), \
         mock.patch.object(we, "multi_league_sections_html", return_value=""), \
         mock.patch("utils.email_preferences.is_enabled", return_value=True), \
         mock.patch("utils.email_events.is_suppressed", return_value=False), \
         mock.patch("utils.email_delivery.send_email") as send, \
         mock.patch("utils.digest_context.DigestRunCache.load_shared", lambda self: None), \
         mock.patch("dashboard_services.db.get_conn", return_value=_Conn()):
        summary = we.send_weekly_digests(dry_run=True)
    send.assert_not_called()
    assert summary["dry_run"] is True
    assert summary["sent"] == 1


def test_send_by_email_scopes_to_one_account(monkeypatch):
    monkeypatch.setenv("BREVO_API_KEY", "xkeysib-test")
    recip = _recip(account_id=9, email="me@example.com")
    digest = {"subject": "Hello", "html": "<p>x {UNSUB}</p>", "tags": ["weekly-digest"]}
    with mock.patch.object(we, "_recipient_by_email", return_value=[recip]), \
         mock.patch.object(we, "_recipients", return_value=[]), \
         mock.patch.object(we, "_recipient_by_id", return_value=[recip]), \
         mock.patch.object(we, "build_digest", return_value=digest), \
         mock.patch.object(we, "other_leagues_for_account", return_value=[]), \
         mock.patch.object(we, "multi_league_sections_html", return_value=""), \
         mock.patch("utils.email_preferences.is_enabled", return_value=True), \
         mock.patch("utils.email_events.is_suppressed", return_value=False), \
         mock.patch("utils.email_events.record_send"), \
         mock.patch("utils.email_delivery.send_email",
                    return_value=SendResult(ok=True, provider="brevo", message_id="mid-me")) as send, \
         mock.patch("utils.digest_context.DigestRunCache.load_shared", lambda self: None), \
         mock.patch("dashboard_services.db.get_conn", return_value=_Conn()):
        summary = we.send_weekly_digests(email="me@example.com", force=True)
    send.assert_called_once()
    assert send.call_args[0][0] == "me@example.com"
    assert summary["sent"] == 1
    assert summary["eligible"] == 1


def test_unknown_email_does_not_mail_everyone(monkeypatch):
    monkeypatch.setenv("BREVO_API_KEY", "xkeysib-test")
    with mock.patch.object(we, "_recipient_by_email", return_value=[]), \
         mock.patch.object(we, "_recipients", return_value=[_recip(), _recip(account_id=2, email="b@x.com")]), \
         mock.patch("utils.email_delivery.send_email") as send, \
         mock.patch("utils.digest_context.DigestRunCache.load_shared", lambda self: None), \
         mock.patch("dashboard_services.db.get_conn", return_value=_Conn()):
        summary = we.send_weekly_digests(email="nobody@example.com")
    send.assert_not_called()
    assert summary["eligible"] == 0
    assert summary["sent"] == 0
    monkeypatch.setenv("BREVO_API_KEY", "xkeysib-test")
    week = we._iso_week()
    with mock.patch.object(we, "_recipients", return_value=[_recip()]), \
         mock.patch("utils.email_preferences.is_enabled", return_value=True), \
         mock.patch("utils.email_events.is_suppressed", return_value=False), \
         mock.patch("utils.email_delivery.send_email") as send, \
         mock.patch("utils.digest_context.DigestRunCache.load_shared", lambda self: None), \
         mock.patch("dashboard_services.db.get_conn", return_value=_Conn(sent_week=week)):
        summary = we.send_weekly_digests(force=True)
    send.assert_not_called()
    assert summary["skipped_already_sent"] == 1


def test_suppressed_address_skipped(monkeypatch):
    monkeypatch.setenv("BREVO_API_KEY", "xkeysib-test")
    with mock.patch.object(we, "_recipients", return_value=[_recip()]), \
         mock.patch("utils.email_preferences.is_enabled", return_value=True), \
         mock.patch("utils.email_events.is_suppressed", return_value=True), \
         mock.patch("utils.email_delivery.send_email") as send, \
         mock.patch("utils.digest_context.DigestRunCache.load_shared", lambda self: None), \
         mock.patch("dashboard_services.db.get_conn", return_value=_Conn()):
        summary = we.send_weekly_digests()
    send.assert_not_called()
    assert summary["skipped_suppressed"] == 1
