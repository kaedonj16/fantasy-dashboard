"""Account notification preferences — weekly_digest with legacy email_opt_out fallback."""
from __future__ import annotations

from unittest import mock

from utils.email_preferences import WEEKLY_DIGEST, is_enabled, set_enabled, unsubscribe_weekly_digest


def test_weekly_digest_defaults_on_when_not_opted_out():
    with mock.patch("utils.email_preferences.ensure_schema"), \
         mock.patch("utils.email_preferences._legacy_opt_out", return_value=False), \
         mock.patch("dashboard_services.db.get_conn") as gc:
        gc.side_effect = RuntimeError("no db")
        assert is_enabled(1, WEEKLY_DIGEST, email_opt_out=False) is True


def test_weekly_digest_legacy_opt_out_respected():
    with mock.patch("utils.email_preferences.ensure_schema"), \
         mock.patch("dashboard_services.db.get_conn") as gc:
        gc.side_effect = RuntimeError("no db")
        assert is_enabled(1, WEEKLY_DIGEST, email_opt_out=True) is False


def test_preference_row_overrides_legacy_opt_out():
    class _Conn:
        def execute(self, *a, **k):
            class R:
                def fetchone(self):
                    return {"enabled": True}
            return R()

    with mock.patch("utils.email_preferences.ensure_schema"), \
         mock.patch("dashboard_services.db.get_conn") as gc:
        ctx = mock.MagicMock()
        ctx.__enter__.return_value = _Conn()
        ctx.__exit__.return_value = False
        gc.return_value = ctx
        # email_opt_out True but explicit weekly_digest enabled row wins.
        assert is_enabled(9, WEEKLY_DIGEST, email_opt_out=True) is True


def test_unknown_types_default_off_without_row():
    class _Conn:
        def execute(self, *a, **k):
            class R:
                def fetchone(self):
                    return None
            return R()

    with mock.patch("utils.email_preferences.ensure_schema"), \
         mock.patch("dashboard_services.db.get_conn") as gc:
        ctx = mock.MagicMock()
        ctx.__enter__.return_value = _Conn()
        ctx.__exit__.return_value = False
        gc.return_value = ctx
        assert is_enabled(1, "product_updates", email_opt_out=False) is False


def test_unsubscribe_sets_weekly_digest_false():
    with mock.patch("utils.email_preferences.set_enabled", return_value=True) as se:
        assert unsubscribe_weekly_digest(5) is True
        se.assert_called_once_with(5, False, WEEKLY_DIGEST)
