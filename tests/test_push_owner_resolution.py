"""Owner-id resolution for push delivery.

Two id-namespace bridges that previously dropped notifications silently:

1. ESPN roster owner ids are SWIDs that may be stored with or without
   ``{braces}``. ``_broadcast_owner`` matched the id exactly, so a device that
   persisted the other spelling was never found.
2. Watchlists key off the account key (``acct:<id>`` for a Google account) but
   push subscriptions key off the platform owner id, so a Google account's
   devices were never matched for watchlist alerts.

These are pure-Python (heavy deps are imported lazily), so no Flask needed.
"""
from unittest import mock

import utils.push_notifications as pn


def test_owner_ids_for_bare_sleeper_key():
    assert pn._subscription_owner_ids_for_user_key("123456") == ["123456"]


def test_owner_ids_for_espn_swid_includes_brace_variants():
    got = set(pn._subscription_owner_ids_for_user_key("{ABC-DEF}"))
    assert got == {"{ABC-DEF}", "ABC-DEF"}


def test_owner_ids_for_account_key_expands_platform_identities():
    with mock.patch(
        "dashboard_services.accounts.list_all_account_platform_ids",
        return_value=["999", "{S-W-1}"],
    ):
        got = set(pn._subscription_owner_ids_for_user_key("acct:42"))
    # The account key itself, the legacy bare account id, and every linked
    # platform identity (with brace variants) are all candidate owner ids.
    assert {"acct:42", "42", "999", "{S-W-1}", "S-W-1"} <= got


def test_owner_ids_for_empty_key():
    assert pn._subscription_owner_ids_for_user_key("") == []
    assert pn._subscription_owner_ids_for_user_key(None) == []


def test_broadcast_owner_matches_espn_swid_across_brace_spellings():
    """A roster owner_id of ``{SWID}`` must reach a device that stored the SWID
    without braces (and vice versa)."""
    captured = {}

    class _R:
        def __init__(self, rows):
            self._rows = rows

        def fetchall(self):
            return self._rows

    class _FakeConn:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def execute(self, q, params=None):
            captured["params"] = params
            _league, variants = params
            stored = "ABC-DEF"  # device persisted the unbraced spelling
            rows = (
                [{"endpoint": "e", "p256dh": "k", "auth": "a", "prefs": None}]
                if stored in variants
                else []
            )
            return _R(rows)

        def commit(self):
            pass

    with mock.patch("dashboard_services.db.get_conn", return_value=_FakeConn()), \
         mock.patch.object(pn, "_send_to_endpoints",
                           side_effect=lambda eps, *a, **k: len(eps)):
        n = pn._broadcast_owner("L1", "{ABC-DEF}", "Title", "Body")

    assert set(captured["params"][1]) == {"{ABC-DEF}", "ABC-DEF"}
    assert n == 1


def test_broadcast_owner_falls_back_to_league_when_no_owner():
    with mock.patch.object(pn, "_broadcast_league", return_value=7) as bl:
        assert pn._broadcast_owner("L1", "", "T", "B") == 7
    bl.assert_called_once()
