"""Broadcast dedupe: one send per PERSON, not per subscription row.

Regression (2026-10-25): every changelog broadcast arrived five times
on the same phone, including the ScoreZone entry deployed AFTER the
#2803 startup-claim fix. That fix stopped worker-vs-worker resends;
these duplicates were inside ONE broadcast. _push_broadcast deduped
in SQL by owner_id, but owner_id is a per-platform spelling (Sleeper
user id, ESPN SWID with or without braces, Yahoo team id, ...), so
the same phone endpoint, stored once per league under several
spellings, survived the dedupe once per spelling and was sent one
copy per spelling.

Rows below mirror the real push_subscriptions shape (dict rows from
dashboard_services.db dict_row, newest first as the query orders
id DESC): one endpoint repeated under many owner spellings, a second
endpoint for the same account, and another manager's row.
"""
from __future__ import annotations

import sys
import types

import pytest

pytest.importorskip("flask")
pytest.importorskip("pandas")

from routes import push_bp


def _row(endpoint, owner_id="", account_key="", platform="sleeper",
         league_id="L1"):
    return {
        "endpoint": endpoint,
        "p256dh": "k-" + endpoint,
        "auth": "a-" + endpoint,
        "league_id": league_id,
        "platform": platform,
        "owner_id": owner_id,
        "account_key": account_key,
    }


# Newest first. One phone ("ep-phone") subscribed once per league under
# five different owner spellings, plus a second endpoint for the same
# account (installed PWA alongside the browser subscription).
KAEDON_ROWS = [
    _row("ep-phone", owner_id="988877766655544433", account_key="acct:7",
         league_id="L5"),
    _row("ep-phone", owner_id="{AAAA-1111-BBBB}", account_key="acct:7",
         platform="espn", league_id="L4"),
    _row("ep-phone", owner_id="AAAA-1111-BBBB", account_key="",
         platform="espn", league_id="L3"),
    _row("ep-phone", owner_id="4", account_key="", platform="yahoo",
         league_id="1307110"),
    _row("ep-phone", owner_id="7", account_key="", platform="sleeper",
         league_id="L2"),
    _row("ep-phone-pwa", owner_id="988877766655544433",
         account_key="acct:7", league_id="L1"),
]

OTHER_ROW = _row("ep-other", owner_id="111122223333444455",
                 account_key="acct:9", league_id="L1")


def test_same_endpoint_many_owner_spellings_sends_once():
    kept = push_bp._dedupe_broadcast_rows(KAEDON_ROWS + [OTHER_ROW])
    assert [r["endpoint"] for r in kept] == ["ep-phone", "ep-other"]


def test_two_endpoints_same_sleeper_owner_sends_once():
    rows = [
        _row("ep-a", owner_id="988877766655544433"),
        _row("ep-b", owner_id="988877766655544433"),
    ]
    kept = push_bp._dedupe_broadcast_rows(rows)
    assert [r["endpoint"] for r in kept] == ["ep-a"]


def test_transitive_link_via_skipped_row():
    # ep-b is the same person's second device, linked only through an
    # owner spelling that first appears on a skipped duplicate row.
    rows = [
        _row("ep-a", owner_id="", account_key="acct:7"),
        _row("ep-a", owner_id="owner-x", account_key=""),
        _row("ep-b", owner_id="owner-x", account_key=""),
    ]
    kept = push_bp._dedupe_broadcast_rows(rows)
    assert [r["endpoint"] for r in kept] == ["ep-a"]


def test_yahoo_team_number_is_not_a_person_key():
    # Two different managers, both stored under team id "4" in their
    # own Yahoo leagues: neither may swallow the other's broadcast.
    rows = [
        _row("ep-a", owner_id="4", platform="yahoo", league_id="111"),
        _row("ep-b", owner_id="4", platform="yahoo", league_id="222"),
    ]
    kept = push_bp._dedupe_broadcast_rows(rows)
    assert [r["endpoint"] for r in kept] == ["ep-a", "ep-b"]


def test_distinct_people_all_kept():
    rows = [
        _row("ep-a", owner_id="988877766655544433", account_key="acct:7"),
        _row("ep-b", owner_id="111122223333444455", account_key="acct:9"),
        _row("ep-c", owner_id="", account_key=""),
    ]
    kept = push_bp._dedupe_broadcast_rows(rows)
    assert [r["endpoint"] for r in kept] == ["ep-a", "ep-b", "ep-c"]


class _FakeResult:
    def __init__(self, rows):
        self._rows = rows

    def fetchall(self):
        return list(self._rows)


class _FakeConn:
    def __init__(self, rows):
        self._rows = rows
        self.sql = []

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def execute(self, sql, params=None):
        self.sql.append(sql)
        return _FakeResult(self._rows)

    def commit(self):
        pass


def test_push_broadcast_sends_once_per_person(monkeypatch):
    import dashboard_services.db as db_mod
    import utils.push_notifications as pn

    conn = _FakeConn(KAEDON_ROWS + [OTHER_ROW])
    monkeypatch.setattr(db_mod, "get_conn", lambda: conn)
    monkeypatch.setattr(push_bp, "_get_vapid_keys",
                        lambda: {"private": "priv", "public": "pub"})
    monkeypatch.setattr(push_bp, "_init_push_table", lambda: None)
    monkeypatch.setattr(pn, "_make_vapid", lambda priv: object())

    calls = []

    class _FakeWebPushException(Exception):
        response = None

    fake_mod = types.ModuleType("pywebpush")
    fake_mod.webpush = lambda subscription_info, **kw: calls.append(
        subscription_info["endpoint"])
    fake_mod.WebPushException = _FakeWebPushException
    monkeypatch.setitem(sys.modules, "pywebpush", fake_mod)

    result, status = push_bp._push_broadcast(
        title="ScoreZone", body="Defensive touchdowns...", url="/",
        tag="changelog-entry-test",
    )
    assert status == 200
    assert result["sent"] == 2
    assert calls == ["ep-phone", "ep-other"]
