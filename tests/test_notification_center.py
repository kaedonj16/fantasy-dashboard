"""Notification center: inbox history, quiet hours, account-level prefs,
and changelog broadcast gating.

The bell inbox (user_notifications) is written at the shared send choke
points: immediate sends record at send time, digest/quiet-hours holds
record when the flush delivers them. Quiet hours hold non-live types
(injury included, and ONLY there) until the window ends; live types are
never held. Prefs are per-account: a PUT mirrors the full prefs dict to
every device row of the account. Changelog broadcasts honor the
"changelog" prefs key.
"""
from __future__ import annotations

import json
import sys
import types
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pytest

pytest.importorskip("flask")
pytest.importorskip("pandas")

import utils.push_notifications as pn
from routes import push_bp

ET = ZoneInfo("America/New_York")


# ── Fake DB serving both push tables ─────────────────────────────────────────

class _Cursor:
    def __init__(self, rows):
        self._rows = rows

    def fetchall(self):
        return list(self._rows)

    def fetchone(self):
        return self._rows[0] if self._rows else None


_DIGEST_COLS = ("endpoint", "p256dh", "auth", "account_key", "league_id",
                "platform", "notif_type", "title", "body", "url", "tag",
                "hour_bucket", "hold_until")
_NOTIF_COLS = ("account_key", "notif_type", "title", "body", "url",
               "league_id", "league_name")


class _FakeConn:
    def __init__(self, digest_store, notif_store):
        self.digest = digest_store
        self.notifs = notif_store
        self._next_digest = (max((r["id"] for r in digest_store), default=0) + 1)
        self._next_notif = (max((r["id"] for r in notif_store), default=0) + 1)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def commit(self):
        pass

    def execute(self, sql, params=None):
        up = " ".join(sql.split()).upper()
        params = params or ()
        if up.startswith("CREATE") or up.startswith("ALTER"):
            return _Cursor([])
        if up.startswith("INSERT INTO PUSH_DIGEST_ITEMS"):
            row = {"id": self._next_digest}
            self._next_digest += 1
            for col, val in zip(_DIGEST_COLS, params):
                row[col] = val
            self.digest.append(row)
            return _Cursor([])
        if up.startswith("DELETE FROM PUSH_DIGEST_ITEMS WHERE ENDPOINT"):
            ep = params[0]
            keep = {r["id"] for r in sorted(
                (r for r in self.digest if r["endpoint"] == ep),
                key=lambda r: r["id"], reverse=True)[:49]}
            self.digest[:] = [r for r in self.digest
                              if r["endpoint"] != ep or r["id"] in keep]
            return _Cursor([])
        if "FROM PUSH_DIGEST_ITEMS ORDER BY ID" in up:
            return _Cursor([dict(r) for r in
                            sorted(self.digest, key=lambda r: r["id"])])
        if up.startswith("DELETE FROM PUSH_DIGEST_ITEMS WHERE ID = ANY"):
            ids = set(params[0] or [])
            self.digest[:] = [r for r in self.digest if r["id"] not in ids]
            return _Cursor([])
        if up.startswith("INSERT INTO USER_NOTIFICATIONS"):
            row = {"id": self._next_notif, "created_at": datetime.now(timezone.utc),
                   "read_at": None}
            self._next_notif += 1
            for col, val in zip(_NOTIF_COLS, params):
                row[col] = val
            self.notifs.append(row)
            return _Cursor([])
        if up.startswith("DELETE FROM USER_NOTIFICATIONS"):
            acct, _acct2, keep_n = params
            mine = sorted((r for r in self.notifs if r["account_key"] == acct),
                          key=lambda r: r["id"], reverse=True)
            keep = {r["id"] for r in mine[:keep_n]}
            self.notifs[:] = [r for r in self.notifs
                              if r["account_key"] != acct or r["id"] in keep]
            return _Cursor([])
        if "SELECT COUNT(*) AS N FROM USER_NOTIFICATIONS" in up:
            n = sum(1 for r in self.notifs
                    if r["account_key"] == params[0] and r["read_at"] is None)
            return _Cursor([{"n": n}])
        if "FROM USER_NOTIFICATIONS" in up and up.startswith("SELECT ID"):
            acct, limit = params
            rows = [dict(r) for r in self.notifs if r["account_key"] == acct]
            rows.sort(key=lambda r: r["id"], reverse=True)
            return _Cursor(rows[:limit])
        if up.startswith("UPDATE USER_NOTIFICATIONS SET READ_AT"):
            if "WHERE ID" in up:
                nid, acct = params
                for r in self.notifs:
                    if (r["id"] == nid and r["account_key"] == acct
                            and r["read_at"] is None):
                        r["read_at"] = datetime.now(timezone.utc)
            else:
                acct = params[0]
                for r in self.notifs:
                    if r["account_key"] == acct and r["read_at"] is None:
                        r["read_at"] = datetime.now(timezone.utc)
            return _Cursor([])
        raise AssertionError("unexpected SQL: %s" % sql[:90])


@pytest.fixture()
def stores(monkeypatch):
    import dashboard_services.db as db

    digest_store: list = []
    notif_store: list = []
    monkeypatch.setattr(
        db, "get_conn", lambda: _FakeConn(digest_store, notif_store))
    monkeypatch.setattr(pn, "_DIGEST_TABLE_INIT", False)
    monkeypatch.setattr(pn, "_NOTIF_TABLE_INIT", False)
    monkeypatch.setattr(pn, "_league_display_name",
                        lambda plat, lid, season: "Blackedraw")
    import dashboard_services.api as api
    monkeypatch.setattr(api, "get_nfl_state",
                        lambda: {"season": "2026", "week": 8})
    return digest_store, notif_store


@pytest.fixture()
def fake_sends(monkeypatch):
    sent = []

    def _send(endpoints, title, body, url="/", tag="update"):
        for ep, _k1, _k2 in endpoints:
            sent.append({"endpoint": ep, "title": title, "body": body,
                         "url": url, "tag": tag})
        return len(endpoints)

    monkeypatch.setattr(pn, "_send_to_endpoints", _send)
    return sent


def _row(endpoint="ep1", prefs=None, account_key="42"):
    return {"endpoint": endpoint, "p256dh": "k", "auth": "a",
            "prefs": json.dumps(prefs or {}), "account_key": account_key,
            "platform": "sleeper"}


# ── Quiet-hours window math ──────────────────────────────────────────────────

def test_quiet_window_default_wraps_midnight():
    prefs = {"quiet_hours": True}
    assert pn._in_quiet_hours(prefs, datetime(2026, 10, 27, 23, 30, tzinfo=ET))
    assert pn._in_quiet_hours(prefs, datetime(2026, 10, 28, 2, 0, tzinfo=ET))
    assert not pn._in_quiet_hours(prefs, datetime(2026, 10, 27, 12, 0, tzinfo=ET))
    # Window end is exclusive: 7:00 AM sharp is out.
    assert not pn._in_quiet_hours(prefs, datetime(2026, 10, 28, 7, 0, tzinfo=ET))


def test_quiet_window_off_and_custom():
    assert not pn._in_quiet_hours({}, datetime(2026, 10, 28, 2, 0, tzinfo=ET))
    prefs = {"quiet_hours": True, "quiet_start": "22:00", "quiet_end": "06:30"}
    assert pn._in_quiet_hours(prefs, datetime(2026, 10, 27, 22, 15, tzinfo=ET))
    assert not pn._in_quiet_hours(prefs, datetime(2026, 10, 28, 6, 45, tzinfo=ET))
    end = pn._quiet_hold_until(prefs, datetime(2026, 10, 27, 23, 0, tzinfo=ET))
    assert (end.hour, end.minute, end.day) == (6, 30, 28)


def test_quiet_hold_until_from_evening_leg():
    prefs = {"quiet_hours": True}
    end = pn._quiet_hold_until(prefs, datetime(2026, 10, 27, 23, 45, tzinfo=ET))
    assert (end.hour, end.minute, end.day) == (7, 0, 28)
    end = pn._quiet_hold_until(prefs, datetime(2026, 10, 28, 3, 0, tzinfo=ET))
    assert (end.hour, end.minute, end.day) == (7, 0, 28)


# ── Send path: holds, live exemption, recording ──────────────────────────────

def _force_quiet(monkeypatch, on=True):
    monkeypatch.setattr(pn, "_in_quiet_hours", lambda prefs, now=None: on)
    monkeypatch.setattr(
        pn, "_quiet_hold_until",
        lambda prefs, now=None: datetime(2026, 10, 28, 7, 0, tzinfo=ET))


def test_quiet_hours_hold_buffers_and_records_nothing_yet(
        stores, fake_sends, monkeypatch):
    digest_store, notif_store = stores
    _force_quiet(monkeypatch)
    sent = pn._send_with_digest(
        [_row(prefs={"quiet_hours": True})],
        "Injury update", "A starter is questionable.", "/x", "t",
        notif_type="injury", league_id="L1", platform="sleeper")
    assert sent == 0 and fake_sends == []
    assert len(digest_store) == 1
    assert digest_store[0]["notif_type"] == "injury"
    assert digest_store[0]["hold_until"] is not None
    # Held alerts enter the inbox only when delivered.
    assert notif_store == []


def test_live_types_never_held_by_quiet_hours(stores, fake_sends, monkeypatch):
    _digest_store, notif_store = stores
    _force_quiet(monkeypatch)
    sent = pn._send_with_digest(
        [_row(prefs={"quiet_hours": True})],
        "Touchdown", "Your RB scored.", "/x", "t",
        notif_type="redzone_scores", league_id="L1", platform="sleeper")
    assert sent == 1 and len(fake_sends) == 1
    # Immediate sends are recorded in the inbox at send time.
    assert len(notif_store) == 1
    assert notif_store[0]["notif_type"] == "redzone_scores"
    assert notif_store[0]["account_key"] == "42"
    assert notif_store[0]["league_name"] == "Blackedraw"


def test_injury_sends_immediately_when_quiet_hours_off(
        stores, fake_sends, monkeypatch):
    digest_store, notif_store = stores
    _force_quiet(monkeypatch, on=False)
    # Even a digest-mode device gets injury immediately: injury is
    # deliberately not digest-eligible.
    sent = pn._send_with_digest(
        [_row(prefs={"digest": True})],
        "Injury update", "A starter is questionable.", "/x", "t",
        notif_type="injury", league_id="L1", platform="sleeper")
    assert sent == 1
    assert digest_store == []
    assert len(notif_store) == 1


def test_type_opt_out_still_wins_during_quiet_hours(
        stores, fake_sends, monkeypatch):
    digest_store, _notif_store = stores
    _force_quiet(monkeypatch)
    sent = pn._send_with_digest(
        [_row(prefs={"quiet_hours": True, "injury": False})],
        "Injury update", "Body.", "/x", "t",
        notif_type="injury", league_id="L1", platform="sleeper")
    assert sent == 0 and digest_store == [] and fake_sends == []


# ── Flush: hold release, legacy gate, delivery recording ─────────────────────

def _digest_row(**kw):
    row = {"id": 1, "endpoint": "ep1", "p256dh": "k", "auth": "a",
           "account_key": "42", "league_id": "L1", "platform": "sleeper",
           "notif_type": "injury", "title": "Injury update",
           "body": "A starter is questionable.", "url": "/x", "tag": "t",
           "hour_bucket": "2026102803", "hold_until": None}
    row.update(kw)
    return row


def test_flush_releases_expired_quiet_hold_before_9am(stores, fake_sends):
    digest_store, notif_store = stores
    digest_store.append(_digest_row(
        hold_until=datetime(2026, 10, 28, 7, 0, tzinfo=ET)))
    sent = pn._flush_digest(now=datetime(2026, 10, 28, 8, 0, tzinfo=ET))
    assert sent == 1
    assert digest_store == []
    assert len(notif_store) == 1
    assert notif_store[0]["notif_type"] == "injury"


def test_flush_keeps_unexpired_hold(stores, fake_sends):
    digest_store, notif_store = stores
    digest_store.append(_digest_row(
        hold_until=datetime(2026, 10, 29, 7, 0, tzinfo=ET)))
    sent = pn._flush_digest(now=datetime(2026, 10, 28, 23, 30, tzinfo=ET))
    assert sent == 0
    assert len(digest_store) == 1
    assert notif_store == []


def test_flush_records_once_per_account_across_devices(stores, fake_sends):
    digest_store, notif_store = stores
    digest_store.append(_digest_row(id=1, endpoint="ep1"))
    digest_store.append(_digest_row(id=2, endpoint="ep2"))
    sent = pn._flush_digest(now=datetime(2026, 10, 28, 10, 0, tzinfo=ET))
    assert sent == 2  # one digest push per device
    assert len(notif_store) == 1  # but one inbox entry per account per alert


# ── Inbox store: list, read, mark-all, prune, scoping ────────────────────────

def test_history_list_read_and_mark_all(stores):
    _d, notif_store = stores
    pn.record_user_notification("42", "close_game", "T1", "B1", "/a")
    pn.record_user_notification("42", "injury", "T2", "B2", "/b")
    pn.record_user_notification("99", "injury", "TX", "BX", "/x")
    items, unread = pn.list_user_notifications("42")
    assert unread == 2
    assert [i["title"] for i in items] == ["T2", "T1"]  # newest first
    assert items[0]["read"] is False
    assert pn.mark_user_notification_read("42", items[0]["id"])
    items, unread = pn.list_user_notifications("42")
    assert unread == 1 and items[0]["read"] is True
    assert pn.mark_all_user_notifications_read("42")
    _items, unread = pn.list_user_notifications("42")
    assert unread == 0
    # The other account is untouched.
    _i99, unread99 = pn.list_user_notifications("99")
    assert unread99 == 1


def test_history_mark_read_is_account_scoped(stores):
    _d, _n = stores
    pn.record_user_notification("99", "injury", "TX", "BX", "/x")
    items99, _ = pn.list_user_notifications("99")
    pn.mark_user_notification_read("42", items99[0]["id"])
    _i, unread99 = pn.list_user_notifications("99")
    assert unread99 == 1


def test_history_prunes_to_latest_50(stores):
    _d, notif_store = stores
    for i in range(55):
        pn.record_user_notification("42", "watchlist", f"T{i}", "B", "/w")
    assert len(notif_store) == 50
    items, _unread = pn.list_user_notifications("42")
    assert items[0]["title"] == "T54"
    assert items[-1]["title"] == "T5"


def test_history_guest_account_records_nothing(stores):
    _d, notif_store = stores
    assert pn.record_user_notification("", "injury", "T", "B", "/x") is False
    assert notif_store == []
    assert pn.list_user_notifications("") == ([], 0)


# ── Inbox API routes ─────────────────────────────────────────────────────────

@pytest.fixture()
def client():
    from flask import Flask
    app = Flask(__name__)
    app.secret_key = "test"
    app.register_blueprint(push_bp.push_bp)
    return app.test_client()


def test_notifications_api_guest_gets_empty_and_writes_401(client, stores):
    resp = client.get("/api/notifications")
    assert resp.status_code == 200
    assert resp.get_json() == {"items": [], "unread": 0}
    assert client.post("/api/notifications/read", json={"id": 1}).status_code == 401
    assert client.post("/api/notifications/read-all", json={}).status_code == 401


def test_notifications_api_round_trip(client, stores):
    pn.record_user_notification("42", "injury", "T", "B", "/x")
    with client.session_transaction() as sess:
        sess["account_id"] = 42
    resp = client.get("/api/notifications")
    data = resp.get_json()
    assert data["unread"] == 1 and len(data["items"]) == 1
    nid = data["items"][0]["id"]
    assert client.post("/api/notifications/read",
                       json={"id": nid}).get_json()["ok"] is True
    assert client.get("/api/notifications").get_json()["unread"] == 0
    pn.record_user_notification("42", "injury", "T2", "B2", "/y")
    assert client.post("/api/notifications/read-all",
                       json={}).get_json()["ok"] is True
    assert client.get("/api/notifications").get_json()["unread"] == 0


# ── Preferences PUT: full per-account mirror ─────────────────────────────────

class _PrefsConn:
    def __init__(self, rows):
        self.rows = rows  # endpoint -> {"prefs": json str, "account_key": str}

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def commit(self):
        pass

    def execute(self, sql, params=None):
        up = " ".join(sql.split()).upper()
        params = params or ()
        if up.startswith("CREATE") or up.startswith("ALTER"):
            return _Cursor([])
        if up.startswith("UPDATE PUSH_SUBSCRIPTIONS SET PREFS"):
            if "ACCOUNT_KEY" in up:
                prefs_json, acct, other_ep = params
                for ep, row in self.rows.items():
                    if row["account_key"] == acct and ep != other_ep:
                        row["prefs"] = prefs_json
            else:
                prefs_json, ep = params
                self.rows[ep]["prefs"] = prefs_json
            return _Cursor([])
        if "SELECT ENDPOINT, PREFS, ACCOUNT_KEY FROM PUSH_SUBSCRIPTIONS" in up:
            ep = params[0]
            row = self.rows[ep]
            return _Cursor([{"endpoint": ep, "prefs": row["prefs"],
                             "account_key": row["account_key"]}])
        raise AssertionError("unexpected SQL: %s" % sql[:90])


def test_preferences_put_mirrors_full_prefs_to_account(client, monkeypatch):
    import dashboard_services.db as db
    rows = {
        "ep-phone": {"prefs": "{}", "account_key": "42"},
        "ep-pwa": {"prefs": "{}", "account_key": "42"},
        "ep-other": {"prefs": "{}", "account_key": "99"},
    }
    monkeypatch.setattr(db, "get_conn", lambda: _PrefsConn(rows))
    monkeypatch.setattr(push_bp, "_init_push_table", lambda: None)
    new_prefs = {"injury": False, "quiet_hours": True,
                 "quiet_start": "22:30", "digest": True}
    resp = client.put("/api/push/preferences",
                      json={"endpoint": "ep-phone", "prefs": new_prefs})
    assert resp.status_code == 200
    # Type toggles AND quiet hours land on the account's other device...
    assert json.loads(rows["ep-pwa"]["prefs"]) == new_prefs
    # ...and never on another account's device.
    assert json.loads(rows["ep-other"]["prefs"]) == {}


# ── Changelog broadcast gating + inbox fold-in ───────────────────────────────

def _broadcast_row(endpoint, account_key, prefs=None):
    return {"endpoint": endpoint, "p256dh": "k-" + endpoint,
            "auth": "a-" + endpoint, "league_id": "L1",
            "platform": "sleeper", "owner_id": "o-" + endpoint,
            "account_key": account_key,
            "prefs": json.dumps(prefs) if prefs is not None else None}


class _BroadcastConn:
    def __init__(self, rows):
        self._rows = rows

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def execute(self, sql, params=None):
        return _Cursor(list(self._rows))

    def commit(self):
        pass


def test_broadcast_skips_changelog_opt_out_and_records_inbox(monkeypatch):
    import dashboard_services.db as db_mod

    rows = [
        _broadcast_row("ep-optout", "42", prefs={"changelog": False}),
        _broadcast_row("ep-fan", "99"),
    ]
    conn = _BroadcastConn(rows)
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

    recorded = []
    monkeypatch.setattr(
        pn, "record_user_notification",
        lambda acct, typ, title, body, url, **kw: recorded.append(
            (acct, typ, title)))

    result, status = push_bp._push_broadcast(
        title="New: Notification center", body="Your alerts, in one place.",
        url="/portfolio", tag="changelog-test")
    assert status == 200 and result["sent"] == 1
    assert calls == ["ep-fan"]
    assert recorded == [("99", "changelog", "New: Notification center")]


# ── Delivery catalog shape ───────────────────────────────────────────────────

def test_delivery_groups_cover_every_settings_type_once():
    keys = [t["key"] for g in pn.PUSH_DELIVERY_GROUPS for t in g["types"]]
    assert len(keys) == len(set(keys)) == 17
    bucket_keys = {t["key"] for b in pn.PUSH_TYPE_BUCKETS for t in b["types"]}
    assert set(keys) == bucket_keys | {"changelog"}
    live = next(g for g in pn.PUSH_DELIVERY_GROUPS if g["id"] == "live")
    assert {t["key"] for t in live["types"]} == set(pn._QUIET_EXEMPT_TYPES)
    for g in pn.PUSH_DELIVERY_GROUPS:
        for t in g["types"]:
            assert t["label"] and t["desc"]
