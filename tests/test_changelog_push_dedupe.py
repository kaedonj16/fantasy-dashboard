"""Startup changelog push dedupe: content key, not calendar date.

Regression (diagnosed 2026-10-15): _notify_changelog_on_startup deduped
on CHANGELOG[0].date via the changelog_last_pushed marker, so only the
first entry deployed on a calendar date could ever push -- the
2026-10-15 BR Assistant entry never pushed because a same-day fix
entry burned the date first -- and the marker was written even when
the broadcast errored and sent 0, silencing the entry forever.

Second regression (2026-10-25): the settled marker is written only
after the slow broadcast, and the notifier runs at import time in
every gunicorn worker, so on the #2801 deploy every worker that
booted inside the send window passed the marker check and sent a
duplicate (five copies of the windows entry). The notifier now
claims the broadcast first via a changelog_push_claim app_state row
(INSERT ... ON CONFLICT DO NOTHING); these tests cover the claim
handshake too.

The store (app_state via dashboard_services.db.get_conn) and the
broadcast (routes.push_bp._push_broadcast) are faked; CHANGELOG is
swapped per test. The function reads all three lazily, so patching
the source modules is enough.
"""
from __future__ import annotations

import logging
import re
import time

import pytest

pytest.importorskip("flask")
pytest.importorskip("pandas")

ASSISTANT_ENTRY = {
    "date": "2026-10-15",
    "tag": "new",
    "text": "BR Assistant answers more of your questions: ask for a "
            "younger, older, or cheaper replacement for any player.",
    "link": "/",
}
FIX_ENTRY = {
    "date": "2026-10-15",  # same calendar date, different entry
    "tag": "fix",
    "text": "Fixed: the dashboard could ask signed-in managers to link "
            "their team again.",
    "link": "/",
}


class _FakeResult:
    def __init__(self, row, rowcount=0):
        self._row = row
        self.rowcount = rowcount

    def fetchone(self):
        return self._row


_KEY_EQ_RE = re.compile(r"key\s*=\s*'([^']+)'")
_VALUES_KEY_RE = re.compile(r"VALUES\s*\(\s*'([^']+)'")


class _FakeConn:
    """Just the app_state slice of the DB the notifier touches.

    The store is a plain dict keyed by app_state key, mirroring the
    real table: the app_state key is parsed out of each statement's
    SQL literal, INSERT ... ON CONFLICT DO NOTHING reports rowcount
    0 without overwriting when the key is already present, and
    DELETE only removes a row whose value matches the parameter
    (the notifier's conditional claim deletes).
    """

    def __init__(self, store):
        self.store = store

    def execute(self, sql, params=None):
        params = params or ()
        if "SELECT value FROM app_state" in sql:
            m = _KEY_EQ_RE.search(sql)
            key = m.group(1) if m else "changelog_last_pushed"
            value = self.store.get(key)
            return _FakeResult({"value": value} if value is not None else None)
        if "INSERT INTO app_state" in sql:
            m = _VALUES_KEY_RE.search(sql)
            key = m.group(1) if m else "changelog_last_pushed"
            if "DO NOTHING" in sql and key in self.store:
                return _FakeResult(None, rowcount=0)
            self.store[key] = params[0]
            return _FakeResult(None, rowcount=1)
        if "DELETE FROM app_state" in sql:
            m = _KEY_EQ_RE.search(sql)
            key = m.group(1) if m else None
            if key is not None and params and self.store.get(key) == params[0]:
                del self.store[key]
                return _FakeResult(None, rowcount=1)
            return _FakeResult(None, rowcount=0)
        return _FakeResult(None)

    def commit(self):
        pass


class _ConnCtx:
    def __init__(self, conn):
        self._conn = conn

    def __enter__(self):
        return self._conn

    def __exit__(self, *exc):
        return False


@pytest.fixture()
def notify(monkeypatch):
    import dashboard_services.changelog as changelog_mod
    import dashboard_services.db as db_mod
    import routes.push_bp as push_bp

    store: dict = {}
    conn = _FakeConn(store)
    monkeypatch.setattr(db_mod, "get_conn", lambda: _ConnCtx(conn))

    calls: list = []
    outcome = {"result": ({"ok": True, "sent": 3, "failed": 0}, 200)}

    def fake_broadcast(**kwargs):
        calls.append(kwargs)
        return outcome["result"]

    monkeypatch.setattr(push_bp, "_push_broadcast", fake_broadcast)
    monkeypatch.setattr(changelog_mod, "CHANGELOG", [dict(ASSISTANT_ENTRY)])

    import app

    # The import-time run above fired once into the fakes; reset so each
    # test starts from the marker state it sets up itself.
    calls.clear()
    store.clear()

    env = {
        "app": app,
        "store": store,
        "calls": calls,
        "outcome": outcome,
        "changelog_mod": changelog_mod,
        "monkeypatch": monkeypatch,
    }
    return env


def _set_changelog(env, entries):
    env["monkeypatch"].setattr(env["changelog_mod"], "CHANGELOG", entries)


def test_content_key_is_stable_and_content_sensitive(notify):
    key = notify["app"]._changelog_entry_key
    assert key(ASSISTANT_ENTRY) == key(dict(ASSISTANT_ENTRY))
    assert key(ASSISTANT_ENTRY) != key(FIX_ENTRY)  # same date, different text
    other_date = dict(ASSISTANT_ENTRY, date="2026-10-16")
    assert key(ASSISTANT_ENTRY) != key(other_date)
    assert key(ASSISTANT_ENTRY).startswith("entry-")


def test_first_boot_pushes_and_marks_with_content_key(notify):
    notify["app"]._notify_changelog_on_startup()
    assert len(notify["calls"]) == 1
    key = notify["app"]._changelog_entry_key(ASSISTANT_ENTRY)
    assert notify["store"]["changelog_last_pushed"] == key
    # The push tag carries the key, so same-day entries do not collapse
    # into one notification slot on the device.
    assert notify["calls"][0]["tag"] == f"changelog-{key}"


def test_second_boot_skips_and_logs(notify, caplog):
    app = notify["app"]
    app._notify_changelog_on_startup()
    with caplog.at_level(logging.INFO):
        app._notify_changelog_on_startup()
    assert len(notify["calls"]) == 1  # no second push
    assert any("[changelog-push]" in r.message and "skipping" in r.message
               for r in caplog.records)


def test_two_entries_sharing_a_date_each_push_once(notify):
    app = notify["app"]
    app._notify_changelog_on_startup()  # assistant entry pushes
    _set_changelog(notify, [dict(FIX_ENTRY)])
    app._notify_changelog_on_startup()  # same date, new entry: pushes too
    app._notify_changelog_on_startup()  # ... but only once
    assert len(notify["calls"]) == 2
    assert notify["store"]["changelog_last_pushed"] == \
        app._changelog_entry_key(FIX_ENTRY)


def test_legacy_bare_date_marker_does_not_suppress_current_entry(notify):
    # Production state after the bug: the marker holds the bare date a
    # same-day fix entry burned. The never-pushed assistant entry at
    # CHANGELOG[0] must still fire exactly once on the next boot.
    notify["store"]["changelog_last_pushed"] = "2026-10-15"
    app = notify["app"]
    app._notify_changelog_on_startup()
    app._notify_changelog_on_startup()
    assert len(notify["calls"]) == 1
    assert notify["store"]["changelog_last_pushed"] == \
        app._changelog_entry_key(ASSISTANT_ENTRY)


def test_edited_text_counts_as_a_new_entry(notify):
    app = notify["app"]
    app._notify_changelog_on_startup()
    edited = dict(ASSISTANT_ENTRY, text=ASSISTANT_ENTRY["text"] + " Really.")
    _set_changelog(notify, [edited])
    app._notify_changelog_on_startup()
    assert len(notify["calls"]) == 2


def test_failed_broadcast_does_not_burn_the_marker(notify, caplog):
    notify["outcome"]["result"] = ({"error": "Push not configured"}, 503)
    app = notify["app"]
    with caplog.at_level(logging.WARNING):
        app._notify_changelog_on_startup()
    assert len(notify["calls"]) == 1
    assert "changelog_last_pushed" not in notify["store"]  # retryable
    assert any("[changelog-push]" in r.message and "broadcast failed" in r.message
               for r in caplog.records)
    # Next boot, healthy broadcast: the entry fires and marks.
    notify["outcome"]["result"] = ({"ok": True, "sent": 5, "failed": 0}, 200)
    app._notify_changelog_on_startup()
    assert len(notify["calls"]) == 2
    assert notify["store"]["changelog_last_pushed"] == \
        app._changelog_entry_key(ASSISTANT_ENTRY)


def test_zero_subscriber_success_still_counts_as_fired(notify):
    notify["outcome"]["result"] = ({"ok": True, "sent": 0, "failed": 0}, 200)
    app = notify["app"]
    app._notify_changelog_on_startup()
    app._notify_changelog_on_startup()
    assert len(notify["calls"]) == 1
    assert notify["store"]["changelog_last_pushed"] == \
        app._changelog_entry_key(ASSISTANT_ENTRY)


def test_fresh_claim_by_another_worker_suppresses_broadcast(notify, caplog):
    # Another worker won the claim moments ago and is mid-broadcast:
    # this boot must not send a duplicate, and must not write the
    # settled marker (the claim holder does that when its send lands).
    app = notify["app"]
    key = app._changelog_entry_key(ASSISTANT_ENTRY)
    notify["store"]["changelog_push_claim"] = f"{key}|{int(time.time())}"
    with caplog.at_level(logging.INFO):
        app._notify_changelog_on_startup()
    assert notify["calls"] == []
    assert "changelog_last_pushed" not in notify["store"]
    assert any("[changelog-push]" in r.message and "claim held" in r.message
               for r in caplog.records)


def test_stale_claim_is_taken_over(notify):
    # The claim holder died mid-send: once the claim is older than
    # the TTL, the next boot re-claims, broadcasts once, settles the
    # marker, and leaves no claim row behind.
    app = notify["app"]
    key = app._changelog_entry_key(ASSISTANT_ENTRY)
    stale_ts = int(time.time()) - app._CHANGELOG_CLAIM_TTL_S - 1
    notify["store"]["changelog_push_claim"] = f"{key}|{stale_ts}"
    app._notify_changelog_on_startup()
    assert len(notify["calls"]) == 1
    assert notify["store"]["changelog_last_pushed"] == key
    assert "changelog_push_claim" not in notify["store"]


def test_broadcast_error_releases_the_claim(notify):
    # A failed broadcast releases the claim without settling, so the
    # next boot retries at once instead of waiting out the claim TTL.
    notify["outcome"]["result"] = ({"error": "Push not configured"}, 503)
    app = notify["app"]
    app._notify_changelog_on_startup()
    assert len(notify["calls"]) == 1
    assert "changelog_last_pushed" not in notify["store"]
    assert "changelog_push_claim" not in notify["store"]
    notify["outcome"]["result"] = ({"ok": True, "sent": 2, "failed": 0}, 200)
    app._notify_changelog_on_startup()
    assert len(notify["calls"]) == 2  # two attempts across the two boots
    assert notify["store"]["changelog_last_pushed"] == \
        app._changelog_entry_key(ASSISTANT_ENTRY)
    assert "changelog_push_claim" not in notify["store"]


def test_claim_for_a_different_entry_does_not_block(notify):
    # A leftover claim for some other entry (its deploy was replaced
    # before it ever settled) is claimable: delete, claim, broadcast.
    app = notify["app"]
    other_key = app._changelog_entry_key(FIX_ENTRY)
    notify["store"]["changelog_push_claim"] = \
        f"{other_key}|{int(time.time())}"
    app._notify_changelog_on_startup()
    assert len(notify["calls"]) == 1
    assert notify["store"]["changelog_last_pushed"] == \
        app._changelog_entry_key(ASSISTANT_ENTRY)
    assert "changelog_push_claim" not in notify["store"]
