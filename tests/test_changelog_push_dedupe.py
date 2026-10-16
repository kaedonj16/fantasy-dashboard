"""Startup changelog push dedupe: content key, not calendar date.

Regression (diagnosed 2026-10-15): _notify_changelog_on_startup deduped
on CHANGELOG[0].date via the changelog_last_pushed marker, so only the
first entry deployed on a calendar date could ever push -- the
2026-10-15 BR Assistant entry never pushed because a same-day fix
entry burned the date first -- and the marker was written even when
the broadcast errored and sent 0, silencing the entry forever.

The store (app_state via dashboard_services.db.get_conn) and the
broadcast (routes.push_bp._push_broadcast) are faked; CHANGELOG is
swapped per test. The function reads all three lazily, so patching
the source modules is enough.
"""
from __future__ import annotations

import logging

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
    def __init__(self, row):
        self._row = row

    def fetchone(self):
        return self._row


class _FakeConn:
    """Just the app_state slice of the DB the notifier touches."""

    def __init__(self, store):
        self.store = store

    def execute(self, sql, params=None):
        if "SELECT value FROM app_state" in sql:
            value = self.store.get("changelog_last_pushed")
            return _FakeResult({"value": value} if value is not None else None)
        if "INSERT INTO app_state" in sql:
            self.store["changelog_last_pushed"] = params[0]
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
