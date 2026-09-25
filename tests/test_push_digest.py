"""Push preferences picker, digest mode, and re-prompt persistence.

Digest mode: per-account, per-hour batching of eligible notification types into
one combined push per device. Existing subscribers with no stored preferences
must behave exactly as before (immediate sends).
"""
import json
import os

import pytest

import utils.push_notifications as pn


# ── Fake digest-table DB ──────────────────────────────────────────────────────

class _FakeCursor:
    def __init__(self, rows):
        self._rows = rows

    def fetchall(self):
        return self._rows

    def fetchone(self):
        return self._rows[0] if self._rows else None


class _FakeDigestConn:
    """Minimal stand-in for the push_digest_items table (list of dicts)."""

    COLUMNS = ("endpoint", "p256dh", "auth", "account_key", "league_id",
               "platform", "notif_type", "title", "body", "url", "tag",
               "hour_bucket")

    def __init__(self, store):
        self.store = store
        self._next_id = (max((r["id"] for r in store), default=0) + 1)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def execute(self, sql, params=None):
        up = " ".join(sql.split()).upper()
        if up.startswith("CREATE") or up.startswith("ALTER"):
            return _FakeCursor([])
        if up.startswith("INSERT INTO PUSH_DIGEST_ITEMS"):
            row = {"id": self._next_id}
            self._next_id += 1
            for col, val in zip(self.COLUMNS, params or ()):
                row[col] = val
            self.store.append(row)
            return _FakeCursor([])
        if "FROM PUSH_DIGEST_ITEMS ORDER BY ID" in up:
            rows = sorted(self.store, key=lambda r: r["id"])
            return _FakeCursor([dict(r) for r in rows])
        if up.startswith("DELETE FROM PUSH_DIGEST_ITEMS WHERE ID = ANY"):
            ids = set(params[0] or [])
            self.store[:] = [r for r in self.store if r["id"] not in ids]
            return _FakeCursor([])
        if up.startswith("DELETE FROM PUSH_DIGEST_ITEMS WHERE ENDPOINT"):
            ep = params[0]
            ep_rows = sorted(
                (r for r in self.store if r["endpoint"] == ep),
                key=lambda r: r["id"], reverse=True,
            )
            keep = {r["id"] for r in ep_rows[:49]}
            self.store[:] = [r for r in self.store
                             if r["endpoint"] != ep or r["id"] in keep]
            return _FakeCursor([])
        raise AssertionError("unexpected digest SQL: %s" % sql[:80])

    def commit(self):
        pass


@pytest.fixture()
def digest_db(monkeypatch):
    import dashboard_services.db as db

    store = []
    monkeypatch.setattr(db, "get_conn", lambda: _FakeDigestConn(store))
    monkeypatch.setattr(pn, "_DIGEST_TABLE_INIT", False)
    return store


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


@pytest.fixture()
def fake_season(monkeypatch):
    import dashboard_services.api as api

    monkeypatch.setattr(api, "get_nfl_state",
                        lambda: {"season": "2026", "week": 4})
    monkeypatch.setattr(pn, "_league_display_name",
                        lambda plat, lid, season: "Blackedraw")


def _row(endpoint, prefs=None, account_key=None, league_id="L1"):
    return {"endpoint": endpoint, "p256dh": "k1", "auth": "a1",
            "prefs": json.dumps(prefs) if prefs is not None else None,
            "platform": "sleeper", "league_id": league_id,
            "account_key": account_key}


# ── Catalog ───────────────────────────────────────────────────────────────────

ALL_TYPE_KEYS = {
    "redzone_scores", "lineup_lock", "injury", "close_game",
    "matchup_preview", "recap_ready", "standings_update", "rival_trades",
    "transaction", "waiver_candidates", "watchlist", "value_drops",
    "breakout_roster", "playoff_odds", "top_movers",
}


def test_catalog_buckets_cover_every_type_key_exactly():
    buckets = pn.PUSH_TYPE_BUCKETS
    assert len(buckets) == 5
    seen = []
    for b in buckets:
        assert b["id"] and b["label"] and b["blurb"]
        assert b["types"], b["id"]
        for t in b["types"]:
            assert t["key"] and t["label"]
            seen.append(t["key"])
    assert set(seen) == ALL_TYPE_KEYS
    assert len(seen) == len(set(seen)), "catalog keys must be unique"


def test_digest_eligible_types_are_catalog_subset():
    assert set(pn._DIGEST_ELIGIBLE_TYPES) <= ALL_TYPE_KEYS
    # Live, time-critical alerts stay immediate even in digest mode.
    assert "redzone_scores" not in pn._DIGEST_ELIGIBLE_TYPES
    assert "top_movers" not in pn._DIGEST_ELIGIBLE_TYPES


# ── Buffering vs immediate ────────────────────────────────────────────────────

def test_digest_on_buffers_instead_of_sending(digest_db, fake_sends):
    rows = [
        _row("ep-digest", {"digest": True}, account_key="acct1"),
        _row("ep-plain", None, account_key="acct1"),
    ]
    sent = pn._send_with_digest(rows, "T", "B", "/w", "tag",
                                notif_type="waiver_candidates", league_id="L1")
    assert sent == 1  # only the no-prefs device went out immediately
    assert [s["endpoint"] for s in fake_sends] == ["ep-plain"]
    assert len(digest_db) == 1
    assert digest_db[0]["endpoint"] == "ep-digest"
    assert digest_db[0]["account_key"] == "acct1"
    assert digest_db[0]["notif_type"] == "waiver_candidates"


def test_no_prefs_behaves_exactly_as_before(digest_db, fake_sends):
    rows = [_row("ep-a", None), _row("ep-b", {})]
    sent = pn._send_with_digest(rows, "T", "B", "/w", "tag",
                                notif_type="waiver_candidates", league_id="L1")
    assert sent == 2
    assert digest_db == []


def test_digest_respects_type_prefs(digest_db, fake_sends):
    rows = [_row("ep-digest",
                 {"digest": True, "waiver_candidates": False},
                 account_key="acct1")]
    sent = pn._send_with_digest(rows, "T", "B", "/w", "tag",
                                notif_type="waiver_candidates", league_id="L1")
    assert sent == 0
    assert fake_sends == []
    assert digest_db == []


def test_live_types_stay_immediate_in_digest_mode(digest_db, fake_sends):
    rows = [_row("ep-digest", {"digest": True}, account_key="acct1")]
    sent = pn._send_with_digest(rows, "T", "B", "/w", "tag",
                                notif_type="redzone_scores", league_id="L1")
    assert sent == 1
    assert [s["endpoint"] for s in fake_sends] == ["ep-digest"]
    assert digest_db == []


# ── Flush: per-account grouping ───────────────────────────────────────────────

def test_flush_groups_per_account_and_cleans_up(digest_db, fake_sends,
                                                fake_season):
    pn._digest_buffer("ep1", "k", "a", "waiver_candidates", "T1", "B1",
                      "/w", "t", league_id="L1", platform="sleeper",
                      account_key="acctA")
    pn._digest_buffer("ep1", "k", "a", "injury", "T2", "B2",
                      "/w", "t", league_id="L2", platform="sleeper",
                      account_key="acctA")
    pn._digest_buffer("ep2", "k", "a", "close_game", "T3", "B3",
                      "/w", "t", league_id="L1", platform="sleeper",
                      account_key="acctA")
    pn._digest_buffer("ep3", "k", "a", "rival_trades", "T4", "B4",
                      "/w", "t", league_id="L9", platform="sleeper",
                      account_key="acctB")
    pn._digest_buffer("ep4", "k", "a", "watchlist", "T5", "B5",
                      "/w", "t", league_id="L7", platform="sleeper")  # legacy

    sent = pn._flush_digest()

    # One combined push per device; nothing left buffered.
    assert sent == 4
    assert digest_db == []
    by_ep = {s["endpoint"]: s for s in fake_sends}
    assert set(by_ep) == {"ep1", "ep2", "ep3", "ep4"}
    assert by_ep["ep1"]["title"] == "BR Fantasy digest"
    assert by_ep["ep1"]["url"] == "/portfolio"
    # ep1 (two leagues) summarizes both of its items; ep2 only its own.
    assert "2 alerts across 2 leagues" in by_ep["ep1"]["body"]
    assert "1 alert in Blackedraw" in by_ep["ep2"]["body"]
    assert "1 alert in Blackedraw" in by_ep["ep4"]["body"]


def test_flush_empty_buffer_sends_nothing(digest_db, fake_sends, fake_season):
    assert pn._flush_digest() == 0
    assert fake_sends == []


def test_digest_summary_counts_and_leagues(fake_season):
    items = [
        {"notif_type": "waiver_candidates", "league_id": "L1"},
        {"notif_type": "waiver_candidates", "league_id": "L2"},
        {"notif_type": "injury", "league_id": "L1"},
    ]
    title, body = pn._digest_summary(items, "2026")
    assert title == "BR Fantasy digest"
    assert body == ("3 alerts across 2 leagues: "
                    "2 waiver targets, 1 injury alert.")


def test_digest_summary_single_item_single_league(fake_season):
    items = [{"notif_type": "close_game", "league_id": "L1", "platform": "sleeper"}]
    title, body = pn._digest_summary(items, "2026")
    assert body == "1 alert in Blackedraw: 1 close game."


# ── Re-prompt persistence sanitization ────────────────────────────────────────

def test_reprompt_ui_prefs_sanitized():
    from routes.ui_prefs_bp import _sanitize_prefs, _ALLOWED_PREF_KEYS

    assert "push_reprompt_dismissed" in _ALLOWED_PREF_KEYS
    assert "push_reprompt_count" in _ALLOWED_PREF_KEYS
    assert "push_reprompt_last" in _ALLOWED_PREF_KEYS

    out = _sanitize_prefs({
        "push_reprompt_dismissed": True,
        "push_reprompt_count": 3,
        "push_reprompt_last": "2026-09-25",
        "push_reprompt_count_bad": "x",  # unknown key dropped
    })
    assert out == {"push_reprompt_dismissed": True,
                   "push_reprompt_count": 3,
                   "push_reprompt_last": "2026-09-25"}

    # Count clamps to a non-negative int; bad dates are dropped.
    out = _sanitize_prefs({"push_reprompt_count": -5,
                           "push_reprompt_last": "tomorrow"})
    assert out == {"push_reprompt_count": 0}


# ── Client static assertions ─────────────────────────────────────────────────

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _app_js():
    with open(os.path.join(REPO, "static", "app.js")) as f:
        return f.read()


def test_subscribe_picker_wired_to_catalog():
    js = _app_js()
    assert "window.openPushPicker" in js
    assert "Tell me about" in js
    assert "/api/push/catalog" in js
    # Every subscribe path opens the picker after a successful subscribe.
    assert js.count("window.openPushPicker(") >= 3


def test_digest_toggle_in_client():
    js = _app_js()
    assert "Hourly digest" in js
    assert 'data-digest' in js
    assert "data-picker-digest" in js
    # Digest toggle persists through the existing per-type preferences API.
    assert 'prefs.digest' in js


def test_reprompt_caps_and_dismissal():
    js = _app_js()
    assert "RP_MAX_SHOWS = 4" in js
    assert "push-reprompt-dismissed" in js
    assert "Don't show again" in js
    assert "/api/ui-prefs" in js
    # High-value moments only: RedZone + weekly hub on game days.
    assert "'redzone'" in js or '"redzone"' in js


def test_portfolio_has_notification_settings_entry():
    with open(os.path.join(REPO, "app.py")) as f:
        src = f.read()
    assert "pf-notif-btn" in src
    assert "openNotifPrefs" in src
