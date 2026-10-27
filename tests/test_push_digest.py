"""Push preferences picker, digest mode, and re-prompt persistence.

Digest mode: per-device batching of eligible notification types into one
combined push per device, held until 9am ET. Existing subscribers with no
stored preferences must behave exactly as before (immediate sends).
"""
import json
import os
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import pytest

import utils.push_notifications as pn

ET = ZoneInfo("America/New_York")
# A flush time safely past the 9am ET quiet-hours gate.
DAYTIME_ET = datetime(2026, 10, 22, 10, 0, tzinfo=ET)


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
    "breakout_roster", "playoff_odds", "top_movers", "breakout_weekly",
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


# ── Flush: per-device grouping + 9am ET quiet hours ───────────────────────────

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

    sent = pn._flush_digest(now=DAYTIME_ET)

    # One combined push per device; nothing left buffered.
    assert sent == 4
    assert digest_db == []
    by_ep = {s["endpoint"]: s for s in fake_sends}
    assert set(by_ep) == {"ep1", "ep2", "ep3", "ep4"}
    assert by_ep["ep1"]["title"] == "BR Fantasy digest"
    # Both of ep1's items link to "/w", so the digest keeps that link.
    assert by_ep["ep1"]["url"] == "/w"
    # ep1 (two leagues) names both of its actual items; single-item devices
    # get their original push verbatim (own title, body, and link).
    assert "B1" in by_ep["ep1"]["body"]
    assert "B2" in by_ep["ep1"]["body"]
    assert "alerts across" not in by_ep["ep1"]["body"]
    assert by_ep["ep2"]["title"] == "T3"
    assert by_ep["ep2"]["body"] == "B3"
    assert by_ep["ep2"]["url"] == "/w"
    assert by_ep["ep4"]["title"] == "T5"
    assert by_ep["ep4"]["body"] == "B5"


def test_flush_empty_buffer_sends_nothing(digest_db, fake_sends, fake_season):
    assert pn._flush_digest(now=DAYTIME_ET) == 0
    assert fake_sends == []


def test_digest_flush_due_boundary_is_9am_et():
    assert pn._digest_flush_due(datetime(2026, 10, 22, 5, 5, tzinfo=ET)) is False
    assert pn._digest_flush_due(datetime(2026, 10, 22, 8, 59, tzinfo=ET)) is False
    assert pn._digest_flush_due(datetime(2026, 10, 22, 9, 0, tzinfo=ET)) is True
    assert pn._digest_flush_due(datetime(2026, 10, 22, 21, 30, tzinfo=ET)) is True
    # Aware times in other zones convert: 12:05 UTC is 8:05am EDT, 13:05 UTC
    # is 9:05am EDT.
    assert pn._digest_flush_due(datetime(2026, 10, 22, 12, 5,
                                         tzinfo=timezone.utc)) is False
    assert pn._digest_flush_due(datetime(2026, 10, 22, 13, 5,
                                         tzinfo=timezone.utc)) is True
    # Naive times are read as Eastern.
    assert pn._digest_flush_due(datetime(2026, 10, 22, 5, 5)) is False
    assert pn._digest_flush_due(datetime(2026, 10, 22, 9, 5)) is True


def test_flush_held_before_9am_then_delivered(digest_db, fake_sends,
                                              fake_season):
    # The 5:05am scenario: the daily job buffered a value drop at ~4am and
    # the hourly runner must NOT flush it before 9am ET.
    pn._digest_buffer("ep1", "k", "a", "value_drops", "T1", "B1",
                      "/w", "t", league_id="L1", platform="sleeper",
                      account_key="acctA")

    sent = pn._flush_digest(now=datetime(2026, 10, 22, 5, 5, tzinfo=ET))
    assert sent == 0
    assert fake_sends == []
    assert len(digest_db) == 1  # still buffered, not dropped

    # The 9:05am hourly run delivers it as the one combined push.
    sent = pn._flush_digest(now=datetime(2026, 10, 22, 9, 5, tzinfo=ET))
    assert sent == 1
    assert [s["endpoint"] for s in fake_sends] == ["ep1"]
    # A lone buffered item is delivered as its original push, not a count.
    assert fake_sends[0]["title"] == "T1"
    assert fake_sends[0]["body"] == "B1"
    assert fake_sends[0]["url"] == "/w"
    assert digest_db == []


def test_flush_combines_same_device_across_account_keys(digest_db, fake_sends,
                                                        fake_season):
    # One league subscribed signed-in (account_key set), another legacy
    # (no account_key): same device must still get ONE combined digest,
    # not the two separate pushes Kaedon screenshotted.
    pn._digest_buffer("ep1", "k", "a", "value_drops", "T1", "B1",
                      "/w", "t", league_id="L1", platform="espn",
                      account_key="acctA")
    pn._digest_buffer("ep1", "k", "a", "value_drops", "T2", "B2",
                      "/w", "t", league_id="L2", platform="yahoo")

    sent = pn._flush_digest(now=DAYTIME_ET)

    assert sent == 1
    assert len(fake_sends) == 1
    assert "B1" in fake_sends[0]["body"]
    assert "B2" in fake_sends[0]["body"]
    assert "value drops" not in fake_sends[0]["body"]
    assert digest_db == []


def _named_leagues(monkeypatch):
    names = {"L1": "Blackedraw", "L2": "League of Greatness"}
    monkeypatch.setattr(pn, "_league_display_name",
                        lambda plat, lid, season: names.get(lid, ""))


def test_digest_summary_lists_items_grouped_by_league(fake_season, monkeypatch):
    _named_leagues(monkeypatch)
    items = [
        {"notif_type": "waiver_candidates", "league_id": "L1",
         "platform": "sleeper", "title": "Waiver of the week",
         "body": "Top target is available.", "url": "/l1/waivers"},
        {"notif_type": "waiver_candidates", "league_id": "L2",
         "platform": "sleeper", "title": "Waiver of the week",
         "body": "Another target is available.", "url": "/l2/waivers"},
        {"notif_type": "value_drops", "league_id": "L1",
         "platform": "sleeper", "title": "Dynasty value dropping",
         "body": "A player is losing dynasty value.", "url": "/l1/trade"},
    ]
    title, body, url = pn._digest_payload(items, "2026")
    assert title == "BR Fantasy digest"
    # The actual alerts, grouped by league, in buffered order. No counts.
    assert body == ("Blackedraw: Top target is available. "
                    "A player is losing dynasty value. "
                    "League of Greatness: Another target is available.")
    assert "alerts across" not in body
    # Named items point at different places, so the tap falls back.
    assert url == "/portfolio"


def test_digest_summary_single_item_is_verbatim(fake_season):
    items = [{"notif_type": "close_game", "league_id": "L1",
              "platform": "sleeper", "title": "Close game in Blackedraw",
              "body": "You are up 2.1 with one player left.",
              "url": "/sleeper/2026/L1/matchup"}]
    title, body, url = pn._digest_payload(items, "2026")
    assert title == "Close game in Blackedraw"
    assert body == "You are up 2.1 with one player left."
    assert url == "/sleeper/2026/L1/matchup"
    # The (title, body) shim agrees.
    assert pn._digest_summary(items, "2026") == (title, body)


def test_digest_payload_caps_named_items(fake_season):
    items = [
        {"notif_type": "value_drops", "league_id": "L1", "platform": "sleeper",
         "title": "T", "body": f"Body {i}.", "url": "/l1/trade"}
        for i in range(5)
    ]
    _title, body, url = pn._digest_payload(items, "2026")
    assert "Body 0." in body
    assert "Body 2." in body
    assert "Body 3." not in body
    assert body.endswith("+2 more")
    # Every named item shares one link, so the digest keeps it.
    assert url == "/l1/trade"


def test_digest_item_text_strips_trailing_league_name(fake_season):
    it = {"notif_type": "value_drops", "title": "Dynasty value dropping",
          "body": "A player is losing value. Check your trade options "
                  "in Blackedraw."}
    assert pn._digest_item_text(it, "Blackedraw") == (
        "A player is losing value. Check your trade options.")


def test_digest_item_text_falls_back_to_title_and_label(fake_season):
    it = {"notif_type": "transaction", "title": "Big drop in Blackedraw",
          "body": ""}
    assert pn._digest_item_text(it, "Blackedraw") == (
        "Big drop in Blackedraw: big drop.")


# ── Re-prompt persistence sanitization ────────────────────────────────────────

def test_reprompt_ui_prefs_sanitized():
    from routes.user_pages_bp import _sanitize_prefs, _ALLOWED_PREF_KEYS

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
    # High-value moments only: ScoreZone + weekly hub on game days.
    assert "'scorezone'" in js or '"scorezone"' in js


def test_portfolio_has_notification_settings_entry():
    with open(os.path.join(REPO, "app.py")) as f:
        src = f.read()
    assert "pf-notif-btn" in src
    assert "openNotifPrefs" in src
