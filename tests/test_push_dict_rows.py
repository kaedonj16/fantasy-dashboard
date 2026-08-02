"""Guards utils.push_notifications against the psycopg dict_row bug class.

get_conn() uses psycopg's dict_row factory, so query rows are dicts keyed by
column name. The push helpers used to read them positionally (r[0], r[3], tuple
unpacking), which raised KeyError / bound column-name strings and, because the
broadcast callers swallow exceptions, silently sent notifications to nobody.
These pin the column-name access so that can't regress.

Pure unit tests - the module's heavy deps (pywebpush, cryptography) are imported
lazily, so importing these helpers needs nothing extra.
"""
from utils.push_notifications import _filter_prefs


def _row(endpoint, p256dh, auth, prefs=None):
    # Shape of a psycopg dict_row from the push_subscriptions SELECTs.
    return {"endpoint": endpoint, "p256dh": p256dh, "auth": auth,
            "prefs": prefs, "owner_id": "o1"}


def test_filter_prefs_reads_dict_rows_without_notif_type():
    rows = [_row("https://e/1", "k1", "a1"), _row("https://e/2", "k2", "a2")]
    out = _filter_prefs(rows, None)
    assert out == [("https://e/1", "k1", "a1"), ("https://e/2", "k2", "a2")]


def test_filter_prefs_respects_disabled_type():
    import json
    rows = [
        _row("https://on", "k", "a", prefs=json.dumps({"lineup_lock": True})),
        _row("https://off", "k", "a", prefs=json.dumps({"lineup_lock": False})),
        _row("https://default", "k", "a", prefs=None),  # unset -> enabled
    ]
    out = _filter_prefs(rows, "lineup_lock")
    endpoints = [t[0] for t in out]
    assert "https://on" in endpoints
    assert "https://default" in endpoints
    assert "https://off" not in endpoints


def test_filter_prefs_accepts_predecoded_jsonb_prefs():
    # dict_row returns JSONB already decoded to a dict, not a string.
    rows = [_row("https://off", "k", "a", prefs={"waiver": False})]
    assert _filter_prefs(rows, "waiver") == []
    assert _filter_prefs(rows, "trade") == [("https://off", "k", "a")]


def test_lineup_lock_sends_bench_points_push():
    """An owner with a legal lineup but a better bench option gets the
    'Points on your bench' push (start Strong RB over Weak RB); an already-
    optimal owner gets only the generic reminder."""
    import pytest
    # Unlike the pure _filter_prefs tests above, this one drives the full
    # lineup-lock broadcast, which patches (and therefore imports) utils.utils
    # and dashboard_services.db. Those pull in bs4 / psycopg, so skip cleanly
    # when the full stack isn't installed — matching how the rest of the suite
    # gates its integration tests (see tests/conftest.py).
    pytest.importorskip("utils.utils")
    pytest.importorskip("dashboard_services.db")

    import sys
    import time
    import types
    from unittest import mock
    import utils.push_notifications as pn

    kick_ms = int((time.time() + 60 * 60) * 1000)  # inside the 40–100 min window

    class FakeConn:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def execute(self, q, params=None):
            class R:
                def fetchall(self_):
                    return [
                        {"endpoint": "e1", "p256dh": "k", "auth": "a", "prefs": None, "owner_id": "O1"},
                        {"endpoint": "e2", "p256dh": "k", "auth": "a", "prefs": None, "owner_id": "O2"},
                    ]
            return R()
        def commit(self): pass

    sent = []
    rosters = [
        {"owner_id": "O1", "starters": ["q1", "r_weak"],
         "players": ["q1", "r_strong", "r_weak"], "reserve": [], "taxi": []},
        {"owner_id": "O2", "starters": ["q2", "r_ok"],
         "players": ["q2", "r_ok"], "reserve": [], "taxi": []},
    ]
    nfl = {
        "q1": {"full_name": "QB One", "team": "KC", "position": "QB", "injury_status": ""},
        "q2": {"full_name": "QB Two", "team": "BUF", "position": "QB", "injury_status": ""},
        "r_weak": {"full_name": "Weak RB", "team": "KC", "position": "RB", "injury_status": ""},
        "r_strong": {"full_name": "Strong RB", "team": "BUF", "position": "RB", "injury_status": ""},
        "r_ok": {"full_name": "OK RB", "team": "BUF", "position": "RB", "injury_status": ""},
    }
    proj = {"q1": 20.0, "q2": 21.0, "r_weak": 6.0, "r_strong": 16.0, "r_ok": 12.0}

    fake_app = types.ModuleType("app")
    fake_app.build_projections_by_week = lambda season, week, ss: {int(week): {"projections": proj}}

    with mock.patch("dashboard_services.api.get_nfl_state",
                    return_value={"season": 2025, "week": 9, "season_type": "reg"}), \
         mock.patch("utils.utils.load_week_schedule",
                    return_value=[{"gameTime_epoch": kick_ms, "home": "KC", "away": "BUF"}]), \
         mock.patch("dashboard_services.db.get_conn", return_value=FakeConn()), \
         mock.patch.object(pn, "_get_subscribed_leagues", return_value=[("L1", "sleeper")]), \
         mock.patch.object(pn, "_app_state_get", return_value=None), \
         mock.patch.object(pn, "_app_state_set", return_value=None), \
         mock.patch.object(pn, "_send_to_endpoints",
                           side_effect=lambda eps, title, body, url="/", tag="update":
                           sent.append({"title": title, "body": body}) or 1), \
         mock.patch.object(pn, "_filter_prefs", side_effect=lambda rows, t: list(rows)), \
         mock.patch("dashboard_services.api.get_nfl_players", return_value=nfl), \
         mock.patch("dashboard_services.api.get_rosters", return_value=rosters), \
         mock.patch("dashboard_services.api.get_league", return_value={"roster_positions": ["QB", "RB"]}), \
         mock.patch.dict(sys.modules, {"app": fake_app}):
        pn.notify_lineup_lock()

    titles = [c["title"] for c in sent]
    assert "Points on your bench" in titles
    assert "Lineups lock soon" in titles  # the optimal owner still gets the generic one
    bench = next(c for c in sent if c["title"] == "Points on your bench")
    assert "Strong RB" in bench["body"] and "Weak RB" in bench["body"]
