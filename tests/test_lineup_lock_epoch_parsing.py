"""Regression: notify_lineup_lock crashed on every hourly run with
"unsupported operand type(s) for /: 'str' and 'int'".

gameTime_epoch arrives as a *string* from the JSON schedule cache, and the
notifier did min(epochs) / 1000 directly on the raw values. The TypeError fired
before the kickoff-window check, so the lineup-lock push never sent at all.
Epochs must be coerced to float (unparseable values skipped).
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

pytest.importorskip("flask")

import utils.push_notifications as pn


def _stub_schedule_module(monkeypatch, games):
    """utils.utils pulls in bs4 at import; stub just the schedule loader."""
    import sys
    import types

    stub = types.ModuleType("utils.utils")
    stub.load_week_schedule = lambda season, w: games
    monkeypatch.setitem(sys.modules, "utils.utils", stub)


def _games_with_string_epochs(kickoff: datetime):
    ms = str(int(kickoff.timestamp() * 1000))  # string, like the JSON cache
    return [
        {"home": "KC", "away": "BUF", "gameTime_epoch": ms},
        {"home": "DAL", "away": "PHI", "gameTime_epoch": None},      # missing
        {"home": "SF", "away": "SEA", "gameTime_epoch": "not-a-time"},  # garbage
    ]


def test_lineup_lock_tolerates_string_epochs(monkeypatch):
    kickoff = datetime.now(tz=timezone.utc) + timedelta(hours=5)  # outside window
    monkeypatch.setattr(
        "dashboard_services.api.get_nfl_state",
        lambda: {"season": "2026", "week": 4, "season_type": "reg"},
    )
    _stub_schedule_module(monkeypatch, _games_with_string_epochs(kickoff))
    # Must not raise TypeError; returns quietly outside the 40-100 min window.
    pn.notify_lineup_lock()


def test_lineup_lock_inside_window_sends(monkeypatch):
    """End-to-end through the send path with string epochs: one league, one
    owner, one subscription -> exactly one push attempted."""
    kickoff = datetime.now(tz=timezone.utc) + timedelta(minutes=60)
    monkeypatch.setattr(
        "dashboard_services.api.get_nfl_state",
        lambda: {"season": "2026", "week": 4, "season_type": "reg"},
    )
    _stub_schedule_module(monkeypatch, _games_with_string_epochs(kickoff))
    monkeypatch.setattr(pn, "_get_subscribed_leagues", lambda: [("L1", "sleeper")])
    monkeypatch.setattr(
        "dashboard_services.platform_api.get_rosters",
        lambda platform, league_id, season: [
            # "0" is an empty starting slot -> find_lineup_issues flags it.
            {"owner_id": "owner1", "starters": ["0", "2"], "players": ["2", "3"]}
        ],
    )
    monkeypatch.setattr(
        "dashboard_services.platform_api.get_league",
        lambda platform, league_id, season: {"roster_positions": ["QB", "RB"]},
    )
    monkeypatch.setattr(
        "dashboard_services.api.get_nfl_players",
        lambda: {
            "1": {"full_name": "A Back", "team": "KC", "injury_status": ""},
            "2": {"full_name": "B Back", "team": "BUF", "injury_status": ""},
            "3": {"full_name": "C Back", "team": "KC", "injury_status": ""},
        },
    )

    sent_calls = []

    class _FakeConn:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def execute(self, query, params=None):
            self.last_query = query
            return _FakeResult(query)

        def commit(self):
            pass

    class _FakeResult:
        def __init__(self, query=""):
            self._query = query

        def fetchone(self):
            return None  # lineup_lock_week not set -> not yet notified

        def fetchall(self):
            if "FROM push_subscriptions" in self._query:
                return [{
                    "endpoint": "https://push.example/ep1",
                    "p256dh": "k", "auth": "a",
                    "prefs": None, "owner_id": "owner1",
                }]
            return []

    monkeypatch.setattr("dashboard_services.db.get_conn", lambda *a, **k: _FakeConn())
    monkeypatch.setattr(
        pn, "_send_to_endpoints",
        lambda endpoints, title, body, url="/", tag="update": sent_calls.append(
            (title, tag)) or len(endpoints),
    )

    pn.notify_lineup_lock()
    assert sent_calls, "expected at least one lineup-lock push inside the window"
    assert all(t.startswith("lineup-lock-2026-4") for _, t in sent_calls)
