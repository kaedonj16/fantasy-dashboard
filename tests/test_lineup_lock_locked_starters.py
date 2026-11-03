"""Regression: the Monday lineup-lock push must not 'fix' a locked starter.

Kaedon's 2026-11-02 screenshot (7:09 PM ET, digest push):

    "League of Greatness: Monday night players lock in about an hour.
     Jeremiyah Love is listed Out. Sit Jeremiyah Love for MarShawn Lloyd
     (+6.6 proj). Check your lineup."

Love (ARI) had played Sunday, got hurt mid-game and flipped to Out, and
his live projection cratered. His slot locked at Sunday kickoff, so both
halves of that push were unactionable: the Out issue could not be
addressed and the swap could not be made.

Root cause: ``_lineup_lock_send`` computed ``locked_teams`` from the
day's games only. ``notify_lineup_lock`` drops past games before
grouping by day, so the day's games are all in the future and the lock
set was always empty. The Next Steps fix (PR #2947) suppressed exactly
this pattern on that surface; the lock push needs the same treatment:
locked teams come from the full week's games, locked starters' issues
are suppressed, and locked players are excluded from both sides of a
swap.
"""
from __future__ import annotations

import sys
import time
import types

import pytest

pytest.importorskip("flask")

import dashboard_services.api as api_mod
import dashboard_services.db as dbmod
import dashboard_services.platform_api as plat_mod
import utils.push_notifications as pn

HOUR = 3600


class _FakeResult:
    def __init__(self, query=""):
        self._query = query

    def fetchall(self):
        if "FROM push_subscriptions" in self._query:
            return [{
                "endpoint": "https://push.example/ep1",
                "p256dh": "k", "auth": "a", "prefs": None,
                "owner_id": "o1", "account_key": "",
            }]
        return []

    def fetchone(self):
        return None


class _FakeConn:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, query, params=None):
        return _FakeResult(query)

    def commit(self):
        pass


def _run_lock_send(monkeypatch, *, day_games, week_games, players,
                     rosters, proj_map, roster_positions=("RB", "RB")):
    """Drive _lineup_lock_send with fakes; return captured push bodies."""
    captured = []

    monkeypatch.setattr(dbmod, "get_conn", lambda *a, **k: _FakeConn())
    monkeypatch.setattr(pn, "_app_state_get", lambda conn, key: None)
    monkeypatch.setattr(pn, "_app_state_set", lambda conn, key, val: None)
    monkeypatch.setattr(pn, "_get_subscribed_leagues",
                        lambda: [("L1", "sleeper")])
    monkeypatch.setattr(pn, "_league_display_name",
                        lambda *a: "League of Greatness")
    monkeypatch.setattr(
        pn, "_send_with_digest",
        lambda rows, title, body, url="/", tag="update", **kw:
            captured.append(body) or 1,
    )
    monkeypatch.setattr(api_mod, "get_nfl_players", lambda: players)
    monkeypatch.setattr(plat_mod, "get_rosters", lambda *a: rosters)
    monkeypatch.setattr(plat_mod, "get_league",
                        lambda *a: {"roster_positions": list(roster_positions)})

    fake_app = types.ModuleType("app")
    fake_app.build_projections_by_week = (
        lambda season, week, *a: {int(week): {"projections": proj_map}}
    )
    monkeypatch.setitem(sys.modules, "app", fake_app)

    pn._lineup_lock_send(
        day_games, 2026, 9,
        dedupe_key="lineup_lock_day", dedupe_value="2026-9-2026-11-02",
        tag="lineup-lock-2026-9-2026-11-02",
        kickoff_line="Monday night players lock in about an hour.",
        soon_line="Monday night players lock soon.",
        log_label="mon",
        week_games=week_games,
    )
    return captured


def _screenshot_setup(love_kickoff):
    """Love (ARI, Out, proj cratered to 0.0) started; Lloyd (GB) benched.

    ``love_kickoff`` is Love's game epoch: past = the screenshot (played
    Sunday, hurt mid-game), future = the pre-kickoff control.
    """
    now = time.time()
    monday = {"home": "GB", "away": "CHI",
              "gameTime_epoch": str(int(now + HOUR))}
    love_game = {"home": "ARI", "away": "SEA",
                 "gameTime_epoch": str(int(love_kickoff))}
    players = {
        "love": {"full_name": "Jeremiyah Love", "team": "ARI",
                 "injury_status": "Out", "position": "RB"},
        "monrb": {"full_name": "Monday Back", "team": "GB",
                  "injury_status": "", "position": "RB"},
        "lloyd": {"full_name": "MarShawn Lloyd", "team": "GB",
                  "injury_status": "", "position": "RB"},
    }
    rosters = [{
        "owner_id": "o1", "roster_id": 1,
        "starters": ["love", "monrb"],
        "players": ["love", "monrb", "lloyd"],
    }]
    proj = {"love": 0.0, "monrb": 12.0, "lloyd": 6.6}
    return monday, love_game, players, rosters, proj


def test_locked_midgame_injured_starter_gets_no_lock_push(monkeypatch):
    now = time.time()
    monday, love_game, players, rosters, proj = _screenshot_setup(now - 26 * HOUR)
    captured = _run_lock_send(
        monkeypatch,
        day_games=[monday], week_games=[love_game, monday],
        players=players, rosters=rosters, proj_map=proj,
    )
    # Love's issue is suppressed and the Love-for-Lloyd swap is excluded:
    # nothing actionable remains, so no push at all.
    assert captured == []


def test_unlocked_out_starter_still_gets_screenshot_push(monkeypatch):
    now = time.time()
    monday, love_game, players, rosters, proj = _screenshot_setup(now + 2 * HOUR)
    captured = _run_lock_send(
        monkeypatch,
        day_games=[love_game], week_games=[love_game, monday],
        players=players, rosters=rosters, proj_map=proj,
    )
    assert len(captured) == 1
    body = captured[0]
    assert "Monday night players lock in about an hour." in body
    assert "Jeremiyah Love is listed Out" in body
    assert "Sit Jeremiyah Love for MarShawn Lloyd (+6.6 proj)" in body


def test_locked_bench_player_never_suggested_as_swap_in(monkeypatch):
    now = time.time()
    monday = {"home": "GB", "away": "CHI",
              "gameTime_epoch": str(int(now + HOUR))}
    sunday = {"home": "ARI", "away": "SEA",
              "gameTime_epoch": str(int(now - 26 * HOUR))}
    players = {
        "weak": {"full_name": "Weak Monday Back", "team": "GB",
                 "injury_status": "", "position": "RB"},
        "monrb": {"full_name": "Monday Back", "team": "GB",
                  "injury_status": "", "position": "RB"},
        "sunday_bench": {"full_name": "Sunday Bench Back", "team": "ARI",
                         "injury_status": "", "position": "RB"},
        "mon_bench": {"full_name": "Monday Bench Back", "team": "GB",
                      "injury_status": "", "position": "RB"},
    }
    rosters = [{
        "owner_id": "o1", "roster_id": 1,
        "starters": ["weak", "monrb"],
        "players": ["weak", "monrb", "sunday_bench", "mon_bench"],
    }]
    # The Sunday bench back out-projects everyone but already played, so
    # he cannot be started; the Monday bench back is the actionable swap.
    proj = {"weak": 1.0, "monrb": 12.0, "sunday_bench": 20.0, "mon_bench": 5.0}
    captured = _run_lock_send(
        monkeypatch,
        day_games=[monday], week_games=[sunday, monday],
        players=players, rosters=rosters, proj_map=proj,
    )
    assert len(captured) == 1
    assert "Sunday Bench Back" not in captured[0]
    assert "Sit Weak Monday Back for Monday Bench Back (+4.0 proj)" in captured[0]
