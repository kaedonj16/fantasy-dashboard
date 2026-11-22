"""Live/final scoreboard verdicts beat the kickoff-epoch window.

``normalize_game_status_from_scoreboard`` used to decide pre/in/post purely
from ``gameTime_epoch`` whenever an epoch was present. A flexed or
mis-parsed kickoff epoch can sit hours in the future while the game is
already being played, so the status map kept every starter on both teams
at not_started and their matchup rows rendered PROJ mid-game (PIT @ PHI,
2026-11-22, Jaylen Warren in Q2). An explicit live/final verdict from the
scoreboard now wins over the window.

Code "0" (scheduled) deliberately does NOT win: the on-disk week schedule
is written pregame with code "0" on every row and never rewritten, so a
scheduled code must still fall through to the epoch window -- that window
is what flips those frozen rows to "in" at kickoff.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

pytest.importorskip("flask")
pytest.importorskip("requests")
pytest.importorskip("bs4")

from utils.game_status import normalize_game_status_from_scoreboard

NOW = datetime(2026, 11, 22, 19, 0, tzinfo=timezone.utc)  # Sun 2:00pm ET


def _epoch(dt: datetime) -> int:
    return int(dt.timestamp())


def test_live_code_beats_future_kickoff_epoch():
    game = {
        "gameStatusCode": "1",
        "gameStatus": "In Progress",
        "gameTime_epoch": _epoch(NOW + timedelta(hours=3)),
    }
    assert normalize_game_status_from_scoreboard(game, now=NOW) == "in"


def test_final_code_beats_future_kickoff_epoch():
    game = {
        "gameStatusCode": "2",
        "gameStatus": "Final",
        "gameTime_epoch": _epoch(NOW + timedelta(hours=1)),
    }
    assert normalize_game_status_from_scoreboard(game, now=NOW) == "post"


def test_live_status_text_beats_future_kickoff_epoch():
    game = {
        "gameStatus": "In Progress - Q2",
        "gameTime_epoch": _epoch(NOW + timedelta(hours=2)),
    }
    assert normalize_game_status_from_scoreboard(game, now=NOW) == "in"


def test_scheduled_code_still_uses_epoch_window():
    future = {
        "gameStatusCode": "0",
        "gameStatus": "Scheduled",
        "gameTime_epoch": _epoch(NOW + timedelta(hours=3)),
    }
    assert normalize_game_status_from_scoreboard(future, now=NOW) == "pre"

    # The frozen pregame schedule row flips live purely on the clock.
    started = {
        "gameStatusCode": "0",
        "gameStatus": "Scheduled",
        "gameTime_epoch": _epoch(NOW - timedelta(hours=1)),
    }
    assert normalize_game_status_from_scoreboard(started, now=NOW) == "in"

    # ...and finalizes once the 3-hour window has passed.
    ended = {
        "gameStatusCode": "0",
        "gameStatus": "Scheduled",
        "gameTime_epoch": _epoch(NOW - timedelta(hours=4)),
    }
    assert normalize_game_status_from_scoreboard(ended, now=NOW) == "post"


def test_no_epoch_falls_back_to_status_fields():
    assert normalize_game_status_from_scoreboard(
        {"gameStatusCode": "1"}, now=NOW) == "in"
    assert normalize_game_status_from_scoreboard(
        {"gameStatusCode": "2"}, now=NOW) == "post"
    assert normalize_game_status_from_scoreboard(
        {"gameStatusCode": "0"}, now=NOW) == "pre"
    assert normalize_game_status_from_scoreboard({}, now=NOW) == "pre"
