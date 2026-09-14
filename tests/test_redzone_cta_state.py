"""The dashboard Redzone CTA has two states, split at kickoff.

``app._redzone_cta_state`` returns ``'pregame'`` in the hour before kickoff (a
calm banner), ``'live'`` once a game is in progress (a pulsing banner), and
``''`` otherwise -- the same overall window as ``_games_live_or_imminent``, just
split so the CTA can render a calm state and a live one.
"""
from datetime import datetime

import pytest

pytest.importorskip("pandas")
pytest.importorskip("flask")

import app as appmod


def _today_str():
    return datetime.now().strftime("%Y%m%d")


def _game(offset_seconds, date=None):
    """A schedule row whose kickoff is ``offset_seconds`` from now."""
    return {
        "gameID": "G1",
        "home": "KC",
        "away": "LAC",
        "gameDate": date or _today_str(),
        "gameTime_epoch": datetime.now().timestamp() + offset_seconds,
    }


def test_empty_when_kickoff_is_hours_away(monkeypatch):
    # Kickoff in 3 hours: game day, but outside the pregame hour and not live.
    monkeypatch.setattr(appmod, "load_week_schedule", lambda *a, **k: [_game(3 * 3600)])
    assert appmod._redzone_cta_state(2025, 3) == ""


def test_pregame_within_hour_before_kickoff(monkeypatch):
    # Kickoff in 45 minutes: the calm pregame window.
    monkeypatch.setattr(appmod, "load_week_schedule", lambda *a, **k: [_game(45 * 60)])
    assert appmod._redzone_cta_state(2025, 3) == "pregame"


def test_live_while_game_is_in_progress(monkeypatch):
    # Kickoff 90 minutes ago: within the kickoff->+4h live window.
    monkeypatch.setattr(appmod, "load_week_schedule", lambda *a, **k: [_game(-90 * 60)])
    assert appmod._redzone_cta_state(2025, 3) == "live"


def test_empty_after_game_has_ended(monkeypatch):
    # Kickoff 5 hours ago: past the live tail.
    monkeypatch.setattr(appmod, "load_week_schedule", lambda *a, **k: [_game(-5 * 3600)])
    assert appmod._redzone_cta_state(2025, 3) == ""


def test_live_wins_over_pregame(monkeypatch):
    # One game live, another kicking off soon: the live state wins.
    monkeypatch.setattr(
        appmod, "load_week_schedule",
        lambda *a, **k: [_game(45 * 60), _game(-30 * 60)],
    )
    assert appmod._redzone_cta_state(2025, 3) == "live"


def test_pregame_when_only_upcoming_game_is_imminent(monkeypatch):
    # A soon game plus a far-off game resolves to pregame, not "".
    monkeypatch.setattr(
        appmod, "load_week_schedule",
        lambda *a, **k: [_game(45 * 60), _game(3 * 3600)],
    )
    assert appmod._redzone_cta_state(2025, 3) == "pregame"


def test_ignores_games_not_dated_today(monkeypatch):
    monkeypatch.setattr(
        appmod, "load_week_schedule", lambda *a, **k: [_game(30 * 60, "20200101")]
    )
    assert appmod._redzone_cta_state(2025, 3) == ""


def test_empty_with_missing_kickoff_epoch(monkeypatch):
    game = _game(30 * 60)
    del game["gameTime_epoch"]
    monkeypatch.setattr(appmod, "load_week_schedule", lambda *a, **k: [game])
    assert appmod._redzone_cta_state(2025, 3) == ""


def test_empty_on_empty_or_bad_inputs(monkeypatch):
    monkeypatch.setattr(appmod, "load_week_schedule", lambda *a, **k: [])
    assert appmod._redzone_cta_state(0, 0) == ""
    assert appmod._redzone_cta_state(2025, None) == ""


def test_survives_schedule_load_failure(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("schedule unavailable")

    monkeypatch.setattr(appmod, "load_week_schedule", boom)
    assert appmod._redzone_cta_state(2025, 3) == ""
