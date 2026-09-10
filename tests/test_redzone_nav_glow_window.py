"""The Redzone nav glow should only pulse while games are live or imminent
(the hour before kickoff), not for the whole calendar day.

Covers ``app._games_live_or_imminent`` — the gate behind the Weekly-button glow
and the pulsing Redzone dropdown dot.
"""
from datetime import datetime

import pytest

pytest.importorskip("pandas")
pytest.importorskip("flask")

import app as appmod


def _today_str():
    return datetime.now().strftime("%Y%m%d")


def _game(offset_seconds):
    """A schedule row dated today whose kickoff is ``offset_seconds`` from now."""
    return {
        "gameID": "G1",
        "home": "KC",
        "away": "LAC",
        "gameDate": _today_str(),
        "gameTime_epoch": datetime.now().timestamp() + offset_seconds,
    }


def test_glow_off_when_kickoff_is_hours_away(monkeypatch):
    # Kickoff in 3 hours: game day, but not live and not within the hour.
    monkeypatch.setattr(appmod, "load_week_schedule", lambda *a, **k: [_game(3 * 3600)])
    assert appmod._games_live_or_imminent(2025, 3) is False


def test_glow_on_within_hour_before_kickoff(monkeypatch):
    # Kickoff in 45 minutes: imminent.
    monkeypatch.setattr(appmod, "load_week_schedule", lambda *a, **k: [_game(45 * 60)])
    assert appmod._games_live_or_imminent(2025, 3) is True


def test_glow_on_while_game_is_live(monkeypatch):
    # Kickoff 90 minutes ago: within the kickoff→+4h live window.
    monkeypatch.setattr(appmod, "load_week_schedule", lambda *a, **k: [_game(-90 * 60)])
    assert appmod._games_live_or_imminent(2025, 3) is True


def test_glow_off_after_game_has_ended(monkeypatch):
    # Kickoff 5 hours ago: past the live tail.
    monkeypatch.setattr(appmod, "load_week_schedule", lambda *a, **k: [_game(-5 * 3600)])
    assert appmod._games_live_or_imminent(2025, 3) is False


def test_glow_ignores_games_not_dated_today(monkeypatch):
    game = _game(30 * 60)
    game["gameDate"] = "20200101"
    monkeypatch.setattr(appmod, "load_week_schedule", lambda *a, **k: [game])
    assert appmod._games_live_or_imminent(2025, 3) is False


def test_glow_off_with_missing_kickoff_epoch(monkeypatch):
    game = _game(30 * 60)
    del game["gameTime_epoch"]
    monkeypatch.setattr(appmod, "load_week_schedule", lambda *a, **k: [game])
    assert appmod._games_live_or_imminent(2025, 3) is False


def test_glow_off_on_empty_or_bad_inputs(monkeypatch):
    monkeypatch.setattr(appmod, "load_week_schedule", lambda *a, **k: [])
    assert appmod._games_live_or_imminent(0, 0) is False
    assert appmod._games_live_or_imminent(2025, None) is False


def test_glow_survives_schedule_load_failure(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("schedule unavailable")

    monkeypatch.setattr(appmod, "load_week_schedule", boom)
    assert appmod._games_live_or_imminent(2025, 3) is False
