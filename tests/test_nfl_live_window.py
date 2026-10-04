"""League-context cache goes short-TTL during live NFL windows.

The matchup page reads scores from the cached league ctx. With a flat 12-hour
TTL, a refresh during live games serves stale pre-game data because the cache
only busts on roster changes. ``_league_ctx_effective_ttl()`` drops to 2
minutes inside the Thursday/Sunday/Monday game windows so scores stay fresh.
"""
import time
from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

# app.py imports pandas (and flask/openai) at module load. The fast "lint" CI
# shard has flask but not pandas, so guard on pandas first -- otherwise
# importing app here raises at COLLECTION and aborts the whole run.
pytest.importorskip("pandas")
pytest.importorskip("flask")
pytest.importorskip("openai")  # app.py pulls openai via dashboard_services.ai.client

import app  # noqa: E402

ET = ZoneInfo("America/New_York")


def _et(year, month, day, hour, minute):
    return datetime(year, month, day, hour, minute, tzinfo=ET)


# 2026-10-04 is a Sunday; neighboring dates cover the other weekdays.


@pytest.mark.parametrize(
    "when,expected",
    [
        # Thursday window: 8:00 PM - 11:30 PM ET
        (_et(2026, 10, 8, 21, 0), True),   # Thu 9 PM
        (_et(2026, 10, 8, 20, 0), True),   # Thu 8 PM (start boundary)
        (_et(2026, 10, 8, 23, 29), True),  # Thu 11:29 PM
        (_et(2026, 10, 8, 23, 30), False),  # Thu 11:30 PM (end boundary)
        (_et(2026, 10, 8, 19, 59), False),  # Thu 7:59 PM
        # Sunday window: 1:00 PM - 11:30 PM ET
        (_et(2026, 10, 4, 15, 0), True),   # Sun 3 PM
        (_et(2026, 10, 4, 13, 0), True),   # Sun 1 PM (start boundary)
        (_et(2026, 10, 4, 23, 29), True),  # Sun 11:29 PM
        (_et(2026, 10, 4, 23, 30), False),  # Sun 11:30 PM (end boundary)
        (_et(2026, 10, 4, 12, 59), False),  # Sun 12:59 PM
        # Monday window: 8:00 PM - 11:30 PM ET
        (_et(2026, 10, 5, 21, 0), True),   # Mon 9 PM
        (_et(2026, 10, 5, 20, 0), True),   # Mon 8 PM (start boundary)
        (_et(2026, 10, 5, 23, 30), False),  # Mon 11:30 PM (end boundary)
        (_et(2026, 10, 5, 10, 0), False),  # Mon 10 AM
        # Non-game days: never live
        (_et(2026, 10, 6, 14, 0), False),  # Tue 2 PM
        (_et(2026, 10, 7, 21, 0), False),  # Wed 9 PM
        (_et(2026, 10, 3, 15, 0), False),  # Sat 3 PM
    ],
)
def test_is_nfl_live_window(when, expected):
    assert app._is_nfl_live_window(when) is expected


def test_effective_ttl_short_during_live_window(monkeypatch):
    monkeypatch.setattr(app, "_is_nfl_live_window", lambda *a, **k: True)
    assert app._league_ctx_effective_ttl() == app._LIVE_SCORE_CACHE_TTL
    assert app._league_ctx_effective_ttl() == 120


def test_effective_ttl_long_outside_live_window(monkeypatch):
    monkeypatch.setattr(app, "_is_nfl_live_window", lambda *a, **k: False)
    assert app._league_ctx_effective_ttl() == app.CACHE_TTL


def test_cache_valid_uses_short_ttl_in_live_window(monkeypatch):
    monkeypatch.setattr(app, "_is_nfl_live_window", lambda *a, **k: True)
    platform, season, league_id = "sleeper", 2026, "livewindowtest"
    now = time.time()
    # 3 minutes old: expired under the 120s live TTL, fresh under 12h.
    assert app._league_ctx_cache_valid(
        {"ts": now - 180, "ctx": {}}, platform, season, league_id
    ) is False
    # 60 seconds old: still fresh under the live TTL.
    assert app._league_ctx_cache_valid(
        {"ts": now - 60, "ctx": {}}, platform, season, league_id
    ) is True


def test_cache_valid_keeps_long_ttl_outside_live_window(monkeypatch):
    monkeypatch.setattr(app, "_is_nfl_live_window", lambda *a, **k: False)
    platform, season, league_id = "sleeper", 2026, "livewindowtest"
    now = time.time()
    # 1 hour old: expired under the live TTL, fresh under the 12h TTL.
    assert app._league_ctx_cache_valid(
        {"ts": now - 3600, "ctx": {}}, platform, season, league_id
    ) is True
    # Older than 12h: expired in both.
    assert app._league_ctx_cache_valid(
        {"ts": now - app.CACHE_TTL - 10, "ctx": {}}, platform, season, league_id
    ) is False
