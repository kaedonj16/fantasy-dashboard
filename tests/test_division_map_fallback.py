"""Tests for the division-map last-good fallback.

A transient provider response that omits per-roster division assignments must
not wipe Div badges: when a league previously had divisions, the last good
map is reused. Leagues without divisions are unaffected.
"""
import logging

from utils import standings_divisions as sd
from utils.standings_divisions import (
    div_map_for_ctx,
    is_division_game,
    roster_division_map,
)


def _rosters_with_divisions():
    return [
        {"roster_id": 1, "settings": {"division": 1}},
        {"roster_id": 2, "settings": {"division": 1}},
        {"roster_id": 3, "settings": {"division": 2}},
        {"roster_id": 4, "settings": {"division": 2}},
    ]


def _rosters_missing_divisions():
    # Provider hiccup: division assignments absent.
    return [
        {"roster_id": 1, "settings": {}},
        {"roster_id": 2, "settings": {}},
        {"roster_id": 3, "settings": {}},
        {"roster_id": 4, "settings": {}},
    ]


def setup_function(_):
    sd._LAST_GOOD_DIV_MAP.clear()


def teardown_function(_):
    sd._LAST_GOOD_DIV_MAP.clear()


def test_good_map_is_cached_and_returned():
    out = roster_division_map(
        _rosters_with_divisions(),
        league_key="sleeper:123",
        settings={"divisions": 2},
    )
    assert out == {1: 1, 2: 1, 3: 2, 4: 2}
    assert sd._LAST_GOOD_DIV_MAP["sleeper:123"] == out


def test_hiccup_falls_back_to_cached_map(caplog):
    roster_division_map(
        _rosters_with_divisions(),
        league_key="sleeper:123",
        settings={"divisions": 2},
    )
    with caplog.at_level(logging.WARNING, logger="utils.standings_divisions"):
        out = roster_division_map(
            _rosters_missing_divisions(),
            league_key="sleeper:123",
            settings={"divisions": 2},
        )
    assert out == {1: 1, 2: 1, 3: 2, 4: 2}
    assert is_division_game(1, 2, out) is True
    assert is_division_game(1, 3, out) is False
    assert any("[divisions] roster data missing divisions" in r.message for r in caplog.records)


def test_explicit_no_divisions_clears_stale_cache():
    roster_division_map(
        _rosters_with_divisions(),
        league_key="sleeper:123",
        settings={"divisions": 2},
    )
    out = roster_division_map(
        _rosters_missing_divisions(),
        league_key="sleeper:123",
        settings={"divisions": 1},
    )
    assert out == {}
    assert "sleeper:123" not in sd._LAST_GOOD_DIV_MAP


def test_divisionless_league_unaffected():
    # No divisions ever: empty map, nothing cached, no fallback.
    out = roster_division_map(
        _rosters_missing_divisions(),
        league_key="sleeper:999",
        settings={"divisions": 1},
    )
    assert out == {}
    assert "sleeper:999" not in sd._LAST_GOOD_DIV_MAP


def test_no_league_key_keeps_old_behavior():
    out = roster_division_map(_rosters_with_divisions())
    assert out == {1: 1, 2: 1, 3: 2, 4: 2}
    assert sd._LAST_GOOD_DIV_MAP == {}
    # And a hiccup without a key still returns empty (no fallback possible).
    assert roster_division_map(_rosters_missing_divisions()) == {}


def test_div_map_for_ctx_derives_key_and_falls_back():
    ctx = {
        "platform": "sleeper",
        "league_id": "123",
        "league_settings": {"divisions": 2},
        "rosters": _rosters_with_divisions(),
    }
    assert div_map_for_ctx(ctx) == {1: 1, 2: 1, 3: 2, 4: 2}
    ctx["rosters"] = _rosters_missing_divisions()
    assert div_map_for_ctx(ctx) == {1: 1, 2: 1, 3: 2, 4: 2}


def test_resolve_divisions_uses_fallback():
    ctx = {
        "platform": "sleeper",
        "league_id": "123",
        "league_settings": {"divisions": 2},
        "league": {"settings": {"divisions": 2}, "metadata": {}},
        "rosters": _rosters_with_divisions(),
    }
    info = sd.resolve_divisions(ctx)
    assert info is not None and info["by_rid"] == {1: 1, 2: 1, 3: 2, 4: 2}
    ctx["rosters"] = _rosters_missing_divisions()
    info = sd.resolve_divisions(ctx)
    assert info is not None and info["by_rid"] == {1: 1, 2: 1, 3: 2, 4: 2}
