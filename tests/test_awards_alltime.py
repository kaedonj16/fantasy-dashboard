"""All-time League Awards: multi-season computation, caching, and rendering.

Covers compute_awards_alltime (history walk, season-capped fetch, top
player-week across seasons, 1h cache, never-raises contract) and the
alltime=True render path (subtitle + "Week N \u00b7 YYYY" details).
"""
from __future__ import annotations

import pytest

pytest.importorskip("pandas")

import pandas as pd

import dashboard_services.api as real_api
import dashboard_services.awards as aw


@pytest.fixture
def _fake_history(monkeypatch):
    """Two-season league: 2026 (current) + 2025 (prior)."""
    calls = {"users": 0, "matchups": 0}

    def fake_history(platform, league_id, season):
        return {2026: "LID2026", 2025: "LID2025"}

    def fake_get_users(platform, lid, s):
        calls["users"] += 1
        assert (lid, s) == ("LID2025", 2025)
        return [
            {"user_id": "u1", "display_name": "Old A", "username": "olda", "metadata": {}},
            {"user_id": "u2", "display_name": "Old B", "username": "oldb", "metadata": {}},
        ]

    def fake_get_rosters(platform, lid, s):
        return [
            {"roster_id": 1, "owner_id": "u1", "metadata": {}},
            {"roster_id": 2, "owner_id": "u2", "metadata": {}},
        ]

    def fake_get_matchups(platform, lid, w, s):
        calls["matchups"] += 1
        assert (lid, s) == ("LID2025", 2025)
        if w == 1:
            return [
                {"roster_id": "1", "points": 200.0, "matchup_id": 1,
                 "players_points": {"999": 41.5}},
                {"roster_id": "2", "points": 90.0, "matchup_id": 1,
                 "players_points": {}},
            ]
        return []

    monkeypatch.setattr(real_api, "build_league_history_map", fake_history)
    monkeypatch.setattr(aw, "get_users", fake_get_users)
    monkeypatch.setattr(aw, "get_rosters", fake_get_rosters)
    monkeypatch.setattr(aw, "get_matchups", fake_get_matchups)
    # Current-season player scan finds nothing in this fixture.
    monkeypatch.setattr(
        aw, "highest_single_game_points",
        lambda *a, **k: ["", 0.0, "", "", "", "", ""],
    )
    aw._ALLTIME_AWARDS_CACHE.clear()
    yield calls
    aw._ALLTIME_AWARDS_CACHE.clear()


@pytest.fixture
def _finalized_df():
    return pd.DataFrame([
        {"owner": "Team A", "week": 5, "points": 150.0, "points_against": 100.0},
        {"owner": "Team B", "week": 5, "points": 100.0, "points_against": 150.0},
    ])


def test_alltime_combines_seasons(_fake_history, _finalized_df):
    players_map = {"999": {"name": "Star", "position": "WR", "team": "CIN"}}
    d = aw.compute_awards_alltime(
        "sleeper", "LID2026", 2026, players_map, [], [], _finalized_df)
    assert d["highest_single_week"] == ("Old A", 1, 200.0, 2025)
    assert d["lowest_single_week"] == ("Old B", 1, 90.0, 2025)
    assert d["highest_player"] == (1, 41.5, "Star", "WR", "CIN", "Old A", "999", 2025)


def test_alltime_cache_hit_no_refetch(_fake_history, _finalized_df):
    aw.compute_awards_alltime("sleeper", "LID2026", 2026, {}, [], [], _finalized_df)
    before = dict(_fake_history)
    aw.compute_awards_alltime("sleeper", "LID2026", 2026, {}, [], [], _finalized_df)
    assert _fake_history == before


def test_alltime_render_labels_seasons(_fake_history, _finalized_df):
    d = aw.compute_awards_alltime("sleeper", "LID2026", 2026, {}, [], [], _finalized_df)
    out = aw.render_awards_section(d, alltime=True)
    assert "All-time league history" in out
    assert "Week 1 \u00b7 2025" in out


def test_alltime_default_render_unchanged():
    d = {"highest_single_week": ("Team Alpha", 3, 152.36)}
    out = aw.render_awards_section(d)
    assert "Season superlatives so far" in out
    assert "Week 3</div>" in out


def test_alltime_empty_history_returns_empty(monkeypatch, _finalized_df):
    monkeypatch.setattr(real_api, "build_league_history_map",
                        lambda platform, league_id, season: {})
    aw._ALLTIME_AWARDS_CACHE.clear()
    try:
        assert aw.compute_awards_alltime(
            "sleeper", "X", 2026, {}, [], [], _finalized_df) == {}
    finally:
        aw._ALLTIME_AWARDS_CACHE.clear()


def test_alltime_never_raises(monkeypatch, _finalized_df):
    def boom(*a, **k):
        raise RuntimeError("nope")

    monkeypatch.setattr(real_api, "build_league_history_map", boom)
    aw._ALLTIME_AWARDS_CACHE.clear()
    try:
        assert aw.compute_awards_alltime(
            "sleeper", "Y", 2026, {}, [], [], _finalized_df) == {}
    finally:
        aw._ALLTIME_AWARDS_CACHE.clear()
