"""Playoff odds must switch to the published H2H schedule once it appears.

Until the platform posts pairings, the sim uses a deterministic round-robin
fallback. The moment real matchup_ids show up, those weeks replace the
fallback — and schedule fingerprints must change so caches rebuild instead of
serving round-robin odds until an unrelated roster change or TTL expiry.
"""
from __future__ import annotations

import pytest

from data_building import simulate_playoff_odds as spo


def _teams(n=4):
    return [
        {"roster_id": i, "name": f"T{i}", "wins": 0, "losses": 0, "ties": 0,
         "pf": 0.0, "avg": 100.0, "std": 15.0}
        for i in range(1, n + 1)
    ]


def test_resolve_schedule_uses_fallback_when_unpublished(monkeypatch):
    # Null matchup_id = schedule not decided yet (Sleeper preseason shape).
    def fake_get_matchups(platform, league_id, week, season):
        return [
            {"matchup_id": None, "roster_id": 1},
            {"matchup_id": None, "roster_id": 2},
            {"matchup_id": None, "roster_id": 3},
            {"matchup_id": None, "roster_id": 4},
        ]

    monkeypatch.setattr(
        "dashboard_services.platform_api.get_matchups", fake_get_matchups,
    )
    teams = _teams()
    weeks = [1, 2, 3]
    resolved = spo._resolve_schedule("sleeper", "lg1", 2026, weeks, teams, {})
    fallback = spo._round_robin_schedule(teams, weeks)
    assert resolved == fallback


def test_resolve_schedule_prefers_real_pairings_per_week(monkeypatch):
    # Week 1 published as 1v4 / 2v3 (not the round-robin default 1v2 / 3v4).
    # Week 2 still unpublished → fallback fills only that week.
    real_week1 = {
        1: [
            {"matchup_id": 1, "roster_id": 1},
            {"matchup_id": 1, "roster_id": 4},
            {"matchup_id": 2, "roster_id": 2},
            {"matchup_id": 2, "roster_id": 3},
        ],
    }

    def fake_get_matchups(platform, league_id, week, season):
        return real_week1.get(week, [
            {"matchup_id": None, "roster_id": rid} for rid in (1, 2, 3, 4)
        ])

    monkeypatch.setattr(
        "dashboard_services.platform_api.get_matchups", fake_get_matchups,
    )
    teams = _teams()
    weeks = [1, 2]
    resolved = spo._resolve_schedule("sleeper", "lg1", 2026, weeks, teams, {})
    assert set(tuple(sorted(p)) for p in resolved[1]) == {(1, 4), (2, 3)}
    fallback_w2 = spo._round_robin_schedule(teams, [2])[2]
    assert set(tuple(sorted(p)) for p in resolved[2]) == set(
        tuple(sorted(p)) for p in fallback_w2
    )


def test_resolve_schedule_fully_real_once_published(monkeypatch):
    published = {
        1: [(1, 4), (2, 3)],
        2: [(1, 3), (2, 4)],
    }

    def fake_get_matchups(platform, league_id, week, season):
        pairs = published.get(week) or []
        rows = []
        for mid, (a, b) in enumerate(pairs, start=1):
            rows.append({"matchup_id": mid, "roster_id": a})
            rows.append({"matchup_id": mid, "roster_id": b})
        return rows

    monkeypatch.setattr(
        "dashboard_services.platform_api.get_matchups", fake_get_matchups,
    )
    teams = _teams()
    weeks = [1, 2]
    resolved = spo._resolve_schedule("sleeper", "lg1", 2026, weeks, teams, {})
    assert set(tuple(sorted(p)) for p in resolved[1]) == {(1, 4), (2, 3)}
    assert set(tuple(sorted(p)) for p in resolved[2]) == {(1, 3), (2, 4)}
    # Must not mix in round-robin once every week is published.
    fallback = spo._round_robin_schedule(teams, weeks)
    assert set(tuple(sorted(p)) for p in resolved[1]) != set(
        tuple(sorted(p)) for p in fallback[1]
    )


def test_schedule_fingerprint_changes_when_slate_appears(monkeypatch):
    state = {"published": False}

    def fake_get_matchups(platform, league_id, week, season):
        if not state["published"]:
            return [{"matchup_id": None, "roster_id": rid} for rid in (1, 2, 3, 4)]
        return [
            {"matchup_id": 1, "roster_id": 1},
            {"matchup_id": 1, "roster_id": 4},
            {"matchup_id": 2, "roster_id": 2},
            {"matchup_id": 2, "roster_id": 3},
        ]

    monkeypatch.setattr(
        "dashboard_services.platform_api.get_matchups", fake_get_matchups,
    )
    weeks = [1, 2]
    before = spo._schedule_fingerprint("sleeper", "lg1", 2026, weeks)
    assert before == "fallback"
    state["published"] = True
    after = spo._schedule_fingerprint("sleeper", "lg1", 2026, weeks)
    assert after != "fallback"
    assert after != before


def test_playoff_schedule_sig_and_ctx_signature_track_publish(monkeypatch):
    state = {"published": False}

    def fake_get_matchups(platform, league_id, week, season):
        if not state["published"]:
            return [{"matchup_id": 0, "roster_id": rid} for rid in (1, 2)]
        return [
            {"matchup_id": 1, "roster_id": 1},
            {"matchup_id": 1, "roster_id": 2},
        ]

    monkeypatch.setattr(
        "dashboard_services.platform_api.get_matchups", fake_get_matchups,
    )
    ctx = {
        "league_id": "lg1",
        "season": 2026,
        "current_week": 0,
        "league_settings": {"playoff_week_start": 15, "playoff_teams": 6},
        "rosters": [
            {"roster_id": 1, "players": ["a"]},
            {"roster_id": 2, "players": ["b"]},
        ],
    }
    before_sched = spo.playoff_schedule_sig(ctx, "sleeper")
    before_ctx = spo._ctx_signature(ctx, "sleeper")
    assert before_sched == "fallback"

    state["published"] = True
    after_sched = spo.playoff_schedule_sig(ctx, "sleeper")
    after_ctx = spo._ctx_signature(ctx, "sleeper")
    assert after_sched != before_sched
    assert after_ctx != before_ctx


def test_fetch_skips_zero_and_null_matchup_ids(monkeypatch):
    def fake_get_matchups(platform, league_id, week, season):
        return [
            {"matchup_id": 0, "roster_id": 1},
            {"matchup_id": 0, "roster_id": 2},
            {"matchup_id": None, "roster_id": 3},
            {"matchup_id": "", "roster_id": 4},
        ]

    monkeypatch.setattr(
        "dashboard_services.platform_api.get_matchups", fake_get_matchups,
    )
    assert spo._fetch_remaining_schedule("sleeper", "lg1", 2026, [1]) == {}


def test_app_playoff_sim_sig_includes_schedule(monkeypatch):
    """The 1h odds cache must not ignore a newly published slate."""
    pytest.importorskip("flask")
    import app as app_mod

    state = {"published": False}

    def fake_get_matchups(platform, league_id, week, season):
        if not state["published"]:
            return []
        return [
            {"matchup_id": 1, "roster_id": 1},
            {"matchup_id": 1, "roster_id": 2},
        ]

    monkeypatch.setattr(
        "dashboard_services.platform_api.get_matchups", fake_get_matchups,
    )
    ctx = {
        "league_id": "lg9",
        "season": 2026,
        "current_week": 0,
        "league_settings": {"playoff_week_start": 15},
        "rosters": [
            {"roster_id": 1, "players": ["x"]},
            {"roster_id": 2, "players": ["y"]},
        ],
    }
    before = app_mod._playoff_sim_sig(ctx, "sleeper")
    state["published"] = True
    after = app_mod._playoff_sim_sig(ctx, "sleeper")
    assert before != after
    assert "|w0|" in before
