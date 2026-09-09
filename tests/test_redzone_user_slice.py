"""Unit tests for the My Leagues (user-scope) Redzone aggregation helpers.

``_redzone_fetch_user`` was refactored into two reusable pieces so the
all-at-once payload and the progressive NDJSON stream share identical logic:

    _redzone_user_portfolio(season)   -> the viewer's league list + identities
    _redzone_user_league_slice(...)   -> one league's namespaced viewer slice

These tests pin the per-league slice shape/namespacing and confirm the
aggregate still dedupes users and stitches slices together correctly, without
needing live providers or a signed-in session.
"""
import pytest

pytest.importorskip("flask")
pytest.importorskip("pandas")

import app  # noqa: E402


def _league_payload(lid):
    """A minimal _redzone_collect return for one league where owner 'u1' is the
    viewer (roster 1) facing 'u2' (roster 2) in matchup 10."""
    pids = {"LA": ("p1", "p2", "p3"), "LB": ("p4", "p5", "p6")}[lid]
    opp_user = {"LA": "u2", "LB": "u3"}[lid]
    return {
        "rosters": [
            {"roster_id": 1, "owner_id": "u1"},
            {"roster_id": 2, "owner_id": opp_user},
        ],
        "matchups": [
            {"roster_id": 1, "matchup_id": 10, "players": [pids[0], pids[1]], "points": 5},
            {"roster_id": 2, "matchup_id": 10, "players": [pids[2]], "points": 3},
        ],
        "users": [
            {"user_id": "u1", "display_name": "Me"},
            {"user_id": opp_user, "display_name": "Opp-" + lid},
        ],
        "player_info": {p: {"name": p, "pos": "WR"} for p in pids},
        "scoring": {"rec": 1.0, "league": lid},
    }


@pytest.fixture
def fake_collect(monkeypatch):
    monkeypatch.setattr(app, "_redzone_collect",
                        lambda plat, lid, season, week: _league_payload(lid))


def test_league_slice_namespaces_and_shapes(fake_collect):
    lg = {"platform": "sleeper", "league_id": "LA", "name": "League A", "season": 2025}
    s = app._redzone_user_league_slice(
        0, lg, 2025, 1, account_id=None, viewer_uid="u1",
        identities_by_platform={}, default_platform="sleeper",
    )
    assert s is not None
    # Both sides of the viewer's matchup, ids namespaced by portfolio index "0:".
    assert {m["matchup_id"] for m in s["matchups"]} == {"0:10"}
    assert {m["roster_id"] for m in s["matchups"]} == {"0:1", "0:2"}
    assert s["viewer_roster_id"] == "0:1"
    assert {r["roster_id"] for r in s["rosters"]} == {"0:1", "0:2"}
    # Every player on the viewer's pair maps back to this league for scoring.
    assert s["pid_league"] == {"p1": "LA", "p2": "LA", "p3": "LA"}
    assert s["scoring_by_league"] == {"LA": {"rec": 1.0, "league": "LA"}}
    assert s["leagues"] == [{"league_id": "LA", "name": "League A", "platform": "sleeper"}]
    assert s["matchups"][0]["league_name"] == "League A"


def test_league_slice_none_when_viewer_absent(fake_collect):
    lg = {"platform": "sleeper", "league_id": "LA", "name": "League A", "season": 2025}
    # A viewer id that matches no roster owner -> no slice.
    s = app._redzone_user_league_slice(
        0, lg, 2025, 1, account_id=None, viewer_uid="nobody",
        identities_by_platform={}, default_platform="sleeper",
    )
    assert s is None


def test_fetch_user_aggregates_slices(monkeypatch, fake_collect):
    portfolio = [
        {"platform": "sleeper", "league_id": "LA", "name": "League A", "season": 2025},
        {"platform": "sleeper", "league_id": "LB", "name": "League B", "season": 2025},
    ]
    monkeypatch.setattr(app, "_redzone_user_portfolio",
                        lambda season: (portfolio, {}, None, "u1"))

    out = app._redzone_fetch_user("sleeper", "LA", 2025, 1)

    assert out["scope"] == "user"
    # One viewer roster per league, namespaced by index.
    assert out["viewer_roster_ids"] == ["0:1", "1:1"]
    assert out["viewer_roster_id"] == "0:1"
    # Matchups from both leagues, distinct namespaced ids.
    assert {m["matchup_id"] for m in out["matchups"]} == {"0:10", "1:10"}
    assert {m["league_name"] for m in out["matchups"]} == {"League A", "League B"}
    # Users deduped across leagues (u1 appears in both, once in output).
    uids = [u["user_id"] for u in out["users"]]
    assert uids.count("u1") == 1
    assert set(uids) == {"u1", "u2", "u3"}
    # Per-league scoring + pid->league map cover both leagues.
    assert set(out["scoring_by_league"]) == {"LA", "LB"}
    assert out["pid_league"] == {
        "p1": "LA", "p2": "LA", "p4": "LB", "p5": "LB",
    }
    assert set(out["player_info"]) == {"p1", "p2", "p3", "p4", "p5", "p6"}
    assert out["leagues"] == [
        {"league_id": "LA", "name": "League A", "platform": "sleeper"},
        {"league_id": "LB", "name": "League B", "platform": "sleeper"},
    ]
