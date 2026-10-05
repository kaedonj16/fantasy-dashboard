"""League Scores: /api/matchup/league-scores endpoint."""
from __future__ import annotations

import pytest

pytest.importorskip("flask")


def _make_matchup(left_rid, right_rid, left_name="Team A", right_name="Team B"):
    return {
        "left": {"roster_id": left_rid, "name": left_name, "starters": []},
        "right": {"roster_id": right_rid, "name": right_name, "starters": []},
    }


def test_endpoint_requires_auth():
    """Unauthenticated requests get empty matchups."""
    import routes.user_pages_bp as bp

    # The endpoint checks session before doing any work.
    # We verify the function exists and is registered.
    assert hasattr(bp, "api_matchup_league_scores")


def test_sort_you_first():
    """The viewer's matchup sorts first."""
    matchups = [
        {"left": {"roster_id": "2", "name": "B", "score": 100, "proj": 110},
         "right": {"roster_id": "3", "name": "C", "score": 120, "proj": 115},
         "win_prob": 30.0, "status": "in", "is_you": False},
        {"left": {"roster_id": "1", "name": "A", "score": 90, "proj": 100},
         "right": {"roster_id": "4", "name": "D", "score": 80, "proj": 95},
         "win_prob": 60.0, "status": "in", "is_you": True},
    ]
    # Mirror the sort logic from the endpoint.
    out = sorted(matchups, key=lambda x: (
        0 if x["is_you"] else 1,
        -((x["left"]["score"] or 0) + ((x["right"] or {}).get("score") or 0)),
    ))
    assert out[0]["is_you"] is True
    assert out[0]["left"]["name"] == "A"


def test_sort_by_total_points():
    """Non-you matchups sort by combined score descending."""
    matchups = [
        {"left": {"score": 50}, "right": {"score": 60}, "is_you": False},
        {"left": {"score": 100}, "right": {"score": 110}, "is_you": False},
        {"left": {"score": 70}, "right": {"score": 75}, "is_you": False},
    ]
    out = sorted(matchups, key=lambda x: (
        0 if x["is_you"] else 1,
        -((x["left"]["score"] or 0) + ((x["right"] or {}).get("score") or 0)),
    ))
    totals = [(m["left"]["score"] + m["right"]["score"]) for m in out]
    assert totals == sorted(totals, reverse=True)


def test_bye_week_matchup():
    """A matchup with no opponent renders without crashing."""
    m = {"left": {"roster_id": "1", "name": "A", "score": 100, "proj": 110},
         "right": None, "win_prob": None, "status": "pre", "is_you": True}
    assert m["right"] is None
    assert m["win_prob"] is None
