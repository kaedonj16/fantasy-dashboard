"""Yahoo scoreboard pairings must follow Yahoo's published matchups.

Yahoo's JSON nests matchups under scoreboard["0"]["matchups"]. The old
extractor only read scoreboard["matchups"], returned [], and the Season Hub
filled in a round-robin that did not match Yahoo.
"""
import pytest

pytest.importorskip("flask")
yahoo_api = pytest.importorskip("dashboard_services.providers.yahoo_api")


def _team_entry(team_id, points="0.00"):
    return {
        "team": [
            [
                {"team_key": f"461.l.99.t.{team_id}"},
                {"team_id": str(team_id)},
                {"name": f"Team {team_id}"},
            ],
            {"team_points": {"coverage_type": "week", "week": "1", "total": str(points)}},
        ]
    }


def _matchup_object(left, right, week=1, pts_l="10.0", pts_r="20.0"):
    return {
        "matchup": {
            "week": str(week),
            "teams": {
                "count": 2,
                "0": _team_entry(left, pts_l),
                "1": _team_entry(right, pts_r),
            },
        }
    }


def _scoreboard_payload(pairs, *, nested=True, league_as_dict=False, week=1):
    matchups = {"count": len(pairs)}
    for i, (a, b) in enumerate(pairs):
        matchups[str(i)] = _matchup_object(a, b, week=week)
    scoreboard = {"week": str(week)}
    if nested:
        scoreboard["0"] = {"matchups": matchups}
    else:
        scoreboard["matchups"] = matchups
    meta = {"league_key": "461.l.99", "current_week": str(week)}
    child = {"scoreboard": scoreboard}
    if league_as_dict:
        league = {"0": meta, "1": child, "count": 2}
    else:
        league = [meta, child]
    return {"fantasy_content": {"league": league}}


def _pairings(rows):
    by_mid = {}
    for row in rows:
        by_mid.setdefault(row["matchup_id"], []).append(row["roster_id"])
    return [tuple(sorted(rids)) for rids in by_mid.values()]


def _run_get_matchups(monkeypatch, payload, week=1):
    monkeypatch.setattr(yahoo_api, "_yahoo_get", lambda *a, **k: payload)
    monkeypatch.setattr(yahoo_api, "_league_key_for_season", lambda *a, **k: "461.l.99")
    return yahoo_api.get_matchups(2026, "99", week, "tok")


def test_nested_scoreboard_keeps_yahoo_pairings_not_round_robin(monkeypatch):
    # Synthetic week-1 for teams 1-4 is 1v2 and 3v4. Yahoo's real slate is 1v4, 2v3.
    payload = _scoreboard_payload([(1, 4), (2, 3)], nested=True)
    rows = _run_get_matchups(monkeypatch, payload)
    assert _pairings(rows) == [(1, 4), (2, 3)]
    assert len(rows) == 4
    pts = {(r["matchup_id"], r["roster_id"]): r["points"] for r in rows}
    assert pts[(1, 1)] == 10.0
    assert pts[(1, 4)] == 20.0


def test_flat_scoreboard_matchups_still_parse(monkeypatch):
    payload = _scoreboard_payload([(1, 3), (2, 4)], nested=False)
    rows = _run_get_matchups(monkeypatch, payload)
    assert _pairings(rows) == [(1, 3), (2, 4)]


def test_dict_shaped_league_scoreboard(monkeypatch):
    payload = _scoreboard_payload([(5, 8), (6, 7)], nested=True, league_as_dict=True)
    rows = _run_get_matchups(monkeypatch, payload)
    assert _pairings(rows) == [(5, 8), (6, 7)]


def test_list_wrapped_matchup_node(monkeypatch):
    payload = _scoreboard_payload([(1, 2)], nested=True)
    payload["fantasy_content"]["league"][1]["scoreboard"]["0"]["matchups"]["0"] = {
        "matchup": [
            {"week": "1", "status": "preevent"},
            {"teams": {
                "count": 2,
                "0": _team_entry(9, "1.0"),
                "1": _team_entry(10, "2.0"),
            }},
        ]
    }
    rows = _run_get_matchups(monkeypatch, payload)
    assert _pairings(rows) == [(9, 10)]


def test_bare_team_array_rows(monkeypatch):
    payload = _scoreboard_payload([(1, 2)], nested=True)
    teams = {
        "count": 2,
        "0": _team_entry(3, "5")["team"],
        "1": _team_entry(6, "7")["team"],
    }
    payload["fantasy_content"]["league"][1]["scoreboard"]["0"]["matchups"]["0"]["matchup"]["teams"] = teams
    rows = _run_get_matchups(monkeypatch, payload)
    assert _pairings(rows) == [(3, 6)]


def test_teams_nested_under_numeric_wrapper(monkeypatch):
    """Yahoo's XML→JSON converter wraps <teams> as matchup["0"]["teams"]."""
    payload = _scoreboard_payload([(1, 2)], nested=True)
    payload["fantasy_content"]["league"][1]["scoreboard"]["0"]["matchups"]["0"] = {
        "matchup": [
            {"week": "1", "status": "preevent", "is_tied": 0},
            {"0": {"teams": {
                "count": 2,
                "0": _team_entry(1, "0.00"),
                "1": _team_entry(4, "0.00"),
            }}},
        ]
    }
    rows = _run_get_matchups(monkeypatch, payload)
    assert _pairings(rows) == [(1, 4)]


def test_matchup_dict_with_numeric_teams_wrapper(monkeypatch):
    payload = _scoreboard_payload([(1, 2)], nested=True)
    payload["fantasy_content"]["league"][1]["scoreboard"]["0"]["matchups"]["0"] = {
        "matchup": {
            "week": "1",
            "status": "preevent",
            "0": {"teams": {
                "count": 2,
                "0": _team_entry(2, "3.0"),
                "1": _team_entry(7, "4.0"),
            }},
        }
    }
    rows = _run_get_matchups(monkeypatch, payload)
    assert _pairings(rows) == [(2, 7)]
    pts = {(r["matchup_id"], r["roster_id"]): r["points"] for r in rows}
    assert pts[(1, 2)] == 3.0
    assert pts[(1, 7)] == 4.0
