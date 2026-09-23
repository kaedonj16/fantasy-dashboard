"""Regression coverage for Fleaflicker's official weekly lineup contract.

Shapes are sanitized examples of the already-supported FetchLeagueBoxscore
group/slot response; this test does not imply live upstream verification.
"""
from dashboard_services.providers.fleaflicker_api import FleaflickerProvider


def _player(fid, name, points=...):
    row = {"proPlayer": {"id": fid, "nameFull": name, "position": "WR"}}
    if points is not ...:
        row["viewingActualPoints"] = points
    return row


def _slot(group, home=None, label="WR"):
    row = {"position": {"group": group, "label": label}}
    if home is not None:
        row["home"] = home
    return row


def test_weekly_boxscore_preserves_slots_bench_scores_and_roster_groups(monkeypatch):
    provider = FleaflickerProvider()
    scoreboard = {"games": [{
        "id": 10, "home": {"id": 1}, "away": {"id": 2},
        "homeScore": {"score": {"value": 21}},
        "awayScore": {"score": {"value": 9}},
    }]}
    boxscore = {"lineups": [
        {"group": "START", "slots": [
            _slot("START", _player(1, "First", {"value": 10}), "WR"),
            _slot("START", None, "WR"),
            _slot("START", _player(999, "Unresolved", {"value": 3}), "FLEX"),
            _slot("START", _player(2, "Last", {"value": -2}), "SUPER_FLEX"),
        ]},
        {"group": "BENCH", "slots": [
            _slot("BENCH", _player(11646, "Jalen Coker", {"value": 7.25}), "BN"),
            _slot("BENCH", _player(3, "Zero", {"value": 0}), "BN"),
            _slot("BENCH", _player(4, "Unknown Points", None), "BN"),
            _slot("BENCH", None, "BN"),
        ]},
        {"group": "INJURED", "slots": [_slot("INJURED", _player(5, "IR"), "IR")]},
        {"group": "TAXI", "slots": [_slot("TAXI", _player(6, "Taxi"), "TAXI")]},
    ]}

    def call(method, *_a, **_kw):
        return scoreboard if method == "FetchLeagueScoreboard" else boxscore

    monkeypatch.setattr(provider, "_call", call)
    monkeypatch.setattr(provider, "_canonical_map", lambda *_a, **_kw: {
        "1": "one", "2": "two", "11646": "11646", "3": "zero",
        "4": "unknown", "5": "ir", "6": "taxi",
    })
    monkeypatch.setattr(provider, "_build_name_index", lambda: {})
    home = provider.get_matchups("1", 2026, 1)[0]

    assert home["starters"] == ["one", "0", "0", "two"]
    assert home["starters_points"] == [10.0, None, None, -2.0]
    assert home["bench_slots"] == ["11646", "zero", "unknown", "0"]
    assert home["bench"] == ["11646", "zero", "unknown"]
    assert home["reserve"] == ["ir"] and home["taxi"] == ["taxi"]
    assert home["players"] == ["one", "two", "11646", "zero", "unknown", "ir", "taxi"]
    assert home["players_points"] == {"one": 10.0, "two": -2.0, "11646": 7.25, "zero": 0.0}
    assert "unknown" in home["metadata"]["unavailable_points"]
    assert home["metadata"]["slot_diagnostics"][0]["status"] == "occupied_unresolved"
    assert sum(p for p in home["starters_points"] if p is not None) == 8.0


def test_coker_can_be_an_official_scoring_starter(monkeypatch):
    provider = FleaflickerProvider()
    lineup = provider._starters_from_boxscore(
        [{"group": "START", "slots": [_slot("START", _player(11646, "Jalen Coker", 12.5))]}],
        "home", {"11646": "11646"}, {},
    )
    assert lineup["starters"] == ["11646"]
    assert lineup["players_points"] == {"11646": 12.5}


def test_failed_boxscore_does_not_skip_next_game(monkeypatch):
    provider = FleaflickerProvider()
    games = {"games": [
        {"id": 10, "home": {"id": 1}, "away": {"id": 2}},
        {"id": 20, "home": {"id": 3}, "away": {"id": 4}},
    ]}
    called = []
    def call(method, *_a, **kw):
        if method == "FetchLeagueScoreboard":
            return games
        called.append(kw["fantasy_game_id"])
        if kw["fantasy_game_id"] == 10:
            raise RuntimeError("one malformed game")
        return {"lineups": [{"group": "START", "slots": [_slot("START", _player(1, "OK", 4))]}]}
    monkeypatch.setattr(provider, "_call", call)
    monkeypatch.setattr(provider, "_canonical_map", lambda *_a, **_kw: {"1": "ok"})
    monkeypatch.setattr(provider, "_build_name_index", lambda: {})
    rows = provider.get_matchups("1", 2026, 1)
    assert called == [10, 20]
    assert next(r for r in rows if r["roster_id"] == 3)["starters"] == ["ok"]
    assert next(r for r in rows if r["roster_id"] == 1)["metadata"]["lineup_state"] == "unavailable"



def test_unresolved_boxscore_slots_are_not_historical():
    """Fleaflicker '0' placeholders must not count as a historical lineup.

    When the boxscore can't resolve players, starters is all "0"s. That must
    not set lineup_is_historical, or Bench Gems / Missed Opportunities render
    empty sections instead of the unavailable message.
    """
    from dashboard_services.recap_calculations import build_lineup_analysis

    # Side with all-"0" starters (unresolved boxscore) but historical flag set
    # (simulating the old buggy behavior where "0"s passed the gate).
    matchups_by_week = {
        1: [{
            "left": {
                "roster_id": "1", "name": "Team A",
                "lineup_is_historical": True,
                "starters": [
                    {"pid": "0", "name": "Unknown", "pos": "WR", "pts": None},
                ],
                "bench": [],
            },
            "right": {
                "roster_id": "2", "name": "Team B",
                "lineup_is_historical": True,
                "starters": [
                    {"pid": "0", "name": "Unknown", "pos": "RB", "pts": None},
                ],
                "bench": [],
            },
        }]
    }
    result = build_lineup_analysis(matchups_by_week, 1, roster_positions=[])
    assert result["available"] is False
    assert "unavailable" in result["reason"].lower()
