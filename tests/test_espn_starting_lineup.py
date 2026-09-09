"""ESPN live starting lineups must come from weekly lineupSlotId, not a guess.

Start/Sit compares the viewer's actual ESPN starters to the computed optimal
lineup. If we miss those slots (bulk mRoster omits them; QB is id 0 which is
falsy in ``a or b`` chains), the advisor can only show the optimal set — which
will not match the ESPN fantasy app.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("espn_api")

from dashboard_services.providers import espn_api as E


def _player(**kwargs):
    defaults = dict(
        slot_position=None,
        slotPosition=None,
        lineupSlot=None,
        lineupSlotId=None,
        proTeam=None,
        proTeamId=None,
    )
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


def test_espn_slot_name_keeps_qb_slot_id_zero():
    # lineupSlotId 0 is QB. `0 or "BE"` used to throw every QB on the bench.
    assert E._espn_slot_name(_player(lineupSlotId=0)) == "QB"
    assert E._espn_is_starter_slot("QB")
    assert E._espn_is_starter_slot(E._espn_slot_name(_player(lineupSlotId=0)))


def test_espn_slot_name_maps_bench_and_flex_ids():
    assert E._espn_slot_name(_player(lineupSlotId=20)) == "BE"
    assert not E._espn_is_starter_slot("BE")
    assert E._espn_slot_name(_player(lineupSlotId=23)) == "RB/WR/TE"
    assert E._espn_is_starter_slot("RB/WR/TE")
    assert E._espn_is_reserve_slot(E._espn_slot_name(_player(lineupSlotId=21)))


def test_espn_slot_name_skips_blank_lineup_slot_to_reach_id():
    # espn-api sets lineupSlot="" when it misses the id; the numeric id is still
    # on the object / raw entry.
    p = _player(lineupSlot="", lineupSlotId=2)
    assert E._espn_slot_name(p) == "RB"
    # BoxPlayer defaults slot_position to "FA" when the id is missing on
    # construct — that placeholder must not hide a real QB id of 0.
    assert E._espn_slot_name(_player(slot_position="FA", lineupSlotId=0)) == "QB"


def test_espn_slot_name_reads_nested_player_pool_entry():
    entry = {
        "playerId": 123,
        "playerPoolEntry": {"id": 123, "lineupSlotId": 4},
    }
    assert E._espn_slot_name(entry) == "WR"


def test_espn_slot_name_blank_and_fa_are_not_starters():
    assert not E._espn_is_starter_slot("")
    assert not E._espn_is_starter_slot("FA")  # BoxPlayer default when id missing
    assert not E._espn_slot_is_assigned("")
    assert not E._espn_slot_is_assigned("FA")


def test_raw_espn_lineup_slot_ids_reads_top_level_and_nested():
    payload = {
        "teams": [{
            "id": 1,
            "roster": {"entries": [
                {"playerId": 11, "lineupSlotId": 0},
                {"playerId": 22, "playerPoolEntry": {"id": 22, "lineupSlotId": 20}},
            ]},
        }],
    }
    lg = SimpleNamespace(espn_request=SimpleNamespace(league_get=lambda params: payload))
    out = E._raw_espn_lineup_slot_ids(lg, week=1)
    assert out[11] == 0
    assert out[22] == 20


def test_get_rosters_detects_qb_and_leaves_bench(monkeypatch):
    qb = _player(playerId=1, lineupSlotId=0, lineupSlot="")
    rb = _player(playerId=2, lineupSlot="RB")
    bn = _player(playerId=3, lineupSlotId=20, lineupSlot="")
    ir = _player(playerId=4, lineupSlot="IR")
    team = SimpleNamespace(
        team_id=1, team_name="Mine", name="Mine",
        owners=[{"id": "{AAA}"}],
        wins=1, losses=0, ties=0, outcomes=["W"],
        points_for=10, points_against=5,
        roster=[qb, rb, bn, ir],
    )
    loaded = []

    def load_roster_week(week):
        loaded.append(week)

    lg = SimpleNamespace(
        teams=[team], current_week=1, scoringPeriodId=1,
        load_roster_week=load_roster_week,
    )
    monkeypatch.setattr(E, "_league", lambda season, league_id: lg)
    monkeypatch.setattr(E, "_espn_to_canon_cached", lambda: {"1": "qb1", "2": "rb1", "3": "bn1", "4": "ir1"})

    rosters = E.get_rosters(2026, "99")
    assert loaded == [1]
    r = rosters[0]
    assert r["starters"] == ["qb1", "rb1"]
    assert "bn1" not in r["starters"]
    assert r["reserve"] == ["ir1"]
    assert set(r["players"]) == {"qb1", "rb1", "bn1", "ir1"}


def test_get_rosters_uses_raw_mroster_when_slots_missing(monkeypatch):
    # Bulk roster often has players and no slot names. Week-scoped mRoster JSON
    # still carries lineupSlotId (including nested playerPoolEntry).
    skill = _player(playerId=3117251, lineupSlot="", lineupSlotId=None)
    bench = _player(playerId=4035, lineupSlot="", lineupSlotId=None)
    team = SimpleNamespace(
        team_id=1, team_name="Mine", name="Mine",
        owners=[{"id": "{AAA}"}],
        wins=0, losses=0, ties=0, outcomes=[],
        points_for=0, points_against=0,
        roster=[skill, bench],
    )
    raw = {
        "teams": [{
            "roster": {"entries": [
                {"playerId": 3117251, "lineupSlotId": 2},
                {"playerId": 4035, "playerPoolEntry": {"lineupSlotId": 20}},
            ]},
        }],
    }
    lg = SimpleNamespace(
        teams=[team], current_week=2,
        espn_request=SimpleNamespace(league_get=lambda params: raw),
    )
    monkeypatch.setattr(E, "_league", lambda season, league_id: lg)
    monkeypatch.setattr(E, "_espn_to_canon_cached", lambda: {"3117251": "4034", "4035": "bn1"})

    r = E.get_rosters(2026, "99")[0]
    assert r["starters"] == ["4034"]
    assert "bn1" not in r["starters"]
    assert "bn1" in r["players"]


def test_get_matchups_does_not_treat_bench_or_missing_as_starters(monkeypatch):
    qb = _player(playerId=1, slot_position="QB", points=20)
    be = _player(playerId=2, slot_position="BE", points=30)
    missing = _player(playerId=3, slot_position="FA", points=40)
    # Numeric 0 must still count as the QB starter on a box-score player.
    qb_id = _player(playerId=4, slot_position=None, lineupSlot="", lineupSlotId=0, points=12)

    home = SimpleNamespace(team_id=1)
    away = SimpleNamespace(team_id=2)
    bs = SimpleNamespace(
        home_team=home, away_team=away,
        home_lineup=[qb, be, missing],
        away_lineup=[qb_id, be],
        home_score=20, away_score=12,
    )
    monkeypatch.setattr(E, "_box_scores_cached", lambda season, league_id, week: [bs])
    monkeypatch.setattr(E, "_espn_to_canon_cached", lambda: {
        "1": "qb1", "2": "be1", "3": "fa1", "4": "qb2",
    })

    rows = E.get_matchups(2026, "99", 1)
    home_row, away_row = rows[0], rows[1]
    assert home_row["starters"] == ["qb1"]
    assert "be1" not in home_row["starters"]
    assert "fa1" not in home_row["starters"]
    assert away_row["starters"] == ["qb2"]
