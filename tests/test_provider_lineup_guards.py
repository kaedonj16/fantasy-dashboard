"""Guards against silent zero Proj% / flat odds from empty provider slots.

Sibling failure modes of the ESPN roster_slots bug: Yahoo reading settings from
the wrong league node, MFL treating rosterSize as starters, and empty
roster_positions collapsing optimal-lineup math to 0 PPG.
"""
from __future__ import annotations

import pytest

pytest.importorskip("espn_api")
pytest.importorskip("flask")

from dashboard_services.providers import mfl_api, yahoo_api
from data_building.simulate_playoff_odds import _position_aware_lineup


def test_position_aware_lineup_wrrb_flex_excludes_te():
    ppg = {
        "rb1": {"ppg": 16.0, "pos": "RB"},
        "wr1": {"ppg": 14.0, "pos": "WR"},
        "te1": {"ppg": 20.0, "pos": "TE"},
        "rb2": {"ppg": 8.0, "pos": "RB"},
    }
    pos = {k: v["pos"] for k, v in ppg.items()}
    flex_avg, flex_starters = _position_aware_lineup(
        list(ppg), ppg, pos, ["RB", "WR", "FLEX"],
    )
    wrrb_avg, wrrb_starters = _position_aware_lineup(
        list(ppg), ppg, pos, ["RB", "WR", "WRRB_FLEX"],
    )
    assert ("TE", 20.0) in flex_starters
    assert ("TE", 20.0) not in wrrb_starters
    assert wrrb_avg < flex_avg


def test_empty_roster_positions_use_default_lineup_instead_of_zero_ppg():
    ppg = {
        "qb": {"ppg": 22.0, "pos": "QB"},
        "rb1": {"ppg": 16.0, "pos": "RB"},
        "rb2": {"ppg": 12.0, "pos": "RB"},
        "wr1": {"ppg": 14.0, "pos": "WR"},
        "wr2": {"ppg": 11.0, "pos": "WR"},
        "te": {"ppg": 9.0, "pos": "TE"},
    }
    pos = {k: v["pos"] for k, v in ppg.items()}
    avg, starters = _position_aware_lineup(list(ppg), ppg, pos, [])
    assert avg > 0
    assert {p for p, _ in starters} >= {"QB", "RB", "WR", "TE"}


def test_yahoo_settings_come_from_league_index_one_not_meta():
    lg = [
        {"league_key": "449.l.1", "num_teams": "10", "scoring_type": "head"},
        {"settings": [{
            "num_playoff_teams": "4",
            "playoff_start_week": "15",
            "roster_positions": [
                {"position": "QB", "count": 1},
                {"position": "RB", "count": 2},
                {"position": "WR", "count": 2},
                {"position": "TE", "count": 1},
                {"position": "W/R/T", "count": 1},
                {"position": "BN", "count": 6},
            ],
            "stat_modifiers": {"stats": [
                {"stat": {"stat_id": 11, "value": "1.0"}},
                {"stat": {"stat_id": 5, "value": "6"}},
            ]},
        }]},
    ]
    settings = yahoo_api._yahoo_settings_dict(lg)
    slots = yahoo_api._yahoo_roster_positions(settings)
    scoring = yahoo_api._yahoo_scoring_settings(lg[0], settings)
    assert slots.count("QB") == 1
    assert slots.count("RB") == 2
    assert slots.count("FLEX") == 1
    assert scoring["rec"] == 1.0
    assert scoring["pass_td"] == 6.0


def test_yahoo_restricted_flex_slots_stay_distinct():
    slots = yahoo_api._yahoo_roster_positions({
        "roster_positions": [
            {"position": "QB", "count": 1},
            {"position": "W/R", "count": 1},
            {"position": "W/T", "count": 1},
            {"position": "R/T", "count": 1},
        ],
    })
    assert slots.count("RB_WR") == 1
    assert slots.count("WR_TE") == 1
    assert slots.count("RB_TE") == 1
    assert slots.count("FLEX") == 0
    assert "W/T" not in slots
    assert "W/R" not in slots


def test_yahoo_get_league_globals_reads_nested_settings(monkeypatch):
    payload = {
        "fantasy_content": {
            "league": [
                {"league_key": "449.l.99", "num_teams": "12", "scoring_type": "head"},
                {"settings": [{
                    "num_playoff_teams": "6",
                    "playoff_start_week": "14",
                    "roster_positions": {"roster_position": [
                        {"position": "QB", "count": "1"},
                        {"position": "RB", "count": "2"},
                        {"position": "WR", "count": "2"},
                        {"position": "TE", "count": "1"},
                        {"position": "Q/W/R/T", "count": "1"},
                    ]},
                    "stat_modifiers": {"stats": {
                        "0": {"stat": {"stat_id": "11", "value": "0.5"}},
                        "1": {"stat": {"stat_id": "5", "value": "4"}},
                        "count": 2,
                    }},
                }]},
            ]
        }
    }
    monkeypatch.setattr(yahoo_api, "_yahoo_get", lambda *a, **k: payload)
    monkeypatch.setattr(yahoo_api, "_league_key", lambda lid: f"449.l.{lid}")
    out = yahoo_api.get_league_globals(2026, "99", access_token="tok")
    assert out["roster_positions"].count("SUPER_FLEX") == 1
    assert out["scoring_settings"]["rec"] == 0.5
    assert out["league_settings"]["playoff_week_start"] == 14
    assert out["league_settings"]["playoff_teams"] == 6


def test_yahoo_get_league_globals_reads_count_keyed_league(monkeypatch):
    payload = {
        "fantasy_content": {
            "league": {
                "count": 2,
                "0": {"league_key": "470.l.99", "num_teams": "10", "scoring_type": "head"},
                "1": {"settings": [{
                    "num_playoff_teams": "4",
                    "playoff_start_week": "15",
                    "roster_positions": [
                        {"position": "QB", "count": 1},
                        {"position": "RB", "count": 2},
                    ],
                    "stat_modifiers": {"stats": [
                        {"stat": {"stat_id": 11, "value": "1.0"}},
                    ]},
                }]},
            }
        }
    }
    monkeypatch.setattr(yahoo_api, "_yahoo_get", lambda *a, **k: payload)
    out = yahoo_api.get_league_globals(2026, "99", access_token="tok")
    assert out["roster_positions"].count("QB") == 1
    assert out["roster_positions"].count("RB") == 2
    assert out["scoring_settings"]["rec"] == 1.0
    assert out["league_settings"]["num_teams"] == 10


def test_yahoo_does_not_treat_head_scoring_type_as_standard_ppr():
    # Competition format "head" must not zero out an explicit PPR modifier.
    settings = {
        "stat_modifiers": {"stats": [{"stat": {"stat_id": 11, "value": "1"}}]},
        "roster_positions": [{"position": "QB", "count": 1}],
    }
    scoring = yahoo_api._yahoo_scoring_settings({"scoring_type": "head"}, settings)
    assert scoring["rec"] == 1.0


def test_yahoo_standard_reception_modifier_stays_zero():
    settings = {
        "stat_modifiers": {"stats": [{"stat": {"stat_id": 11, "value": "0"}}]},
    }
    scoring = yahoo_api._yahoo_scoring_settings({"scoring_type": "head"}, settings)
    assert scoring["rec"] == 0.0


def test_yahoo_300_yard_bonus_does_not_overwrite_per_yard_rate():
    settings = {
        "stat_modifiers": {"stats": [
            {"stat": {"stat_id": 4, "value": "0.04"}},
            {"stat": {
                "stat_id": 4, "value": "3",
                "bonuses": {"bonus": [{"target": "300", "points": "3"}]},
            }},
            {"stat": {"stat_id": 11, "value": "0"}},
        ]},
    }
    scoring = yahoo_api._yahoo_scoring_settings({"scoring_type": "head"}, settings)
    assert scoring["pass_yd"] == 0.04
    assert scoring["rec"] == 0.0


def test_yahoo_standard_weekly_proj_is_not_one_point_per_yard():
    from utils.fantasy_scoring import projection_points
    from utils.league_scoring import normalize_league_scoring

    settings = {
        "stat_modifiers": {"stats": [
            {"stat": {"stat_id": 4, "value": "0.04"}},
            {"stat": {
                "stat_id": 4, "value": "3",
                "bonuses": {"bonus": [{"target": "300", "points": "3"}]},
            }},
            {"stat": {"stat_id": 5, "value": "4"}},
            {"stat": {"stat_id": 6, "value": "-2"}},
            {"stat": {"stat_id": 9, "value": "0.1"}},
            {"stat": {"stat_id": 10, "value": "6"}},
            {"stat": {"stat_id": 11, "value": "0"}},
            {"stat": {"stat_id": 12, "value": "0.1"}},
            {"stat": {"stat_id": 18, "value": "-2"}},
        ]},
    }
    ss = normalize_league_scoring(
        "yahoo", yahoo_api._yahoo_scoring_settings({"scoring_type": "head"}, settings),
    )
    allen = projection_points(
        {"raw_stats": {
            "pass_yd": 235.21, "pass_td": 1.8, "pass_int": 0.6,
            "rush_yd": 26.3, "rush_td": 0.55, "fum_lost": 0.2,
        }},
        ss, "QB",
    )
    assert ss["rec"] == 0.0
    assert ss["pointsPerReception"] == 0.0
    assert allen < 40
    assert allen != pytest.approx(268.9, abs=1)


def test_mfl_positions_ignore_numeric_roster_size():
    provider = mfl_api.MFLProvider
    assert provider._positions({"id": "1", "rosterSize": "20"}) == []
    assert provider._positions({"id": "1", "starters": "QB,RB,WR,TE"}) == [
        "QB", "RB", "WR", "TE",
    ]
    # rosterSize must not contaminate a missing starters field into ["20"].
    assert "20" not in provider._positions({"id": "1", "rosterSize": 20, "starters": ""})
    assert provider._positions({"id": "1", "starters": "QB,RB,WR,TE,RB+WR,WR+TE"}) == [
        "QB", "RB", "WR", "TE", "RB_WR", "WR_TE",
    ]


def test_fleaflicker_positions_do_not_leave_slash_labels():
    """Sibling of the ESPN/Yahoo/MFL slot bugs: raw Fleaflicker flex/DST labels
    are invisible to draft-room and count_roster_positions."""
    from dashboard_services.providers.fleaflicker_api import FleaflickerProvider

    slots = FleaflickerProvider._positions({
        "rosterPositions": [
            {"label": "QB", "group": "START", "start": 1},
            {"label": "RB/WR/TE", "group": "START", "start": 1},
            {"label": "QB/RB/WR/TE", "group": "START", "start": 1},
            {"label": "D/ST", "group": "START", "start": 1},
        ],
    })
    assert slots == ["QB", "FLEX", "SUPER_FLEX", "DEF"]
    assert "RB/WR/TE" not in slots
    assert "D/ST" not in slots
