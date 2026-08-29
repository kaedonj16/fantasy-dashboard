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


def test_yahoo_does_not_treat_head_scoring_type_as_standard_ppr():
    # Competition format "head" must not zero out an explicit PPR modifier.
    settings = {
        "stat_modifiers": {"stats": [{"stat": {"stat_id": 11, "value": "1"}}]},
        "roster_positions": [{"position": "QB", "count": 1}],
    }
    scoring = yahoo_api._yahoo_scoring_settings({"scoring_type": "head"}, settings)
    assert scoring["rec"] == 1.0


def test_mfl_positions_ignore_numeric_roster_size():
    provider = mfl_api.MFLProvider
    assert provider._positions({"id": "1", "rosterSize": "20"}) == []
    assert provider._positions({"id": "1", "starters": "QB,RB,WR,TE"}) == [
        "QB", "RB", "WR", "TE",
    ]
    # rosterSize must not contaminate a missing starters field into ["20"].
    assert "20" not in provider._positions({"id": "1", "rosterSize": 20, "starters": ""})
