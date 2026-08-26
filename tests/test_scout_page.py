"""Scout tab smoke test: player rows include projected PPG.

Does not import Flask. Live value-table lookup is stubbed.
"""
from __future__ import annotations

from dashboard_services.pages.scout_page import build_scout_body


def test_scout_unsigned_in_prompt():
    html = build_scout_body({"viewer": {}})
    assert "Sign in to view your scouting report" in html


def test_scout_renders_player_proj_ppg(monkeypatch):
    import dashboard_services.pages.scout_page as scout_page

    monkeypatch.setattr(scout_page, "_live_model_value_table", lambda: [])

    ctx = {
        "viewer": {"viewer_roster_id": "1"},
        "platform": "sleeper",
        "league_id": "L1",
        "season": 2026,
        "current_week": 3,
        "offseason_mode": False,
        "rosters": [
            {"roster_id": 1, "players": ["111"], "starters": ["111"]},
            {"roster_id": 2, "players": ["222"], "starters": ["222"]},
        ],
        "roster_map": {"1": "You", "2": "Rival FC"},
        "standings_map": {
            "2": {"wins": 3, "losses": 1, "pf": 412.4, "pa": 355.1},
        },
        "model_value_table": [
            {
                "id": "222",
                "name": "Rival Star",
                "position": "WR",
                "value": 8200,
                "team": "KC",
                "pos_rank_label": "WR12",
            }
        ],
        "players_index": {"222": {"name": "Rival Star", "pos": "WR", "team": "KC"}},
        "matchups_by_week": {
            3: [
                {
                    "team1": {"roster_id": 1, "starters": ["111"]},
                    "team2": {"roster_id": 2, "starters": ["222"], "pts_total": None},
                }
            ]
        },
        "statuses": {3: {"statuses": {}}},
        "proj_by_roster": {(3, "2"): 118.4},
        "proj_by_week": {3: {"222": 18.4}},
    }
    html = build_scout_body(ctx)
    assert "Rival Star" in html
    assert "18.4 PPG" in html
    assert "scout-ppg" in html
    assert "Rival FC" in html
