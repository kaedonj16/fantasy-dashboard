"""Compact cheat-sheet player payload: skill-only, board columns, no junk."""
from pathlib import Path

from dashboard_services.league_players_board import slim_board_payload
from dashboard_services.pages.cheat_sheet_page import build_cheat_sheet_body


def test_slim_board_payload_keeps_skill_values_and_drops_the_rest():
    payload = {
        "historical_available": True,
        "market_vs_adp_available": True,
        "adp_source_options": {"redraft": [{"value": "consensus", "label": "Consensus"}]},
        "tier_thresholds": {"1qb": {"12": [100]}},
        "adp_columns": [{"value": "sleeper", "label": "Sleeper"}],
        "players": [
            {
                "id": "1",
                "name": "Starter WR",
                "position": "WR",
                "age": 24.1,
                "value": 800,
                "sf_value": 700,
                "redraft_value_1qb": 90,
                "redraft_value_sf": 80,
                "redraft_avg_pick": 12.4,
                "proj_ppg": 16.2,
                "projected_offense_rank": 7,
                "espnHeadshot": "https://example/x.png",
                "bDay": "2001-01-01",
                "ppg": 18.0,
                "vorp": 40,
                "projection": {"ppg": 16.2, "source": "sleeper", "cache_version": "x"},
                "market_vs_adp_1qb": -3,
                "sf_market_vs_adp": -5,
                "market_expected_adp": 9.0,
                "market_confidence": 0.8,
                "market_confidence_label": "High",
                "market_basis": "season_props",
                "historical": {"p_hit_pct": 37, "n": 40},
                "adp_by_source": {
                    "consensus": {
                        "redraft_avg_pick": 12.4,
                        "sf_redraft_avg_pick": 11.1,
                        "avg_pick": None,
                    },
                    "sleeper": {"redraft_avg_pick": 13.0},
                },
            },
            {
                "id": "pick",
                "name": "2026 1.01",
                "position": "PICK",
                "value": 999,
            },
            {
                "id": "0",
                "name": "Practice squad",
                "position": "RB",
                "value": 0,
                "redraft_value_1qb": 0,
            },
            {
                "id": "k",
                "name": "Kicker",
                "position": "K",
                "proj_ppg": 8.0,
            },
        ],
    }

    slim = slim_board_payload(payload, is_superflex=False)
    assert [p["id"] for p in slim["players"]] == ["1"]
    row = slim["players"][0]
    assert row["name"] == "Starter WR"
    assert row["proj_ppg"] == 16.2
    assert row["projected_offense_rank"] == 7
    assert row["redraft_avg_pick"] == 12.4
    assert row["market_vs_adp"] == -3
    assert row["historical"]["p_hit_pct"] == 37
    assert row["adp_by_source"]["consensus"]["redraft_avg_pick"] == 12.4
    assert "avg_pick" not in row["adp_by_source"]["consensus"]
    assert "espnHeadshot" not in row
    assert "projection" not in row
    assert "vorp" not in row
    assert "tier_thresholds" not in slim
    assert "adp_columns" not in slim
    assert slim["adp_source_options"]["redraft"][0]["value"] == "consensus"
    assert slim["market_vs_adp_available"] is True
    assert slim["historical_available"] is True

    sf = slim_board_payload(payload, is_superflex=True)
    assert sf["players"][0]["market_vs_adp"] == -5


def test_cheat_sheet_prefetches_the_board_view():
    body = build_cheat_sheet_body("league-123", 2026, "sleeper", is_superflex=True)
    script = (Path(__file__).parents[1] / "static" / "cheat_sheet.js").read_text(encoding="utf-8")

    assert "view=board" in body
    assert "window.__cheatPlayersP" in body
    assert "c.isSuperflex?'sf':'1qb'" in body
    assert "params = ['view=board']" in script
    assert "pending.url === url" in script
