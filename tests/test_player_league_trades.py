"""Unit tests for player league trade helpers + pick resolution."""
from __future__ import annotations

from dashboard_services.player_league_trades import (
    _format_trade_sides,
    _pick_label,
    attach_drafted_players_to_trade_db_assets,
    resolve_pick_asset,
)


def test_pick_label_with_slot():
    assert _pick_label(2026, 1, 4) == "2026 Pick 1.04"


def test_pick_label_without_slot_uses_order():
    assert _pick_label(2026, 1, None, "early") == "2026 Round 1 (early)"


def test_resolve_pick_asset_attaches_drafted_player():
    resolution = {
        (2026, 1, 4): {
            "type": "player",
            "player_id": "999",
            "name": "Tetairoa McMillan",
            "position": "WR",
            "is_focus": False,
        }
    }
    asset = resolve_pick_asset(
        pick_season=2026,
        pick_round=1,
        pick_slot=4,
        resolution_map=resolution,
    )
    assert asset["type"] == "pick"
    assert asset["drafted_player"]["name"] == "Tetairoa McMillan"
    assert "→ Tetairoa McMillan" in asset["name"]
    assert asset["name"].startswith("2026 Pick 1.04")


def test_resolve_pick_asset_undrafted_keeps_pick_label():
    asset = resolve_pick_asset(
        pick_season=2027,
        pick_round=1,
        pick_slot=6,
        resolution_map={},
    )
    assert asset["name"] == "2027 Pick 1.06"
    assert "drafted_player" not in asset


def test_format_trade_sides_focus_and_teams():
    txn = {
        "roster_ids": [1, 2],
        "adds": {"111": 2, "222": 1},  # 111 (Chase) to team 2; 222 to team 1
        "drops": {"111": 1, "222": 2},
        "draft_picks": [
            {
                "season": "2026",
                "round": 1,
                "roster_id": 3,
                "owner_id": 1,  # team 1 receives the pick
                "previous_owner_id": 2,
            }
        ],
    }
    roster_names = {"1": "Film Room", "2": "Hoodie's Heroes"}
    players = {
        "111": {"name": "Chase Brown", "pos": "RB"},
        "222": {"name": "Depth Piece", "pos": "WR"},
    }
    slot_map = {("2026", "3"): 4}
    resolution = {
        (2026, 1, 4): {
            "player_id": "999",
            "name": "Tetairoa McMillan",
            "position": "WR",
        }
    }

    sides = _format_trade_sides(
        txn,
        focus_pid="111",
        roster_names=roster_names,
        players_index=players,
        slot_map=slot_map,
        resolution_map=resolution,
    )
    assert sides is not None
    side_a, side_b = sides
    # Receiver of Chase Brown
    assert side_a["team_name"] == "Hoodie's Heroes"
    assert any(a.get("is_focus") for a in side_a["assets"])
    # Sender received the 2026 1st which drafted McMillan
    assert side_b["team_name"] == "Film Room"
    pick = next(a for a in side_b["assets"] if a.get("type") == "pick")
    assert pick["drafted_player"]["name"] == "Tetairoa McMillan"
    assert "1.04" in pick["name"]


def test_attach_drafted_players_to_trade_db_assets(monkeypatch):
    trades = [
        {
            "league_id": "L1",
            "side_a": [{"type": "player", "name": "Chase Brown", "player_id": "111", "is_focus": True}],
            "side_b": [{
                "type": "pick",
                "name": "2026 Pick 1.04",
                "pick_season": 2026,
                "pick_round": 1,
                "pick_slot": 4,
            }],
        }
    ]

    def fake_map(platform, league_id, seasons=None):
        assert league_id == "L1"
        return {
            (2026, 1, 4): {
                "player_id": "999",
                "name": "Tetairoa McMillan",
                "position": "WR",
            }
        }

    monkeypatch.setattr(
        "dashboard_services.player_league_trades.build_draft_resolution_map",
        fake_map,
    )
    out = attach_drafted_players_to_trade_db_assets(trades, platform="sleeper")
    pick = out[0]["side_b"][0]
    assert pick["drafted_player"]["name"] == "Tetairoa McMillan"
    assert "→ Tetairoa McMillan" in pick["name"]
