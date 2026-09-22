"""Unit tests for player league trade helpers + pick resolution."""
from __future__ import annotations

from dashboard_services.player_league_trades import (
    _format_trade_sides,
    _pick_label,
    attach_drafted_players_to_trade_db_assets,
    resolve_pick_asset,
)


def test_resolution_uses_original_owner_and_actual_pick_number(monkeypatch):
    """A snake/custom selection's display slot comes from pick_no, not start slot."""
    import sys
    import types
    from dashboard_services import player_league_trades as plt

    api = types.ModuleType("dashboard_services.api")
    api.get_drafts = lambda league_id: [{"status": "complete", "season": "2026", "draft_id": "D1"}]
    api.get_draft = lambda draft_id: {
        "settings": {"teams": 12}, "slot_to_roster_id": {"8": 44},
    }
    api.get_draft_picks = lambda draft_id: [
        {"player_id": "love", "round": 2, "draft_slot": 8, "pick_no": 17},
    ]
    utils = types.ModuleType("utils.utils")
    utils.load_players_index = lambda: {"love": {"name": "Jeremiyah Love", "pos": "RB"}}
    monkeypatch.setitem(sys.modules, "dashboard_services.api", api)
    monkeypatch.setitem(sys.modules, "utils.utils", utils)

    out = plt.build_draft_resolution_map("sleeper", "L1")
    assert out[(2026, 2, "44")]["within_round"] == 5
    assert out[(2026, 2, "44")]["draft_id"] == "D1"


def test_multiple_same_season_drafts_are_ambiguous(monkeypatch):
    import sys
    import types
    from dashboard_services import player_league_trades as plt

    api = types.ModuleType("dashboard_services.api")
    api.get_drafts = lambda league_id: [
        {"status": "complete", "season": "2026", "draft_id": "D1"},
        {"status": "complete", "season": "2026", "draft_id": "D2"},
    ]
    api.get_draft = lambda draft_id: {"settings": {"teams": 12}, "slot_to_roster_id": {"1": 7}}
    api.get_draft_picks = lambda draft_id: [
        {"player_id": draft_id, "round": 1, "draft_slot": 1, "pick_no": 1},
    ]
    utils = types.ModuleType("utils.utils")
    utils.load_players_index = lambda: {}
    monkeypatch.setitem(sys.modules, "dashboard_services.api", api)
    monkeypatch.setitem(sys.modules, "utils.utils", utils)
    assert plt.build_draft_resolution_map("sleeper", "L1") == {}


def test_pick_label_with_slot():
    assert _pick_label(2026, 1, 4) == "2026 1.04"


def test_pick_label_without_slot_uses_order():
    assert _pick_label(2026, 1, None, "early") == "2026 1st (early)"


def test_resolve_pick_asset_attaches_drafted_player():
    resolution = {
        (2026, 1, "3"): {
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
        pick_roster_id="3",
        resolution_map=resolution,
    )
    assert asset["type"] == "pick"
    assert asset["drafted_player"]["name"] == "Tetairoa McMillan"
    assert "(Tetairoa McMillan)" in asset["name"]
    assert asset["name"].startswith("2026 1.04")


def test_resolve_pick_asset_undrafted_keeps_pick_label():
    asset = resolve_pick_asset(
        pick_season=2027,
        pick_round=1,
        pick_slot=6,
        resolution_map={},
    )
    assert asset["name"] == "2027 1.06"
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
        (2026, 1, "3"): {
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


def test_pick_linked_trade_uses_participants_and_complete_original_package():
    txn = {
        "roster_ids": [10, 20],
        "adds": {}, "drops": {},
        "draft_picks": [
            {"season": "2026", "round": 1, "roster_id": 1,
             "owner_id": 10, "previous_owner_id": 20},
            {"season": "2026", "round": 1, "roster_id": 5,
             "owner_id": 20, "previous_owner_id": 10},
            {"season": "2027", "round": 1, "roster_id": 10,
             "owner_id": 20, "previous_owner_id": 10},
        ],
    }
    resolutions = {
        (2026, 1, "1"): {"player_id": "love", "name": "Jeremiyah Love",
                           "position": "RB", "draft_id": "D26", "within_round": 1},
        (2026, 1, "5"): {"player_id": "other", "name": "Actual 1.05 Player",
                           "position": "WR", "draft_id": "D26", "within_round": 5},
    }
    sides = _format_trade_sides(
        txn, focus_pid="love", roster_names={"10": "Selected Team", "20": "Partner"},
        players_index={}, slot_map={}, resolution_map=resolutions,
    )
    assert sides is not None
    received, sent = sides
    assert received["direction"] == "Received"
    assert received["team_name"] == "Selected Team"
    assert received["assets"][0]["name"] == "2026 1.01 (Jeremiyah Love)"
    assert received["assets"][0]["via_draft_pick"] is True
    assert sent["direction"] == "Sent"
    assert {a["name"] for a in sent["assets"]} == {
        "2026 1.05 (Actual 1.05 Player)", "2027 1st",
    }


def test_attach_drafted_players_to_trade_db_assets(monkeypatch):
    trades = [
        {
            "league_id": "L1",
            "side_a": [{"type": "player", "name": "Chase Brown", "player_id": "111", "is_focus": True}],
            "side_b": [{
                "type": "pick",
                "name": "2026 1.04",
                "pick_season": 2026,
                "pick_round": 1,
                "pick_slot": 4,
                "pick_roster_id": "3",
            }],
        }
    ]

    def fake_map(platform, league_id, seasons=None):
        assert league_id == "L1"
        return {
            (2026, 1, "3"): {
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
    assert "(Tetairoa McMillan)" in pick["name"]


def _trade_txn(pid="111", other="222", txn_id="tx-1", ts=1_700_000_000_000):
    return {
        "type": "trade",
        "status": "complete",
        "transaction_id": txn_id,
        "adds": {pid: 2, other: 1},
        "drops": {pid: 1, other: 2},
        "draft_picks": [],
        "roster_ids": [1, 2],
        "status_updated": ts,
    }


def test_get_player_league_trades_walks_every_season_including_week_zero(monkeypatch):
    import sys
    import types
    from dashboard_services import player_league_trades as plt

    calls = []
    fake_api = types.ModuleType("dashboard_services.api")
    fake_api.build_league_history_map = lambda plat, lid, season: {
        2026: "L26", 2025: "L25", 2024: "L24",
    }
    fake_service = types.ModuleType("dashboard_services.service")

    def fake_tx(league_id, weeks, platform="sleeper", season=0):
        calls.append((str(league_id), int(season), list(weeks)))
        return {0: [_trade_txn(txn_id=f"tx-{season}", ts=1_700_000_000_000 + int(season))]}

    fake_service.get_transactions_by_week = fake_tx
    fake_utils = types.ModuleType("utils.utils")
    fake_utils.load_players_index = lambda: {
        "111": {"name": "Chase Brown", "pos": "RB"},
        "222": {"name": "Depth", "pos": "WR"},
    }
    monkeypatch.setitem(sys.modules, "dashboard_services.api", fake_api)
    monkeypatch.setitem(sys.modules, "dashboard_services.service", fake_service)
    monkeypatch.setitem(sys.modules, "utils.utils", fake_utils)
    monkeypatch.setattr(plt, "build_draft_resolution_map", lambda *a, **k: {})
    monkeypatch.setattr(plt, "_slot_map_for_league", lambda *a, **k: {})
    monkeypatch.setattr(plt, "_roster_names", lambda *a, **k: {"1": "Film Room", "2": "Hoodie's Heroes"})

    out = plt.get_player_league_trades(
        player_id="111", platform="sleeper", league_id="L26", season=2026, limit=20,
    )
    assert {c[0] for c in calls} == {"L26", "L25", "L24"}
    assert {c[1] for c in calls} == {2026, 2025, 2024}
    assert all(0 in c[2] for c in calls), "offseason week 0 must be fetched every season"
    assert out["total"] == 3
    assert {t["season"] for t in out["trades"]} == {2026, 2025, 2024}


def test_get_player_league_trades_dedupes_repeated_transaction_ids(monkeypatch):
    import sys
    import types
    from dashboard_services import player_league_trades as plt

    fake_api = types.ModuleType("dashboard_services.api")
    fake_api.build_league_history_map = lambda plat, lid, season: {2026: "L26"}
    fake_service = types.ModuleType("dashboard_services.service")
    same = _trade_txn(txn_id="yahoo-dup")
    fake_service.get_transactions_by_week = lambda league_id, weeks, platform="sleeper", season=0: {
        w: [same] for w in weeks
    }
    fake_utils = types.ModuleType("utils.utils")
    fake_utils.load_players_index = lambda: {
        "111": {"name": "Chase Brown", "pos": "RB"},
        "222": {"name": "Depth", "pos": "WR"},
    }
    monkeypatch.setitem(sys.modules, "dashboard_services.api", fake_api)
    monkeypatch.setitem(sys.modules, "dashboard_services.service", fake_service)
    monkeypatch.setitem(sys.modules, "utils.utils", fake_utils)
    monkeypatch.setattr(plt, "build_draft_resolution_map", lambda *a, **k: {})
    monkeypatch.setattr(plt, "_slot_map_for_league", lambda *a, **k: {})
    monkeypatch.setattr(plt, "_roster_names", lambda *a, **k: {"1": "A", "2": "B"})

    out = plt.get_player_league_trades(
        player_id="111", platform="yahoo", league_id="L26", season=2026, limit=20,
    )
    assert out["total"] == 1


def test_get_player_league_trades_includes_pick_link_before_limit(monkeypatch):
    import sys
    import types
    from dashboard_services import player_league_trades as plt

    pick_trade = {
        "type": "trade", "status": "complete", "transaction_id": "pick-love",
        "adds": {}, "drops": {}, "roster_ids": [1, 2], "status_updated": 1000,
        "draft_picks": [{"season": "2026", "round": 1, "roster_id": 9,
                         "owner_id": 1, "previous_owner_id": 2}],
    }
    api = types.ModuleType("dashboard_services.api")
    api.build_league_history_map = lambda *args: {2026: "L26", 2025: "L25"}
    service = types.ModuleType("dashboard_services.service")
    service.get_transactions_by_week = lambda lid, weeks, **kwargs: {0: [pick_trade]}
    utils = types.ModuleType("utils.utils")
    utils.load_players_index = lambda: {"love": {"name": "Jeremiyah Love", "pos": "RB"}}
    monkeypatch.setitem(sys.modules, "dashboard_services.api", api)
    monkeypatch.setitem(sys.modules, "dashboard_services.service", service)
    monkeypatch.setitem(sys.modules, "utils.utils", utils)
    monkeypatch.setattr(plt, "build_draft_resolution_map", lambda platform, lid, seasons=None: {
        (2026, 1, "9"): {"player_id": "love", "name": "Jeremiyah Love", "position": "RB",
                          "draft_id": "D26", "within_round": 1},
    })
    monkeypatch.setattr(plt, "_slot_map_for_league", lambda *args: {})
    monkeypatch.setattr(plt, "_roster_names", lambda *args: {"1": "Mine", "2": "Theirs"})

    out = plt.get_player_league_trades(
        player_id="love", platform="sleeper", league_id="L26", season=2026, limit=1,
    )
    # The same transaction returned through two lineage years is counted once,
    # and pick-linked matching occurs before pagination.
    assert out["total"] == 1
    assert out["trades"][0]["via_draft_pick"] is True


def test_roster_names_calls_build_roster_map_with_league_id(monkeypatch):
    import sys
    import types
    from dashboard_services import player_league_trades as plt

    captured = {}
    fake_players = types.ModuleType("dashboard_services.players")

    def fake_map(league_id, platform, season, users=None, rosters=None):
        captured["args"] = (league_id, platform, season)
        return {1: "Film Room", "5": "Hoodie's Heroes"}

    fake_players.build_roster_map = fake_map
    monkeypatch.setitem(sys.modules, "dashboard_services.players", fake_players)

    names = plt._roster_names("sleeper", "L26", 2025)
    assert captured["args"] == ("L26", "sleeper", 2025)
    assert names == {"1": "Film Room", "5": "Hoodie's Heroes"}
