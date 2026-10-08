"""Trade-calculator slim payload: QB/RB/WR/TE + picks, value fields only,
deltas/indicators folded in."""
import pytest

pytest.importorskip("pandas")

from dashboard_services.league_players_board import slim_trade_payload, slim_trade_player


def _player(pid, pos, **kw):
    base = {
        "id": pid,
        "name": f"Player {pid}",
        "team": "KC",
        "position": pos,
        "age": 25.0,
        "value": 500.0,
        "sf_value": 450.0,
        "value_8": 480.0,
        "value_12": 520.0,
        "value_14": 540.0,
        "pos_rank_label": "WR12",
        "is_rookie": False,
        "search_name": f"player {pid}",
        # Junk the trade view must drop:
        "vorp": 40,
        "ppg": 18.0,
        "projection": {"ppg": 16.2, "unit": "x"},
        "adp_by_source": {"sleeper": {"avg_pick": 12.0}},
        "historical": {"2024": {}},
    }
    base.update(kw)
    return base


def test_slim_trade_player_keeps_value_fields_drops_junk():
    row = slim_trade_player(_player("1", "WR"))
    assert row["id"] == "1"
    assert row["value"] == 500.0
    assert row["pos_rank_label"] == "WR12"
    assert "vorp" not in row
    assert "ppg" not in row
    assert "projection" not in row
    assert "adp_by_source" not in row
    assert "historical" not in row


def test_slim_trade_player_drops_non_skill_positions():
    assert slim_trade_player(_player("2", "K")) is None
    assert slim_trade_player(_player("3", "DEF")) is None
    assert slim_trade_player(_player("4", "PICK")) is not None
    assert slim_trade_player(_player("5", "QB")) is not None


def test_slim_trade_payload_folds_in_deltas_and_indicators():
    payload = {
        "tier_thresholds": {"1qb": {"10": [100]}},
        "players": [_player("1", "WR"), _player("2", "RB")],
    }
    out = slim_trade_payload(
        payload,
        delta_map={"1": 12.5},
        indicator_sets={"breakouts": ["1"], "elites": ["2"], "prospects": []},
    )
    by_id = {p["id"]: p for p in out["players"]}
    assert by_id["1"]["delta_7d"] == 12.5
    assert by_id["1"]["is_breakout"] is True
    assert by_id["2"]["is_elite"] is True
    assert "delta_7d" not in by_id["2"]
    assert out["tier_thresholds"] == {"1qb": {"10": [100]}}


def test_slim_trade_payload_does_not_mutate_input():
    p = _player("1", "WR")
    payload = {"players": [p]}
    slim_trade_payload(payload, delta_map={"1": 5.0},
                       indicator_sets={"breakouts": ["1"]})
    assert "delta_7d" not in p
    assert "is_breakout" not in p
