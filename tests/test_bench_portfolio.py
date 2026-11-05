"""Frontier 7: bench as portfolio (utils/bench_portfolio.py).

End-of-bench spots are priced by path to value: one-away depth, the
handcuff premium for the viewer's own starters, live injury windows
with real return horizons, and role momentum. Funding order for
claims is weakest path first.
"""
from utils.bench_portfolio import bench_portfolio, drop_order


def _player(team, pos, order, name):
    return {"team": team, "position": pos, "depth_chart_order": order,
            "depth_chart_position": pos, "full_name": name,
            "injury_status": "", "status": "Active"}


def _ctx():
    players = {
        # Viewer's starter RB and his handcuff (bench).
        "rb1": _player("SF", "RB", 1, "Star Runner"),
        "rb2": _player("SF", "RB", 2, "Handcuff Back"),
        # Another team's backup, one away, nobody's handcuff here.
        "rb3": _player("DAL", "RB", 2, "Other Backup"),
        # A third-stringer with no path.
        "wr9": _player("NYJ", "WR", 3, "Deep Reserve"),
        # A bench WR with no depth signal at all.
        "wr8": _player("NYJ", "WR", None, "No Signal"),
    }
    return {
        "players": players,
        "rosters": [{
            "roster_id": 1,
            "players": ["rb1", "rb2", "rb3", "wr9", "wr8"],
            "starters": ["rb1"],
            "reserve": [], "taxi": [],
        }],
        "model_value_table": [
            {"player_id": "rb2", "value": 100.0},
            {"player_id": "rb3", "value": 400.0},
            {"player_id": "wr9", "value": 50.0},
            {"player_id": "wr8", "value": 300.0},
        ],
        "current_week": 9,
    }


def test_handcuff_beats_higher_value_backup_on_path():
    rows = bench_portfolio(_ctx(), 1, role_states={},
                           weeks_out_fn=lambda pid: None)
    by_id = {r["player_id"]: r for r in rows}
    # rb2: one-away (30) + handcuff (25) = 55, despite value 100.
    assert by_id["rb2"]["path_score"] == 55
    assert by_id["rb2"]["handcuff_of"] == "rb1"
    # rb3: one-away only = 30, despite value 400.
    assert by_id["rb3"]["path_score"] == 30
    assert rows[0]["player_id"] == "rb2"


def test_drop_order_funds_from_weakest_path_not_cheapest_sticker():
    order = drop_order(_ctx(), 1, role_states={},
                       weeks_out_fn=lambda pid: None)
    # wr9 (10 path, value 50) and wr8 (0 path, value 300): the
    # no-signal WR funds first even though he is worth more.
    assert order[0] == "wr8"
    assert order[-1] == "rb2"     # the handcuff is the last drop


def test_live_window_uses_the_return_horizon():
    rows = bench_portfolio(_ctx(), 1, role_states={},
                           weeks_out_fn=lambda pid: 4.0
                           if pid == "rb1" else None)
    by_id = {r["player_id"]: r for r in rows}
    # rb2 gains 4 weeks * 3 = 12 on top of 55.
    assert by_id["rb2"]["path_score"] == 67
    assert any("4 more weeks" in r for r in by_id["rb2"]["reasons"])


def test_window_without_a_date_uses_the_role_signal_only():
    rows = bench_portfolio(
        _ctx(), 1,
        role_states={"rb1": {"state": "starter_out"}},
        weeks_out_fn=lambda pid: None)
    by_id = {r["player_id"]: r for r in rows}
    assert by_id["rb2"]["path_score"] == 65   # 55 + 10, no fake date


def test_starters_and_reserve_never_appear():
    ctx = _ctx()
    ctx["rosters"][0]["reserve"] = ["wr9"]
    rows = bench_portfolio(ctx, 1, role_states={},
                           weeks_out_fn=lambda pid: None)
    ids = {r["player_id"] for r in rows}
    assert "rb1" not in ids and "wr9" not in ids


def test_empty_without_roster():
    assert bench_portfolio(_ctx(), 99) == []
    assert bench_portfolio({}, 1) == []
