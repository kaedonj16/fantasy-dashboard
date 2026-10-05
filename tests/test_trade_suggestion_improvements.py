"""Tests for trade suggestion improvements: 3-asset packages and bye warnings."""
import pytest

pytest.importorskip("pandas")

from dashboard_services.trade_acquire_packages import value_matched_acquire_packages
from utils.bye_outlook import trade_bye_coverage_warnings as trade_bye_warnings


def _player(pid, name, pos, value):
    return {
        "player_id": pid, "id": pid, "name": name,
        "position": pos, "value": value,
    }


def test_three_asset_packages_surface():
    """A 3-for-1 at fair value should appear in acquire packages."""
    players = [
        _player("p1", "Star WR", "WR", 300.0),
        _player("p2", "Mid RB1", "RB", 200.0),
        _player("p3", "Mid RB2", "RB", 180.0),
        _player("p4", "Mid WR", "WR", 170.0),
        _player("p5", "Depth TE", "TE", 100.0),
    ]
    # Focus worth ~550: no single player close, no 2-player combo in band,
    # but p2+p3+p4 = 550 hits it.
    pkgs = value_matched_acquire_packages(
        550.0, players, [], max_options=12, league_size=10,
    )
    three_asset = [p for p in pkgs if len(p["assets"]) == 3]
    assert three_asset, "expected a 3-asset package for the 550-value focus"


def test_three_asset_capped_to_top_assets():
    """3-asset combos only consider the top 10 by value (performance)."""
    players = [_player(f"p{i}", f"Player {i}", "WR", 100.0 + i) for i in range(20)]
    pkgs = value_matched_acquire_packages(
        350.0, players, [], max_options=12, league_size=10,
    )
    # Should complete without combinatorial blowup; just verify it runs
    assert isinstance(pkgs, list)


def test_bye_warnings_new_crunch_only():
    """Trading away your only QB cover flags Week 9; pre-existing crunches do not."""
    bye_by_team = {"KC": 9, "BUF": 7, "DAL": 9}
    lineup_reqs = {"QB": 1, "RB": 2, "WR": 2, "TE": 1}

    # Pre: QB1 (KC, bye 9) + QB2 (BUF, bye 7). Post: trade away QB2.
    pre = [
        {"position": "QB", "team": "KC"},
        {"position": "QB", "team": "BUF"},
        {"position": "RB", "team": "DAL"},
    ]
    post = [
        {"position": "QB", "team": "KC"},
        {"position": "RB", "team": "DAL"},
    ]
    # QB2's bye (week 7) is before from_week, so no warning there.
    # But wait: post has only KC QB with bye 9. Pre had BUF covering week 9.
    # In week 9: post has 1 QB on bye, needs 1 starter -> crunch.
    # In week 9: pre had 1 QB on bye (KC), BUF not on bye -> not crunch.
    warnings = trade_bye_warnings(bye_by_team, pre, post, lineup_reqs, from_week=1)
    assert len(warnings) == 1
    assert warnings[0]["week"] == 9
    assert warnings[0]["positions"] == ["QB"]
    assert "Week 9" in warnings[0]["message"]
    assert "QB" in warnings[0]["message"]


def test_bye_warnings_shared_bye_acquisition():
    """Acquiring two starters who share a bye flags the coverage gap."""
    bye_by_team = {"KC": 9, "SF": 9, "DAL": 10, "BUF": 11}
    lineup_reqs = {"QB": 1, "RB": 2, "WR": 2, "TE": 1}

    pre = [
        {"position": "WR", "team": "KC"},
        {"position": "WR", "team": "DAL"},
        {"position": "WR", "team": "BUF"},
    ]
    post = [
        {"position": "WR", "team": "KC"},
        {"position": "WR", "team": "SF"},  # acquired, shares bye 9 with KC
        {"position": "WR", "team": "BUF"},
    ]
    # Week 9 pre: KC out, DAL+BUF available = 2, need 2 -> no gap.
    # Week 9 post: KC+SF out, BUF available = 1, need 2 -> gap.
    warnings = trade_bye_warnings(bye_by_team, pre, post, lineup_reqs, from_week=1)
    assert len(warnings) == 1
    assert warnings[0]["week"] == 9
    assert warnings[0]["positions"] == ["WR"]


def test_bye_warnings_no_new_crunch():
    """A trade that does not create a new crunch produces no warnings."""
    bye_by_team = {"KC": 9, "BUF": 10}
    lineup_reqs = {"QB": 1, "RB": 2, "WR": 2, "TE": 1}

    pre = [
        {"position": "QB", "team": "KC"},
        {"position": "QB", "team": "BUF"},
    ]
    post = [
        {"position": "QB", "team": "KC"},
        {"position": "QB", "team": "BUF"},
    ]
    warnings = trade_bye_warnings(bye_by_team, pre, post, lineup_reqs, from_week=1)
    assert warnings == []


def test_bye_warnings_empty_bye_map():
    """No schedule data means no warnings, never a crash."""
    warnings = trade_bye_warnings({}, [{"position": "QB", "team": "KC"}], [], {"QB": 1})
    assert warnings == []
