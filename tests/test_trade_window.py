"""Tests for utils.trade_window (buy/sell window advisor)."""
from utils.trade_window import (
    BUY_THRESHOLD,
    SELL_THRESHOLD,
    trade_partners,
    trade_window_verdict,
)


class TestVerdict:
    def test_contender_buys(self):
        assert trade_window_verdict(82.0)["verdict"] == "buy"

    def test_rebuilder_sells(self):
        assert trade_window_verdict(12.0)["verdict"] == "sell"

    def test_bubble_holds(self):
        assert trade_window_verdict(50.0)["verdict"] == "hold"

    def test_thresholds_inclusive(self):
        assert trade_window_verdict(BUY_THRESHOLD)["verdict"] == "buy"
        assert trade_window_verdict(SELL_THRESHOLD)["verdict"] == "sell"

    def test_urgent_near_deadline(self):
        assert trade_window_verdict(80, weeks_to_deadline=2)["urgent"]
        assert trade_window_verdict(80, weeks_to_deadline=0)["urgent"]
        assert not trade_window_verdict(80, weeks_to_deadline=6)["urgent"]
        assert not trade_window_verdict(80, weeks_to_deadline=None)["urgent"]

    def test_all_in_modifier(self):
        vw = trade_window_verdict(80, age_rank=1, n_teams=12)
        assert vw["modifier"] == "all_in"

    def test_youth_modifier(self):
        vw = trade_window_verdict(10, age_rank=12, n_teams=12)
        assert vw["modifier"] == "youth"

    def test_aging_bubble_modifier(self):
        vw = trade_window_verdict(50, age_rank=2, n_teams=12)
        assert vw["modifier"] == "aging_bubble"

    def test_no_modifier_without_age_data(self):
        assert trade_window_verdict(80)["modifier"] == ""
        assert trade_window_verdict(80, age_rank=1, n_teams=None)["modifier"] == ""

    def test_young_buyer_gets_no_modifier(self):
        assert trade_window_verdict(80, age_rank=12, n_teams=12)["modifier"] == ""


class TestPartners:
    TEAMS = [
        {"name": "Me", "playoff_pct": 82, "is_viewer": True},
        {"name": "Tanker", "playoff_pct": 5},
        {"name": "Rebuilder", "playoff_pct": 25},
        {"name": "Bubble", "playoff_pct": 50},
        {"name": "Contender", "playoff_pct": 90},
    ]

    def test_buyer_gets_clearest_sellers_first(self):
        assert trade_partners(self.TEAMS, "buy") == ["Tanker", "Rebuilder"]

    def test_seller_gets_strongest_buyers_first(self):
        assert trade_partners(self.TEAMS, "sell") == ["Contender"]

    def test_bubble_teams_are_not_partners(self):
        assert "Bubble" not in trade_partners(self.TEAMS, "buy")
        assert "Bubble" not in trade_partners(self.TEAMS, "sell")

    def test_viewer_excluded(self):
        assert "Me" not in trade_partners(self.TEAMS, "sell")

    def test_hold_gets_no_partners(self):
        assert trade_partners(self.TEAMS, "hold") == []

    def test_limit(self):
        teams = [{"name": f"T{i}", "playoff_pct": i} for i in range(10)]
        assert len(trade_partners(teams, "buy", limit=3)) == 3

    def test_empty(self):
        assert trade_partners([], "buy") == []
        assert trade_partners(None, "buy") == []
