from __future__ import annotations

from utils.league_format import (
    auction_budget,
    detect_league_format,
    is_auction_draft,
    is_best_ball,
)


def test_sleeper_auction_draft_type():
    assert is_auction_draft({"type": "auction", "settings": {"budget": 200}})
    assert not is_auction_draft({"type": "snake", "settings": {"rounds": 15}})


def test_snake_draft_ignores_league_auction_budget():
    """ESPN may expose auctionBudget even on snake leagues — draft type wins."""
    league = {"settings": {"draftSettings": {"type": "SNAKE", "auctionBudget": 200}}}
    assert not is_auction_draft({"type": "snake"}, league=league)
    assert not is_auction_draft(league=league)


def test_espn_auction_budget_signal():
    league = {"settings": {"draftSettings": {"type": "AUCTION", "auctionBudget": 200}}}
    assert is_auction_draft(league=league)
    assert auction_budget(league=league) == 200


def test_espn_numeric_type_needs_budget():
    assert not is_auction_draft(league={"settings": {"draftSettings": {"type": "2"}}})
    assert is_auction_draft(
        league={"settings": {"draftSettings": {"type": "2", "auctionBudget": 100}}}
    )


def test_best_ball_settings_flag():
    assert is_best_ball({"settings": {"best_ball": 1}})
    assert is_best_ball({"settings": {"bestBall": True}})
    assert not is_best_ball({"settings": {"best_ball": 0}, "name": "Best Ball Bros"})


def test_detect_league_format_combined():
    fmt = detect_league_format(
        league={"name": "X", "settings": {"best_ball": 1}},
        drafts=[{"type": "auction", "settings": {"budget": 150}}],
    )
    assert fmt["is_auction"] is True
    assert fmt["is_best_ball"] is True
    assert fmt["auction_budget"] == 150
    assert fmt["draft_type"] == "auction"


def test_detect_league_format_prefers_full_snake_over_mock_auction():
    fmt = detect_league_format(
        drafts=[
            {"type": "auction", "status": "complete", "settings": {"rounds": 3, "budget": 200}},
            {"type": "snake", "status": "complete", "settings": {"rounds": 15}},
        ],
    )
    assert fmt["is_auction"] is False
    assert fmt["draft_type"] == "snake"
