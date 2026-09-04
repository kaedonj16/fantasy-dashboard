"""Honest draft-capital availability: ESPN/Yahoo never invent future picks."""
from utils.draft_capital import (
    draft_capital_unavailable_copy,
    has_future_draft_capital,
    provider_exposes_draft_capital,
)


def test_provider_exposes_only_hosts_with_a_pick_feed():
    assert provider_exposes_draft_capital("sleeper")
    assert provider_exposes_draft_capital("mfl")
    assert provider_exposes_draft_capital("fleaflicker")
    assert not provider_exposes_draft_capital("espn")
    assert not provider_exposes_draft_capital("yahoo")
    assert not provider_exposes_draft_capital("unknown")


def test_espn_and_yahoo_never_have_future_capital_even_if_dynasty():
    dynasty = {"settings": {"type": "dynasty"}}
    assert has_future_draft_capital("espn", league=dynasty, settings=dynasty["settings"]) is False
    assert has_future_draft_capital("yahoo", league=dynasty, settings=dynasty["settings"]) is False


def test_sleeper_dynasty_has_capital_redraft_does_not():
    assert has_future_draft_capital(
        "sleeper",
        league={"settings": {"type": 2}},
        settings={"type": 2},
    )
    assert has_future_draft_capital(
        "sleeper",
        league={"settings": {"type": 0}},
        settings={"type": 0},
    ) is False


def test_unavailable_copy_names_the_host():
    espn = draft_capital_unavailable_copy("espn")
    yahoo = draft_capital_unavailable_copy("yahoo")
    assert "ESPN" in espn and "not available" in espn
    assert "Yahoo" in yahoo and "not available" in yahoo
    assert "—" not in espn and "—" not in yahoo
