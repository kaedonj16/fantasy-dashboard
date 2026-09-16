"""Draft-pick labels on the league activity page."""

from dashboard_services.pages.activity_page import (
    _activity_pick_label,
    _activity_pick_order_is_final,
)


def test_future_pick_does_not_show_unfinalized_slot():
    """Current standings must not masquerade as next year's draft order."""
    assert _activity_pick_label(2027, 3, 2026, 0) == "2027 3rd"
    assert _activity_pick_label(2027, 3, 2026, 4) == "2027 3rd"


def test_finalized_pick_keeps_exact_slot():
    assert _activity_pick_label(2026, 1, 2026, 4) == "2026 1.04"


def test_unordered_pick_uses_ordinal_round():
    assert _activity_pick_label(2026, 1, 2026) == "2026 1st"
    assert _activity_pick_label(2026, 2, 2026) == "2026 2nd"
    assert _activity_pick_label(2026, 4, 2026) == "2026 4th"


def test_missing_pick_data_does_not_trigger_order_lookup():
    assert _activity_pick_label(0, 3, 2026, 4) == "Pick"
    assert _activity_pick_label(2027, 0, 2026, 4) == "Pick"
    assert not _activity_pick_order_is_final(0, 2026)
    assert not _activity_pick_order_is_final(2026, 0)
