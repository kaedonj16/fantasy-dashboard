"""Unit tests for the game-log upcoming-weeks projection median.

_game_log_upcoming_median decides what the player-modal Stats tab fills into
future game-log weeks that lack a weekly projection. Only weeks at/after the
projection boundary count: a player whose projections all predate it
(benched/cut -- the feed no longer projects them) must get 0 so stale
early-season numbers never masquerade as future forecasts.
"""
import pytest

# app.py imports pandas (and flask) at module load. The fast "lint" CI shard has
# flask but not pandas, so guard on pandas first -- otherwise importing app here
# raises at COLLECTION and aborts the whole run. The helper itself is pure; this
# test runs in the full-dependency shard.
pytest.importorskip("pandas")
pytest.importorskip("flask")
pytest.importorskip("openai")  # app.py pulls openai via dashboard_services.ai.client

from app import _game_log_upcoming_median  # noqa: E402


def test_median_over_upcoming_weeks_only():
    # Stale early-season values (weeks 1-2) must not drag the fill: only
    # weeks 6+ count.
    vals = {1: 8.0, 2: 9.0, 6: 20.0, 7: 22.0}
    assert _game_log_upcoming_median(vals, 6) == pytest.approx(21.0)


def test_stale_only_projections_yield_zero():
    # The Drew Lock case: every projection predates the boundary (benched, no
    # longer in the feed) -> 0 so future weeks stay blank.
    vals = {1: 10.0, 2: 12.6, 3: 15.0, 4: 11.0, 5: 13.2}
    assert _game_log_upcoming_median(vals, 6) == 0.0


def test_preseason_uses_all_weeks():
    # Boundary 1 (preseason/future season): everything counts.
    vals = {1: 10.0, 2: 14.0, 3: 12.0}
    assert _game_log_upcoming_median(vals, 1) == pytest.approx(12.0)


def test_explicit_upcoming_zeros_yield_zero():
    # Real zeros are values, not missing data: median is 0 and the fill gate
    # (_fill_med > 0) leaves unfetched weeks blank rather than inventing one.
    assert _game_log_upcoming_median({6: 0.0, 7: 0.0}, 6) == 0.0


def test_empty_and_bad_input_yield_zero():
    assert _game_log_upcoming_median({}, 6) == 0.0
    assert _game_log_upcoming_median(None, 6) == 0.0
    assert _game_log_upcoming_median({6: "x"}, 6) == 0.0
