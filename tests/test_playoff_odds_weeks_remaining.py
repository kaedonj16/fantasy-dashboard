"""Playoff-odds header: weeks remaining must match the sim, not the provider week.

Regression for the reported bug: with 4 games played, the Playoff Odds tab
showed "9 weeks remaining" instead of 10. The sim derives remaining weeks
from finalized W/L/T records (10 weeks: 5..14 for a week-15 playoff start),
but the header computed ``playoff_week_start - current_week - 1`` where
``current_week`` is Sleeper's *in-progress* week (5) -- the extra ``-1``
treated the week being played as already finished.

The fix: ``remaining_regular_season_weeks`` (same source as the sim) feeds
``weeks_remaining`` in the /api/playoff-odds payload, and the JS prefers it.
"""

import re
from pathlib import Path

import pytest

# simulate_playoff_odds imports numpy at module load. The fast "lint" CI shard
# has flask but not numpy, so guard on numpy first -- otherwise importing the
# sim module here raises at COLLECTION and aborts the whole run. The full
# integration shard has numpy and runs these tests normally.
pytest.importorskip("numpy")

from data_building.simulate_playoff_odds import remaining_regular_season_weeks

ROOT = Path(__file__).resolve().parents[1]
APP_PY = (ROOT / "app.py").read_text(encoding="utf-8")
APP_JS = (ROOT / "static" / "app.js").read_text(encoding="utf-8")


def _teams(records, playoff_week_start=15):
    """records: list of (wins, losses, ties)."""
    return [
        {"roster_id": i, "wins": w, "losses": l, "ties": t}
        for i, (w, l, t) in enumerate(records)
    ]


# ---------------------------------------------------------------------------
# remaining_regular_season_weeks: the reported bug and its edges
# ---------------------------------------------------------------------------

def test_reported_bug_four_games_played_shows_ten_not_nine():
    # Her league: 4-0, 3-1, 2-2, 2-2, 2-2, 1-3, 1-3, 2-2, 2-2, 0-4.
    # Sleeper's current_week is 5 (the week being played), but only 4 weeks
    # are finalized, so weeks 5..14 = 10 remain.
    records = [(4, 0, 0), (3, 1, 0), (2, 2, 0), (2, 2, 0), (2, 2, 0),
               (1, 3, 0), (2, 2, 0), (1, 3, 0), (2, 2, 0), (0, 4, 0)]
    assert remaining_regular_season_weeks(
        _teams(records), 15, current_week=5) == 10


def test_preseson_all_zero_records_shows_fourteen():
    records = [(0, 0, 0)] * 10
    assert remaining_regular_season_weeks(
        _teams(records), 15, current_week=0) == 14


def test_tie_counts_as_a_completed_game():
    # 3-0-1 is 4 games played, same as 4-0.
    records = [(3, 0, 1)] * 10
    assert remaining_regular_season_weeks(
        _teams(records), 15, current_week=5) == 10


def test_final_week_in_progress_shows_one():
    # Week 14 being played, 13 finalized: only week 14 remains.
    records = [(10, 3, 0)] * 10
    assert remaining_regular_season_weeks(
        _teams(records), 15, current_week=14) == 1


def test_season_complete_shows_zero():
    records = [(11, 3, 0)] * 10
    assert remaining_regular_season_weeks(
        _teams(records), 15, current_week=15) == 0


def test_lagging_team_does_not_hide_weeks():
    # One team's record lags (3 games vs 4): the sim uses the minimum, so
    # the header stays consistent with what was simulated.
    records = [(4, 0, 0)] * 9 + [(3, 0, 0)]
    assert remaining_regular_season_weeks(
        _teams(records), 15, current_week=5) == 11


# ---------------------------------------------------------------------------
# Source contracts: the payload and the JS must use the backend count
# ---------------------------------------------------------------------------

def test_api_payload_includes_weeks_remaining():
    # /api/playoff-odds must serve the backend count the header reads.
    route_start = APP_PY.index('def api_playoff_odds():')
    route_end = APP_PY.index('def api_playoff_scenarios()', route_start)
    route_src = APP_PY[route_start:route_end]
    assert "remaining_regular_season_weeks(" in route_src
    assert '"weeks_remaining"' in route_src


def test_js_prefers_backend_weeks_remaining_over_week_arithmetic():
    # _renderPlayoffOdds must destructure weeks_remaining and prefer it;
    # the playoff_week_start - current_week - 1 formula stays only as a
    # fallback for older cached payloads.
    fn_start = APP_JS.index("function _renderPlayoffOdds(data)")
    fn_end = APP_JS.index("function _poScenClass", fn_start)
    fn_src = APP_JS[fn_start:fn_end]
    assert re.search(r"const\s*\{[^}]*\bweeks_remaining\b[^}]*\}\s*=\s*data", fn_src)
    assert "weeks_remaining != null" in fn_src
    # fallback still present, not primary
    assert "playoff_week_start - current_week - 1" in fn_src
