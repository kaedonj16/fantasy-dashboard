"""Depth-chart injury vacancy → breakout boost.

Sleeper's live depth chart (depth_chart_order + injury_status) already powers
waiver targets: an injured starter sitting AHEAD of a player frees the role.
These tests lock that same signal into the breakout engine's competition_removed
component — a hurt starter ahead should lift a buried-but-ready backup, and clear
the opportunity gate — while staying inert for historical rebuilds that supply no
live feed.
"""
import sys
import types

import pytest

# openai is a real dependency in CI but absent from the lean base suite; stub it
# so importing the engine package (core → projections → openai) doesn't skip the
# pure-function tests below.
if "openai" not in sys.modules:
    _stub = types.ModuleType("openai")
    for _n in ("OpenAI", "RateLimitError", "APIConnectionError", "APIStatusError",
               "APITimeoutError", "BadRequestError"):
        setattr(_stub, _n, type(_n, (Exception,), {}))
    sys.modules["openai"] = _stub

comp = pytest.importorskip("data_building.breakout_engine.components")

from data_building.breakout_engine.config import (  # noqa: E402
    INJURY_VACANCY_STARTER_POINTS,
    INJURY_VACANCY_MAX,
    BREAKOUT_GATE_COMP_MIN,
)


def _di(vacated, healthy_ahead=0):
    return {"vacated": vacated, "healthy_ahead": healthy_ahead}


# --- pure injury-vacancy scorer -------------------------------------------------

def test_no_injury_scores_zero():
    assert comp._injury_vacancy_score(None) == (0.0, [])
    assert comp._injury_vacancy_score(_di([]))[0] == 0.0


def test_season_ending_injury_directly_ahead_clears_gate():
    """IR to the starter immediately ahead (no healthy blockers) is a full
    opening — big enough on its own to clear the opportunity gate."""
    score, keys = comp._injury_vacancy_score(
        _di([{"status": "IR", "pid": "99", "name": "Starter X"}])
    )
    assert score == pytest.approx(INJURY_VACANCY_STARTER_POINTS)
    assert score >= BREAKOUT_GATE_COMP_MIN
    assert keys[0]["name"] == "Starter X"
    assert keys[0]["injury_status"] == "IR"


def test_healthy_blocker_ahead_discounts_the_vacancy():
    """A healthy body still between the candidate and the vacancy means it's not
    his role yet — the benefit is sharply discounted, and a deep player gets none."""
    full = comp._injury_vacancy_score(_di([{"status": "IR", "pid": "9"}], healthy_ahead=0))[0]
    one = comp._injury_vacancy_score(_di([{"status": "IR", "pid": "9"}], healthy_ahead=1))[0]
    deep = comp._injury_vacancy_score(_di([{"status": "IR", "pid": "9"}], healthy_ahead=3))[0]
    assert full > one > 0
    assert deep == 0.0


def test_softer_status_scores_less_than_season_ending():
    ir = comp._injury_vacancy_score(_di([{"status": "IR", "pid": "9"}]))[0]
    q = comp._injury_vacancy_score(_di([{"status": "QUESTIONABLE", "pid": "9"}]))[0]
    assert ir > q > 0


def test_multiple_injuries_stack_but_cap():
    many = [{"status": "IR", "pid": str(i)} for i in range(5)]
    score = comp._injury_vacancy_score(_di(many))[0]
    assert score == pytest.approx(INJURY_VACANCY_MAX)


# --- folded into competition_removed -------------------------------------------

def test_injury_only_produces_competition_removed_score():
    """No permanent departures, just a hurt starter ahead — competition_removed
    still fires (and clears the gate), and records the injury for explainability."""
    di = _di([{"status": "IR", "pid": "99", "name": "Starter X"}])
    score, details = comp.calculate_competition_removed_score(
        "1", "KC", "WR", 2026, {}, departures_cache={}, depth_injury=di
    )
    assert score >= BREAKOUT_GATE_COMP_MIN
    assert details["injury_vacancy_score"] == score
    assert len(details["injury_vacancies"]) == 1


def test_no_depth_injury_leaves_score_unchanged():
    """Historical rebuilds pass no live feed — behavior must be identical to
    before this feature (departure-only, zero when the roster is stable)."""
    base = comp.calculate_competition_removed_score(
        "1", "KC", "WR", 2026, {}, departures_cache={}
    )[0]
    assert base == 0.0


# --- core wiring: "who is injured ahead" reuses the waiver helper --------------

def test_depth_injury_for_finds_hurt_starter_ahead_and_names_them():
    """The engine's per-player lookup should surface a starter ranked ahead who
    is on IR, enriched with a display name, and ignore healthy teammates."""
    from data_building.breakout_engine.core import BreakoutEngine
    from utils.waiver_score import build_depth_index

    full_players = {
        "starter": {"team": "KC", "position": "WR", "depth_chart_order": 1,
                    "injury_status": "IR", "full_name": "Hurt Starter"},
        "backup": {"team": "KC", "position": "WR", "depth_chart_order": 2,
                   "injury_status": None, "full_name": "Buried Backup"},
        "other": {"team": "KC", "position": "WR", "depth_chart_order": 3,
                  "injury_status": None, "full_name": "Deeper Guy"},
    }

    # Build an engine shell without the DB-touching __init__.
    eng = BreakoutEngine.__new__(BreakoutEngine)
    eng.full_players = full_players
    eng.depth_index = build_depth_index(full_players)

    di = eng._depth_injury_for("backup")
    assert di is not None
    assert di["healthy_ahead"] == 0  # the only man ahead is hurt
    assert di["vacated"][0]["name"] == "Hurt Starter"
    assert di["vacated"][0]["status"] == "IR"

    # The starter himself has nobody hurt ahead of him → no vacancy.
    assert eng._depth_injury_for("starter") is None


def test_depth_injury_for_returns_none_without_live_feed():
    """No live players map (historical rebuild) → never a vacancy."""
    from data_building.breakout_engine.core import BreakoutEngine
    eng = BreakoutEngine.__new__(BreakoutEngine)
    eng.full_players = {}
    eng.depth_index = {}
    assert eng._depth_injury_for("anyone") is None
