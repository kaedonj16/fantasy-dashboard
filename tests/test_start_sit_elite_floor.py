"""Unit tests for the start/sit elite form floor (2026-11-13).

The floor is keyed on last-4-played PPG: when it is at or above the
position line (QB 22.0, RB 20.0, WR 20.0, TE 14.0), the form factor
may not sit below 1.0 -- a player still producing at an elite absolute
level is not discounted for running below his own season baseline
(the Gibbs case from the 2026-11-13 diagnosis). Bonuses are never
trimmed, below-line players are untouched, and a caller that passes
no last4_ppg gets no floor.
"""
import pytest

from utils.start_sit_score import (
    ELITE_FORM_FLOOR_PPG,
    FORM_SCALES,
    compute_start_score,
)


def _form(**kw):
    _, factors, _ = compute_start_score(20.0, **kw)
    return factors["form"]


# A standard "cooling off" shape: last-3 below the season line, 8 played
# games (shrink 8/12). The factors dict rounds form to 3dp, so raw
# expectations compare with a 1e-3 tolerance.
COOLING = dict(recent_ppg=16.0, season_ppg=20.0, played_games=8)


def _raw_form(pos):
    delta = (16.0 - 20.0) / FORM_SCALES[pos]
    return 1.0 + max(-1.0, min(1.0, delta)) * 0.08 * (8 / 12)


def test_thresholds_are_exact():
    assert ELITE_FORM_FLOOR_PPG == {
        "QB": 22.0, "RB": 20.0, "WR": 20.0, "TE": 14.0}


@pytest.mark.parametrize("pos,line", [
    ("QB", 22.0), ("RB", 20.0), ("WR", 20.0), ("TE", 14.0)])
def test_floor_fires_at_threshold(pos, line):
    assert _form(position=pos, last4_ppg=line, **COOLING) == 1.0


@pytest.mark.parametrize("pos,line", [
    ("QB", 22.0), ("RB", 20.0), ("WR", 20.0), ("TE", 14.0)])
def test_floor_silent_just_below_threshold(pos, line):
    form = _form(position=pos, last4_ppg=line - 0.1, **COOLING)
    assert form == pytest.approx(_raw_form(pos), abs=1e-3)


def test_rb_19_9_keeps_the_discount():
    # The named edge from the sign-off: 19.9 is good, not elite.
    form = _form(position="RB", last4_ppg=19.9, **COOLING)
    assert form == pytest.approx(_raw_form("RB"), abs=1e-3)
    assert form < 1.0


def test_bonus_is_never_trimmed():
    # Hot form (bonus) with an elite last-4: the floor lifts to 1.0 at
    # most, it must not pull a bonus down to neutral.
    hot = dict(recent_ppg=25.0, season_ppg=20.0, played_games=8)
    bonus = _form(position="RB", **hot)
    assert bonus > 1.0
    assert _form(position="RB", last4_ppg=23.0, **hot) == bonus


def test_no_last4_caller_is_unchanged():
    # Callers that never pass last4_ppg (no recorded form window) get
    # exactly the pre-floor behaviour.
    assert _form(position="RB", **COOLING) == pytest.approx(
        _raw_form("RB"), abs=1e-3)
    assert _form(position="RB", last4_ppg=None,
                 **COOLING) == pytest.approx(_raw_form("RB"), abs=1e-3)


def test_gibbs_shape_is_floored_to_neutral():
    # The diagnosis case: RB, last-3 18.5 vs a 26.2 season line
    # saturated the form clamp even after shrinkage; a 23.3 last-4
    # means the stretch is still elite in absolute terms.
    kw = dict(position="RB", recent_ppg=18.5, season_ppg=26.2,
              played_games=8, usage_delta=-0.1)
    _, factors, _ = compute_start_score(24.6, **kw)
    assert factors["form"] < 1.0
    _, factors, _ = compute_start_score(24.6, last4_ppg=23.3, **kw)
    assert factors["form"] == 1.0
    # Usage sat in the dead zone, so form_usage follows form exactly.
    assert factors["form_usage"] == 1.0


def test_floor_applies_after_shrinkage_not_instead_of_it():
    # A thin sample (2 played games, shrink 1/3) whose raw delta is
    # small: the floor decision is on the finished form factor, and a
    # below-line player keeps his (small) shrunk discount.
    kw = dict(position="WR", recent_ppg=18.0, season_ppg=20.0,
              played_games=2)
    shrunk = 1.0 + (-2.0 / 4.0) * 0.08 * (2 / 6)
    assert _form(last4_ppg=19.9, **kw) == pytest.approx(shrunk, abs=1e-3)
    assert _form(last4_ppg=20.0, **kw) == 1.0
