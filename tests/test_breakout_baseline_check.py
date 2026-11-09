"""weekly-grading-v6: prior-season baseline check, partial games, and
the exact verdict word.

The baseline-check cases are shaped on the real Week 2 record the fix
exists for (diagnosed 2026-10-14): Dalton Kincaid's call graded against
a prior-season baseline snap share of 44.2% (his injury-depressed 2025:
12 games) while his actual 2026 Weeks 1-2 snap share was 69.3%. The old
grader recorded snap_delta +21.23 and a HELD role on a call whose grade
was a production-collapse MISS, and the Breakout Report graphic printed
that role as "GREW". The numbers below are the engine's own recorded
values for that call (baseline PPG 10.51, outcome 65.43% snaps and
3.07 PPG over Weeks 3-5, ppg_delta -7.44).
"""
from __future__ import annotations

import json
from datetime import date

import pytest

from data_building.breakout_engine import forecasts
from data_building.breakout_engine import weekly_grading as wg


def wk(week, snap=None, tgt=None, car=None, ppr=0.0, snaps=50):
    """One player_weekly_metrics-shaped row (shares are 0-100)."""
    return {
        "week": week, "snap_pct": snap, "targets": tgt, "carries": car,
        "ppr_pts": ppr, "snaps": snaps,
    }


def prior_season_call(*, as_of_week=2, baseline_snap=44.2, position="TE"):
    """A prior-season-baseline call: the stored evidence carries the
    prior-season snap baseline and no PPG (the engine's prior-season
    pseudo-row has no fantasy points), exactly the stored shape the
    grader documented for these calls."""
    return {
        "player_id": "10236", "player_name": "Dalton Kincaid",
        "season": 2026, "as_of_week": as_of_week,
        "as_of_date": date(2026, 9, 22),
        "scoring_version": "weekly-v6", "classification": "early_watch",
        "breakout_score": 40.7, "confidence": 0.8,
        "position": position, "baseline_source": "prior_season",
        "baseline_weeks": [],
        "evidence": {
            "fantasy": {},
            "signals": {"snap_share": {"baseline": baseline_snap,
                                       "recent": 69.3}},
        },
    }


def kincaid_prior_rows():
    """2025: 12 games at the recorded 10.51 PPG / 44.2% snap line."""
    return [wk(w, snap=44.2, tgt=5, car=0, ppr=10.51)
            for w in range(1, 13)]


def kincaid_current_rows():
    """2026 Weeks 1-2 at his real 69.3% snap share; Weeks 3-5 outcome
    at the recorded 65.43% / 3.07 PPG."""
    return [
        wk(1, snap=69.3, tgt=6, car=0, ppr=12.0),
        wk(2, snap=69.3, tgt=6, car=0, ppr=10.0),
        wk(3, snap=65.43, tgt=4, car=0, ppr=3.07),
        wk(4, snap=65.43, tgt=4, car=0, ppr=3.07),
        wk(5, snap=65.43, tgt=4, car=0, ppr=3.07),
    ]


# ---------------------------------------------------------------------------
# baseline check: the Kincaid case
# ---------------------------------------------------------------------------

def test_kincaid_baseline_aligned_to_current_role():
    grade = wg.grade_call(prior_season_call(), kincaid_current_rows(),
                          kincaid_prior_rows())
    # The miss stands: it was always a production verdict (-7.44 PPG
    # against the prior-season 10.51 baseline, which the current-season
    # 11.0 PPG does not materially exceed, so PPG is NOT aligned).
    assert grade["grade"] == wg.GRADE_MISS
    assert grade["ppg_delta"] == -7.44
    assert grade["baseline_ppg"] == 10.51
    # The role no longer grades against the injury-depressed 44.2%:
    # the snap baseline aligns to his actual 69.3% current role, so
    # the outcome reads as the slight fade it was (-3.87), not +21.23.
    assert grade["baseline_snap_pct"] == 69.3
    assert grade["snap_delta"] == -3.87
    assert grade["detail"]["role_state"] == "reverted"
    check = grade["detail"]["baseline_check"]
    assert check["aligned_fields"] == ["snap_pct"]
    assert check["current_games"] == 2
    assert check["compared"]["snap_pct"] == {
        "baseline": 44.2, "current": 69.3}
    assert grade["grading_version"] == "weekly-grading-v6"


def test_baseline_check_needs_two_current_games():
    # A Week 1 call shape: only one current-season game exists, so the
    # prior-season baseline is used as-is and no check is recorded.
    call = prior_season_call(as_of_week=1)
    rows = [wk(1, snap=69.3, tgt=6, car=0, ppr=12.0),
            wk(2, snap=65.43, tgt=4, car=0, ppr=3.07),
            wk(3, snap=65.43, tgt=4, car=0, ppr=3.07),
            wk(4, snap=65.43, tgt=4, car=0, ppr=3.07)]
    grade = wg.grade_call(call, rows, kincaid_prior_rows())
    assert "baseline_check" not in grade["detail"]
    assert grade["baseline_snap_pct"] == 44.2
    assert grade["snap_delta"] == 21.23


def test_baseline_check_never_lowers_a_baseline():
    # Current role BELOW the prior baseline: the check runs, aligns
    # nothing, and the prior baseline stands untouched.
    call = prior_season_call()
    rows = [wk(1, snap=30.0, tgt=3, car=0, ppr=5.0),
            wk(2, snap=30.0, tgt=3, car=0, ppr=5.0),
            wk(3, snap=32.0, tgt=3, car=0, ppr=6.0),
            wk(4, snap=32.0, tgt=3, car=0, ppr=6.0),
            wk(5, snap=32.0, tgt=3, car=0, ppr=6.0)]
    grade = wg.grade_call(call, rows, kincaid_prior_rows())
    check = grade["detail"]["baseline_check"]
    assert check["aligned_fields"] == []
    assert grade["baseline_snap_pct"] == 44.2
    assert grade["baseline_ppg"] == 10.51


def test_baseline_check_skips_current_season_calls():
    call = prior_season_call()
    call["baseline_source"] = "current_season"
    baseline, source, check = wg.resolve_baseline(
        call, kincaid_current_rows(), kincaid_prior_rows())
    assert check is None
    assert baseline["snap_pct"] == 44.2


def test_ppg_alignment_can_change_the_grade():
    # Prior-season PPG baseline 6.0, but the player's current-season
    # line before the call is already 11.0 PPG: grading the outcome
    # against 6.0 would manufacture a hit out of a flat continuation.
    call = prior_season_call(as_of_week=3, baseline_snap=30.0,
                             position="WR")
    call["player_name"] = "Test Receiver"
    prior_rows = [wk(w, snap=30.0, tgt=4, car=0, ppr=6.0)
                  for w in range(1, 13)]
    rows = [wk(1, snap=32.0, tgt=7, car=0, ppr=11.0),
            wk(2, snap=32.0, tgt=7, car=0, ppr=11.0),
            wk(3, snap=32.0, tgt=7, car=0, ppr=11.0),
            wk(4, snap=34.0, tgt=7, car=0, ppr=12.5),
            wk(5, snap=34.0, tgt=7, car=0, ppr=12.5),
            wk(6, snap=34.0, tgt=7, car=0, ppr=12.5)]
    grade = wg.grade_call(call, rows, prior_rows)
    assert grade["baseline_ppg"] == 11.0     # aligned, not the stale 6.0
    assert grade["ppg_delta"] == 1.5
    # Against the unchecked 6.0 baseline this same outcome is a hit
    # (+6.5 PPG at a startable level); against the real current role
    # it is flat production, a partial.
    assert grade["grade"] == wg.GRADE_PARTIAL
    assert grade["detail"]["baseline_check"]["aligned_fields"] == ["ppr_ppg"]


# ---------------------------------------------------------------------------
# partial games
# ---------------------------------------------------------------------------

def test_is_partial_game_threshold():
    assert wg.is_partial_game(wk(4, snap=16.9, snaps=11)) is True
    assert wg.is_partial_game(wk(4, snap=19.9)) is True
    assert wg.is_partial_game(wk(4, snap=20.0)) is False
    assert wg.is_partial_game(wk(4, snap=65.0)) is False
    assert wg.is_partial_game(wk(4, snap=None)) is False


def test_window_stats_excludes_partial_games():
    stats = wg.window_stats([
        wk(3, snap=65.0, tgt=5, car=0, ppr=10.0),
        wk(4, snap=16.9, tgt=1, car=0, ppr=1.5, snaps=11),  # injury exit
        wk(5, snap=60.0, tgt=6, car=0, ppr=12.0),
    ])
    assert stats["games"] == 2
    assert stats["partial_games"] == 1
    assert stats["ppr_ppg"] == 11.0      # not dragged to 7.83 by the exit
    assert stats["snap_pct"] == 62.5
    assert stats["opp_pg"] == 5.5


def test_grade_call_partial_game_not_full_weight():
    # Concepcion shape: two full outcome games plus an 11-snap early
    # exit in the third. The exit must not drag the outcome PPG at
    # full weight, and the two real games still grade the call.
    call = {
        "player_id": "999", "player_name": "Test Wideout", "season": 2026,
        "as_of_week": 2, "as_of_date": date(2026, 9, 22),
        "scoring_version": "weekly-v6", "classification": "early_watch",
        "breakout_score": 38.0, "confidence": 0.7, "position": "WR",
        "baseline_source": "current_season", "baseline_weeks": [1, 2],
        "evidence": {
            "fantasy": {"baseline_ppg": 8.0},
            "signals": {"snap_share": {"baseline": 40.0, "recent": 55.0}},
        },
    }
    rows = [wk(3, snap=55.0, tgt=5, car=0, ppr=9.0),
            wk(4, snap=58.0, tgt=6, car=0, ppr=10.0),
            wk(5, snap=16.9, tgt=1, car=0, ppr=0.8, snaps=11)]
    grade = wg.grade_call(call, rows)
    assert grade["outcome_games"] == 2
    assert grade["outcome_ppg"] == 9.5
    assert grade["detail"]["outcome_partial_games"] == 1


# ---------------------------------------------------------------------------
# exact verdict word
# ---------------------------------------------------------------------------

def test_role_verdict_word_is_the_engine_word_verbatim():
    assert wg.role_verdict_word("held") == "HELD"
    assert wg.role_verdict_word("reverted") == "REVERTED"
    assert wg.role_verdict_word("mixed") == "MIXED"
    assert wg.role_verdict_word("injured") == "INJURED"
    assert wg.role_verdict_word(None) == "UNKNOWN"
    # The whole point: held is never strengthened into GREW.
    assert wg.role_verdict_word("held") != "GREW"


def _grade_row(**overrides):
    row = {
        "player_id": "10236", "player_name": "Dalton Kincaid",
        "classification": "early_watch", "as_of_week": 2,
        "breakout_score": 40.7, "outcome_games": 3, "grade": "miss",
        "ppg_delta": -7.44, "opp_delta": None, "snap_delta": 21.23,
        "detail": {"role_state": "held", "reason": "production_collapsed"},
    }
    row.update(overrides)
    return row


def test_track_record_view_carries_engine_verdict_verbatim():
    # The historical row that shipped as GREW: a miss whose role the
    # engine recorded as held. The view the graphic consumes must now
    # carry the engine's words themselves, for dict and JSON-text
    # detail alike.
    for detail in (_grade_row()["detail"],
                   json.dumps(_grade_row()["detail"])):
        view = forecasts.top_grade_rows(
            [_grade_row(detail=detail)], wg.GRADE_MISS)[0]
        assert view["grade"] == "miss"
        assert view["role_state"] == "held"
        assert view["role_word"] == "HELD"
        assert view["role_word"] != "GREW"
