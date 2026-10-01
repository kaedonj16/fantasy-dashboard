"""Honest 'monitored' label for sub-threshold weekly calls.

A player the engine evaluates but scores below WATCHLIST_MIN_SCORE is not a
breakout candidate: new runs store "monitored" instead of "watchlist", and
by-classification aggregates bucket already-stored sub-18 "watchlist" rows
as "monitored" at read time. Stored rows are never rewritten, and every
other classification passes through untouched wherever it occurs.
"""
import pytest

wb = pytest.importorskip("data_building.breakout_engine.weekly_breakout")
from data_building.breakout_engine import calibration as cal
from data_building.breakout_engine import forecasts
from data_building.breakout_engine import weekly_grading as wg


def wk(week, snap=None, ts=None, tgt=None, car=None, pa=None, ppr=0.0, snaps=1):
    return {
        "week": week, "snap_pct": snap, "target_share": ts, "targets": tgt,
        "carries": car, "pass_att": pa, "ppr_pts": ppr, "snaps": snaps,
    }


WR = {"player_id": "1", "player_name": "Test WR", "team": "AAA", "position": "WR"}


# ---------------------------------------------------------------------------
# display_classification: the read-time mapping
# ---------------------------------------------------------------------------

def test_display_classification_sub_threshold_watchlist_becomes_monitored():
    assert wb.display_classification("watchlist", 17.9) == "monitored"
    assert wb.display_classification("watchlist", 0.0) == "monitored"


def test_display_classification_threshold_edge_stays_watchlist():
    # The gate sits exactly on the 18 threshold: 18 keeps watchlist.
    assert wb.display_classification("watchlist", 18.0) == "watchlist"
    assert wb.display_classification("watchlist", 35.0) == "watchlist"


def test_display_classification_missing_score_passes_through():
    assert wb.display_classification("watchlist", None) == "watchlist"
    assert wb.display_classification("watchlist", "nope") == "watchlist"


def test_display_classification_other_labels_never_remapped():
    # Real classifications are never bucketed as monitored, even far
    # below 18 - the residual gate is watchlist-only.
    assert wb.display_classification("emerging_breakout", 5.0) == "emerging_breakout"
    assert wb.display_classification("temporary_opportunity", 2.0) == "temporary_opportunity"
    assert wb.display_classification("early_watch", 10.0) == "early_watch"
    assert wb.display_classification(None, 5.0) == "unknown"


# ---------------------------------------------------------------------------
# scorer: the residual label is stored, other labels are byte-identical
# ---------------------------------------------------------------------------

def _final(**overrides):
    base = dict(final_score=5.0, recent_games=3, persistent=False,
                meaningful_role=False, supporting_signals=0,
                established=False, injury_vacated=False,
                garbage_time=False, provisional=False)
    base.update(overrides)
    return wb._classify_final_evidence(**base)


def test_scorer_residual_below_floor_is_monitored():
    assert _final() == "monitored"


def test_scorer_residual_at_floor_stays_watchlist():
    assert _final(final_score=18.0) == "watchlist"


def test_scorer_garbage_time_sub_threshold_is_monitored():
    assert _final(garbage_time=True, final_score=5.0) == "monitored"
    # A garbage-time game that still scores at/above the floor keeps the
    # label it has always had.
    assert _final(garbage_time=True, final_score=19.0) == "watchlist"


def test_scorer_temporary_opportunity_below_floor_untouched():
    # Role-driven labels are not score-gated: a teammate absence driving
    # the work is still a temporary opportunity under 18.
    assert _final(injury_vacated=True, meaningful_role=True,
                  final_score=5.0) == "temporary_opportunity"


def test_scorer_emerging_and_early_watch_untouched():
    assert _final(final_score=60.0, recent_games=3, persistent=True,
                  meaningful_role=True, supporting_signals=2) == "emerging_breakout"
    assert _final(final_score=25.0, meaningful_role=True,
                  supporting_signals=1) == "early_watch"


def test_scorer_end_to_end_flat_usage_is_monitored():
    # Flat usage with a one-game fantasy spike scores 0.0 and now stores
    # the honest residual label through the whole scoring path.
    rows = [wk(1, 55, 18, 6, ppr=8), wk(2, 54, 17, 6, ppr=7),
            wk(3, 56, 18, 6, ppr=7), wk(4, 55, 17, 5, ppr=28)]
    res = wb.score_player(WR, rows, cutoff_week=4)
    assert res["breakout_score"] < wb.WATCHLIST_MIN_SCORE
    assert res["classification"] == "monitored"


# ---------------------------------------------------------------------------
# aggregates: stored rows bucket identically in both summarizers
# ---------------------------------------------------------------------------

def _rows():
    rows = []
    for i in range(12):
        rows.append({"player_id": f"m{i}", "grade": "hit" if i < 6 else "miss",
                     "classification": "watchlist", "breakout_score": 10.0,
                     "scoring_version": "weekly-v6"})
    for i in range(12):
        rows.append({"player_id": f"w{i}", "grade": "hit" if i < 4 else "miss",
                     "classification": "watchlist", "breakout_score": 30.0,
                     "scoring_version": "weekly-v6"})
    for i in range(12):
        rows.append({"player_id": f"e{i}", "grade": "hit" if i < 10 else "miss",
                     "classification": "emerging_breakout",
                     "breakout_score": 55.0,
                     "scoring_version": "weekly-v6"})
    return rows


def test_summarize_grades_buckets_stored_sub_threshold_watchlist():
    summary = wg.summarize_grade_rows(_rows(), min_sample=10)
    groups = summary["by_classification"]
    assert "watchlist" in groups and "monitored" in groups
    assert groups["monitored"]["graded"] == 12
    assert groups["monitored"]["hit_rate"] == pytest.approx(6 / 12)
    assert groups["watchlist"]["graded"] == 12
    assert groups["watchlist"]["hit_rate"] == pytest.approx(4 / 12, abs=1e-4)
    assert groups["emerging_breakout"]["graded"] == 12
    # Bucketing moves rows between groups; it never changes the pool.
    assert summary["overall"]["graded"] == 36
    assert summary["overall"]["hit"] == 20


def test_calibration_bands_agree_with_grader_bucketing():
    bands = cal.summarize_bands(_rows(), min_sample=10)
    groups = bands["by_classification"]
    assert sorted(groups) == ["emerging_breakout", "monitored", "watchlist"]
    assert groups["monitored"]["graded"] == 12
    assert groups["watchlist"]["graded"] == 12
    assert groups["monitored"]["hit_rate"] == pytest.approx(6 / 12)


def test_track_record_payload_shows_monitored_group(monkeypatch):
    rows = [
        {"player_id": "a", "player_name": "A", "classification": "watchlist",
         "as_of_week": 2, "breakout_score": 3.0, "grade": "hit",
         "outcome_games": 3, "ppg_delta": 3.0, "opp_delta": 3.0,
         "snap_delta": 9.0},
        {"player_id": "b", "player_name": "B", "classification": "watchlist",
         "as_of_week": 2, "breakout_score": 50.0, "grade": "hit",
         "outcome_games": 3, "ppg_delta": 3.0, "opp_delta": 3.0,
         "snap_delta": 9.0},
    ]
    monkeypatch.setattr(forecasts, "load_weekly_grade_rows",
                        lambda season: rows)
    record = forecasts.weekly_track_record(2026)
    groups = {g["classification"]: g for g in record["groups"]}
    assert groups["monitored"]["graded"] == 1
    assert groups["watchlist"]["graded"] == 1
