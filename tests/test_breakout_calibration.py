"""Self-learning calibration: band helpers, isotonic curve, findings.

Design rules pinned here:
- Bands align with the engine thresholds (watchlist 18, emerging 42) and
  the confidence bands; assignment is lower-inclusive at every edge.
- Bucket shape and the 10-graded rate floor mirror the grader exactly.
- The isotonic fit is non-decreasing by construction; a partial verdict
  is worth 0.5 of a hit (expected-value style).
- Findings never conclude anything below the sample guards (200 graded
  for the current version, 30 graded per compared band) and never
  recommend a change; zero rows is a sane input everywhere.
"""
from __future__ import annotations

import pytest

from data_building.breakout_engine import calibration as cal


def _row(verdict="hit", score=50.0, conf=50.0, classification="watchlist",
         version="weekly-v6", **extra):
    row = {
        "player_id": "p", "verdict": verdict, "breakout_score": score,
        "confidence": conf, "classification": classification,
        "scoring_version": version,
    }
    row.update(extra)
    return row


def _rows(verdict, n, **kw):
    return [_row(verdict, **kw) for _ in range(n)]


# ---------------------------------------------------------------------------
# band assignment boundaries
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("score,label", [
    (0.0, "Under 18"), (17.9, "Under 18"),
    (18.0, "18-41"), (41.9, "18-41"),
    (42.0, "42-59"), (59.9, "42-59"),
    (60.0, "60+"), (100.0, "60+"),
])
def test_score_band_boundaries(score, label):
    assert cal.score_band(score) == label


@pytest.mark.parametrize("conf,label", [
    (0.0, "Under 40"), (39.9, "Under 40"),
    (40.0, "40-69"), (69.9, "40-69"),
    (70.0, "70+"), (100.0, "70+"),
])
def test_confidence_band_boundaries(conf, label):
    assert cal.confidence_band(conf) == label


def test_band_of_missing_value_is_none():
    assert cal.score_band(None) is None
    assert cal.confidence_band(None) is None
    assert cal.score_band("not-a-number") is None


# ---------------------------------------------------------------------------
# summarize_bands
# ---------------------------------------------------------------------------

def test_summarize_bands_counts_and_floor():
    rows = []
    # 42-59 band: 12 graded (6 hit / 3 partial / 3 miss) -> rate shown.
    rows += _rows("hit", 6, score=50.0, conf=80.0)
    rows += _rows("partial", 3, score=50.0, conf=80.0)
    rows += _rows("miss", 3, score=50.0, conf=80.0)
    # 18-41 band: 4 graded -> below the floor, rate hidden, counts real.
    rows += _rows("hit", 4, score=30.0, conf=20.0)
    # One ungraded call: counted in calls, never in graded.
    rows += _rows("ungraded", 1, score=30.0, conf=20.0)

    summary = cal.summarize_bands(rows)
    bands = {b["band"]: b for b in summary["score_bands"]}
    assert [b["band"] for b in summary["score_bands"]] == \
        ["Under 18", "18-41", "42-59", "60+"]
    assert bands["42-59"]["graded"] == 12
    assert bands["42-59"]["hit"] == 6
    assert bands["42-59"]["partial"] == 3
    assert bands["42-59"]["miss"] == 3
    assert bands["42-59"]["hit_rate"] == pytest.approx(0.5)
    assert bands["18-41"]["calls"] == 5
    assert bands["18-41"]["graded"] == 4
    assert bands["18-41"]["ungraded"] == 1
    assert bands["18-41"]["hit_rate"] is None
    assert bands["Under 18"]["calls"] == 0
    assert bands["60+"]["graded"] == 0

    conf = {b["band"]: b for b in summary["confidence_bands"]}
    assert conf["70+"]["graded"] == 12
    assert conf["Under 40"]["graded"] == 4
    assert summary["overall"]["graded"] == 16
    assert summary["by_classification"]["watchlist"]["graded"] == 16
    assert summary["by_scoring_version"]["weekly-v6"]["graded"] == 16


def test_summarize_bands_missing_values_excluded_from_that_band_only():
    rows = _rows("hit", 10, score=50.0, conf=50.0)
    rows.append(_row("hit", score=None, conf=50.0))       # no score
    rows.append(_row("hit", score=50.0, conf=None))       # no confidence

    summary = cal.summarize_bands(rows)
    score = {b["band"]: b for b in summary["score_bands"]}
    conf = {b["band"]: b for b in summary["confidence_bands"]}
    assert score["42-59"]["graded"] == 11       # scoreless row excluded
    assert conf["40-69"]["graded"] == 11        # confidenceless excluded
    assert summary["overall"]["graded"] == 12   # both count overall
    assert summary["by_classification"]["watchlist"]["graded"] == 12


def test_summarize_bands_accepts_grade_key_rows():
    # Persisted grade rows name the verdict "grade", not "verdict".
    rows = [{"grade": "hit", "breakout_score": 65.0, "confidence": 90.0,
             "classification": "emerging_breakout",
             "scoring_version": "weekly-v6"} for _ in range(10)]
    summary = cal.summarize_bands(rows)
    assert summary["overall"]["hit"] == 10
    band = {b["band"]: b for b in summary["score_bands"]}
    assert band["60+"]["graded"] == 10


def test_summarize_bands_min_sample_override():
    rows = _rows("hit", 4, score=30.0)
    assert cal.summarize_bands(rows)["score_bands"][1]["hit_rate"] is None
    lowered = cal.summarize_bands(rows, min_sample=3)
    assert lowered["score_bands"][1]["hit_rate"] == pytest.approx(1.0)


def test_summarize_bands_zero_rows():
    summary = cal.summarize_bands([])
    assert summary["overall"] == {
        "calls": 0, "graded": 0, "hit": 0, "partial": 0, "miss": 0,
        "ungraded": 0, "hit_rate": None, "partial_rate": None,
        "miss_rate": None}
    assert len(summary["score_bands"]) == 4
    assert all(b["calls"] == 0 for b in summary["score_bands"])
    assert len(summary["confidence_bands"]) == 3
    assert summary["by_classification"] == {}
    assert summary["by_scoring_version"] == {}


# ---------------------------------------------------------------------------
# isotonic score curve
# ---------------------------------------------------------------------------

def test_fit_score_curve_empty_and_unusable():
    assert cal.fit_score_curve([]) == []
    assert cal.fit_score_curve(_rows("ungraded", 5, score=50.0)) == []
    assert cal.fit_score_curve(_rows("hit", 5, score=None)) == []


def test_fit_score_curve_partial_counts_half():
    curve = cal.fit_score_curve([_row("partial", score=50.0)])
    assert curve == [{"score": 50.0, "probability": pytest.approx(0.5)}]


def test_fit_score_curve_monotone_data_unchanged():
    rows = (_rows("miss", 2, score=10.0) + _rows("partial", 2, score=50.0)
            + _rows("hit", 2, score=90.0))
    curve = cal.fit_score_curve(rows)
    assert [p["score"] for p in curve] == [10.0, 50.0, 90.0]
    assert [p["probability"] for p in curve] == \
        [pytest.approx(0.0), pytest.approx(0.5), pytest.approx(1.0)]


def test_fit_score_curve_pools_violators_weighted():
    rows = []
    rows += _rows("miss", 4, score=10.0)                    # mean 0.00
    rows += _rows("hit", 3, score=20.0) + _rows("miss", 1, score=20.0)
    rows += _rows("hit", 1, score=30.0) + _rows("miss", 3, score=30.0)
    rows += _rows("hit", 4, score=90.0)                    # mean 1.00
    curve = cal.fit_score_curve(rows)
    # The 20 (0.75) and 30 (0.25) groups violate monotonicity and pool
    # into one weighted step at their combined mean, (3 + 1) / 8 = 0.5.
    assert curve == [
        {"score": 10.0, "probability": pytest.approx(0.0)},
        {"score": 20.0, "probability": pytest.approx(0.5)},
        {"score": 90.0, "probability": pytest.approx(1.0)},
    ]
    probs = [p["probability"] for p in curve]
    assert probs == sorted(probs)


# ---------------------------------------------------------------------------
# findings: guards first, conclusions only with data
# ---------------------------------------------------------------------------

def _band_block(score, hits, partials, misses, conf_for):
    rows = []
    for verdict, n in (("hit", hits), ("partial", partials),
                       ("miss", misses)):
        for _ in range(n):
            rows.append(_row(verdict, score=score, conf=conf_for(verdict)))
    return rows


def _healthy_rows():
    """200 graded weekly-v6 calls: value and hit rate rise with score,
    and confidence separates (hits confident, misses not)."""
    by_score = lambda v: {"hit": 80.0, "partial": 55.0, "miss": 20.0}[v]
    rows = []
    rows += _band_block(10.0, 4, 8, 28, by_score)     # Under 18: value .20
    rows += _band_block(30.0, 18, 12, 30, by_score)   # 18-41:    value .40
    rows += _band_block(50.0, 30, 12, 18, by_score)   # 42-59:    value .60
    rows += _band_block(80.0, 28, 6, 6, by_score)     # 60+:      value .775
    return rows


def test_findings_zero_rows():
    assert cal.findings([]) == ["insufficient data: 0 graded calls"]
    assert cal.findings(_rows("ungraded", 9)) == \
        ["insufficient data: 0 graded calls"]


def test_findings_below_version_guard_reports_counts():
    rows = _rows("hit", 30, score=50.0) + _rows("miss", 20, score=50.0)
    out = cal.findings(rows)
    assert len(out) == 4
    assert all(f.startswith("insufficient data: 50 graded calls under "
                            "weekly-v6 (need 200)") for f in out)


def test_findings_version_defaults_to_the_most_graded():
    rows = (_rows("hit", 150, version="weekly-v5")
            + _rows("hit", 60, version="weekly-v6"))
    out = cal.findings(rows)
    assert all("weekly-v5" in f and "150 graded calls" in f for f in out)
    out_v6 = cal.findings(rows, "weekly-v6")
    assert all("weekly-v6" in f and "60 graded calls" in f for f in out_v6)


def test_findings_healthy_engine_all_positive():
    out = cal.findings(_healthy_rows())
    assert not any(f.startswith("insufficient data") for f in out)
    assert any("rises with score" in f for f in out)
    assert any("out-hit the watchlist band" in f for f in out)
    assert any("hit more than low-confidence" in f for f in out)
    # Curve means by score: 10 -> .20, 30 -> .40, 50 -> .60, 80 -> .775.
    assert any("reaches a 50% hit value at a breakout score of 50" in f
               for f in out)


def test_findings_inversion_and_emerging_flags_fire():
    by_score = lambda v: {"hit": 80.0, "partial": 55.0, "miss": 20.0}[v]
    rows = []
    rows += _band_block(10.0, 4, 8, 28, by_score)     # value .20
    rows += _band_block(30.0, 36, 12, 12, by_score)   # value .70, hit .60
    rows += _band_block(50.0, 12, 12, 36, by_score)   # value .30, hit .20
    rows += _band_block(80.0, 28, 6, 6, by_score)     # value .775
    out = cal.findings(rows)
    inversion = [f for f in out if "not monotonic" in f]
    assert inversion and "42-59" in inversion[0] and "18-41" in inversion[0]
    assert any("do not out-hit the watchlist band" in f for f in out)


def test_findings_confidence_flag_fires_when_backwards():
    inverted = lambda v: {"hit": 20.0, "partial": 55.0, "miss": 85.0}[v]
    rows = []
    rows += _band_block(10.0, 4, 8, 28, inverted)
    rows += _band_block(30.0, 18, 12, 30, inverted)
    rows += _band_block(50.0, 30, 12, 18, inverted)
    rows += _band_block(80.0, 28, 6, 6, inverted)
    out = cal.findings(rows)
    assert any("do not hit more than low-confidence" in f for f in out)


def test_findings_thin_bands_stay_insufficient_per_check():
    # 200 graded overall, but 170 sit in one band: band comparisons have
    # no qualifying pair and must say so instead of concluding.
    rows = _rows("hit", 100, score=80.0, conf=50.0)
    rows += _rows("miss", 70, score=80.0, conf=50.0)
    rows += _rows("hit", 10, score=10.0, conf=50.0)
    rows += _rows("hit", 10, score=30.0, conf=50.0)
    rows += _rows("hit", 10, score=50.0, conf=50.0)
    out = cal.findings(rows)
    assert any(f.startswith("insufficient data") and "monotonicity" in f
               for f in out)
    assert any(f.startswith("insufficient data") and "separation" in f
               for f in out)
    assert any(f.startswith("insufficient data") and "confidence" in f
               for f in out)
