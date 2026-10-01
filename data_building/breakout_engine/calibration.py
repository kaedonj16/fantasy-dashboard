"""Self-learning v1: offline calibration analysis over graded weekly calls.

This module learns FROM the breakout engine's own finished grades. It is
pure analysis over graded-call rows (dicts carrying ``breakout_score``,
``confidence``, ``classification``, ``scoring_version`` and a verdict under
``verdict`` or ``grade``): band summaries, an isotonic score-to-outcome
curve, and plain-language findings. It never writes anything and never
changes scoring.

OPERATING RULE: this report informs a human-approved SCORING_VERSION bump.
It recommends no new thresholds and applies nothing by itself; a person
reads the findings, decides, and ships any change as a new scoring version
through the normal review path. The breakout sidebar's Track Record and
``scripts/calibrate_weekly_breakouts.py`` both read these helpers, so the
rail and the CLI can never disagree about a band.

Conventions:

* A call's outcome value is expected-value style: hit = 1.0, partial = 0.5,
  miss = 0.0. Ungraded calls carry no outcome value and never enter a rate
  or the curve; they are counted in ``calls`` / ``ungraded`` only.
* Score bands align with the engine's own thresholds
  (``weekly_breakout.WATCHLIST_MIN_SCORE`` = 18, ``EMERGING_MIN_SCORE`` =
  42): "Under 18" [0, 18), "18-41" [18, 42), "42-59" [42, 60), "60+"
  [60, 100]. Confidence bands: "Under 40" [0, 40), "40-69" [40, 70),
  "70+" [70, 100].
* Bucket shape mirrors ``weekly_grading._rate_bucket`` exactly (calls,
  graded, hit, partial, miss, ungraded, hit_rate, partial_rate, miss_rate)
  and rates are None below the same 10-graded display floor the grader
  uses; counts are always reported.
* A row whose score (or confidence) is missing is excluded from that band
  set only; it still counts toward the overall / classification / version
  summaries.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

GRADE_HIT = "hit"
GRADE_PARTIAL = "partial"
GRADE_MISS = "miss"
GRADE_UNGRADED = "ungraded"

# Expected-value outcome weights (see module docstring).
HIT_VALUES = {GRADE_HIT: 1.0, GRADE_PARTIAL: 0.5, GRADE_MISS: 0.0}

# Display-rate floor: identical to weekly_grading.MIN_SUMMARY_SAMPLE.
MIN_RATE_SAMPLE = 10

# Findings guards. Below these counts a check reports "insufficient data"
# with the actual counts instead of a conclusion: small samples produce
# noise, not signal, and a calibration finding a human might act on needs
# more than the display floor.
MIN_FINDING_GRADED = 200
MIN_FINDING_BAND_GRADED = 30

# (label, lower bound inclusive, upper bound exclusive or None)
SCORE_BANDS: Tuple[Tuple[str, float, Optional[float]], ...] = (
    ("Under 18", 0.0, 18.0),
    ("18-41", 18.0, 42.0),
    ("42-59", 42.0, 60.0),
    ("60+", 60.0, None),
)
CONFIDENCE_BANDS: Tuple[Tuple[str, float, Optional[float]], ...] = (
    ("Under 40", 0.0, 40.0),
    ("40-69", 40.0, 70.0),
    ("70+", 70.0, None),
)


# =============================================================================
# small pure helpers
# =============================================================================

def _num(value: Any) -> Optional[float]:
    """Float or None. Mirrors the grader's rule: missing stays missing,
    never a fabricated zero (Decimal-safe)."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _verdict(row: Dict[str, Any]) -> str:
    return str(row.get("verdict") or row.get("grade") or "")


def _band_label(bands, value: Any) -> Optional[str]:
    score = _num(value)
    if score is None:
        return None
    for label, low, high in bands:
        if score >= low and (high is None or score < high):
            return label
    return None


def score_band(score: Any) -> Optional[str]:
    """Score-band label for a breakout score, or None when the score is
    missing (such rows join no score band)."""
    return _band_label(SCORE_BANDS, score)


def confidence_band(confidence: Any) -> Optional[str]:
    """Confidence-band label, or None when confidence is missing."""
    return _band_label(CONFIDENCE_BANDS, confidence)


def _bucket(rows: Sequence[Dict[str, Any]], min_sample: int) -> Dict[str, Any]:
    """Rate bucket with the grader's exact shape and floor semantics."""
    counts = {GRADE_HIT: 0, GRADE_PARTIAL: 0, GRADE_MISS: 0, GRADE_UNGRADED: 0}
    for row in rows:
        verdict = _verdict(row)
        if verdict in counts:
            counts[verdict] += 1
    graded = counts[GRADE_HIT] + counts[GRADE_PARTIAL] + counts[GRADE_MISS]
    enough = graded >= int(min_sample)

    def _rate(n: int) -> Optional[float]:
        if not enough or graded == 0:
            return None
        return round(n / graded, 4)

    return {
        "calls": len(rows),
        "graded": graded,
        "hit": counts[GRADE_HIT],
        "partial": counts[GRADE_PARTIAL],
        "miss": counts[GRADE_MISS],
        "ungraded": counts[GRADE_UNGRADED],
        "hit_rate": _rate(counts[GRADE_HIT]),
        "partial_rate": _rate(counts[GRADE_PARTIAL]),
        "miss_rate": _rate(counts[GRADE_MISS]),
    }


def _grouped(rows: Sequence[Dict[str, Any]], key: str) -> Dict[str, List[Dict[str, Any]]]:
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(str(row.get(key) or "unknown"), []).append(row)
    return groups


def summarize_bands(
    rows: Sequence[Dict[str, Any]],
    min_sample: int = MIN_RATE_SAMPLE,
) -> Dict[str, Any]:
    """Band + group summaries over graded-call rows. Pure.

    Returns the overall bucket, every score band and confidence band in
    fixed band order (empty bands included, zeroed), and by-classification
    / by-scoring-version buckets, all in the grader's bucket shape with
    rates None below ``min_sample`` graded in the group.
    """
    rows = list(rows)

    def _band_entries(bands, field, band_fn):
        grouped: Dict[str, List[Dict[str, Any]]] = {label: [] for label, _lo, _hi in bands}
        for row in rows:
            label = band_fn(row.get(field))
            if label is not None:
                grouped[label].append(row)
        return [
            {"band": label, **_bucket(grouped[label], min_sample)}
            for label, _lo, _hi in bands
        ]

    by_classification = _grouped(rows, "classification")
    by_version = _grouped(rows, "scoring_version")
    return {
        "min_sample": int(min_sample),
        "overall": _bucket(rows, min_sample),
        "score_bands": _band_entries(SCORE_BANDS, "breakout_score", score_band),
        "confidence_bands": _band_entries(
            CONFIDENCE_BANDS, "confidence", confidence_band),
        "by_classification": {
            key: _bucket(group, min_sample)
            for key, group in sorted(by_classification.items())
        },
        "by_scoring_version": {
            key: _bucket(group, min_sample)
            for key, group in sorted(by_version.items())
        },
    }


# =============================================================================
# isotonic score curve (pool-adjacent-violators, weighted, dependency-free)
# =============================================================================

def fit_score_curve(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, float]]:
    """Isotonic fit of outcome value on breakout score. Pure.

    Graded rows (hit / partial / miss) with a recorded score are grouped
    by exact score; each group's mean outcome value (hit = 1.0, partial =
    0.5, miss = 0.0) is fitted with pool-adjacent-violators weighted by
    group size, yielding the closest non-decreasing step function. Returns
    one point per fitted step, ``{"score": <lowest score in the step>,
    "probability": <fitted value>}``, ascending by score with
    non-decreasing probabilities. Empty or unusable input returns [].
    """
    totals: Dict[float, List[float]] = {}  # score -> [value sum, count]
    for row in rows:
        value = HIT_VALUES.get(_verdict(row))
        score = _num(row.get("breakout_score"))
        if value is None or score is None:
            continue
        slot = totals.setdefault(score, [0.0, 0.0])
        slot[0] += value
        slot[1] += 1.0
    if not totals:
        return []
    # Blocks: [lowest score, weight, weighted mean outcome value].
    blocks: List[List[float]] = []
    for score in sorted(totals):
        value_sum, count = totals[score]
        blocks.append([score, count, value_sum / count])
        while len(blocks) >= 2 and blocks[-2][2] > blocks[-1][2]:
            upper = blocks.pop()
            lower = blocks.pop()
            weight = lower[1] + upper[1]
            mean = (lower[2] * lower[1] + upper[2] * upper[1]) / weight
            blocks.append([lower[0], weight, mean])
    return [{"score": block[0], "probability": round(block[2], 4)}
            for block in blocks]


# =============================================================================
# findings (plain language, hard sample guards, observations only)
# =============================================================================

def _graded(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [r for r in rows if _verdict(r) in HIT_VALUES]


def _mean_hit_value(rows: Sequence[Dict[str, Any]]) -> Optional[float]:
    values = [HIT_VALUES[_verdict(r)] for r in rows if _verdict(r) in HIT_VALUES]
    if not values:
        return None
    return sum(values) / len(values)


def _hit_rate(rows: Sequence[Dict[str, Any]]) -> Optional[float]:
    graded = _graded(rows)
    if not graded:
        return None
    return sum(1 for r in graded if _verdict(r) == GRADE_HIT) / len(graded)


def _pct(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{value:.0%}"


def _band_rows(rows: Sequence[Dict[str, Any]], field: str,
               band_fn) -> Dict[str, List[Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        label = band_fn(row.get(field))
        if label is not None:
            grouped.setdefault(label, []).append(row)
    return grouped


def findings(rows: Sequence[Dict[str, Any]],
             scoring_version: Optional[str] = None) -> List[str]:
    """Plain-language calibration findings over graded-call rows. Pure.

    Checks run on the current scoring version's graded calls (the given
    ``scoring_version``, else the version with the most graded rows).
    Guards: below ``MIN_FINDING_GRADED`` graded calls for that version,
    every check reports insufficient data; a band comparison also needs
    ``MIN_FINDING_BAND_GRADED`` graded calls in each band it compares.
    Findings state observed facts only: they never recommend a threshold
    or any other change.
    """
    rows = list(rows)
    graded = _graded(rows)
    if not graded:
        return ["insufficient data: 0 graded calls"]

    if scoring_version is None:
        by_version = _grouped(graded, "scoring_version")
        scoring_version = sorted(
            by_version.items(), key=lambda kv: (-len(kv[1]), kv[0]))[0][0]
    current = [r for r in graded
               if str(r.get("scoring_version") or "unknown") == scoring_version]
    n_current = len(current)

    checks = ("score-band monotonicity", "emerging-band separation",
              "confidence validity", "the fitted score curve")
    if n_current < MIN_FINDING_GRADED:
        return [
            f"insufficient data: {n_current} graded calls under "
            f"{scoring_version} (need {MIN_FINDING_GRADED}) to check {check}"
            for check in checks
        ]

    out: List[str] = []
    score_groups = _band_rows(current, "breakout_score", score_band)
    conf_groups = _band_rows(current, "confidence", confidence_band)

    def _insufficient(check: str, detail: str) -> str:
        return f"insufficient data: {detail} to check {check}"

    # (a) Monotonicity: hit value should not fall between adjacent score
    # bands that both carry enough graded calls.
    labels = [label for label, _lo, _hi in SCORE_BANDS]
    pairs = []
    for lower, upper in zip(labels, labels[1:]):
        lower_rows = score_groups.get(lower, [])
        upper_rows = score_groups.get(upper, [])
        if (len(lower_rows) >= MIN_FINDING_BAND_GRADED
                and len(upper_rows) >= MIN_FINDING_BAND_GRADED):
            pairs.append((lower, lower_rows, upper, upper_rows))
    if not pairs:
        counts = ", ".join(
            f"{label} {len(score_groups.get(label, []))}" for label in labels)
        out.append(_insufficient(
            "score-band monotonicity",
            f"no adjacent score-band pair has {MIN_FINDING_BAND_GRADED} "
            f"graded calls in both bands under {scoring_version} "
            f"(graded by band: {counts})"))
    else:
        inversions = []
        for lower, lower_rows, upper, upper_rows in pairs:
            lower_value = _mean_hit_value(lower_rows)
            upper_value = _mean_hit_value(upper_rows)
            if upper_value is not None and lower_value is not None \
                    and upper_value < lower_value:
                inversions.append(
                    f"the {upper} band averaged a {_pct(upper_value)} hit "
                    f"value, below the {_pct(lower_value)} of the lower "
                    f"{lower} band ({len(upper_rows)} and "
                    f"{len(lower_rows)} graded calls)")
        if inversions:
            out.extend(f"Score bands are not monotonic under "
                       f"{scoring_version}: {text}" for text in inversions)
        else:
            compared = sorted({p[0] for p in pairs} | {p[2] for p in pairs},
                              key=labels.index)
            detail = ", ".join(
                f"{label} {_pct(_mean_hit_value(score_groups[label]))}"
                for label in compared)
            out.append(
                f"Hit value rises with score under {scoring_version} "
                f"across the bands with enough data ({detail}).")

    # (b) Emerging separation: the emerging-range bands (42-59, 60+)
    # should out-hit the watchlist band (18-41).
    base_rows = score_groups.get("18-41", [])
    if len(base_rows) < MIN_FINDING_BAND_GRADED:
        out.append(_insufficient(
            "emerging-band separation",
            f"the 18-41 watchlist band has {len(base_rows)} graded calls "
            f"under {scoring_version} (need {MIN_FINDING_BAND_GRADED})"))
    else:
        base_rate = _hit_rate(base_rows)
        weak = []
        strong = []
        for label in ("42-59", "60+"):
            band_rows = score_groups.get(label, [])
            if len(band_rows) < MIN_FINDING_BAND_GRADED:
                out.append(_insufficient(
                    "emerging-band separation",
                    f"the {label} band has {len(band_rows)} graded calls "
                    f"under {scoring_version} "
                    f"(need {MIN_FINDING_BAND_GRADED})"))
                continue
            rate = _hit_rate(band_rows)
            entry = f"{label} at {_pct(rate)}"
            if rate is not None and base_rate is not None \
                    and rate <= base_rate:
                weak.append(entry)
            else:
                strong.append(entry)
        if weak:
            out.append(
                f"Emerging-range calls do not out-hit the watchlist band "
                f"under {scoring_version}: {', '.join(weak)} vs the 18-41 "
                f"band at {_pct(base_rate)}.")
        if strong and not weak:
            out.append(
                f"Emerging-range bands out-hit the watchlist band under "
                f"{scoring_version}: {', '.join(strong)} vs the 18-41 "
                f"band at {_pct(base_rate)}.")

    # (c) Confidence validity: high-confidence calls should hit more
    # than low-confidence calls.
    low_rows = conf_groups.get("Under 40", [])
    high_rows = conf_groups.get("70+", [])
    if len(low_rows) < MIN_FINDING_BAND_GRADED \
            or len(high_rows) < MIN_FINDING_BAND_GRADED:
        out.append(_insufficient(
            "confidence validity",
            f"the Under 40 and 70+ confidence bands have "
            f"{len(low_rows)} and {len(high_rows)} graded calls under "
            f"{scoring_version} (need {MIN_FINDING_BAND_GRADED} each)"))
    else:
        low_rate = _hit_rate(low_rows)
        high_rate = _hit_rate(high_rows)
        if high_rate is not None and low_rate is not None \
                and high_rate <= low_rate:
            out.append(
                f"High-confidence calls do not hit more than "
                f"low-confidence calls under {scoring_version}: the 70+ "
                f"confidence band hit {_pct(high_rate)} vs "
                f"{_pct(low_rate)} for the Under 40 band "
                f"({len(high_rows)} and {len(low_rows)} graded calls).")
        else:
            out.append(
                f"High-confidence calls hit more than low-confidence "
                f"calls under {scoring_version}: the 70+ confidence band "
                f"hit {_pct(high_rate)} vs {_pct(low_rate)} for the "
                f"Under 40 band ({len(high_rows)} and {len(low_rows)} "
                f"graded calls).")

    # (d) Where the fitted curve crosses a 50% hit value: an observed
    # fact about the fitted curve, never a threshold instruction.
    curve = fit_score_curve(current)
    if not curve:
        out.append(_insufficient(
            "the fitted score curve",
            f"no graded calls with a recorded score under "
            f"{scoring_version}"))
    else:
        crossing = next((p for p in curve if p["probability"] >= 0.5), None)
        if crossing is not None:
            out.append(
                f"The fitted score curve reaches a 50% hit value at a "
                f"breakout score of {crossing['score']:g} under "
                f"{scoring_version} ({n_current} graded calls).")
        else:
            top = max(curve, key=lambda p: p["probability"])
            out.append(
                f"The fitted score curve never reaches a 50% hit value "
                f"under {scoring_version}; its highest fitted value is "
                f"{_pct(top['probability'])} at a breakout score of "
                f"{top['score']:g} ({n_current} graded calls).")
    return out
