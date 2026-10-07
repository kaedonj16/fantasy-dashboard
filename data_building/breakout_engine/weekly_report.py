"""Weekly breakout grading and calibration report (the Wednesday email).

This module builds the weekly self-learning report for the breakout
engine: how the surfaced calls that finished grading recently actually
did, the season record so far (from
:mod:`data_building.breakout_engine.calibration`, the same helpers the
sidebar Track Record and the calibrate CLI use, so every surface
agrees), and a concrete suggested-changes plan Kaedon can hand back to
Muse to apply. It is pure analysis over grade-row dicts: it writes
nothing and changes no scoring, threshold, weight, or version. Only
surfaced calls (breakout score >= 30, not watchlist/monitored, per
:func:`data_building.breakout_engine.weekly_grading.is_surfaced_row`)
are reported; anything never shown on the board is excluded from every
section.

Inputs are rows as loaded from ``weekly_breakout_grades`` (see
``scripts/calibrate_weekly_breakouts.load_graded_calls``): each row
carries ``grade`` (or ``verdict``), ``breakout_score``, ``confidence``,
``classification``, ``scoring_version``, ``player_id``, ``player_name``,
``as_of_week`` and the outcome columns the grader stored (``ppg_delta``,
``baseline_ppg``, ``outcome_ppg``, ``opp_delta``, ``snap_delta``).

NEWLY-GRADED WINDOW: grade rows carry ``graded_at`` (``TIMESTAMP
DEFAULT NOW()`` in ``weekly_grading.init_weekly_breakout_grades_db``),
so "this week" means the grades recorded in the ``WINDOW_DAYS`` days
before the report's ``as_of`` moment. A row whose ``graded_at`` is
missing or unparseable is never counted as new; it still counts toward
the cumulative season record. The send script passes the live clock;
tests pass a fixed ``as_of``.

Suggested changes are driven by ``calibration.findings`` plus the fitted
score curve, under calibration's own sample guards. A conversion is
suggested only for the one change that maps to exact constants: when the
fitted curve's 50% hit-value crossing sits at least
``CONVERT_MIN_SHIFT`` points away from the current
``EMERGING_MIN_SCORE``, the plan names the constant, the from and to
values, and the next scoring version. Band inversions and invalid
confidence do not map to a single constant, so they surface as
"Investigate:" lines with their evidence. Every line carries its sample
size. Copy rule: no em dashes anywhere in the rendered email.
"""
from __future__ import annotations

import re
from datetime import date, datetime, time, timedelta, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

from data_building.breakout_engine import calibration
from data_building.breakout_engine.weekly_breakout import (
    EMERGING_MIN_SCORE,
    SCORING_VERSION,
)
from data_building.breakout_engine.weekly_grading import is_surfaced_row

WINDOW_DAYS = 7
NOTABLE_LIMIT = 5
# How far the fitted 50% crossing must sit from EMERGING_MIN_SCORE
# before the plan suggests converting the threshold to it.
CONVERT_MIN_SHIFT = 5.0

_GRADED = (calibration.GRADE_HIT, calibration.GRADE_PARTIAL,
           calibration.GRADE_MISS)


# =============================================================================
# small pure helpers
# =============================================================================

def _num(value: Any) -> Optional[float]:
    """Float or None. Missing stays missing, never a fabricated zero."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _verdict(row: Dict[str, Any]) -> str:
    return str(row.get("verdict") or row.get("grade") or "")


def _parse_ts(value: Any) -> Optional[datetime]:
    """Parse a graded_at value (datetime, date, or ISO string) to a naive
    datetime, or None when it cannot be read. Aware values are converted
    to UTC and made naive so they compare against the naive ``as_of``."""
    if value is None:
        return None
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, date):
        parsed = datetime.combine(value, time.min)
    elif isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            return None
    else:
        return None
    if parsed.tzinfo is not None:
        parsed = parsed.astimezone(timezone.utc).replace(tzinfo=None)
    return parsed


def _fmt_rate(rate: Optional[float]) -> str:
    return "n/a" if rate is None else f"{rate:.1%}"


def _fmt_score(value: Any) -> str:
    number = _num(value)
    return "n/a" if number is None else f"{number:g}"


def _next_version(version: str) -> str:
    """weekly-v6 -> weekly-v7. A version with no trailing number gets a
    plain suffix instead of a guess."""
    match = re.match(r"^(.*?)(\d+)$", version or "")
    if not match:
        return f"{version}-next"
    return f"{match.group(1)}{int(match.group(2)) + 1}"


# =============================================================================
# sections
# =============================================================================

def newly_graded_rows(rows: Sequence[Dict[str, Any]], as_of: datetime,
                      window_days: int = WINDOW_DAYS) -> List[Dict[str, Any]]:
    """Rows whose graded_at falls in the window ending at ``as_of``."""
    start = as_of - timedelta(days=window_days)
    out = []
    for row in rows:
        stamp = _parse_ts(row.get("graded_at"))
        if stamp is not None and start <= stamp <= as_of:
            out.append(row)
    return out


def _notable(rows: Sequence[Dict[str, Any]], verdict: str) -> List[Dict[str, Any]]:
    """Up to NOTABLE_LIMIT rows with the given verdict, biggest outcome
    first: hits by largest PPG rise, misses by largest PPG fall, ties by
    higher breakout score. Rows without a PPG delta sort last."""
    picked = [r for r in rows if _verdict(r) == verdict]

    def _key(row: Dict[str, Any]):
        delta = _num(row.get("ppg_delta"))
        score = _num(row.get("breakout_score")) or 0.0
        if verdict == calibration.GRADE_HIT:
            return (-(delta if delta is not None else -1e18), -score)
        return ((delta if delta is not None else 1e18), -score)

    picked.sort(key=_key)
    return [{
        "player": str(row.get("player_name") or row.get("player_id")
                      or "unknown"),
        "classification": str(row.get("classification") or "unknown"),
        "as_of_week": row.get("as_of_week"),
        "breakout_score": _num(row.get("breakout_score")),
        "confidence": _num(row.get("confidence")),
        "ppg_delta": _num(row.get("ppg_delta")),
        "baseline_ppg": _num(row.get("baseline_ppg")),
        "outcome_ppg": _num(row.get("outcome_ppg")),
        "opp_delta": _num(row.get("opp_delta")),
        "snap_delta": _num(row.get("snap_delta")),
    } for row in picked[:NOTABLE_LIMIT]]


def _nothing_new_line(rows: Sequence[Dict[str, Any]],
                      latest_week: Optional[int],
                      window_days: int) -> str:
    """The honest one-liner for the Graded-this-week section when no
    verdict landed in the window, with the reason when it is knowable."""
    total_graded = sum(1 for r in rows if _verdict(r) in _GRADED)
    week_note = (f" Stored weekly data currently reaches week "
                 f"{latest_week}." if latest_week is not None else "")
    if total_graded == 0:
        return ("Nothing has been graded yet this season: no breakout "
                "call has finished its 3-week outcome window."
                + week_note)
    stamps = [s for s in (_parse_ts(r.get("graded_at")) for r in rows)
              if s is not None]
    recent = (f" The most recent grades were recorded on "
              f"{max(stamps):%Y-%m-%d}." if stamps else "")
    return (f"Nothing new graded in the last {window_days} days."
            + recent + week_note)


def suggested_changes(rows: Sequence[Dict[str, Any]],
                      current_rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """The approval plan: conversions, investigations, or the reason no
    change is suggested. Driven by calibration.findings plus the fitted
    curve, under calibration's sample guards."""
    graded_current = [r for r in current_rows if _verdict(r) in _GRADED]
    n_current = len(graded_current)
    findings = calibration.findings(rows, SCORING_VERSION)

    conversions: List[str] = []
    crossing_score: Optional[float] = None
    if n_current >= calibration.MIN_FINDING_GRADED:
        curve = calibration.fit_score_curve(graded_current)
        crossing = next((p for p in curve if p["probability"] >= 0.5), None)
        if crossing is not None:
            crossing_score = float(crossing["score"])
            if abs(crossing_score - EMERGING_MIN_SCORE) >= CONVERT_MIN_SHIFT:
                conversions.append(
                    f"Convert: EMERGING_MIN_SCORE "
                    f"{EMERGING_MIN_SCORE:g} -> {crossing_score:g} and "
                    f"bump SCORING_VERSION {SCORING_VERSION} -> "
                    f"{_next_version(SCORING_VERSION)} (fitted 50% "
                    f"hit-value crossing at {crossing_score:g}, based on "
                    f"{n_current} graded {SCORING_VERSION} calls).")

    investigations: List[str] = []
    for finding in findings:
        if finding.startswith("insufficient data"):
            continue
        if finding.startswith("The fitted score curve reaches a 50%"):
            continue  # the conversion's evidence, not a problem to chase
        if finding.startswith(("Hit value rises with score",
                               "Emerging-range bands out-hit",
                               "High-confidence calls hit more")):
            continue  # a passing check, not a problem to chase
        investigations.append(f"Investigate: {finding}")

    reason: Optional[str] = None
    if not conversions and not investigations:
        insufficient = [f for f in findings
                        if f.startswith("insufficient data")]
        if insufficient:
            first = insufficient[0]
            reason = first[0].upper() + first[1:] + "."
        elif crossing_score is not None:
            reason = (
                f"The fitted 50% hit-value crossing sits at "
                f"{crossing_score:g}, within {CONVERT_MIN_SHIFT:g} points "
                f"of EMERGING_MIN_SCORE {EMERGING_MIN_SCORE:g}, and every "
                f"other check passed ({n_current} graded "
                f"{SCORING_VERSION} calls).")
        else:
            reason = (
                f"All calibration checks passed under {SCORING_VERSION} "
                f"({n_current} graded calls).")
    return {
        "conversions": conversions,
        "investigations": investigations,
        "no_change_reason": reason,
    }


# =============================================================================
# report assembly + rendering
# =============================================================================

def build_weekly_report(rows: Sequence[Dict[str, Any]], season: int,
                        as_of: Optional[datetime] = None,
                        latest_week: Optional[int] = None,
                        window_days: int = WINDOW_DAYS) -> Dict[str, Any]:
    """Structured weekly report over already-loaded grade rows. Pure.

    ``rows`` spans every scoring version (the week's grading activity is
    reported as it happened); the cumulative season record and the
    suggested-changes plan cover the current SCORING_VERSION only.
    ``latest_week`` is the season's most recent stored week when the
    caller knows it; it only feeds the honest reason line when nothing
    new graded.

    Every section is surfaced-calls only (see
    :func:`data_building.breakout_engine.weekly_grading.is_surfaced_row`):
    watchlist/monitored rows and sub-30 scores were never on the board,
    so they are excluded from the new grades, the season record, the
    notable calls, and the calibration plan alike.
    """
    rows = [r for r in rows if is_surfaced_row(r)]
    as_of = as_of or datetime.now()
    current = [r for r in rows
               if str(r.get("scoring_version") or "") == SCORING_VERSION]
    new_rows = newly_graded_rows(rows, as_of, window_days)
    new_bucket = calibration.summarize_bands(new_rows)["overall"]
    cumulative = calibration.summarize_bands(current)
    return {
        "season": int(season),
        "as_of": as_of,
        "window_days": int(window_days),
        "scoring_version": SCORING_VERSION,
        "latest_week": latest_week,
        "total_graded_all_versions":
            calibration.summarize_bands(rows)["overall"]["graded"],
        "newly_graded": {
            "bucket": new_bucket,
            "notable_hits": _notable(new_rows, calibration.GRADE_HIT),
            "notable_misses": _notable(new_rows, calibration.GRADE_MISS),
        },
        "nothing_new_line": (
            _nothing_new_line(rows, latest_week, window_days)
            if new_bucket["graded"] == 0 else None),
        "cumulative": {
            "overall": cumulative["overall"],
            "by_classification": cumulative["by_classification"],
            "score_bands": cumulative["score_bands"],
            "confidence_bands": cumulative["confidence_bands"],
        },
        "suggested_changes": suggested_changes(rows, current),
    }


def _bucket_line(label: str, bucket: Dict[str, Any]) -> str:
    return (
        f"  {label:<28} calls={bucket['calls']:<5} "
        f"graded={bucket['graded']:<5} hit={bucket['hit']:<4} "
        f"partial={bucket['partial']:<4} miss={bucket['miss']:<4} "
        f"ungraded={bucket['ungraded']:<4} "
        f"hit_rate={_fmt_rate(bucket['hit_rate'])} "
        f"partial_rate={_fmt_rate(bucket['partial_rate'])} "
        f"miss_rate={_fmt_rate(bucket['miss_rate'])}"
    )


def _notable_line(entry: Dict[str, Any]) -> str:
    week = entry["as_of_week"]
    week_text = f"week {week} call" if week is not None else "call"
    head = (f"- {entry['player']} ({entry['classification']}, "
            f"{week_text}, score {_fmt_score(entry['breakout_score'])})")
    if entry["ppg_delta"] is not None:
        detail = f"PPG delta {entry['ppg_delta']:+.1f}"
        if entry["baseline_ppg"] is not None \
                and entry["outcome_ppg"] is not None:
            detail += (f" ({entry['baseline_ppg']:g} to "
                       f"{entry['outcome_ppg']:g})")
        return f"{head}: {detail}"
    if entry["opp_delta"] is not None:
        return f"{head}: opportunities per game delta {entry['opp_delta']:+.1f}"
    if entry["snap_delta"] is not None:
        return f"{head}: snap share delta {entry['snap_delta']:+.1f} points"
    return f"{head}: no outcome deltas recorded"


def render_email(report: Dict[str, Any]) -> Tuple[str, str]:
    """Render the report dict as (subject, plain-text body)."""
    season = report["season"]
    version = report["scoring_version"]
    new = report["newly_graded"]
    bucket = new["bucket"]

    if report["total_graded_all_versions"] == 0:
        subject = "Breakout report: no calls graded yet"
    else:
        subject = (f"Breakout report: {bucket['graded']} calls graded "
                   f"this week")

    lines = [
        f"Weekly breakout report for season {season}",
        f"Grades recorded in the last {report['window_days']} days, "
        f"as of {report['as_of']:%Y-%m-%d}",
        f"Current scoring version: {version}",
        "",
        "Graded this week",
        "----------------",
    ]
    if report["nothing_new_line"] is not None:
        lines.append(report["nothing_new_line"])
    else:
        lines.append(
            f"{bucket['graded']} calls graded: {bucket['hit']} hits, "
            f"{bucket['partial']} partials, {bucket['miss']} misses "
            f"(hit rate {_fmt_rate(bucket['hit_rate'])}).")
        if bucket["ungraded"]:
            lines.append(
                f"{bucket['ungraded']} more calls in the window were "
                f"marked ungraded (no games or no baseline) and count "
                f"in neither direction.")
        for title, entries in (("Notable hits", new["notable_hits"]),
                               ("Notable misses", new["notable_misses"])):
            if entries:
                lines.append("")
                lines.append(f"{title}:")
                lines.extend(_notable_line(e) for e in entries)

    lines += ["", f"Season record so far ({version})", "-" * 28]
    overall = report["cumulative"]["overall"]
    if overall["graded"] == 0:
        lines.append(f"No graded calls yet under {version}.")
    else:
        lines.append(_bucket_line("overall", overall))
        lines.append("")
        lines.append("By classification:")
        lines.extend(
            _bucket_line(name, group)
            for name, group in
            report["cumulative"]["by_classification"].items())
        lines.append("")
        lines.append("By score:")
        lines.extend(
            _bucket_line(e["band"], e)
            for e in report["cumulative"]["score_bands"])
        lines.append("")
        lines.append("By confidence:")
        lines.extend(
            _bucket_line(e["band"], e)
            for e in report["cumulative"]["confidence_bands"])

    plan = report["suggested_changes"]
    lines += ["", "Suggested changes", "-----------------"]
    if plan["conversions"] or plan["investigations"]:
        lines.extend(plan["conversions"])
        lines.extend(plan["investigations"])
    else:
        lines.append(
            f"No changes suggested this week. {plan['no_change_reason']}")

    lines += [
        "",
        "Hand this email to Muse to apply the suggested changes.",
        "This report is read-only: it changes nothing by itself. Any "
        "change ships as a human-approved SCORING_VERSION bump.",
    ]
    return subject, "\n".join(lines)
