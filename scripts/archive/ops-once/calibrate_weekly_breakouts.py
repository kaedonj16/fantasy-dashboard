#!/usr/bin/env python3
"""
Calibration report for graded weekly breakout calls (self-learning v1).

Reads the season's finished weekly breakout grades and reports how the
engine's own calls actually did: hit rates by breakout-score band, by
confidence band, by classification, and by scoring version, an isotonic
fit of outcome value against breakout score, and plain-language findings
(monotonicity, emerging-band separation, confidence validity, and where
the fitted curve crosses a 50% hit value). Findings carry hard sample
guards: below 200 graded calls for the current scoring version (or 30
graded calls in a compared band) a check reports insufficient data with
the actual counts instead of a conclusion.

OPERATING RULE: this report informs a human-approved SCORING_VERSION
bump; it never changes scoring itself. It recommends no new thresholds,
applies nothing, and is read-only: it writes neither to the database nor
to any file.

Usage:
    python scripts/calibrate_weekly_breakouts.py
    python scripts/calibrate_weekly_breakouts.py --season 2026
    python scripts/calibrate_weekly_breakouts.py --season 2026 --json
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()


def _default_season() -> int:
    try:
        from data_building.sleeper_data import get_nfl_state
        state = get_nfl_state() or {}
        season = state.get("season")
        if season:
            return int(season)
    except Exception:
        pass
    from datetime import date
    return date.today().year


def load_graded_calls(season: int) -> list:
    """Every persisted grade row for the season, ALL scoring versions.

    Read-only: when the grades table does not exist yet the season simply
    has no graded calls, so this returns [] rather than creating anything.
    """
    from dashboard_services.db import get_conn
    from data_building.breakout_engine.weekly_grading import GRADES_TABLE

    with get_conn() as conn:
        found = conn.execute(
            "SELECT 1 AS x FROM information_schema.tables "
            "WHERE table_name = %s",
            (GRADES_TABLE,),
        ).fetchone()
        if not found:
            return []
        rows = conn.execute(
            f"SELECT * FROM {GRADES_TABLE} WHERE season = %s "
            f"ORDER BY scoring_version, as_of_week, player_id",
            (int(season),),
        ).fetchall()
    return [dict(r) for r in rows]


def build_report(season: int, rows: list) -> dict:
    """The calibration report for already-loaded grade rows. Pure.

    The headline band tables, curve, and findings cover the CURRENT
    scoring version only; the by-version table spans every version so an
    old version's record stays visible next to the current one.
    """
    from data_building.breakout_engine import calibration
    from data_building.breakout_engine.weekly_breakout import SCORING_VERSION

    current = [r for r in rows
               if str(r.get("scoring_version") or "") == SCORING_VERSION]
    bands_current = calibration.summarize_bands(current)
    bands_all = calibration.summarize_bands(rows)
    return {
        "season": int(season),
        "scoring_version": SCORING_VERSION,
        "graded_calls": {
            "current_version": bands_current["overall"]["graded"],
            "all_versions": bands_all["overall"]["graded"],
        },
        "current_version_summary": {
            "overall": bands_current["overall"],
            "score_bands": bands_current["score_bands"],
            "confidence_bands": bands_current["confidence_bands"],
            "by_classification": bands_current["by_classification"],
        },
        "by_scoring_version": bands_all["by_scoring_version"],
        "score_curve": calibration.fit_score_curve(current),
        "findings": calibration.findings(rows, SCORING_VERSION),
    }


def _fmt_rate(rate) -> str:
    return "n/a" if rate is None else f"{rate:.1%}"


def _bucket_line(label: str, bucket: dict) -> str:
    return (
        f"  {label:<28} calls={bucket['calls']:<5} "
        f"graded={bucket['graded']:<5} hit={bucket['hit']:<4} "
        f"partial={bucket['partial']:<4} miss={bucket['miss']:<4} "
        f"ungraded={bucket['ungraded']:<4} "
        f"hit_rate={_fmt_rate(bucket['hit_rate'])} "
        f"partial_rate={_fmt_rate(bucket['partial_rate'])} "
        f"miss_rate={_fmt_rate(bucket['miss_rate'])}"
    )


def format_report(report: dict) -> str:
    version = report["scoring_version"]
    graded = report["graded_calls"]
    summary = report["current_version_summary"]
    lines = [
        f"Weekly breakout calibration report for season {report['season']} "
        f"(read-only)",
        f"Current scoring version: {version} "
        f"({graded['current_version']} graded calls; "
        f"{graded['all_versions']} across all versions)",
        "",
        f"Overall ({version}):",
        _bucket_line("overall", summary["overall"]),
    ]
    for title, entries in (("By score", summary["score_bands"]),
                           ("By confidence", summary["confidence_bands"])):
        lines.append("")
        lines.append(f"{title} ({version}):")
        lines.extend(_bucket_line(e["band"], e) for e in entries)

    lines.append("")
    lines.append(f"By classification ({version}):")
    if summary["by_classification"]:
        lines.extend(
            _bucket_line(name, bucket)
            for name, bucket in summary["by_classification"].items())
    else:
        lines.append("  (no graded calls)")

    lines.append("")
    lines.append("By scoring version (all graded calls):")
    if report["by_scoring_version"]:
        lines.extend(
            _bucket_line(name, bucket)
            for name, bucket in report["by_scoring_version"].items())
    else:
        lines.append("  (no graded calls)")

    lines.append("")
    lines.append(f"Fitted score curve ({version}), score -> fitted hit value:")
    if report["score_curve"]:
        for point in report["score_curve"]:
            lines.append(
                f"  {point['score']:g} -> {point['probability']:.1%}")
    else:
        lines.append("  (no graded calls with a recorded score)")

    lines.append("")
    lines.append("Findings:")
    lines.extend(f"  - {finding}" for finding in report["findings"])
    lines.append("")
    lines.append(
        "This report never changes scoring. Any change ships as a "
        "human-approved SCORING_VERSION bump.")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season", type=int, default=None,
                        help="Season to calibrate (default: current NFL "
                             "season)")
    parser.add_argument("--json", action="store_true",
                        help="Emit the report as JSON instead of text")
    args = parser.parse_args()

    season = args.season if args.season is not None else _default_season()
    report = build_report(season, load_graded_calls(season))
    if args.json:
        print(json.dumps(report, indent=2, default=str))
    else:
        print(format_report(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
