"""Compare headless CPU draft results with a production real-draft audit.

Example:
    python scripts/compare_draft_realism.py \
      --real artifacts/draft-realism.json \
      --cpu artifacts/cpu-redraft-1qb.json \
      --output artifacts/draft-realism-comparison.md \
      --json artifacts/draft-realism-comparison.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

POSITIONS = ("QB", "RB", "WR", "TE", "K", "DEF")
PHASES = {"early": "early (1-4)", "middle": "middle (5-9)", "late": "late (10+)"}


def load_report(path: Path, label: str) -> object:
    """Load an input report with an actionable error for ephemeral Render shells."""
    if not path.exists():
        nearby = sorted(path.parent.glob("*.json")) if path.parent.exists() else []
        found = ", ".join(str(item) for item in nearby) or "none"
        raise FileNotFoundError(
            f"{label} report not found: {path}. JSON files in {path.parent}: {found}. "
            "Render shell files are ephemeral across deploys; regenerate this report in the current shell."
        )
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} report is not valid JSON: {path} ({exc})") from exc


def cohort_key(draft_type: str, superflex: bool) -> tuple[str, str]:
    dtype = "dynasty startup" if draft_type == "startup" else draft_type
    return dtype, "Superflex" if superflex else "1QB"


def compare(real_rows: list[dict], cpu: dict) -> dict:
    cfg = cpu["configuration"]
    key = cohort_key(cfg["type"], bool(cfg.get("sf")))
    real = next((row for row in real_rows if (row["draft_type"], row["format"]) == key), None)
    if real is None:
        raise ValueError(f"Real audit has no {key[0]} — {key[1]} cohort")
    timing_real = {**real.get("qb_rounds", {}), **real.get("te_rounds", {})}
    timing = {name: {"real": timing_real.get(name), "cpu": value,
                     "delta": None if value is None or timing_real.get(name) is None else round(value - timing_real[name], 2)}
              for name, value in cpu.get("medianRound", {}).items()}
    counts = {pos: {"real": real.get("position_counts", {}).get(pos, 0),
                    "cpu": cpu.get("medianFinalCount", {}).get(pos, 0),
                    "delta": round(cpu.get("medianFinalCount", {}).get(pos, 0)
                                   - real.get("position_counts", {}).get(pos, 0), 2)}
              for pos in POSITIONS}
    phase = {}
    for short, real_name in PHASES.items():
        phase[short] = {pos: {"real": real.get("phase_shares", {}).get(real_name, {}).get(pos, 0),
                              "cpu": cpu.get("phaseShare", {}).get(short, {}).get(pos, 0),
                              "delta": round(cpu.get("phaseShare", {}).get(short, {}).get(pos, 0)
                                             - real.get("phase_shares", {}).get(real_name, {}).get(pos, 0), 2)}
                        for pos in POSITIONS}
    absolute_deltas = [abs(row["delta"]) for row in timing.values() if row["delta"] is not None]
    absolute_deltas += [abs(row["delta"]) for row in counts.values()]
    absolute_deltas += [abs(row["delta"]) for rows in phase.values() for row in rows.values()]
    warnings = []
    if real.get("drafts", 0) < 200:
        warnings.append("Real cohort has fewer than 200 drafts; treat small deltas as directional.")
    for pos, setting in (("K", cfg.get("k", 0)), ("DEF", cfg.get("def", 0))):
        if setting and real.get("position_counts", {}).get(pos, 0) == 0:
            warnings.append(f"CPU requires {setting} {pos}, but the real cohort median is 0; "
                            "this is a mixed-roster comparison, not a like-for-like special-teams calibration.")
    if abs(float(cfg.get("rounds", 0)) - float(real.get("median_rounds", cfg.get("rounds", 0)))) > 1:
        warnings.append("CPU rounds differ materially from the real cohort median draft length.")
    return {"cohort": {"draftType": key[0], "format": key[1], "realDrafts": real.get("drafts", 0),
                       "cpuDrafts": cfg.get("drafts", 0)}, "timing": timing, "finalCounts": counts,
            "phaseShare": phase, "invariants": cpu.get("invariants", {}),
            "waitingCalibration": cpu.get("waitingCalibration", {}),
            "warnings": warnings,
            "meanAbsoluteDelta": round(sum(absolute_deltas) / max(1, len(absolute_deltas)), 2)}


def markdown(report: dict) -> str:
    cohort = report["cohort"]
    lines = ["# CPU vs Real Draft Comparison", "",
             f"**{cohort['draftType'].title()} — {cohort['format']}** · "
             f"{cohort['realDrafts']:,} real drafts vs {cohort['cpuDrafts']:,} CPU drafts", "",
             f"- Mean absolute metric delta: **{report['meanAbsoluteDelta']:.2f}**", "",
             "## Position timing", "", "| Metric | Real | CPU | Delta |", "|---|---:|---:|---:|"]
    if report.get("warnings"):
        lines[6:6] = ["## Comparison warnings", ""] + [f"- ⚠ {warning}" for warning in report["warnings"]] + [""]
    for name, row in report["timing"].items():
        lines.append(f"| {name} | {row['real'] if row['real'] is not None else '—'} | "
                     f"{row['cpu'] if row['cpu'] is not None else '—'} | "
                     f"{row['delta'] if row['delta'] is not None else '—'} |")
    lines += ["", "## Final roster counts", "", "| Position | Real | CPU | Delta |", "|---|---:|---:|---:|"]
    for pos, row in report["finalCounts"].items():
        lines.append(f"| {pos} | {row['real']} | {row['cpu']} | {row['delta']} |")
    lines += ["", "## Safety invariants", ""]
    lines += [f"- **{name}:** {value}" for name, value in report["invariants"].items()]
    lines += ["", "## Value-of-waiting calibration", "", "| Predicted bin | Samples | Predicted | Actual |",
              "|---|---:|---:|---:|"]
    for name, row in report["waitingCalibration"].items():
        lines.append(f"| {name} | {row['samples']} | {row['predictedPct']}% | {row['actualPct']}% |")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real", type=Path, required=True, help="JSON from audit_draft_realism.py")
    parser.add_argument("--cpu", type=Path, required=True, help="JSON from benchmark_cpu_drafts.js")
    parser.add_argument("--output", type=Path, help="Write Markdown comparison")
    parser.add_argument("--json", type=Path, dest="json_output", help="Write machine-readable comparison")
    parser.add_argument("--max-mean-delta", type=float, help="Exit 1 when mean absolute delta exceeds this")
    args = parser.parse_args(argv)
    try:
        report = compare(load_report(args.real, "Real-draft audit"), load_report(args.cpu, "CPU benchmark"))
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))
    text = markdown(report)
    print(text, end="")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(report, indent=2) + "\n")
    return int(args.max_mean_delta is not None and report["meanAbsoluteDelta"] > args.max_mean_delta)


if __name__ == "__main__":
    raise SystemExit(main())
