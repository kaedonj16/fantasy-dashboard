#!/usr/bin/env python3
"""
Grade matured BR Assistant advice and (optionally) build the weekly
review packet.

Runs the stage 2 outcome graders (dashboard_services/assistant/grading.py)
over the advice log: start/sit calls once their week has finalized, and
buy/sell/hold calls at the 4-week mark. Grades land on the log rows
themselves; re-running is safe (already-graded rows are never rewritten).

With --packet it also writes Kaedon's weekly review packet (Markdown)
via dashboard_services/assistant/review_packet.py.

Usage:
    python3 scripts/grade_assistant_advice.py
    python3 scripts/grade_assistant_advice.py --dry-run
    python3 scripts/grade_assistant_advice.py --packet
    python3 scripts/grade_assistant_advice.py --packet --packet-dir /tmp/packets

Runs from the Render daily-job (cron) service shell, not the web shell:
it walks the advice log and the value history store, and the web dyno
should not spend its memory on batch grading. Suggested cadence: daily
for grading (start/sit grades land as soon as a week finalizes), and
once a week with --packet before the Wednesday review ritual.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:  # local .env loading is a convenience; env vars still apply
    pass


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true",
                        help="compute grades and print what would be written, write nothing")
    parser.add_argument("--limit", type=int, default=500,
                        help="max ungraded rows per intent per run (default 500)")
    parser.add_argument("--packet", action="store_true",
                        help="also generate the weekly review packet (Markdown file)")
    parser.add_argument("--packet-dir", default=None,
                        help="where the packet file goes (default: ASSISTANT_PACKET_DIR "
                             "env var, else ./assistant_review_packets)")
    args = parser.parse_args()

    from dashboard_services.assistant.grading import run_graders

    summary = run_graders(dry_run=args.dry_run, limit=args.limit)
    mode = "DRY RUN (nothing written)" if args.dry_run else "grades written"
    print(f"Assistant outcome grading: {mode}")
    for intent, c in summary.items():
        print(
            f"  {intent:<14} considered={c['considered']:<4} graded={c['graded']:<4} "
            f"hit={c['hit']:<4} partial={c['partial']:<4} miss={c['miss']:<4} "
            f"skipped={c['skipped']:<4}"
        )

    if args.packet:
        from dashboard_services.assistant.review_packet import generate_packet
        path, packet = generate_packet(out_dir=args.packet_dir)
        print(f"Review packet written: {path}")
        graded_total = sum(g["graded"] for g in packet["graded_by_type"].values())
        print(
            f"  packet: {sum(v['count'] for v in packet['volume_by_type'].values())} answers, "
            f"{graded_total} graded this cycle, {len(packet['worst_misses'])} misses listed, "
            f"{len(packet['draft_tickets'])} draft data tickets, "
            f"{len(packet['grading_debt_flags'])} grading debt flags"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
