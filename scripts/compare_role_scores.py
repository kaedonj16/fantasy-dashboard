#!/usr/bin/env python3
"""
A/B compare role_score v1 (calculate_role_score) vs v2 (percentile of the
team-relative opportunity index) on the current usage table.

Usage:
    python scripts/compare_role_scores.py            # top 25 per position by v2
    python scripts/compare_role_scores.py --pos WR   # one position
    python scripts/compare_role_scores.py --limit 50

Reads data/usage_table.json (built by the daily cron). Does not touch the DB,
so it is safe to run anywhere the usage table is present.
"""
from __future__ import annotations

import argparse

from data_building.advanced_metrics import (
    calculate_player_metrics,
    finalize_role_scores_v2,
)
from utils.utils import load_players_index, load_usage_table


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pos", choices=["QB", "RB", "WR", "TE"], help="limit to one position")
    ap.add_argument("--limit", type=int, default=25, help="rows per section")
    ap.add_argument("--movers", action="store_true",
                    help="rank by |v2 - v1| and show biggest risers + fallers")
    args = ap.parse_args()

    usage_table = load_usage_table()
    if not usage_table:
        print("No usage_table.json found — run the usage build first.")
        return
    names = {}
    for pid, meta in (load_players_index() or {}).items():
        names[str(pid)] = (meta.get("name") or "").strip()

    # v1 lives inside calculate_player_metrics; capture it before v2 overwrites.
    metrics = []
    for p in usage_table:
        pid, pos, usage = p.get("id"), p.get("position"), p.get("usage") or {}
        if not pid or pos not in ("QB", "RB", "WR", "TE") or (usage.get("games") or 0) == 0:
            continue
        try:
            metrics.append(calculate_player_metrics(str(pid), usage, pos))
        except Exception:
            continue
    v1 = {m["player_id"]: m.get("role_score") for m in metrics}

    finalize_role_scores_v2(metrics, usage_table)  # mutates role_score -> v2

    def line(m):
        pid = m["player_id"]
        old, new = v1.get(pid), m.get("role_score")
        d = (new - old) if (old is not None and new is not None) else None
        return (f"{m.get('position',''):4s}{names.get(pid, pid)[:22]:22s} "
                f"{('-' if old is None else f'{old:7.1f}')} "
                f"{('-' if new is None else f'{new:7.1f}')} "
                f"{('-' if d is None else f'{d:+8.1f}')}")

    if args.movers:
        rows = [m for m in metrics
                if (args.pos is None or m.get("position") == args.pos)
                and v1.get(m["player_id"]) is not None and m.get("role_score") is not None]
        rows.sort(key=lambda m: m["role_score"] - v1[m["player_id"]])
        hdr = f"{'pos':4s}{'player':22s}{'v1':>8s}{'v2':>8s}{'Δ':>9s}"
        print(f"\n=== BIGGEST RISERS (v2 - v1) ===\n{hdr}")
        for m in reversed(rows[-args.limit:]):
            print(line(m))
        print(f"\n=== BIGGEST FALLERS (v2 - v1) ===\n{hdr}")
        for m in rows[: args.limit]:
            print(line(m))
        return

    positions = [args.pos] if args.pos else ["QB", "RB", "WR", "TE"]
    for pos in positions:
        rows = [m for m in metrics if m.get("position") == pos]
        rows.sort(key=lambda m: (m.get("role_score") or 0), reverse=True)
        print(f"\n=== {pos} (top {args.limit} by v2) ===")
        print(f"{'player':22s} {'v1':>7s} {'v2':>7s} {'Δ':>7s}")
        for m in rows[: args.limit]:
            pid = m["player_id"]
            old = v1.get(pid)
            new = m.get("role_score")
            d = (new - old) if (old is not None and new is not None) else None
            print(f"{names.get(pid, pid)[:22]:22s} "
                  f"{('-' if old is None else f'{old:7.1f}')} "
                  f"{('-' if new is None else f'{new:7.1f}')} "
                  f"{('-' if d is None else f'{d:+7.1f}')}")


if __name__ == "__main__":
    main()
