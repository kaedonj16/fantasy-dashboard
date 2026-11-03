#!/usr/bin/env python3
"""Bucketed calibration check for live matchup win probability.

Reads the archived live snapshots in the ``matchup_moments`` table
(written by dashboard_services/sunday_drama.py on live polls) and scores
the probabilities that were actually shown to users: Brier score and
mean |predicted - outcome| per bucket, overall and split by how many
NFL games were still live at the snapshot (the closest time-left proxy
the table stores).

Outcome definition: snapshots are recorded while games are in progress,
so a matchup's LAST snapshot is at/near the final whistle. A group whose
last snapshot still reports live games is excluded (its score is not
final). Outcome is 1.0 / 0.0 / 0.5 from that snapshot's points.

What this cannot do, honestly: the table stores only the probability
that was shown (the pre-existing model's output), not the per-player
state behind it, so the NEW small-pool model cannot be replayed from
this data, and true "players left" buckets are impossible from it.
Scoring the new model on real Sundays needs a per-player live snapshot
capture joined to these rows; until that exists, the new model is
pinned by the scenario tests in tests/test_win_prob_small_pool.py and
this script tracks the shipped model's calibration drift.

Run it where the app database is reachable (e.g. the Render web shell,
where DATABASE_URL is set). From this repo's dev VM the production
database is not reachable and the script says so and exits non-zero.

Usage: python scripts/calibrate_win_prob.py [--season 2026]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _prob_band(p: float) -> str:
    for lo, hi in ((0.0, 0.1), (0.1, 0.3), (0.3, 0.5), (0.5, 0.7),
                   (0.7, 0.9), (0.9, 1.01)):
        if lo <= p < hi:
            return "%.1f-%.1f" % (lo, min(hi, 1.0))
    return "?"


def _games_band(g: int) -> str:
    if g <= 0:
        return "no games live"
    if g == 1:
        return "1 game live"
    return "2+ games live"


def _summarize(rows):
    """rows: list of (pred, outcome, games_live). Returns per-bucket
    (n, mean_pred, actual_rate, brier, mae) keyed by (games, band)."""
    buckets: dict = {}
    for pred, outcome, games_live in rows:
        key = (_games_band(int(games_live or 0)), _prob_band(float(pred)))
        b = buckets.setdefault(key, [0, 0.0, 0.0, 0.0, 0.0])
        b[0] += 1
        b[1] += float(pred)
        b[2] += float(outcome)
        b[3] += (float(pred) - float(outcome)) ** 2
        b[4] += abs(float(pred) - float(outcome))
    out = {}
    for key, (n, sp, so, sb, sa) in buckets.items():
        out[key] = (n, sp / n, so / n, sb / n, sa / n)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--season", default=None,
                    help="season filter, e.g. 2026 (default: all)")
    args = ap.parse_args()

    try:
        from dashboard_services.db import get_conn
    except Exception as exc:  # pragma: no cover - environment dependent
        print("cannot import the app DB layer: %s" % exc)
        return 2

    sql = (
        "SELECT league_id, season, week, matchup_key, observed_at, "
        "left_win_prob, left_pts, right_pts, games_live "
        "FROM matchup_moments"
    )
    params: list = []
    if args.season:
        sql += " WHERE season = %s"
        params.append(str(args.season))
    sql += " ORDER BY league_id, season, week, matchup_key, observed_at"

    try:
        with get_conn() as conn:
            rows = conn.execute(sql, params).fetchall()
    except Exception as exc:
        print("matchup_moments is not reachable from here (%s).\n"
              "The archived snapshots live only in the production "
              "database; run this script where DATABASE_URL points at "
              "it (e.g. the Render web shell)." % exc)
        return 2

    def _get(row, key, idx):
        return row[key] if isinstance(row, dict) else row[idx]

    # Group snapshots per matchup; outcome from each group's last row.
    groups: dict = {}
    for row in rows:
        key = (_get(row, "league_id", 0), _get(row, "season", 1),
               _get(row, "week", 2), _get(row, "matchup_key", 3))
        groups.setdefault(key, []).append(row)

    scored = []
    excluded = 0
    for key, snaps in groups.items():
        last = snaps[-1]
        if int(_get(last, "games_live", 8) or 0) > 0:
            excluded += 1
            continue
        lp = float(_get(last, "left_pts", 6))
        rp = float(_get(last, "right_pts", 7))
        outcome = 1.0 if lp > rp else (0.0 if lp < rp else 0.5)
        for s in snaps:
            scored.append((float(_get(s, "left_win_prob", 5)), outcome,
                           int(_get(s, "games_live", 8) or 0)))

    if not scored:
        print("no scorable snapshots (%d matchups, %d excluded as "
              "not-final)" % (len(groups), excluded))
        return 1

    buckets = _summarize(scored)
    n = len(scored)
    brier = sum((p - o) ** 2 for p, o, _ in scored) / n
    mae = sum(abs(p - o) for p, o, _ in scored) / n
    print("snapshots scored: %d across %d matchups (%d excluded)\n"
          "overall: Brier %.4f  mean|pred-outcome| %.4f\n"
          % (n, len(groups) - excluded, excluded, brier, mae))
    print("%-14s %-8s %6s %8s %8s %8s %8s"
          % ("games live", "prob", "n", "pred", "actual", "brier", "mae"))
    for (games, band), (bn, mp, ar, bb, mm) in sorted(buckets.items()):
        print("%-14s %-8s %6d %8.3f %8.3f %8.4f %8.4f"
              % (games, band, bn, mp, ar, bb, mm))
    print("\nNote: stored probabilities are the shipped model's outputs "
          "at poll time.\nBuckets by players-left are not possible from "
          "matchup_moments (it stores no per-player state).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
