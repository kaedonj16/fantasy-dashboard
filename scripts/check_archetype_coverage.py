#!/usr/bin/env python3
"""
Check whether player_advanced_metrics has the archetype fields (aDOT, slot/wide/
inline rate, YAC) populated for the NFL WR/TE we'd use in breakout role-fit.

Run on a machine with DB access:
    python scripts/check_archetype_coverage.py [season]

Prints per-season coverage counts and a few sample players so you can confirm
the data is real and discriminating before we wire archetype matching in.
"""
import sys
from dashboard_services.db import get_conn

ARCHETYPE_COLS = [
    "avg_depth_of_target", "slot_rate", "wide_rate", "inline_rate",
    "yards_after_catch_per_reception", "contested_catch_rate",
]


def main():
    season = int(sys.argv[1]) if len(sys.argv) > 1 else None

    with get_conn() as conn:
        cur = conn.cursor()

        # Per-season coverage: how many WR/TE rows have non-null archetype fields.
        nonnull = " AND ".join(f"{c} IS NOT NULL" for c in ARCHETYPE_COLS)
        cur.execute(f"""
            SELECT season,
                   COUNT(*) FILTER (WHERE position IN ('WR','TE')) AS wr_te_rows,
                   COUNT(*) FILTER (WHERE position IN ('WR','TE') AND {nonnull}) AS with_archetype
            FROM player_advanced_metrics
            GROUP BY season
            ORDER BY season
        """)
        print(f"{'season':<8}{'WR/TE rows':>12}{'w/ archetype':>14}")
        for r in cur.fetchall():
            r = dict(r)
            print(f"{str(r['season']):<8}{r['wr_te_rows']:>12}{r['with_archetype']:>14}")

        # Sample some well-known names for the target season to eyeball values.
        target = season or "(SELECT MAX(season) FROM player_advanced_metrics)"
        cur.execute(f"""
            SELECT DISTINCT ON (player_id)
                   player_id, position, season, as_of_date,
                   avg_depth_of_target, slot_rate, wide_rate, inline_rate,
                   yards_after_catch_per_reception, contested_catch_rate
            FROM player_advanced_metrics
            WHERE position IN ('WR','TE')
              AND season = {target if season else '(SELECT MAX(season) FROM player_advanced_metrics)'}
              AND avg_depth_of_target IS NOT NULL
            ORDER BY player_id, as_of_date DESC
            LIMIT 15
        """)
        print("\nSample WR/TE archetype rows (target season):")
        print(f"{'player_id':<14}{'pos':<4}{'aDOT':>6}{'slot%':>7}{'wide%':>7}{'inline%':>8}{'yac/r':>7}")
        for r in cur.fetchall():
            r = dict(r)
            def f(x): return f"{float(x):.1f}" if x is not None else "-"
            print(f"{str(r['player_id']):<14}{r['position']:<4}{f(r['avg_depth_of_target']):>6}"
                  f"{f(r['slot_rate']):>7}{f(r['wide_rate']):>7}{f(r['inline_rate']):>8}"
                  f"{f(r['yards_after_catch_per_reception']):>7}")


if __name__ == "__main__":
    main()
