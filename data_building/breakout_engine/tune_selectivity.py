#!/usr/bin/env python3
"""
Tune breakout selectivity (curve pivot/slope) against your real data.

The breakout list is pared down by two mechanisms in
PhaseDetector.calculate_aggregate_score:

  1. Qualification gates  - a player must have a real opportunity AND the
                            readiness/trajectory to seize it.
  2. A score curve        - pivot + stretch so the mediocre middle drops below
                            the page's candidate floor (50).

This script reads the stored per-component scores from
breakout_opportunity_scores, then replays the gate + curve logic for a grid of
(pivot, slope) values and prints how many candidates each pair would surface —
so you can pick the pair that lands in your target range WITHOUT rebuilding the
scores repeatedly.

Usage:
    python -m data_building.breakout_engine.tune_selectivity [season] [floor]

    season  optional, defaults to the current NFL-state season
    floor   optional, the page candidate cutoff (defaults to 50)

Once you find a good pair, set BREAKOUT_CURVE_PIVOT / BREAKOUT_CURVE_SLOPE in
config.py and rebuild once:
    python -m data_building.breakout_engine.calculate_breakouts_with_real_data
"""

import sys

from dashboard_services.db import get_conn
from data_building.breakout_engine import phases as _phases
from data_building.breakout_engine.config import (
    BREAKOUT_CURVE_PIVOT,
    BREAKOUT_CURVE_SLOPE,
)

PhaseDetector = _phases.PhaseDetector

# Map DB columns -> the component names calculate_aggregate_score expects.
_COMPONENT_COLS = {
    "opportunity_opened": "opportunity_opened_score",
    "competition_removed": "competition_removed_score",
    "competition_added_penalty": "competition_added_penalty",
    "team_environment": "team_environment_score",
    "player_readiness": "player_readiness_score",
    "role_trajectory": "role_trajectory_score",
    "confidence": "confidence_score",
}


def _load_candidates(season: int):
    """Return [(phase, {component_scores...}), ...] for the season."""
    cols = ", ".join(sorted(set(_COMPONENT_COLS.values())))
    query = f"""
        SELECT DISTINCT ON (player_id)
            player_id, phase, {cols}
        FROM breakout_opportunity_scores
        WHERE season = %s
        ORDER BY player_id, as_of_date DESC, calculated_at DESC
    """
    rows = []
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(query, [season])
        for r in cur.fetchall():
            phase = r.get("phase") or "post_draft"
            comp = {
                name: float(r.get(col) or 0.0)
                for name, col in _COMPONENT_COLS.items()
            }
            rows.append((phase, comp))
    return rows


def _count_for(rows, pivot: float, slope: float, floor: float) -> int:
    """Replay the real gate+curve logic with the given pivot/slope."""
    orig_p, orig_s = _phases.BREAKOUT_CURVE_PIVOT, _phases.BREAKOUT_CURVE_SLOPE
    _phases.BREAKOUT_CURVE_PIVOT = pivot
    _phases.BREAKOUT_CURVE_SLOPE = slope
    try:
        return sum(
            1
            for phase, comp in rows
            if PhaseDetector.calculate_aggregate_score(comp, phase) >= floor
        )
    finally:
        _phases.BREAKOUT_CURVE_PIVOT = orig_p
        _phases.BREAKOUT_CURVE_SLOPE = orig_s


def main():
    season = None
    floor = 50.0
    if len(sys.argv) > 1:
        season = int(sys.argv[1])
    if len(sys.argv) > 2:
        floor = float(sys.argv[2])

    if season is None:
        try:
            from dashboard_services.api import get_nfl_state
            season = int((get_nfl_state() or {}).get("season", 2026))
        except Exception:
            season = 2026

    rows = _load_candidates(season)
    print(f"Season {season}: {len(rows)} scored players in the DB.")
    if not rows:
        print("No rows found — rebuild scores first.")
        return

    current = _count_for(rows, BREAKOUT_CURVE_PIVOT, BREAKOUT_CURVE_SLOPE, floor)
    print(
        f"Current config (pivot={BREAKOUT_CURVE_PIVOT}, "
        f"slope={BREAKOUT_CURVE_SLOPE}) -> {current} candidates at floor {floor:.0f}\n"
    )

    pivots = [48, 50, 52, 54, 56, 58, 60]
    slopes = [1.4, 1.6, 1.8, 2.0, 2.2]
    print("Candidate counts by pivot (rows) x slope (cols):")
    print("pivot\\slope " + "".join(f"{s:>7}" for s in slopes))
    for p in pivots:
        cells = "".join(f"{_count_for(rows, p, s, floor):>7}" for s in slopes)
        print(f"{p:>10} {cells}")
    print(
        "\nPick the pivot/slope landing in your target range, set them in "
        "config.py (BREAKOUT_CURVE_PIVOT / BREAKOUT_CURVE_SLOPE), then rebuild."
    )


if __name__ == "__main__":
    main()
