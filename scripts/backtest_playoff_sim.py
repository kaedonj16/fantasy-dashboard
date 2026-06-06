#!/usr/bin/env python3
"""
Calibration backtest for the playoff-odds simulator.

For each completed league-season, it predicts every team's playoff odds from the
preseason roster-projection model (the same code path the app uses in the
offseason, including per-week Sleeper projections, the injury model, and cached
manager efficiency) and compares those predictions to who ACTUALLY made the
playoffs. It then reports calibration buckets, a Brier score, and log loss so you
can tell whether, say, teams given ~60% odds really made it ~60% of the time —
and tune the hazard / std / blend with evidence instead of guesswork.

Usage:
    python scripts/backtest_playoff_sim.py \
        --leagues sleeper:1048291234567:2024 sleeper:998877665544:2023 \
        --sims 5000

Note: rosters are read as the platform currently reports them, so this best
reflects end-of-season rosters. It validates the model's structure and
calibration, not a true first-week-of-the-season forecast.
"""
from __future__ import annotations

import argparse
import math
import sys
from typing import List, Tuple


def calibration_report(samples: List[Tuple[float, bool]], n_bins: int = 10) -> str:
    """Build a calibration table + Brier score + log loss from (pred, actual)."""
    if not samples:
        return "No samples collected."

    bins: list[list] = [[] for _ in range(n_bins)]
    for prob, actual in samples:
        b = min(n_bins - 1, max(0, int(prob * n_bins)))
        bins[b].append((prob, 1.0 if actual else 0.0))

    brier = sum((p - a) ** 2 for p, a in samples) / len(samples)
    eps = 1e-6
    logloss = -sum(
        a * math.log(min(1 - eps, max(eps, p))) + (1 - a) * math.log(min(1 - eps, max(eps, 1 - p)))
        for p, a in ((p, 1.0 if a else 0.0) for p, a in samples)
    ) / len(samples)

    lines = []
    lines.append(f"{'bucket':>10} {'n':>5} {'pred':>7} {'actual':>7} {'gap':>7}")
    lines.append("-" * 40)
    for i, b in enumerate(bins):
        if not b:
            continue
        lo, hi = i * 100 // n_bins, (i + 1) * 100 // n_bins
        pred = sum(p for p, _ in b) / len(b) * 100
        act = sum(a for _, a in b) / len(b) * 100
        lines.append(
            f"{lo:>3}-{hi:<3}% {len(b):>5} {pred:>6.1f}% {act:>6.1f}% {act - pred:>+6.1f}"
        )
    lines.append("-" * 40)
    lines.append(f"samples={len(samples)}  Brier={brier:.4f}  LogLoss={logloss:.4f}")
    lines.append("(lower Brier/LogLoss is better; gap near 0 = well calibrated)")
    return "\n".join(lines)


def _made_playoffs(ctx: dict) -> Tuple[set, int]:
    """Set of roster_ids that actually made the playoffs from final standings."""
    from data_building.simulate_playoff_odds import _build_teams

    settings = ctx.get("league_settings") or {}
    playoff_teams = int(settings.get("playoff_teams") or 6)
    teams = _build_teams(ctx.get("team_stats"))
    ranked = sorted(teams, key=lambda t: (-t["wins"], -t["pf"]))
    return {t["roster_id"] for t in ranked[:playoff_teams]}, playoff_teams


def backtest_one(platform: str, league_id: str, season: int, sims: int) -> List[Tuple[float, bool]]:
    """Return (predicted_prob, made_playoffs) for every team in one league-season."""
    from app import build_league_context
    from data_building.simulate_playoff_odds import simulate_playoff_odds

    ctx = build_league_context(platform, league_id, season)
    team_stats = ctx.get("team_stats")
    if team_stats is None or team_stats.empty:
        print(f"  [skip] {platform}:{league_id}:{season} — no completed games", file=sys.stderr)
        return []

    made, _ = _made_playoffs(ctx)

    # Preseason-style prediction: drop realized games so the sim projects from
    # rosters + per-week projections + efficiency, exactly like the offseason path.
    pre_ctx = dict(ctx)
    pre_ctx["team_stats"] = None
    pre_ctx["current_week"] = 0
    preds = simulate_playoff_odds(pre_ctx, platform, n_sims=sims)

    samples: List[Tuple[float, bool]] = []
    for row in preds:
        rid = row["roster_id"]
        samples.append((float(row.get("playoff_pct", 0)) / 100.0, rid in made))
    return samples


def main() -> int:
    ap = argparse.ArgumentParser(description="Calibration backtest for the playoff simulator")
    ap.add_argument(
        "--leagues", nargs="+", required=True,
        help="One or more platform:league_id:season tuples (completed seasons)",
    )
    ap.add_argument("--sims", type=int, default=5000, help="Monte Carlo iterations")
    args = ap.parse_args()

    all_samples: List[Tuple[float, bool]] = []
    for spec in args.leagues:
        try:
            platform, league_id, season = spec.split(":")
        except ValueError:
            print(f"[error] bad --leagues entry '{spec}', expected platform:league_id:season", file=sys.stderr)
            continue
        print(f"Backtesting {platform}:{league_id}:{season} …", file=sys.stderr)
        try:
            all_samples.extend(backtest_one(platform, league_id, int(season), args.sims))
        except Exception as exc:
            print(f"  [error] {spec}: {exc}", file=sys.stderr)

    print(calibration_report(all_samples))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
