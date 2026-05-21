"""
Backtest the multitask predictions (hit_probability, cumulative_ppr, peak_ppr)
against historical outcomes.

How it works
------------
1. Pull historical breakout component scores from the DB for a past season (e.g. 2023).
2. Re-run compute_multitask_predictions() on those scores — this works because the
   multitask predictions are derived purely from existing component scores, so we can
   reconstruct them retroactively for any season in the DB.
3. Load actual PPR outcomes from cache/player_history/usage_rows_{season+1}.json
   (season+1 because we're predicting next season's performance).
4. Report:
   - Hit-rate calibration: among players predicted at each probability bucket,
     what fraction actually finished top-12?
   - PPR accuracy: mean absolute error and % within 20% of predicted value.
   - Feature importance: which component scores correlate most with actual outcomes.

Usage
-----
    python -m data_building.breakout_engine.backtest_multitask --season 2023
    python -m data_building.breakout_engine.backtest_multitask --season 2023 --min-score 30
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from dotenv import load_dotenv
load_dotenv()


def load_actual_outcomes(season: int) -> dict[str, dict]:
    """
    Load actual PPR outcomes for a season from the cache files.
    Returns player_id -> {total_ppr, ppr_ppg, games, position}.
    """
    cache_path = Path("cache/player_history") / f"usage_rows_{season}.json"
    if not cache_path.exists():
        raise FileNotFoundError(f"No outcome data found at {cache_path}")

    with open(cache_path) as f:
        rows = json.load(f)

    outcomes = {}
    for r in rows:
        pid = str(r.get("id") or "")
        if not pid:
            continue
        u = r.get("usage", {})
        games = int(u.get("games") or 0)
        ppr_ppg = float(u.get("ppr_ppg") or 0)
        outcomes[pid] = {
            "total_ppr": round(ppr_ppg * games, 1),
            "ppr_ppg": ppr_ppg,
            "games": games,
            "position": r.get("position", ""),
            "name": r.get("name", ""),
        }
    return outcomes


def get_top12_pids(outcomes: dict[str, dict]) -> set[str]:
    """Return the set of player IDs who finished top-12 at their position."""
    by_pos: dict[str, list] = {}
    for pid, o in outcomes.items():
        if o["games"] < 10:
            continue
        by_pos.setdefault(o["position"], []).append((pid, o["total_ppr"]))

    top12: set[str] = set()
    for pos, players in by_pos.items():
        cutoff = 6 if pos in ("QB", "TE") else 12
        for pid, _ in sorted(players, key=lambda x: x[1], reverse=True)[:cutoff]:
            top12.add(pid)
    return top12


def load_breakout_scores_from_db(season: int, min_score: float = 0.0) -> list[dict]:
    """
    Pull the most recent breakout component scores per player for a season.
    Returns list of dicts with all component scores.
    """
    from dashboard_services.db import get_conn

    with get_conn() as conn:
        rows = conn.execute(
            """
            SELECT DISTINCT ON (player_id)
                player_id, player_name, position, team,
                breakout_opportunity_score,
                player_readiness_score,
                confidence_score,
                role_trajectory_score,
                opportunity_opened_score,
                competition_removed_score,
                competition_added_penalty,
                team_environment_score,
                component_details,
                season
            FROM breakout_opportunity_scores
            WHERE season = %s
              AND breakout_opportunity_score >= %s
            ORDER BY player_id, as_of_date DESC
            """,
            (season, min_score),
        ).fetchall()
    return [dict(r) for r in rows]


def reconstruct_multitask(row: dict, prev_usage: dict | None = None) -> dict:
    """Re-derive multitask predictions from stored component scores."""
    from data_building.breakout_engine.multitask_predictions import compute_multitask_predictions

    cd = row.get("component_details") or {}
    if isinstance(cd, str):
        try:
            cd = json.loads(cd)
        except Exception:
            cd = {}

    # Extract efficiency metrics from stored component_details
    readiness_details = cd.get("player_readiness", {})
    efficiency_metrics = {
        "yards_per_target": readiness_details.get("yards_per_target"),
        "yards_per_carry":  readiness_details.get("yards_per_carry"),
        "catch_rate":       readiness_details.get("catch_rate"),
    } if readiness_details else None

    age = readiness_details.get("age") if readiness_details else None

    return compute_multitask_predictions(
        position=row.get("position", "WR"),
        breakout_score=float(row.get("breakout_opportunity_score") or 0),
        readiness_score=float(row.get("player_readiness_score") or 0),
        confidence_score=float(row.get("confidence_score") or 0),
        role_trajectory_score=float(row.get("role_trajectory_score") or 0),
        projected_usage={},
        efficiency_metrics=efficiency_metrics,
        prev_usage=prev_usage,
        age=age,
    )


def calibration_report(predictions: list[tuple[float, bool]]) -> None:
    """Print hit-rate calibration: predicted probability vs actual hit rate per bucket."""
    buckets: dict[str, list[bool]] = {}
    for prob, hit in predictions:
        label = f"{int(prob * 10) * 10:02d}-{int(prob * 10) * 10 + 10:02d}%"
        buckets.setdefault(label, []).append(hit)

    print("\n  Hit-rate calibration (predicted prob vs actual hit rate):")
    print(f"  {'Bucket':<10} {'Predicted':>10} {'Actual':>10} {'N':>6} {'Δ':>8}")
    print("  " + "-" * 48)
    for label in sorted(buckets):
        vals = buckets[label]
        pred_mid = (int(label[:2]) + 5) / 100
        actual = sum(vals) / len(vals)
        delta = actual - pred_mid
        flag = " ◄ overconfident" if delta < -0.08 else (" ◄ underconfident" if delta > 0.08 else "")
        print(f"  {label:<10} {pred_mid:>9.0%} {actual:>9.0%} {len(vals):>6}  {delta:>+.0%}{flag}")


def ppr_accuracy_report(pairs: list[tuple[float, float, float]], label: str) -> None:
    """
    Print accuracy report comparing predicted vs actual PPG.
    pairs: list of (predicted_ppg, actual_ppg, actual_games)
    Primary metric: within ±10% of actual PPG.
    """
    if not pairs:
        return
    errors = [abs(pred - actual) for pred, actual, _ in pairs]
    within10 = sum(1 for pred, actual, _ in pairs if abs(pred - actual) / max(actual, 0.1) <= 0.10) / len(pairs)
    within20 = sum(1 for pred, actual, _ in pairs if abs(pred - actual) / max(actual, 0.1) <= 0.20) / len(pairs)
    mean_pred = sum(p for p, _, _ in pairs) / len(pairs)
    mean_actual = sum(a for _, a, _ in pairs) / len(pairs)
    print(f"\n  {label}:")
    print(f"    N={len(pairs)}, MAE={sum(errors)/len(errors):.2f} ppg, "
          f"within ±10%: {within10:.0%},  within ±20%: {within20:.0%}")
    print(f"    mean predicted: {mean_pred:.1f} ppg,  mean actual: {mean_actual:.1f} ppg  "
          f"(bias: {mean_pred - mean_actual:+.1f})")


def run_backtest(season: int, min_score: float = 0.0, verbose: bool = False) -> None:
    outcome_season = season + 1
    print(f"\n=== Multitask backtest: predicted for {season}, outcomes from {outcome_season} ===")

    print(f"Loading breakout scores for season {season}...")
    try:
        candidates = load_breakout_scores_from_db(season, min_score)
    except Exception as e:
        print(f"  ERROR loading DB scores: {e}")
        return
    print(f"  {len(candidates)} candidates (min_score={min_score})")

    print(f"Loading actual outcomes for season {outcome_season}...")
    try:
        outcomes = load_actual_outcomes(outcome_season)
    except FileNotFoundError as e:
        print(f"  ERROR: {e}")
        print(f"  Available outcome files:")
        for p in sorted(Path("cache/player_history").glob("usage_rows_*.json")):
            print(f"    {p}")
        return
    print(f"  {len(outcomes)} players with outcomes")

    top12 = get_top12_pids(outcomes)
    print(f"  {len(top12)} top-12 finishers across all positions")

    # Reconstruct predictions and pair with outcomes
    hit_pairs: list[tuple[float, bool]] = []
    # tuples of (predicted_ppg, actual_ppg, actual_games)
    ppg_pairs: list[tuple[float, float, float]] = []
    missed: list[str] = []

    for row in candidates:
        pid = str(row["player_id"])
        mt = reconstruct_multitask(row)

        actual = outcomes.get(pid)
        if actual is None or actual["games"] < 8:
            missed.append(f"{row.get('player_name','?')} ({pid})")
            continue

        hit_prob = mt["hit_probability"]
        if hit_prob is not None:
            hit_pairs.append((hit_prob, pid in top12))

        cum_ppr = mt["cumulative_ppr"]
        actual_ppg = actual["ppr_ppg"]
        actual_games = float(actual["games"])
        if cum_ppr is not None and actual_ppg > 0:
            # Convert cumulative 2-season estimate → per-game PPG for season 1.
            # Assume a 17-game season; cumulative / 2 gives season-1 total.
            predicted_ppg = (cum_ppr / 2.0) / 17.0
            ppg_pairs.append((predicted_ppg, actual_ppg, actual_games))

        if verbose and hit_prob is not None:
            hit_flag = "✓" if pid in top12 else "✗"
            pred_ppg = (cum_ppr / 2.0 / 17.0) if cum_ppr else 0
            print(f"  {hit_flag} {row.get('player_name','?'):<22} "
                  f"score={float(row.get('breakout_opportunity_score',0)):.0f}  "
                  f"hit_prob={hit_prob:.0%}  "
                  f"pred_ppg={pred_ppg:.1f}  actual_ppg={actual_ppg:.1f}")

    if not hit_pairs:
        print("\n  No matched players with outcomes — check that the outcome season has data.")
        return

    print(f"\n  Matched: {len(hit_pairs)} players, unmatched: {len(missed)}")

    # --- Hit probability calibration ---
    actual_top12_rate = sum(1 for _, h in hit_pairs if h) / len(hit_pairs)
    print(f"\n  Overall top-12 hit rate among candidates: {actual_top12_rate:.0%} "
          f"(base rate in full player pool ~15-22%)")

    calibration_report(hit_pairs)

    # --- PPG accuracy ---
    ppr_accuracy_report(ppg_pairs, "Season-1 PPG accuracy (predicted vs actual PPG)")

    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backtest breakout multitask predictions")
    parser.add_argument("--season", type=int, default=2023,
                        help="Prediction season (outcomes loaded from season+1)")
    parser.add_argument("--min-score", type=float, default=30.0,
                        help="Minimum breakout score to include")
    parser.add_argument("--verbose", action="store_true",
                        help="Print per-player results")
    args = parser.parse_args()

    run_backtest(args.season, args.min_score, args.verbose)
