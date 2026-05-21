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
4. Load source stats from cache/player_history/usage_rows_{season}.json to reconstruct
   projected_usage (vacated opportunity + baseline) for each candidate.
5. Report:
   - Hit-rate calibration: among players predicted at each probability bucket,
     what fraction actually finished top-12?
   - PPR accuracy: mean absolute error and % within 10/20% of predicted value.
   - Brier score: probabilistic accuracy of hit_probability predictions.
   - Feature importance: Pearson r of each component score vs actual outcomes.
   - Precision@K: top-K candidates by breakout score, what % hit top-12?

Usage
-----
    python -m data_building.breakout_engine.backtest_multitask --season 2023
    python -m data_building.breakout_engine.backtest_multitask --season 2023 --min-score 30
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from dotenv import load_dotenv
load_dotenv()

# openai is transitively imported via the breakout engine's projections module.
# Mock it so this standalone script doesn't require the full AI stack.
from unittest.mock import MagicMock as _MagicMock
for _mod in ("openai", "openai.types", "openai.types.chat"):
    sys.modules.setdefault(_mod, _MagicMock())


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


def load_source_stats(season: int) -> dict[str, dict]:
    """
    Load full usage fields for the source/prediction season (for projected_usage
    reconstruction in reconstruct_multitask).
    Returns player_id -> usage dict.
    """
    cache_path = Path("cache/player_history") / f"usage_rows_{season}.json"
    if not cache_path.exists():
        return {}
    with open(cache_path) as f:
        rows = json.load(f)
    result = {}
    for r in rows:
        pid = str(r.get("id") or "")
        if pid:
            result[pid] = r.get("usage", {})
    return result


def load_position_map(outcome_season: int, from_json: str | None) -> dict[str, str]:
    """
    Load a player_id → position map.  Tries the supplemental file written by
    build_historical_scores (most accurate), then falls back to the candidate
    scores JSON, then gives up gracefully.
    """
    if from_json:
        pos_path = Path(from_json) / f"player_positions_{outcome_season}.json"
        if pos_path.exists():
            with open(pos_path) as f:
                raw = json.load(f)
            return {pid: v["position"] for pid, v in raw.items() if v.get("position")}
    return {}


def get_top12_pids(
    outcomes: dict[str, dict],
    position_map: dict[str, str] | None = None,
) -> set[str]:
    """Return the set of player IDs who finished top-12 at their position."""
    pm = position_map or {}
    by_pos: dict[str, list] = {}
    for pid, o in outcomes.items():
        if o["games"] < 10:
            continue
        pos = o["position"] or pm.get(pid, "")
        if not pos:
            continue
        by_pos.setdefault(pos, []).append((pid, o["total_ppr"]))

    top12: set[str] = set()
    for pos, players in by_pos.items():
        cutoff = 6 if pos in ("QB", "TE") else 12
        for pid, _ in sorted(players, key=lambda x: x[1], reverse=True)[:cutoff]:
            top12.add(pid)
    return top12


def get_breakout_pids(
    outcomes: dict[str, dict],
    source_stats: dict[str, dict] | None = None,
    position_map: dict[str, str] | None = None,
) -> set[str]:
    """
    A breakout = meaningfully better than the player's OWN prior season.

    Rules (≥10 games in outcome season required):
      - Prior baseline exists (≥6 games, ≥4 ppg):
          actual ≥ prior × 1.15  AND  actual ≥ 7.0 ppg
          (15% relative jump + minimum absolute floor)
      - No meaningful prior baseline (minimal/missing source data):
          actual ≥ 10.0 ppg  (they established themselves from scratch)

    Example: player at 12.1 ppg prior → breakout at ≥ 13.9 ppg (×1.15).
    """
    ss = source_stats or {}
    breakout: set[str] = set()

    for pid, o in outcomes.items():
        if o["games"] < 10:
            continue
        actual_ppg  = o["ppr_ppg"]
        prior       = ss.get(pid, {})
        prior_ppg   = float(prior.get("ppr_ppg") or 0)
        prior_games = int(prior.get("games") or 0)

        if prior_games >= 6 and prior_ppg >= 4.0:
            if actual_ppg >= prior_ppg * 1.15 and actual_ppg >= 7.0:
                breakout.add(pid)
        else:
            if actual_ppg >= 10.0:
                breakout.add(pid)

    return breakout


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


def load_breakout_scores_from_json(
    season: int, json_dir: str, min_score: float = 0.0
) -> list[dict]:
    """Load breakout scores from a JSON file produced by build_historical_scores --output-json."""
    path = Path(json_dir) / f"breakout_scores_{season}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"No JSON scores found at {path}. "
            f"Run build_historical_scores.py --season {season} --output-json {json_dir} first."
        )
    with open(path) as f:
        rows = json.load(f)
    return [r for r in rows if float(r.get("breakout_opportunity_score") or 0) >= min_score]


def _compute_projected_usage_from_details(
    position: str, prev_usage: dict, component_details: dict
) -> dict:
    """
    Reconstruct projected usage from stored component_details, mirroring the
    logic in build_historical_scores._compute_projected_usage().
    """
    opp = component_details.get("opportunity_opened", {})
    vac_targets = float(opp.get("vacated_targets", 0))
    vac_carries = float(opp.get("vacated_carries", 0))
    vac_snaps   = float(opp.get("vacated_snap_share", 0))

    prev_snap = float((prev_usage or {}).get("snap_share", 0))

    if position == "QB":
        opp_share = 0.90
    elif position == "RB":
        opp_share = 0.48 if prev_snap >= 0.55 else (0.32 if prev_snap >= 0.30 else 0.18)
    elif position in ("WR", "TE"):
        opp_share = 0.40 if prev_snap >= 0.70 else (0.27 if prev_snap >= 0.45 else 0.16)
    else:
        opp_share = 0.25

    return {
        "targets":    float((prev_usage or {}).get("targets", 0))  + vac_targets * opp_share,
        "carries":    float((prev_usage or {}).get("carries", 0))  + vac_carries * opp_share,
        "snap_share": min(float((prev_usage or {}).get("snap_share", 0)) + vac_snaps * opp_share, 0.95),
    }


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

    # Reconstruct projected_usage from stored opportunity signals + prev_usage baseline
    position = row.get("position", "WR")
    projected_usage = _compute_projected_usage_from_details(position, prev_usage or {}, cd)

    return compute_multitask_predictions(
        position=position,
        breakout_score=float(row.get("breakout_opportunity_score") or 0),
        readiness_score=float(row.get("player_readiness_score") or 0),
        confidence_score=float(row.get("confidence_score") or 0),
        role_trajectory_score=float(row.get("role_trajectory_score") or 0),
        projected_usage=projected_usage,
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


def pearson_r(xs: list[float], ys: list[float]) -> float:
    """Pearson correlation coefficient between two equal-length lists."""
    n = len(xs)
    if n < 3:
        return 0.0
    mx, my = sum(xs) / n, sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den_x = sum((x - mx) ** 2 for x in xs)
    den_y = sum((y - my) ** 2 for y in ys)
    den = math.sqrt(den_x * den_y)
    return num / den if den > 0 else 0.0


def feature_importance_report(feature_data: list[dict], hit_key: str = "is_top12") -> None:
    """
    Pearson r of each component score vs actual_ppg and vs a hit binary outcome.
    Higher |r| = stronger predictor.
    """
    if len(feature_data) < 5:
        return

    features = [
        ("breakout_score",          "Breakout opportunity score"),
        ("readiness_score",         "Player readiness"),
        ("confidence_score",        "Confidence"),
        ("role_trajectory_score",   "Role trajectory"),
        ("opportunity_opened_score","Opportunity opened"),
        ("competition_removed_score","Competition removed"),
        ("competition_added_penalty","Competition added (penalty)"),
        ("team_environment_score",  "Team environment"),
        ("hit_probability",         "Hit probability (derived)"),
    ]

    actual_ppg = [d["actual_ppg"] for d in feature_data]
    hit_vals   = [float(d.get(hit_key, 0)) for d in feature_data]
    hit_label  = "r→breakout" if hit_key == "is_breakout" else "r→top12"

    print("\n  Feature importance (Pearson r with actual outcomes):")
    print(f"  {'Feature':<30} {'r→ppg':>8}  {hit_label:>10}")
    print("  " + "-" * 54)

    rows_out = []
    for key, label in features:
        vals = [d.get(key) for d in feature_data]
        if any(v is None for v in vals):
            continue
        vals_f = [float(v) for v in vals]
        r_ppg = pearson_r(vals_f, actual_ppg)
        r_hit  = pearson_r(vals_f, hit_vals)
        rows_out.append((abs(r_ppg), label, r_ppg, r_hit))

    for _, label, r_ppg, r_hit in sorted(rows_out, reverse=True):
        print(f"  {label:<30} {r_ppg:>+.3f}    {r_hit:>+.3f}")


def precision_at_k_report(feature_data: list[dict], hit_key: str = "is_top12") -> None:
    """
    Of the top-K candidates ranked by breakout_score, what fraction hit the target outcome?
    """
    ranked = sorted(feature_data, key=lambda d: -d["breakout_score"])
    n = len(ranked)
    label = "breakout" if hit_key == "is_breakout" else "top-12"

    print(f"\n  Precision@K (top-K by breakout score → {label} hit rate):")
    print(f"  {'K':<6} {'Hits':>6} {'Precision':>10}")
    print("  " + "-" * 26)
    for k in [10, 20, 30, 50]:
        if k > n:
            break
        hits = sum(1 for d in ranked[:k] if d.get(hit_key))
        print(f"  {k:<6} {hits:>6}  {hits/k:>9.0%}")


def list_available_seasons() -> list[int]:
    """Return all seasons that have breakout scores in the DB."""
    from dashboard_services.db import get_conn
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT season, COUNT(DISTINCT player_id) as n "
            "FROM breakout_opportunity_scores GROUP BY season ORDER BY season"
        ).fetchall()
    return [(int(r["season"]), int(r["n"])) for r in rows]


def run_backtest(
    season: int,
    min_score: float = 0.0,
    verbose: bool = False,
    from_json: str | None = None,
) -> None:
    outcome_season = season + 1
    print(f"\n=== Multitask backtest: predicted for {season}, outcomes from {outcome_season} ===")

    print(f"Loading breakout scores for season {season}...")
    try:
        if from_json:
            candidates = load_breakout_scores_from_json(season, from_json, min_score)
        else:
            # DB path: check available seasons first
            try:
                available = list_available_seasons()
            except Exception as e:
                print(f"  ERROR connecting to DB: {e}")
                return
            available_seasons = [s for s, _ in available]
            if season not in available_seasons:
                print(f"  Season {season} not in DB. Available: {available_seasons}")
                return
            candidates = load_breakout_scores_from_db(season, min_score)
    except Exception as e:
        print(f"  ERROR loading scores: {e}")
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

    # Source stats (season N) used to reconstruct projected_usage per candidate
    source_stats = load_source_stats(season)
    print(f"  {len(source_stats)} players with source stats (season {season})")

    position_map = load_position_map(outcome_season, from_json)
    top12 = get_top12_pids(outcomes, position_map)
    breakouts = get_breakout_pids(outcomes, source_stats, position_map)
    print(f"  {len(top12)} top-12 finishers, {len(breakouts)} breakout players "
          f"(≥15% improvement from prior season; {len(position_map)} with position data)")

    # Reconstruct predictions and pair with outcomes
    hit_pairs_top12:    list[tuple[float, bool]] = []
    hit_pairs_breakout: list[tuple[float, bool]] = []
    # tuples of (predicted_ppg, actual_ppg, actual_games)
    ppg_pairs: list[tuple[float, float, float]] = []
    feature_data: list[dict] = []
    missed: list[str] = []

    for row in candidates:
        pid = str(row["player_id"])
        prev_usage = source_stats.get(pid)
        mt = reconstruct_multitask(row, prev_usage=prev_usage)

        actual = outcomes.get(pid)
        if actual is None or actual["games"] < 8:
            missed.append(f"{row.get('player_name','?')} ({pid})")
            continue

        hit_prob = mt["hit_probability"]
        is_top12   = pid in top12
        is_breakout = pid in breakouts
        cum_ppr  = mt["cumulative_ppr"]
        actual_ppg   = actual["ppr_ppg"]
        actual_games = float(actual["games"])

        if hit_prob is not None:
            hit_pairs_top12.append((hit_prob, is_top12))
            hit_pairs_breakout.append((hit_prob, is_breakout))

        if cum_ppr is not None and actual_ppg > 0:
            season1_ppr = mt.get("season1_ppr") or (cum_ppr / 2.0)
            predicted_ppg = season1_ppr / 17.0
            ppg_pairs.append((predicted_ppg, actual_ppg, actual_games))

        if verbose and hit_prob is not None:
            pos = row.get("position", "")
            prior_src   = source_stats.get(pid, {})
            prior_ppg   = float(prior_src.get("ppr_ppg") or 0)
            prior_games = int(prior_src.get("games") or 0)
            if is_breakout:
                hit_flag = "✓"
            elif prior_games >= 6 and prior_ppg >= 4.0 and actual_ppg >= prior_ppg * 1.08 and actual_ppg >= 7.0:
                hit_flag = "~"  # near miss: 8-15% improvement
            else:
                hit_flag = "✗"
            season1_ppr = mt.get("season1_ppr") or (cum_ppr / 2.0 if cum_ppr else 0)
            pred_ppg = season1_ppr / 17.0
            if prior_ppg > 0:
                delta = actual_ppg - prior_ppg
                outcome_str = f"prior={prior_ppg:.1f}  actual={actual_ppg:.1f} ({delta:+.1f})"
            else:
                outcome_str = f"actual={actual_ppg:.1f}"
            print(f"  {hit_flag} {row.get('player_name','?'):<22} {pos:<3} "
                  f"score={float(row.get('breakout_opportunity_score',0)):.0f}  "
                  f"hit_prob={hit_prob:.0%}  "
                  f"pred={pred_ppg:.1f}  {outcome_str}")

        # Collect feature data for correlation / precision reports
        if hit_prob is not None:
            feature_data.append({
                "breakout_score":           float(row.get("breakout_opportunity_score") or 0),
                "readiness_score":          float(row.get("player_readiness_score") or 0),
                "confidence_score":         float(row.get("confidence_score") or 0),
                "role_trajectory_score":    float(row.get("role_trajectory_score") or 0),
                "opportunity_opened_score": float(row.get("opportunity_opened_score") or 0),
                "competition_removed_score":float(row.get("competition_removed_score") or 0),
                "competition_added_penalty":float(row.get("competition_added_penalty") or 0),
                "team_environment_score":   float(row.get("team_environment_score") or 0),
                "hit_probability":          hit_prob,
                "actual_ppg":               actual_ppg,
                "is_top12":                 int(is_top12),
                "is_breakout":              int(is_breakout),
            })

    if not hit_pairs_top12:
        print("\n  No matched players with outcomes — check that the outcome season has data.")
        return

    n = len(hit_pairs_top12)
    print(f"\n  Matched: {n} players, unmatched: {len(missed)}")

    top12_rate    = sum(1 for _, h in hit_pairs_top12    if h) / n
    breakout_rate = sum(1 for _, h in hit_pairs_breakout if h) / n
    print(f"\n  Hit rates among candidates:")
    print(f"    Top-12 strict:       {top12_rate:.0%}  "
          f"(NFL-wide top-12; dominated by established stars)")
    print(f"    Breakout:            {breakout_rate:.0%}  "
          f"(≥15% improvement from prior season + ≥7 ppg floor; better signal for this candidate pool)")

    # --- Calibration vs breakout threshold (more meaningful) ---
    calibration_report(hit_pairs_breakout)

    # --- Brier score (breakout threshold) ---
    brier = sum((prob - int(hit)) ** 2 for prob, hit in hit_pairs_breakout) / n
    brier_naive = breakout_rate * (1 - breakout_rate)
    print(f"\n  Brier score (breakout): {brier:.4f}  "
          f"(naive baseline: {brier_naive:.4f}; "
          f"{'better' if brier < brier_naive else 'worse'} than naive by "
          f"{abs(brier_naive - brier):.4f})")

    # --- PPG accuracy ---
    ppr_accuracy_report(ppg_pairs, "Season-1 PPG accuracy (predicted vs actual PPG)")

    # --- Feature importance (vs breakout threshold) ---
    feature_importance_report(feature_data, hit_key="is_breakout")

    # --- Precision@K (by breakout threshold) ---
    precision_at_k_report(feature_data, hit_key="is_breakout")

    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backtest breakout multitask predictions")
    parser.add_argument("--season", type=int, default=2023,
                        help="Prediction season (outcomes loaded from season+1)")
    parser.add_argument("--min-score", type=float, default=30.0,
                        help="Minimum breakout score to include")
    parser.add_argument("--verbose", action="store_true",
                        help="Print per-player results")
    parser.add_argument("--from-json", metavar="DIR", default=None,
                        help="Load scores from JSON files in DIR instead of the database "
                             "(produced by build_historical_scores.py --output-json)")
    args = parser.parse_args()

    run_backtest(args.season, args.min_score, args.verbose, from_json=args.from_json)
