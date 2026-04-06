"""
Historical backtesting for the breakout detection model.

Evaluates how well the breakout scoring engine would have predicted
actual breakouts in past seasons (2021-2024).

A "breakout" is defined as a player who:
  - Was NOT a top-12 at their position the prior season (position rank 13+), OR
    was a rookie, AND
  - Achieved a top-12 finish at their position the FOLLOWING season.

Usage:
    python -m data_building.breakout_engine.backtest_breakout_model \
        --seasons 2022 2023 2024 \
        --min-score 40 \
        --output backtest_results.json

Outputs:
  - Precision / Recall / F1 by phase and position
  - AUC-ROC for breakout_opportunity_score
  - Top false positives and false negatives for manual review
  - Per-component feature importance (point-biserial correlation)
"""

import argparse
import json
import os
import sys
from dataclasses import dataclass, asdict
from datetime import date
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Optional heavy deps — only needed if running full eval
# ---------------------------------------------------------------------------
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

try:
    from scipy import stats as scipy_stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    from sklearn.metrics import (
        roc_auc_score,
        precision_recall_fscore_support,
        average_precision_score,
    )
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

# ---------------------------------------------------------------------------
# Internal imports
# ---------------------------------------------------------------------------
from .db_helpers import (
    batch_load_all_breakout_data,
    load_all_player_usage,
    load_all_team_stats,
)
from .core import BreakoutEngine


# ==============================================================================
# DATA STRUCTURES
# ==============================================================================

@dataclass
class BacktestRecord:
    """Single player-season evaluation record."""
    player_id: str
    player_name: str
    team: str
    position: str
    season: int                       # Season being predicted (e.g. 2023)
    as_of_date: date                  # Date scores were calculated
    phase: str

    # Model outputs
    breakout_opportunity_score: float
    opportunity_opened_score: float
    competition_removed_score: float
    competition_added_penalty: float
    team_environment_score: float
    player_readiness_score: float
    role_trajectory_score: float
    confidence_score: float

    # Ground truth
    prior_position_rank: Optional[int]   # Previous season fantasy rank at position
    actual_position_rank: Optional[int]  # This season's actual fantasy rank
    is_actual_breakout: bool             # Ground truth label


@dataclass
class BacktestMetrics:
    """Aggregate metrics for a single season/phase/position cohort."""
    season: int
    position: str
    phase: str
    n_players: int
    n_breakouts: int
    n_predicted_breakouts: int          # Players with score >= threshold

    precision: float
    recall: float
    f1: float
    auc_roc: float
    avg_precision: float

    # Top false positives / negatives for debugging
    false_positives: List[Dict]         # High score but didn't break out
    false_negatives: List[Dict]         # Low score but broke out anyway

    # Per-component correlations with breakout label
    component_correlations: Dict[str, float]


# ==============================================================================
# BREAKOUT LABEL GENERATION
# ==============================================================================

BREAKOUT_RANK_THRESHOLD = 12   # Top-12 at position = "broke out"
PRIOR_NON_STARTER_RANK = 13    # Must have been ranked 13+ previously (or rookie)
TOP_N_FALSE_EXAMPLES = 10      # How many false pos/neg to surface in report


def _load_historical_fantasy_rankings(season: int) -> Dict[str, Dict]:
    """
    Load historical fantasy finish rankings from cache.

    Looks for cache/player_history/fantasy_rankings_{season}.json.
    Each entry: {player_id: {position_rank: int, total_points: float}}

    Falls back to usage-derived proxy (targets + carries rank) if no
    dedicated rankings file exists.

    Returns:
        Dict mapping player_id (str) → {'position_rank': int, 'total_points': float}
    """
    rankings_path = os.path.join(
        "cache", "player_history", f"fantasy_rankings_{season}.json"
    )

    if os.path.exists(rankings_path):
        try:
            with open(rankings_path, "r") as f:
                data = json.load(f)
            print(f"[backtest] Loaded {len(data)} fantasy rankings for {season}")
            return {str(k): v for k, v in data.items()}
        except (json.JSONDecodeError, IOError) as e:
            print(f"[backtest] Warning: could not load rankings for {season}: {e}")

    # Fallback: derive proxy ranking from usage (targets + carries as opportunity)
    usage_path = os.path.join(
        "cache", "player_history", f"usage_rows_{season}.json"
    )
    if not os.path.exists(usage_path):
        print(f"[backtest] No usage data found for {season}, skipping")
        return {}

    try:
        with open(usage_path, "r") as f:
            usage_data = json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        print(f"[backtest] Warning: could not load usage for {season}: {e}")
        return {}

    # Build position groups and sort by estimated fantasy points
    by_position: Dict[str, List] = {}
    for player in usage_data:
        pos = player.get("position", "")
        if pos not in ["QB", "RB", "WR", "TE"]:
            continue
        usage = player.get("usage") or {}
        games = max(usage.get("games", 0) or 0, 1)

        # Estimate PPR fantasy points from usage
        targets = (usage.get("avg_targets") or 0) * games
        receptions = (usage.get("avg_receptions") or 0) * games
        carries = (usage.get("avg_carries") or 0) * games
        rush_yards = (usage.get("avg_rush_yards") or 0) * games
        rec_yards = (usage.get("avg_rec_yards") or 0) * games
        rush_tds = (usage.get("avg_rush_tds") or 0) * games
        rec_tds = (usage.get("avg_rec_tds") or 0) * games
        pass_yards = (usage.get("avg_pass_yards") or 0) * games
        pass_tds = (usage.get("avg_pass_tds") or 0) * games
        ints = (usage.get("avg_interceptions") or 0) * games

        pts = (
            rush_yards * 0.1 + rush_tds * 6 +
            rec_yards * 0.1 + rec_tds * 6 + receptions * 1 +
            pass_yards * 0.04 + pass_tds * 4 - ints * 2
        )

        by_position.setdefault(pos, []).append({
            "player_id": str(player.get("id")),
            "total_points": round(pts, 1),
        })

    rankings: Dict[str, Dict] = {}
    for pos, players in by_position.items():
        players.sort(key=lambda x: x["total_points"], reverse=True)
        for rank, p in enumerate(players, start=1):
            rankings[p["player_id"]] = {
                "position_rank": rank,
                "total_points": p["total_points"],
                "position": pos,
            }

    print(f"[backtest] Derived proxy rankings for {season}: {len(rankings)} players")
    return rankings


def _label_breakouts(
        player_list: List[Dict],
        prior_rankings: Dict[str, Dict],
        current_rankings: Dict[str, Dict],
        top_n: int = BREAKOUT_RANK_THRESHOLD,
        prior_non_starter: int = PRIOR_NON_STARTER_RANK,
) -> List[Dict]:
    """
    Assign is_actual_breakout=True for players who:
      1. Were outside top-(prior_non_starter) at their position last season
         OR had no prior season data (rookie), AND
      2. Finished inside top-(top_n) at their position THIS season.

    Adds 'prior_position_rank', 'actual_position_rank', 'is_actual_breakout'
    to each player dict.
    """
    labeled = []
    for player in player_list:
        pid = str(player.get("player_id"))

        prior = prior_rankings.get(pid, {})
        current = current_rankings.get(pid, {})

        prior_rank = prior.get("position_rank")
        actual_rank = current.get("position_rank")

        # Condition 1: was not an established starter last year
        was_non_starter = (prior_rank is None) or (prior_rank > prior_non_starter)

        # Condition 2: became a top-N finisher this year
        broke_out = (actual_rank is not None) and (actual_rank <= top_n)

        labeled.append({
            **player,
            "prior_position_rank": prior_rank,
            "actual_position_rank": actual_rank,
            "is_actual_breakout": was_non_starter and broke_out,
        })
    return labeled


# ==============================================================================
# METRIC CALCULATION
# ==============================================================================

def _compute_metrics(
        records: List[BacktestRecord],
        threshold: float = 50.0,
) -> Dict:
    """
    Compute classification metrics for a list of BacktestRecords.

    Returns a dict with precision, recall, f1, auc_roc, avg_precision,
    component_correlations, false_positives, false_negatives.
    """
    if not records:
        return {
            "n_players": 0,
            "n_breakouts": 0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "auc_roc": 0.0,
            "avg_precision": 0.0,
            "component_correlations": {},
            "false_positives": [],
            "false_negatives": [],
        }

    scores = [r.breakout_opportunity_score for r in records]
    labels = [int(r.is_actual_breakout) for r in records]
    predicted = [int(s >= threshold) for s in scores]

    n_players = len(records)
    n_breakouts = sum(labels)

    # Precision / Recall / F1
    if HAS_SKLEARN and n_breakouts > 0:
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predicted, average="binary", zero_division=0
        )
        try:
            auc = roc_auc_score(labels, scores) if len(set(labels)) > 1 else 0.0
        except Exception:
            auc = 0.0
        try:
            ap = average_precision_score(labels, scores) if len(set(labels)) > 1 else 0.0
        except Exception:
            ap = 0.0
    else:
        # Manual fallback
        tp = sum(1 for p, l in zip(predicted, labels) if p == 1 and l == 1)
        fp = sum(1 for p, l in zip(predicted, labels) if p == 1 and l == 0)
        fn = sum(1 for p, l in zip(predicted, labels) if p == 0 and l == 1)

        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = (2 * precision * recall / max(precision + recall, 1e-9))
        auc = 0.0
        ap = 0.0

    # Component correlations with breakout label (point-biserial)
    component_fields = [
        "opportunity_opened_score",
        "competition_removed_score",
        "competition_added_penalty",
        "team_environment_score",
        "player_readiness_score",
        "role_trajectory_score",
        "confidence_score",
    ]
    component_correlations: Dict[str, float] = {}

    for field in component_fields:
        field_scores = [getattr(r, field) for r in records]
        if HAS_SCIPY and len(set(labels)) > 1 and len(set(field_scores)) > 1:
            try:
                corr, _ = scipy_stats.pointbiserialr(labels, field_scores)
                component_correlations[field] = round(corr, 4)
            except Exception:
                component_correlations[field] = 0.0
        elif HAS_NUMPY and len(set(labels)) > 1:
            # Simple correlation fallback
            try:
                corr = float(np.corrcoef(labels, field_scores)[0, 1])
                component_correlations[field] = round(corr, 4)
            except Exception:
                component_correlations[field] = 0.0
        else:
            component_correlations[field] = 0.0

    # False positives (predicted breakout, didn't happen)
    fp_records = sorted(
        [r for r, p, l in zip(records, predicted, labels) if p == 1 and l == 0],
        key=lambda r: r.breakout_opportunity_score, reverse=True
    )[:TOP_N_FALSE_EXAMPLES]

    # False negatives (didn't predict, but broke out)
    fn_records = sorted(
        [r for r, p, l in zip(records, predicted, labels) if p == 0 and l == 1],
        key=lambda r: r.breakout_opportunity_score
    )[:TOP_N_FALSE_EXAMPLES]

    def _summarize(r: BacktestRecord) -> Dict:
        return {
            "player_id": r.player_id,
            "player_name": r.player_name,
            "team": r.team,
            "position": r.position,
            "season": r.season,
            "score": r.breakout_opportunity_score,
            "prior_rank": r.prior_position_rank,
            "actual_rank": r.actual_position_rank,
        }

    return {
        "n_players": n_players,
        "n_breakouts": n_breakouts,
        "n_predicted": sum(predicted),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "auc_roc": round(auc, 4),
        "avg_precision": round(ap, 4),
        "component_correlations": component_correlations,
        "false_positives": [_summarize(r) for r in fp_records],
        "false_negatives": [_summarize(r) for r in fn_records],
    }


# ==============================================================================
# MAIN BACKTESTING LOOP
# ==============================================================================

def run_backtest(
        seasons: List[int],
        min_score: float = 40.0,
        threshold: float = 55.0,
        phase_override: Optional[str] = None,
        positions: Optional[List[str]] = None,
        output_path: Optional[str] = None,
) -> Dict:
    """
    Run backtesting across multiple seasons.

    For each season S:
      - Build a player list from usage_rows_{S-1}.json (prior season)
      - Run the BreakoutEngine as of April 1 of season S (post-FA phase)
      - Load actual fantasy rankings for season S
      - Label breakouts and compute metrics

    Args:
        seasons: List of target seasons to evaluate (e.g. [2022, 2023, 2024])
        min_score: Minimum score to include in results (engine filter)
        threshold: Classification threshold for precision/recall computation
        phase_override: Force a specific phase for all evals (default: post_free_agency)
        positions: Filter to specific positions (default: all skill positions)
        output_path: If provided, write JSON results here

    Returns:
        Dict with per-season metrics and aggregate summary
    """
    positions = positions or ["QB", "RB", "WR", "TE"]
    all_results: Dict[str, Dict] = {}
    all_records: List[BacktestRecord] = []

    for season in seasons:
        print(f"\n{'='*60}")
        print(f"[backtest] Evaluating season {season}")
        print(f"{'='*60}")

        # Use April 1st as the evaluation date (post-FA, pre-draft)
        eval_date = date(season, 4, 1)
        effective_phase = phase_override or "post_free_agency"

        # Load actual fantasy rankings for THIS season (ground truth)
        current_rankings = _load_historical_fantasy_rankings(season)
        if not current_rankings:
            print(f"[backtest] No ranking data for {season}, skipping")
            continue

        # Load prior season rankings (to determine who was "not established")
        prior_rankings = _load_historical_fantasy_rankings(season - 1)

        # Build engine for this season
        try:
            engine = BreakoutEngine(season=season, as_of_date=eval_date)
        except Exception as e:
            print(f"[backtest] Could not initialize engine for {season}: {e}")
            continue

        # Build player list from prior year usage cache (who to evaluate)
        player_list = _build_candidate_player_list(
            engine.usage_cache, season, positions
        )
        if not player_list:
            print(f"[backtest] No players found for {season}, skipping")
            continue

        print(f"[backtest] Scoring {len(player_list)} players for {season}...")

        # Run engine
        candidates = engine.calculate_breakout_scores(
            player_list=player_list, min_score=0.0  # Include all for recall
        )

        # Convert to evaluation records, label breakouts
        score_by_id = {c.player_id: c for c in candidates}

        labeled_players = _label_breakouts(
            player_list, prior_rankings, current_rankings
        )

        season_records: List[BacktestRecord] = []
        for player in labeled_players:
            pid = str(player["player_id"])
            pos = player.get("position", "")
            if pos not in positions:
                continue

            candidate = score_by_id.get(pid)
            if candidate is None:
                # Engine filtered this player (score below 0 threshold)
                # Build a minimal record with zero scores for recall tracking
                record = BacktestRecord(
                    player_id=pid,
                    player_name=player.get("player_name", "Unknown"),
                    team=player.get("team", ""),
                    position=pos,
                    season=season,
                    as_of_date=eval_date,
                    phase=effective_phase,
                    breakout_opportunity_score=0.0,
                    opportunity_opened_score=0.0,
                    competition_removed_score=0.0,
                    competition_added_penalty=0.0,
                    team_environment_score=0.0,
                    player_readiness_score=0.0,
                    role_trajectory_score=0.0,
                    confidence_score=0.0,
                    prior_position_rank=player.get("prior_position_rank"),
                    actual_position_rank=player.get("actual_position_rank"),
                    is_actual_breakout=player.get("is_actual_breakout", False),
                )
            else:
                record = BacktestRecord(
                    player_id=pid,
                    player_name=candidate.player_name,
                    team=candidate.team,
                    position=candidate.position,
                    season=season,
                    as_of_date=eval_date,
                    phase=candidate.phase,
                    breakout_opportunity_score=candidate.breakout_opportunity_score,
                    opportunity_opened_score=candidate.opportunity_opened_score,
                    competition_removed_score=candidate.competition_removed_score,
                    competition_added_penalty=candidate.competition_added_penalty,
                    team_environment_score=candidate.team_environment_score,
                    player_readiness_score=candidate.player_readiness_score,
                    role_trajectory_score=candidate.role_trajectory_score,
                    confidence_score=candidate.confidence_score,
                    prior_position_rank=player.get("prior_position_rank"),
                    actual_position_rank=player.get("actual_position_rank"),
                    is_actual_breakout=player.get("is_actual_breakout", False),
                )

            season_records.append(record)
            all_records.append(record)

        # Compute per-season metrics
        season_metrics = _compute_metrics(season_records, threshold=threshold)
        all_results[str(season)] = {
            **season_metrics,
            "season": season,
            "eval_date": str(eval_date),
            "phase": effective_phase,
            "threshold": threshold,
        }

        n = season_metrics["n_players"]
        b = season_metrics["n_breakouts"]
        p = season_metrics["precision"]
        r = season_metrics["recall"]
        f = season_metrics["f1"]
        a = season_metrics["auc_roc"]
        print(f"  Players: {n}  |  Actual breakouts: {b}")
        print(f"  Precision: {p:.3f}  Recall: {r:.3f}  F1: {f:.3f}  AUC: {a:.3f}")

        # Per-position breakdown
        pos_breakdown = {}
        for pos in positions:
            pos_records = [rec for rec in season_records if rec.position == pos]
            if pos_records:
                pos_metrics = _compute_metrics(pos_records, threshold=threshold)
                pos_breakdown[pos] = {
                    "n_players": pos_metrics["n_players"],
                    "n_breakouts": pos_metrics["n_breakouts"],
                    "precision": pos_metrics["precision"],
                    "recall": pos_metrics["recall"],
                    "f1": pos_metrics["f1"],
                    "auc_roc": pos_metrics["auc_roc"],
                }
                print(f"    {pos}: P={pos_metrics['precision']:.3f} "
                      f"R={pos_metrics['recall']:.3f} "
                      f"F1={pos_metrics['f1']:.3f}")

        all_results[str(season)]["by_position"] = pos_breakdown

        # Component importance for this season
        print(f"\n  Component correlations with breakout label:")
        for comp, corr in sorted(
            season_metrics["component_correlations"].items(),
            key=lambda x: abs(x[1]), reverse=True
        ):
            bar = "+" * int(abs(corr) * 20)
            direction = "+" if corr > 0 else "-"
            print(f"    {direction} {comp:<38} {corr:+.4f}  {bar}")

    # Aggregate across all seasons
    if all_records:
        aggregate = _compute_metrics(all_records, threshold=threshold)
        all_results["aggregate"] = {
            **aggregate,
            "seasons": seasons,
            "threshold": threshold,
        }
        print(f"\n{'='*60}")
        print(f"[backtest] AGGREGATE ({min(seasons)}-{max(seasons)})")
        print(f"  Players: {aggregate['n_players']}  Breakouts: {aggregate['n_breakouts']}")
        print(f"  Precision: {aggregate['precision']:.3f}  "
              f"Recall: {aggregate['recall']:.3f}  "
              f"F1: {aggregate['f1']:.3f}  "
              f"AUC: {aggregate['auc_roc']:.3f}")
    else:
        all_results["aggregate"] = {}

    # Write output
    if output_path:
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f"\n[backtest] Results saved to {output_path}")

    return all_results


def _build_candidate_player_list(
        usage_cache: Dict[str, Dict],
        season: int,
        positions: List[str]
) -> List[Dict]:
    """
    Build the list of players to evaluate from the usage cache.

    The usage_cache is expected to contain the flat format produced by
    load_all_player_usage() (i.e. direct fields like 'targets', 'carries',
    'age', 'years_exp' — NOT nested under a 'usage' sub-dict).

    Returns a list of player dicts with the fields required by
    BreakoutEngine.calculate_player_breakout_score().
    """
    players = []

    for player_id, player_data in usage_cache.items():
        position = player_data.get("position", "")
        if position not in positions:
            continue

        # Flat format: age and years_exp are top-level after load_all_player_usage fix
        age = player_data.get("age")
        years_exp = player_data.get("years_exp", 0) or 0
        games = player_data.get("games", 0) or 0

        # Skip players with no meaningful usage data and not a rookie
        if games == 0 and years_exp > 0:
            continue

        # Skip players whose age is entirely unknown (can't score readiness)
        if age is None:
            continue

        players.append({
            "player_id": player_id,
            "player_name": player_data.get("name") or player_data.get("player_name", f"Player_{player_id}"),
            "team": player_data.get("team", ""),
            "position": position,
            "age": age,
            "years_exp": years_exp,
            "is_rookie": years_exp == 0,
        })

    return players


# ==============================================================================
# CLI ENTRY POINT
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Backtest the breakout detection model against historical seasons."
    )
    parser.add_argument(
        "--seasons", nargs="+", type=int,
        default=[2022, 2023, 2024],
        help="Seasons to evaluate (ground truth year, e.g. 2023 predicts 2023 breakouts)"
    )
    parser.add_argument(
        "--threshold", type=float, default=55.0,
        help="Classification threshold for precision/recall (default: 55)"
    )
    parser.add_argument(
        "--min-score", type=float, default=0.0,
        help="Minimum breakout score to include (default: 0 for full recall tracking)"
    )
    parser.add_argument(
        "--positions", nargs="+", default=["QB", "RB", "WR", "TE"],
        help="Positions to evaluate"
    )
    parser.add_argument(
        "--phase", default=None,
        help="Force a specific phase for all evaluations (default: post_free_agency)"
    )
    parser.add_argument(
        "--output", default="backtest_results.json",
        help="Output path for JSON results"
    )
    args = parser.parse_args()

    results = run_backtest(
        seasons=args.seasons,
        min_score=args.min_score,
        threshold=args.threshold,
        phase_override=args.phase,
        positions=args.positions,
        output_path=args.output,
    )

    return results


if __name__ == "__main__":
    main()
