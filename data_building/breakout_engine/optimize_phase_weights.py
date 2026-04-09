"""
ML-based weight optimizer for the breakout detection model.

Replaces the hand-tuned PHASE_WEIGHTS dict in config.py with weights learned
from historical breakout outcomes using gradient-based optimization.

Strategy:
  1. Run the backtest to collect (component_scores, is_breakout) pairs per phase.
  2. Treat the weighted sum as a logistic regression without an intercept —
     each phase gets its own weight vector that maps 7 component scores → breakout probability.
  3. Optimize weights via scipy.optimize.minimize (L-BFGS-B) maximizing log-likelihood.
  4. Post-process: normalize weights to sum to 1 and enforce positivity (except penalty component).
  5. Write updated PHASE_WEIGHTS back to config.py (or a separate override file).

Usage:
    python -m data_building.breakout_engine.optimize_phase_weights \
        --seasons 2022 2023 2024 \
        --output-config optimized_weights.json

    # Apply directly to config.py:
    python -m data_building.breakout_engine.optimize_phase_weights \
        --seasons 2022 2023 2024 \
        --apply

Optional heavy deps: numpy, scipy, sklearn (same as backtest script).
If unavailable, falls back to a grid-search approximation.
"""

import argparse
import json
import math
import os
import re
import sys
from copy import deepcopy
from datetime import date
from typing import Dict, List, Optional, Tuple

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

try:
    from scipy import optimize as scipy_optimize
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

from .backtest_breakout_model import (
    run_backtest,
    BacktestRecord,
    _load_historical_fantasy_rankings,
    _label_breakouts,
    _build_candidate_player_list,
)
from .config import PHASE_WEIGHTS
from .core import BreakoutEngine


# ==============================================================================
# COMPONENT FIELD ORDER (must match PHASE_WEIGHTS keys)
# ==============================================================================

COMPONENT_ORDER = [
    "opportunity_opened_score",
    "competition_removed_score",
    "competition_added_penalty",
    "team_environment_score",
    "player_readiness_score",
    "role_trajectory_score",
    "confidence_score",
]

WEIGHT_KEY_ORDER = [
    "opportunity_opened",
    "competition_removed",
    "competition_added_penalty",
    "team_environment",
    "player_readiness",
    "role_trajectory",
    "confidence",
]

# Phases to optimize weights for
TARGET_PHASES = [
    "offseason",
    "post_free_agency",
    "post_draft",
    "preseason",
    "in_season",
]

# Map phase → approximate evaluation date for backtesting
PHASE_EVAL_DATES = {
    "offseason": date(2000, 2, 1),       # Feb 1
    "post_free_agency": date(2000, 4, 1),  # Apr 1 (post-FA)
    "post_draft": date(2000, 5, 15),      # May 15 (post-draft)
    "preseason": date(2000, 8, 15),       # Aug 15 (preseason)
    "in_season": date(2000, 10, 1),       # Oct 1 (mid-season)
}


# ==============================================================================
# DATA COLLECTION
# ==============================================================================

def collect_training_data(
        seasons: List[int],
        phases: List[str],
        positions: Optional[List[str]] = None,
) -> Dict[str, Tuple[List, List]]:
    """
    Collect (X, y) training pairs for each phase.

    Returns:
        Dict mapping phase → (X, y) where:
          X: list of component score vectors [7 floats each]
          y: list of binary breakout labels [0 or 1]
    """
    positions = positions or ["RB", "WR", "TE"]  # Exclude QB (different dynamics)
    phase_data: Dict[str, Tuple[List, List]] = {p: ([], []) for p in phases}

    for season in seasons:
        print(f"\n[optimizer] Collecting data for season {season}...")

        current_rankings = _load_historical_fantasy_rankings(season)
        if not current_rankings:
            print(f"[optimizer] No rankings for {season}, skipping")
            continue
        prior_rankings = _load_historical_fantasy_rankings(season - 1)

        for phase in phases:
            base_date = PHASE_EVAL_DATES[phase]
            eval_date = date(season, base_date.month, base_date.day)

            try:
                engine = BreakoutEngine(season=season, as_of_date=eval_date)
            except Exception as e:
                print(f"[optimizer] Engine init failed for {season}/{phase}: {e}")
                continue

            player_list = _build_candidate_player_list(
                engine.usage_cache, season, positions
            )
            if not player_list:
                continue

            # Score all players with min_score=0 to get full distribution
            candidates = engine.calculate_breakout_scores(
                player_list=player_list, min_score=0.0
            )
            score_by_id = {c.player_id: c for c in candidates}

            labeled = _label_breakouts(player_list, prior_rankings, current_rankings)

            X_phase, y_phase = phase_data[phase]

            for player in labeled:
                pid = str(player["player_id"])
                if player.get("position") not in positions:
                    continue

                candidate = score_by_id.get(pid)
                if candidate is None:
                    # Zero-scored player (never made it through engine)
                    x = [0.0] * 7
                else:
                    x = [
                        candidate.opportunity_opened_score,
                        candidate.competition_removed_score,
                        candidate.competition_added_penalty,
                        candidate.team_environment_score,
                        candidate.player_readiness_score,
                        candidate.role_trajectory_score,
                        candidate.confidence_score,
                    ]

                X_phase.append(x)
                y_phase.append(int(player.get("is_actual_breakout", False)))

    # Report collection summary
    for phase in phases:
        X, y = phase_data[phase]
        n_breakouts = sum(y)
        print(f"[optimizer] {phase}: {len(X)} players, {n_breakouts} breakouts")

    return phase_data


# ==============================================================================
# WEIGHT OPTIMIZATION
# ==============================================================================

def _sigmoid(z):
    """Numerically stable sigmoid."""
    if HAS_NUMPY:
        return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))
    return 1.0 / (1.0 + math.exp(max(-500, min(500, -z))))


def _log_loss_with_l2(weights, X, y, l2_lambda=0.1):
    """
    Logistic log-loss with L2 regularization.

    The breakout_opportunity_score = sum(w_i * component_i).
    We treat this raw score as the logit, scaled to reasonable range.
    """
    if not (HAS_NUMPY and HAS_SCIPY):
        return 0.0  # Fallback

    X_arr = np.array(X)    # (n, 7)
    y_arr = np.array(y)    # (n,)
    w = np.array(weights)  # (7,)

    # Scale scores into logit space: raw score is 0-100, center around 50
    logits = (X_arr @ w - 50.0) / 10.0
    probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -500, 500)))

    # Binary cross-entropy
    eps = 1e-9
    loss = -np.mean(
        y_arr * np.log(probs + eps) + (1 - y_arr) * np.log(1 - probs + eps)
    )

    # L2 regularization (encourages weights to stay near prior)
    l2_penalty = l2_lambda * np.sum(w ** 2)

    return loss + l2_penalty


def _log_loss_gradient(weights, X, y, l2_lambda=0.1):
    """Gradient of log_loss_with_l2 w.r.t. weights."""
    if not HAS_NUMPY:
        return [0.0] * len(weights)

    X_arr = np.array(X)
    y_arr = np.array(y)
    w = np.array(weights)

    logits = (X_arr @ w - 50.0) / 10.0
    probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -500, 500)))
    errors = probs - y_arr                 # (n,)
    grad = (X_arr.T @ errors) / (len(y_arr) * 10.0)  # scale factor from logit formula
    grad += 2 * l2_lambda * w             # L2 gradient

    return grad.tolist()


def optimize_weights_scipy(
        X: List[List[float]],
        y: List[int],
        prior_weights: List[float],
        l2_lambda: float = 0.05,
) -> Tuple[List[float], float]:
    """
    Optimize weights using L-BFGS-B (allows box constraints).

    The penalty component (competition_added_penalty, index=2) can go negative,
    so we allow it to have values in [-0.40, 0.0].
    All other weights are bounded [0.01, 0.40].
    """
    if not (HAS_NUMPY and HAS_SCIPY):
        return prior_weights, 0.0

    # Bounds: penalty component can be negative
    bounds = []
    for i in range(len(WEIGHT_KEY_ORDER)):
        if WEIGHT_KEY_ORDER[i] == "competition_added_penalty":
            bounds.append((-0.40, 0.0))
        else:
            bounds.append((0.01, 0.45))

    result = scipy_optimize.minimize(
        fun=_log_loss_with_l2,
        x0=prior_weights,
        args=(X, y, l2_lambda),
        jac=_log_loss_gradient,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": 500, "ftol": 1e-10, "gtol": 1e-7},
    )

    optimized = result.x.tolist()
    final_loss = float(result.fun)

    # Normalize positive weights to sum to ~1 (keep penalty separate)
    pos_indices = [i for i, k in enumerate(WEIGHT_KEY_ORDER) if k != "competition_added_penalty"]
    neg_indices = [i for i, k in enumerate(WEIGHT_KEY_ORDER) if k == "competition_added_penalty"]

    pos_sum = sum(optimized[i] for i in pos_indices)
    if pos_sum > 0:
        for i in pos_indices:
            optimized[i] /= pos_sum
        # Rescale so positives sum to 1 - |penalty_weight|
        penalty_mag = abs(optimized[neg_indices[0]])
        scale = 1.0 - penalty_mag
        for i in pos_indices:
            optimized[i] *= scale

    return optimized, final_loss


def optimize_weights_sklearn(
        X: List[List[float]],
        y: List[int],
) -> List[float]:
    """
    Fallback: use sklearn LogisticRegression to estimate feature importances.
    Returns normalized weights as a list in WEIGHT_KEY_ORDER.
    """
    if not (HAS_NUMPY and HAS_SKLEARN):
        return None

    X_arr = np.array(X)
    y_arr = np.array(y)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_arr)

    clf = LogisticRegression(
        C=5.0,           # Moderate regularization
        max_iter=500,
        class_weight="balanced",   # Handle class imbalance
        solver="lbfgs",
    )
    clf.fit(X_scaled, y_arr)

    # Raw coefficients (in standard-score space)
    raw_coefs = clf.coef_[0].tolist()

    # Normalize to sum=1 (preserve sign for penalty component)
    pos_sum = sum(max(c, 0) for c in raw_coefs)
    if pos_sum > 0:
        normalized = [c / pos_sum for c in raw_coefs]
    else:
        normalized = raw_coefs

    return normalized


def grid_search_weights(
        X: List[List[float]],
        y: List[int],
        prior_weights: List[float],
        n_iter: int = 500,
) -> List[float]:
    """
    Pure-Python fallback: random perturbation hill-climbing.
    Much slower and less accurate than scipy, but works without deps.
    """
    import random

    def score(weights, X, y):
        """Log-loss approximation."""
        total_loss = 0.0
        for xi, yi in zip(X, y):
            raw = sum(w * x for w, x in zip(weights, xi))
            logit = (raw - 50.0) / 10.0
            logit = max(-500, min(500, logit))
            prob = 1.0 / (1.0 + math.exp(-logit))
            prob = max(1e-9, min(1 - 1e-9, prob))
            total_loss -= yi * math.log(prob) + (1 - yi) * math.log(1 - prob)
        return total_loss / max(len(y), 1)

    best = list(prior_weights)
    best_loss = score(best, X, y)

    for _ in range(n_iter):
        # Random perturbation
        candidate = [
            max(-0.40, min(0.45, w + random.gauss(0, 0.02)))
            for w in best
        ]
        # Force penalty negative
        candidate[2] = min(0.0, candidate[2])

        # Normalize positive weights
        pos_sum = sum(max(c, 0) for i, c in enumerate(candidate) if i != 2)
        if pos_sum > 0:
            scale = (1.0 + candidate[2]) / pos_sum  # keep budget
            candidate = [
                (c * scale if i != 2 else c)
                for i, c in enumerate(candidate)
            ]

        loss = score(candidate, X, y)
        if loss < best_loss:
            best = candidate
            best_loss = loss

    return best


def optimize_phase_weights(
        seasons: List[int],
        phases: Optional[List[str]] = None,
        positions: Optional[List[str]] = None,
        l2_lambda: float = 0.05,
) -> Dict[str, Dict[str, float]]:
    """
    Optimize component weights for each phase using historical breakout data.

    Returns:
        Dict with same structure as PHASE_WEIGHTS, with learned values.
    """
    phases = phases or TARGET_PHASES
    positions = positions or ["RB", "WR", "TE"]

    # Collect training data for all phases
    phase_data = collect_training_data(seasons, phases, positions)

    optimized_weights: Dict[str, Dict[str, float]] = {}

    for phase in phases:
        X, y = phase_data[phase]
        n_breakouts = sum(y)

        print(f"\n[optimizer] Optimizing weights for phase: {phase}")
        print(f"  Training samples: {len(X)}  |  Breakouts: {n_breakouts}")

        if len(X) < 20 or n_breakouts < 3:
            print(f"  [!] Insufficient data for {phase}, using prior weights")
            optimized_weights[phase] = deepcopy(PHASE_WEIGHTS[phase])
            continue

        # Prior weights as initial guess
        prior = [PHASE_WEIGHTS[phase][k] for k in WEIGHT_KEY_ORDER]

        # Attempt optimization
        if HAS_NUMPY and HAS_SCIPY:
            learned, final_loss = optimize_weights_scipy(X, y, prior, l2_lambda)
            method = "L-BFGS-B"
        elif HAS_NUMPY and HAS_SKLEARN:
            sklearn_weights = optimize_weights_sklearn(X, y)
            if sklearn_weights:
                # Blend 50/50 with prior to avoid overfitting on small data
                learned = [0.5 * s + 0.5 * p for s, p in zip(sklearn_weights, prior)]
                final_loss = 0.0
            else:
                learned = prior
                final_loss = 0.0
            method = "sklearn (blended)"
        else:
            learned = grid_search_weights(X, y, prior)
            final_loss = 0.0
            method = "grid_search"

        # Build weight dict
        weight_dict = {k: round(max(0.0, v) if k != "competition_added_penalty" else v, 4)
                       for k, v in zip(WEIGHT_KEY_ORDER, learned)}

        optimized_weights[phase] = weight_dict

        print(f"  Method: {method}  |  Final loss: {final_loss:.4f}")
        print(f"  Prior → Learned:")
        for key, prior_v in zip(WEIGHT_KEY_ORDER, prior):
            learned_v = weight_dict[key]
            delta = learned_v - prior_v
            arrow = "↑" if delta > 0.01 else ("↓" if delta < -0.01 else "→")
            print(f"    {arrow} {key:<38} {prior_v:.4f} → {learned_v:.4f}  ({delta:+.4f})")

    return optimized_weights


# ==============================================================================
# OUTPUT / APPLICATION
# ==============================================================================

def format_weights_as_python(weights: Dict[str, Dict[str, float]]) -> str:
    """Format optimized weights as a Python PHASE_WEIGHTS dict literal."""
    lines = ["PHASE_WEIGHTS: Dict[str, Dict[str, float]] = {"]
    for phase, w in weights.items():
        lines.append(f"    '{phase}': {{")
        for key, val in w.items():
            lines.append(f"        '{key}': {val},")
        lines.append("    },")
    lines.append("}")
    return "\n".join(lines)


def apply_weights_to_config(
        weights: Dict[str, Dict[str, float]],
        config_path: str = None,
) -> None:
    """
    Write learned weights back into config.py, replacing the PHASE_WEIGHTS block.
    Makes a backup of the original file first.
    """
    if config_path is None:
        config_path = os.path.join(
            os.path.dirname(__file__), "config.py"
        )

    # Backup original
    backup_path = config_path + ".bak"
    with open(config_path, "r") as f:
        original = f.read()
    with open(backup_path, "w") as f:
        f.write(original)
    print(f"[optimizer] Backed up config.py → {backup_path}")

    # Build replacement block
    new_block_lines = [
        "PHASE_WEIGHTS: Dict[str, Dict[str, float]] = {",
    ]
    phase_comments = {
        "offseason": "# Offseason: no in-season data, focus on opportunity & readiness",
        "post_free_agency": "# Post-FA: signings impact competition heavily",
        "post_draft": "# Post-draft: draft picks drive competition penalty",
        "preseason": "# Preseason: usage trends start to matter",
        "in_season": "# In-season: recent trajectory dominates",
    }
    for phase, w in weights.items():
        comment = phase_comments.get(phase, f"# {phase}")
        new_block_lines.append(f"    '{phase}': {{  {comment}")
        for key, val in w.items():
            new_block_lines.append(f"        '{key}': {val},")
        new_block_lines.append("    },")
    new_block_lines.append("}")
    new_block = "\n".join(new_block_lines)

    # Replace the PHASE_WEIGHTS block in the file using regex
    pattern = r"PHASE_WEIGHTS:\s*Dict\[str,\s*Dict\[str,\s*float\]\]\s*=\s*\{.*?\n\}"
    replacement = new_block
    updated = re.sub(pattern, replacement, original, flags=re.DOTALL)

    if updated == original:
        print("[optimizer] Warning: could not find PHASE_WEIGHTS block to replace.")
        print("[optimizer] Printing new weights instead:")
        print(new_block)
        return

    with open(config_path, "w") as f:
        f.write(updated)
    print(f"[optimizer] Applied optimized weights to {config_path}")


# ==============================================================================
# CLI ENTRY POINT
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Optimize breakout model phase weights using historical data."
    )
    parser.add_argument(
        "--seasons", nargs="+", type=int,
        default=[2022, 2023, 2024],
        help="Seasons to use for training (e.g. 2022 2023 2024)"
    )
    parser.add_argument(
        "--phases", nargs="+",
        default=TARGET_PHASES,
        choices=TARGET_PHASES,
        help="Phases to optimize (default: all)"
    )
    parser.add_argument(
        "--positions", nargs="+",
        default=["RB", "WR", "TE"],
        help="Positions to include in training"
    )
    parser.add_argument(
        "--l2", type=float, default=0.05,
        help="L2 regularization strength (default: 0.05)"
    )
    parser.add_argument(
        "--output-config", default="optimized_weights.json",
        help="Write JSON output of optimized weights here"
    )
    parser.add_argument(
        "--apply", action="store_true",
        help="Apply learned weights directly to config.py (makes backup first)"
    )
    args = parser.parse_args()

    optimized = optimize_phase_weights(
        seasons=args.seasons,
        phases=args.phases,
        positions=args.positions,
        l2_lambda=args.l2,
    )

    # Write JSON output
    output_path = args.output_config
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(optimized, f, indent=2)
    print(f"\n[optimizer] Saved optimized weights → {output_path}")
    print(format_weights_as_python(optimized))

    # Optionally apply to config.py
    if args.apply:
        apply_weights_to_config(optimized)

    return optimized


if __name__ == "__main__":
    main()
