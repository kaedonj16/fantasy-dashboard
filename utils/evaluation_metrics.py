"""Decision-specific evaluation metrics shared by model backtests."""
from __future__ import annotations

import math


def brier_score(predictions, outcomes) -> float:
    pairs = list(zip(predictions, outcomes))
    return sum((float(p) - float(y)) ** 2 for p, y in pairs) / len(pairs) if pairs else 0.0


def log_loss(predictions, outcomes) -> float:
    pairs, eps = list(zip(predictions, outcomes)), 1e-6
    if not pairs:
        return 0.0
    return -sum(float(y) * math.log(min(1-eps, max(eps, float(p))))
                + (1-float(y)) * math.log(min(1-eps, max(eps, 1-float(p))))
                for p, y in pairs) / len(pairs)


def precision_at_k(scores, outcomes, k: int) -> float:
    pairs = list(zip(scores, outcomes))
    chosen = sorted(pairs, key=lambda pair: pair[0], reverse=True)[:max(0, k)]
    return sum(bool(y) for _, y in chosen) / len(chosen) if chosen else 0.0


def decision_regret(recommended_value: float, optimal_value: float) -> float:
    """Lost realized utility versus the best legal hindsight decision."""
    return max(0.0, float(optimal_value) - float(recommended_value))
