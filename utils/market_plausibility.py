"""Truthful trade market-plausibility model with a calibratable boundary."""
from __future__ import annotations
import math


def market_plausibility(send_value: float, receive_value: float, fit: float = 0,
                        model: dict | None = None) -> dict:
    """Score market similarity; only call it probability for labeled models."""
    ratio = float(send_value or 0) / max(float(receive_value or 0), 1.0)
    features = (1.0, ratio - 1.0, float(fit) / 100.0)
    calibrated = bool(model and model.get("rejected_offer_labels") and model.get("weights"))
    weights = tuple(model["weights"]) if calibrated else (-0.05, 5.0, 1.0)
    raw = sum(a * b for a, b in zip(weights, features))
    score = round(100 / (1 + math.exp(-raw)))
    return {"score": min(95, max(5, score)),
            "label": "acceptance probability" if calibrated else "market plausibility",
            "calibrated": calibrated}
