"""Pure, dependency-free value-model guardrails.

Kept import-light (stdlib only) so the logic can be unit-tested without pulling
in pandas/numpy/DB. Used by value_model_training.rewrite_value_table_with_model.
"""
from __future__ import annotations

# Defaults — see rewrite_value_table_with_model for rationale.
OVERMARKET_TRIGGER = 2.5     # model must be >= this x the external consensus to act
OVERMARKET_MIN_GAP = 100.0   # ...and at least this many points above it (skip low-stakes)


def overmarket_capped(
    value: float,
    fc_val: float,
    dp_val: float,
    *,
    trigger: float = OVERMARKET_TRIGGER,
    min_gap: float = OVERMARKET_MIN_GAP,
) -> float:
    """Pull ``value`` down when it sits drastically above BOTH external markets.

    ``fc_val`` / ``dp_val`` are the external values normalized to the model's
    0–999.9 scale (0 or falsy = the source doesn't cover the player, which counts
    as a low signal). The consensus is ``max(fc_val, dp_val)`` so a player either
    source rates decently is left alone — only when both are far below the model
    is it pulled down (to ``trigger`` × the consensus). Returns the value
    unchanged when no external covers the player or the divergence isn't extreme.
    """
    if value is None or value <= 0:
        return value
    fc = fc_val or 0.0
    dp = dp_val or 0.0
    if fc <= 0 and dp <= 0:
        return value  # no external coverage at all → don't touch
    market_ref = max(fc, dp)
    if value >= trigger * max(market_ref, 1.0) and (value - market_ref) >= min_gap:
        return max(market_ref * trigger, market_ref)
    return value
