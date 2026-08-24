"""Pure, dependency-free value-model guardrails.

Kept import-light (stdlib only) so the logic can be unit-tested without pulling
in pandas/numpy/DB. Used by value_model_training.rewrite_value_table_with_model.
"""
from __future__ import annotations

# Defaults — see rewrite_value_table_with_model for rationale.
OVERMARKET_TRIGGER = 2.5     # model must be >= this x the external consensus to act
OVERMARKET_MIN_GAP = 100.0   # ...and at least this many points above it (skip low-stakes)
SF_NONQB_FLOOR_RATIO = 0.85  # non-QB SF safety floor, as a fraction of 1QB value


def sf_nonqb_floor(sf_value: float, value_1qb: float,
                   *, ratio: float = SF_NONQB_FLOOR_RATIO) -> float:
    """Floor a non-QB's Superflex value at ``ratio`` × its 1QB value.

    In Superflex, QBs absorb value, so elite non-QBs trade slightly BELOW their
    1QB value (~0.90-0.92× per the market). This is only a safety net against a
    bad/missing DP-2QB read cratering a player: the ratio sits below the real
    market ratio so it never inflates a non-QB above its market SF value. Flooring
    at the FULL 1QB value (ratio=1.0) inverts the market and, once the higher SF
    calibration scale is applied, pushes top RBs above the QBs on the SF board.
    """
    if sf_value is None:
        return sf_value
    return max(float(sf_value), ratio * float(value_1qb or 0.0))


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
