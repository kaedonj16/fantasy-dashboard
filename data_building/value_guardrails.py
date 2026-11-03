"""Pure, dependency-free value-model guardrails.

Kept import-light (stdlib only) so the logic can be unit-tested without pulling
in pandas/numpy/DB. Used by value_model_training.rewrite_value_table_with_model.
"""
from __future__ import annotations

def quantile_map_to_reference(source_values, reference_values):
    """Map raw values from one market source onto a reference source's
    value distribution, by rank (quantile normalization).

    Market sources publish values on incompatible scales: FantasyCalc's
    values decay steeply by rank while KeepTradeCut's are comparatively
    flat, so a linear min-max stretch of both to 0-999.9 leaves the same
    player at wildly different normalized values (2026-11-03: Terrance
    Ferguson normalized to 126 from FC but 351 from KTC, and the blend
    inherited the inflated read — FC TE15 / KTC TE14 became board TE8).
    Mapping each source through the reference distribution instead —
    the player at rank N in the source receives the reference's value
    at rank N — makes the sources commensurable before they are
    blended: only the source's ORDER carries across, priced in
    reference units.

    Ties in the source share the mean of the reference values across
    their rank span, so ordering inside a tie never matters. Source
    ranks beyond the reference's depth clamp to its tail values. An
    empty reference maps everything to 0.0 (the source then contributes
    nothing and the blend renormalizes over the sources that remain).
    Returns a list aligned with ``source_values``.
    """
    src = [float(v) for v in (source_values or [])]
    if not src:
        return []
    ref = sorted((float(v) for v in (reference_values or [])), reverse=True)
    if not ref:
        return [0.0] * len(src)
    order = sorted(range(len(src)), key=lambda i: src[i], reverse=True)
    out = [0.0] * len(src)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and src[order[j]] == src[order[i]]:
            j += 1
        span = [ref[min(k, len(ref) - 1)] for k in range(i, j)]
        avg = sum(span) / len(span)
        for k in range(i, j):
            out[order[k]] = avg
        i = j
    return out


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
    is it pulled down (to ``trigger`` × the consensus). When NEITHER source
    covers the player, a value at/above ``min_gap`` is a phantom and is zeroed;
    smaller uncovered values are left alone (low-stakes data gap).
    """
    if value is None or value <= 0:
        return value
    fc = fc_val or 0.0
    dp = dp_val or 0.0
    if fc <= 0 and dp <= 0:
        # No external coverage at all: there is no market to be "over", so a
        # small value is a harmless data gap — but a value at/above the
        # min-gap stake has no market behind it whatsoever and cannot stand.
        # That was the Phil Mafah case: neither source priced him (FC tail
        # ~9 normalized, DP 3 raw) while the board served 330, and this
        # exemption let the phantom through every run.
        return 0.0 if value >= min_gap else value
    market_ref = max(fc, dp)
    if value >= trigger * max(market_ref, 1.0) and (value - market_ref) >= min_gap:
        return max(market_ref * trigger, market_ref)
    return value


def blend_wls_value(market_value: float, wls_value, backing,
                    *, k: float = 6.0) -> float:
    """Confidence-weighted blend of a market value toward a WLS trade value.

    WLS may only ADJUST a price a market actually set. When the market blend
    is 0 (neither external source covers the player) the WLS value is NOT
    adopted outright: the old outright adoption made a stale/phantom WLS row
    the player's entire value for exactly the players no market could check,
    and the WLS prior is itself derived from the model board, so the two
    sides re-anchored each other with no external signal in the loop (the
    Phil Mafah case: vendor blend ~6, served 330). ``backing`` is the
    decayed trade weight; at ``backing == k`` the blend is 50/50.
    """
    market = float(market_value or 0.0)
    if wls_value is None or float(wls_value) <= 0:
        return market
    if market <= 0:
        return market
    b = float(backing or 0.0)
    conf = b / (b + k) if b > 0 else 0.0
    return conf * float(wls_value) + (1.0 - conf) * market
