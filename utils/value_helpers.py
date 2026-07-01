"""Pure TE-premium value helpers.

Extracted from app.py so this logic can be unit-tested without importing the
full application (pandas / DB) stack. A league that awards bonus points per TE
reception ("TE premium") makes tight ends more valuable; these helpers snap the
league's Sleeper ``bonus_rec_te`` to the supported tiers (0 / 0.5 / 1.0) and
scale TE values by +20% per full point — matching the trade calculator, activity
feed and player modal so values stay consistent on every page that shows them.
"""
from __future__ import annotations


def te_premium_from_settings(scoring_settings) -> float:
    """Snap a league's Sleeper ``bonus_rec_te`` to a supported premium tier.

    Returns 1.0 (full), 0.5 (half), or 0.0 (none). Non-dict / non-numeric input
    yields 0.0 rather than raising.
    """
    try:
        b = float((scoring_settings or {}).get("bonus_rec_te") or 0)
    except (TypeError, ValueError, AttributeError):
        return 0.0
    return 1.0 if b >= 0.75 else 0.5 if b >= 0.25 else 0.0


def apply_te_premium(value, position, te_premium) -> float:
    """Scale a TE's value up for TE-premium leagues; pass-through otherwise.

    +20% per full premium point. ``value`` is coerced to float (DB values arrive
    as Decimal, which would otherwise raise on ``Decimal * float``), returning
    0.0 on non-numeric input rather than raising.
    """
    try:
        v = float(value or 0)
    except (TypeError, ValueError):
        return 0.0
    if te_premium and str(position or "").upper() == "TE":
        return v * (1.0 + te_premium * 0.20)
    return v
