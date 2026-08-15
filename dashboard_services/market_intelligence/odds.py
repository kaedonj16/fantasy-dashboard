from __future__ import annotations

from typing import Optional


def american_implied_probability(price) -> Optional[float]:
    try:
        odds = float(price)
    except (TypeError, ValueError):
        return None
    if odds == 0:
        return None
    return abs(odds) / (abs(odds) + 100.0) if odds < 0 else 100.0 / (odds + 100.0)


def no_vig_over_probability(over_price, under_price) -> Optional[float]:
    over = american_implied_probability(over_price)
    under = american_implied_probability(under_price)
    if over is None or under is None or over + under <= 0:
        return None
    return over / (over + under)
