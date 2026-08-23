"""Shared watchlist value-alert threshold.

Single source of truth for "did this player move enough to alert?", used by
both the in-app pull endpoint (/api/watchlist-alerts) and the push notifier so
they never disagree.

Dynasty values live on a ~0-999.9 scale, so a flat cutoff is scale-blind: 150
is a modest 15% move for a 999 stud but a massive 75% move for a 200-value
depth piece. Use a percentage of the player's current value with an absolute
floor — roughly a 10% weekly swing, but never less than a 50-point move (below
which a small percentage on a low-value player is just noise).
"""
from __future__ import annotations

_VALUE_ALERT_PCT = 0.10
_VALUE_ALERT_FLOOR = 50.0


def value_alert_threshold(value) -> float:
    """The 7-day value move (absolute points) required to alert on a player of
    this current value."""
    try:
        v = float(value or 0)
    except (TypeError, ValueError):
        v = 0.0
    return max(_VALUE_ALERT_FLOOR, _VALUE_ALERT_PCT * v)


def is_value_alert(delta, value) -> bool:
    """True when a 7-day ``delta`` clears the value-aware threshold for a player
    currently worth ``value``."""
    if delta is None:
        return False
    try:
        return abs(float(delta)) >= value_alert_threshold(value)
    except (TypeError, ValueError):
        return False
