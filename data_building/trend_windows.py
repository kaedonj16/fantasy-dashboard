"""Shared recent-vs-baseline trend windows.

One canonical implementation of the "is this player's usage/production
rising?" comparison, shared by the Advanced Metrics trend columns
(``xfp_trend`` / ``opportunity_trend``) and the weekly usage-trends payload.
Three copies of this math with divergent small-sample rules already caused
a real bug (early-season boards showing a forced 0.00% trend for everyone),
so the window rule lives here exactly once.

Canonical window rule (values ordered oldest -> newest, ``recent_n`` = 3):

- Fewer than 2 usable values: no comparison exists -> ``(None, None)``.
- 2..``recent_n`` values (early season): the recent window would cover the
  whole sample, forcing every ratio to exactly 0.0, so instead the LATEST
  value is compared against the average of the values before it.
- More than ``recent_n`` values: the recent window is the last ``recent_n``
  values and the baseline is the average of the WHOLE series (the season
  average, recent window included). That is deliberate: the trend reads
  "recent stretch vs the player's season norm", not "recent vs the rest".

Missing values (``None`` entries) are SKIPPED - they shrink the sample,
they are never coerced to 0, which would fabricate a real observation.

Pure and dependency-free: plain sequences in, floats/None out.
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

DEFAULT_RECENT_N = 3


def _usable(values: Sequence[Optional[float]]) -> List[float]:
    """The present values as floats, order preserved. None (and anything
    non-numeric) is a missing sample and is dropped, never zero-filled."""
    out: List[float] = []
    for v in values:
        if v is None:
            continue
        try:
            out.append(float(v))
        except (TypeError, ValueError):
            continue
    return out


def trend_window(
    values: Sequence[Optional[float]],
    recent_n: int = DEFAULT_RECENT_N,
) -> Tuple[Optional[float], Optional[float]]:
    """(recent_avg, baseline_avg) for a trend comparison, or (None, None).

    See the module docstring for the canonical window rule.
    """
    vals = _usable(values)
    if len(vals) < 2:
        return None, None
    if len(vals) <= recent_n:
        return vals[-1], sum(vals[:-1]) / (len(vals) - 1)
    recent = vals[-recent_n:]
    return sum(recent) / len(recent), sum(vals) / len(vals)


def recent_vs_baseline_ratio(
    values: Sequence[Optional[float]],
    recent_n: int = DEFAULT_RECENT_N,
) -> Optional[float]:
    """Recent-window average / baseline average - 1. None when unusable.

    Positive = trending up (0.15 means the recent stretch runs 15% above
    the baseline). None when there is no comparison or the baseline is 0.
    """
    recent_avg, baseline_avg = trend_window(values, recent_n)
    if recent_avg is None or baseline_avg is None or baseline_avg == 0:
        return None
    return recent_avg / baseline_avg - 1.0


def recent_vs_baseline_delta(
    values: Sequence[Optional[float]],
    recent_n: int = DEFAULT_RECENT_N,
) -> Optional[float]:
    """Recent-window average minus baseline average, in the stat's own
    units. None when there is no comparison (fewer than 2 usable values).
    """
    recent_avg, baseline_avg = trend_window(values, recent_n)
    if recent_avg is None or baseline_avg is None:
        return None
    return recent_avg - baseline_avg
