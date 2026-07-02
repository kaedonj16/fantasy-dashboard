"""Pure history-season selection helper.

Extracted from app.py so the default-season logic can be unit-tested without
the pandas/DB stack. (The cached, league-chain-traversing
``get_available_history_seasons`` stays in app.py; only this pure selector moves.)
"""
from __future__ import annotations

from typing import List


def get_default_history_season(available_seasons: List[int], current_season: int) -> int:
    """
    Default to the most recent completed season, not the current season.
    If there is no prior season, fall back to the newest available season.
    """
    available = sorted({int(s) for s in available_seasons if s}, reverse=True)
    if not available:
        return int(current_season)

    past = [s for s in available if s < int(current_season)]
    if past:
        return past[0]

    return available[0]
