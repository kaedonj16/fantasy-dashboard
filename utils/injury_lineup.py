"""Simultaneous injury-state lineup re-optimization."""
from __future__ import annotations

from utils.optimal_lineup import compute_optimal_lineup


def healthy_lineup_total(values, positions, slots, player_ids, unavailable=()):
    """Remove every unavailable player before one exact re-optimization."""
    unavailable_ids = frozenset(str(x) for x in unavailable)
    healthy = [pid for pid in player_ids if str(pid) not in unavailable_ids]
    return compute_optimal_lineup(values, positions, slots, healthy)
