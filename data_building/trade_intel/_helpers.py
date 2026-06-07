"""Shared leaf helpers for the trade-intel package.

These tiny pure functions were previously copy-pasted across analytics.py,
trade_value_model.py and trade_pattern_model.py (and the diagnose_wls script).
Kept in a dependency-free module so any trade-intel module can import them
without risk of an import cycle.
"""
from __future__ import annotations


def _decay_weight(days_ago: float) -> float:
    if days_ago <= 14:
        return 1.0
    if days_ago <= 30:
        return 0.6
    if days_ago <= 60:
        return 0.25
    return 0.08


def _size_bucket(num_teams: int) -> str:
    """Map raw team count to one of four canonical size buckets."""
    if num_teams <= 9:
        return "8"
    if num_teams <= 11:
        return "10"
    if num_teams == 12:
        return "12"
    return "14"
