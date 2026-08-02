"""Shared, dependency-free label helpers for the breakout engine.

Kept as a leaf module (no imports beyond the stdlib) so any breakout-engine
module can pull in the shared mapping without risking an import cycle.
"""
from __future__ import annotations


def departure_verb(change_type: str) -> str:
    """Convert a roster-change type to a past-tense verb for narration."""
    verbs = {
        'retirement': 'retired',
        'free_agent': 'left in FA',
        'trade': 'traded away',
        'cut': 'released',
    }
    return verbs.get(change_type, 'departed')
