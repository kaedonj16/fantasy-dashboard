"""Small pure formatting helpers.

Extracted from app.py, where the same ordinal logic was reimplemented in at
least three places (_ord_str, _dash_ord, _ordinal) plus several inline
`{1:"st",2:"nd",3:"rd"}.get(...)` snippets. Centralized and unit-tested so the
"11th/12th/13th" special case can't be gotten wrong in one copy and right in
another.
"""
from __future__ import annotations


def ord_suffix(n) -> str:
    """Ordinal suffix for an integer: 1->'st', 2->'nd', 3->'rd', 4->'th',
    correctly returning 'th' for the 11-13 teens (11th, 12th, 13th)."""
    try:
        n = int(n)
    except (TypeError, ValueError):
        return "th"
    if 10 <= (n % 100) <= 20:
        return "th"
    return {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")


def ordinal(n) -> str:
    """Full ordinal string: 1 -> '1st', 2 -> '2nd', 11 -> '11th', 23 -> '23rd'."""
    try:
        n = int(n)
    except (TypeError, ValueError):
        return str(n)
    return f"{n}{ord_suffix(n)}"
