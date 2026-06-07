"""Canonical numeric coercion helpers.

These replace a family of near-identical private ``_safe_float`` / ``_safe_int``
helpers that had been copy-pasted across the codebase. Only the call sites whose
local helper was *proven* behaviorally identical (tested across a full input
battery and both call patterns) import from here; variants with intentionally
different semantics — pandas ``pd.isna`` handling, ``default=None``, NaN
stripping, or ``int(float(s))`` string parsing — were deliberately left in place.

Behavior: ``None`` and blank/whitespace-only strings return ``default``;
everything else is coerced via ``float()`` / ``int()``; any ``TypeError`` or
``ValueError`` returns ``default``. NaN is intentionally NOT stripped
(``float('nan')`` round-trips), matching the consolidated call sites.
"""
from __future__ import annotations


def safe_float(value, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        if isinstance(value, str) and not value.strip():
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def safe_int(value, default: int = 0) -> int:
    try:
        if value is None:
            return default
        if isinstance(value, str) and not value.strip():
            return default
        return int(value)
    except (TypeError, ValueError):
        return default
