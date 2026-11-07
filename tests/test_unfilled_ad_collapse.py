"""Unfilled AdSense slots must collapse, including the bare top/bottom slots.

The shell serves the top and bottom slots as bare ``<ins class="adsbygoogle">``
elements with no ``.ad-container`` wrapper (app.py ``_AD_TOP`` / ``_AD_BOTTOM``).
Google still stamps a fixed height on an unfilled slot
(``data-ad-status="unfilled"``), which left a dead band for every
free/logged-out viewer. The collapse rule in dashboard.css therefore has to
target the bare slot itself, not only a wrapping container, and it needs
``!important`` to beat the slot's inline ``display:block``.
"""
from __future__ import annotations

import re
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]


def _read(rel: str) -> str:
    return (_ROOT / rel).read_text(encoding="utf-8")


def _collapse_rules(css: str):
    """Yield (selector list, body) for every rule in the stylesheet."""
    for match in re.finditer(r"([^{}]+)\{([^{}]*)\}", css):
        yield match.group(1), match.group(2)


def test_bare_unfilled_ad_slot_collapses():
    css = _read("static/dashboard.css")
    # The wrapped-only :has() selector contains the same substring, so match
    # the bare selector as a standalone entry in a rule's selector list whose
    # body collapses it.
    for selectors, body in _collapse_rules(css):
        parts = [part.strip() for part in selectors.split(",")]
        if 'ins.adsbygoogle[data-ad-status="unfilled"]' in parts:
            assert re.search(r"display:\s*none\s*!important", body)
            return
    raise AssertionError("no collapse rule targets the bare unfilled ad slot")


def test_wrapped_container_collapse_still_present():
    css = _read("static/dashboard.css")
    assert '.ad-container:has(ins.adsbygoogle[data-ad-status="unfilled"])' in css
