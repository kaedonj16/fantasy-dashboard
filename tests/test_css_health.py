"""Guards for static/dashboard.css hygiene.

These don't verify layout (that needs a browser) — they catch two things that
have actually bitten us: dead empty rule blocks, and breakpoint sprawl. Related
components using near-duplicate breakpoints (e.g. one grid switching at 480px,
its sibling at 700px) is what caused several mobile bugs where the two drifted
out of sync. The ratchet below freezes the current number of distinct
max-width breakpoints so new one-off values can't creep in — lower MAX_BREAKPOINTS
as the file is consolidated toward the canonical scale documented at the top of
dashboard.css (480 / 600 / 768 / 900 / 1180).
"""
import os
import re

_CSS_PATH = os.path.join(os.path.dirname(__file__), "..", "static", "dashboard.css")

# Ratchet: the file uses this many distinct max-width @media breakpoints. It may
# only shrink from here — never add another near-duplicate "mobile" width.
MAX_BREAKPOINTS = 19


def _css():
    with open(_CSS_PATH, encoding="utf-8") as fh:
        return fh.read()


def test_no_empty_rule_blocks():
    css = _css()
    empty = re.findall(r"([^{}]+)\{\s*\}", css)
    # Report the offending selectors if any slipped in.
    assert not empty, "empty CSS rule blocks: " + ", ".join(e.strip()[:40] for e in empty)


def test_breakpoint_count_does_not_grow():
    css = _css()
    widths = set()
    for cond in re.findall(r"@media([^{]+)\{", css):
        widths.update(int(w) for w in re.findall(r"max-width:\s*(\d+)px", cond))
    assert len(widths) <= MAX_BREAKPOINTS, (
        f"distinct @media max-width breakpoints grew to {len(widths)} "
        f"(cap {MAX_BREAKPOINTS}). Reuse a canonical breakpoint instead of adding "
        f"a new one. Values: {sorted(widths)}"
    )
