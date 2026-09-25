"""Guards for static/dashboard.css hygiene.

These don't verify layout (that needs a browser) — they catch two things that
have actually bitten us: dead empty rule blocks, and breakpoint sprawl. Related
components using near-duplicate breakpoints (e.g. one grid switching at 480px,
its sibling at 700px) is what caused several mobile bugs where the two drifted
out of sync. The ratchet below freezes the current number of distinct
max-width breakpoints so new one-off values can't creep in — lower MAX_BREAKPOINTS
as the file is consolidated toward the canonical scale documented at the top of
dashboard.css (480 / 600 / 768 / 900 / 1180). A third guard locks tooltip chrome
to the shared --tooltip-* tokens (advanced-metrics .adv-def-tip).
"""
import os
import re

_CSS_PATH = os.path.join(os.path.dirname(__file__), "..", "static", "dashboard.css")

# Ratchet: the file uses this many distinct max-width @media breakpoints. It may
# only shrink from here — never add another near-duplicate "mobile" width.
MAX_BREAKPOINTS = 12


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


def test_tooltip_chrome_uses_shared_tokens():
    """Hover/tap text tooltips must share the advanced-metrics .adv-def-tip chrome.

    The pre-unification [data-tooltip] bubble was a hardcoded #333 chip that
    ignored the theme. Guard the shared tokens and that regression.
    """
    css = _css()
    for token in (
        "--tooltip-bg:",
        "--tooltip-fg:",
        "--tooltip-border:",
        "--tooltip-radius:",
        "--tooltip-pad:",
        "--tooltip-fs:",
        "--tooltip-lh:",
        "--tooltip-shadow:",
    ):
        assert token in css, f"missing tooltip token {token}"
    assert "background: #333" not in css
    wk = re.search(r"\.wk-tip\s*\{([^}]+)\}", css)
    assert wk, "missing .wk-tip rule"
    assert "var(--tooltip-bg)" in wk.group(1)
    assert "background: var(--text)" not in wk.group(1)


def test_discord_logo_inverts_white_in_dark_mode():
    """The Discord PNG is a black glyph. Dark mode must invert it to white.

    The More-sheet row (.br-sheet-icon-img) had size/opacity only, so the logo
    stayed black on the dark sheet. Cover the src selector (pill, sheet,
    contact, banner) and the sheet class itself.
    """
    css = _css()
    assert 'html[data-theme="dark"] img[src*="discord-brands-solid.png"]' in css
    assert "invert(100%)" in css
    sheet = re.search(
        r'html\[data-theme="dark"\]\s+\.br-sheet-icon-img\s*\{([^}]+)\}',
        css,
    )
    assert sheet, "missing dark-mode rule for .br-sheet-icon-img"
    assert "invert(100%)" in sheet.group(1)


def test_hub_alert_strips_share_tone_token():
    """Season-hub status strips (lineup / roster / trade / bench) share one
    --alert-tone so variants recolor instead of forking layout."""
    css = _css()
    assert "border-left: 3px solid #f59e0b" not in css
    shared = re.search(
        r"\.lineup-alert-card\s*,\s*\.trade-window-card\s*,\s*\.bench-check-card\s*\{([^}]+)\}",
        css,
    )
    assert shared, "hub alert strips must share one rule"
    body = shared.group(1)
    assert "--alert-tone:" in body
    assert "var(--warning)" in body
    assert "var(--card)" in body
    assert "linear-gradient" not in body
    rail = re.search(
        r"\.lineup-alert-card::before\s*,\s*\.trade-window-card::before\s*,\s*\.bench-check-card::before\s*\{([^}]+)\}",
        css,
    )
    assert rail, "hub alert strips must share a left rail"
    assert "var(--alert-tone)" in rail.group(1)
    assert ".roster-moves-card" in css
    assert ".trade-window-card.tw-buy" in css
    assert ".bench-check-card.bench-ok" in css
    assert ".bench-check-card.bench-miss" in css
    wl = re.search(r"\.wl-alerts\s*\{([^}]+)\}", css)
    assert wl and "--alert-tone:" in wl.group(1)


def test_rz_tokens_defined_on_root_for_out_of_page_surfaces():
    """Regression: the box-score sheet is appended to document.body (outside
    .rz-page), so the --rz-* tokens must exist on :root, not only .rz-page.
    Without them the sheet rendered transparent with unstyled text."""
    css = _css()
    root_block = re.search(r":root,\s*\.rz-page\s*\{([^}]*)\}", css)
    assert root_block, ":root, .rz-page token block missing"
    for token in ("--rz-card", "--rz-border", "--rz-text", "--rz-muted"):
        assert token in root_block.group(1), f"{token} not defined on :root"


def test_rz_boxscore_sheet_clears_mobile_dock():
    """Regression: the game box-score sheet used z-index 90/91, below the
    mobile dock (.br-tabbar at --z-chrome), so its lower rows slid under the
    dock and the body never scrolled on iOS. The sheet must sit at modal
    level and its body must be a shrinking flex child of the max-height
    column sheet."""
    css = _css()
    sheet = re.search(r"\.rz-bs-sheet\s*\{([^}]*)\}", css)
    assert sheet, ".rz-bs-sheet block missing"
    assert "var(--z-modal)" in sheet.group(1), \
        ".rz-bs-sheet must clear the dock at --z-modal"
    body = re.search(r"\.rz-bs-body\s*\{([^}]*)\}", css)
    assert body, ".rz-bs-body block missing"
    assert "min-height: 0" in body.group(1), \
        ".rz-bs-body needs min-height: 0 to shrink and scroll inside the sheet"
    assert "env(safe-area-inset-bottom)" in body.group(1), \
        ".rz-bs-body bottom padding must clear the home indicator"
