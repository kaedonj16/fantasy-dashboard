#!/usr/bin/env python3
"""Extract critical above-the-fold CSS for the signed-in dashboard homepage.

Parses static/dashboard.css with a simple brace-matching parser and keeps
rules whose selectors match the critical patterns (nav, hero, action rail,
layout, theme tokens, icon box-sizing). Outputs minified critical CSS.

Usage: python3 scripts/extract_critical_css.py > /tmp/critical.css
"""
import re
import sys
from pathlib import Path

CSS_PATH = Path(__file__).parent.parent / "static" / "dashboard.css"

# Selector patterns (regex) for above-the-fold dashboard homepage content.
# Keep this list tight: every rule adds bytes to every page load.
CRITICAL_PATTERNS = [
    # Theme tokens (required for all styled content)
    r"^:root$",
    r"^:root\[data-theme",
    # Base reset
    r"^\*$",
    r"^\*,",
    r"^html$",
    r"^html,",
    r"^body$",
    r"^body\.",
    # Page shell / layout
    r"\.page-shell",
    r"\.os-layout",
    r"\.os-main-col",
    r"\.os-left-col",
    r"\.os-right-col",
    r"\.os-side-rail",
    r"\.os-col-fill",
    # Top nav
    r"\.top-nav",
    r"\.br-mnav",
    r"\.changelog-bell",
    r"\.nav-new-badge",
    r"\.site-logo",
    # Hero (Season Hub)
    r"\.os-hero-card",
    r"\.os-hero-top",
    r"\.os-hero-title",
    r"\.os-hero-copy",
    r"\.os-hero-stats",
    r"\.os-hero-tag",
    r"\.os-stat-card",
    r"\.os-stat-label",
    r"\.os-stat-value",
    r"\.os-stat-sub",
    r"\.os-stat-playoff",
    # Action rail (jump nav directly under hero)
    r"\.os-jump-nav",
    r"\.std-jump-nav",
    # Base cards + section heads (first viewport)
    r"^\.card$",
    r"^\.card,",
    r", \.card$",
    r"\.os-card",
    r"\.os-section-head",
    r"\.os-section-title",
    r"\.os-section-subtitle",
    r"\.os-tab-panel",
    r"\.os-tab-active",
    # ScoreZone CTA + since-last-visit (above fold on dashboard)
    r"\.weekly-rz-cta",
    r"\.slv-wrap",
    # Mobile bottom dock (visible on phones)
    r"\.br-tabbar",
    r"\.has-tabbar",
    # Icon box sizing (prevents 0x0 flash / CLS while full CSS loads async)
    r"^\.fa$",
    r"^\.fa,",
    r"\.fa-solid",
    r"\.fa-regular",
    r"^\.fas",
    r"^\.far",
    # Loading skeleton (hero tiles can render as skeletons)
    r"\.skeleton",
]

_COMPILED = [re.compile(p) for p in CRITICAL_PATTERNS]


def _strip_comments(css: str) -> str:
    return re.sub(r"/\*.*?\*/", "", css, flags=re.DOTALL)


def _parse_rules(css: str):
    """Yield (selector, body) for each top-level rule; @media blocks yielded
    as ('@media ...', inner_css). Simple brace matching, no @import handling."""
    css = _strip_comments(css)
    i, n = 0, len(css)
    while i < n:
        # Skip whitespace and stray semicolons/braces
        while i < n and css[i] in " \t\n\r;":
            i += 1
        if i >= n:
            break
        # Find selector end (opening brace)
        sel_start = i
        while i < n and css[i] != "{":
            i += 1
        if i >= n:
            break
        selector = css[sel_start:i].strip()
        i += 1  # skip {
        depth = 1
        body_start = i
        while i < n and depth > 0:
            if css[i] == "{":
                depth += 1
            elif css[i] == "}":
                depth -= 1
            i += 1
        body = css[body_start:i - 1]
        if selector:
            yield selector, body


def _selector_matches(selector: str) -> bool:
    # Split comma-separated selectors; match if ANY matches
    for part in selector.split(","):
        # Source CSS contains literal \n sequences in some selectors;
        # normalize them to spaces before matching.
        part = part.replace("\\n", " ").replace("\\r", " ").replace("\\t", " ").strip()
        # For [data-theme="dark"] prefixed selectors, check the remainder
        # against critical patterns (don't blanket-include all dark rules)
        _dark_m = re.match(r'^\[data-theme="dark"\]\s*(.*)$', part)
        _check = _dark_m.group(1) if _dark_m else part
        if _dark_m and not _check:
            continue
        # Strip pseudo-classes/elements for matching (keep :root)
        base = re.sub(r"::?[a-zA-Z-]+(\([^)]*\))?", "", _check).strip()
        for pat in _COMPILED:
            if pat.search(_check) or (base and pat.search(base)):
                return True
    return False


def _minify_body(body: str) -> str:
    body = re.sub(r"\s+", " ", body).strip()
    body = re.sub(r"\s*([:;,{}])\s*", r"\1", body)
    body = re.sub(r";}", "}", body)
    return body


def _clean_selector(selector: str) -> str:
    """Normalize literal \n sequences and whitespace in selectors."""
    s = selector.replace("\\n", " ").replace("\\r", " ").replace("\\t", " ")
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"\s*,\s*", ",", s)
    return s


def extract(css_path: Path | None = None) -> str:
    """Extract and return minified critical CSS."""
    css = (css_path or CSS_PATH).read_text(encoding="utf-8")
    out = []
    for selector, body in _parse_rules(css):
        if selector.startswith("@media"):
            # Keep media block only with matching inner rules
            inner = []
            for sel2, body2 in _parse_rules(body):
                if _selector_matches(sel2):
                    inner.append(f"{_clean_selector(sel2)}{{{_minify_body(body2)}}}")
            if inner:
                out.append(f"{selector}{{{''.join(inner)}}}")
        elif selector.startswith("@"):
            # Skip @font-face, @keyframes etc. for critical CSS
            # (fonts load async; animations are non-critical)
            continue
        elif _selector_matches(selector):
            out.append(f"{_clean_selector(selector)}{{{_minify_body(body)}}}")
    return "".join(out)


def main() -> None:
    critical = extract()
    sys.stdout.write(critical)
    sys.stderr.write(f"critical bytes: {len(critical)}\n")


if __name__ == "__main__":
    main()
