"""Critical CSS + async dashboard bundle (homepage speed).

Verifies:
1. static/critical-home.css exists and contains above-the-fold selectors.
2. The icon merge script rebuilds dashboard.css merge sections.
3. BASE_HTML inlines critical CSS and loads the bundle async.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

pytest.importorskip("flask")

STATIC = Path(__file__).parent.parent / "static"


def test_critical_css_exists_and_has_key_selectors():
    p = STATIC / "critical-home.css"
    assert p.exists(), "static/critical-home.css missing -- run scripts/extract_critical_css.py"
    css = p.read_text(encoding="utf-8")
    assert len(css) > 5000, f"critical CSS suspiciously small: {len(css)} bytes"
    assert len(css) < 60000, f"critical CSS too large: {len(css)} bytes"
    for sel in (".top-nav", ".os-hero-card", ".os-jump-nav", ".os-stat-card",
                ".fa-solid", ":root", "body"):
        assert sel in css, f"critical CSS missing {sel}"


def test_critical_css_has_no_literal_escapes():
    css = (STATIC / "critical-home.css").read_text(encoding="utf-8")
    assert "\\n" not in css, "critical CSS contains literal backslash-n sequences"


def test_dashboard_css_contains_merged_icons():
    css = (STATIC / "dashboard.css").read_text(encoding="utf-8")
    assert "MERGED: icons.css" in css
    assert "MERGED: font-awesome.css" in css
    # PNG icon mappings from icons.css are present
    assert "arrow-trend-up-solid.png" in css


def test_base_html_inlines_critical_and_async_bundle():
    import app as app_module
    html = app_module.BASE_HTML
    # Critical CSS inlined
    assert "{critical_css}" in html
    # Bundle loads async via media="print" trick (attr injected)
    assert "{css_async_attr}" in html
    assert "{css_noscript}" in html
    # Old separate icon stylesheets are gone (no <link> tags; the comment
    # explaining the merge may still mention the filenames)
    assert 'href="/static/icons.css' not in html
    assert 'href="/static/font-awesome.css' not in html


def test_merge_script_is_idempotent():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "merge_icon_css",
        Path(__file__).parent.parent / "scripts" / "merge_icon_css.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    before = (STATIC / "dashboard.css").read_text(encoding="utf-8")
    mod.main()
    after = (STATIC / "dashboard.css").read_text(encoding="utf-8")
    assert before == after, "merge_icon_css.py is not idempotent"
