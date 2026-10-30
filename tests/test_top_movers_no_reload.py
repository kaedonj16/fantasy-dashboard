"""Top Movers filters must not reload the page.

Kaedon (2026-10-30): "top movers page shouldnt reload when the filters
are changed". Both filter groups on /top-movers (timeframe 7d/30d/90d
and the Dynasty/Redraft format toggle) were plain <a href> links, so
every tap was a full document load: shell, assets, scroll reset.

Fix: the links stay real crawlable hrefs (SEO and the no-JS fallback),
and build_risers_fallers_body now appends _TOP_MOVERS_SWAP_SCRIPT,
which intercepts plain clicks, fetches the same URL, and swaps the
title, subtitle, toggle groups, grid and explainer (#rfWhy) from the
returned page. That page comes from the exact same server render, so
the swapped numbers can never drift from a full load. The URL updates
via pushState, popstate re-swaps, clicking the active filter is a
no-op, and any fetch/parse failure falls back to a normal navigation.

These tests pin the rendered body contract the script depends on
plus the script and CSS hooks themselves.
"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAGE = (ROOT / "dashboard_services" / "pages" / "dynasty_pages.py").read_text()
CSS = (ROOT / "static" / "dashboard.css").read_text()
SEO_CSS = (ROOT / "static" / "seo_lite.css").read_text()

_MOVERS = {
    "risers": [{"name": "Riser One", "position": "RB", "team": "BUF",
                "new_value": 700.0, "old_value": 640.0, "delta": 60.0,
                "player_id": "42"}],
    "fallers": [],
    "latest_date": "2026-10-29", "comparison_date": "2026-10-22",
}


def _body(**kwargs):
    from dashboard_services.pages.dynasty_pages import (
        build_risers_fallers_body,
    )

    return build_risers_fallers_body(
        dict(_MOVERS), as_of_date="October 29, 2026", **kwargs)


def test_filter_links_stay_real_hrefs_for_seo_and_no_js():
    body = _body(days=30, scoring_format="redraft")
    assert 'href="/top-movers?days=7&format=redraft"' in body
    assert 'href="/top-movers?days=90&format=redraft"' in body
    assert 'href="/top-movers?days=30">Dynasty</a>' in body


def test_toggle_groups_carry_swap_group_hooks():
    body = _body(days=7)
    assert 'data-rf-group="days"' in body
    assert 'data-rf-group="format"' in body


def test_explainer_is_wrapped_for_swapping():
    # A format change swaps the why/explainer copy too, so it needs a
    # stable container the script can replace as one unit.
    assert 'id="rfWhy"' in _body(days=7)
    assert 'id="rfWhy"' in _body(days=7, scoring_format="redraft")


def test_swap_script_is_appended_once_per_render():
    body = _body(days=7)
    assert body.count("window.__brTopMoversSwap") == 2, (
        "guard check + assignment: the script ships exactly once"
    )
    assert body.rstrip().endswith("</script>")


def test_script_intercepts_clicks_and_swaps_in_place():
    seg = PAGE[PAGE.index("_TOP_MOVERS_SWAP_SCRIPT"):]
    seg = seg[: seg.index('"""\n', seg.index("popstate"))]
    # Delegated click handling on the real filter links.
    assert 'e.target.closest("a.rf-tf-opt")' in seg
    assert "e.preventDefault()" in seg
    # Clicking the already-active filter must do nothing (it used to
    # reload the identical page).
    assert 'a.classList.contains("is-active")' in seg
    # The board comes from the same URL's normal server render.
    assert "fetch(url, { credentials: \"same-origin\" })" in seg
    assert "new DOMParser().parseFromString(html, \"text/html\")" in seg
    for sel in ('".rf-title"', '".rf-subtitle"', '".rf-grid"', '"#rfWhy"'):
        assert sel in seg, f"swap must replace {sel}"
    assert "data-rf-group" in seg
    # URL stays shareable and Back steps through viewed filters.
    assert "history.pushState" in seg
    assert '"popstate"' in seg
    # Failure falls back to the old behaviour, never a dead board.
    assert seg.count("window.location.href = url") == 2


def test_script_is_rerunnable_under_soft_nav():
    seg = PAGE[PAGE.index("_TOP_MOVERS_SWAP_SCRIPT"):]
    seg = seg[: seg.index('"""\n', seg.index("popstate"))]
    # app.js soft nav re-executes inline page scripts: the window guard
    # keeps the document listeners single, and no top-level let/const
    # may redeclare on the second run.
    assert "if (window.__brTopMoversSwap) return;" in seg
    assert not re.search(r"(?m)^\s*(let|const)\s", seg)


def test_loading_state_dims_grid_in_both_stylesheets():
    # Signed-in pages load dashboard.css; guests load seo_lite.css.
    assert ".rf-page.rf-loading .rf-grid" in CSS
    assert ".rf-page.rf-loading .rf-grid" in SEO_CSS
    assert "rf-loading" in PAGE
