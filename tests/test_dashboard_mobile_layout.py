"""Regression guards for dashboard hub mobile layout (Sleeper + Fleaflicker).

The Fleaflicker spacing bug was caused by side columns (especially the teams
roster card) stretching the hub grid / reserving height for hidden panels — not
by missing dashboard HTML. These tests lock the shared layout contract without
 brittle pixel assertions.
"""
from __future__ import annotations

import re
from pathlib import Path

_CSS = Path(__file__).resolve().parents[1] / "static" / "dashboard.css"
_PAGES = Path(__file__).resolve().parents[1] / "dashboard_services" / "pages"


def _css() -> str:
    return _CSS.read_text(encoding="utf-8")


def _page_sources() -> str:
    return (
        (_PAGES / "dashboard_page.py").read_text(encoding="utf-8")
        + (_PAGES / "offseason_dashboard_page.py").read_text(encoding="utf-8")
    )


def _mobile_block(css: str) -> str:
    """Return the first max-width:1180px block that hides .os-tab-panel."""
    parts = re.split(r"@media\s*\(\s*max-width:\s*1180px\s*\)\s*\{", css)
    for chunk in parts[1:]:
        depth = 1
        body = []
        for ch in chunk:
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    break
            if depth >= 1:
                body.append(ch)
        text = "".join(body)
        if ".os-tab-panel" in text and "display: none" in text:
            return text
    raise AssertionError("expected a mobile @media (max-width: 1180px) os-tab-panel block")


def test_os_layout_does_not_stretch_columns_to_tallest_sidebar():
    css = _css()
    m = re.search(r"\.os-layout\s*\{([^}]+)\}", css)
    assert m, ".os-layout rule missing"
    body = m.group(1)
    assert "align-items: start" in body
    assert "align-items: stretch" not in body
    assert "minmax(0, 1fr)" in body
    assert "min-width: 0" in body


def test_mobile_inactive_tab_panels_removed_from_layout():
    mobile = _mobile_block(_css())
    assert re.search(r"\.os-tab-panel\s*\{[^}]*display:\s*none\s*!important", mobile)
    assert re.search(
        r"\.os-tab-panel\.os-tab-active\s*\{[^}]*display:\s*block\s*!important",
        mobile,
    )


def test_mobile_hub_uses_flex_column_not_grid_tracks():
    mobile = _mobile_block(_css())
    m = re.search(r"\.os-layout\s*\{([^}]+)\}", mobile)
    assert m, "mobile .os-layout rule missing"
    body = m.group(1)
    assert "display: flex" in body
    assert "flex-direction: column" in body


def test_inactive_team_panels_use_display_none_globally():
    css = _css()
    m = re.search(r"\.team-panel\s*\{([^}]+)\}", css)
    assert m, ".team-panel rule missing"
    body = m.group(1)
    assert "display: none" in body or "display:none" in body.replace(" ", "")
    active = re.search(r"\.team-panel\.active\s*\{([^}]+)\}", css)
    assert active, ".team-panel.active rule missing"
    assert "display: flex" in active.group(1)


def test_dashboard_hubs_share_os_layout_tab_structure():
    """Fleaflicker and Sleeper render the same hub skeleton from shared builders."""
    src = _page_sources()
    assert src.count('<div class="os-layout">') == 2
    for pattern in (
        r'<aside class="os-left-col os-tab-panel"',
        r'<main class="os-main-col">',
        r'<aside class="os-right-col os-tab-panel"',
        r'os-tab-panel os-tab-active',
        r'class="os-jump-nav"',
    ):
        assert len(re.findall(pattern, src)) >= 2, f"missing shared pattern: {pattern}"


def test_offseason_jump_nav_targets_match_tab_panel_ids():
    src = (_PAGES / "offseason_dashboard_page.py").read_text(encoding="utf-8")
    jumps = re.findall(r'data-jump="(os-jump-[^"]+)"', src)
    assert jumps == [
        "os-jump-actions",
        "os-jump-report",
        "os-jump-roster",
        "os-jump-waivers",
        "os-jump-teams",
    ]
    for panel_id in jumps:
        assert f'id="{panel_id}"' in src, f"missing tab panel id {panel_id}"


def test_inseason_jump_nav_targets_match_tab_panel_ids():
    src = (_PAGES / "dashboard_page.py").read_text(encoding="utf-8")
    jumps = re.findall(r'data-jump="(os-jump-[^"]+)"', src)
    assert jumps == [
        "os-jump-actions",
        "os-jump-report",
        "os-jump-standings",
        "os-jump-waivers",
        "os-jump-teams",
    ]
    for panel_id in jumps:
        assert f'id="{panel_id}"' in src or f"id='{panel_id}'" in src, (
            f"missing tab panel id {panel_id}"
        )


def test_no_platform_specific_spacer_before_os_layout():
    """Hub builders must start body with .os-layout — no provider wrapper."""
    for name in ("dashboard_page.py", "offseason_dashboard_page.py"):
        src = (_PAGES / name).read_text(encoding="utf-8")
        m = re.search(r'body = f"""\s*\n\s*(<[^>]+>)', src)
        assert m, f"{name}: could not find body template start"
        assert 'class="os-layout"' in m.group(1), (
            f"{name}: body must begin with .os-layout, got {m.group(1)!r}"
        )
