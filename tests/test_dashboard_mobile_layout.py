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


def test_os_layout_side_columns_capped_to_center_height():
    css = _css()
    m = re.search(r"\.os-layout\s*\{([^}]+)\}", css)
    assert m, ".os-layout rule missing"
    body = m.group(1)
    assert "align-items: start" in body
    assert "align-items: stretch" not in body
    assert "minmax(0, 1fr)" in body
    assert "min-width: 0" in css
    assert "os-hub-cols-synced" in css
    assert "--os-hub-main-h" in css


def test_mobile_inactive_tab_panels_removed_from_layout():
    mobile = _mobile_block(_css())
    assert re.search(r"\.os-tab-panel\s*\{[^}]*display:\s*none\s*!important", mobile)
    assert re.search(
        r"\.os-tab-panel\.os-tab-active\s*\{[^}]*display:\s*block\s*!important",
        mobile,
    )


def test_mobile_hides_inseason_left_rail_when_neither_child_tab_is_active():
    """Standings + Report share the left rail; empty rail must not leave a gap."""
    mobile = _mobile_block(_css())
    assert ".os-left-col:not(.os-tab-panel):not(:has(.os-tab-active))" in mobile
    assert "display: none" in mobile


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
        r'<aside class="os-left-col',
        r'<main class="os-main-col">',
        r'<aside class="os-right-col os-tab-panel"',
        r'os-tab-panel os-tab-active',
        r'class="os-jump-nav"',
    ):
        assert len(re.findall(pattern, src)) >= 2, f"missing shared pattern: {pattern}"
    # Offseason left rail is still a single tab panel (team snapshot).
    os_dash = (_PAGES / "offseason_dashboard_page.py").read_text(encoding="utf-8")
    assert '<aside class="os-left-col os-tab-panel"' in os_dash


def test_offseason_jump_nav_targets_match_tab_panel_ids():
    src = (_PAGES / "offseason_dashboard_page.py").read_text(encoding="utf-8")
    jumps = re.findall(r'data-jump="(os-jump-[^"]+)"', src)
    assert jumps == [
        "os-jump-actions",
        "os-jump-report",
        "os-jump-roster",
        "os-jump-teams",
    ]
    for panel_id in jumps:
        assert f'id="{panel_id}"' in src, f"missing tab panel id {panel_id}"


def test_hubs_swap_empty_roster_sidebar_for_cheat_sheet():
    for name in ("dashboard_page.py", "offseason_dashboard_page.py"):
        src = (_PAGES / name).read_text(encoding="utf-8")
        assert "render_dashboard_teams_sidebar" in src, name
        assert "{teams_tab_label}" in src, name


def test_inseason_jump_nav_targets_match_tab_panel_ids():
    src = (_PAGES / "dashboard_page.py").read_text(encoding="utf-8")
    jumps = re.findall(r'data-jump="(os-jump-[^"]+)"', src)
    assert jumps == [
        "os-jump-actions",
        "os-jump-matchup",
        "os-jump-report",
        "os-jump-standings",
        "os-jump-teams",
    ]
    for panel_id in jumps:
        assert f'id="{panel_id}"' in src or f"id='{panel_id}'" in src, (
            f"missing tab panel id {panel_id}"
        )


def test_inseason_matchup_preview_has_own_tab():
    """Matchup Preview must not be buried under the Report tab on mobile."""
    src = (_PAGES / "dashboard_page.py").read_text(encoding="utf-8")
    assert 'data-jump="os-jump-matchup"' in src
    assert 'id="os-jump-matchup"' in src
    assert 'id="os-jump-report"' in src
    # Front Office lives in the left rail (Report tab); matchup stays in main.
    left = src[src.index('class="os-left-col"'): src.index('class="os-main-col"')]
    main = src[src.index('class="os-main-col"'): src.index('class="os-right-col')]
    assert "{gm_card_html}" in left
    assert "{gm_card_html}" not in main
    assert "{matchup_html}" in main
    assert "{matchup_html}" not in left
    assert 'id="os-jump-report"' in left
    assert 'id="os-jump-standings"' in left


def test_dashboard_matchup_preview_is_compact_with_full_page_link():
    src = (_PAGES / "dashboard_page.py").read_text(encoding="utf-8")
    assert "compact=True" in src
    assert "title_href" in src
    assert '"page_weekly"' in src
    hub = (_PAGES / "weekly_hub_page.py").read_text(encoding="utf-8")
    assert "compact=True" not in hub
    assert "subtle-label" not in src


def test_changelog_announces_compact_dashboard_matchup_preview():
    from dashboard_services.changelog import CHANGELOG

    entry = next(
        e for e in CHANGELOG
        if "matchup preview" in e.get("text", "").lower()
        and "win bar" in e.get("text", "").lower()
    )
    assert entry["date"] == "2026-08-31"
    assert entry["tag"] == "update"
    assert "—" not in entry["text"]
    assert "–" not in entry["text"]


def test_changelog_announces_front_office_label_removed():
    from dashboard_services.changelog import CHANGELOG

    entry = next(
        e for e in CHANGELOG
        if "front office report" in e.get("text", "").lower()
        and "team-name" in e.get("text", "").lower()
    )
    assert entry["date"] == "2026-08-31"
    assert entry["tag"] == "update"
    assert "—" not in entry["text"]
    assert "–" not in entry["text"]


def test_inseason_hides_matchup_preview_until_undrafted_non_dynasty_drafts():
    src = (_PAGES / "dashboard_page.py").read_text(encoding="utf-8")
    assert "show_matchup_preview" in src
    assert "_show_matchup_preview" in src
    assert "is_dynasty=not _league_is_redraft(ctx)" in src
    hub = (_PAGES / "weekly_hub_page.py").read_text(encoding="utf-8")
    assert "show_matchup_preview" in hub
    assert "is_dynasty=not _league_is_redraft(ctx)" in hub


def test_changelog_announces_front_office_left_rail_and_bulletins_off():
    from dashboard_services.changelog import CHANGELOG

    entry = next(
        e for e in CHANGELOG
        if "front office report" in e.get("text", "").lower()
        and "standings" in e.get("text", "").lower()
        and "bulletins" in e.get("text", "").lower()
    )
    assert entry["date"] == "2026-08-31"
    assert entry["tag"] == "update"
    assert "—" not in entry["text"]
    assert "–" not in entry["text"]


def test_changelog_announces_undrafted_matchup_preview_hide():
    from dashboard_services.changelog import CHANGELOG

    entry = next(
        e for e in CHANGELOG
        if "matchup preview" in e.get("text", "").lower() and "dynasty" in e.get("text", "").lower()
    )
    assert entry["date"] == "2026-08-31"
    assert entry["tag"] == "fix"
    assert "—" not in entry["text"]
    assert "–" not in entry["text"]


def test_no_platform_specific_spacer_before_os_layout():
    """Hub builders must start body with .os-layout — no provider wrapper."""
    for name in ("dashboard_page.py", "offseason_dashboard_page.py"):
        src = (_PAGES / name).read_text(encoding="utf-8")
        m = re.search(r'body = f"""\s*\n\s*(<[^>]+>)', src)
        assert m, f"{name}: could not find body template start"
        assert 'class="os-layout"' in m.group(1), (
            f"{name}: body must begin with .os-layout, got {m.group(1)!r}"
        )
