"""Mobile hybrid dashboard (approved mock: mobile-dashboard-hybrid-mock).

Phone-only additions to the in-season league dashboard home: a sticky
Jump to bar under the top chrome, a Tools grid after Next Steps with
the six real tools, mini standings, and reference cards that start
collapsed. Desktop is unchanged: both new surfaces are display:none
outside the phone breakpoint. These guards assert the contract
statically, like the other dashboard placement tests.
"""
from __future__ import annotations

import re
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_PAGE = _ROOT / "dashboard_services" / "pages" / "dashboard_page.py"
_CSS = _ROOT / "static" / "dashboard.css"


def _page() -> str:
    return _PAGE.read_text(encoding="utf-8")


def _css() -> str:
    return _CSS.read_text(encoding="utf-8")


def _mobile_reorg_block(css: str) -> str:
    start = css.index("Mobile dashboard reorg")
    end = css.index("MERGED: icons.css", start)
    return css[start:end]


def test_new_surfaces_hidden_outside_phone_breakpoint():
    css = _css()
    m = re.search(
        r"\.dash-jump-nav,\s*\.dash-tools-card,\s*\.dash-st-toggle\s*\{\s*display:\s*none\s*;",
        css,
    )
    assert m, "base rule must hide the jump bar, tools grid and standings toggle"
    block = _mobile_reorg_block(css)
    assert ".dash-jump-nav {" in block
    assert re.search(r"\.dash-tools-card\s*\{\s*display:\s*block\s*;", block)


def test_jump_bar_chips_cover_the_stack_in_order():
    src = _page()
    labels = re.findall(r'_jump_chips\.append\(\("([^"]+)", "([^"]+)"', src)
    assert labels == [
        ("Matchup", "dash-sec-matchup"),
        ("Next Steps", "os-jump-actions"),
        ("Tools", "dash-sec-tools"),
        ("Injury Watch", "dashInjuryWatch"),
        ("Risers", "dash-sec-risers"),
        ("Front Office", "os-jump-report-card"),
        ("Standings", "dash-sec-standings"),
        ("Activity", "dash-sec-activity"),
        ("Awards", "dash-sec-awards"),
    ]
    # Every chip target is an id the page actually renders.
    for _label, target in labels:
        assert f'id="{target}"' in src or f'"{target}"' in src


def test_jump_bar_is_sticky_with_scrollspy_and_pruning():
    src = _page()
    assert 'class="dash-jump-nav"' in src
    assert "Jump to" in src
    assert "pruneChips" in src
    assert "syncInjuryBadge" in src
    block = _mobile_reorg_block(_css())
    nav_bodies = re.findall(r"\.dash-jump-nav\s*\{([^}]+)\}", block)
    assert any("position: sticky" in body for body in nav_bodies)
    assert "scroll-margin-top" in block


def test_injury_count_comes_from_rendered_rows():
    src = _page()
    assert '_injury_watch_html.count("cc-inj-row")' in src
    # No other chip carries a count: Front Office has no real count.
    assert src.count("_jump_chips.append") == 9


def test_tools_grid_exact_six_real_tools():
    src = _page()
    for label in (
        "ScoreZone", "Waivers & Start/Sit", "Lineup Lab",
        "Opponent Scout", "Schedule Rankings", "Trade Calculator",
    ):
        assert f'"{label}"' in src or f'"{label}",' in src
    for endpoint in (
        "page_scorezone", "league_pages.page_waivers",
        "page_weekly", "page_schedule", "trade.page_trade",
    ):
        assert endpoint in src
    assert "?tab=lab" in src and "?tab=scout" in src
    # Tiles are label-only anchors in a two-column grid.
    assert 'class="dash-tool-tile"' in src
    block = _mobile_reorg_block(_css())
    grid = re.search(r"\.dash-tools-grid\s*\{([^}]+)\}", block)
    assert grid and "grid-template-columns: 1fr 1fr" in grid.group(1)


def test_tools_grid_respects_best_ball_gate():
    src = _page()
    # Nav parity: in Best Ball the Waivers tile drops Start/Sit and
    # Lineup Lab is omitted entirely.
    assert '"Waivers" if _is_bb_tools else "Waivers & Start/Sit"' in src
    assert "if not _is_bb_tools:" in src


def test_tools_grid_after_next_steps_in_main_column():
    src = _page()
    main = src[src.index('class="os-main-col"'):src.index('class="os-right-col')]
    assert "{_action_queue_html}" in main and "{_tools_html}" in main
    assert main.index("{_action_queue_html}") < main.index("{_tools_html}")
    assert main.index("{_hero_strip_html}") < main.index("{_jump_html}")
    assert main.index("{_jump_html}") < main.index("{_matchup_preview_html}")


def test_reference_cards_start_collapsed_on_mobile():
    src = _page()
    assert '"dash-act-body", "dash-awards-body"' in src
    assert 'bodyEl.classList.add("collapsed")' in src
    assert 'mobile.matches' in src


def test_standings_mini_top4_plus_viewer_with_expand():
    src = _page()
    assert 'id="dash-sec-standings"' in src
    assert 'data-viewer-team=' in src
    assert "dash-st-extra" in src and "dash-st-expanded" in src
    assert "Show full standings" in src
    block = _mobile_reorg_block(_css())
    assert ".cc-st-card:not(.dash-st-expanded) .dash-st-extra" in block


def test_no_autopilot_or_invented_nav_in_hybrid_surfaces():
    src = _page()
    tools = src[src.index("_tool_defs = ["):src.index("_tiles = []")]
    assert "autopilot" not in tools.lower()
    assert "My Team" not in src
