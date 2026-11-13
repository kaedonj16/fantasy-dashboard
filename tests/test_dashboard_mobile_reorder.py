"""Mobile dashboard stack order (approved mock: mobile-dashboard-layout).

Phone-only (<=768px): the flattened .os-layout stack runs hero, ScoreZone
banner, Matchup, Next Steps, Injury Watch, Usage Risers, Usage Fallers,
Front Office, Standings, League Activity, League Awards. The order lives
in the CSS `order` values of the mobile reorg block; the Fallers card is
rendered by the page builder directly after the Risers card and hidden
outside the phone breakpoint so the desktop three-column layout is
unchanged. These guards assert the contract statically, like the other
dashboard placement tests.
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


def _order(block: str, selector: str) -> int:
    m = re.search(
        r"\.os-layout \." + re.escape(selector) + r"\s*\{\s*order:\s*(\d+)\s*;",
        block,
    )
    assert m, f"no mobile order rule for .{selector}"
    return int(m.group(1))


def test_fallers_render_directly_after_risers_in_main_column():
    src = _page()
    assert src.count("{usage_fallers_html}") == 1
    main = src[src.index('class="os-main-col"'):src.index('class="os-right-col')]
    assert "{usage_fallers_html}" in main
    assert main.index("{usage_movers_html}") < main.index("{usage_fallers_html}")


def test_fallers_card_hidden_outside_phone_breakpoint():
    css = _css()
    m = re.search(r"\.os-layout \.usage-fallers-card\s*\{\s*display:\s*none\s*;", css)
    assert m, "base rule must hide the Fallers card (desktop unchanged)"
    block = _mobile_reorg_block(css)
    assert re.search(
        r"\.os-layout \.usage-fallers-card\s*\{\s*display:\s*block\s*;", block
    ), "mobile reorg block must reveal the Fallers card on phones"


def test_left_rail_restored_in_phone_stack():
    # The <=1280px jump-tab block hides a non-tab-panel left rail with no
    # active tab child; the in-season hub stacks sections instead of
    # tabbing, so the phone reorg block must restore the rail (offseason's
    # tab-panel rail stays untouched) or Front Office/Standings vanish.
    block = _mobile_reorg_block(_css())
    assert re.search(
        r"\.os-layout > \.os-left-col:not\(\.os-tab-panel\)\s*\{"
        r"\s*display:\s*contents\s*!important\s*;",
        block,
    ), "mobile reorg block must restore the in-season left rail on phones"


def test_mobile_stack_order():
    css = _css()
    anchor = css.index("Approved mobile flow")
    block = css[anchor:anchor + 1500]
    expected = [
        "os-hero-strip",       # hero stays on top
        "weekly-rz-cta",       # banner directly under the hero
        "os-matchup-preview",  # 1. Your Matchup
        "os-action-queue",     # 2. Next Steps
        "cc-inj-card",         # 3. Injury Watch
        "usage-movers-card",   # 4. Usage Risers
        "usage-fallers-card",  # 5. Usage Fallers
        "cc-fo-card",          # 6. Front Office
        "cc-st-card",          # 7. Standings
        "cc-act-card",         # 8. League Activity
        "cc-aw-card",          # 9. League Awards
    ]
    orders = [_order(block, sel) for sel in expected]
    assert orders == sorted(orders), f"mobile stack order wrong: {orders}"
    assert len(set(orders)) == len(orders), f"duplicate order values: {orders}"
