"""Teams drawer must sit in front of the floating bottom chrome.

Kaedon's 2026-10-30 screenshot: with the Teams page team drawer open,
the BR Assistant launcher covered a player row's value outright, and
the Discord / data-freshness pills floated on top of the drawer's
bottom rows.

Stacking facts on unfixed main:
- .td-scrim is z-index 40 and .td-drawer is z-index 50, the lowest
  modal stack on the site.
- #floating-pill-group is also z-index 50 but initPills() appends it
  to the body after the drawer markup, so it wins the tie and paints
  (and clicks) on top of the drawer, undimmed by the scrim.
- .br-asst-launcher / .br-asst-teaser / .br-asst-bubble sit at
  z-index 9994-9995, far above the drawer.

Fix: while the drawer is open, teams.js puts .td-lock on the body, and
one scoped CSS rule drops that floating chrome to z-index 30, below
the scrim (40), so it dims with the rest of the page and the scrim
swallows its clicks. The chrome's own base stacking is NOT lowered:
with the drawer closed the assistant launcher/teaser and the pill
group keep the exact z-indexes their own stacking contracts rely on.

These tests read the sources directly (no Flask app import).
"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CSS = (ROOT / "static" / "dashboard.css").read_text(encoding="utf-8")
TEAMS_JS = (ROOT / "static" / "teams.js").read_text(encoding="utf-8")


def _zindex(selector: str, css: str = CSS) -> int:
    m = re.search(re.escape(selector) + r"\s*\{([^}]+)\}", css)
    assert m, f"missing CSS rule for {selector}"
    z = re.search(r"z-index:\s*(\d+)", m.group(1))
    assert z, f"rule for {selector} has no numeric z-index"
    return int(z.group(1))


# The assistant's own delimited section (same slice the assistant UI
# contract tests use), so base-value checks never match the scoped
# body.td-lock rule, which lives in the team drawer section.
ASSISTANT_CSS = CSS[
    CSS.index("BR ASSISTANT (Stage 1 popup chat)"):CSS.index("/* == BR ASSISTANT END == */")
]


def test_drawer_stack_is_scrim_below_drawer():
    assert _zindex(".td-scrim") < _zindex(".td-drawer")


def test_td_lock_drops_floating_chrome_below_the_scrim():
    # One scoped rule naming every floating-chrome selector under
    # body.td-lock, with a z-index below the team drawer scrim, so the
    # chrome dims with the page instead of floating over the drawer.
    m = re.search(
        r"body\.td-lock\s+#floating-pill-group,\s*"
        r"body\.td-lock\s+\.br-asst-launcher,\s*"
        r"body\.td-lock\s+\.br-asst-teaser,\s*"
        r"body\.td-lock\s+\.br-asst-bubble\s*\{([^}]+)\}",
        CSS,
    )
    assert m, "no body.td-lock rule covering the floating chrome"
    z = re.search(r"z-index:\s*(\d+)", m.group(1))
    assert z, "td-lock floating chrome rule has no z-index"
    assert int(z.group(1)) < _zindex(".td-scrim")


def test_floating_chrome_base_stacking_is_unchanged():
    # The fix is scoped to the drawer-open state. Lowering these base
    # values would break the assistant popup/teaser stacking contracts
    # (tests/test_assistant_ui_contract.py) and the pills' normal use.
    assert _zindex("#floating-pill-group") == 50
    assert _zindex(".br-asst-launcher", ASSISTANT_CSS) == 9995
    assert _zindex(".br-asst-teaser", ASSISTANT_CSS) == 9994
    assert _zindex(".br-asst-bubble", ASSISTANT_CSS) == 9994
    # ...which is what made this a bug: every one of them paints above
    # the closed drawer stack until td-lock scopes them under it.
    assert _zindex("#floating-pill-group") >= _zindex(".td-drawer")
    for selector in (".br-asst-launcher", ".br-asst-teaser", ".br-asst-bubble"):
        assert _zindex(selector, ASSISTANT_CSS) > _zindex(".td-drawer")


def test_teams_js_still_marks_body_while_drawer_open():
    # The CSS hook only fires if open/close keep toggling td-lock.
    assert "document.body.classList.add('td-lock')" in TEAMS_JS
    assert "document.body.classList.remove('td-lock')" in TEAMS_JS
