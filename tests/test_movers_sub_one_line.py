"""Player Insights Movers scope line must fit on one line.

Regression test for Kaedon's 2026-10-30 screenshot: in the trade
calculator's Player Insights rail (360px wide, 326px of content room),
the Movers subtitle

    BIGGEST 7-DAY CHANGES IN 1QB DYNASTY BR VALUE

wrapped, leaving "VALUE" alone on a second line. Measured with the
site's Archivo 800 at the old styling (11px, 0.08em tracking, uppercase)
that string is 335.6px wide, so it cannot fit the rail.

Fix, three parts:
- Both JS writers (loadTopMovers and the Movers tab-switch) plus the
  server-rendered initial label use the compact phrasing
  "Biggest {days}-day {1QB|SF} {size} {dynasty|redraft} BR value movers",
  which drops the redundant "changes in" (the panel already lists Top
  Risers / Top Fallers). Worst variant (30-day, 12/14-team) measures
  ~315px at the new styling.
- .otc-mini-sub is nowrap at 10px / 0.04em tracking (the <=640px
  override must agree), with ellipsis as a safety net on screens too
  narrow for even the compact line.
- The old freshness suffix is gone: /api/value-movers rebuilds its
  payload with only used_days/risers/fallers and never returns
  latest_date, so the suffix could never render; keeping it would
  re-break the one line if the field ever appeared.

These tests read the sources directly (no Flask app import).
"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP_JS = (ROOT / "static" / "app.js").read_text()
CSS = (ROOT / "static" / "dashboard.css").read_text()
PAGE = (ROOT / "dashboard_services" / "pages" / "trade_calculator_page.py").read_text()


def _base_rule(selector: str) -> str:
    m = re.search(r"(?m)^" + re.escape(selector) + r" \{\n(.*?)\n\}", CSS, re.S)
    assert m, f"no base CSS rule found for {selector}"
    return m.group(1)


def test_movers_sub_is_nowrap_at_compact_size():
    rule = _base_rule(".otc-mini-sub")
    assert re.search(r"white-space\s*:\s*nowrap", rule), (
        ".otc-mini-sub must be white-space: nowrap so the scope line "
        "can never wrap to a second line"
    )
    assert re.search(r"font-size\s*:\s*10px", rule), (
        ".otc-mini-sub must be 10px (11px measured too wide for the rail)"
    )
    assert re.search(r"letter-spacing\s*:\s*0\.04em", rule), (
        ".otc-mini-sub tracking must be 0.04em (0.08em measured too wide)"
    )


def test_movers_sub_mobile_override_agrees_on_size():
    # The <=640px block re-declares .otc-mini-sub font-size; it must not
    # restore the too-wide 11px on phones.
    for m in re.finditer(r"\.otc-mini-sub \{\s*font-size\s*:\s*(\d+)px", CSS):
        assert m.group(1) == "10", (
            f".otc-mini-sub override uses {m.group(1)}px, expected 10px"
        )


def test_both_movers_writers_use_compact_one_line_phrasing():
    writers = re.findall(r"Biggest \$\{[^}]*\}[^`]*BR value movers`", APP_JS)
    assert len(writers) == 2, (
        "expected both moversSub writers (load + tab switch) to use the "
        f"compact 'BR value movers' phrasing, found {len(writers)}"
    )
    assert "changes in ${" not in APP_JS, (
        "the old wrapping 'Biggest ...-day changes in ... BR value' "
        "phrasing must not come back"
    )


def test_initial_label_matches_compact_phrasing():
    assert 'id="moversSub">Biggest 7-day 1QB dynasty BR value movers<' in PAGE, (
        "the server-rendered moversSub label must match the compact "
        "phrasing the JS writers use, so the line does not change shape "
        "or wrap before the first fetch lands"
    )


def test_no_freshness_suffix_on_movers_sub():
    assert "freshnessText" not in APP_JS, (
        "/api/value-movers never returns latest_date, so a freshness "
        "suffix on moversSub is dead code that would break the one "
        "line if the field ever appeared; keep it off the sub"
    )
