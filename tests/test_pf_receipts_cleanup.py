"""Focused tests for the Projected Finals receipts cleanup (mock v3):
full league names on a two-line row, winning projected score bold, a
signed margin, a small W/L pill, and one totals footer. Hero / summary
strip markup and the tile toggle contract are unchanged."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP_PY = (ROOT / "app.py").read_text(encoding="utf-8")
APP_JS = (ROOT / "static" / "app.js").read_text(encoding="utf-8")


def _receipts_js():
    start = APP_JS.index("function renderReceipts()")
    end = APP_JS.index("window.__pfReceiptPush", start)
    return APP_JS[start:end]


def _css_rule(selector):
    start = APP_PY.index(f'"{selector}{{')
    end = APP_PY.index('}"', start)
    return APP_PY[start:end]


def test_receipts_rows_stack_full_league_name_over_opponent():
    js = _receipts_js()
    assert "pf-rc-main" in js
    assert "pf-rc-lg" in js and "esc(r.name)" in js
    assert "pf-rc-opp" in js and "vs ' + esc(r.opp)" in js
    # The main complaint: league names truncated mid-word. The league-name
    # rule must not ellipsize or force a single line anymore.
    lg_rule = _css_rule(".pf-rc-lg")
    assert "ellipsis" not in lg_rule
    assert "nowrap" not in lg_rule


def test_receipts_rows_are_compact():
    row_rule = _css_rule(".pf-rc-row")
    assert "min-height:46px" in row_rule


def test_winning_projected_score_is_bold_and_margin_is_signed():
    js = _receipts_js()
    assert "pf-rc-win" in js
    assert "y > o ? 'pf-rc-win'" in js
    assert "o > y ? 'pf-rc-win'" in js
    assert "pf-rc-margin" in js
    # Margin is my proj minus opp proj, rendered with an explicit sign.
    assert "Math.abs(md).toFixed(1)" in js
    assert "(md < 0 ? '-' : '+')" in js


def test_totals_footer_sums_the_same_row_values_and_fails_soft():
    js = _receipts_js()
    assert "pf-rc-total" in js
    assert "' for · '" in js and "' against · net '" in js
    assert "totYou += y" in js and "totOpp += o" in js
    # Fail-soft: a non-finite row value suppresses the footer entirely
    # instead of rendering fake or partial totals.
    assert "totOk" in js and "if (totOk)" in js


def test_no_filters_sorts_or_share_bars_in_receipts():
    js = _receipts_js()
    for marker in ("data-filter", "pf-rc-sort", "pf-rc-share", "pf-rc-bar"):
        assert marker not in js
        assert marker not in APP_PY


def test_hero_and_toggle_contract_unchanged():
    # Summary strip shell: proj tile still a button, receipts still
    # hidden by default with the toss-up header preserved verbatim.
    assert "id='pfProjTile' aria-expanded='false'" in APP_PY
    assert "id='pfReceipts' hidden" in APP_PY
    assert "Projected finals · Toss-up = within 3 pts" in APP_PY
    js = APP_JS
    assert "getElementById('pfProjTile')" in js
    assert "getElementById('pfReceipts')" in js
    assert "panel.hidden = !open" in js
