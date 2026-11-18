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
    # Tightened ~11% in the approved states redesign (46px -> 41px).
    assert "min-height:41px" in row_rule
    assert "padding:4px 0" in row_rule


def test_score_columns_are_fixed_width_and_tabular():
    right_rule = _css_rule(".pf-rc-right")
    assert "grid-template-columns:46px 10px 46px" in right_rule
    big_rule = _css_rule(".pf-rc-big")
    assert "tabular-nums" in big_rule


def test_tossup_pill_is_a_compact_t_the_same_size_as_w_and_l():
    js = _receipts_js()
    # The wide "Toss-up" pill is gone: the tag renders a single T (the
    # panel header still spells the band out) in the one fixed-size pill
    # every tag shares, so no row's scores shift for a wider pill.
    assert "tagStr === 'Toss-up' ? 'T' : tagStr" in js
    tag_rule = _css_rule(".pf-rc-tag")
    assert "width:30px" in tag_rule and "height:22px" in tag_rule


def test_live_rows_keep_projection_big_with_current_points_underneath():
    js = _receipts_js()
    assert "r.status === 'in'" in js
    assert "pf-rc-live" in js
    # Current points render small in the same fixed columns (y2/o2 grid
    # row) from the real delivered scores, never invented.
    assert "pf-rc-now pf-rc-y2" in js and "fmt1(r.youScore)" in js
    assert "pf-rc-now pf-rc-o2" in js and "fmt1(r.oppScore)" in js
    # Phone widths used to hide the current-points line entirely.
    assert ".pf-rc-now{display:none" not in APP_PY


def test_final_rows_show_actual_score_and_solid_pill():
    js = _receipts_js()
    # bigVals graduates the actual score into the big slot only at final;
    # every other state keeps the projected final there.
    assert "r.status === 'final' && r.youScore != null" in js
    assert "pf-rc-state" in js
    assert "(isFinal ? ' solid' : '')" in js
    tag_rule = _css_rule(".pf-rc-tag.w.solid")
    assert "background:#15803d" in tag_rule


def test_footer_net_is_the_bold_payoff():
    js = _receipts_js()
    assert "pf-rc-net" in js
    net_rule = _css_rule(".pf-rc-net")
    assert "font-weight:800" in net_rule
    net_w_rule = _css_rule(".pf-rc-net.w")
    assert "color:var(--win)" in net_w_rule


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
