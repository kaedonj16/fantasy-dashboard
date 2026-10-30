"""Player Rankings phone layout (approved rework, 2026-10-30).

History: PR #2716 made the page's inline sheet the single source of phone
grid rules, moved the sparkline + pos rank + team into a player subline,
and flipped the Total Points sort on phones (total is the big number,
Value the labeled sublabel). The 2026-10-30 rework keeps those semantics
and changes the geometry and control stack around them: rank | player |
PPG | Value on a tighter grid, the rank delta inline to the RIGHT of the
rank number (replacing the stacked-under delta), a flush headshot in the
player cell, and a phone control stack of title row, search, full-width
sliding sort pill, one nowrap row of six equal-flex position pills and
the count line (no settings/tags row).
"""
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
PAGE = (ROOT / "dashboard_services" / "pages" / "players_page.py").read_text(encoding="utf-8")
CSS = (ROOT / "static" / "dashboard.css").read_text(encoding="utf-8")
JS = (ROOT / "static" / "rankings.js").read_text(encoding="utf-8")
APP = (ROOT / "app.py").read_text(encoding="utf-8")
SW = (ROOT / "static" / "sw.js").read_text(encoding="utf-8")


def test_phone_grid_is_single_sourced_in_the_page_sheet():
    # The one phone grid: rank | player | PPG | Value with tight gaps.
    assert "@media (max-width: 640px)" in PAGE
    assert "grid-template-columns: 40px minmax(0, 1fr) 34px 60px !important;" in PAGE
    assert "column-gap: 3px !important;" in PAGE
    # dashboard.css must not carry a competing rankings grid again.
    assert ".pr-grid-row" not in CSS
    assert "38px 1fr 54px 44px 54px" not in CSS
    assert "34px 1fr 52px 52px" not in CSS


def test_phone_hides_the_total_points_header_column():
    # The Total Points column header hides on phones; its sort flips the
    # trailing Value column instead (see the flip test below). The subline
    # (sparkline + pos rank + team + age) renders at every width now.
    assert "#prTableHeader:not(.pr-adp-mode) span:nth-child(4) { display: none !important; }" in PAGE
    assert ".pr-sub {" in PAGE


def test_rank_delta_is_inline_right_of_the_rank_number():
    # Her explicit steer: the delta no longer stacks under the number.
    assert ".pr-rank {\n        font-size: 14px;" in PAGE
    assert "flex-direction: column;\n          align-items: flex-start;\n          gap: 0;" not in PAGE
    # JS renders rank + delta in the one rank cell, delta after the number.
    assert "'<span class=\"pr-rank\">'  + (displayRank ? '#' + displayRank : '–') + rankDeltaHTML + '</span>'" in JS


def test_phone_control_stack_has_no_settings_or_tags_row():
    # Full-width sliding sort pill with equal segments; the Total Points
    # button reads "Total" on phones via the short label span.
    assert ".pr-sort-toggle { flex: 1 1 100%; }" in PAGE
    assert ".pr-sort-toggle button { flex: 1; padding: 7px 0; font-size: 12px; }" in PAGE
    assert ".pr-sort-full { display: none; }" in PAGE
    assert '<span class="pr-sort-short">Total</span>' in PAGE
    # Settings / Legend / sort label / divider are gone from the phone stack.
    assert ".pr-grouplab, .pr-vdiv, .pr-pinright { display: none; }" in PAGE
    assert "#prFormatSeg { display: none; }" in PAGE
    # One nowrap row of six equal-flex position pills.
    assert ".filter-positions { flex: 1 1 100%; flex-wrap: nowrap; overflow: visible; }" in PAGE
    assert ".filter-positions .pos-pill { flex: 1; min-height: 40px; }" in PAGE
    # The old active-settings tags row is gone from the shell entirely.
    assert 'id="prActiveSettings"' not in PAGE


def test_total_points_sort_flips_on_phone_only():
    # Phone: total is the big number, Value the labeled sublabel.
    assert "VAL ${sortDisplay}" in JS
    assert "sortBy === 'total_pts') sortHeaderEl.textContent = 'Total';" in JS
    # Desktop: Total Points is its own column; Value stays in its column.
    assert '<span class="pr-total">' in JS
    assert "totalCellHTML" in JS


def test_subline_markup_in_js_rows_and_ssr_rows():
    assert "pr-spark-sub-canvas" in JS
    assert "pr-name-main" in JS
    assert "_prDrawSparkline(cnvSub, sparkData, _prSparkAnimate, 26, 14)" in JS
    assert '<span class="pr-sub">{_sub_html}</span>' in APP
    assert '_sub = pos_rank' in APP


def test_export_csv_lives_on_the_count_row():
    assert 'id="prCountRow"' in PAGE
    assert PAGE.count("prExportCSV") == 1
    # app.py's SSR reveal targets the exact shell strings (no silent no-op).
    assert "'<div id=\"prCountRow\" style=\"display:none;'" in APP
    assert "countRow.style.display = 'flex';" in JS


def test_service_worker_bumped_for_the_new_css():
    from pin_helpers import assert_sw_cache_current

    assert_sw_cache_current(SW)
