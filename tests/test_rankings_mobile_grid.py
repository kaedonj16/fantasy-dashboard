"""Player Rankings phone layout (approved mock, 2026-10-19).

Kaedon's screenshots showed "PPG VALUE" jammed together on phones: two
conflicting mobile grids (dashboard.css vs the page's inline sheet) put the
header and rows on different tracks, the Value column was 50px wide with
gap:0, the rank cell ate 76px beside a separate sparkline column, Sort by
sat full-width under centered setting tags, and a Total Points sort stacked
Value / total / pos-rank as three unlabeled lines in one cell.

The fix makes the page's inline sheet the single source of phone grid
rules, moves the sparkline + pos rank + team into a player subline, puts
Sort by compact on the setting-tags row, and flips the Total Points sort
on phones (total is the big number, Value the labeled sublabel).
"""
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
PAGE = (ROOT / "dashboard_services" / "pages" / "players_page.py").read_text(encoding="utf-8")
CSS = (ROOT / "static" / "dashboard.css").read_text(encoding="utf-8")
JS = (ROOT / "static" / "rankings.js").read_text(encoding="utf-8")
APP = (ROOT / "app.py").read_text(encoding="utf-8")
SW = (ROOT / "static" / "sw.js").read_text(encoding="utf-8")


def test_phone_grid_is_single_sourced_in_the_page_sheet():
    # The one phone grid: rank | player | PPG | Value with real gaps.
    assert "@media (max-width: 640px)" in PAGE
    assert "grid-template-columns: 40px minmax(0, 1fr) 46px 70px !important;" in PAGE
    # The header's inline gap:0 must not narrow its tracks vs the rows'.
    assert "column-gap: 10px !important;" in PAGE
    # dashboard.css must not carry a competing rankings grid again.
    assert ".pr-grid-row" not in CSS
    assert "38px 1fr 54px 44px 54px" not in CSS
    assert "34px 1fr 52px 52px" not in CSS


def test_phone_hides_the_columns_that_move_into_the_subline():
    assert ".pr-arrows, .pr-pos-cell, .pr-age, .pr-team," in PAGE
    assert ".pr-sub { display: flex; }" in PAGE
    # Rank delta stacks under the number instead of beside it.
    assert ".pr-rank {\n          flex-direction: column;" in PAGE


def test_sort_by_sits_on_the_setting_tags_row_on_mobile():
    # The stacked full-width sort row is gone; the group is a compact flex.
    assert "grid-auto-columns: minmax(0, 1fr)" not in PAGE
    assert ".filter-sort-group {\n          display: flex;" in PAGE
    assert "max-width: 148px;" in PAGE


def test_total_points_sort_flips_on_phone_only():
    # Phone: total is the big number, Value the labeled sublabel.
    assert "VAL ${sortDisplay}" in JS
    assert "sortBy === 'total_pts') sortHeaderEl.textContent = 'Total';" in JS
    # Desktop keeps Value on top with the total + its rank underneath.
    assert "const tSub = Number(p.total_pts).toFixed(1) + (tRank ? ' ' + tRank : '');" in JS


def test_subline_markup_in_js_rows_and_ssr_rows():
    assert "pr-spark-sub-canvas" in JS
    assert "pr-name-main" in JS
    assert "_prDrawSparkline(cnvSub, sparkData, _prSparkAnimate, 26, 14)" in JS
    assert '<span class="pr-sub">{_sub}</span>' in APP
    assert '_sub = pos_rank' in APP


def test_export_csv_lives_on_the_count_row():
    assert 'id="prCountRow"' in PAGE
    assert PAGE.count("prExportCSV") == 1
    # app.py's SSR reveal targets the exact shell strings (no silent no-op).
    assert "'<div id=\"prCountRow\" style=\"display:none;'" in APP
    assert "countRow.style.display = 'flex';" in JS


def test_service_worker_bumped_for_the_new_css():
    assert "br-fantasy-v76" in SW
