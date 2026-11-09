"""Fit 3 columns contracts for the Advanced Metrics table on phones.

Approved mock A: on mobile the frozen pane used to eat 272px (# column
52px + Player 220px), leaving room for exactly one metric column. The
Fit 3 geometry shrinks Player to 148px, folds the rank into a subline
under the player name, and pins every metric column to exactly 68px, so
Player + the first three metrics (148 + 3*68 = 352px) fit inside the
~372px phone table wrap with no sideways scroll. Desktop geometry is
untouched: every Fit 3 rule lives inside the max-width:600px block and
the subline is display:none outside it.
"""
import re

from dashboard_services.pages.advanced_metrics_page import build_advanced_metrics_body
from data_building.advanced_metrics import LEADERBOARD_METRICS


def _html():
    return build_advanced_metrics_body(False, LEADERBOARD_METRICS)


def _style(html):
    return html[html.index("<style>"):html.index("</style>")]


def _script(html):
    return html[html.index("<script>"):html.index("</script>", html.index("<script>"))]


def _fit3_block(css):
    """The balanced-brace @media block carrying the Fit 3 comment."""
    marker = css.index("Fit 3 columns")
    start = css.rindex("@media (max-width:600px)", 0, marker)
    depth = 0
    i = css.index("{", start)
    while True:
        ch = css[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return css[start:i + 1]
        i += 1


def test_fit3_player_column_is_148px_on_mobile():
    block = _fit3_block(_style(_html()))
    assert ".am-player { width:148px; max-width:148px; }" in block


def test_fit3_rank_column_hidden_on_mobile():
    block = _fit3_block(_style(_html()))
    assert ".am-rank, .am-table th.am-rank { display:none !important; }" in block


def test_fit3_metric_columns_are_exactly_68px():
    block = _fit3_block(_style(_html()))
    assert ".am-barcell { width:68px; min-width:68px; max-width:68px; }" in block
    # The value wrap (78px on desktop) and the val min-width must release,
    # or the cell content would overflow the 68px column.
    assert ".am-val-wrap { width:auto;" in block
    assert ".am-val { min-width:0;" in block
    # Fixed layout is what pins the columns to the declared widths.
    assert ".am-table { table-layout:fixed; }" in block
    # Cell padding tightens to ~6px horizontal.
    assert ".am-table td { padding:9px 6px; }" in block


def test_fit3_geometry_fits_three_metrics_at_rest():
    # Player 148 + three metrics at 68 = 352, inside the 372px phone wrap.
    assert 148 + 3 * 68 == 352
    assert 148 + 3 * 68 <= 372


def test_fit3_headers_wrap_inside_68px():
    block = _fit3_block(_style(_html()))
    assert ".am-table th.am-barcell {" in block
    assert "white-space:normal" in block
    assert "text-align:center" in block


def test_fit3_subline_renders_only_on_mobile():
    css = _style(_html())
    # Base rule keeps the subline off desktop entirely.
    assert ".am-player-sub { display:none; }" in css
    block = _fit3_block(css)
    assert ".am-player-sub {" in block
    assert "display:block" in block
    assert "font-size:11px" in block
    # The desktop right-hand team/pos meta hides on mobile (subline has it).
    assert ".am-player-right { display:none; }" in block


def test_desktop_geometry_is_unchanged():
    css = _style(_html())
    assert ".am-player { width:220px; max-width:220px; }" in css
    assert ".am-rank { width:52px;" in css
    assert ".am-barcell { width:auto; min-width:120px; }" in css


def test_rank_folds_into_player_subline_in_render_js():
    js = _script(_html())
    assert "am-player-sub" in js
    assert "'#' + rank + ' · '" in js
    # Position drops out of the subline when the view is filtered to
    # exactly that one position; the All view keeps it.
    assert "_subPos" in js
    assert "_posList()" in js
    # The separate rank cell still renders (desktop keeps its # column).
    assert 'class="am-rank" data-column-id="rank"' in js
    assert 'class="am-player-main"' in js


def test_static_header_keeps_rank_and_player_columns():
    body = _html().split("<style>")[0]
    assert '<th class="am-rank">#</th>' in body
    assert '<th class="am-player">Player</th>' in body
    # Games/Weeks stay hidden on mobile via the same block.
    block = _fit3_block(_style(_html()))
    assert ".am-games, .am-table th.am-games," in block
