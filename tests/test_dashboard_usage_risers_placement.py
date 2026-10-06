"""Usage risers placement on the Season Hub.

The "Usage risers" card (the viewer's top-3 rostered players with rising
usage) used to render inside the left rail's Report tab panel, which is not
active by default, so the card was effectively hidden until the user tapped
Report. It now renders at the end of the default-active Actions panel,
after the action-queue content.

The League Awards section made the opposite trip in the same change: out
of the Standings tab panel (which now holds the Standings card only) and
into the Report panel after the Season Review card, making Report the
retrospective tab.

The dashboard render path isn't unit-tested (it needs a full league ctx +
DB), so like test_dashboard_mobile_layout and
test_dashboard_actions_empty_state, these guards assert the placement
statically against the page builder source.
"""
from __future__ import annotations

from pathlib import Path

_PAGE = Path(__file__).resolve().parents[1] / "dashboard_services" / "pages" / "dashboard_page.py"


def _src() -> str:
    return _PAGE.read_text(encoding="utf-8")


def _block(src: str, start_marker: str, end_marker: str) -> str:
    start = src.index(start_marker)
    end = src.index(end_marker, start)
    return src[start:end]


def test_usage_risers_still_computed_for_viewer():
    src = _src()
    assert "usage_movers_html = _render_usage_movers(ctx, viewer_roster_id)" in src


def test_usage_risers_render_once_inside_actions_panel_after_action_inner():
    src = _src()
    # Exactly one render site for the card in the whole page body.
    assert src.count("{usage_movers_html}") == 1
    queue = _block(src, "_action_queue_html = f", "</div>")
    assert 'id="os-jump-actions"' in queue
    # Risers come after the action-queue content, inside the same panel, so
    # they sit below the action cards (or the all-clear / link fallbacks).
    assert queue.index("{_action_inner}") < queue.index("{usage_movers_html}")


def test_report_panel_holds_season_review_then_awards_without_usage_risers():
    src = _src()
    panel = _block(src, '<div id="os-jump-report" class="os-tab-panel">', "</div>")
    assert "{season_review_html}" in panel
    assert "{awards_html}" in panel
    assert "{usage_movers_html}" not in panel
    # Retrospective order: season review first, League Awards after it.
    assert panel.index("{season_review_html}") < panel.index("{awards_html}")
    # The panel and its jump-nav button are untouched.
    assert 'data-jump="os-jump-report"' in src


def test_awards_render_once_and_not_in_standings_panel():
    src = _src()
    # Exactly one render site for the awards section in the whole page body.
    assert src.count("{awards_html}") == 1
    standings = _block(src, 'id="os-jump-standings"', '<div id="os-jump-report"')
    assert "{awards_html}" not in standings
    # The standings panel keeps the Standings card itself.
    assert "{standings_html}" in standings
    assert 'data-target="dash-standings-body"' in standings


def test_risers_do_not_suppress_actions_fallback():
    src = _src()
    # The nonblank-fallback decision still considers only the four action
    # cards: risers alone must not suppress the "You're all set" all-clear.
    # Cards are (key, html) tuples sorted by urgency; usage_movers_html must
    # not be among them.
    assert "usage_movers_html" not in src.split("_action_cards =")[1].split("if _action_cards:")[0]
    for name in ("roster_moves_html", "trade_window_html", "do_next_waiver_html"):
        assert name in src
    assert "You're all set for Week" in src
