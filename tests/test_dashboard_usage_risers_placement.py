"""Usage risers and League Awards placement on the Season Hub (redesign).

The "Usage risers" card (the viewer's top-3 rostered players with rising
usage) renders as its own card in the main column, below the Next steps
action queue.

The League Awards section renders in the right column as a single-column
stack of clean cards.

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


def test_usage_risers_render_as_separate_card_in_main_column():
    src = _src()
    # Exactly one render site for the card in the whole page body.
    assert src.count("{usage_movers_html}") == 1
    # Not nested inside the action queue: it is its own card in main.
    queue = _block(src, "_action_queue_html = f", "</div>")
    assert "{usage_movers_html}" not in queue
    main = _block(src, 'class="os-main-col"', 'class="os-right-col')
    assert "{usage_movers_html}" in main
    # Order: action queue first, then the risers card.
    assert main.index("{_action_queue_html}") < main.index("{usage_movers_html}")


def test_awards_in_right_column_not_left():
    """Dashboard redesign: League Awards live in the right column."""
    src = _src()
    right = _block(src, '<aside class="os-right-col', "</aside>")
    assert "{awards_html}" in right
    left = _block(src, '<aside class="os-left-col', "</aside>")
    assert "{awards_html}" not in left
    main = _block(src, 'class="os-main-col"', 'class="os-right-col')
    assert "{awards_html}" not in main


def test_awards_render_once_and_not_in_standings_card():
    src = _src()
    # Exactly one render site for the awards section in the whole page body.
    assert src.count("{awards_html}") == 1
    left = _block(src, '<aside class="os-left-col', "</aside>")
    assert "{awards_html}" not in left
    # The left column keeps the Standings card itself.
    assert "{standings_html}" in left
    assert 'data-target="dash-standings-body"' in left


def test_risers_do_not_suppress_actions_fallback():
    src = _src()
    # The nonblank-fallback decision still considers only the action cards:
    # risers alone must not suppress the "You're all set" all-clear.
    # Cards are (key, html) tuples sorted by urgency; usage_movers_html must
    # not be among them. (roster_moves_html was folded into the Next steps
    # queue, so it is no longer a standalone card.)
    assert "usage_movers_html" not in src.split("_action_cards =")[1].split("if _action_cards:")[0]
    for name in ("do_next_waiver_html", "losing_trade_html"):
        assert name in src
    assert "You're all set for Week" in src
