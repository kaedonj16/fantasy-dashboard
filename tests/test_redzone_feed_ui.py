"""Source contracts for the Redzone play feed controls and score summary."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REDZONE_JS = (ROOT / "static" / "redzone.js").read_text(encoding="utf-8")
DASHBOARD_CSS = (ROOT / "static" / "dashboard.css").read_text(encoding="utf-8")


def test_redzone_feed_is_always_latest_without_sort_switch():
    assert "var list = _chronoSort(filtered);" in REDZONE_JS
    assert "_feedSort" not in REDZONE_JS
    assert "function _softRank(" not in REDZONE_JS
    assert "rz-sort-btn" not in REDZONE_JS
    assert "For You</button>" not in REDZONE_JS


def test_play_summary_groups_clock_delta_and_labeled_total():
    delta_markup = REDZONE_JS.split(
        "'<div class=\"rz-event-delta ' + deltaCls + '\">'", 1
    )[1][:500]

    assert "rz-event-clock" in delta_markup
    assert "rz-event-delta-pts" in delta_markup
    assert "rz-event-total" in delta_markup
    assert "</span> total" in delta_markup
    assert ".rz-event-delta {" in DASHBOARD_CSS
    assert "border-radius: 10px;" in DASHBOARD_CSS


def test_latest_order_is_comparable_across_simultaneous_games():
    """Quarter/clock wall time must win over game-local provider sequence."""
    chrono_key = REDZONE_JS.split("function _chronoKey(ev) {", 1)[1].split(
        "function _chronoSort(list) {", 1
    )[0]

    assert chrono_key.index("var q = parseInt(ev.gameQuarter, 10);") < chrono_key.index(
        "if (ev.gameId && ev.seq != null)"
    )
    assert "return list.slice().sort" in REDZONE_JS
