"""Source contracts for the Redzone play feed controls and score summary."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REDZONE_JS = (ROOT / "static" / "redzone.js").read_text(encoding="utf-8")
DASHBOARD_CSS = (ROOT / "static" / "dashboard.css").read_text(encoding="utf-8")


def test_redzone_feed_is_always_latest_without_sort_switch():
    assert "var list = _chronoSort(filtered);" in REDZONE_JS
    assert "_feedSort" not in REDZONE_JS
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
