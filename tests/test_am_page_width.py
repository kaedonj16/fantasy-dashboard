"""Advanced Metrics page width (Kaedon, 2026-10-19: "isnt wide enough").

The page card was capped at 1400px, so wide presets (Key Metrics = 9
metric columns, the RB preset = 12) squeezed their bars or scrolled on
desktop monitors. The cap is now 1920px: effectively full width on a
1920px display (body margin aside), centred with margins on anything
larger. The pinned-player compare modal widens in step (1400 -> 1760).

A single metric is the exception (Kaedon's Yards/Carry screenshot,
2026-10-19): one bar stretched across the whole wide card while player
names still truncated at 220px. With no extra-metric header present the
card stays at a readable 1200px and Player widens to 300px on desktop.

Phones are untouched: the card is width:100% under the cap either way,
and the wider Player column is desktop-only.
"""
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
PAGE = (ROOT / "dashboard_services" / "pages" / "advanced_metrics_page.py").read_text(encoding="utf-8")
SW = (ROOT / "static" / "sw.js").read_text(encoding="utf-8")


def test_page_card_is_widened_to_1920():
    assert ".card.central:has(#amTable) { max-width:1920px; transition:max-width .35s ease; }" in PAGE
    assert ".card.central:has(#amTable) { max-width:1400px; }" not in PAGE


def test_width_switch_animates_instead_of_snapping():
    # The 1200 <-> 1920 switch (one metric vs several) eases over .35s;
    # without the transition on the base card rule it jumps in one frame.
    assert "transition:max-width .35s ease;" in PAGE


def test_single_metric_view_stays_readable():
    # No th.am-extra-header in the table -> no extra metric columns.
    assert ".card.central:has(#amTable):not(:has(#amTable th.am-extra-header)) { max-width:1200px; }" in PAGE
    assert ".card.central:has(#amTable):not(:has(#amTable th.am-extra-header)) .am-player { width:300px; max-width:300px; }" in PAGE


def test_compare_modal_widens_in_step():
    assert ".am-cmp-card { max-width:min(95vw,1760px); }" in PAGE
    assert ".am-cmp-card { max-width:min(95vw,1400px); }" not in PAGE


def test_sw_cache_bumped():
    assert "br-fantasy-v65" in SW
