"""Matchup H2H line placement (Kaedon's screenshot, 2026-10-19).

The head-to-head lines (this-season H2H + the client-filled all-time
rivalry line) used to render under the team names at the bottom of the
matchup head. Kaedon moved them to the top: centre slot of the badge
row, on the same row as the GOTW and Divisional badges, always in the
middle. The badge row is a 1fr / auto / 1fr grid so the centre is the
true middle regardless of badge widths; on phones the centre slot is
capped and wraps inside its pill so the badges never overlap it.

Verified in a Playwright replica of the head: centre-slot centre ==
card centre at 874px, 390px and 360px, zero overlap, zero page overflow.
"""
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
CSS = (ROOT / "static" / "dashboard.css").read_text(encoding="utf-8")
MATCHUPS = (
    ROOT / "dashboard_services" / "matchups.py"
).read_text(encoding="utf-8")


def test_h2h_renders_in_the_badge_row_centre_slot():
    assert "<div class='m-espn-badges-c'>{h2h_html}</div>" in MATCHUPS
    # Exactly one emission site for the H2H block: the centre slot.
    assert MATCHUPS.count("{h2h_html}") == 1
    # The badge row now renders even when only the H2H lines exist
    # (no GOTW, no divisional game), so the centre is always populated.
    assert "if gotw_badge_html or div_badge_html or h2h_html:" in MATCHUPS


def test_badge_row_grid_keeps_the_centre_at_the_true_middle():
    assert (
        ".m-espn-badges { display: grid; "
        "grid-template-columns: 1fr auto 1fr;"
    ) in CSS
    assert ".m-espn-badges-l { justify-self: start; }" in CSS
    assert ".m-espn-badges-r { justify-self: end; }" in CSS
    assert ".m-espn-badges-c {" in CSS
    # The old bottom-of-head spacing is gone now the lines live up top.
    assert "margin-top: 8px; }" not in CSS.split(".m-head-espn .m-rivalry")[1].split("\n")[0]


def test_phone_centre_slot_is_capped_and_badges_shrink():
    assert ".m-espn-badges-c { max-width: 144px; }" in CSS
    assert ".m-espn-badges .div-badge { font-size: 10px; padding: 2px 6px; }" in CSS
    assert ".m-espn-badges .m-gotw-badge { font-size: 10px; padding: 3px 6px; }" in CSS
