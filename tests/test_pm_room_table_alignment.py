"""Regression guard: player modal Team-tab room table alignment.

Every `.pm-troom-row` (header included) is its own CSS grid, so the column
tracks must be identical no matter what a row contains. Content-sized
(`auto`) tracks resolve per row: the header row sizes its columns to the
wide uppercase labels while data rows size theirs to the narrow values,
and the two drift apart (worst at the left, converging at the right edge).
The RB room shipped `repeat(5, minmax(58px, auto))` while QB/WR/TE used
fixed px tracks, so only the RB room headers misaligned with the data.
"""

import re
from pathlib import Path

CSS = (Path(__file__).resolve().parent.parent / "static" / "dashboard.css").read_text(
    encoding="utf-8"
)


def _rule_body(selector_pattern):
    match = re.search(selector_pattern + r"\s*\{([^}]*)\}", CSS)
    assert match, f"missing CSS rule for {selector_pattern}"
    return match.group(1)


def _template_tracks(body):
    match = re.search(r"grid-template-columns:\s*([^;]+);", body)
    assert match, "room row rule must declare grid-template-columns"
    return match.group(1).split()


def test_rb_room_columns_are_fixed_tracks():
    body = _rule_body(r"\.pm-room-rb \.pm-troom-row")
    assert "auto" not in body, (
        "RB room rows must not use content-sized tracks: each row is its own "
        "grid, so auto tracks make the header and data columns drift apart"
    )
    assert "repeat(" not in body
    tracks = _template_tracks(body)
    # slot, name, snap, target share, carry share, touch share, PPR PPG
    assert len(tracks) == 7
    for track in tracks[2:]:
        assert re.fullmatch(r"\d+px", track), f"metric track {track!r} must be fixed px"


def test_rb_room_tracks_fit_header_labels():
    # Header labels render at 9px/800 uppercase with 0.04em letter-spacing;
    # TARGET SHARE is the widest at ~74px. Each share column must clear its
    # label or the header text wraps/spills instead of aligning.
    tracks = _template_tracks(_rule_body(r"\.pm-room-rb \.pm-troom-row"))
    widths = [int(t[:-2]) for t in tracks[2:]]
    assert widths[0] >= 88  # snap cell: bar + bold % (base table uses 92px)
    assert widths[1] >= 78  # TARGET SHARE
    assert widths[2] >= 72  # CARRY SHARE
    assert widths[3] >= 74  # TOUCH SHARE
    assert widths[4] >= 50  # PPR PPG


def test_other_room_columns_stay_fixed():
    for pattern in (
        r"\.pm-room-qb \.pm-troom-row,\s*\.pm-room-wr \.pm-troom-row,\s*\.pm-room-te \.pm-troom-row",
        r"\.pm-troom-row",
    ):
        body = _rule_body(pattern)
        assert "auto" not in body


def test_mobile_rb_room_layout_unchanged():
    # Phones hide the three share columns and show the stacked details line;
    # that 4-track template is fixed already and must keep its shape.
    bodies = re.findall(r"\.pm-room-rb \.pm-troom-row\s*\{([^}]*)\}", CSS)
    mobile = [b for b in bodies if "grid-template-columns:18px" in b]
    assert mobile, "mobile RB room rule missing"
    assert len(_template_tracks(mobile[0])) == 4
