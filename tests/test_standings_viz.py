"""Tests for utils.standings_viz SVG builders."""
from utils.standings_viz import luck_quadrant_svg, value_age_svg


def _analysis(n):
    out = {}
    for i in range(n):
        out[f"T{i}"] = {
            "games": 4,
            "all_play_pct": i / max(n - 1, 1),
            "actual_wins": float(i % 4),
            "expected_wins": float(i),
            "luck_delta": float(i - 2),
        }
    return out


def test_luck_quadrant_empty_below_three():
    assert luck_quadrant_svg({}, "") == ""
    assert luck_quadrant_svg(_analysis(2), "") == ""


def test_luck_quadrant_renders_svg():
    svg = luck_quadrant_svg(_analysis(4), "T1")
    assert svg.startswith("<svg")
    assert svg.rstrip().endswith("</svg>")
    assert "Lucky" in svg and "Unlucky" in svg
    # Viewer team gets the thicker highlight ring.
    assert 'stroke-width="2"' in svg


def test_luck_quadrant_skips_teams_with_no_games():
    a = _analysis(4)
    a["T0"]["games"] = 0
    # Only 3 teams remain with games -> still renders.
    svg = luck_quadrant_svg(a, "")
    assert svg.startswith("<svg")


def test_luck_quadrant_hidden_before_any_wins():
    # Season start: everyone has games but nobody has won yet -> hide it.
    a = _analysis(5)
    for v in a.values():
        v["actual_wins"] = 0.0
    assert luck_quadrant_svg(a, "") == ""


def test_luck_quadrant_uses_owner_colors():
    svg = luck_quadrant_svg(_analysis(4), "", owner_colors={"T1": "#123456"})
    assert 'fill="#123456"' in svg


def _vrows(n):
    return [
        {"owner": f"T{i}", "total_value": 1000 + i * 100, "avg_age": 24 + i * 0.5, "n": 15}
        for i in range(n)
    ]


def test_value_age_empty_below_three():
    assert value_age_svg([], "") == ""
    assert value_age_svg(_vrows(2), "") == ""


def test_value_age_renders_svg_with_quadrant_labels():
    svg = value_age_svg(_vrows(5), "T2")
    assert svg.startswith("<svg")
    assert "Young &amp; loaded" in svg
    assert "Win-now" in svg
    assert "Rebuilding" in svg
    assert "Aging out" in svg


def test_value_age_ignores_zero_value_teams():
    rows = _vrows(3) + [{"owner": "Empty", "total_value": 0, "avg_age": 0, "n": 0}]
    svg = value_age_svg(rows, "")
    assert svg.startswith("<svg")
    assert "Empty" not in svg


def test_value_age_uses_owner_colors():
    svg = value_age_svg(_vrows(4), "", owner_colors={"T0": "#abcdef"})
    assert 'fill="#abcdef"' in svg


LONG_NAMES = ["Caleb's Casting Couch", "Make Kamaraca Great Again", "Pittsburgh Pilots"]


def _long_vrows():
    return [
        {"owner": LONG_NAMES[0], "total_value": 7108, "avg_age": 25.2, "n": 15},
        {"owner": LONG_NAMES[1], "total_value": 4123, "avg_age": 29.5, "n": 15},
        {"owner": LONG_NAMES[2], "total_value": 5886, "avg_age": 26.0, "n": 15},
    ]


def test_value_age_shows_full_long_team_names():
    svg = value_age_svg(_long_vrows(), LONG_NAMES[0])
    # No hard 12-char truncation anywhere: every full name is in the SVG.
    for name in LONG_NAMES:
        assert name.replace("'", "&#x27;") in svg or name in svg
    assert "Caleb&#x27;s Cast<" not in svg and "Caleb's Cast<" not in svg


def test_value_age_right_edge_label_is_end_anchored():
    svg = value_age_svg(_long_vrows(), "")
    # The oldest team sits at the right edge; its long label must flip to
    # end-anchored so it stays inside the frame.
    assert 'text-anchor="end"' in svg


def test_luck_shows_full_long_team_names():
    a = {
        LONG_NAMES[0]: {"games": 7, "all_play_pct": 0.8, "actual_wins": 6.0,
                        "expected_wins": 5.6, "luck_delta": 0.4},
        LONG_NAMES[1]: {"games": 7, "all_play_pct": 0.3, "actual_wins": 2.0,
                        "expected_wins": 2.1, "luck_delta": -0.1},
        LONG_NAMES[2]: {"games": 7, "all_play_pct": 0.55, "actual_wins": 4.0,
                        "expected_wins": 3.9, "luck_delta": 0.1},
    }
    svg = luck_quadrant_svg(a, LONG_NAMES[0])
    for name in LONG_NAMES:
        assert name.replace("'", "&#x27;") in svg or name in svg
