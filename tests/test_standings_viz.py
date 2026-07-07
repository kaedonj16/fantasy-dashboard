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
