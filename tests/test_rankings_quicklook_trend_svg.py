"""Rankings hover quick-look: the 30-day value trend graph geometry.

The trend SVG is drawn in a 200x64 viewBox but the chart box on the card
is taller than that ratio. The original render stretched the SVG to the
box with preserveAspectRatio="none", which turned the end dot into a
tall oval, made the stroke uneven, and steepened every slope; its
min-max domain also slammed peaks into the chart edges and pinned a
flat series to the bottom edge, and the raw polyline drew the daily
value steps as a jagged zigzag under a flat washed-out tint.

The fixed graph follows the value-history charts' locked design
(BRValueChart): no non-uniform stretch, a padded domain, a soft
Fritsch-Carlson monotone curve through every true point (no data
smoothing), a gradient fill that fades down from the line, and a flat
solid end dot with no ring. The Node runs below stub window.BRValueChart
with the renderer's real monotone functions extracted from app.js, so
the curve under test is the site's own math.
"""
from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
JS = (ROOT / "static" / "rankings.js").read_text(encoding="utf-8")
APP_JS = (ROOT / "static" / "app.js").read_text(encoding="utf-8")


def _svg_src() -> str:
    start = JS.index("function prQlTrendSVG")
    end = JS.index("\n}\n", start) + 3
    return JS[start:end]


def _app_fn(name: str) -> str:
    # Slice from the function's declaration to the next sibling function
    # inside the BRValueChart closure (handles one-line bodies like _n).
    start = APP_JS.index("function " + name + "(")
    end = APP_JS.index("\n  function ", start + 10)
    return APP_JS[start:end]


def _mono_stub() -> str:
    return (
        "var window = { BRValueChart: { monotonePath: _monoPath } };\n"
        + _app_fn("_n")
        + "\n"
        + _app_fn("_tangents")
        + "\n"
        + _app_fn("_monoPath")
        + "\n"
    )


def _run(vals, with_chart=True, box=None) -> str:
    node = shutil.which("node")
    if not node:
        pytest.skip("node not available")
    call = "prQlTrendSVG(" + repr(list(vals)).replace("'", "")
    if box:
        call += ", " + str(box[0]) + ", " + str(box[1])
    call += ")"
    script = (_mono_stub() if with_chart else "") + _svg_src() + (
        "\nconsole.log(" + call + ");"
    )
    out = subprocess.run(
        [node, "-e", script], capture_output=True, text=True, timeout=30
    )
    assert out.returncode == 0, out.stderr
    return out.stdout.strip()


def _line_path(svg: str) -> str:
    m = re.search(r'<path d="([^"]+)" fill="none"', svg)
    assert m, svg
    return m.group(1)


def _path_ys(svg: str):
    d = _line_path(svg)
    return [float(y) for _, y in re.findall(r"(-?[\d.]+),(-?[\d.]+)", d)]


def test_no_nonuniform_stretch():
    svg = _run([100, 110, 105, 120])
    assert 'preserveAspectRatio="none"' not in svg


def test_line_is_the_shared_monotone_curve():
    svg = _run([100, 110, 105, 130, 120])
    d = _line_path(svg)
    assert "C" in d  # cubic segments, not a raw polyline
    assert "<polyline" not in svg


def test_polyline_fallback_without_brvaluechart():
    svg = _run([100, 110, 105, 120], with_chart=False)
    assert "<polyline" in svg


def test_end_dot_is_flat_and_ringless():
    svg = _run([100, 110, 105, 120])
    m = re.search(r"<circle[^>]+>", svg)
    assert m, svg
    assert 'r="3.4"' in m.group(0)
    assert "stroke" not in m.group(0)


def test_domain_is_padded_off_the_edges():
    # A two-point min/max series must not touch y=PAD or y=H-PAD; the
    # monotone control points stay inside the endpoints' band too.
    ys = _path_ys(_run([100, 120]))
    assert min(ys) > 5 and max(ys) < 59, ys


def test_flat_series_sits_at_mid_height():
    ys = _path_ys(_run([640, 640, 640, 640]))
    assert all(abs(y - 32.0) < 0.6 for y in ys), ys


def test_gradient_fill_under_the_line():
    svg = _run([100, 110, 105, 120])
    assert "<linearGradient" in svg and 'fill="url(#qlTrendFillUp)"' in svg


def test_box_matched_sizing_fills_the_zone():
    # The card's chart zone is taller than the old fixed 200x64
    # drawing; drawing at a fixed size letterboxed the graph into a
    # floating band with empty space around it. With the box's own
    # size passed in, the viewBox IS the box and the line uses it.
    svg = _run([640, 640, 640, 640], box=(240, 132))
    assert 'viewBox="0 0 240 132"' in svg
    ys = _path_ys(svg)
    assert all(abs(y - 66.0) < 0.6 for y in ys), ys
    ys = _path_ys(_run([100, 120], box=(240, 132)))
    assert min(ys) > 6 and max(ys) < 126, ys
    assert min(ys) < 40 and max(ys) > 95, ys


def test_fill_quick_measures_the_chart_box():
    fill = JS[JS.index("function prQlFillQuick"):JS.index("function prQlFillGames")]
    assert "chartEl.clientWidth, chartEl.clientHeight" in fill


def test_direction_colors_kept():
    assert "#22c55e" in _run([100, 120])
    down = _run([120, 100])
    assert "#ef4444" in down
    # The fill gradient is keyed per direction so two charts on one
    # page can never resolve each other's gradient by id.
    assert 'fill="url(#qlTrendFillDn)"' in down
    assert 'fill="url(#qlTrendFillUp)"' in _run([100, 120])
