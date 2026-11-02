"""Rankings hover quick-look: the 30-day value trend graph geometry.

The trend SVG is drawn in a 200x64 viewBox but the chart box on the card
is taller than that ratio. The original render stretched the SVG to the
box with preserveAspectRatio="none", which turned the end dot into a
tall oval, made the stroke uneven, and steepened every slope; its
min-max domain also slammed peaks into the chart edges and pinned a
flat series to the bottom edge. These tests pin the fixed geometry:
aspect preserved, domain padded, flat series centered, and a soft area
fill under the line.
"""
from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
JS = (ROOT / "static" / "rankings.js").read_text(encoding="utf-8")


def _svg_src() -> str:
    start = JS.index("function prQlTrendSVG")
    end = JS.index("\n}\n", start) + 3
    return JS[start:end]


def _run(vals) -> str:
    node = shutil.which("node")
    if not node:
        pytest.skip("node not available")
    script = (
        _svg_src()
        + "\nconsole.log(prQlTrendSVG("
        + repr(list(vals)).replace("'", "")
        + "));"
    )
    out = subprocess.run(
        [node, "-e", script], capture_output=True, text=True, timeout=30
    )
    assert out.returncode == 0, out.stderr
    return out.stdout.strip()


def _points(svg: str):
    m = re.search(r'points="([^"]+)"', svg)
    assert m, svg
    return [tuple(float(c) for c in p.split(",")) for p in m.group(1).split(" ")]


def test_no_nonuniform_stretch():
    svg = _run([100, 110, 105, 120])
    assert 'preserveAspectRatio="none"' not in svg


def test_end_dot_is_a_circle_marker():
    svg = _run([100, 110, 105, 120])
    assert "<circle" in svg and 'r="3.2"' in svg


def test_domain_is_padded_off_the_edges():
    # A two-point min/max series must not touch y=PAD or y=H-PAD.
    pts = _points(_run([100, 120]))
    ys = [p[1] for p in pts]
    assert min(ys) > 5 and max(ys) < 59, ys


def test_flat_series_sits_at_mid_height():
    pts = _points(_run([640, 640, 640, 640]))
    ys = [p[1] for p in pts]
    assert all(y == pytest.approx(32.0, abs=0.6) for y in ys), ys


def test_area_fill_under_the_line():
    svg = _run([100, 110, 105, 120])
    assert "<path" in svg and 'opacity="0.13"' in svg


def test_direction_colors_kept():
    assert "#22c55e" in _run([100, 120])
    assert "#ef4444" in _run([120, 100])
