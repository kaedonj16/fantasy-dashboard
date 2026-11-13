"""RZ badge must not fire at a team's own end on an abbreviation alias.

Reported live (Week 10, WAS vs NYG): Rachaad White's 2-yard rush was
tagged RZ at "1st & 10 @ WSH 8". White's team is stored as WAS while the
play-by-play yard line spells Washington WSH, so the raw string compare
in _isInRedZone / gsIsInRedZone read his own 8 as the opponent's 20.
Both renderers (ScoreZone feed and the app.js game sheet) must normalize
team abbreviations on both sides before comparing.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SZ = (ROOT / "static" / "scorezone.js").read_text(encoding="utf-8")
APP = (ROOT / "static" / "app.js").read_text(encoding="utf-8")


def _fn(src: str, name: str) -> str:
    needle = "function %s(" % name
    start = src.index(needle)
    end = src.index("\n  }\n", start) + len("\n  }\n")
    return src[start:end]


def _assignment(src: str, lhs: str) -> str:
    start = src.index(lhs)
    end = src.index("};", start) + len("};")
    return src[start:end]


CASES = {
    # The reported play: own 8, feed alias vs stored team.
    "white_own8": ["WSH 8", "WAS"],
    "white_own8_reverse": ["WAS 8", "WSH"],
    # A genuine red-zone snap for Washington's offense.
    "was_at_nyg8": ["NYG 8", "WAS"],
    "nyg_at_wsh8": ["WSH 8", "NYG"],
    # The other live aliases (JAC/JAX, LA/LAR).
    "jax_own12": ["JAC 12", "JAX"],
    "jax_own12_reverse": ["JAX 12", "JAC"],
    "lar_own15": ["LA 15", "LAR"],
    # Plain same-team / opponent checks without aliases.
    "sea_own16": ["SEA 16", "SEA"],
    "sea_at_sf16": ["SEA 16", "SF"],
    # Outside the 20 is never the red zone, alias or not.
    "was_own25_vs_nyg": ["WSH 25", "NYG"],
    # No offense team: never guess.
    "no_offense": ["NYG 8", ""],
}

EXPECTED = {
    "white_own8": False,
    "white_own8_reverse": False,
    "was_at_nyg8": True,
    "nyg_at_wsh8": True,
    "jax_own12": False,
    "jax_own12_reverse": False,
    "lar_own15": False,
    "sea_own16": False,
    "sea_at_sf16": True,
    "was_own25_vs_nyg": False,
    "no_offense": False,
}


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js not available")
def test_scorezone_is_in_red_zone_normalizes_aliases():
    script = f"""
var window = {{}};
{_fn(SZ, "_canonTeam")}
{_fn(SZ, "_isInRedZone")}
var cases = {json.dumps(CASES)};
var out = {{}};
Object.keys(cases).forEach(function(k) {{ out[k] = _isInRedZone(cases[k][0], cases[k][1]); }});
console.log(JSON.stringify(out));
"""
    out = json.loads(subprocess.check_output(["node", "-e", script], text=True))
    assert out == EXPECTED


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js not available")
def test_game_sheet_is_in_red_zone_normalizes_aliases():
    script = f"""
var window = {{}};
{_assignment(APP, "window._rzNormalizeTeam = function")}
{_assignment(APP, "window.brCanonNflTeam = function")}
{_fn(APP, "gsIsInRedZone")}
var cases = {json.dumps(CASES)};
var out = {{}};
Object.keys(cases).forEach(function(k) {{ out[k] = gsIsInRedZone(cases[k][0], cases[k][1]); }});
console.log(JSON.stringify(out));
"""
    out = json.loads(subprocess.check_output(["node", "-e", script], text=True))
    assert out == EXPECTED
