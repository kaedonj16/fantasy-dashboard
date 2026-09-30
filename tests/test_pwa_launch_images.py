"""Contracts for the iOS PWA launch images (apple-touch-startup-image).

A cold PWA launch shows iOS's own launch screen from icon tap until first
HTML paint. Without per-size startup-image links that screen is blank white,
and iOS ignores any image whose pixel dimensions do not exactly match the
device. These tests lock the matrix declared in app.py's BASE_HTML head
against the generated files in static/splash/ (scripts/gen_splash_images.py):

- every link resolves to a real file,
- each PNG's IHDR dimensions exactly equal media (device-width x ratio,
  device-height x ratio),
- every size ships in both light and dark, on the exact #appSplash
  backgrounds (#f8fafc / #020617), with the BR logo actually present,
- hrefs carry no query string (some iOS versions fail to match them).

Pure file/source tests so they run in the light unit shard; the pixel test
importorskips PIL, mirroring the repo's optional-dependency precedent.
"""
import re
import struct
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
APP_PY = (ROOT / "app.py").read_text(encoding="utf-8")

LIGHT_BG = (248, 250, 252)  # #f8fafc
DARK_BG = (2, 6, 23)  # #020617

# (device-width CSS px, device-height CSS px, -webkit-device-pixel-ratio).
# 414x896 exists at BOTH @3 (1242x2688) and @2 (828x1792); the ratio in the
# media query is what disambiguates them, so both must be emitted.
EXPECTED_SIZES = [
    (430, 932, 3),
    (393, 852, 3),
    (390, 844, 3),
    (428, 926, 3),
    (414, 896, 3),
    (375, 812, 3),
    (360, 780, 3),
    (414, 896, 2),
    (375, 667, 2),
    (414, 736, 3),
    (1024, 1366, 2),
    (834, 1194, 2),
]

LINK_RE = re.compile(
    r'<link rel="apple-touch-startup-image" media="([^"]+)" href="([^"]+)">'
)
MEDIA_RE = re.compile(
    r"\(prefers-color-scheme: (light|dark)\) and "
    r"\(device-width: (\d+)px\) and \(device-height: (\d+)px\) and "
    r"\(-webkit-device-pixel-ratio: (\d+)\) and \(orientation: portrait\)"
)


def _links():
    out = []
    for media, href in LINK_RE.findall(APP_PY):
        m = MEDIA_RE.fullmatch(media)
        assert m, f"startup-image media query not in the expected shape: {media}"
        scheme, css_w, css_h, ratio = m.group(1), int(m.group(2)), int(m.group(3)), int(m.group(4))
        out.append(
            {
                "scheme": scheme,
                "css_w": css_w,
                "css_h": css_h,
                "ratio": ratio,
                "href": href,
                "path": ROOT / href.lstrip("/"),
            }
        )
    return out


LINKS = _links()


def _ihdr_size(path):
    data = path.read_bytes()[:24]
    assert data[:8] == b"\x89PNG\r\n\x1a\n", f"{path.name} is not a PNG"
    assert data[12:16] == b"IHDR", f"{path.name} has no IHDR chunk"
    return struct.unpack(">II", data[16:24])


def test_startup_image_matrix_is_complete():
    assert len(LINKS) == 2 * len(EXPECTED_SIZES)
    seen = {(l["scheme"], l["css_w"], l["css_h"], l["ratio"]) for l in LINKS}
    assert len(seen) == len(LINKS), "duplicate startup-image link"
    for css_w, css_h, ratio in EXPECTED_SIZES:
        assert ("light", css_w, css_h, ratio) in seen
        assert ("dark", css_w, css_h, ratio) in seen


def test_every_link_resolves_to_an_existing_file():
    for link in LINKS:
        assert link["path"].is_file(), f"missing file for {link['href']}"


def test_ihdr_dimensions_match_media_query_exactly():
    # iOS ignores a startup image whose pixels are not the exact device size.
    for link in LINKS:
        expected = (link["css_w"] * link["ratio"], link["css_h"] * link["ratio"])
        assert _ihdr_size(link["path"]) == expected, link["href"]
        stem = f"splash-{expected[0]}x{expected[1]}"
        if link["scheme"] == "dark":
            stem += "-dark"
        assert link["path"].name == stem + ".png", link["href"]


def test_startup_hrefs_carry_no_query_string():
    for link in LINKS:
        assert "?" not in link["href"], link["href"]


def test_corner_pixels_and_logo_presence():
    pytest.importorskip("PIL")
    from PIL import Image

    for link in LINKS:
        bg = LIGHT_BG if link["scheme"] == "light" else DARK_BG
        im = Image.open(link["path"]).convert("RGB")
        w, h = im.size
        corners = [
            im.getpixel((0, 0)),
            im.getpixel((w - 1, 0)),
            im.getpixel((0, h - 1)),
            im.getpixel((w - 1, h - 1)),
        ]
        assert all(c == bg for c in corners), (link["href"], corners)
        center = im.crop((int(w * 0.2), int(h * 0.3), int(w * 0.8), int(h * 0.7)))
        px = list(center.getdata())
        non_bg = sum(1 for p in px if p != bg) / len(px)
        assert non_bg > 0.01, f"no logo pixels found in {link['href']}"


def test_active_theme_splash_logo_is_preloaded():
    # The head boot script creates a preload for ONLY the resolved theme's
    # #appSplash logo, so the logo paints with its background on cold load.
    assert "l.rel='preload';l.as='image'" in APP_PY
    assert "'/static/BR_Logo_dark.png?v=6c0c4828'" in APP_PY
    assert "'/static/BR_Logo.png?v=6c0c4828'" in APP_PY


def test_generator_script_stays_in_sync():
    script = (ROOT / "scripts" / "gen_splash_images.py").read_text(encoding="utf-8")
    for css_w, css_h, ratio in EXPECTED_SIZES:
        assert f"({css_w}, {css_h}, {ratio})" in script
