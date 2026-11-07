#!/usr/bin/env python3
"""Rebuild the PWA / home-screen icons: the BR mark lifted off a textured
white surface.

The logo art (static/BR_Mark.png) is used untouched. Depth comes only from
a two-layer shadow under the mark (tight contact + soft ambient) that fades
out well inside the canvas, over a white background with a faint radial
light falloff and very fine grain. "any" icons draw the mark at 80% width;
maskable icons at 62% so Android masks never crop the logo or its shadow.
All icons are fully opaque RGB (the OS must not punch a theme-color hole).

    python3 scripts/make_pwa_icons.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

ROOT = Path(__file__).resolve().parents[1]
STATIC = ROOT / "static"
SS = 2048  # supersample canvas

TARGETS = (
    ("app-icon-180.png", 180, 0.80),
    ("app-icon-192.png", 192, 0.80),
    ("app-icon-192-maskable.png", 192, 0.62),
    ("app-icon-512.png", 512, 0.80),
    ("app-icon-512-maskable.png", 512, 0.62),
    # Legacy apple-touch / manifest names. icon-512x512.png stays untouched:
    # it is the flat source art other tooling derives from.
    ("icon-180x180.png", 180, 0.80),
    ("icon-192x192.png", 192, 0.80),
    ("icon-192x192-maskable.png", 192, 0.62),
    ("icon-512x512-maskable.png", 512, 0.62),
)


def _background(size: int) -> Image.Image:
    rng = np.random.default_rng(7)
    y, x = np.mgrid[0:size, 0:size].astype(np.float32)
    r = np.sqrt((x - size * 0.5) ** 2 + (y - size * 0.42) ** 2) / (size * 0.85)
    m = np.clip((0.70 - r) / 0.70, 0, 1) ** 1.2  # texture fades to pure white at edges
    base = 255 - (np.clip(r, 0, 1) ** 1.6) * 13 * m
    img = np.stack([base, base + 1 * m, base + 3 * m], axis=-1)
    img = img + (rng.normal(0, 2.2, (size, size)).astype(np.float32) * m)[..., None]
    return Image.fromarray(np.clip(img, 0, 255).astype(np.uint8), "RGB").convert("RGBA")


def _silhouette(img: Image.Image, color) -> Image.Image:
    sil = Image.new("RGBA", img.size, color)
    sil.putalpha(img.getchannel("A"))
    return sil


def _render(mark_src: Image.Image, fill: float) -> Image.Image:
    tw = int(SS * fill)
    th = int(mark_src.height * (tw / mark_src.width))
    mark = mark_src.resize((tw, th), Image.LANCZOS)
    mx = (SS - tw) // 2
    my = (SS - th) // 2 - int(SS * 0.010)
    canvas = _background(SS)
    amb = _silhouette(mark, (13, 38, 70, 255)).filter(ImageFilter.GaussianBlur(SS * 0.020))
    amb.putalpha(amb.getchannel("A").point(lambda v: int(v * 0.20)))
    canvas.alpha_composite(amb, (mx, my + int(SS * 0.020)))
    con = _silhouette(mark, (13, 38, 70, 255)).filter(ImageFilter.GaussianBlur(SS * 0.006))
    con.putalpha(con.getchannel("A").point(lambda v: int(v * 0.24)))
    canvas.alpha_composite(con, (mx, my + int(SS * 0.008)))
    canvas.alpha_composite(mark, (mx, my))
    return canvas.convert("RGB")


def main() -> None:
    mark_src = Image.open(STATIC / "BR_Mark.png").convert("RGBA")
    mark_src = mark_src.crop(mark_src.getbbox())
    for name, size, fill in TARGETS:
        _render(mark_src, fill).resize((size, size), Image.LANCZOS).save(STATIC / name)
        print(f"wrote static/{name} ({size}x{size}, fill={fill})")


if __name__ == "__main__":
    main()
