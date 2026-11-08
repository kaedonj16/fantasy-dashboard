#!/usr/bin/env python3
"""Rebuild the PWA / home-screen icons: the BR mark as a shaded glyph on a
site-navy gradient tile (the Stripe/ESPN depth recipe, in site colours).

Depth lives in the glyph: the white BR (static/BR_Mark_dark.png, shapes
untouched) is re-shaded with a vertical gradient, white at the top to
accent-soft #e9eef5 at the bottom, the football shaded with the same light,
a light edge on top and a two-layer soft shadow underneath. The tile runs
from site navy #122d4b (lightened with site blue #2563eb) at the top-left
to the site's dark surface #0d2137 at the bottom-right, with pooled light
tinted by the sky accent #38bdf8.

Placement is optical, not geometric: BR_Mark_dark.png carries faint stray
alpha past the visible mark, so the mark is trimmed on a hard alpha
threshold and positioned with its alpha centroid on the tile centre.
"any" icons fill 90% of the width; maskable icons 70%, inside the Android
safe zone including the shadow. All icons are fully opaque RGB.

    python3 scripts/make_pwa_icons.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

ROOT = Path(__file__).resolve().parents[1]
STATIC = ROOT / "static"
SS = 2048  # supersample canvas

NAVY = (0x12, 0x2D, 0x4B)
DARK = (0x0D, 0x21, 0x37)
BLUE = (0x25, 0x63, 0xEB)
SKY = (0x38, 0xBD, 0xF8)
SOFT = (0xE9, 0xEE, 0xF5)

TARGETS = (
    ("app-icon-180.png", 180, 0.90),
    ("app-icon-192.png", 192, 0.90),
    ("app-icon-192-maskable.png", 192, 0.70),
    ("app-icon-512.png", 512, 0.90),
    ("app-icon-512-maskable.png", 512, 0.70),
    # Legacy apple-touch / manifest names. icon-512x512.png stays untouched:
    # it is the flat source art other tooling derives from.
    ("icon-180x180.png", 180, 0.90),
    ("icon-192x192.png", 192, 0.90),
    ("icon-192x192-maskable.png", 192, 0.70),
    ("icon-512x512-maskable.png", 512, 0.70),
)


def _mix(c1, c2, t):
    return tuple(int(a + (b - a) * t) for a, b in zip(c1, c2))


def _tile(size: int) -> Image.Image:
    y, x = np.mgrid[0:size, 0:size].astype(np.float32)
    t = np.clip((x * 0.45 + y) / (size * 1.45), 0, 1) ** 1.1
    c1 = np.array(_mix(NAVY, BLUE, 0.30), np.float32)
    c2 = np.array(DARK, np.float32)
    img = c1[None, None, :] * (1 - t[..., None]) + c2[None, None, :] * t[..., None]
    r = np.sqrt((x - size * 0.30) ** 2 + (y - size * 0.10) ** 2) / (size * 0.75)
    glow = (np.clip(1 - r, 0, 1) ** 2 * 0.16)[..., None]
    img = img * (1 - glow) + np.array(SKY, np.float32)[None, None, :] * glow
    return Image.fromarray(np.clip(img, 0, 255).astype(np.uint8), "RGB").convert("RGBA")


def _shaded_mark() -> Image.Image:
    im = Image.open(STATIC / "BR_Mark_dark.png").convert("RGBA")
    a = np.asarray(im)
    ys, xs = np.where(a[..., 3] > 128)  # hard trim: stray faint alpha is not mark
    im = im.crop((xs.min(), ys.min(), xs.max() + 1, ys.max() + 1))
    arr = np.asarray(im).astype(np.float32)
    h, w = arr.shape[0], arr.shape[1]
    t = np.tile(((np.arange(h, dtype=np.float32) / max(h - 1, 1)) ** 1.15)[:, None], (1, w))
    rgb, alpha = arr[..., :3], arr[..., 3]
    letter = (alpha > 180) & (rgb.mean(axis=-1) > 200)
    grad = np.array((255, 255, 255), np.float32)[None, None, :] * (1 - t[..., None]) + \
        np.array(SOFT, np.float32)[None, None, :] * t[..., None]
    out = rgb.copy()
    out[letter] = grad[letter]
    f = (1.09 - 0.20 * t)[..., None]
    rest = ~letter
    out[rest] = np.clip(rgb[rest] * f[rest], 0, 255)
    return Image.fromarray(np.dstack([out, alpha]).astype(np.uint8), "RGBA")


def _silhouette(img: Image.Image, color) -> Image.Image:
    sil = Image.new("RGBA", img.size, color)
    sil.putalpha(img.getchannel("A"))
    return sil


def _render(mark: Image.Image, fill: float) -> Image.Image:
    tw = int(SS * fill)
    m = mark.resize((tw, int(mark.height * tw / mark.width)), Image.LANCZOS)
    wgt = np.asarray(m)[..., 3].astype(np.float32) / 255.0
    cx = (wgt.sum(axis=0) * np.arange(wgt.shape[1])).sum() / wgt.sum()
    cy = (wgt.sum(axis=1) * np.arange(wgt.shape[0])).sum() / wgt.sum()
    mx, my = int(SS / 2 - cx), int(SS / 2 - cy)  # alpha centroid on tile centre
    canvas = _tile(SS)
    amb = _silhouette(m, (2, 10, 24, 255)).filter(ImageFilter.GaussianBlur(SS * 0.020))
    amb.putalpha(amb.getchannel("A").point(lambda v: int(v * 0.34)))
    canvas.alpha_composite(amb, (mx, my + int(SS * 0.020)))
    con = _silhouette(m, (2, 10, 24, 255)).filter(ImageFilter.GaussianBlur(SS * 0.006))
    con.putalpha(con.getchannel("A").point(lambda v: int(v * 0.30)))
    canvas.alpha_composite(con, (mx, my + int(SS * 0.008)))
    lite = _silhouette(m, (255, 255, 255, 255))
    lite.putalpha(lite.getchannel("A").point(lambda v: int(v * 0.38)))
    canvas.alpha_composite(lite, (mx, my - max(2, int(SS * 0.0016))))
    canvas.alpha_composite(m, (mx, my))
    return canvas.convert("RGB")


def main() -> None:
    mark = _shaded_mark()
    for name, size, fill in TARGETS:
        _render(mark, fill).resize((size, size), Image.LANCZOS).save(STATIC / name)
        print(f"wrote static/{name} ({size}x{size}, fill={fill})")


if __name__ == "__main__":
    main()
