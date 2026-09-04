#!/usr/bin/env python3
"""Rebuild PWA / home-screen icons on an opaque white canvas.

Android and iOS fill transparent (or near-transparent) pixels with the
theme color — which was navy — so the installed app icon looked dark.
These icons are fully opaque #FFFFFF with the BR mark centered.

    python3 scripts/make_pwa_icons.py
"""
from __future__ import annotations

import struct
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "static" / "icon-512x512.png"
OUT = ROOT / "static"

# "any" icons keep most of the artwork; maskable icons stay inside the
# Android safe zone (~80% of the canvas) so the mask never crops the mark.
TARGETS = (
    ("app-icon-180.png", 180, 0.82),
    ("app-icon-192.png", 192, 0.82),
    ("app-icon-192-maskable.png", 192, 0.62),
    ("app-icon-512.png", 512, 0.82),
    ("app-icon-512-maskable.png", 512, 0.62),
    # Legacy apple-touch / manifest names so existing <link> URLs pick up the
    # white canvas. Do not overwrite icon-512x512.png — it is the source art.
    ("icon-180x180.png", 180, 0.82),
    ("icon-192x192.png", 192, 0.82),
    ("icon-192x192-maskable.png", 192, 0.62),
    ("icon-512x512-maskable.png", 512, 0.62),
)


def _chunks(data: bytes):
    assert data[:8] == b"\x89PNG\r\n\x1a\n", "not a PNG"
    i = 8
    while i < len(data):
        (length,) = struct.unpack(">I", data[i : i + 4])
        tag = data[i + 4 : i + 8]
        payload = data[i + 8 : i + 8 + length]
        yield tag, payload
        i += 12 + length


def _paeth(a, b, c):
    p = a + b - c
    pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
    if pa <= pb and pa <= pc:
        return a
    return b if pb <= pc else c


def decode_png(path: Path):
    data = path.read_bytes()
    w = h = bit = color = 0
    plte = b""
    trns = b""
    idat = b""
    for tag, payload in _chunks(data):
        if tag == b"IHDR":
            w, h, bit, color, _comp, _filt, inter = struct.unpack(">IIBBBBB", payload)
            assert bit == 8 and inter == 0, "expected 8-bit non-interlaced PNG"
        elif tag == b"PLTE":
            plte = payload
        elif tag == b"tRNS":
            trns = payload
        elif tag == b"IDAT":
            idat += payload
    raw = zlib.decompress(idat)
    if color == 3:
        bpp = 1
        palette = [(plte[i], plte[i + 1], plte[i + 2]) for i in range(0, len(plte), 3)]
        alpha = list(trns) + [255] * (len(palette) - len(trns))
    elif color == 2:
        bpp = 3
        palette = alpha = None
    elif color == 6:
        bpp = 4
        palette = alpha = None
    else:
        raise SystemExit(f"unsupported PNG color type {color}")

    stride = w * bpp
    prev = bytearray(stride)
    rows = []
    pos = 0
    for _y in range(h):
        ftype = raw[pos]
        pos += 1
        line = bytearray(raw[pos : pos + stride])
        pos += stride
        if ftype == 1:
            for x in range(bpp, stride):
                line[x] = (line[x] + line[x - bpp]) & 0xFF
        elif ftype == 2:
            for x in range(stride):
                line[x] = (line[x] + prev[x]) & 0xFF
        elif ftype == 3:
            for x in range(stride):
                a = line[x - bpp] if x >= bpp else 0
                line[x] = (line[x] + ((a + prev[x]) >> 1)) & 0xFF
        elif ftype == 4:
            for x in range(stride):
                a = line[x - bpp] if x >= bpp else 0
                c = prev[x - bpp] if x >= bpp else 0
                line[x] = (line[x] + _paeth(a, prev[x], c)) & 0xFF
        elif ftype != 0:
            raise SystemExit(f"bad PNG filter {ftype}")
        row = []
        if color == 3:
            for idx in line:
                r, g, b = palette[idx]
                al = alpha[idx]
                if al < 255:
                    f = al / 255.0
                    r = int(r * f + 255 * (1 - f))
                    g = int(g * f + 255 * (1 - f))
                    b = int(b * f + 255 * (1 - f))
                row.append((r, g, b))
        elif color == 2:
            for x in range(0, stride, 3):
                row.append((line[x], line[x + 1], line[x + 2]))
        else:
            for x in range(0, stride, 4):
                r, g, b, al = line[x], line[x + 1], line[x + 2], line[x + 3]
                if al < 255:
                    f = al / 255.0
                    r = int(r * f + 255 * (1 - f))
                    g = int(g * f + 255 * (1 - f))
                    b = int(b * f + 255 * (1 - f))
                row.append((r, g, b))
        rows.append(row)
        prev = line
    return w, h, rows


def content_bbox(px, w, h):
    x0, y0, x1, y1 = w, h, 0, 0
    for y in range(h):
        row = px[y]
        for x in range(w):
            r, g, b = row[x]
            if r < 250 or g < 250 or b < 250:
                if x < x0:
                    x0 = x
                if x > x1:
                    x1 = x
                if y < y0:
                    y0 = y
                if y > y1:
                    y1 = y
    return x0, y0, x1 + 1, y1 + 1


def render(px, w, h, size, fill):
    x0, y0, x1, y1 = content_bbox(px, w, h)
    cw, ch = x1 - x0, y1 - y0
    target = max(1, int(size * fill))
    scale = target / max(cw, ch)
    nw, nh = max(1, int(cw * scale)), max(1, int(ch * scale))
    ox, oy = (size - nw) // 2, (size - nh) // 2
    canvas = [[(255, 255, 255)] * size for _ in range(size)]
    for y in range(nh):
        sy = y0 + min(ch - 1, int(y / scale))
        srow = px[sy]
        trow = canvas[oy + y]
        for x in range(nw):
            sx = x0 + min(cw - 1, int(x / scale))
            trow[ox + x] = srow[sx]
    return canvas


def write_png(path: Path, rows):
    n = len(rows)
    raw = bytearray()
    for row in rows:
        raw.append(0)
        for r, g, b in row:
            raw += bytes((r, g, b))
    comp = zlib.compress(bytes(raw), 9)

    def chunk(tag, d):
        return struct.pack(">I", len(d)) + tag + d + struct.pack(">I", zlib.crc32(tag + d) & 0xFFFFFFFF)

    # Color type 2 = opaque RGB, no tRNS — the OS cannot punch a navy hole.
    ihdr = struct.pack(">IIBBBBB", n, n, 8, 2, 0, 0, 0)
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", comp) + chunk(b"IEND", b""))


def main():
    w, h, px = decode_png(SRC)
    for name, size, fill in TARGETS:
        dest = OUT / name
        write_png(dest, render(px, w, h, size, fill))
        print(f"wrote {dest.relative_to(ROOT)} ({size}x{size}, fill={fill})")


if __name__ == "__main__":
    main()
