#!/usr/bin/env python3
"""Generate the extension's PNG icons from the BR Fantasy logo (no deps).

Decodes static/icon-512x512.png (8-bit palette PNG), crops to the BR + football
mark (dropping the "FANTASY FOOTBALL" wordmark so the small sizes stay legible),
centers it on a square white tile, and box-downscales to 16/32/48/128.

    python3 extension/icons/make_icons.py
"""
import os
import struct
import zlib

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.normpath(os.path.join(HERE, "..", "..", "static", "icon-512x512.png"))
SIZES = (16, 32, 48, 128)


def _chunks(data):
    assert data[:8] == b"\x89PNG\r\n\x1a\n", "not a PNG"
    i = 8
    while i < len(data):
        (length,) = struct.unpack(">I", data[i:i + 4])
        tag = data[i + 4:i + 8]
        payload = data[i + 8:i + 8 + length]
        yield tag, payload
        i += 12 + length


def _paeth(a, b, c):
    p = a + b - c
    pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
    if pa <= pb and pa <= pc:
        return a
    return b if pb <= pc else c


def decode_palette_png(path):
    data = open(path, "rb").read()
    w = h = 0
    plte = b""
    trns = b""
    idat = b""
    for tag, payload in _chunks(data):
        if tag == b"IHDR":
            w, h, bd, ct, comp, filt, inter = struct.unpack(">IIBBBBB", payload)
            assert bd == 8 and ct == 3 and inter == 0, "expected 8-bit non-interlaced palette PNG"
        elif tag == b"PLTE":
            plte = payload
        elif tag == b"tRNS":
            trns = payload
        elif tag == b"IDAT":
            idat += payload
    raw = zlib.decompress(idat)
    palette = [(plte[i], plte[i + 1], plte[i + 2]) for i in range(0, len(plte), 3)]
    alpha = list(trns) + [255] * (len(palette) - len(trns))

    # Un-filter: 1 byte per pixel (palette index), bpp = 1.
    stride = w
    prev = bytearray(stride)
    out = []
    pos = 0
    for _y in range(h):
        ftype = raw[pos]; pos += 1
        line = bytearray(raw[pos:pos + stride]); pos += stride
        if ftype == 1:      # Sub
            for x in range(1, stride):
                line[x] = (line[x] + line[x - 1]) & 0xFF
        elif ftype == 2:    # Up
            for x in range(stride):
                line[x] = (line[x] + prev[x]) & 0xFF
        elif ftype == 3:    # Average
            for x in range(stride):
                a = line[x - 1] if x else 0
                line[x] = (line[x] + ((a + prev[x]) >> 1)) & 0xFF
        elif ftype == 4:    # Paeth
            for x in range(stride):
                a = line[x - 1] if x else 0
                c = prev[x - 1] if x else 0
                line[x] = (line[x] + _paeth(a, prev[x], c)) & 0xFF
        # map indices -> RGBA, compositing over white for any transparency
        row = []
        for idx in line:
            r, g, b = palette[idx]
            al = alpha[idx]
            if al < 255:
                f = al / 255.0
                r = int(r * f + 255 * (1 - f))
                g = int(g * f + 255 * (1 - f))
                b = int(b * f + 255 * (1 - f))
            row.append((r, g, b))
        out.append(row)
        prev = line
    return w, h, out


def content_bbox(px, w, h):
    """Bounding box of the mark, ignoring the wordmark band near the bottom."""
    y_limit = int(h * 0.63)
    x0, y0, x1, y1 = w, h, 0, 0
    for y in range(y_limit):
        row = px[y]
        for x in range(w):
            r, g, b = row[x]
            if max(r, g, b) < 235:  # any non-near-white ink
                if x < x0: x0 = x
                if x > x1: x1 = x
                if y < y0: y0 = y
                if y > y1: y1 = y
    return x0, y0, x1 + 1, y1 + 1


def square_canvas(px, w, h):
    x0, y0, x1, y1 = content_bbox(px, w, h)
    cw, ch = x1 - x0, y1 - y0
    side = max(cw, ch)
    pad = int(side * 0.12)
    canvas = side + 2 * pad
    tile = [[(255, 255, 255)] * canvas for _ in range(canvas)]
    ox = (canvas - cw) // 2
    oy = (canvas - ch) // 2
    for y in range(ch):
        srow = px[y0 + y]
        trow = tile[oy + y]
        for x in range(cw):
            trow[ox + x] = srow[x0 + x]
    return canvas, tile


def box_downscale(tile, csize, n):
    scale = csize / n
    out = []
    for oy in range(n):
        iy0, iy1 = int(oy * scale), max(int(oy * scale) + 1, int((oy + 1) * scale))
        row = []
        for ox in range(n):
            ix0, ix1 = int(ox * scale), max(int(ox * scale) + 1, int((ox + 1) * scale))
            r = g = b = cnt = 0
            for yy in range(iy0, iy1):
                trow = tile[yy]
                for xx in range(ix0, ix1):
                    pr, pg, pb = trow[xx]
                    r += pr; g += pg; b += pb; cnt += 1
            row.append((r // cnt, g // cnt, b // cnt, 255))
        out.append(row)
    return out


def write_png(path, rows):
    n = len(rows)
    raw = bytearray()
    for row in rows:
        raw.append(0)
        for px in row:
            raw += bytes(px)
    comp = zlib.compress(bytes(raw), 9)

    def chunk(tag, d):
        return struct.pack(">I", len(d)) + tag + d + struct.pack(">I", zlib.crc32(tag + d) & 0xFFFFFFFF)

    ihdr = struct.pack(">IIBBBBB", n, n, 8, 6, 0, 0, 0)
    with open(path, "wb") as fh:
        fh.write(b"\x89PNG\r\n\x1a\n")
        fh.write(chunk(b"IHDR", ihdr))
        fh.write(chunk(b"IDAT", comp))
        fh.write(chunk(b"IEND", b""))


def main():
    w, h, px = decode_palette_png(SRC)
    csize, tile = square_canvas(px, w, h)
    for size in SIZES:
        write_png(os.path.join(HERE, f"icon{size}.png"), box_downscale(tile, csize, size))
        print(f"wrote icon{size}.png")


if __name__ == "__main__":
    main()
