#!/usr/bin/env python3
"""Generate the extension's PNG icons (no third-party deps).

A navy rounded-square tile with a tan football on it. Run from the repo root:

    python3 extension/icons/make_icons.py
"""
import os
import struct
import zlib

NAVY = (18, 45, 75, 255)        # brand --accent
TAN = (226, 170, 101, 255)      # football body
TAN_DARK = (150, 104, 52, 255)  # football edge
WHITE = (245, 247, 250, 255)    # seam + laces
CLEAR = (0, 0, 0, 0)


def _pixels(n):
    cx = cy = (n - 1) / 2.0
    r = n * 0.22                 # corner radius
    half = n / 2.0
    bx, by = n * 0.40, n * 0.245  # football radii
    seam_h = max(1.0, n * 0.022)
    lace_w = max(1.0, n * 0.016)
    rows = []
    for y in range(n):
        row = []
        for x in range(n):
            # rounded-square background mask
            dx = abs(x - cx) - (half - r)
            dy = abs(y - cy) - (half - r)
            if dx > 0 and dy > 0 and (dx * dx + dy * dy) > r * r:
                row.append(CLEAR)
                continue
            col = NAVY
            ex, ey = (x - cx) / bx, (y - cy) / by
            d = ex * ex + ey * ey
            if d <= 1.0:
                col = TAN_DARK if d > 0.80 else TAN
                if abs(y - cy) <= seam_h and abs(x - cx) < bx * 0.86:
                    col = WHITE
                for lx in (-0.24, -0.08, 0.08, 0.24):
                    if abs((x - cx) - lx * bx) <= lace_w and abs(y - cy) <= by * 0.36:
                        col = WHITE
            row.append(col)
        rows.append(row)
    return rows


def _write_png(path, rows):
    n = len(rows)
    raw = bytearray()
    for row in rows:
        raw.append(0)  # filter type 0
        for px in row:
            raw += bytes(px)
    comp = zlib.compress(bytes(raw), 9)

    def chunk(tag, data):
        return (struct.pack(">I", len(data)) + tag + data +
                struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF))

    ihdr = struct.pack(">IIBBBBB", n, n, 8, 6, 0, 0, 0)  # 8-bit RGBA
    with open(path, "wb") as fh:
        fh.write(b"\x89PNG\r\n\x1a\n")
        fh.write(chunk(b"IHDR", ihdr))
        fh.write(chunk(b"IDAT", comp))
        fh.write(chunk(b"IEND", b""))


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    for size in (16, 32, 48, 128):
        _write_png(os.path.join(here, f"icon{size}.png"), _pixels(size))
        print(f"wrote icon{size}.png")


if __name__ == "__main__":
    main()
