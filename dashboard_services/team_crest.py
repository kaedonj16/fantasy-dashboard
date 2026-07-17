"""Auto-generated hex team crests.

A fallback identity mark for teams with no profile picture — a hex shield in a
color derived from the team name, with a two-letter monogram taken from the
*team name* (not the username). Deliberately a fallback only: callers show the
real avatar when one exists and fall back to this otherwise, so a crest never
sits next to a photo.
"""
from __future__ import annotations

import html
import re
from urllib.parse import quote

# Mid-saturation hues that stay legible on both the light and dark card grounds.
# Index chosen by a stable hash of the name so a team always gets the same color.
_PALETTE = [
    "#2563eb",  # blue
    "#16a34a",  # green
    "#d97706",  # amber
    "#7c3aed",  # violet
    "#0891b2",  # cyan
    "#db2777",  # pink
    "#4f46e5",  # indigo
    "#ca8a04",  # gold
    "#dc2626",  # red
    "#0d9488",  # teal
]


def crest_initials(name: str) -> str:
    """Two-letter monogram from a team name, skipping symbol-only tokens.

    "Bijan Believers" -> "BB", "Nabers & Chill" -> "NC", "Super Cena -" -> "SC",
    "Juggernaut" -> "JU".
    """
    words = [w for w in re.split(r"\s+", (name or "").strip()) if w and w[0].isalnum()]
    if len(words) >= 2:
        return (words[0][0] + words[1][0]).upper()
    if words:
        return words[0][:2].upper()
    return "?"


def _color_for(name: str) -> str:
    h = 0
    for ch in (name or ""):
        h = (h * 31 + ord(ch)) & 0xFFFFFFFF
    return _PALETTE[h % len(_PALETTE)]


def _mix(hex_a: str, hex_b: str, t: float) -> str:
    """Blend hex_a toward hex_b by fraction t, returning a #rrggbb hex."""
    a = tuple(int(hex_a[i:i + 2], 16) for i in (1, 3, 5))
    b = tuple(int(hex_b[i:i + 2], 16) for i in (1, 3, 5))
    return "#" + "".join(f"{round(a[i] + (b[i] - a[i]) * t):02x}" for i in range(3))


def team_crest(name: str, size: int = 32) -> str:
    """Return an inline-SVG hex crest for `name`, sized to `size` px square.

    Self-contained (its own gradient def), so it drops into any avatar slot.
    """
    initials = html.escape(crest_initials(name))
    color = _color_for(name)
    uid = f"tc{abs(hash(name)) % 100000}"
    fs = max(9, int(size * 0.42))
    return (
        f'<svg width="{size}" height="{size}" viewBox="0 0 44 48" '
        f'style="flex:none;filter:drop-shadow(0 1px 2px rgba(0,0,0,.2))" role="img" '
        f'aria-label="{initials}">'
        f'<defs><linearGradient id="{uid}" x1="0" y1="0" x2="0" y2="1">'
        f'<stop offset="0%" stop-color="color-mix(in srgb, {color} 82%, #fff)"/>'
        f'<stop offset="100%" stop-color="{color}"/></linearGradient></defs>'
        f'<path d="M22 2 L40 12 V36 L22 46 L4 36 V12 Z" fill="url(#{uid})" '
        f'stroke="color-mix(in srgb, {color} 68%, #000)" stroke-width="1.2"/>'
        f'<path d="M22 2 L40 12 V36 L22 46 L4 36 V12 Z" fill="none" stroke="#fff" '
        f'stroke-width="1" opacity="0.22"/>'
        f'<text x="22" y="29" text-anchor="middle" fill="#fff" '
        f'style="font:800 {fs * 44 // size}px/1 system-ui,sans-serif">{initials}</text>'
        f'</svg>'
    )


def team_crest_data_uri(name: str) -> str:
    """A crest as a `data:` URI, for use as an <img src> in circular avatar slots.

    Drawn on a 48×48 square canvas with the hex centered and sized to fit inside
    the inscribed circle, so the app's `border-radius:50%` avatar frame clips only
    the transparent corners — the shield stays whole. Colors are pre-computed to
    plain hex (no color-mix/currentColor), since an SVG loaded as an image renders
    without the page's CSS context.
    """
    initials = html.escape(crest_initials(name))
    base = _color_for(name)
    top = _mix(base, "#ffffff", 0.20)
    edge = _mix(base, "#000000", 0.35)
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" width="48" height="48" viewBox="0 0 48 48">'
        f'<defs><linearGradient id="g" x1="0" y1="0" x2="0" y2="1">'
        f'<stop offset="0" stop-color="{top}"/><stop offset="1" stop-color="{base}"/>'
        '</linearGradient></defs>'
        f'<path d="M24 3 L42 13 V35 L24 45 L6 35 V13 Z" fill="url(#g)" stroke="{edge}" stroke-width="1.4"/>'
        f'<text x="24" y="30" text-anchor="middle" fill="#ffffff" '
        'font-family="system-ui,-apple-system,sans-serif" font-weight="800" font-size="17">'
        f'{initials}</text>'
        '</svg>'
    )
    return "data:image/svg+xml," + quote(svg, safe="")
