"""Coin-style rank medals for leaderboards (standings, power rankings, awards).

A shared helper so every record/achievement board renders the same podium marks:
gold #1 (crowned), silver #2, bronze #3, and a plain numbered ring for the rest.
Deliberately NOT used on value boards — value fluctuates, so there's no meaningful
"first place" to crown.

The medals are self-contained inline SVG (their own <defs>), so callers just drop
the returned string into a cell — no page-level sprite or extra CSS required. IDs
are suffixed per metal, unique within a single board (one #1/#2/#3 apiece).
"""
from __future__ import annotations

# highlight → base → shade → dark-rim, plus the embossed-numeral ink.
_METALS = {
    "gold":   dict(face=("#fff7d6", "#f6d375", "#d9a531", "#a9781f"),
                   rim=("#ffe89a", "#c99a2e", "#7d5a12"), edge="#7d5a12", num="#8a6410"),
    "silver": dict(face=("#ffffff", "#dde3ea", "#a7b2be", "#727d89"),
                   rim=("#eef2f6", "#aab4bf", "#66707b"), edge="#66707b", num="#5c6670"),
    "bronze": dict(face=("#ffe6cc", "#e0a56a", "#bd7539", "#7f4a22"),
                   rim=("#f0c39a", "#b87a44", "#7a4620"), edge="#7a4620", num="#7a4620"),
}

_RANK_METAL = {1: "gold", 2: "silver", 3: "bronze"}

_CROWN = (
    '<path d="M-7 5 L7 5 L5.4 -3 L1.8 0.3 L0 -5 L-1.8 0.3 L-5.4 -3 Z" fill="url(#rmCrown)" '
    'stroke="#8a6410" stroke-width="0.6" stroke-linejoin="round"/>'
    '<circle cx="-7" cy="-3.4" r="1.4" fill="#f6d375" stroke="#8a6410" stroke-width="0.5"/>'
    '<circle cx="7" cy="-3.4" r="1.4" fill="#f6d375" stroke="#8a6410" stroke-width="0.5"/>'
    '<circle cx="0" cy="-6.4" r="1.5" fill="#fff7d6" stroke="#8a6410" stroke-width="0.5"/>'
)


def _medal_svg(metal: str, label: str, size: int, crown: bool) -> str:
    m = _METALS[metal]
    uid = metal  # unique per metal within a board
    face, rim = m["face"], m["rim"]
    crown_svg = f'<g transform="translate(32 7)">{_CROWN}</g>' if crown else ""
    return (
        f'<svg viewBox="0 0 64 66" width="{size}" height="{size * 66 // 64}" role="img" '
        f'aria-label="Rank {label}" style="overflow:visible;flex:none;'
        f'filter:drop-shadow(0 2px 3px rgba(0,0,0,.28))">'
        f'<defs>'
        f'<radialGradient id="rmFace-{uid}" cx="38%" cy="30%" r="78%">'
        f'<stop offset="0%" stop-color="{face[0]}"/><stop offset="42%" stop-color="{face[1]}"/>'
        f'<stop offset="80%" stop-color="{face[2]}"/><stop offset="100%" stop-color="{face[3]}"/>'
        f'</radialGradient>'
        f'<linearGradient id="rmRim-{uid}" x1="0" y1="0" x2="0" y2="1">'
        f'<stop offset="0%" stop-color="{rim[0]}"/><stop offset="52%" stop-color="{rim[1]}"/>'
        f'<stop offset="100%" stop-color="{rim[2]}"/></linearGradient>'
        f'<radialGradient id="rmCrown" cx="40%" cy="30%" r="80%"><stop offset="0%" stop-color="#fff7d6"/>'
        f'<stop offset="60%" stop-color="#f4cf6a"/><stop offset="100%" stop-color="#d59f2e"/></radialGradient>'
        f'</defs>'
        f'{crown_svg}'
        f'<circle cx="32" cy="32" r="22" fill="url(#rmRim-{uid})"/>'
        f'<circle cx="32" cy="32" r="20.5" fill="none" stroke="{m["edge"]}" stroke-width="2.4" '
        f'stroke-dasharray="1.7 2.2" opacity="0.45"/>'
        f'<circle cx="32" cy="32" r="16.5" fill="url(#rmFace-{uid})" stroke="{m["edge"]}" stroke-width="1"/>'
        f'<circle cx="32" cy="32" r="16.5" fill="none" stroke="#ffffff" stroke-width="1" opacity="0.30"/>'
        f'<path d="M20 24 A16 16 0 0 1 44 22" fill="none" stroke="#ffffff" stroke-width="2.4" '
        f'stroke-linecap="round" opacity="0.45"/>'
        f'<text x="32" y="33" text-anchor="middle" dominant-baseline="central" '
        f'style="font:800 18px system-ui,sans-serif" fill="#ffffff" opacity="0.5">{label}</text>'
        f'<text x="32" y="32" text-anchor="middle" dominant-baseline="central" '
        f'style="font:800 18px system-ui,sans-serif" fill="{m["num"]}">{label}</text>'
        f'</svg>'
    )


def _numbered_ring(label: str, size: int) -> str:
    d = size - 6
    return (
        f'<span style="width:{d}px;height:{d}px;border:1.5px solid var(--border);border-radius:50%;'
        f'display:inline-flex;align-items:center;justify-content:center;font-size:{max(11, d // 2)}px;'
        f'font-weight:800;color:var(--text-subtle);flex:none;font-variant-numeric:tabular-nums">{label}</span>'
    )


def rank_mark(rank, size: int = 36, wrap: bool = True, ring_others: bool = True) -> str:
    """Return a medal (ranks 1–3) or a fallback for a leaderboard row.

    `rank` may be an int or anything int-coercible; non-numeric ranks fall back to
    the original text. `wrap` centers the mark in a flex box so it sits nicely in a
    table cell or grid column. `ring_others` controls the 4+ fallback: a numbered
    ring (good for spacious card lists) when True, or a plain bold number (keeps
    dense tables from growing taller) when False.
    """
    try:
        r = int(rank)
        label = str(r)
    except (TypeError, ValueError):
        r, label = 0, str(rank)

    metal = _RANK_METAL.get(r)
    if metal:
        # #1 runs a touch larger and wears the crown.
        msize = size + 4 if r == 1 else size
        inner = _medal_svg(metal, label, msize, crown=(r == 1))
    elif ring_others:
        inner = _numbered_ring(label, size)
    else:
        inner = (f'<span style="font-weight:800;color:var(--text-muted);'
                 f'font-variant-numeric:tabular-nums">{label}</span>')

    if not wrap:
        return inner
    return (
        f'<span style="display:inline-flex;align-items:center;justify-content:center;'
        f'min-width:{size + 8}px">{inner}</span>'
    )
