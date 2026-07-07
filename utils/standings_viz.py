"""Server-rendered SVG scatter charts for league-wide team views.

Two league scatters shared by the Graphs page and the team modal's Graphs tab
(both call the same functions so the two surfaces can't drift):

  - luck_quadrant_svg: all-play win rate (true strength, x) vs actual win rate
    (y), with a dashed y=x "deserved" diagonal. Above the line = lucky, below =
    unlucky. Position is the signal; an amber/blue point tint (CVD-safe) only
    reinforces it.
  - value_age_svg: average roster age (x) vs total dynasty value (y), split into
    quadrants by the league medians (young+loaded / win-now / rebuilding / aging).

Pure string builders (no app/pandas imports) so they unit-test cleanly and run
identically server-side on either surface. Colors use CSS custom properties with
literal fallbacks so they adapt to light/dark themes.
"""
from __future__ import annotations

import html
from typing import List, Tuple

# CVD-safe reinforcement colors (position is the primary encoding).
_LUCK = "#f59e0b"    # amber: luckier than scoring earned
_UNLUCK = "#3b82f6"  # blue: unluckier than scoring earned
_NEU = "#94a3b8"     # gray: within a game of deserved
_ACCENT = "#6366f1"  # indigo: neutral team marker for value/age


def _esc(s) -> str:
    return html.escape(str(s))


def luck_quadrant_svg(analysis: dict, viewer_owner: str = "") -> str:
    """SVG scatter of actual win% (y) vs all-play win% (x) with a 'deserved'
    diagonal. Returns '' when fewer than 3 teams have played a game."""
    rows: List[Tuple[str, dict]] = [
        (o, a) for o, a in (analysis or {}).items()
        if a.get("games") and a["games"] > 0
    ]
    if len(rows) < 3:
        return ""

    W, H, pad = 460, 340, 44
    x0, y0, x1, y1 = pad, 16, W - 16, H - pad

    def sx(pct):  # all-play % -> x
        return x0 + pct * (x1 - x0)

    def sy(pct):  # actual % -> y (inverted)
        return y1 - pct * (y1 - y0)

    parts = [
        f'<svg viewBox="0 0 {W} {H}" class="luck-quadrant" role="img" '
        f'aria-label="Performance vs luck: each team plotted by all-play win rate against actual win rate">'
    ]
    parts.append(
        f'<rect x="{x0}" y="{y0}" width="{x1-x0}" height="{y1-y0}" fill="none" '
        f'stroke="var(--border,#e2e8f0)" stroke-width="1"/>'
    )
    for g in (0.25, 0.5, 0.75):
        parts.append(f'<line x1="{sx(g):.1f}" y1="{y0}" x2="{sx(g):.1f}" y2="{y1}" stroke="var(--border,#e2e8f0)" stroke-width="0.5" opacity="0.5"/>')
        parts.append(f'<line x1="{x0}" y1="{sy(g):.1f}" x2="{x1}" y2="{sy(g):.1f}" stroke="var(--border,#e2e8f0)" stroke-width="0.5" opacity="0.5"/>')
    # "Deserved" diagonal (y = x).
    parts.append(f'<line x1="{sx(0):.1f}" y1="{sy(0):.1f}" x2="{sx(1):.1f}" y2="{sy(1):.1f}" stroke="var(--text-muted,#94a3b8)" stroke-width="1.5" stroke-dasharray="5 4"/>')
    parts.append(f'<text x="{x0+8}" y="{y0+16}" font-size="11" font-weight="700" fill="{_LUCK}">Lucky</text>')
    parts.append(f'<text x="{x1-8}" y="{y1-8}" font-size="11" font-weight="700" fill="{_UNLUCK}" text-anchor="end">Unlucky</text>')
    parts.append(f'<text x="{(x0+x1)/2:.0f}" y="{H-8}" font-size="11" fill="var(--text-muted,#94a3b8)" text-anchor="middle">All-play win rate (true strength) &rarr;</text>')
    parts.append(f'<text x="14" y="{(y0+y1)/2:.0f}" font-size="11" fill="var(--text-muted,#94a3b8)" text-anchor="middle" transform="rotate(-90 14 {(y0+y1)/2:.0f})">Actual win rate &rarr;</text>')

    for owner, a in sorted(rows, key=lambda r: r[1]["all_play_pct"]):
        ax = sx(a["all_play_pct"])
        ay = sy(a["actual_wins"] / a["games"])
        delta = a.get("luck_delta", 0)
        col = _LUCK if delta >= 1 else (_UNLUCK if delta <= -1 else _NEU)
        is_me = viewer_owner and str(owner) == str(viewer_owner)
        r = 7 if is_me else 5
        ring = ' stroke="var(--text,#0f172a)" stroke-width="2"' if is_me else ' stroke="#fff" stroke-width="1"'
        short = _esc(str(owner)[:12])
        sign = "+" if delta > 0 else ""
        parts.append(
            f'<g><title>{_esc(owner)}: {a["actual_wins"]:.0f} actual wins vs '
            f'{a["expected_wins"]:.1f} expected ({sign}{delta:.1f})</title>'
            f'<circle cx="{ax:.1f}" cy="{ay:.1f}" r="{r}" fill="{col}"{ring}/>'
        )
        if ax > x1 - 70:
            parts.append(f'<text x="{ax-r-3:.1f}" y="{ay+3:.1f}" font-size="9.5" fill="var(--text,#334155)" text-anchor="end">{short}</text></g>')
        else:
            parts.append(f'<text x="{ax+r+3:.1f}" y="{ay+3:.1f}" font-size="9.5" fill="var(--text,#334155)">{short}</text></g>')

    parts.append("</svg>")
    return "".join(parts)


def _median(vals: List[float]) -> float:
    s = sorted(vals)
    n = len(s)
    if n == 0:
        return 0.0
    mid = n // 2
    return s[mid] if n % 2 else (s[mid - 1] + s[mid]) / 2.0


def value_age_svg(rows: List[dict], viewer_owner: str = "") -> str:
    """SVG scatter of total dynasty value (y) vs average roster age (x), split
    into quadrants by the league medians. Younger + more valuable (top-left) is
    the ascending-dynasty corner. Returns '' with fewer than 3 valued teams."""
    pts = [
        r for r in (rows or [])
        if (r.get("total_value") or 0) > 0 and (r.get("avg_age") or 0) > 0
    ]
    if len(pts) < 3:
        return ""

    ages = [float(r["avg_age"]) for r in pts]
    vals = [float(r["total_value"]) for r in pts]
    a_min, a_max = min(ages), max(ages)
    v_min, v_max = min(vals), max(vals)
    # Pad the ranges a touch so edge points aren't glued to the frame.
    a_pad = max((a_max - a_min) * 0.12, 0.5)
    v_pad = max((v_max - v_min) * 0.12, 1.0)
    a_lo, a_hi = a_min - a_pad, a_max + a_pad
    v_lo, v_hi = v_min - v_pad, v_max + v_pad
    a_med, v_med = _median(ages), _median(vals)

    W, H, pad = 460, 340, 46
    x0, y0, x1, y1 = pad, 16, W - 16, H - pad

    def sx(age):
        return x0 + (age - a_lo) / max(a_hi - a_lo, 1e-9) * (x1 - x0)

    def sy(val):
        return y1 - (val - v_lo) / max(v_hi - v_lo, 1e-9) * (y1 - y0)

    parts = [
        f'<svg viewBox="0 0 {W} {H}" class="luck-quadrant" role="img" '
        f'aria-label="Dynasty value versus average roster age for each team">'
    ]
    parts.append(
        f'<rect x="{x0}" y="{y0}" width="{x1-x0}" height="{y1-y0}" fill="none" '
        f'stroke="var(--border,#e2e8f0)" stroke-width="1"/>'
    )
    # Median split lines make the four quadrants.
    mx, my = sx(a_med), sy(v_med)
    parts.append(f'<line x1="{mx:.1f}" y1="{y0}" x2="{mx:.1f}" y2="{y1}" stroke="var(--text-muted,#94a3b8)" stroke-width="1" stroke-dasharray="4 4" opacity="0.7"/>')
    parts.append(f'<line x1="{x0}" y1="{my:.1f}" x2="{x1}" y2="{my:.1f}" stroke="var(--text-muted,#94a3b8)" stroke-width="1" stroke-dasharray="4 4" opacity="0.7"/>')
    # Corner labels (younger is left, more valuable is up).
    parts.append(f'<text x="{x0+8}" y="{y0+15}" font-size="10" font-weight="700" fill="var(--text-muted,#94a3b8)">Young &amp; loaded</text>')
    parts.append(f'<text x="{x1-8}" y="{y0+15}" font-size="10" font-weight="700" fill="var(--text-muted,#94a3b8)" text-anchor="end">Win-now</text>')
    parts.append(f'<text x="{x0+8}" y="{y1-8}" font-size="10" font-weight="700" fill="var(--text-muted,#94a3b8)">Rebuilding</text>')
    parts.append(f'<text x="{x1-8}" y="{y1-8}" font-size="10" font-weight="700" fill="var(--text-muted,#94a3b8)" text-anchor="end">Aging out</text>')
    # Axis titles.
    parts.append(f'<text x="{(x0+x1)/2:.0f}" y="{H-8}" font-size="11" fill="var(--text-muted,#94a3b8)" text-anchor="middle">Average roster age &rarr;</text>')
    parts.append(f'<text x="14" y="{(y0+y1)/2:.0f}" font-size="11" fill="var(--text-muted,#94a3b8)" text-anchor="middle" transform="rotate(-90 14 {(y0+y1)/2:.0f})">Total dynasty value &rarr;</text>')

    for r in sorted(pts, key=lambda r: r["total_value"]):
        owner = r["owner"]
        px, py = sx(float(r["avg_age"])), sy(float(r["total_value"]))
        is_me = viewer_owner and str(owner) == str(viewer_owner)
        rad = 7 if is_me else 5
        col = _LUCK if is_me else _ACCENT
        ring = ' stroke="var(--text,#0f172a)" stroke-width="2"' if is_me else ' stroke="#fff" stroke-width="1"'
        short = _esc(str(owner)[:12])
        parts.append(
            f'<g><title>{_esc(owner)}: {r["total_value"]:.0f} total value, '
            f'{r["avg_age"]:.1f} avg age</title>'
            f'<circle cx="{px:.1f}" cy="{py:.1f}" r="{rad}" fill="{col}"{ring}/>'
        )
        if px > x1 - 70:
            parts.append(f'<text x="{px-rad-3:.1f}" y="{py+3:.1f}" font-size="9.5" fill="var(--text,#334155)" text-anchor="end">{short}</text></g>')
        else:
            parts.append(f'<text x="{px+rad+3:.1f}" y="{py+3:.1f}" font-size="9.5" fill="var(--text,#334155)">{short}</text></g>')

    parts.append("</svg>")
    return "".join(parts)
