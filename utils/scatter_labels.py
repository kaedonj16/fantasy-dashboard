"""Pure helpers for Plotly scatter point labels.

Used by the graphs page PF-vs-PA scatters (season + career / tour mock) so
dense clusters offset or hide overlapping team names instead of stacking
every label at ``top center``.
"""
from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

# Prefer above-the-dot first; then sides; then below as last resorts.
_PLOTLY_ANCHORS: Tuple[str, ...] = (
    "top center",
    "top left",
    "top right",
    "middle left",
    "middle right",
    "bottom center",
    "bottom left",
    "bottom right",
)


def _norm_span(vals: Sequence[float]) -> Tuple[float, float, float]:
    lo = float(min(vals))
    hi = float(max(vals))
    span = hi - lo
    if span <= 1e-9:
        span = 1.0
    return lo, hi, span


def _anchor_box(nx: float, ny: float, tw: float, th: float, anchor: str) -> Tuple[float, float, float, float]:
    """Axis-aligned box in normalized space for a label at ``(nx, ny)``."""
    gap = 0.02
    if "left" in anchor:
        x1, x2 = nx - gap - tw, nx - gap
    elif "right" in anchor:
        x1, x2 = nx + gap, nx + gap + tw
    else:
        x1, x2 = nx - tw / 2.0, nx + tw / 2.0
    if anchor.startswith("top"):
        y1, y2 = ny + gap, ny + gap + th
    elif anchor.startswith("bottom"):
        y1, y2 = ny - gap - th, ny - gap
    else:
        y1, y2 = ny - th / 2.0, ny + th / 2.0
    return x1, y1, x2, y2


def _overlap(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> bool:
    return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])


def scatter_label_placements(
    xs: Sequence[float],
    ys: Sequence[float],
    labels: Sequence[str],
    *,
    char_width: float = 0.035,
    label_height: float = 0.07,
    hide_when_crowded: bool = True,
    x_range: Sequence[float] | None = None,
    y_range: Sequence[float] | None = None,
    axis_pad_frac: float = 0.12,
) -> List[Tuple[str, str]]:
    """Assign Plotly ``text`` / ``textposition`` pairs with collision avoidance.

    Points earlier in the sequences keep their labels preferentially (callers
    should put high-priority teams first — e.g. viewer, then power-rank order).
    When every anchor still overlaps an already-placed label, the text is
    cleared (hover/name still available on the marker) unless
    ``hide_when_crowded`` is False, in which case the least-crowded anchor is kept.

    Normalization uses the chart axis range when provided; otherwise the data
    span plus ``axis_pad_frac`` (matching graphs_page PF/PA padding) so a tight
    cluster stays dense in normalized space instead of being stretched to [0,1].

    Returns one ``(text, textposition)`` per input point.
    """
    n = len(labels)
    if n == 0:
        return []
    if not (len(xs) == len(ys) == n):
        raise ValueError("xs, ys, and labels must be the same length")

    xf = [float(v) for v in xs]
    yf = [float(v) for v in ys]

    if x_range is not None and len(x_range) >= 2:
        x_lo, x_hi = float(x_range[0]), float(x_range[1])
        x_span = max(x_hi - x_lo, 1e-9)
    else:
        x_lo, x_hi, raw = _norm_span(xf)
        pad = max(raw * axis_pad_frac, 1.0)
        x_lo -= pad
        x_span = (x_hi + pad) - x_lo

    if y_range is not None and len(y_range) >= 2:
        y_lo, y_hi = float(y_range[0]), float(y_range[1])
        y_span = max(y_hi - y_lo, 1e-9)
    else:
        y_lo, y_hi, raw = _norm_span(yf)
        pad = max(raw * axis_pad_frac, 1.0)
        y_lo -= pad
        y_span = (y_hi + pad) - y_lo

    placed: List[Tuple[float, float, float, float]] = []
    out: List[Tuple[str, str]] = []

    for i in range(n):
        label = str(labels[i] or "")
        if not label:
            out.append(("", "top center"))
            continue
        nx = (xf[i] - x_lo) / x_span
        ny = (yf[i] - y_lo) / y_span
        tw = max(char_width * 2.0, char_width * len(label))
        th = label_height

        chosen = None
        best = None
        best_hits = None
        for anchor in _PLOTLY_ANCHORS:
            box = _anchor_box(nx, ny, tw, th, anchor)
            hits = sum(1 for p in placed if _overlap(box, p))
            if hits == 0:
                chosen = (label, anchor, box)
                break
            if best_hits is None or hits < best_hits:
                best_hits = hits
                best = (label, anchor, box)

        if chosen is None:
            if hide_when_crowded or best is None:
                out.append(("", "top center"))
                continue
            chosen = best

        text, anchor, box = chosen
        placed.append(box)
        out.append((text, anchor))

    return out


def scatter_label_placements_from_rows(
    rows: Iterable[dict],
    *,
    x_key: str = "x",
    y_key: str = "y",
    label_key: str = "label",
    **kwargs,
) -> List[Tuple[str, str]]:
    """Convenience wrapper over dict rows (tour-mock / test fixtures)."""
    rows = list(rows)
    return scatter_label_placements(
        [r[x_key] for r in rows],
        [r[y_key] for r in rows],
        [r[label_key] for r in rows],
        **kwargs,
    )
