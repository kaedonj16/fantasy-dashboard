"""Brand styling for Plotly figures.

Default Plotly (white paper, its own font, heavy gridlines) reads "data tool"
and breaks dark mode (white chart blocks on dark cards). This applies the site
look to every server-built figure: the brand font, transparent backgrounds so
the chart adopts the card surface in both themes, and quiet neutral gridlines
that work on light and dark.

Client-built charts (team modal, radar, history) apply the same values via the
mirrored constants in static/app.js (window.brandPlotlyLayout).
"""
from __future__ import annotations

BRAND_FONT = "InterVariable, Inter, system-ui, -apple-system, sans-serif"
# Mid-grey that stays legible on white cards and on the dark navy card.
AXIS_TEXT = "#7c8798"
GRID = "rgba(148, 163, 184, 0.22)"
AXIS_LINE = "rgba(148, 163, 184, 0.38)"


def apply_brand_layout(fig):
    """Apply brand font / transparent surfaces / quiet grid to a go.Figure.
    Called after the figure's own update_layout so axis titles etc. survive."""
    fig.update_layout(
        font=dict(family=BRAND_FONT, size=12.5, color=AXIS_TEXT),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        hoverlabel=dict(font=dict(family=BRAND_FONT, size=12)),
    )
    fig.update_xaxes(gridcolor=GRID, zerolinecolor=AXIS_LINE, linecolor=AXIS_LINE, tickcolor=AXIS_LINE)
    fig.update_yaxes(gridcolor=GRID, zerolinecolor=AXIS_LINE, linecolor=AXIS_LINE, tickcolor=AXIS_LINE)
    return fig
