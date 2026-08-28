"""Shared analytics terminology for consistent labels and tooltips.

Categories:
  VALUE    — BR Value, VOR, trade value, roster value
  MARKET   — ADP, market movement, trade frequency, market vs ADP
  PROJECTION — projected PPG, start score, projected wins, playoff simulation
  HISTORY  — historical hit rate, comps, historical trends, career outcomes
"""

from __future__ import annotations

# Category keys used in title attributes and helper text.
CATEGORY_VALUE = "VALUE"
CATEGORY_MARKET = "MARKET"
CATEGORY_PROJECTION = "PROJECTION"
CATEGORY_HISTORY = "HISTORY"

LABELS: dict[str, dict[str, str]] = {
    "br_value": {
        "label": "BR Value",
        "category": CATEGORY_VALUE,
        "tooltip": "Dynasty trade value from the BR model for this league format.",
    },
    "vor": {
        "label": "VOR",
        "category": CATEGORY_VALUE,
        "tooltip": "Value over replacement — points above a rosterable baseline at this position.",
    },
    "trade_value": {
        "label": "Trade value",
        "category": CATEGORY_VALUE,
        "tooltip": "Fair-trade value used in the calculator and trade database.",
    },
    "roster_value": {
        "label": "Roster value",
        "category": CATEGORY_VALUE,
        "tooltip": "Combined player and pick value on a team's roster.",
    },
    "adp": {
        "label": "ADP",
        "category": CATEGORY_MARKET,
        "tooltip": "Average draft position from the selected consensus source.",
    },
    "market_vs_adp": {
        "label": "Market vs ADP",
        "category": CATEGORY_MARKET,
        "tooltip": "Where independent market signals imply this player should be drafted relative to current ADP.",
    },
    "market_movement": {
        "label": "Market movement",
        "category": CATEGORY_MARKET,
        "tooltip": "Recent change in trade frequency or market price for this player.",
    },
    "trade_frequency": {
        "label": "Trade frequency",
        "category": CATEGORY_MARKET,
        "tooltip": "How often this player appears in real dynasty trades over a recent window.",
    },
    "projected_ppg": {
        "label": "Proj PPG",
        "category": CATEGORY_PROJECTION,
        "tooltip": "Projected fantasy points per game for the current scoring settings.",
    },
    "start_score": {
        "label": "Start score",
        "category": CATEGORY_PROJECTION,
        "tooltip": "Start/Sit recommendation strength for this week's lineup decision.",
    },
    "playoff_odds": {
        "label": "Playoff odds",
        "category": CATEGORY_PROJECTION,
        "tooltip": "Simulated chance to make the playoffs from current rosters and schedule.",
    },
    "historical_hit_rate": {
        "label": "Hist",
        "category": CATEGORY_HISTORY,
        "tooltip": "Historical top-12 chance for players with this career profile and situation. Not a ranking input.",
    },
    "historical_comps": {
        "label": "Players like this",
        "category": CATEGORY_HISTORY,
        "tooltip": "Historical outcomes for players with a similar career arc and usage profile.",
    },
    "adp_range_comps": {
        "label": "ADP range history",
        "category": CATEGORY_HISTORY,
        "tooltip": "Historical outcomes for players drafted in this ADP range, regardless of profile.",
    },
}


def label(key: str) -> str:
    return LABELS.get(key, {}).get("label", key)


def tooltip(key: str) -> str:
    return LABELS.get(key, {}).get("tooltip", "")


def title_attr(key: str) -> str:
    entry = LABELS.get(key, {})
    if not entry:
        return ""
    cat = entry.get("category", "")
    tip = entry.get("tooltip", "")
    if cat and tip:
        return f"{cat}: {tip}"
    return tip
