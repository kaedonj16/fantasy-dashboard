"""NFL team abbreviation helpers.

Extracted from app.py. The abbreviation->name map is hoisted to module level so
it is built once at import instead of on every call, and the mapping can be
unit-tested without the pandas/DB stack.
"""
from __future__ import annotations

# Abbreviation -> full team name. Includes both WAS and WSH spellings for
# Washington since upstream feeds disagree.
TEAM_FULL_NAMES = {
    "ARI": "Arizona Cardinals",
    "ATL": "Atlanta Falcons",
    "BAL": "Baltimore Ravens",
    "BUF": "Buffalo Bills",
    "CAR": "Carolina Panthers",
    "CHI": "Chicago Bears",
    "CIN": "Cincinnati Bengals",
    "CLE": "Cleveland Browns",
    "DAL": "Dallas Cowboys",
    "DEN": "Denver Broncos",
    "DET": "Detroit Lions",
    "GB": "Green Bay Packers",
    "HOU": "Houston Texans",
    "IND": "Indianapolis Colts",
    "JAX": "Jacksonville Jaguars",
    "KC": "Kansas City Chiefs",
    "LV": "Las Vegas Raiders",
    "LAC": "Los Angeles Chargers",
    "LAR": "Los Angeles Rams",
    "MIA": "Miami Dolphins",
    "MIN": "Minnesota Vikings",
    "NE": "New England Patriots",
    "NO": "New Orleans Saints",
    "NYG": "New York Giants",
    "NYJ": "New York Jets",
    "PHI": "Philadelphia Eagles",
    "PIT": "Pittsburgh Steelers",
    "SF": "San Francisco 49ers",
    "SEA": "Seattle Seahawks",
    "TB": "Tampa Bay Buccaneers",
    "TEN": "Tennessee Titans",
    "WAS": "Washington Commanders",
    "WSH": "Washington Commanders",
}


def get_team_full_name(abbreviation: str) -> str:
    """Map a team abbreviation to its full team name.

    Case-insensitive. Unknown abbreviations pass through unchanged.
    """
    return TEAM_FULL_NAMES.get(str(abbreviation).upper(), abbreviation)
