"""Static NFL stadium / game-environment metadata.

Fantasy output is meaningfully shaped by where a game is played: indoor
(dome or fixed/closed roof) games are weather-proof and slightly friendlier to
passing and kicking, while late-season games at cold-weather outdoor sites carry
real downside for passers, receivers, and kickers. Those are *structural* facts
about the venue - they need no live feed, so this module ships them offline.

We deliberately do NOT fabricate live weather, wind, or betting totals here: a
"Dome" / "Cold" tag we can always stand behind beats a wind speed we'd have to
guess. If a real weather or odds source is wired up later, ``game_environment``
is the single place to enrich the returned tag.

``dome`` covers true domes and retractable/fixed roofs that play climate-
controlled in practice (ATL, DAL, HOU, ARI, IND, plus SoFi's fixed canopy).
``climate`` is the outdoor-weather profile used for the late-season cold flag:
"dome" (n/a), "cold", "mild", or "warm".
"""
from __future__ import annotations

from typing import Optional

# team abbr -> (stadium name, dome?, outdoor climate profile)
STADIUMS: dict[str, dict] = {
    "ARI": {"name": "State Farm Stadium", "dome": True,  "climate": "dome"},
    "ATL": {"name": "Mercedes-Benz Stadium", "dome": True, "climate": "dome"},
    "BAL": {"name": "M&T Bank Stadium", "dome": False, "climate": "cold"},
    "BUF": {"name": "Highmark Stadium", "dome": False, "climate": "cold"},
    "CAR": {"name": "Bank of America Stadium", "dome": False, "climate": "mild"},
    "CHI": {"name": "Soldier Field", "dome": False, "climate": "cold"},
    "CIN": {"name": "Paycor Stadium", "dome": False, "climate": "cold"},
    "CLE": {"name": "Huntington Bank Field", "dome": False, "climate": "cold"},
    "DAL": {"name": "AT&T Stadium", "dome": True, "climate": "dome"},
    "DEN": {"name": "Empower Field", "dome": False, "climate": "cold"},
    "DET": {"name": "Ford Field", "dome": True, "climate": "dome"},
    "GB":  {"name": "Lambeau Field", "dome": False, "climate": "cold"},
    "HOU": {"name": "NRG Stadium", "dome": True, "climate": "dome"},
    "IND": {"name": "Lucas Oil Stadium", "dome": True, "climate": "dome"},
    "JAX": {"name": "EverBank Stadium", "dome": False, "climate": "warm"},
    "KC":  {"name": "Arrowhead Stadium", "dome": False, "climate": "cold"},
    "LV":  {"name": "Allegiant Stadium", "dome": True, "climate": "dome"},
    "LAC": {"name": "SoFi Stadium", "dome": True, "climate": "dome"},
    "LAR": {"name": "SoFi Stadium", "dome": True, "climate": "dome"},
    "MIA": {"name": "Hard Rock Stadium", "dome": False, "climate": "warm"},
    "MIN": {"name": "U.S. Bank Stadium", "dome": True, "climate": "dome"},
    "NE":  {"name": "Gillette Stadium", "dome": False, "climate": "cold"},
    "NO":  {"name": "Caesars Superdome", "dome": True, "climate": "dome"},
    "NYG": {"name": "MetLife Stadium", "dome": False, "climate": "cold"},
    "NYJ": {"name": "MetLife Stadium", "dome": False, "climate": "cold"},
    "PHI": {"name": "Lincoln Financial Field", "dome": False, "climate": "cold"},
    "PIT": {"name": "Acrisure Stadium", "dome": False, "climate": "cold"},
    "SEA": {"name": "Lumen Field", "dome": False, "climate": "mild"},
    "SF":  {"name": "Levi's Stadium", "dome": False, "climate": "mild"},
    "TB":  {"name": "Raymond James Stadium", "dome": False, "climate": "warm"},
    "TEN": {"name": "Nissan Stadium", "dome": False, "climate": "mild"},
    "WAS": {"name": "Northwest Stadium", "dome": False, "climate": "cold"},
}

# Common alternate abbreviations seen across Sleeper / Tank01 / ESPN feeds.
ALIASES: dict[str, str] = {
    "JAC": "JAX", "LA": "LAR", "STL": "LAR", "SD": "LAC", "OAK": "LV",
    "WSH": "WAS", "WFT": "WAS", "LVR": "LV", "SFO": "SF", "TAM": "TB",
    "GNB": "GB", "KAN": "KC", "NWE": "NE", "NOR": "NO",
}

# NFL weeks from ~mid-December on, when cold-weather sites actually play cold.
_COLD_WEEK_START = 14


def normalize_team(team: str) -> str:
    """Uppercase and de-alias an NFL team abbreviation."""
    t = str(team or "").strip().upper()
    return ALIASES.get(t, t)


def game_environment(home_team: str, week: Optional[int] = None) -> Optional[dict]:
    """Environment tag for a game hosted by ``home_team``.

    Returns ``None`` for unknown teams (e.g. a bye or bad abbr). Otherwise a
    dict: ``env`` ("dome"/"outdoor"), ``label``, ``dome`` (bool), ``cold``
    (bool - only for cold-climate outdoor sites in the late-season window),
    ``stadium``, and a short human ``note``. Warm/mild outdoor games return a
    tag with no ``cold`` flag so the UI can leave them unmarked.
    """
    st = STADIUMS.get(normalize_team(home_team))
    if not st:
        return None
    if st["dome"]:
        return {
            "env": "dome", "label": "Dome", "dome": True, "cold": False,
            "stadium": st["name"], "note": "Indoor - weather-proof",
        }
    cold = st["climate"] == "cold" and week is not None and int(week) >= _COLD_WEEK_START
    return {
        "env": "outdoor", "label": "Outdoor", "dome": False, "cold": bool(cold),
        "stadium": st["name"],
        "note": "Cold-weather site, late season" if cold else "Outdoor",
    }
