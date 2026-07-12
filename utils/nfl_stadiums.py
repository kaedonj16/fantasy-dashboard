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

# team abbr -> stadium name, dome?, outdoor climate profile, and lat/lon (for
# weather lookups; domes carry coords too but weather is skipped for them).
STADIUMS: dict[str, dict] = {
    "ARI": {"name": "State Farm Stadium", "dome": True,  "climate": "dome", "lat": 33.5276, "lon": -112.2626},
    "ATL": {"name": "Mercedes-Benz Stadium", "dome": True, "climate": "dome", "lat": 33.7554, "lon": -84.4009},
    "BAL": {"name": "M&T Bank Stadium", "dome": False, "climate": "cold", "lat": 39.2780, "lon": -76.6227},
    "BUF": {"name": "Highmark Stadium", "dome": False, "climate": "cold", "lat": 42.7738, "lon": -78.7870},
    "CAR": {"name": "Bank of America Stadium", "dome": False, "climate": "mild", "lat": 35.2258, "lon": -80.8528},
    "CHI": {"name": "Soldier Field", "dome": False, "climate": "cold", "lat": 41.8623, "lon": -87.6167},
    "CIN": {"name": "Paycor Stadium", "dome": False, "climate": "cold", "lat": 39.0955, "lon": -84.5161},
    "CLE": {"name": "Huntington Bank Field", "dome": False, "climate": "cold", "lat": 41.5061, "lon": -81.6995},
    "DAL": {"name": "AT&T Stadium", "dome": True, "climate": "dome", "lat": 32.7473, "lon": -97.0945},
    "DEN": {"name": "Empower Field", "dome": False, "climate": "cold", "lat": 39.7439, "lon": -105.0201},
    "DET": {"name": "Ford Field", "dome": True, "climate": "dome", "lat": 42.3400, "lon": -83.0456},
    "GB":  {"name": "Lambeau Field", "dome": False, "climate": "cold", "lat": 44.5013, "lon": -88.0622},
    "HOU": {"name": "NRG Stadium", "dome": True, "climate": "dome", "lat": 29.6847, "lon": -95.4107},
    "IND": {"name": "Lucas Oil Stadium", "dome": True, "climate": "dome", "lat": 39.7601, "lon": -86.1639},
    "JAX": {"name": "EverBank Stadium", "dome": False, "climate": "warm", "lat": 30.3239, "lon": -81.6373},
    "KC":  {"name": "Arrowhead Stadium", "dome": False, "climate": "cold", "lat": 39.0489, "lon": -94.4839},
    "LV":  {"name": "Allegiant Stadium", "dome": True, "climate": "dome", "lat": 36.0909, "lon": -115.1833},
    "LAC": {"name": "SoFi Stadium", "dome": True, "climate": "dome", "lat": 33.9535, "lon": -118.3392},
    "LAR": {"name": "SoFi Stadium", "dome": True, "climate": "dome", "lat": 33.9535, "lon": -118.3392},
    "MIA": {"name": "Hard Rock Stadium", "dome": False, "climate": "warm", "lat": 25.9580, "lon": -80.2389},
    "MIN": {"name": "U.S. Bank Stadium", "dome": True, "climate": "dome", "lat": 44.9736, "lon": -93.2575},
    "NE":  {"name": "Gillette Stadium", "dome": False, "climate": "cold", "lat": 42.0909, "lon": -71.2643},
    "NO":  {"name": "Caesars Superdome", "dome": True, "climate": "dome", "lat": 29.9511, "lon": -90.0812},
    "NYG": {"name": "MetLife Stadium", "dome": False, "climate": "cold", "lat": 40.8135, "lon": -74.0745},
    "NYJ": {"name": "MetLife Stadium", "dome": False, "climate": "cold", "lat": 40.8135, "lon": -74.0745},
    "PHI": {"name": "Lincoln Financial Field", "dome": False, "climate": "cold", "lat": 39.9008, "lon": -75.1675},
    "PIT": {"name": "Acrisure Stadium", "dome": False, "climate": "cold", "lat": 40.4468, "lon": -80.0158},
    "SEA": {"name": "Lumen Field", "dome": False, "climate": "mild", "lat": 47.5952, "lon": -122.3316},
    "SF":  {"name": "Levi's Stadium", "dome": False, "climate": "mild", "lat": 37.4030, "lon": -121.9700},
    "TB":  {"name": "Raymond James Stadium", "dome": False, "climate": "warm", "lat": 27.9759, "lon": -82.5033},
    "TEN": {"name": "Nissan Stadium", "dome": False, "climate": "mild", "lat": 36.1665, "lon": -86.7713},
    "WAS": {"name": "Northwest Stadium", "dome": False, "climate": "cold", "lat": 38.9076, "lon": -76.8645},
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


def stadium_coords(team: str) -> Optional[tuple]:
    """(lat, lon) for a team's home stadium, or None for an unknown team."""
    st = STADIUMS.get(normalize_team(team))
    if not st:
        return None
    return (st["lat"], st["lon"])


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
