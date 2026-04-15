"""
Fetch, parse, and cache Jeff Sagarin college football predictor ratings.

Sagarin publishes annual CFB predictor ratings at sagarin.com. This module
scrapes those pages, normalises team names to match CFBD conventions, and
caches results to disk so the pipeline only hits the network once per year.

URL pattern:
    Current season : http://sagarin.com/sports/cfsend.htm
    Historical      : http://sagarin.com/sports/cf{yy}end.htm
                      e.g. cf20end.htm = 2020 season
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Dict, Optional

import requests

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_DATA_DIR = Path(__file__).parent.parent.parent / "data"
_BASE_URL  = "http://sagarin.com/sports/cfsend.htm"
_HIST_URL  = "http://sagarin.com/sports/cf{yy}end.htm"

# ---------------------------------------------------------------------------
# Team name normalisation aliases
# Sagarin uses shortened names; map them to CFBD full names (lower-cased,
# punctuation stripped) so lookups succeed.
# ---------------------------------------------------------------------------
ALIASES: Dict[str, str] = {
    # SEC
    "miss state":         "mississippi state",
    "ole miss":           "mississippi",
    "s carolina":         "south carolina",
    "n carolina":         "north carolina",
    "n carolina st":      "north carolina state",
    "nc state":           "north carolina state",
    "fla":                "florida",
    "fla state":          "florida state",
    "tenn":               "tennessee",
    "vandy":              "vanderbilt",
    "ga tech":            "georgia tech",
    "miami fl":           "miami",
    # Big Ten
    "ohio st":            "ohio state",
    "penn st":            "penn state",
    "michigan st":        "michigan state",
    "minn":               "minnesota",
    "nw":                 "northwestern",
    "neb":                "nebraska",
    "iowa st":            "iowa state",      # Big 12, but same pattern
    # Big 12
    "okla":               "oklahoma",
    "oklahoma st":        "oklahoma state",
    "kansas st":          "kansas state",
    "texas tech":         "texas tech",      # already fine, belt-and-suspenders
    "tcu":                "tcu",
    "wvu":                "west virginia",
    "baylor":             "baylor",
    # Pac-12
    "washington st":      "washington state",
    "oregon st":          "oregon state",
    "arizona st":         "arizona state",
    "cal":                "california",
    "usc":                "usc",
    "ucla":               "ucla",
    # ACC
    "bc":                 "boston college",
    "wake forest":        "wake forest",
    # G5
    "fla atlantic":       "florida atlantic",
    "middle tenn":        "middle tennessee",
    "s miss":             "southern miss",
    "la tech":            "louisiana tech",
    "la lafayette":       "louisiana",
    "unt":                "north texas",
    "utsa":               "utsa",
    "utep":               "utep",
    "fiu":                "fiu",
    "e carolina":         "east carolina",
    "c florida":          "ucf",
    "ucf":                "ucf",
    "cinn":               "cincinnati",
    "s florida":          "south florida",
    "colo st":            "colorado state",
    "boise st":           "boise state",
    "fresno st":          "fresno state",
    "s diego st":         "san diego state",
    "san jose st":        "san jose state",
    "nevada":             "nevada",
    "unlv":               "unlv",
    "hawaii":             "hawaii",
    "ball st":            "ball state",
    "cent mich":          "central michigan",
    "e mich":             "eastern michigan",
    "w mich":             "western michigan",
    "n ill":              "northern illinois",
    "ohio":               "ohio",
    "bowling grn":        "bowling green",
    "kent st":            "kent state",
    "buff":               "buffalo",
    "miami oh":           "miami (oh)",
    "akron":              "akron",
    "toledo":             "toledo",
    "app state":          "appalachian state",
    "app st":             "appalachian state",
    "troy":               "troy",
    "georgia st":         "georgia state",
    "georgia so":         "georgia southern",
    "la monroe":          "louisiana monroe",
    "ark state":          "arkansas state",
    "ark st":             "arkansas state",
    "s ala":              "south alabama",
    "tx state":           "texas state",
    "texas st":           "texas state",
    "w kentucky":         "western kentucky",
    "n texas":            "north texas",
    "old dominion":       "old dominion",
    "coastal car":        "coastal carolina",
    "coastal caro":       "coastal carolina",
}

# ---------------------------------------------------------------------------
# Module-level in-memory cache: {year: {normalised_team: rating}}
# ---------------------------------------------------------------------------
_CACHE: Dict[int, Dict[str, float]] = {}


def _url_for_year(year: int) -> str:
    """Return the Sagarin page URL for the given season year."""
    import datetime
    if year >= datetime.date.today().year:
        return _BASE_URL
    yy = str(year)[2:]   # 2020 → "20"
    return _HIST_URL.format(yy=yy)


def _normalise(name: str) -> str:
    """Lower-case, strip punctuation, apply ALIASES."""
    n = name.strip().lower()
    n = re.sub(r"[^a-z0-9 ]", "", n)
    n = re.sub(r"\s+", " ", n).strip()
    return ALIASES.get(n, n)


def _parse_ratings(html: str) -> Dict[str, float]:
    """
    Extract team→Predictor rating from Sagarin's <pre> text block.

    Each data row looks like:
        "    1  Alabama               =  98.12  12-  0  88.45  65.23"

    We capture:
        group(1) = team name (everything between rank and "=")
        group(2) = Predictor rating (first decimal number after "=")
    """
    ratings: Dict[str, float] = {}
    pattern = re.compile(
        r"^\s{0,6}\d{1,3}\s{1,4}(.+?)\s*=\s*([\d]+\.[\d]+)",
        re.MULTILINE,
    )
    for m in pattern.finditer(html):
        raw_name = m.group(1).strip()
        rating   = float(m.group(2))
        key      = _normalise(raw_name)
        if key:
            ratings[key] = rating
    return ratings


def fetch_ratings(year: int) -> Dict[str, float]:
    """
    Return {normalised_team_name: predictor_rating} for the given season year.

    Look-up order:
        1. Module-level in-memory cache (_CACHE)
        2. Disk cache  (data/sagarin_ratings_{year}.json)
        3. HTTP fetch  from sagarin.com, then write disk cache

    Raises requests.RequestException on network failure (caller should handle).
    """
    if year in _CACHE:
        return _CACHE[year]

    cache_path = _DATA_DIR / f"sagarin_ratings_{year}.json"
    if cache_path.exists():
        try:
            ratings = json.loads(cache_path.read_text())
            _CACHE[year] = ratings
            return ratings
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Sagarin disk cache corrupt for %d (%s) — re-fetching", year, exc)

    url = _url_for_year(year)
    logger.info("Fetching Sagarin ratings for %d from %s", year, url)
    resp = requests.get(url, timeout=20)
    resp.raise_for_status()

    ratings = _parse_ratings(resp.text)
    if not ratings:
        logger.warning("Sagarin parse returned 0 teams for year %d — check HTML format", year)

    try:
        _DATA_DIR.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(ratings, indent=2, sort_keys=True))
    except OSError as exc:
        logger.warning("Could not write Sagarin cache for %d: %s", year, exc)

    _CACHE[year] = ratings
    return ratings


def get_team_rating(team: str, year: int) -> Optional[float]:
    """
    Public API: look up Sagarin predictor rating for *team* in *year*.

    Returns None if the team is not found (FCS, non-D1, or name mismatch)
    or if the fetch fails.  Callers should treat None as the non-D1 floor.
    """
    if not team or not year:
        return None
    try:
        ratings = fetch_ratings(year)
    except Exception as exc:
        logger.warning("Sagarin fetch failed for year %d: %s", year, exc)
        return None
    return ratings.get(_normalise(team))
