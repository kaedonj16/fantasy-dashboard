"""
Robust ESPN college football player age / DOB scraper.

Scraping strategy (tried in order for each player):
  1. ESPN search API  → find athlete ID (handles name variants, suffixes)
  2. ESPN athlete API → full profile JSON including dateOfBirth
  3. ESPN player HTML → BeautifulSoup scrape of bio section + embedded JSON
  4. Return None with low confidence if all three fail

Public API
----------
  search_player_on_espn(name, team=None, position=None)
      → (espn_id: str, confidence: float)

  scrape_player_profile(player_id)
      → {"dob": "YYYY-MM-DD", "age": float, "team": str, "position": str,
         "espn_id": str, "url": str, "source": str, "height": str, "weight": str}

  parse_dob_and_calculate_age(dob_str, ref_date=None)
      → (dob_iso: str, age: float)

  get_player_age(name, team=None, position=None, draft_year=None)
      → {"player_name", "espn_id", "team", "position",
         "dob", "age", "source", "url", "confidence"}

  fetch_espn_ages_robust(names, draft_year, prospects_meta=None, delay=0.3)
      → {name_lower: age_float}  (drop-in replacement for fetch_espn_ages)
"""
from __future__ import annotations

import json
import re
import time
from datetime import date, datetime
from typing import Any, Dict, List, Optional, Tuple

import requests

try:
    from bs4 import BeautifulSoup
    _BS4 = True
except ImportError:
    _BS4 = False

# ─────────────────────────────────────────────────────────────────────────────
# ESPN endpoints
# ─────────────────────────────────────────────────────────────────────────────

_SEARCH_URL    = "https://site.api.espn.com/apis/common/v3/search"
_ATHLETE_URL   = "https://site.api.espn.com/apis/site/v2/sports/football/college-football/athletes/{id}"
_ATHLETE_CORE  = "https://sports.core.api.espn.com/v2/sports/football/leagues/college-football/athletes/{id}"
_PLAYER_PAGE   = "https://www.espn.com/college-football/player/_/id/{id}"

_BROWSER_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)

_JSON_HEADERS = {
    "User-Agent":      _BROWSER_UA,
    "Accept":          "application/json",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer":         "https://www.espn.com/",
}
_HTML_HEADERS = {**_JSON_HEADERS, "Accept": "text/html,application/xhtml+xml,*/*;q=0.8"}

# ─────────────────────────────────────────────────────────────────────────────
# In-process cache  {name_lower: result_dict}  avoids duplicate requests
# ─────────────────────────────────────────────────────────────────────────────
_CACHE: Dict[str, Dict[str, Any]] = {}

# ─────────────────────────────────────────────────────────────────────────────
# Name normalisation helpers
# ─────────────────────────────────────────────────────────────────────────────

_SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}


def _norm_name(n: str) -> str:
    """Lowercase, strip suffixes, collapse whitespace, remove punctuation."""
    n = n.lower().strip()
    n = re.sub(r"\.", "", n)               # D.J. → DJ
    n = re.sub(r"['\-\u2019]", " ", n)    # apostrophe/hyphen → space
    n = re.sub(r"\s+", " ", n).strip()
    parts = [p for p in n.split() if p not in _SUFFIXES]
    return " ".join(parts)


def _name_score(item_name: str, query_name: str) -> float:
    """
    0.0 → no match at all
    0.5 → last-name-only match (weak)
    0.8 → full normalized match
    1.0 → exact full-name match (including suffixes)
    """
    if item_name.lower().strip() == query_name.lower().strip():
        return 1.0
    ni = _norm_name(item_name)
    nq = _norm_name(query_name)
    if ni == nq:
        return 0.8
    # Last-name match as a weak fallback
    ni_last = ni.split()[-1] if ni.split() else ""
    nq_last = nq.split()[-1] if nq.split() else ""
    if ni_last and nq_last and ni_last == nq_last:
        return 0.5
    return 0.0


def _team_score(item_team: str, query_team: str) -> float:
    """1.0 if the query team name appears anywhere in the item team string."""
    if not query_team:
        return 0.5  # neutral when we don't know team
    qt = query_team.lower().strip()
    it = item_team.lower().strip()
    if not it:
        return 0.0  # no team in result — can't confirm
    if qt in it or it in qt:
        return 1.0
    # Partial: last word of query in item (e.g., "Buffaloes" in "Colorado Buffaloes")
    qt_parts = qt.split()
    if qt_parts and qt_parts[-1] in it:
        return 0.7
    if qt_parts and qt_parts[0] in it:
        return 0.6
    return 0.0


# ─────────────────────────────────────────────────────────────────────────────
# HTTP helper — retries on transient errors, immediate fail on 4xx client errors
# ─────────────────────────────────────────────────────────────────────────────

def _get(url: str, params: Optional[Dict] = None, headers: Optional[Dict] = None,
         retries: int = 3, timeout: int = 12) -> Optional[requests.Response]:
    """GET with exponential backoff.  Returns None on permanent failure."""
    h = headers or _JSON_HEADERS
    for attempt in range(retries):
        try:
            resp = requests.get(url, params=params, headers=h, timeout=timeout)
            if resp.status_code in (400, 401, 403, 404):
                return None  # permanent client error — don't retry
            if resp.status_code == 429:
                time.sleep(4.0 * (attempt + 1))
                continue
            resp.raise_for_status()
            return resp
        except requests.Timeout:
            time.sleep(1.5 ** attempt)
        except requests.exceptions.ProxyError:
            # Proxy blocking the request (restricted network environment)
            return None
        except requests.RequestException:
            time.sleep(1.5 ** attempt)
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Part 3 — DOB parsing + age calculation
# ─────────────────────────────────────────────────────────────────────────────

# Common date formats ESPN and its APIs return
_DOB_FORMATS = [
    "%Y-%m-%dT%H:%M:%SZ",   # 2004-08-26T07:00:00Z
    "%Y-%m-%dT%H:%M:%S",    # 2004-08-26T07:00:00
    "%Y-%m-%dT%H:%MZ",      # 2004-08-26T07:00Z
    "%Y-%m-%d",             # 2004-08-26
    "%m/%d/%Y",             # 08/26/2004
    "%B %d, %Y",            # August 26, 2004
    "%b %d, %Y",            # Aug 26, 2004
]


def parse_dob_and_calculate_age(
    dob_str: str,
    ref_date: Optional[date] = None,
) -> Tuple[Optional[str], Optional[float]]:
    """
    Parse a DOB string into ISO format and calculate fractional age at ref_date.

    Args:
        dob_str:  raw DOB string from ESPN (various formats)
        ref_date: reference date for age calculation (default: today)

    Returns:
        (dob_iso, age_float) — either may be None on parse failure.
    """
    if not dob_str:
        return None, None

    raw = dob_str.strip()
    # Remove timezone offset if present (keep just the date+time)
    raw_clean = re.sub(r"\+\d{2}:\d{2}$", "", raw).strip()

    dob: Optional[date] = None
    for fmt in _DOB_FORMATS:
        try:
            dob = datetime.strptime(raw_clean, fmt).date()
            break
        except ValueError:
            # Also try the first N chars to handle trailing timezone/garbage
            try:
                # Only useful for short fixed-length formats (ISO variants)
                if len(raw_clean) > 10 and "%" not in fmt.replace("%Y", "").replace("%m", "").replace("%d", ""):
                    dob = datetime.strptime(raw_clean[:10], "%Y-%m-%d").date()
                    break
            except ValueError:
                pass

    if dob is None:
        # Last-resort: look for 4-digit year + 1-2 digit month/day
        m = re.search(r"(\d{4})-(\d{1,2})-(\d{1,2})", raw)
        if m:
            try:
                dob = date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
            except ValueError:
                pass

    if dob is None:
        return None, None

    ref = ref_date or date.today()
    age = round((ref - dob).days / 365.25, 2)
    return dob.isoformat(), age


# ─────────────────────────────────────────────────────────────────────────────
# Part 1 — player search
# ─────────────────────────────────────────────────────────────────────────────

def _parse_search_items(data: Dict) -> List[Dict]:
    """Flatten both ESPN search response shapes into a list of candidate items."""
    # Shape A: {"items": [...]}
    items = data.get("items") or []
    if items:
        return list(items)
    # Shape B: {"results": [{"contents": [...]}]}
    flat: List[Dict] = []
    for result in (data.get("results") or []):
        flat.extend(result.get("contents") or [])
    return flat


def search_player_on_espn(
    name: str,
    team:     Optional[str] = None,
    position: Optional[str] = None,
) -> Tuple[Optional[str], float, Optional[Dict]]:
    """
    Search ESPN for a college football player.

    Returns:
        (espn_id, confidence, best_item_dict)
        confidence: 0.0–1.0  (combination of name + team match quality)
        espn_id is None when no viable match found.
    """
    # Try multiple search strategies in order
    search_configs = [
        # (sport, type_param)
        ("college-football", "athlete"),
        ("college-football", "player"),
        ("college-football", None),       # no type filter
        ("football",         "athlete"),  # covers NFL too — useful for drafted players
    ]

    best_id:         Optional[str]  = None
    best_conf:       float          = 0.0
    best_item:       Optional[Dict] = None
    dob_in_search:   Optional[str]  = None

    for sport, type_param in search_configs:
        params: Dict[str, Any] = {
            "query": name,
            "sport": sport,
            "limit": "10",
        }
        if type_param:
            params["type"] = type_param

        resp = _get(_SEARCH_URL, params=params, headers=_JSON_HEADERS)
        if resp is None:
            continue

        try:
            data = resp.json()
        except ValueError:
            continue

        for item in _parse_search_items(data):
            # Accept items with these type values (ESPN uses both)
            item_type = (item.get("type") or "").lower()
            if item_type not in ("athlete", "player", ""):
                if item_type and item_type not in ("athlete", "player"):
                    # Skip non-player items (news, teams, etc.)
                    if item_type in ("story", "video", "team", "event", "league"):
                        continue

            display = item.get("displayName") or item.get("name") or ""
            if not display:
                continue

            ns = _name_score(display, name)
            if ns < 0.5:
                continue  # not even a last-name match — skip

            # Determine team string from various possible locations in item
            item_team = (
                (item.get("team") or {}).get("displayName") or
                (item.get("team") or {}).get("location") or
                item.get("teamName") or
                ""
            )
            ts = _team_score(item_team, team or "")

            # Prefer college football entries over NFL
            league_name = (
                (item.get("league") or {}).get("name") or
                item.get("leagueName") or ""
            ).lower()
            league_bonus = 0.1 if "college" in league_name else 0.0

            # DOB in search result = high signal of a good profile
            has_dob = bool(item.get("dateOfBirth"))
            dob_bonus = 0.05 if has_dob else 0.0

            conf = ns * 0.55 + ts * 0.30 + league_bonus + dob_bonus
            conf = min(conf, 1.0)

            if conf > best_conf:
                best_conf  = conf
                best_id    = str(item.get("id") or item.get("uid") or "")
                best_item  = item
                if has_dob:
                    dob_in_search = item["dateOfBirth"]

        # If we already found a high-confidence CFB match with a DOB, stop searching
        if best_conf >= 0.75 and dob_in_search:
            break

    # Strip non-numeric prefixes from ESPN UIDs ("s:20~l:23~a:4426354" → "4426354")
    if best_id and "~a:" in best_id:
        best_id = best_id.split("~a:")[-1]

    return best_id, best_conf, best_item


# ─────────────────────────────────────────────────────────────────────────────
# Part 2 — player profile data (athlete API + HTML fallback)
# ─────────────────────────────────────────────────────────────────────────────

def _extract_dob_from_athlete_api(player_id: str) -> Optional[str]:
    """
    Call the ESPN athlete detail API and return the raw dateOfBirth string.
    Tries both the site API and the core API.
    """
    for url_template in (_ATHLETE_URL, _ATHLETE_CORE):
        url  = url_template.format(id=player_id)
        resp = _get(url, headers=_JSON_HEADERS)
        if resp is None:
            continue
        try:
            data = resp.json()
        except ValueError:
            continue

        # Site API: {"athlete": {..., "dateOfBirth": "..."}}
        athlete = data.get("athlete") or data
        dob = (
            athlete.get("dateOfBirth") or
            athlete.get("dateOfBirth") or
            data.get("dateOfBirth")
        )
        if dob:
            return str(dob)

    return None


def _extract_dob_from_html(player_id: str) -> Optional[str]:
    """
    Fetch the ESPN college-football player HTML page and extract DOB by:
      1. Searching all <script> tags for a "dateOfBirth" JSON key
      2. Looking for Birthday/Born/DOB text patterns in the bio section
    """
    if not _BS4:
        return None

    url  = _PLAYER_PAGE.format(id=player_id)
    resp = _get(url, headers=_HTML_HEADERS)
    if resp is None:
        return None

    html = resp.text
    soup = BeautifulSoup(html, "lxml")

    # ── Strategy 1: find dateOfBirth in embedded JSON (any <script> tag) ─────
    dob_json_re = re.compile(
        r'"dateOfBirth"\s*:\s*"([^"]{6,})"',
        re.IGNORECASE,
    )
    for script in soup.find_all("script"):
        text = script.string or ""
        m = dob_json_re.search(text)
        if m:
            return m.group(1)

    # Also check the raw HTML directly (covers inline JS not inside <script>)
    m = dob_json_re.search(html)
    if m:
        return m.group(1)

    # ── Strategy 2: Bio section text patterns ─────────────────────────────────
    # ESPN uses several different class/element patterns across years:
    # "Birthday", "Born", "DOB" labels near date values.
    bio_labels = re.compile(
        r"\b(birthday|born|date of birth|dob)\b",
        re.IGNORECASE,
    )
    # Date patterns: M/D/YYYY, MM/DD/YYYY, Month D YYYY, Month D, YYYY
    date_re = re.compile(
        r"\b(\d{1,2}/\d{1,2}/\d{4}|\w+ \d{1,2},?\s*\d{4})\b",
        re.IGNORECASE,
    )

    # Try structured bio elements first
    for el in soup.select(
        "[class*='Bio'], [class*='bio'], "
        "[class*='PlayerHeader'], [class*='Athlete__Bio'], "
        ".playerinfo, #player-bio, .player-bio, "
        "[data-testid*='bio']"
    ):
        text = el.get_text(" ", strip=True)
        if bio_labels.search(text):
            dm = date_re.search(text)
            if dm:
                return dm.group(1)

    # Broader fallback: any element that has a birthday label near a date
    full_text = soup.get_text(" ", strip=True)
    for m in re.finditer(
        r"(?:birthday|born|date of birth|dob)\s*[:\s]+(\d{1,2}/\d{1,2}/\d{4}|\w+ \d{1,2},?\s*\d{4})",
        full_text,
        re.IGNORECASE,
    ):
        return m.group(1)

    return None


def scrape_player_profile(player_id: str) -> Dict[str, Any]:
    """
    Fetch full player profile for the given ESPN athlete ID.

    Returns a dict with: dob, age, team, position, height, weight,
    espn_id, url, source.
    """
    result: Dict[str, Any] = {
        "espn_id":   player_id,
        "url":       _PLAYER_PAGE.format(id=player_id),
        "dob":       None,
        "age":       None,
        "team":      None,
        "position":  None,
        "height":    None,
        "weight":    None,
        "source":    None,
    }

    # ── Step A: Athlete detail API (most reliable) ────────────────────────────
    for url_template in (_ATHLETE_URL, _ATHLETE_CORE):
        url  = url_template.format(id=player_id)
        resp = _get(url, headers=_JSON_HEADERS)
        if resp is None:
            continue
        try:
            data = resp.json()
        except ValueError:
            continue

        ath = data.get("athlete") or data

        # Extract all available fields
        result["team"]     = (
            (ath.get("team") or {}).get("displayName") or
            (ath.get("team") or {}).get("location") or
            ath.get("teamName")
        )
        result["position"] = (
            (ath.get("position") or {}).get("abbreviation") or
            ath.get("position")
        )
        result["height"]   = ath.get("displayHeight") or ath.get("height")
        result["weight"]   = ath.get("displayWeight") or ath.get("weight")

        raw_dob = ath.get("dateOfBirth")
        if raw_dob:
            dob_iso, age = parse_dob_and_calculate_age(str(raw_dob))
            if dob_iso:
                result["dob"]    = dob_iso
                result["age"]    = age
                result["source"] = "espn_athlete_api"
                return result

        # API responded but no DOB — break and fall through to HTML
        break

    # ── Step B: HTML scraping fallback ────────────────────────────────────────
    if result["dob"] is None:
        raw_dob = _extract_dob_from_html(player_id)
        if raw_dob:
            dob_iso, age = parse_dob_and_calculate_age(raw_dob)
            if dob_iso:
                result["dob"]    = dob_iso
                result["age"]    = age
                result["source"] = "espn_html"

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Part 4 — main combined function
# ─────────────────────────────────────────────────────────────────────────────

def get_player_age(
    name:       str,
    team:       Optional[str]   = None,
    position:   Optional[str]   = None,
    draft_year: Optional[int]   = None,
) -> Dict[str, Any]:
    """
    Full pipeline: search → profile → DOB → age.

    Returns a structured result dict.  age and dob may be None on failure.
    """
    cache_key = f"{_norm_name(name)}|{(team or '').lower()}"
    if cache_key in _CACHE:
        return _CACHE[cache_key]

    result: Dict[str, Any] = {
        "player_name": name,
        "espn_id":     None,
        "team":        team,
        "position":    position,
        "dob":         None,
        "age":         None,
        "source":      None,
        "url":         None,
        "confidence":  0.0,
    }

    # Reference date for age: late April of draft_year (NFL Draft timing)
    ref = date(draft_year, 4, 25) if draft_year else date.today()

    # ── Step 1: Find ESPN athlete ID ─────────────────────────────────────────
    espn_id, search_conf, best_item = search_player_on_espn(name, team, position)

    if espn_id is None:
        result["confidence"] = 0.0
        _CACHE[cache_key] = result
        return result

    result["espn_id"]    = espn_id
    result["url"]        = _PLAYER_PAGE.format(id=espn_id)
    result["confidence"] = round(search_conf, 2)

    # Check if the search result itself already had a DOB
    if best_item and best_item.get("dateOfBirth"):
        dob_iso, age = parse_dob_and_calculate_age(best_item["dateOfBirth"], ref)
        if dob_iso:
            result["dob"]    = dob_iso
            result["age"]    = age
            result["source"] = "espn_search"
            _CACHE[cache_key] = result
            return result

    # ── Step 2: Full profile scrape ──────────────────────────────────────────
    profile = scrape_player_profile(espn_id)

    # Recalculate age at draft reference date if we have dob
    if profile.get("dob"):
        _, age_at_draft = parse_dob_and_calculate_age(profile["dob"], ref)
        profile["age"] = age_at_draft

    result.update({
        "team":     profile.get("team") or team,
        "position": profile.get("position") or position,
        "dob":      profile.get("dob"),
        "age":      profile.get("age"),
        "source":   profile.get("source"),
        "height":   profile.get("height"),
        "weight":   profile.get("weight"),
    })

    _CACHE[cache_key] = result
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Part 7 — batch function (drop-in replacement for ingestion.fetch_espn_ages)
# ─────────────────────────────────────────────────────────────────────────────

def fetch_espn_ages_robust(
    names:          List[str],
    draft_year:     int,
    prospects_meta: Optional[List[Dict[str, Any]]] = None,
    delay:          float = 0.30,
) -> Dict[str, float]:
    """
    Look up ages for a list of prospect names.

    Args:
        names:          player names (same order as prospects_meta if provided)
        draft_year:     used as reference year for age calculation
        prospects_meta: optional list of dicts with "name", "school", "position"
                        for disambiguation; if None, school/position not used.
        delay:          base seconds between requests (default 0.30)

    Returns:
        {name_lower: age_at_draft_float}
    """
    meta_by_name: Dict[str, Dict] = {}
    if prospects_meta:
        for m in prospects_meta:
            key = _norm_name(m.get("name") or "")
            if key:
                meta_by_name[key] = m

    found = 0
    no_match = 0
    no_dob = 0
    result: Dict[str, float] = {}

    print(f"[espn] Starting age lookup for {len(names)} prospects (draft_year={draft_year})")

    for i, name in enumerate(names):
        if (i + 1) % 20 == 0:
            print(
                f"[espn] {i + 1}/{len(names)} processed — "
                f"{found} found, {no_match} no match, {no_dob} no DOB"
            )

        nk = _norm_name(name)
        meta = meta_by_name.get(nk, {})
        team     = meta.get("school") or meta.get("team")
        position = meta.get("position")

        r = get_player_age(name, team=team, position=position, draft_year=draft_year)

        if r["espn_id"] is None:
            no_match += 1
            print(f"[espn] {name}: no ESPN match found")
        elif r["age"] is None:
            no_dob += 1
            print(f"[espn] {name}: matched espn_id={r['espn_id']} but no DOB")
        else:
            found += 1
            result[name.lower().strip()] = r["age"]
            print(
                f"[espn] {name}: dob={r['dob']} → age={r['age']:.2f} "
                f"(src={r['source']}, conf={r['confidence']:.2f})"
            )

        time.sleep(delay)

    print(
        f"[espn] COMPLETE: {found}/{len(names)} ages resolved "
        f"({no_match} no match, {no_dob} no DOB)"
    )
    return result


# ─────────────────────────────────────────────────────────────────────────────
# CLI testing helper
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    test_players = [
        ("Travis Hunter",      "Colorado",     "WR"),
        ("Shedeur Sanders",    "Colorado",     "QB"),
        ("Tetairoa McMillan",  "Arizona",      "WR"),
        ("Will Campbell",      "LSU",          "OL"),
        ("Jeremiyah Love",     "Notre Dame",   "RB"),
        ("Ashton Jeanty",      "Boise State",  "RB"),
        ("Colston Loveland",   "Michigan",     "TE"),
        ("Mason Graham",       "Michigan",     "DL"),
    ]

    if len(sys.argv) > 1:
        test_players = [(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None, None)]

    print("=" * 60)
    for pname, pteam, ppos in test_players:
        r = get_player_age(pname, team=pteam, position=ppos, draft_year=2026)
        print(
            f"{pname:<25s} | "
            f"id={r.get('espn_id') or 'NONE':<10s} | "
            f"dob={r.get('dob') or 'NONE':<12s} | "
            f"age={str(r.get('age') or 'NONE'):<6s} | "
            f"src={r.get('source') or 'none':<20s} | "
            f"conf={r.get('confidence', 0):.2f}"
        )
        time.sleep(0.4)
