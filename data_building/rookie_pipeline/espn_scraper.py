"""
Robust ESPN college football player age / DOB scraper.

Scraping strategy (tried in order for each player):
  1. ESPN search API  → find athlete ID
  2. ESPN athlete API → full profile JSON including dateOfBirth
  3. ESPN player HTML → BeautifulSoup scrape of embedded JSON / page text
  4. Return None with low confidence if all fail
"""

from __future__ import annotations

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

_SEARCH_URL = "https://site.api.espn.com/apis/common/v3/search"
_ATHLETE_URL = "https://site.api.espn.com/apis/site/v2/sports/football/college-football/athletes/{id}"
_ATHLETE_CORE = "https://sports.core.api.espn.com/v2/sports/football/leagues/college-football/athletes/{id}"
_PLAYER_PAGE = "https://www.espn.com/college-football/player/_/id/{id}"

_BROWSER_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)

_JSON_HEADERS = {
    "User-Agent": _BROWSER_UA,
    "Accept": "application/json",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.espn.com/",
}
_HTML_HEADERS = {**_JSON_HEADERS, "Accept": "text/html,application/xhtml+xml,*/*;q=0.8"}

_CACHE: Dict[str, Dict[str, Any]] = {}

_SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}

_DOB_FORMATS = [
    "%Y-%m-%dT%H:%M:%SZ",
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%dT%H:%MZ",
    "%Y-%m-%d",
    "%m/%d/%Y",
    "%B %d, %Y",
    "%b %d, %Y",
]


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _norm_name(n: str) -> str:
    n = (n or "").lower().strip()
    n = re.sub(r"\.", "", n)
    n = re.sub(r"['\-\u2019]", " ", n)
    n = re.sub(r"\s+", " ", n).strip()
    parts = [p for p in n.split() if p not in _SUFFIXES]
    return " ".join(parts)


def _name_score(item_name: str, query_name: str) -> float:
    if not item_name or not query_name:
        return 0.0

    if item_name.lower().strip() == query_name.lower().strip():
        return 1.0

    ni = _norm_name(item_name)
    nq = _norm_name(query_name)

    if ni == nq:
        return 0.9

    ni_parts = ni.split()
    nq_parts = nq.split()
    if not ni_parts or not nq_parts:
        return 0.0

    # strong first+last overlap
    if ni_parts[0] == nq_parts[0] and ni_parts[-1] == nq_parts[-1]:
        return 0.8

    # weak last-name-only fallback
    if ni_parts[-1] == nq_parts[-1]:
        return 0.5

    return 0.0


def _team_score(item_team: str, query_team: str) -> float:
    if not query_team:
        return 0.5
    it = (item_team or "").lower().strip()
    qt = query_team.lower().strip()
    if not it:
        return 0.0
    if qt in it or it in qt:
        return 1.0

    qt_parts = qt.split()
    if qt_parts and qt_parts[-1] in it:
        return 0.7
    if qt_parts and qt_parts[0] in it:
        return 0.6
    return 0.0


def _clean_espn_id(raw_id: Any) -> Optional[str]:
    if raw_id is None:
        return None
    s = str(raw_id).strip()
    if not s:
        return None
    if "~a:" in s:
        s = s.split("~a:")[-1]
    m = re.search(r"(\d+)$", s)
    return m.group(1) if m else None


def _get(
    url: str,
    params: Optional[Dict] = None,
    headers: Optional[Dict] = None,
    retries: int = 3,
    timeout: int = 12,
) -> Optional[requests.Response]:
    h = headers or _JSON_HEADERS
    for attempt in range(retries):
        try:
            resp = requests.get(url, params=params, headers=h, timeout=timeout)
            if resp.status_code in (400, 401, 403, 404):
                                return None
            if resp.status_code == 429:
                time.sleep(2.0 * (attempt + 1))
                continue

            resp.raise_for_status()
            return resp

        except requests.Timeout as e:
                        time.sleep(1.5 ** attempt)

        except requests.exceptions.ProxyError as e:
                        return None

        except requests.RequestException as e:
                        time.sleep(1.5 ** attempt)

        return None


def parse_dob_and_calculate_age(
    dob_str: str,
    ref_date: Optional[date] = None,
) -> Tuple[Optional[str], Optional[float]]:
    if not dob_str:
        return None, None

    raw = dob_str.strip()

    # remove trailing timezone offsets like +00:00
    raw = re.sub(r"\+\d{2}:\d{2}$", "", raw)

    parsed_dob: Optional[date] = None

    for fmt in _DOB_FORMATS:
        try:
            parsed_dob = datetime.strptime(raw, fmt).date()
            break
        except ValueError:
            continue

    if parsed_dob is None:
        iso_match = re.search(r"(\d{4}-\d{2}-\d{2})", raw)
        if iso_match:
            try:
                parsed_dob = datetime.strptime(iso_match.group(1), "%Y-%m-%d").date()
            except ValueError:
                pass

    if parsed_dob is None:
        return None, None

    ref = ref_date or date.today()
    age = round((ref - parsed_dob).days / 365.25, 2)
    return parsed_dob.isoformat(), age


# ─────────────────────────────────────────────────────────────────────────────
# Search
# ─────────────────────────────────────────────────────────────────────────────

def _parse_search_items(data: Dict) -> List[Dict]:
    items = data.get("items") or []
    if items:
        return list(items)

    flat: List[Dict] = []
    for result in (data.get("results") or []):
        flat.extend(result.get("contents") or [])
    return flat


def _name_variants(name: str) -> List[str]:
    variants = []
    clean = name.strip()

    variants.append(clean)

    # remove suffix
    parts = clean.split()
    if parts and parts[-1].lower().replace(".", "") in _SUFFIXES:
        variants.append(" ".join(parts[:-1]))

    # CJ -> C.J.
    if len(parts) >= 2 and len(parts[0]) == 2 and parts[0].isalpha() and "." not in parts[0]:
        variants.append(f"{parts[0][0]}.{parts[0][1]}. {' '.join(parts[1:])}")

    # remove punctuation
    variants.append(re.sub(r"[^\w\s]", "", clean))

    # dedupe
    out = []
    seen = set()
    for v in variants:
        v = v.strip()
        if v and v not in seen:
            seen.add(v)
            out.append(v)
    return out


def search_player_on_espn(
    name: str,
    team: Optional[str] = None,
    position: Optional[str] = None,
) -> Tuple[Optional[str], float, Optional[Dict]]:
    best_id: Optional[str] = None
    best_conf = 0.0
    best_item: Optional[Dict] = None

    def _score_items(items: List[Dict]) -> None:
        nonlocal best_id, best_conf, best_item

        for item in items:
            item_type = (item.get("type") or "").lower()
            if item_type in {"story", "video", "team", "event", "league"}:
                continue

            display = (
                item.get("displayName")
                or item.get("name")
                or item.get("fullName")
                or ""
            )
            if not display:
                continue

            ns = _name_score(display, name)
            if ns < 0.5:
                continue

            item_team = (
                (item.get("team") or {}).get("displayName")
                or (item.get("team") or {}).get("location")
                or item.get("teamName")
                or ""
            )
            ts = _team_score(item_team, team or "")

            item_pos = (
                (item.get("position") or {}).get("abbreviation")
                or item.get("position")
                or ""
            )
            pos_bonus = 0.05 if position and item_pos and str(item_pos).upper() == str(position).upper() else 0.0

            league_name = (
                (item.get("league") or {}).get("name")
                or item.get("leagueName")
                or ""
            ).lower()
            league_bonus = 0.10 if "college" in league_name else 0.0
            dob_bonus = 0.05 if item.get("dateOfBirth") else 0.0

            conf = min(ns * 0.55 + ts * 0.25 + pos_bonus + league_bonus + dob_bonus, 1.0)
            item_id = _clean_espn_id(item.get("id") or item.get("uid"))

            if item_id and conf > best_conf:
                best_conf = conf
                best_id = item_id
                best_item = item

    for query in _name_variants(name):
        resp = _get(_SEARCH_URL, params={"query": query, "limit": "10"}, headers=_JSON_HEADERS)
        if resp is None:
            continue

        try:
            raw = resp.json()
            items = _parse_search_items(raw)
            _score_items(items)
        except ValueError:
            continue

        if best_id:
            break

    if best_conf < 0.30:
        return None, 0.0, None

    return best_id, round(best_conf, 2), best_item


# ─────────────────────────────────────────────────────────────────────────────
# Athlete/profile extraction
# ─────────────────────────────────────────────────────────────────────────────

def _extract_dob_from_html(player_id: str) -> Optional[str]:
    if not _BS4:
        return None

    url = _PLAYER_PAGE.format(id=player_id)
    resp = _get(url, headers=_HTML_HEADERS)
    if resp is None:
        return None

    html = resp.text
    soup = BeautifulSoup(html, "html.parser")

    dob_json_re = re.compile(r'"dateOfBirth"\s*:\s*"([^"]+)"', re.IGNORECASE)

    for script in soup.find_all("script"):
        text = script.string or script.get_text() or ""
        m = dob_json_re.search(text)
        if m:
            return m.group(1)

    m = dob_json_re.search(html)
    if m:
        return m.group(1)

    full_text = soup.get_text(" ", strip=True)
    text_match = re.search(
        r"(?:birthday|born|date of birth|dob)\s*[:\s]+(\d{1,2}/\d{1,2}/\d{4}|\w+ \d{1,2},?\s*\d{4})",
        full_text,
        re.IGNORECASE,
    )
    if text_match:
        return text_match.group(1)

    return None

def scrape_player_profile(player_id: str) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "espn_id": player_id,
        "url": _PLAYER_PAGE.format(id=player_id),
        "dob": None,
        "age": None,
        "team": None,
        "position": None,
        "height": None,
        "weight": None,
        "source": None,
    }

    # Step A: athlete API first
    for url_template in (_ATHLETE_URL, _ATHLETE_CORE):
        url = url_template.format(id=player_id)
        resp = _get(url, headers=_JSON_HEADERS)
        if resp is None:
            continue

        try:
            data = resp.json()
        except ValueError:
            continue

        athlete = data.get("athlete") or data

        result["team"] = (
            (athlete.get("team") or {}).get("displayName")
            or (athlete.get("team") or {}).get("location")
            or athlete.get("teamName")
            or result["team"]
        )
        result["position"] = (
            (athlete.get("position") or {}).get("abbreviation")
            or athlete.get("position")
            or result["position"]
        )
        result["height"] = athlete.get("displayHeight") or athlete.get("height") or result["height"]
        result["weight"] = athlete.get("displayWeight") or athlete.get("weight") or result["weight"]

        raw_dob = athlete.get("dateOfBirth") or data.get("dateOfBirth")
        if raw_dob:
            dob_iso, age = parse_dob_and_calculate_age(str(raw_dob))
            if dob_iso:
                result["dob"] = dob_iso
                result["age"] = age
                result["source"] = "espn_athlete_api"
                return result

    # Step B: HTML fallback
    raw_dob = _extract_dob_from_html(player_id)
    if raw_dob:
        dob_iso, age = parse_dob_and_calculate_age(raw_dob)
        if dob_iso:
            result["dob"] = dob_iso
            result["age"] = age
            result["source"] = "espn_html"

    return result

# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def get_player_age(
    name: str,
    team: Optional[str] = None,
    position: Optional[str] = None,
    draft_year: Optional[int] = None,
) -> Dict[str, Any]:
    cache_key = f"{_norm_name(name)}|{(team or '').lower()}|{(position or '').upper()}"
    if cache_key in _CACHE:
        return _CACHE[cache_key]

    result: Dict[str, Any] = {
        "player_name": name,
        "espn_id": None,
        "team": team,
        "position": position,
        "dob": None,
        "age": None,
        "source": None,
        "url": None,
        "confidence": 0.0,
        "height": None,
        "weight": None,
    }

    ref = date(draft_year, 4, 25) if draft_year else date.today()

    espn_id, search_conf, best_item = search_player_on_espn(name, team, position)

    if espn_id is None:
        _CACHE[cache_key] = result
        return result

    result["espn_id"] = espn_id
    result["url"] = _PLAYER_PAGE.format(id=espn_id)
    result["confidence"] = round(search_conf, 2)

    search_dob = best_item.get("dateOfBirth") if best_item else None
    if search_dob:
        dob_iso, age = parse_dob_and_calculate_age(search_dob, ref)
        if dob_iso:
            result["dob"] = dob_iso
            result["age"] = age
            result["source"] = "espn_search"
            _CACHE[cache_key] = result
            return result

    profile = scrape_player_profile(espn_id)

    if profile.get("dob"):
        _, age_at_ref = parse_dob_and_calculate_age(profile["dob"], ref)
    else:
        age_at_ref = None

    result.update({
        "team": profile.get("team") or team,
        "position": profile.get("position") or position,
        "dob": profile.get("dob"),
        "age": age_at_ref,
        "source": profile.get("source"),
        "height": profile.get("height"),
        "weight": profile.get("weight"),
    })

    _CACHE[cache_key] = result
    return result


def fetch_espn_ages_robust(
    names: List[str],
    draft_year: int,
    prospects_meta: Optional[List[Dict[str, Any]]] = None,
    delay: float = 0.30,
) -> Dict[str, float]:
    meta_by_name: Dict[str, Dict[str, Any]] = {}
    if prospects_meta:
        for m in prospects_meta:
            key = _norm_name(m.get("name") or "")
            if key:
                meta_by_name[key] = m

    result: Dict[str, float] = {}

    print(f"[espn] Starting age lookup for {len(names)} prospects (draft_year={draft_year})")

    # Quick connectivity test — skip entire loop if ESPN is unreachable
    _probe = _get(_SEARCH_URL, params={"query": "Travis Hunter", "limit": "3"})
    if _probe is None:
        print("[espn] WARNING: ESPN API unreachable — skipping age lookup (combine data is fallback)")
        return {}
    try:
        _probe_items = len(_parse_search_items(_probe.json()))
        print(f"[espn] Connectivity OK — test query returned {_probe_items} items")
    except ValueError:
        print("[espn] WARNING: ESPN returned unparseable response — skipping")
        return {}

    for name in names:
        nk = _norm_name(name)
        meta = meta_by_name.get(nk, {})
        team = meta.get("school") or meta.get("team")
        position = meta.get("position")

        r = get_player_age(name, team=team, position=position, draft_year=draft_year)
        if r.get("age") is not None:
            result[name.lower().strip()] = float(r["age"])

        time.sleep(delay)

    return result


if __name__ == "__main__":
    test_players = [
        ("Travis Hunter", "Colorado", "WR"),
        ("Shedeur Sanders", "Colorado", "QB"),
        ("Tetairoa McMillan", "Arizona", "WR"),
        ("Will Campbell", "LSU", "OL"),
        ("Jeremiyah Love", "Notre Dame", "RB"),
        ("Ashton Jeanty", "Boise State", "RB"),
        ("Colston Loveland", "Michigan", "TE"),
        ("Mason Graham", "Michigan", "DL"),
    ]

    print("=" * 80)
    for pname, pteam, ppos in test_players:
        r = get_player_age(pname, team=pteam, position=ppos, draft_year=2026)
        print(
            f"{pname:<25} | "
            f"id={str(r.get('espn_id') or 'NONE'):<10} | "
            f"dob={str(r.get('dob') or 'NONE'):<12} | "
            f"age={str(r.get('age') or 'NONE'):<6} | "
            f"src={str(r.get('source') or 'none'):<18} | "
            f"conf={r.get('confidence', 0):.2f}"
        )
        time.sleep(0.4)