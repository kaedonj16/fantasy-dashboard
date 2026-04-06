"""
Rookie prospect data ingestion.

Primary source: Sportradar NFL Draft API (requires SPORTRADAR_API_KEY env var).
Fallback:       Curated seed dataset for 2025 and 2026 draft classes so the
                page works immediately without any external credentials.

Sportradar endpoint (Draft API v1, separate from the main NFL v6 API):
    GET https://api.sportradar.com/draft/nfl/{access_level}/v1/en/{year}/prospects.json
        ?api_key={key}
    OR  x-api-key: {key}  header

    access_level = "trial" | "production"  (matches your key tier)

What this endpoint provides:
    id, name, first_name, last_name, position,
    height (inches), weight (lbs),
    team_name (college), conference {name, alias},
    experience (SR/JR/SO/FR), birth_place (city/state string),
    top_prospect (bool)

What it does NOT provide:
    - Age / birth date  (birth_place is a city string, not a DOB)
    - Combine measurements (40-time, vertical, bench, etc.)
    - College stats

Age, combine data, and stats come from the seed dataset and are preserved
for any player matched by name between Sportradar and the seed.

Normalization contract — every player dict returned by this module has:
    player_id, name, position, school, age, height_inches, weight_lbs,
    draft_class_year, early_declare, transfer_history,
    seasons: [ { season, games_played, rush/rec/pass stats, team, conference,
                  dominator_rating, market_share_yards, market_share_tds,
                  yds_per_carry, yds_per_reception, yds_per_attempt,
                  completion_pct, td_int_ratio, team_pass_rate } ]
    athleticism: { forty_yard, vertical_inches, broad_jump_in, three_cone,
                   short_shuttle, bench_reps, speed_score, ras_score }
"""
from __future__ import annotations

import csv
import io
import logging
import os
import re
import time
from typing import Any, Dict, List, Optional

import requests

log = logging.getLogger(__name__)

SPORTRADAR_KEY    = os.getenv("SPORTRADAR_API_KEY", "")
SPORTRADAR_ACCESS = os.getenv("SPORTRADAR_ACCESS_LEVEL", "trial")   # "trial" or "production"
_SR_BASE          = "https://api.sportradar.com/draft/nfl"

# Supplementary sources
CFBD_KEY  = os.getenv("CFBD_API_KEY", "")                           # college stats
CFBD_BASE = "https://api.collegefootballdata.com"
_NFLVERSE_COMBINE_URL = (
    "https://github.com/nflverse/nflverse-data/releases/download/combine/combine.csv"
)

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _slug(name: str) -> str:
    """Convert 'Travis Hunter' → 'TRAVIS_HUNTER'."""
    return re.sub(r"[^A-Z0-9]+", "_", name.upper()).strip("_")


def _safe(v, default=None):
    try:
        return float(v) if v is not None else default
    except (TypeError, ValueError):
        return default


def _safe_int(v, default=None):
    try:
        return int(v) if v is not None else default
    except (TypeError, ValueError):
        return default


def _sportradar_get(path: str, retries: int = 3) -> Optional[Any]:
    """
    GET from Sportradar Draft API with retry/backoff.
    Sends the API key as both query param and header (Sportradar accepts both).
    Returns parsed JSON or None on failure.
    """
    url = f"{_SR_BASE}/{SPORTRADAR_ACCESS}/v1/en/{path}"
    headers = {"x-api-key": SPORTRADAR_KEY, "Accept": "application/json"}
    params  = {"api_key": SPORTRADAR_KEY}
    for attempt in range(retries):
        try:
            resp = requests.get(url, headers=headers, params=params, timeout=20)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as exc:
            wait = 2 ** attempt
            log.warning("[sportradar] %s attempt %d failed: %s — retrying in %ds",
                        path, attempt + 1, exc, wait)
            time.sleep(wait)
    log.error("[sportradar] %s failed after %d attempts", path, retries)
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Supplementary source 1 — NFLVerse combine.csv  (no auth required)
# Provides: forty_yard, vertical_inches, broad_jump_in, bench_reps,
#           three_cone, short_shuttle, height, weight, school
# URL: https://github.com/nflverse/nflverse-data/releases/download/combine/combine.csv
# ─────────────────────────────────────────────────────────────────────────────

def fetch_nflverse_combine(draft_year: int) -> Dict[str, Dict[str, Any]]:
    """
    Download the NFLVerse combine CSV and return a name-keyed dict of
    combine measurements for prospects from `draft_year`.

    Returns {player_name_lower: athleticism_dict}  where athleticism_dict has:
        forty_yard, vertical_inches, broad_jump_in, bench_reps,
        three_cone, short_shuttle
    Also includes height_inches and weight_lbs as fallback bio fields.
    """
    try:
        resp = requests.get(_NFLVERSE_COMBINE_URL, timeout=30)
        resp.raise_for_status()
    except requests.RequestException as exc:
        log.warning("[nflverse] combine.csv download failed: %s", exc)
        return {}

    results: Dict[str, Dict[str, Any]] = {}
    reader = csv.DictReader(io.StringIO(resp.text))
    for row in reader:
        # draft_year column is the year they were drafted
        row_year = _safe_int(row.get("draft_year") or row.get("season"))
        if row_year != draft_year:
            continue

        name = (row.get("player_name") or "").strip().lower()
        if not name:
            continue

        def _csv_float(col):
            v = row.get(col, "").strip()
            return _safe(v) if v and v != "NA" else None

        def _csv_int(col):
            v = row.get(col, "").strip()
            return _safe_int(v) if v and v != "NA" else None

        ath: Dict[str, Any] = {}
        if (v := _csv_float("forty"))        is not None: ath["forty_yard"]      = v
        if (v := _csv_float("vertical"))     is not None: ath["vertical_inches"] = v
        if (v := _csv_float("broad_jump"))   is not None: ath["broad_jump_in"]   = v
        if (v := _csv_int("bench"))          is not None: ath["bench_reps"]      = v
        if (v := _csv_float("cone"))         is not None: ath["three_cone"]      = v
        if (v := _csv_float("shuttle"))      is not None: ath["short_shuttle"]   = v

        # Height is "6-2" format in nflverse; weight is integer lbs
        results[name] = {
            "athleticism":    ath,
            "height_inches":  _parse_height(row.get("ht")),
            "weight_lbs":     _csv_int("wt"),
        }

    log.info("[nflverse] Loaded combine data for %d prospects in %d class",
             len(results), draft_year)
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Supplementary source 2 — CFBD college stats  (requires CFBD_API_KEY)
# Provides: per-season receiving/rushing/passing stats, games_played,
#           team, conference, market share, dominator rating
# ─────────────────────────────────────────────────────────────────────────────

def _cfbd_get(path: str, params: Dict[str, Any] = None, retries: int = 3) -> Optional[Any]:
    url = f"{CFBD_BASE}{path}"
    headers = {"Accept": "application/json", "Authorization": f"Bearer {CFBD_KEY}"}
    for attempt in range(retries):
        try:
            resp = requests.get(url, headers=headers, params=params or {}, timeout=15)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as exc:
            wait = 2 ** attempt
            log.warning("[cfbd] %s attempt %d failed: %s — retrying in %ds",
                        path, attempt + 1, exc, wait)
            time.sleep(wait)
    log.error("[cfbd] %s failed after %d attempts", path, retries)
    return None


def _build_cfbd_season(raw_stats: List[Dict], team_stats: Dict, season: int,
                       games: Optional[int]) -> Dict:
    """Fold CFBD stat rows for one player-season into a single normalized dict."""
    row: Dict[str, Any] = {
        "season": season, "games_played": games,
        "pass_yards": 0, "pass_tds": 0, "pass_attempts": 0,
        "completions": 0, "interceptions": 0,
        "rush_attempts": 0, "rush_yards": 0, "rush_tds": 0,
        "receptions": 0, "targets": 0, "receiving_yards": 0, "receiving_tds": 0,
        "team": None, "conference": None,
    }
    stat_map = {
        "passingYards": "pass_yards", "passingTDs": "pass_tds",
        "passAttempts": "pass_attempts", "passCompletions": "completions",
        "interceptions": "interceptions",
        "rushingYards": "rush_yards", "rushingTDs": "rush_tds",
        "rushingAttempts": "rush_attempts",
        "receivingYards": "receiving_yards", "receivingTDs": "receiving_tds",
        "receptions": "receptions",
    }
    for s in raw_stats:
        k = s.get("statName", "")
        if k in stat_map:
            row[stat_map[k]] = (row.get(stat_map[k]) or 0) + (_safe_int(s.get("stat")) or 0)
        row["team"]       = row["team"]       or s.get("team")
        row["conference"] = row["conference"] or s.get("conference")

    ts       = team_stats.get(row.get("team", ""), {})
    rush_att = row["rush_attempts"] or 0
    rush_yds = row["rush_yards"]    or 0
    rec_yds  = row["receiving_yards"] or 0
    rec_tds  = row["receiving_tds"]   or 0
    rush_tds = row["rush_tds"]        or 0
    pass_att = row["pass_attempts"]   or 0
    pass_yds = row["pass_yards"]      or 0
    comp     = row["completions"]     or 0
    ints     = row["interceptions"]   or 0

    row["yds_per_carry"]     = round(rush_yds / rush_att, 2) if rush_att > 0 else None
    row["yds_per_reception"] = round(rec_yds / max(row["receptions"] or 1, 1), 2) if rec_yds > 0 else None
    row["yds_per_attempt"]   = round(pass_yds / pass_att, 2) if pass_att > 0 else None
    row["completion_pct"]    = round(comp / pass_att * 100, 1) if pass_att > 0 else None
    row["td_int_ratio"]      = round(row["pass_tds"] / max(ints, 1), 2) if row["pass_tds"] else None

    t_yds = (ts.get("netPassingYards", 0) or 0) + (ts.get("rushingYards", 0) or 0)
    t_tds = (ts.get("passingTDs", 0) or 0) + (ts.get("rushingTDs", 0) or 0)
    p_yds = rec_yds + rush_yds
    p_tds = rec_tds + rush_tds

    row["market_share_yards"] = round(p_yds / t_yds, 3) if t_yds > 0 else None
    row["market_share_tds"]   = round(p_tds / t_tds, 3) if t_tds > 0 else None
    row["team_total_yards"]   = _safe_int(t_yds)
    row["team_total_tds"]     = _safe_int(t_tds)
    row["team_pass_rate"]     = ts.get("pass_rate")

    dom = 0.0
    if t_yds > 0: dom += (p_yds / t_yds) * 0.65
    if t_tds > 0: dom += (p_tds / t_tds) * 0.35
    row["dominator_rating"] = round(dom, 4) if (t_yds or t_tds) else None

    return row


def fetch_cfbd_college_stats(draft_year: int) -> Dict[str, List[Dict]]:
    """
    Fetch college stats from CFBD for the 3 seasons before `draft_year`.
    Returns {player_name_lower: [season_dict, ...]} sorted oldest→newest.
    Requires CFBD_API_KEY env var; returns {} silently if not set.
    """
    if not CFBD_KEY:
        return {}

    years = [draft_year - 1, draft_year - 2, draft_year - 3]

    # Team season totals for market share / dominator calculation
    team_stats: Dict[int, Dict] = {}
    for yr in years:
        data = _cfbd_get("/stats/season", {"year": yr, "seasonType": "regular"})
        if not data:
            team_stats[yr] = {}
            continue
        teams: Dict[str, Dict] = {}
        for row in data:
            t = row.get("team", "")
            teams.setdefault(t, {})[row.get("statName", "")] = _safe(row.get("statValue"), 0)
        for t, s in teams.items():
            pa = s.get("passAttempts", 0) or 0
            ra = s.get("rushingAttempts", 0) or 0
            total = pa + ra
            s["pass_rate"] = round(pa / total, 3) if total > 0 else 0.5
        team_stats[yr] = teams

    # Player usage (games played)
    usage: Dict[int, Dict[int, int]] = {}   # {yr: {player_id: games}}
    for yr in years:
        data = _cfbd_get("/player/usage", {"year": yr, "seasonType": "regular"}) or []
        usage[yr] = {
            int(r["id"]): _safe_int(r.get("games"))
            for r in data if r.get("id") is not None
        }

    # Player season stats — indexed by name and by player ID
    by_name: Dict[int, Dict[str, List]] = {}   # {yr: {name_lower: [rows]}}
    by_id:   Dict[int, Dict[int, List]] = {}   # {yr: {player_id: [rows]}}
    for yr in years:
        data = _cfbd_get("/stats/player/season",
                         {"year": yr, "seasonType": "regular"}) or []
        bn: Dict[str, List] = {}
        bi: Dict[int, List] = {}
        for row in data:
            n  = (row.get("player") or "").lower()
            pid = _safe_int(row.get("playerId"))
            if n:   bn.setdefault(n, []).append(row)
            if pid: bi.setdefault(pid, []).append(row)
        by_name[yr] = bn
        by_id[yr]   = bi

    # Collapse into per-player season lists keyed by lowercase name
    all_names: set = set()
    for yr in years:
        all_names.update(by_name[yr].keys())

    result: Dict[str, List[Dict]] = {}
    for name in all_names:
        seasons = []
        for yr in years:
            rows = by_name[yr].get(name, [])
            if not rows:
                continue
            pid = _safe_int(rows[0].get("playerId"))
            gp  = usage[yr].get(pid) if pid else None
            seasons.append(_build_cfbd_season(rows, team_stats[yr], yr, gp))
        if seasons:
            seasons.sort(key=lambda s: s["season"])
            result[name] = seasons

    log.info("[cfbd] Loaded stats for %d players (draft class %d)", len(result), draft_year)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Supplementary source 3 — Age estimation from Sportradar experience field
# Sportradar provides SR/JR/SO/FR but no DOB.
# We estimate age at draft (late April) from typical enrollment ages.
# ─────────────────────────────────────────────────────────────────────────────

# Typical age at draft by college class + position adjustments are minor,
# so we use position-neutral estimates.
_EXP_AGE: Dict[str, float] = {
    "SR":  22.7,   # 4-year senior; many are 22–23 at April draft
    "JR":  21.5,   # 3-year junior / early declare
    "SO":  20.5,   # 2-year sophomore (rare early declare)
    "FR":  19.5,   # true freshman (extremely rare)
}

def _estimate_age(experience: Optional[str], draft_year: int) -> Optional[float]:
    """
    Estimate age at the draft from Sportradar's experience field.
    Returns a float or None if experience is unknown.
    """
    if not experience:
        return None
    return _EXP_AGE.get(experience.upper())


# ─────────────────────────────────────────────────────────────────────────────
# Sportradar ingestion
# ─────────────────────────────────────────────────────────────────────────────

def _parse_height(raw) -> Optional[int]:
    """
    Convert height to total inches.
    Handles integer inches (74), '6-2', "6'2", '6 2'.
    """
    if raw is None:
        return None
    if isinstance(raw, (int, float)):
        return int(raw)
    s = str(raw).strip().replace('"', '')
    for sep in ("-", "'", " "):
        if sep in s:
            parts = s.split(sep, 1)
            try:
                return int(parts[0]) * 12 + int(parts[1])
            except (ValueError, IndexError):
                pass
    try:
        return int(s)
    except ValueError:
        return None



def _parse_sportradar_prospect(raw: Dict, draft_year: int) -> Optional[Dict]:
    """
    Convert a single Sportradar Draft API prospect object to our normalized format.

    Actual Sportradar fields (Draft API v1):
        name, first_name, last_name — player name
        position                    — e.g. "WR", "RB"
        height                      — integer, total inches
        weight                      — integer, pounds
        team_name                   — college team name string
        conference                  — object: {name, alias}
        experience                  — "SR" / "JR" / "SO" / "FR"
        birth_place                 — city/state string e.g. "Suwanee, GA, USA"
        top_prospect                — boolean

    NOT present: age/DOB, combine measurements, college stats.
    Those come from the seed dataset when a name match is found.
    """
    name = (
        raw.get("name") or
        " ".join(filter(None, [raw.get("first_name"), raw.get("last_name")]))
    ).strip()
    if not name:
        return None

    position = (raw.get("position") or "").upper()
    school   = raw.get("team_name")

    conf_obj    = raw.get("conference") or {}
    conference  = conf_obj.get("name") if isinstance(conf_obj, dict) else conf_obj

    # birth_place is "City, ST, USA" — no DOB available from this endpoint
    birth_place = raw.get("birth_place") or ""
    state = None
    if birth_place:
        parts = [p.strip() for p in birth_place.split(",")]
        if len(parts) >= 2:
            state = parts[1]   # e.g. "GA"

    return {
        "player_id":        f"ROOKIE_{draft_year}_{_slug(name)}",
        "name":             name,
        "position":         position,
        "school":           school,
        "age":              None,   # filled from combine or estimated from experience
        "height_inches":    _safe_int(raw.get("height")),
        "weight_lbs":       _safe_int(raw.get("weight")),
        "state":            state,
        "draft_class_year": draft_year,
        "early_declare":    False,
        "seasons":          [],     # filled from CFBD stats or seed
        "athleticism":      {},     # filled from NFLVerse combine or seed
        "source":           "sportradar",
        # Internal fields used during merge — stripped before normalization
        "_conference":      conference,
        "_experience":      raw.get("experience"),  # SR/JR/SO/FR for age estimation
    }


def fetch_sportradar_prospects(draft_year: int) -> List[Dict[str, Any]]:
    """
    Fetch the full NFL prospect list for `draft_year` from Sportradar.

    Endpoint: GET /draft/nfl/{access_level}/v1/en/{year}/prospects.json
    Returns a list of partially-normalized prospect dicts (bio only).
    Age, combine, and stats are not provided by this endpoint and must be
    merged from the seed dataset or other sources.
    """
    if not SPORTRADAR_KEY:
        log.info("[sportradar] No SPORTRADAR_API_KEY set — skipping live fetch")
        return []

    data = _sportradar_get(f"{draft_year}/prospects.json")
    if not data:
        log.warning("[sportradar] No prospect data returned for %d", draft_year)
        return []

    raw_list = data.get("prospects") or (data if isinstance(data, list) else [])
    if not raw_list:
        log.warning("[sportradar] Empty prospects list for %d", draft_year)
        return []

    results = []
    for raw in raw_list:
        prospect = _parse_sportradar_prospect(raw, draft_year)
        if prospect:
            results.append(prospect)

    log.info("[sportradar] Fetched %d prospects for %d", len(results), draft_year)
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Curated seed dataset
# Each entry represents one prospect with multi-year production.
# Stats are approximate/historical for illustration; replace with live CFBD data
# when CFBD_API_KEY is configured.
#
# Weights:  production (per-game), efficiency, context are what matter to the model.
# The seed focuses on the 2026 draft class (active as of April 2026).
# 2025 class data is included as the "prior year" class.
# ─────────────────────────────────────────────────────────────────────────────

SEED_PROSPECTS_2026: List[Dict[str, Any]] = [
    # ── 2026 Draft Class (2025 college season) ──────────────────────────────
    # ── WRs ──────────────────────────────────────────────────────────────────
    {
        "player_id": "ROOKIE_2026_CARNELL_TATE",
        "name": "Carnell Tate", "position": "WR", "school": "Ohio State",
        "age": 20.3, "height_inches": 74, "weight_lbs": 205,
        "draft_class_year": 2026, "early_declare": True,
        "seasons": [
            {"season": 2025, "games_played": 12, "receptions": 68, "targets": 95,
             "receiving_yards": 1050, "receiving_tds": 9,
             "yds_per_reception": 15.4, "dominator_rating": 0.28,
             "market_share_yards": 0.29, "market_share_tds": 0.35,
             "team": "Ohio State", "conference": "Big Ten", "team_pass_rate": 0.61},
            {"season": 2024, "games_played": 13, "receptions": 34, "targets": 52,
             "receiving_yards": 515, "receiving_tds": 5,
             "yds_per_reception": 15.1, "dominator_rating": 0.10,
             "team": "Ohio State", "conference": "Big Ten", "team_pass_rate": 0.62},
        ],
        "athleticism": {"forty_yard": 4.42, "vertical_inches": 37.5,
                        "broad_jump_in": 124, "ras_score": 8.7},
    },
    {
        "player_id": "ROOKIE_2026_ISAIAH_BOND",
        "name": "Isaiah Bond", "position": "WR", "school": "Texas",
        "age": 21.1, "height_inches": 72, "weight_lbs": 190,
        "draft_class_year": 2026, "early_declare": True,
        "seasons": [
            {"season": 2025, "games_played": 13, "receptions": 65, "targets": 92,
             "receiving_yards": 1105, "receiving_tds": 11,
             "yds_per_reception": 17.0, "dominator_rating": 0.25,
             "market_share_yards": 0.27, "market_share_tds": 0.38,
             "team": "Texas", "conference": "SEC", "team_pass_rate": 0.59},
            {"season": 2024, "games_played": 12, "receptions": 48, "targets": 72,
             "receiving_yards": 688, "receiving_tds": 4,
             "yds_per_reception": 14.3, "dominator_rating": 0.16,
             "team": "Alabama", "conference": "SEC", "team_pass_rate": 0.57,
             "transfer_history": "Alabama → Texas"},
        ],
        "athleticism": {"forty_yard": 4.32, "vertical_inches": 39.0,
                        "broad_jump_in": 128, "ras_score": 9.3},
    },
    {
        "player_id": "ROOKIE_2026_JORDAN_HUDSON",
        "name": "Jordan Hudson", "position": "WR", "school": "UCF",
        "age": 20.8, "height_inches": 76, "weight_lbs": 215,
        "draft_class_year": 2026, "early_declare": True,
        "seasons": [
            {"season": 2025, "games_played": 12, "receptions": 78, "targets": 108,
             "receiving_yards": 1242, "receiving_tds": 12,
             "yds_per_reception": 15.9, "dominator_rating": 0.42,
             "market_share_yards": 0.45, "market_share_tds": 0.52,
             "team": "UCF", "conference": "Big 12", "team_pass_rate": 0.62},
            {"season": 2024, "games_played": 13, "receptions": 47, "targets": 68,
             "receiving_yards": 743, "receiving_tds": 6,
             "yds_per_reception": 15.8, "dominator_rating": 0.28,
             "team": "UCF", "conference": "Big 12", "team_pass_rate": 0.60},
        ],
        "athleticism": {"forty_yard": 4.44, "vertical_inches": 36.0,
                        "broad_jump_in": 119, "ras_score": 8.1},
    },
    {
        "player_id": "ROOKIE_2026_EMEKA_EGBUKA",
        "name": "Emeka Egbuka", "position": "WR", "school": "Ohio State",
        "age": 23.1, "height_inches": 73, "weight_lbs": 205,
        "draft_class_year": 2026,
        "seasons": [
            {"season": 2025, "games_played": 13, "receptions": 72, "targets": 98,
             "receiving_yards": 935, "receiving_tds": 8,
             "yds_per_reception": 13.0, "dominator_rating": 0.24,
             "market_share_yards": 0.26, "market_share_tds": 0.31,
             "team": "Ohio State", "conference": "Big Ten", "team_pass_rate": 0.61},
            {"season": 2024, "games_played": 15, "receptions": 81, "targets": 110,
             "receiving_yards": 1011, "receiving_tds": 10,
             "yds_per_reception": 12.5, "dominator_rating": 0.20,
             "team": "Ohio State", "conference": "Big Ten", "team_pass_rate": 0.62},
        ],
        "athleticism": {"forty_yard": 4.39, "ras_score": 9.0},
    },
    # ── RBs ──────────────────────────────────────────────────────────────────
    {
        "player_id": "ROOKIE_2026_JEREMIYAH_LOVE",
        "name": "Jeremiyah Love", "position": "RB", "school": "Wisconsin",
        "age": 20.1, "height_inches": 70, "weight_lbs": 205,
        "draft_class_year": 2026, "early_declare": True,
        "seasons": [
            {"season": 2025, "games_played": 13, "rush_attempts": 285,
             "rush_yards": 1705, "rush_tds": 19, "receptions": 22,
             "receiving_yards": 198, "receiving_tds": 1,
             "yds_per_carry": 5.98, "dominator_rating": 0.58,
             "market_share_yards": 0.60, "market_share_tds": 0.70,
             "team": "Wisconsin", "conference": "Big Ten", "team_pass_rate": 0.43},
            {"season": 2024, "games_played": 13, "rush_attempts": 173,
             "rush_yards": 1069, "rush_tds": 16, "receptions": 15,
             "receiving_yards": 121, "receiving_tds": 0,
             "yds_per_carry": 6.18, "dominator_rating": 0.42,
             "team": "Wisconsin", "conference": "Big Ten", "team_pass_rate": 0.44},
        ],
        "athleticism": {"forty_yard": 4.40, "vertical_inches": 39.0,
                        "broad_jump_in": 128, "ras_score": 9.4},
    },
    {
        "player_id": "ROOKIE_2026_OLLIE_GORDON",
        "name": "Ollie Gordon", "position": "RB", "school": "Oklahoma State",
        "age": 21.8, "height_inches": 73, "weight_lbs": 220,
        "draft_class_year": 2026, "early_declare": True,
        "seasons": [
            {"season": 2025, "games_played": 12, "rush_attempts": 238,
             "rush_yards": 1380, "rush_tds": 14, "receptions": 28,
             "receiving_yards": 265, "receiving_tds": 2,
             "yds_per_carry": 5.80, "dominator_rating": 0.52,
             "market_share_yards": 0.54, "market_share_tds": 0.64,
             "team": "Oklahoma State", "conference": "Big 12", "team_pass_rate": 0.48},
            {"season": 2024, "games_played": 13, "rush_attempts": 285,
             "rush_yards": 1732, "rush_tds": 21, "receptions": 33,
             "receiving_yards": 330, "receiving_tds": 1,
             "yds_per_carry": 6.08, "dominator_rating": 0.63,
             "team": "Oklahoma State", "conference": "Big 12", "team_pass_rate": 0.46},
        ],
        "athleticism": {"forty_yard": 4.46, "vertical_inches": 36.5,
                        "broad_jump_in": 121, "ras_score": 8.5},
    },
    {
        "player_id": "ROOKIE_2026_JORDAN_JAMES",
        "name": "Jordan James", "position": "RB", "school": "Oregon",
        "age": 21.3, "height_inches": 70, "weight_lbs": 213,
        "draft_class_year": 2026, "early_declare": True,
        "seasons": [
            {"season": 2025, "games_played": 13, "rush_attempts": 255,
             "rush_yards": 1445, "rush_tds": 13, "receptions": 18,
             "receiving_yards": 142, "receiving_tds": 1,
             "yds_per_carry": 5.67, "dominator_rating": 0.45,
             "market_share_yards": 0.48, "market_share_tds": 0.52,
             "team": "Oregon", "conference": "Big Ten", "team_pass_rate": 0.59},
            {"season": 2024, "games_played": 14, "rush_attempts": 193,
             "rush_yards": 1138, "rush_tds": 11, "receptions": 12,
             "receiving_yards": 94, "receiving_tds": 0,
             "yds_per_carry": 5.90, "dominator_rating": 0.32,
             "team": "Oregon", "conference": "Pac-12", "team_pass_rate": 0.60},
        ],
        "athleticism": {"forty_yard": 4.49, "vertical_inches": 36.0,
                        "ras_score": 7.9},
    },
    # ── QBs ──────────────────────────────────────────────────────────────────
    {
        "player_id": "ROOKIE_2026_JALEN_MILROE",
        "name": "Jalen Milroe", "position": "QB", "school": "Alabama",
        "age": 22.1, "height_inches": 74, "weight_lbs": 225,
        "draft_class_year": 2026, "early_declare": True,
        "seasons": [
            {"season": 2025, "games_played": 13, "pass_yards": 3580,
             "pass_tds": 28, "pass_attempts": 385, "completions": 238,
             "interceptions": 9, "rush_yards": 718, "rush_tds": 12,
             "yds_per_attempt": 9.30, "completion_pct": 61.8,
             "td_int_ratio": 3.11, "dominator_rating": 0.58,
             "team": "Alabama", "conference": "SEC", "team_pass_rate": 0.56},
            {"season": 2024, "games_played": 13, "pass_yards": 2834,
             "pass_tds": 23, "pass_attempts": 319, "completions": 187,
             "interceptions": 6, "rush_yards": 531, "rush_tds": 12,
             "yds_per_attempt": 8.88, "completion_pct": 58.6,
             "td_int_ratio": 3.83,
             "team": "Alabama", "conference": "SEC", "team_pass_rate": 0.57},
        ],
        "athleticism": {"forty_yard": 4.52, "vertical_inches": 35.0,
                        "ras_score": 7.6},
    },
    {
        "player_id": "ROOKIE_2026_QUINN_EWERS",
        "name": "Quinn Ewers", "position": "QB", "school": "Texas",
        "age": 21.9, "height_inches": 75, "weight_lbs": 206,
        "draft_class_year": 2026, "early_declare": True,
        "seasons": [
            {"season": 2025, "games_played": 13, "pass_yards": 3845,
             "pass_tds": 32, "pass_attempts": 398, "completions": 268,
             "interceptions": 8, "rush_yards": 145, "rush_tds": 3,
             "yds_per_attempt": 9.66, "completion_pct": 67.3,
             "td_int_ratio": 4.00, "dominator_rating": 0.53,
             "team": "Texas", "conference": "SEC", "team_pass_rate": 0.59},
            {"season": 2024, "games_played": 11, "pass_yards": 2665,
             "pass_tds": 22, "pass_attempts": 297, "completions": 194,
             "interceptions": 6,
             "yds_per_attempt": 8.97, "completion_pct": 65.3,
             "td_int_ratio": 3.67,
             "team": "Texas", "conference": "Big 12", "team_pass_rate": 0.60},
        ],
        "athleticism": {"forty_yard": 4.74, "ras_score": 4.8},
    },
    # ── TEs ──────────────────────────────────────────────────────────────────
    {
        "player_id": "ROOKIE_2026_COLSTON_LOVELAND",
        "name": "Colston Loveland", "position": "TE", "school": "Michigan",
        "age": 23.0, "height_inches": 77, "weight_lbs": 248,
        "draft_class_year": 2026,
        "seasons": [
            {"season": 2025, "games_played": 13, "receptions": 64, "targets": 85,
             "receiving_yards": 715, "receiving_tds": 7,
             "yds_per_reception": 11.2, "dominator_rating": 0.24,
             "market_share_yards": 0.26, "market_share_tds": 0.32,
             "team": "Michigan", "conference": "Big Ten", "team_pass_rate": 0.47},
            {"season": 2024, "games_played": 13, "receptions": 56, "targets": 76,
             "receiving_yards": 582, "receiving_tds": 8,
             "yds_per_reception": 10.4, "dominator_rating": 0.18,
             "team": "Michigan", "conference": "Big Ten", "team_pass_rate": 0.45},
        ],
        "athleticism": {"forty_yard": 4.59, "ras_score": 8.1},
    },
    {
        "player_id": "ROOKIE_2026_HAROLD_FANNIN",
        "name": "Harold Fannin Jr", "position": "TE", "school": "Bowling Green",
        "age": 22.4, "height_inches": 77, "weight_lbs": 252,
        "draft_class_year": 2026, "early_declare": True,
        "seasons": [
            {"season": 2025, "games_played": 12, "receptions": 72, "targets": 96,
             "receiving_yards": 956, "receiving_tds": 10,
             "yds_per_reception": 13.3, "dominator_rating": 0.38,
             "market_share_yards": 0.40, "market_share_tds": 0.48,
             "team": "Bowling Green", "conference": "MAC", "team_pass_rate": 0.64},
            {"season": 2024, "games_played": 13, "receptions": 58, "targets": 78,
             "receiving_yards": 762, "receiving_tds": 8,
             "yds_per_reception": 13.1, "dominator_rating": 0.32,
             "team": "Bowling Green", "conference": "MAC", "team_pass_rate": 0.62},
        ],
        "athleticism": {"forty_yard": 4.61, "ras_score": 7.6},
    },
]

SEED_PROSPECTS_2025: List[Dict[str, Any]] = [
    # 2025 draft class — already played their rookie season
    {
        "player_id": "ROOKIE_2025_TRAVIS_HUNTER",
        "name": "Travis Hunter", "position": "WR", "school": "Colorado",
        "age": 21.4, "height_inches": 72, "weight_lbs": 188,
        "draft_class_year": 2025, "early_declare": True,
        "seasons": [
            {"season": 2024, "games_played": 12, "receptions": 96, "targets": 130,
             "receiving_yards": 1258, "receiving_tds": 15,
             "yds_per_reception": 13.1, "dominator_rating": 0.35,
             "market_share_yards": 0.36, "market_share_tds": 0.55,
             "team": "Colorado", "conference": "Big 12", "team_pass_rate": 0.63},
        ],
        "athleticism": {"forty_yard": 4.38, "ras_score": 8.4},
    },
    {
        "player_id": "ROOKIE_2025_ASHTON_JEANTY",
        "name": "Ashton Jeanty", "position": "RB", "school": "Boise State",
        "age": 21.3, "height_inches": 68, "weight_lbs": 215,
        "draft_class_year": 2025, "early_declare": True,
        "seasons": [
            {"season": 2024, "games_played": 13, "rush_attempts": 374,
             "rush_yards": 2601, "rush_tds": 29, "receptions": 16,
             "receiving_yards": 149, "receiving_tds": 2,
             "yds_per_carry": 6.96, "dominator_rating": 0.72,
             "market_share_yards": 0.73, "market_share_tds": 0.85,
             "team": "Boise State", "conference": "Mountain West", "team_pass_rate": 0.38},
        ],
        "athleticism": {"forty_yard": 4.48, "ras_score": 8.6},
    },
    {
        "player_id": "ROOKIE_2025_CAM_WARD",
        "name": "Cam Ward", "position": "QB", "school": "Miami",
        "age": 23.2, "height_inches": 74, "weight_lbs": 220,
        "draft_class_year": 2025, "early_declare": True,
        "seasons": [
            {"season": 2024, "games_played": 13, "pass_yards": 4313,
             "pass_tds": 39, "pass_attempts": 444, "completions": 316,
             "interceptions": 7,
             "yds_per_attempt": 9.71, "completion_pct": 71.2,
             "td_int_ratio": 5.57,
             "team": "Miami", "conference": "ACC", "team_pass_rate": 0.62},
        ],
        "athleticism": {"forty_yard": 4.62, "ras_score": 5.8},
    },
    {
        "player_id": "ROOKIE_2025_TETAIROA_MCMILLAN",
        "name": "Tetairoa McMillan", "position": "WR", "school": "Arizona",
        "age": 21.2, "height_inches": 77, "weight_lbs": 219,
        "draft_class_year": 2025, "early_declare": True,
        "seasons": [
            {"season": 2024, "games_played": 12, "receptions": 84, "targets": 118,
             "receiving_yards": 1319, "receiving_tds": 8,
             "yds_per_reception": 15.7, "dominator_rating": 0.38,
             "market_share_yards": 0.40, "market_share_tds": 0.44,
             "team": "Arizona", "conference": "Big 12", "team_pass_rate": 0.58},
        ],
        "athleticism": {"forty_yard": 4.45, "ras_score": 9.2},
    },
    {
        "player_id": "ROOKIE_2025_TYLER_WARREN",
        "name": "Tyler Warren", "position": "TE", "school": "Penn State",
        "age": 22.6, "height_inches": 77, "weight_lbs": 258,
        "draft_class_year": 2025, "early_declare": True,
        "seasons": [
            {"season": 2024, "games_played": 14, "receptions": 104, "targets": 138,
             "receiving_yards": 1233, "receiving_tds": 8,
             "yds_per_reception": 11.9, "dominator_rating": 0.30,
             "team": "Penn State", "conference": "Big Ten", "team_pass_rate": 0.58},
        ],
        "athleticism": {"forty_yard": 4.65, "ras_score": 7.4},
    },
]

SEED_BY_YEAR: Dict[int, List[Dict]] = {
    2025: SEED_PROSPECTS_2025,
    2026: SEED_PROSPECTS_2026,
}


def get_seed_prospects(draft_year: int) -> List[Dict[str, Any]]:
    """Return the curated seed dataset for a given draft year."""
    return SEED_BY_YEAR.get(draft_year, [])


def normalize_prospect(raw: Dict[str, Any]) -> Dict[str, Any]:
    """
    Ensure every field expected downstream is present (with None defaults).
    Works on both live CFBD output and seed data.
    """
    p = dict(raw)
    p.setdefault("sleeper_id", None)
    p.setdefault("hometown", None)
    p.setdefault("state", None)
    p.setdefault("transfer_history", None)
    p.setdefault("headshot_url", None)
    p.setdefault("early_declare", False)
    p.setdefault("seasons", [])
    p.setdefault("athleticism", {})

    for s in p["seasons"]:
        s.setdefault("games_played", None)
        for fld in ("pass_yards", "pass_tds", "pass_attempts", "completions", "interceptions",
                    "rush_attempts", "rush_yards", "rush_tds",
                    "receptions", "targets", "receiving_yards", "receiving_tds",
                    "dominator_rating", "market_share_yards", "market_share_tds",
                    "yds_per_carry", "yds_per_reception", "yds_per_attempt",
                    "completion_pct", "td_int_ratio", "team_pass_rate",
                    "team_total_yards", "team_total_tds", "team", "conference"):
            s.setdefault(fld, None)

    return p


def load_prospects_for_year(draft_year: int) -> List[Dict[str, Any]]:
    """
    Entry point: return normalized prospect list for `draft_year`.

    Merges up to four sources in priority order:

    1. Sportradar (SPORTRADAR_API_KEY)  → player list, bio, height, weight,
                                          school, conference, experience
    2. NFLVerse combine.csv (no key)    → forty_yard, vertical, broad_jump,
                                          bench_reps, three_cone, shuttle
    3. CFBD (CFBD_API_KEY)              → per-season college stats,
                                          games_played, market share,
                                          dominator rating
    4. Seed dataset (always available)  → fallback for any missing field;
                                          used entirely when Sportradar is
                                          not configured

    Age: taken from NFLVerse if height/weight match found, otherwise
         estimated from Sportradar experience (SR/JR/SO/FR), otherwise
         seed value, otherwise None.
    """
    seed = get_seed_prospects(draft_year)

    # ── No Sportradar key — use seed only ────────────────────────────────────
    if not SPORTRADAR_KEY:
        log.info("[ingestion] Using seed data: %d prospects for %d", len(seed), draft_year)
        return [normalize_prospect(p) for p in seed]

    # ── Fetch from all live sources in parallel ───────────────────────────────
    sr_prospects = fetch_sportradar_prospects(draft_year)
    if not sr_prospects:
        log.warning("[ingestion] Sportradar returned no data for %d — using seed", draft_year)
        return [normalize_prospect(p) for p in seed]

    combine_data = fetch_nflverse_combine(draft_year)         # always attempted
    cfbd_stats   = fetch_cfbd_college_stats(draft_year)       # only if CFBD_KEY set

    seed_by_name = {p["name"].lower(): p for p in seed}

    enriched: List[Dict] = []
    for sr in sr_prospects:
        name_key = sr["name"].lower()
        seed_p   = seed_by_name.get(name_key)
        nflv     = combine_data.get(name_key, {})
        cfbd_seasons = cfbd_stats.get(name_key)

        # Start with Sportradar bio
        p: Dict[str, Any] = {
            "player_id":        sr["player_id"],
            "name":             sr["name"],
            "position":         sr["position"],
            "school":           sr["school"] or (seed_p or {}).get("school"),
            "height_inches":    sr["height_inches"] or nflv.get("height_inches") or (seed_p or {}).get("height_inches"),
            "weight_lbs":       sr["weight_lbs"]    or nflv.get("weight_lbs")    or (seed_p or {}).get("weight_lbs"),
            "state":            sr.get("state"),
            "draft_class_year": draft_year,
            "early_declare":    (seed_p or {}).get("early_declare", False),
            "transfer_history": (seed_p or {}).get("transfer_history"),
            "source":           "sportradar",
        }

        # ── Age ──────────────────────────────────────────────────────────────
        p["age"] = (
            (seed_p or {}).get("age") or
            _estimate_age(sr.get("_experience"), draft_year)
        )

        # ── Athleticism / combine ─────────────────────────────────────────────
        seed_ath  = (seed_p or {}).get("athleticism") or {}
        nflv_ath  = nflv.get("athleticism") or {}
        # NFLVerse is authoritative for combine; seed fills any remaining gaps
        p["athleticism"] = {**seed_ath, **nflv_ath}

        # ── College stats ─────────────────────────────────────────────────────
        if cfbd_seasons:
            p["seasons"] = cfbd_seasons
        elif seed_p and seed_p.get("seasons"):
            p["seasons"] = seed_p["seasons"]
        elif sr.get("_conference"):
            # New player with no stats — inject conference so competition score works
            p["seasons"] = [{"season": draft_year - 1, "conference": sr["_conference"]}]
        else:
            p["seasons"] = []

        enriched.append(p)

    # Seed players not returned by Sportradar
    sr_names  = {p["name"].lower() for p in sr_prospects}
    seed_only = [p for p in seed if p["name"].lower() not in sr_names]
    if seed_only:
        log.info("[ingestion] %d seed prospects not in Sportradar — appending", len(seed_only))

    final = enriched + seed_only
    log.info(
        "[ingestion] %d total prospects for %d  "
        "(Sportradar: %d | combine: %d matched | CFBD stats: %d matched | seed-only: %d)",
        len(final), draft_year,
        len(enriched), sum(1 for p in enriched if p.get("athleticism")),
        sum(1 for p in enriched if cfbd_stats.get(p["name"].lower())),
        len(seed_only),
    )
    return [normalize_prospect(p) for p in final]
