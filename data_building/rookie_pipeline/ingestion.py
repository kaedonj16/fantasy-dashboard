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
        "age":              None,   # not in this endpoint; filled from seed on merge
        "height_inches":    _safe_int(raw.get("height")),
        "weight_lbs":       _safe_int(raw.get("weight")),
        "state":            state,
        "draft_class_year": draft_year,
        "early_declare":    False,
        "seasons":          [],     # not in this endpoint; filled from seed on merge
        "athleticism":      {},     # not in this endpoint; filled from seed on merge
        "source":           "sportradar",
        # Attach raw conference name so competition scoring works for new players
        "_conference":      conference,
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

    When SPORTRADAR_API_KEY is set:
      - Fetches the official prospect list from Sportradar, which provides
        name, position, height, weight, school, and conference.
      - For any player whose name matches a seed entry, the seed's age,
        college seasons, and athleticism/combine data are merged in, since
        the Sportradar prospects endpoint does not include those fields.
      - Players in Sportradar but not in the seed get scored on bio alone
        (stats/combine default to neutral).
      - Seed players not returned by Sportradar are appended so the page
        always has a complete roster.

    Falls back entirely to seed data when no API key is configured.
    """
    seed = get_seed_prospects(draft_year)

    if SPORTRADAR_KEY:
        live = fetch_sportradar_prospects(draft_year)
        if live:
            # Build a name-keyed lookup from seed for enrichment
            seed_by_name = {p["name"].lower(): p for p in seed}

            enriched = []
            for sr in live:
                seed_match = seed_by_name.get(sr["name"].lower())
                if seed_match:
                    # Prefer Sportradar bio fields; fill gaps from seed
                    merged = dict(seed_match)
                    merged["height_inches"] = sr["height_inches"] or seed_match.get("height_inches")
                    merged["weight_lbs"]    = sr["weight_lbs"]    or seed_match.get("weight_lbs")
                    merged["school"]        = sr["school"]        or seed_match.get("school")
                    merged["source"]        = "sportradar"
                else:
                    # New player not in seed — use Sportradar bio, inject conference
                    # into each season placeholder so competition scoring works
                    merged = dict(sr)
                    if sr.get("_conference"):
                        merged["seasons"] = [{
                            "season": draft_year - 1,
                            "conference": sr["_conference"],
                        }]
                # Drop internal keys
                merged.pop("_conference", None)
                merged.pop("_draft_round", None)
                merged.pop("_draft_pick", None)
                enriched.append(merged)

            # Keep seed players not returned by Sportradar
            sr_names   = {p["name"].lower() for p in live}
            seed_only  = [p for p in seed if p["name"].lower() not in sr_names]
            if seed_only:
                log.info("[ingestion] %d seed prospects not in Sportradar — keeping seed data",
                         len(seed_only))

            merged_list = enriched + seed_only
            log.info("[ingestion] %d total prospects for %d (%d Sportradar, %d seed-only)",
                     len(merged_list), draft_year, len(enriched), len(seed_only))
            return [normalize_prospect(p) for p in merged_list]

        log.warning("[ingestion] Sportradar returned no data for %d — using seed", draft_year)

    log.info("[ingestion] Using seed data: %d prospects for %d", len(seed), draft_year)
    return [normalize_prospect(p) for p in seed]
