"""
Rookie prospect data ingestion.

Primary source: Sportradar NFL API (requires SPORTRADAR_API_KEY env var).
Fallback:       Curated seed dataset for 2025 and 2026 draft classes so the
                page works immediately without any external credentials.

Sportradar endpoint:
    GET https://api.sportradar.com/nfl/official/{version}/en/prospects.json
        ?api_key={key}&year={draft_year}

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
from datetime import date
from typing import Any, Dict, List, Optional

import requests

log = logging.getLogger(__name__)

SPORTRADAR_KEY     = os.getenv("SPORTRADAR_API_KEY", "")
SPORTRADAR_BASE    = "https://api.sportradar.com/nfl/official"
SPORTRADAR_VERSION = os.getenv("SPORTRADAR_NFL_VERSION", "trial/v7")

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


def _sportradar_get(path: str, params: Dict[str, Any] = None, retries: int = 3) -> Optional[Any]:
    """GET from Sportradar API with retry/backoff. Returns None on failure."""
    url = f"{SPORTRADAR_BASE}/{SPORTRADAR_VERSION}/en/{path}"
    p = dict(params or {})
    p["api_key"] = SPORTRADAR_KEY
    for attempt in range(retries):
        try:
            resp = requests.get(url, params=p, timeout=20)
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

def _age_at_draft(birth_date: Optional[str], draft_year: int) -> Optional[float]:
    """Calculate age at the approximate draft date (late April of draft_year)."""
    if not birth_date:
        return None
    try:
        bdate = date.fromisoformat(str(birth_date)[:10])
        draft_date = date(draft_year, 4, 25)
        return round((draft_date - bdate).days / 365.25, 1)
    except (ValueError, TypeError):
        return None


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


def _parse_sportradar_season(season_obj: Dict, position: str) -> Dict:
    """
    Convert one Sportradar season statistics object into our normalized format.

    Sportradar nests stats under keys like 'passing', 'rushing', 'receiving'
    within the season object (or a nested 'statistics' sub-object).
    """
    # Stats may live directly on season_obj or under a 'statistics' sub-key
    stats = season_obj.get("statistics") or season_obj

    passing   = stats.get("passing")   or {}
    rushing   = stats.get("rushing")   or {}
    receiving = stats.get("receiving") or {}

    pass_att  = _safe_int(passing.get("attempts")      or passing.get("att"),         0)
    pass_yds  = _safe_int(passing.get("yards")         or passing.get("pass_yards"),  0)
    pass_tds  = _safe_int(passing.get("touchdowns")    or passing.get("pass_tds"),    0)
    comp      = _safe_int(passing.get("completions")   or passing.get("comp"),        0)
    ints      = _safe_int(passing.get("interceptions") or passing.get("int"),         0)

    rush_att  = _safe_int(rushing.get("attempts")   or rushing.get("att"),        0)
    rush_yds  = _safe_int(rushing.get("yards")      or rushing.get("rush_yards"), 0)
    rush_tds  = _safe_int(rushing.get("touchdowns") or rushing.get("rush_tds"),   0)

    rec_yds   = _safe_int(receiving.get("yards")       or receiving.get("rec_yards"), 0)
    rec_tds   = _safe_int(receiving.get("touchdowns")  or receiving.get("rec_tds"),   0)
    recs      = _safe_int(receiving.get("receptions")  or receiving.get("rec"),       0)
    targets   = _safe_int(receiving.get("targets"),                                   0)

    # games_played: may sit at season level or under stats
    gp = _safe_int(
        season_obj.get("games_played") or season_obj.get("games") or
        stats.get("games_played")      or stats.get("games")
    )

    # Team / conference
    team_obj = season_obj.get("team") or {}
    team = team_obj.get("name") if isinstance(team_obj, dict) else team_obj
    conf_obj = season_obj.get("conference") or {}
    conference = conf_obj.get("name") if isinstance(conf_obj, dict) else conf_obj

    row: Dict[str, Any] = {
        "season":          _safe_int(season_obj.get("season") or season_obj.get("year")),
        "games_played":    gp,
        "pass_yards":      pass_yds,
        "pass_tds":        pass_tds,
        "pass_attempts":   pass_att,
        "completions":     comp,
        "interceptions":   ints,
        "rush_attempts":   rush_att,
        "rush_yards":      rush_yds,
        "rush_tds":        rush_tds,
        "receptions":      recs,
        "targets":         targets,
        "receiving_yards": rec_yds,
        "receiving_tds":   rec_tds,
        "team":            team,
        "conference":      conference,
        # Sportradar doesn't supply team-level totals needed for market share /
        # dominator rating, so these remain None until enriched externally.
        "dominator_rating":   None,
        "market_share_yards": None,
        "market_share_tds":   None,
        "team_total_yards":   None,
        "team_total_tds":     None,
        "team_pass_rate":     None,
    }

    # Derived efficiency metrics
    row["yds_per_carry"]     = round(rush_yds / rush_att, 2) if rush_att > 0 else None
    row["yds_per_reception"] = round(rec_yds / max(recs, 1), 2) if rec_yds > 0 else None
    row["yds_per_attempt"]   = round(pass_yds / pass_att, 2) if pass_att > 0 else None
    row["completion_pct"]    = round(comp / pass_att * 100, 1) if pass_att > 0 else None
    row["td_int_ratio"]      = round(pass_tds / max(ints, 1), 2) if pass_tds else None

    return row


def _parse_sportradar_prospect(raw: Dict, draft_year: int) -> Optional[Dict]:
    """Convert a single Sportradar prospect object to our normalized format."""
    # Name — try full name first, fall back to first+last
    name = (
        raw.get("name_full") or
        " ".join(filter(None, [raw.get("name_first"), raw.get("name_last")])) or
        raw.get("name", "")
    ).strip()
    if not name:
        return None

    position = (raw.get("position") or "").upper()

    school_obj = raw.get("school") or {}
    school = school_obj.get("name") if isinstance(school_obj, dict) else school_obj

    age = _age_at_draft(raw.get("birth_date"), draft_year)

    # Combine / pro-day measurements
    combine_obj = raw.get("combine") or {}
    athleticism: Dict[str, Any] = {}
    mapping = {
        "forty_yard":      ("forty_yard_dash",       _safe),
        "vertical_inches": ("vertical_jump",          _safe),
        "broad_jump_in":   ("broad_jump",             _safe_int),
        "bench_reps":      ("bench_press",            _safe_int),
        "three_cone":      ("three_cone_drill",       _safe),
        "short_shuttle":   ("twenty_yard_shuttle",    _safe),
    }
    for our_key, (sr_key, cast) in mapping.items():
        val = combine_obj.get(sr_key)
        if val is not None:
            athleticism[our_key] = cast(val)

    # College seasons
    # Sportradar may return seasons under: statistics.seasons[], seasons[], or
    # a top-level list keyed by year.
    stats_root  = raw.get("statistics") or {}
    season_list = (
        stats_root.get("seasons") or
        raw.get("seasons") or
        (stats_root if isinstance(stats_root, list) else [])
    )

    seasons = []
    for s in season_list:
        yr = _safe_int(s.get("season") or s.get("year"))
        if not yr:
            continue
        # Keep only the last 3 college seasons before the draft
        if yr < draft_year - 3 or yr >= draft_year:
            continue
        season_row = _parse_sportradar_season(s, position)
        season_row["season"] = yr
        seasons.append(season_row)

    # Draft info (populated after the draft)
    draft_obj = raw.get("draft") or {}

    return {
        "player_id":        f"ROOKIE_{draft_year}_{_slug(name)}",
        "name":             name,
        "position":         position,
        "school":           school,
        "age":              age,
        "height_inches":    _parse_height(raw.get("height")),
        "weight_lbs":       _safe_int(raw.get("weight")),
        "hometown":         raw.get("hometown"),
        "state":            raw.get("birth_place", {}).get("state") if isinstance(raw.get("birth_place"), dict) else None,
        "draft_class_year": draft_year,
        "early_declare":    False,
        "seasons":          seasons,
        "athleticism":      athleticism,
        "source":           "sportradar",
        # Preserved for mock consensus enrichment downstream
        "_draft_round":     _safe_int(draft_obj.get("round")),
        "_draft_pick":      _safe_int(draft_obj.get("pick")),
    }


def fetch_sportradar_prospects(draft_year: int) -> List[Dict[str, Any]]:
    """
    Fetch the full NFL prospect list for `draft_year` from Sportradar.

    Tries two URL patterns:
      1. /prospects.json?year={draft_year}           (year-filtered)
      2. /draft/{draft_year}/prospects.json          (draft-scoped path)

    Returns a list of normalized prospect dicts.
    """
    if not SPORTRADAR_KEY:
        log.info("[sportradar] No SPORTRADAR_API_KEY set — skipping live fetch")
        return []

    data = _sportradar_get("prospects.json", {"year": draft_year})
    if not data:
        data = _sportradar_get(f"draft/{draft_year}/prospects.json")
    if not data:
        log.warning("[sportradar] No prospect data returned for %d", draft_year)
        return []

    # Response is either {"prospects": [...]} or a bare list
    raw_list = data.get("prospects") or (data if isinstance(data, list) else [])
    if not raw_list:
        log.warning("[sportradar] Empty prospects list for %d", draft_year)
        return []

    results = []
    for raw in raw_list:
        prospect = _parse_sportradar_prospect(raw, draft_year)
        if prospect:
            results.append(prospect)

    log.info("[sportradar] Parsed %d prospects for %d", len(results), draft_year)
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
      - Calls the Sportradar NFL prospects endpoint for the full class with
        bio (name, position, school, height, weight, age/DOB), combine
        measurements (40-time, vertical, broad jump, bench, 3-cone, shuttle),
        and multi-season college stats.
      - Any player returned by Sportradar but with no college stats still
        appears in the list — scoring falls back to graceful defaults.
      - Seed prospects not found in the Sportradar response are appended so
        the page always has a complete roster even during API outages.

    Falls back entirely to seed data when no API key is configured.
    """
    seed = get_seed_prospects(draft_year)

    if SPORTRADAR_KEY:
        live = fetch_sportradar_prospects(draft_year)
        if live:
            live_ids = {p["player_id"] for p in live}
            seed_only = [p for p in seed if p["player_id"] not in live_ids]
            if seed_only:
                log.info("[ingestion] %d seed prospects not in Sportradar response — keeping seed data",
                         len(seed_only))
            merged = live + seed_only
            log.info("[ingestion] %d total prospects for %d (%d Sportradar, %d seed)",
                     len(merged), draft_year, len(live), len(seed_only))
            return [normalize_prospect(p) for p in merged]
        log.warning("[ingestion] Sportradar returned no data for %d — using seed", draft_year)

    log.info("[ingestion] Using seed data: %d prospects for %d", len(seed), draft_year)
    return [normalize_prospect(p) for p in seed]
