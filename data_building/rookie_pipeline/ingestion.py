"""
Rookie prospect data ingestion.

Primary source: CollegeFootballData.com API (free, requires CFBD_API_KEY env var).
Fallback:       Curated seed dataset for 2025 and 2026 draft classes so the
                page works immediately without any external credentials.

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

CFBD_BASE = "https://api.collegefootballdata.com"
CFBD_KEY  = os.getenv("CFBD_API_KEY", "")

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


def _cfbd_headers() -> Dict[str, str]:
    headers = {"Accept": "application/json"}
    if CFBD_KEY:
        headers["Authorization"] = f"Bearer {CFBD_KEY}"
    return headers


def _cfbd_get(path: str, params: Dict[str, Any] = None, retries: int = 3) -> Optional[Any]:
    """GET from CFBD API with retry/backoff. Returns None on failure."""
    url = f"{CFBD_BASE}{path}"
    for attempt in range(retries):
        try:
            resp = requests.get(url, headers=_cfbd_headers(), params=params or {}, timeout=15)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as exc:
            wait = 2 ** attempt
            log.warning("[cfbd] %s attempt %d failed: %s — retrying in %ds", path, attempt + 1, exc, wait)
            time.sleep(wait)
    log.error("[cfbd] %s failed after %d attempts", path, retries)
    return None


# ─────────────────────────────────────────────────────────────────────────────
# CFBD ingestion
# ─────────────────────────────────────────────────────────────────────────────

def fetch_cfbd_player_season_stats(year: int) -> List[Dict[str, Any]]:
    """
    Fetch all player season stat lines from CFBD for `year`.
    Returns a flat list of stat dicts keyed by athlete name + team.
    """
    if not CFBD_KEY:
        log.info("[cfbd] No CFBD_API_KEY set — skipping live fetch for %d", year)
        return []

    data = _cfbd_get("/stats/player/season", {"year": year, "seasonType": "regular"})
    if not data:
        return []

    log.info("[cfbd] Fetched %d stat lines for %d", len(data), year)
    return data


def fetch_cfbd_team_stats(year: int) -> Dict[str, Dict[str, Any]]:
    """
    Returns {team_name: {total_yards, total_tds, pass_rate, ...}}.
    Used to compute market share metrics.
    """
    if not CFBD_KEY:
        return {}

    data = _cfbd_get("/stats/season", {"year": year, "seasonType": "regular"})
    if not data:
        return {}

    teams: Dict[str, Dict[str, Any]] = {}
    for row in data:
        team = row.get("team", "")
        stat = row.get("statName", "")
        val  = _safe(row.get("statValue"), 0)
        entry = teams.setdefault(team, {})
        entry[stat] = val

    # Compute pass rate per team
    for team, stats in teams.items():
        pass_att = stats.get("passAttempts", 0) or 0
        rush_att = stats.get("rushingAttempts", 0) or 0
        total    = pass_att + rush_att
        stats["pass_rate"] = round(pass_att / total, 3) if total > 0 else 0.5

    return teams


def _build_season_row(raw_stats: List[Dict], team_stats: Dict[str, Any], season: int) -> Dict:
    """
    Fold a list of raw CFBD stat rows for one player-season into a single dict.
    """
    row: Dict[str, Any] = {
        "season": season,
        "games_played": None,
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
        key = s.get("statName", "")
        val = _safe_int(s.get("stat"), 0)
        if key in stat_map:
            row[stat_map[key]] = (row.get(stat_map[key]) or 0) + (val or 0)
        if not row["team"]:
            row["team"] = s.get("team")
        if not row["conference"]:
            row["conference"] = s.get("conference")

    team = row.get("team", "")
    ts   = team_stats.get(team, {})

    # Derived metrics
    gp = row["games_played"] or 1
    rush_att = row["rush_attempts"] or 0
    rush_yds = row["rush_yards"] or 0
    rec_yds  = row["receiving_yards"] or 0
    rec_tds  = row["receiving_tds"] or 0
    rush_tds = row["rush_tds"] or 0
    pass_att = row["pass_attempts"] or 0
    pass_yds = row["pass_yards"] or 0
    comp     = row["completions"] or 0

    row["yds_per_carry"]     = round(rush_yds / rush_att, 2) if rush_att > 0 else None
    row["yds_per_reception"] = round(rec_yds / max(row["receptions"] or 1, 1), 2) if rec_yds > 0 else None
    row["yds_per_attempt"]   = round(pass_yds / pass_att, 2) if pass_att > 0 else None
    row["completion_pct"]    = round(comp / pass_att * 100, 1) if pass_att > 0 else None
    ints = row["interceptions"] or 0
    row["td_int_ratio"]      = round(row["pass_tds"] / max(ints, 1), 2) if row["pass_tds"] else None

    # Market share
    team_total_yds = (ts.get("netPassingYards", 0) or 0) + (ts.get("rushingYards", 0) or 0)
    team_total_tds = (ts.get("passingTDs", 0) or 0) + (ts.get("rushingTDs", 0) or 0)
    player_yds = rec_yds + rush_yds
    player_tds = rec_tds + rush_tds

    row["market_share_yards"] = round(player_yds / team_total_yds, 3) if team_total_yds > 0 else None
    row["market_share_tds"]   = round(player_tds / team_total_tds, 3) if team_total_tds > 0 else None
    row["team_total_yards"]   = _safe_int(team_total_yds)
    row["team_total_tds"]     = _safe_int(team_total_tds)
    row["team_pass_rate"]     = ts.get("pass_rate")

    # Dominator rating = (player_yds / team_total_yds)*0.65 + (player_tds / team_total_tds)*0.35
    dom = 0.0
    if team_total_yds > 0:
        dom += (player_yds / team_total_yds) * 0.65
    if team_total_tds > 0:
        dom += (player_tds / team_total_tds) * 0.35
    row["dominator_rating"] = round(dom, 4) if (team_total_yds or team_total_tds) else None

    return row


def fetch_cfbd_prospects(draft_year: int, player_names: List[str]) -> List[Dict[str, Any]]:
    """
    For a list of prospect names, pull their last 2-3 college seasons from CFBD.
    Returns normalized player dicts ready for DB insertion.
    """
    if not CFBD_KEY:
        log.info("[cfbd] No API key — returning empty for live fetch")
        return []

    results = []
    years_to_check = [draft_year - 1, draft_year - 2, draft_year - 3]

    # Pre-fetch team stats for all years
    all_team_stats: Dict[int, Dict] = {}
    for yr in years_to_check:
        all_team_stats[yr] = fetch_cfbd_team_stats(yr)

    # Pre-fetch all player stat lines for each year
    all_season_stats: Dict[int, List[Dict]] = {}
    for yr in years_to_check:
        all_season_stats[yr] = fetch_cfbd_player_season_stats(yr)

    for name in player_names:
        name_lower = name.lower()
        player_seasons = []

        for yr in years_to_check:
            stats_for_year = [
                s for s in all_season_stats.get(yr, [])
                if (s.get("player") or "").lower() == name_lower
            ]
            if stats_for_year:
                season_row = _build_season_row(stats_for_year, all_team_stats.get(yr, {}), yr)
                player_seasons.append(season_row)

        if player_seasons:
            player_id = f"ROOKIE_{draft_year}_{_slug(name)}"
            results.append({
                "player_id": player_id,
                "name": name,
                "draft_class_year": draft_year,
                "seasons": player_seasons,
                "source": "cfbd",
            })

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
    Uses live CFBD data when API key is set; otherwise falls back to seed data.
    """
    if CFBD_KEY:
        seed = get_seed_prospects(draft_year)
        names = [p["name"] for p in seed]
        live  = fetch_cfbd_prospects(draft_year, names)
        if live:
            log.info("[ingestion] Loaded %d prospects via CFBD API for %d", len(live), draft_year)
            return [normalize_prospect(p) for p in live]
        log.warning("[ingestion] CFBD returned no data, falling back to seed for %d", draft_year)

    seed = get_seed_prospects(draft_year)
    log.info("[ingestion] Using seed data: %d prospects for %d", len(seed), draft_year)
    return [normalize_prospect(p) for p in seed]
