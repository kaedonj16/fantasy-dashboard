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
import os
import re
import time
import warnings
from typing import Any, Dict, List, Optional

import requests

# Suppress urllib3 SSL warning for LibreSSL compatibility
warnings.filterwarnings('ignore', message='.*urllib3 v2 only supports OpenSSL.*')

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
    
    print(f"[sportradar] Request: GET {url} (access={SPORTRADAR_ACCESS}, key={SPORTRADAR_KEY[:8] if SPORTRADAR_KEY else 'NONE'}...)")
    
    last_error = None
    for attempt in range(retries):
        try:
            resp = requests.get(url, headers=headers, params=params, timeout=20)
            print(f"[sportradar] Response: HTTP {resp.status_code} for {path}")
            resp.raise_for_status()
            return resp.json()
        except requests.Timeout as exc:
            last_error = f"Timeout after 20s"
            wait = 2 ** attempt
            print(f"[sportradar] {path} attempt {attempt + 1}/{retries}: TIMEOUT — retrying in {wait}s")
            time.sleep(wait)
        except requests.HTTPError as exc:
            last_error = f"HTTP {exc.response.status_code}: {exc.response.reason}"
            if exc.response.status_code == 401:
                print(f"[sportradar] {path} FAILED: Authentication error (401) — check SPORTRADAR_API_KEY")
                return None
            elif exc.response.status_code == 403:
                print(f"[sportradar] {path} FAILED: Forbidden (403) — check API key permissions or access level (current: {SPORTRADAR_ACCESS})")
                return None
            elif exc.response.status_code == 404:
                print(f"[sportradar] {path} FAILED: Not found (404) — invalid path or year")
                return None
            elif exc.response.status_code == 429:
                wait = 2 ** attempt
                print(f"[sportradar] {path} attempt {attempt + 1}/{retries}: RATE LIMITED (429) — retrying in {wait}s")
                time.sleep(wait)
            else:
                wait = 2 ** attempt
                print(f"[sportradar] {path} attempt {attempt + 1}/{retries}: HTTP {exc.response.status_code} — retrying in {wait}s")
                time.sleep(wait)
        except requests.RequestException as exc:
            last_error = f"Request failed: {type(exc).__name__}: {exc}"
            wait = 2 ** attempt
            print(f"[sportradar] {path} attempt {attempt + 1}/{retries}: {type(exc).__name__} — retrying in {wait}s")
            time.sleep(wait)
        except Exception as exc:
            last_error = f"Unexpected error: {type(exc).__name__}: {exc}"
            print(f"[sportradar] {path} attempt {attempt + 1}/{retries}: UNEXPECTED ERROR — {last_error}")
            return None
    
    print(f"[sportradar] {path} FAILED after {retries} attempts — Last error: {last_error}")
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
    print(f"[nflverse] Starting combine data fetch for draft year {draft_year}")
    print(f"[nflverse] URL: {_NFLVERSE_COMBINE_URL}")
    
    try:
        print("[nflverse] Downloading combine.csv from NFLVerse")
        resp = requests.get(_NFLVERSE_COMBINE_URL, timeout=30)
        resp.raise_for_status()
        print(f"[nflverse] Downloaded {len(resp.text)} bytes")
    except requests.Timeout:
        print("[nflverse] FAILED: Download timeout after 30s — NFLVerse may be slow or unreachable")
        return {}
    except requests.HTTPError as exc:
        print(f"[nflverse] FAILED: HTTP {exc.response.status_code} ({exc.response.reason}) — URL may have changed or GitHub rate limit")
        return {}
    except requests.RequestException as exc:
        print(f"[nflverse] FAILED: Network error — {type(exc).__name__}: {exc}")
        return {}
    except Exception as exc:
        print(f"[nflverse] FAILED: Unexpected error — {type(exc).__name__}: {exc}")
        return {}

    results: Dict[str, Dict[str, Any]] = {}
    reader = csv.DictReader(io.StringIO(resp.text))
    total_rows = 0
    matching_rows = 0
    parse_errors = 0
    
    try:
        for row in reader:
            total_rows += 1
            # draft_year column is the year they were drafted
            try:
                row_year = _safe_int(row.get("draft_year") or row.get("season"))
            except Exception as exc:
                print(f"[nflverse] Parse error for draft_year in row {total_rows}: {exc}")
                parse_errors += 1
                continue
                
            if row_year != draft_year:
                continue

            matching_rows += 1
            name = (row.get("player_name") or "").strip().lower()
            if not name:
                print(f"[nflverse] Skipping row {total_rows}: no player_name")
                continue

            try:
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
                height = _parse_height(row.get("ht"))
                weight = _csv_int("wt")

                raw_bd = (row.get("birthdate") or row.get("birth_date") or "").strip()
                birthdate = raw_bd if raw_bd and raw_bd not in ("NA", "na", "") else None

                results[name] = {
                    "athleticism":    ath,
                    "height_inches":  height,
                    "weight_lbs":     weight,
                    "birthdate":      birthdate,
                }
            except Exception as exc:
                print(f"[nflverse] Parse error for player '{name or 'UNKNOWN'}' in row {total_rows}: {exc}")
                parse_errors += 1
                continue
    except csv.Error as exc:
        print(f"[nflverse] FAILED: CSV parsing error at row {total_rows} — {exc}")
        return results  # Return what we have so far
    except Exception as exc:
        print(f"[nflverse] FAILED: Unexpected error during CSV processing — {type(exc).__name__}: {exc}")
        return results

    print(f"[nflverse] Loaded combine data for {len(results)} prospects for {draft_year} ({total_rows} total rows, {parse_errors} parse errors)")
    print(f"[nflverse] Found {matching_rows} matching rows for {draft_year} in combine data")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Supplementary source 2 — CFBD college stats  (requires CFBD_API_KEY)
# Provides: per-season receiving/rushing/passing stats, games_played,
#           team, conference, market share, dominator rating
# ─────────────────────────────────────────────────────────────────────────────

def _cfbd_get(path: str, params: Dict[str, Any] = None, retries: int = 3) -> Optional[Any]:
    url = f"{CFBD_BASE}{path}"
    headers = {"Accept": "application/json", "Authorization": f"Bearer {CFBD_KEY}"}
    
    print(f"[cfbd] Request: GET {url} (params={params}, key={CFBD_KEY[:8] if CFBD_KEY else 'NONE'}...)")
    
    last_error = None
    for attempt in range(retries):
        try:
            resp = requests.get(url, headers=headers, params=params or {}, timeout=15)
            print(f"[cfbd] Response: HTTP {resp.status_code} for {path}")
            resp.raise_for_status()
            return resp.json()
        except requests.Timeout:
            last_error = "Timeout after 15s"
            wait = 2 ** attempt
            print(f"[cfbd] {path} attempt {attempt + 1}/{retries}: TIMEOUT — retrying in {wait}s")
            time.sleep(wait)
        except requests.HTTPError as exc:
            last_error = f"HTTP {exc.response.status_code}: {exc.response.reason}"
            if exc.response.status_code == 401:
                print(f"[cfbd] {path} FAILED: Authentication error (401) — check CFBD_API_KEY is valid")
                return None
            elif exc.response.status_code == 403:
                print(f"[cfbd] {path} FAILED: Forbidden (403) — API key may not have required permissions")
                return None
            elif exc.response.status_code == 404:
                print(f"[cfbd] {path} FAILED: Not found (404) — invalid endpoint or parameters: {params}")
                return None
            elif exc.response.status_code == 400:
                print(f"[cfbd] {path} FAILED: Bad Request (400) — invalid parameters: {params}")
                return None
            elif exc.response.status_code == 429:
                wait = 2 ** attempt * 2
                print(f"[cfbd] {path} attempt {attempt + 1}/{retries}: RATE LIMITED (429) — backing off {wait}s")
                time.sleep(wait)
            else:
                wait = 2 ** attempt
                print(f"[cfbd] {path} attempt {attempt + 1}/{retries}: HTTP {exc.response.status_code} — retrying in {wait}s")
                time.sleep(wait)
        except requests.RequestException as exc:
            last_error = f"Request failed: {type(exc).__name__}"
            wait = 2 ** attempt
            print(f"[cfbd] {path} attempt {attempt + 1}/{retries}: {type(exc).__name__} — retrying in {wait}s")
            time.sleep(wait)
        except Exception as exc:
            last_error = f"Unexpected error: {type(exc).__name__}: {exc}"
            print(f"[cfbd] {path} FAILED: UNEXPECTED ERROR — {last_error}")
            return None
    
    print(f"[cfbd] {path} FAILED after {retries} attempts — Last error: {last_error}")
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
    # CFBD API uses abbreviated statType values (e.g., "YDS", "TD", "REC")
    # NOT full names like "passingYards" or "receivingYards"
    stat_map = {
        # Passing stats (category: "passing")
        "YDS": "pass_yards",
        "TD": "pass_tds",
        "ATT": "pass_attempts",
        "COMPLETIONS": "completions",
        "INT": "interceptions",
        # Rushing stats (category: "rushing")
        "CAR": "rush_attempts",
        # Receiving stats (category: "receiving")
        "REC": "receptions",
    }

    # Track stats by category to handle duplicate stat types (e.g., "YDS" exists for passing/rushing/receiving)
    for s in raw_stats:
        category = s.get("category", "").lower()
        stat_type = s.get("statType", "")  # CFBD uses "statType" (camelCase), not "stat_type"
        stat_value = _safe_int(s.get("stat")) or 0

        # Map stat_type to our field, considering category for ambiguous types
        if stat_type == "YDS":
            if category == "passing":
                row["pass_yards"] = (row.get("pass_yards") or 0) + stat_value
            elif category == "rushing":
                row["rush_yards"] = (row.get("rush_yards") or 0) + stat_value
            elif category == "receiving":
                row["receiving_yards"] = (row.get("receiving_yards") or 0) + stat_value
        elif stat_type == "TD":
            if category == "passing":
                row["pass_tds"] = (row.get("pass_tds") or 0) + stat_value
            elif category == "rushing":
                row["rush_tds"] = (row.get("rush_tds") or 0) + stat_value
            elif category == "receiving":
                row["receiving_tds"] = (row.get("receiving_tds") or 0) + stat_value
        elif stat_type in stat_map:
            row[stat_map[stat_type]] = (row.get(stat_map[stat_type]) or 0) + stat_value

        # Always capture team/conference from any record
        row["team"]       = row["team"]       or s.get("team")
        row["conference"] = row["conference"] or s.get("conference")

    team_name = row.get("team", "")
    ts = team_stats.get(team_name, {})

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
    # print(row)
    return row


def fetch_cfbd_games_played(draft_year: int) -> Dict[str, Dict[int, int]]:
    """
    Fetch exact games played per player per season via CFBD /games/players endpoint.

    The endpoint requires `year` + `week` — it does NOT accept a season-level query
    with just `year` + `seasonType`.  We loop weeks 1-17 and aggregate unique game
    IDs per player, giving an exact games-played count rather than the default 12.

    Returns:
        {player_name_lower: {year: games_played_count}}

    ~51 API calls total (17 weeks × 3 years) with 0.2s rate-limit sleep ≈ ~10s.
    Empty weeks (bye weeks, season end) are skipped automatically.
    """
    if not CFBD_KEY:
        print("[cfbd_gp] No CFBD_API_KEY — skipping games-played lookup")
        return {}

    years   = [draft_year - 1, draft_year - 2, draft_year - 3]
    result: Dict[str, Dict[int, int]] = {}   # name_lower → {yr: count}

    for yr in years:
        print(f"[cfbd_gp] Fetching /games/players week-by-week for {yr}")
        player_games: Dict[str, set] = {}   # name_lower → set of game_ids
        empty_streak = 0

        for week in range(1, 18):   # weeks 1-17 covers FBS regular season
            try:
                data = _cfbd_get(
                    "/games/players",
                    {"year": yr, "week": week, "seasonType": "regular"},
                    retries=2,
                ) or []
            except Exception as exc:
                print(f"[cfbd_gp] Week {week} error: {exc}")
                continue

            if not data:
                empty_streak += 1
                if empty_streak >= 3:
                    # 3 consecutive empty weeks = season has ended; stop early
                    break
                continue

            empty_streak = 0

            # Each element in `data` is one game.  Structure:
            # { "id": <game_id>, "teams": [ { "categories": [
            #     { "name": "receiving", "types": [
            #         { "athletes": [ { "name": "Travis Hunter", ... } ] }
            #     ]}
            # ]}]}
            for game_obj in data:
                game_id = game_obj.get("id")
                if not game_id:
                    continue
                for team in (game_obj.get("teams") or []):
                    for category in (team.get("categories") or []):
                        for stat_type in (category.get("types") or []):
                            for athlete in (stat_type.get("athletes") or []):
                                a_name = (athlete.get("name") or "").lower().strip()
                                if a_name:
                                    player_games.setdefault(a_name, set()).add(game_id)

            time.sleep(0.2)   # light rate-limiting between week calls

        # Convert game-id sets → counts and store
        count = 0
        for player_name, game_ids in player_games.items():
            result.setdefault(player_name, {})[yr] = len(game_ids)
            count += 1

        print(f"[cfbd_gp] {yr}: resolved games-played for {count} players")

    print(f"[cfbd_gp] COMPLETE: {len(result)} players across {len(years)} seasons")
    return result


def fetch_cfbd_college_stats(draft_year: int) -> Dict[str, List[Dict]]:
    """
    Fetch college stats from CFBD for the 3 seasons before `draft_year`.
    Returns {player_name_lower: [season_dict, ...]} sorted oldest→newest.
    Requires CFBD_API_KEY env var; returns {} silently if not set.
    """
    print(f"[cfbd] Starting college stats fetch for draft year {draft_year}")
    
    if not CFBD_KEY:
        print("[cfbd] No CFBD_API_KEY set — skipping college stats")
        return {}

    years = [draft_year - 1, draft_year - 2, draft_year - 3]
    print(f"[cfbd] Fetching college stats for years: {years}")

    try:
        # Team season totals for market share / dominator calculation
        print("[cfbd] Fetching team season totals for market share calculation")
        team_stats: Dict[int, Dict] = {}
        for yr in years:
            print(f"[cfbd] Fetching team stats for {yr}")
            data = _cfbd_get("/stats/season", {"year": yr, "seasonType": "regular"})
            if not data:
                print(f"[cfbd] No team stats data for {yr}")
                team_stats[yr] = {}
                continue
            try:
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
                print(f"[cfbd] Loaded team stats for {yr}: {len(teams)} teams")
            except Exception as exc:
                print(f"[cfbd] ERROR processing team stats for {yr} — {type(exc).__name__}: {exc}")
                team_stats[yr] = {}

        # Games-played lookup from /games/players (separate endpoint)
        print("[cfbd] Fetching games played from /games/players endpoint")
        try:
            games_played_map = fetch_cfbd_games_played(draft_year)
            print(f"[cfbd] Games played resolved for {len(games_played_map)} players")
        except Exception as exc:
            print(f"[cfbd] WARNING: games-played fetch failed ({exc}), will default to None")
            games_played_map = {}

        # Player season stats — indexed by name and by player ID
        print("[cfbd] Fetching player season stats")
        by_name: Dict[int, Dict[str, List]] = {}   # {yr: {name_lower: [rows]}}
        by_id:   Dict[int, Dict[int, List]] = {}   # {yr: {player_id: [rows]}}
        for yr in years:
            print(f"[cfbd] Fetching player stats for {yr}")
            try:
                data = _cfbd_get("/stats/player/season",
                                 {"year": yr, "seasonType": "regular"}) or []
                bn: Dict[str, List] = {}
                bi: Dict[int, List] = {}
                for row in data:
                    n  = (row.get("player") or "").lower()
                    pid = _safe_int(row.get("playerId"))
                    position = (row.get("position") or "").upper()
                    
                    if position not in {"QB", "WR", "RB", "TE"}:
                        continue
                    
                    if n:   bn.setdefault(n, []).append(row)
                    if pid: bi.setdefault(pid, []).append(row)
                by_name[yr] = bn
                by_id[yr]   = bi
                print(f"[cfbd] Loaded player stats for {yr}: {len(bn)} players")
            except Exception as exc:
                print(f"[cfbd] ERROR loading player stats for {yr} — {type(exc).__name__}: {exc}")
                by_name[yr] = {}
                by_id[yr] = {}

        # Collapse into per-player season lists keyed by lowercase name
        print("[cfbd] Building player season summaries")
        all_names: set = set()
        for yr in years:
            all_names.update(by_name[yr].keys())

        result: Dict[str, List[Dict]] = {}
        for name in all_names:
            try:
                seasons = []
                for yr in years:
                    rows = by_name.get(yr, {}).get(name, [])
                    if not rows:
                        continue
                    try:
                        gp = (games_played_map.get(name) or {}).get(yr)
                        seasons.append(_build_cfbd_season(rows, team_stats.get(yr, {}), yr, gp))
                        # print(name)
                        # print(seasons)
                    except Exception as exc:
                        print(f"[cfbd] ERROR building season for '{name}' year {yr} — {type(exc).__name__}: {exc}")
                if seasons:
                    seasons.sort(key=lambda s: s["season"])
                    result[name] = seasons
            except Exception as exc:
                print(f"[cfbd] ERROR processing player '{name}' — {type(exc).__name__}: {exc}")

        print(f"[cfbd] COMPLETE: Loaded stats for {len(result)} players (draft class {draft_year})")
        return result
        
    except Exception as exc:
        print(f"[cfbd] FAILED: Unexpected error fetching college stats for {draft_year} — {type(exc).__name__}: {exc}")
        return {}


# ─────────────────────────────────────────────────────────────────────────────
# Supplementary source 3 — ESPN athlete search  (no auth required)
# Provides: dateOfBirth → exact age at draft
#
# Endpoint: https://site.api.espn.com/apis/common/v3/search
#   ?query={name}&sport=football&type=athlete&limit=3
#
# Response shape (relevant portion):
#   { "results": [{ "contents": [{ "type": "athlete",
#                                   "displayName": "Travis Hunter",
#                                   "dateOfBirth": "2004-12-15",
#                                   "league": {"name": "College Football"}, ... }] }] }
#
# We match on normalized display name and pick the college-football entry when
# multiple sport entries are returned (e.g. the same name in NFL + CFB).
# ─────────────────────────────────────────────────────────────────────────────

_ESPN_SEARCH_URL = "https://site.api.espn.com/apis/common/v3/search"


def _extract_espn_items(data: dict) -> list:
    """
    Handle both ESPN search response shapes:
      Format A (current):  {"items": [...]}
      Format B (older):    {"results": [{"contents": [...]}]}
    Returns a flat list of candidate item dicts.
    """
    items = data.get("items") or []
    if items:
        return list(items)
    # Older format: results > contents
    flat: list = []
    for result in (data.get("results") or []):
        flat.extend(result.get("contents") or [])
    return flat


_NAME_SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}


def _names_match(item_name: str, query_name: str) -> bool:
    """
    Tolerant name comparison: strip common suffixes, normalise
    hyphens/apostrophes/periods to spaces, and compare case-insensitively.
    Handles: "Patrick Mahomes II" == "Patrick Mahomes",
             "Brian Robinson Jr." == "Brian Robinson",
             "Ja'Marr Chase" == "JaMarr Chase", etc.
    """
    def _norm(n: str) -> str:
        n = n.lower().strip()
        n = re.sub(r"\.", "", n)                # remove periods entirely (D.J. → DJ)
        n = re.sub(r"['\-\u2019]", " ", n)     # apostrophes/hyphens → space
        n = re.sub(r"\s+", " ", n).strip()
        parts = [p for p in n.split() if p not in _NAME_SUFFIXES]
        return " ".join(parts)

    return _norm(item_name) == _norm(query_name)


def _age_at_date(dob_str: str, ref_year: int, ref_month: int = 4, ref_day: int = 25) -> Optional[float]:
    """
    Compute fractional age (years) at ref_year-ref_month-ref_day given a
    'YYYY-MM-DD' (or 'YYYY-MM-DDTHH:MM:SSZ') date-of-birth string.
    """
    try:
        dob_part = dob_str[:10]   # keep only 'YYYY-MM-DD'
        from datetime import date
        dob  = date.fromisoformat(dob_part)
        ref  = date(ref_year, ref_month, ref_day)
        days = (ref - dob).days
        return round(days / 365.25, 2)
    except (ValueError, TypeError):
        return None


def fetch_espn_ages(names: List[str], draft_year: int,
                    delay: float = 0.25) -> Dict[str, float]:
    """
    Look up each prospect by name via ESPN's public search API and return a
    dict of {name_lower: age_at_draft_float}.

    Uses a small delay between requests to avoid rate-limiting.
    Silently skips any name that can't be resolved.

    Args:
        names:       list of player names to look up
        draft_year:  draft year (used as reference date — late April)
        delay:       seconds to sleep between requests (default 0.25 s)
    """
    result: Dict[str, float] = {}
    failed_requests = 0
    failed_parsing = 0
    no_dob_found = 0
    
    print(f"[espn_ages] Starting age lookup for {len(names)} prospects (draft_year={draft_year})")
    print(f"[espn_ages] URL: {_ESPN_SEARCH_URL}")
    
    for i, name in enumerate(names):
        if (i + 1) % 25 == 0:
            print(f"[espn_ages] Processed {i + 1}/{len(names)} prospects ({len(result)} ages found, {failed_requests} request fails, {no_dob_found} no DOB)")
        
        name_norm = name.lower().strip()

        # Try sport=football first (covers both NFL + CFB entries).
        # If no DOB found, retry with sport=college-football which sometimes
        # returns a different result set with DOB populated.
        sports_to_try = ["football", "college-football"]
        dob: Optional[str] = None
        found_match = False

        for sport_param in sports_to_try:
            if dob:
                break  # already resolved from a previous sport attempt

            data = None
            request_ok = False

            # Up to 2 attempts per sport (retry on empty/network errors)
            for attempt in range(2):
                try:
                    resp = requests.get(
                        _ESPN_SEARCH_URL,
                        params={"query": name, "sport": sport_param,
                                "type": "player", "limit": "5"},
                        headers={"Accept": "application/json"},
                        timeout=10,
                    )
                    print(f"[espn_ages] {name} ({sport_param}): HTTP {resp.status_code}")
                    resp.raise_for_status()
                    data = resp.json()
                    request_ok = True
                    break  # successful fetch

                except requests.Timeout:
                    print(f"[espn_ages] {name}: TIMEOUT (attempt {attempt + 1})")
                    time.sleep(1.0)
                except requests.HTTPError as exc:
                    code = exc.response.status_code
                    if code == 429:
                        print(f"[espn_ages] {name}: RATE LIMITED (429) — backing off 2s")
                        time.sleep(delay * 2)
                    else:
                        print(f"[espn_ages] {name}: HTTP {code} {exc.response.reason}")
                    break  # don't retry on HTTP errors other than timeout
                except requests.RequestException as exc:
                    print(f"[espn_ages] {name}: Request error — {type(exc).__name__}: {exc}")
                    time.sleep(delay)
                except Exception as exc:
                    print(f"[espn_ages] {name}: Unexpected error — {type(exc).__name__}: {exc}")
                    break

            if not request_ok or data is None:
                failed_requests += 1
                continue  # try next sport or give up

            # Parse response — handle both ESPN response shapes via helper
            try:
                items = _extract_espn_items(data)
                if not items:
                    print(f"[espn_ages] {name} ({sport_param}): No items in response")
                    continue  # try next sport

                for item in items:
                    item_type = item.get("type")
                    if item_type not in ("player", "athlete"):
                        continue

                    item_display = item.get("displayName") or ""
                    if not _names_match(item_display, name):
                        continue

                    found_match = True
                    league = item.get("league", "")
                    league_name = league.get("name", "") if isinstance(league, dict) else str(league)

                    candidate = item.get("dateOfBirth")
                    if candidate:
                        dob = candidate
                        print(f"[espn_ages] {name}: Found DOB={dob} (league={league_name}, sport={sport_param})")
                        if "college" in league_name.lower():
                            break  # best match — prefer CFB entry
                    else:
                        print(f"[espn_ages] {name}: Match found but no dateOfBirth (sport={sport_param})")

            except Exception as exc:
                print(f"[espn_ages] {name}: ERROR parsing ESPN response — {type(exc).__name__}: {exc}")
                failed_parsing += 1

        # Resolve age from DOB (or record failure)
        if not found_match:
            print(f"[espn_ages] {name}: No matching player found in ESPN results")
            no_dob_found += 1
        elif not dob:
            print(f"[espn_ages] {name}: Match found but no DOB available")
            no_dob_found += 1
        else:
            try:
                age = _age_at_date(dob, draft_year)
                if age is not None:
                    result[name_norm] = age
                    print(f"[espn_ages] {name}: SUCCESS — DOB={dob} → age={age:.2f}")
                else:
                    print(f"[espn_ages] {name}: DOB={dob} is unparseable")
                    failed_parsing += 1
            except Exception as exc:
                print(f"[espn_ages] {name}: ERROR calculating age from DOB={dob} — {exc}")
                failed_parsing += 1

        time.sleep(delay)

    print(f"[espn_ages] COMPLETE: {len(result)}/{len(names)} ages resolved ({failed_requests} request fails, {no_dob_found} no DOB, {failed_parsing} parse errors)")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Age fallback — estimation from Sportradar experience field
# Used only when ESPN lookup fails.
# ─────────────────────────────────────────────────────────────────────────────

# Typical age at draft by college class (position-neutral)
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
    try:
        name = (
            raw.get("name") or
            " ".join(filter(None, [raw.get("first_name"), raw.get("last_name")]))
        ).strip()
        if not name:
            print(f"[sportradar] Skipping prospect: no name found (raw keys: {list(raw.keys())})")
            return None

        position = (raw.get("position") or "").upper()
        
        # Filter to only include QB, RB, WR, TE positions
        if position not in {"QB", "RB", "WR", "TE"}:
            return None
        
        school = raw.get("team_name")

        conf_obj = raw.get("conference") or {}
        conference = conf_obj.get("name") if isinstance(conf_obj, dict) else conf_obj

        # birth_place is "City, ST, USA" — no DOB available from this endpoint
        birth_place = raw.get("birth_place") or ""
        state = None
        if birth_place:
            parts = [p.strip() for p in birth_place.split(",")]
            if len(parts) >= 2:
                state = parts[1]   # e.g. "GA"

        height = _safe_int(raw.get("height"))
        weight = _safe_int(raw.get("weight"))
        experience = raw.get("experience")

        return {
            "player_id":        f"ROOKIE_{draft_year}_{_slug(name)}",
            "name":             name,
            "position":         position,
            "school":           school,
            "age":              None,   # filled from combine or estimated from experience
            "height_inches":    height,
            "weight_lbs":       weight,
            "state":            state,
            "draft_class_year": draft_year,
            "early_declare":    False,
            "seasons":          [],     # filled from CFBD stats or seed
            "athleticism":      {},     # filled from NFLVerse combine or seed
            "source":           "sportradar",
            # Internal fields used during merge — stripped before normalization
            "_conference":      conference,
            "_experience":      experience,  # SR/JR/SO/FR for age estimation
        }
    except Exception as exc:
        print(f"[sportradar] ERROR parsing prospect: {raw.get('name', 'UNKNOWN')} — {type(exc).__name__}: {exc}")
        return None


def fetch_sportradar_prospects(draft_year: int) -> List[Dict[str, Any]]:
    """
    Fetch the full NFL prospect list for `draft_year` from Sportradar.

    Endpoint: GET /draft/nfl/{access_level}/v1/en/{year}/prospects.json
    Returns a list of partially-normalized prospect dicts (bio only).
    Age, combine, and stats are not provided by this endpoint and must be
    merged from the seed dataset or other sources.
    """
    print(f"[sportradar] Starting prospect fetch for draft year {draft_year}")
    print(f"[sportradar] API key present: {'YES' if SPORTRADAR_KEY else 'NO'}")
    print(f"[sportradar] Access level: {SPORTRADAR_ACCESS}")
    
    if not SPORTRADAR_KEY:
        print("[sportradar] FAILED: No SPORTRADAR_API_KEY set — cannot fetch prospects")
        return []

    try:
        data = _sportradar_get(f"{draft_year}/prospects.json")
    except Exception as exc:
        print(f"[sportradar] FAILED: Unexpected error fetching prospects for {draft_year} — {type(exc).__name__}: {exc}")
        return []
        
    if not data:
        print(f"[sportradar] FAILED: No data returned for {draft_year} — API may be down or key invalid")
        return []

    raw_list = data.get("prospects") or (data if isinstance(data, list) else [])
    if not raw_list:
        print(f"[sportradar] FAILED: Empty prospects list for {draft_year} — response format may have changed")
        print(f"[sportradar] Response keys: {list(data.keys()) if isinstance(data, dict) else 'N/A (not dict)'}")
        return []

    print(f"[sportradar] Processing {len(raw_list)} raw prospects from API")
    results = []
    filtered_count = 0
    parse_errors = 0
    
    for i, raw in enumerate(raw_list):
        try:
            prospect = _parse_sportradar_prospect(raw, draft_year)
            if prospect:
                results.append(prospect)
            else:
                filtered_count += 1
        except Exception as exc:
            print(f"[sportradar] ERROR parsing prospect at index {i} — {type(exc).__name__}: {exc}")
            parse_errors += 1
        
        if (i + 1) % 100 == 0:
            print(f"[sportradar] Processed {i + 1}/{len(raw_list)} raw prospects ({len(results)} kept, {filtered_count} filtered, {parse_errors} errors)")

    print(f"[sportradar] COMPLETE: {len(results)} prospects for {draft_year} (filtered {filtered_count} non-QB/RB/WR/TE, {parse_errors} parse errors)")
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


def get_seed_prospects(draft_year: int) -> List[Dict[str, Any]]:
    """Return the curated seed dataset for a given draft year."""
    return []


def prospects_from_mock_draft(mock_picks: List[Dict[str, Any]], draft_year: int) -> List[Dict[str, Any]]:
    """
    Create minimal prospect records from mock draft data.

    This is used as a fallback when no seed data or API data is available.
    """
    prospects = []
    seen_players = set()

    for pick in mock_picks:
        player_name = pick.get("player_name", "").strip()
        if not player_name or player_name in seen_players:
            continue

        seen_players.add(player_name)

        # Generate player_id
        player_id = f"ROOKIE_{draft_year}_{_slug(player_name)}"

        prospects.append({
            "player_id": player_id,
            "name": player_name,
            "position": pick.get("position", ""),
            "school": pick.get("school", ""),
            "draft_class_year": draft_year,
            "age": None,
            "height_inches": None,
            "weight_lbs": None,
            "hometown": None,
            "state": None,
            "early_declare": False,
            "transfer_history": None,
            "headshot_url": None,
        })

    print(f"[ingestion] Created {len(prospects)} prospects from mock draft data")
    return prospects

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

    Merges up to five sources in priority order:

    1. Sportradar (SPORTRADAR_API_KEY)  → player list, bio, height, weight,
                                          school, conference, experience
    2. ESPN search (no key)             → exact age from dateOfBirth
    3. NFLVerse combine.csv (no key)    → forty_yard, vertical, broad_jump,
                                          bench_reps, three_cone, shuttle
    4. CFBD (CFBD_API_KEY)              → per-season college stats,
                                          games_played, market share,
                                          dominator rating
    5. Seed dataset (always available)  → fallback for any missing field;
                                          used entirely when Sportradar is
                                          not configured

    Age priority: ESPN DOB lookup → seed value → Sportradar experience estimate
    """
    print(f"[ingestion] Starting prospect loading for draft year {draft_year}")
    print(f"[ingestion] SPORTRADAR_API_KEY present: {'YES' if SPORTRADAR_KEY else 'NO'}")
    print(f"[ingestion] CFBD_API_KEY present: {'YES' if CFBD_KEY else 'NO'}")
    
    try:
        seed = get_seed_prospects(draft_year) or []
    except Exception as exc:
        print(f"[ingestion] ERROR loading seed prospects — {type(exc).__name__}: {exc}")
        seed = []
    print(f"[ingestion] Loaded {len(seed)} seed prospects")

    # ── No Sportradar key — use seed only ────────────────────────────────────
    if not SPORTRADAR_KEY:
        print(f"[ingestion] FAILED: No SPORTRADAR_API_KEY set — returning seed data only ({len(seed)} prospects)")
        return [normalize_prospect(p) for p in seed]

    # ── Fetch from all live sources ───────────────────────────────────────────
    print(f"[ingestion] Fetching Sportradar prospects for {draft_year}")
    try:
        sr_prospects = fetch_sportradar_prospects(draft_year)
    except Exception as exc:
        print(f"[ingestion] ERROR fetching Sportradar prospects — {type(exc).__name__}: {exc}")
        print(f"[ingestion] Falling back to seed data ({len(seed)} prospects)")
        return [normalize_prospect(p) for p in seed]
    
    if not sr_prospects:
        print(f"[ingestion] FAILED: Sportradar returned no data for {draft_year} — using seed ({len(seed)} prospects)")
        return [normalize_prospect(p) for p in seed]
    
    print(f"[ingestion] Sportradar returned {len(sr_prospects)} prospects")

    # ESPN age lookup — use the robust scraper (search + athlete API + HTML fallback)
    print(f"[ingestion] Starting ESPN age lookup for {len(sr_prospects)} prospects")
    try:
        from .espn_scraper import fetch_espn_ages_robust
        espn_ages = fetch_espn_ages_robust(
            [p["name"] for p in sr_prospects],
            draft_year,
            prospects_meta=sr_prospects,   # provides school + position for disambiguation
        )
    except Exception as exc:
        print(f"[ingestion] ERROR in ESPN age lookup — {type(exc).__name__}: {exc}")
        espn_ages = {}
    print(f"[ingestion] ESPN ages resolved for {len(espn_ages)} prospects")

    print(f"[ingestion] Fetching NFLverse combine data for {draft_year}")
    try:
        combine_data = fetch_nflverse_combine(draft_year)
    except Exception as exc:
        print(f"[ingestion] ERROR fetching NFLverse combine data — {type(exc).__name__}: {exc}")
        combine_data = {}
    print(f"[ingestion] NFLverse combine data loaded for {len(combine_data)} prospects")
    
    print(f"[ingestion] Fetching CFBD college stats for {draft_year}")
    try:
        cfbd_stats = fetch_cfbd_college_stats(draft_year)
    except Exception as exc:
        print(f"[ingestion] ERROR fetching CFBD stats — {type(exc).__name__}: {exc}")
        cfbd_stats = {}
    print(f"[ingestion] CFBD stats loaded for {len(cfbd_stats)} prospects")

    try:
        seed_by_name = {p["name"].lower(): p for p in seed}
    except Exception as exc:
        print(f"[ingestion] ERROR building seed lookup — {type(exc).__name__}: {exc}")
        seed_by_name = {}
    
    print(f"[ingestion] Starting enrichment of {len(sr_prospects)} Sportradar prospects")

    enriched: List[Dict] = []
    enrichment_errors = 0
    
    for i, sr in enumerate(sr_prospects):
        try:
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

            # ── Age: ESPN DOB (exact) → seed → experience estimate ───────────────
            try:
                p["age"] = (
                    espn_ages.get(name_key) or
                    (seed_p or {}).get("age") or
                    _estimate_age(sr.get("_experience"), draft_year)
                )
            except Exception as exc:
                print(f"[ingestion] ERROR calculating age for '{sr['name']}' — {type(exc).__name__}: {exc}")
                p["age"] = None

            # ── Athleticism / combine ─────────────────────────────────────────────
            try:
                seed_ath  = (seed_p or {}).get("athleticism") or {}
                nflv_ath  = nflv.get("athleticism") or {}
                # NFLVerse is authoritative for combine; seed fills any remaining gaps
                p["athleticism"] = {**seed_ath, **nflv_ath}
            except Exception as exc:
                print(f"[ingestion] ERROR merging athleticism for '{sr['name']}' — {type(exc).__name__}: {exc}")
                p["athleticism"] = {}

            # ── College stats ─────────────────────────────────────────────────────
            try:
                if cfbd_seasons:
                    p["seasons"] = cfbd_seasons
                elif seed_p and seed_p.get("seasons"):
                    p["seasons"] = seed_p["seasons"]
                elif sr.get("_conference"):
                    # New player with no stats — inject conference so competition score works
                    p["seasons"] = [{"season": draft_year - 1, "conference": sr["_conference"]}]
                else:
                    p["seasons"] = []
            except Exception as exc:
                print(f"[ingestion] ERROR setting seasons for '{sr['name']}' — {type(exc).__name__}: {exc}")
                p["seasons"] = []

            enriched.append(p)
        except Exception as exc:
            print(f"[ingestion] ERROR enriching prospect '{sr.get('name', 'UNKNOWN')}' (index {i}) — {type(exc).__name__}: {exc}")
            enrichment_errors += 1
        
        if len(enriched) % 50 == 0:
            print(f"[ingestion] Enriched {len(enriched)}/{len(sr_prospects)} prospects ({enrichment_errors} errors)")

    # Seed players not returned by Sportradar
    try:
        sr_names  = {p["name"].lower() for p in sr_prospects}
        seed_only = [p for p in seed if p["name"].lower() not in sr_names]
        if seed_only:
            print(f"[ingestion] {len(seed_only)} seed prospects not in Sportradar — appending")
    except Exception as exc:
        print(f"[ingestion] ERROR processing seed-only prospects — {type(exc).__name__}: {exc}")
        seed_only = []

    final = enriched + seed_only
    print(
        f"[ingestion] COMPLETE: {len(final)} total prospects for {draft_year}  "
        f"(Sportradar: {len(enriched)} | combine: {sum(1 for p in enriched if p.get('athleticism'))} matched | CFBD stats: {sum(1 for p in enriched if cfbd_stats.get(p['name'].lower()))} matched | seed-only: {len(seed_only)} | enrichment errors: {enrichment_errors})"
    )
    
    try:
        return [normalize_prospect(p) for p in final]
    except Exception as exc:
        print(f"[ingestion] ERROR normalizing prospects — {type(exc).__name__}: {exc}")
        # Return unnormalized data as fallback
        return final
