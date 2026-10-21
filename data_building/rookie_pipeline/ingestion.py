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

Normalization contract - every player dict returned by this module has:
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
import csv
import datetime
import hashlib
import io
import os
import re
import time
import warnings
from typing import Any, Dict, List, Optional, Set, Tuple

import requests
def normalize_name(name: str) -> str:
    """
    THE canonical player-name key for name joins in this pipeline.

    Lowercases; unifies punctuation variants (periods and apostrophes are
    dropped, hyphens and any other punctuation become spaces); collapses
    whitespace; drops generational suffix tokens (Jr./Sr./II/III/IV/V); and
    collapses spaced single-letter initials ("K C" -> "KC").  Both sides of
    a join must pass through here so spelling variants of one player land
    on the same key: "C.J. Carr" == "CJ Carr", "Trey'Dez Green" ==
    "Treydez Green", "Ryan Coleman-Williams" == "Ryan Coleman Williams".
    """
    _SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}
    if not name:
        return ""
    s = str(name).lower()
    s = s.replace(".", "").replace("'", "").replace("\u2019", "")
    s = re.sub(r"[^a-z0-9]+", " ", s)
    parts = [p for p in s.split() if p not in _SUFFIXES]
    s = " ".join(parts)
    # Collapse spaced single-letter initials: "k c concepcion" -> "kc concepcion"
    s = re.sub(r"\b([a-z])\s([a-z])\b", r"\1\2", s)
    return s.strip()
from .sagarin import get_team_rating as _sagarin_get_team_rating

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

# CFBD API caching
_CFBD_CACHE: Dict[str, Dict[str, Any]] = {}  # {cache_key: {"data": data, "timestamp": timestamp}}
_CFBD_CACHE_TTL = 3600  # 1 hour cache TTL for CFBD data

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _slug(name: str) -> str:
    """Convert 'Travis Hunter' → 'TRAVIS_HUNTER'. Strips periods so 'K.C.' → 'KC'."""
    return re.sub(r"[^A-Z0-9]+", "_", name.upper().replace(".", "")).strip("_")


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

    last_error = None
    for attempt in range(retries):
        try:
            resp = requests.get(url, headers=headers, params=params, timeout=20)
            resp.raise_for_status()
            return resp.json()
        except requests.Timeout as exc:
            last_error = f"Timeout after 20s"
            wait = 2 ** attempt
            print(f"[sportradar] {path} attempt {attempt + 1}/{retries}: TIMEOUT - retrying in {wait}s")
            time.sleep(wait)
        except requests.HTTPError as exc:
            last_error = f"HTTP {exc.response.status_code}: {exc.response.reason}"
            if exc.response.status_code == 401:
                print(f"[sportradar] {path} FAILED: Authentication error (401) - check SPORTRADAR_API_KEY")
                return None
            elif exc.response.status_code == 403:
                print(f"[sportradar] {path} FAILED: Forbidden (403) - check API key permissions or access level (current: {SPORTRADAR_ACCESS})")
                return None
            elif exc.response.status_code == 404:
                print(f"[sportradar] {path} FAILED: Not found (404) - invalid path or year")
                return None
            elif exc.response.status_code == 429:
                wait = 2 ** attempt
                print(f"[sportradar] {path} attempt {attempt + 1}/{retries}: RATE LIMITED (429) - retrying in {wait}s")
                time.sleep(wait)
            else:
                wait = 2 ** attempt
                print(f"[sportradar] {path} attempt {attempt + 1}/{retries}: HTTP {exc.response.status_code} - retrying in {wait}s")
                time.sleep(wait)
        except requests.RequestException as exc:
            last_error = f"Request failed: {type(exc).__name__}: {exc}"
            wait = 2 ** attempt
            print(f"[sportradar] {path} attempt {attempt + 1}/{retries}: {type(exc).__name__} - retrying in {wait}s")
            time.sleep(wait)
        except Exception as exc:
            last_error = f"Unexpected error: {type(exc).__name__}: {exc}"
            print(f"[sportradar] {path} attempt {attempt + 1}/{retries}: UNEXPECTED ERROR - {last_error}")
            return None
    
    print(f"[sportradar] {path} FAILED after {retries} attempts - Last error: {last_error}")
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Supplementary source 1 - NFLVerse combine.csv  (no auth required)
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
        print("[nflverse] FAILED: Download timeout after 30s - NFLVerse may be slow or unreachable")
        return {}
    except requests.HTTPError as exc:
        print(f"[nflverse] FAILED: HTTP {exc.response.status_code} ({exc.response.reason}) - URL may have changed or GitHub rate limit")
        return {}
    except requests.RequestException as exc:
        print(f"[nflverse] FAILED: Network error - {type(exc).__name__}: {exc}")
        return {}
    except Exception as exc:
        print(f"[nflverse] FAILED: Unexpected error - {type(exc).__name__}: {exc}")
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
        print(f"[nflverse] FAILED: CSV parsing error at row {total_rows} - {exc}")
        return results  # Return what we have so far
    except Exception as exc:
        print(f"[nflverse] FAILED: Unexpected error during CSV processing - {type(exc).__name__}: {exc}")
        return results

    print(f"[nflverse] Loaded combine data for {len(results)} prospects for {draft_year} ({total_rows} total rows, {parse_errors} parse errors)")
    print(f"[nflverse] Found {matching_rows} matching rows for {draft_year} in combine data")
    return results


def fetch_local_combine_csv(draft_year: int) -> Dict[str, Dict[str, Any]]:
    """
    Process the local combine CSV file and return a name-keyed dict of
    combine measurements for prospects from `draft_year`.

    Returns {player_name_lower: athleticism_dict}  where athleticism_dict has:
        forty_yard, vertical_inches, broad_jump_in, bench_reps,
        three_cone, short_shuttle
    Also includes height_inches and weight_lbs as fallback bio fields.
    """
    import pandas as pd
    import re
    import time
    from pathlib import Path
    
    print(f"[local_combine] Processing local combine CSV for draft year {draft_year}")
    
    csv_dir = Path(__file__).parent.parent.parent / "cache"
    csv_path = csv_dir / "athleticism" / f"{draft_year}.csv"
    if not csv_path.exists():
        csv_path = csv_dir / f"Official Times & Measurements - {draft_year}.csv"
    
    if not csv_path.exists():
        print(f"[local_combine] ERROR: CSV file not found at {csv_path}")
        return {}
    mtime = csv_path.stat().st_mtime
    print(f"[local_combine] Using {csv_path} (mtime {mtime})")
    stale_days = 400
    if (time.time() - mtime) > stale_days * 86400:
        print(f"[local_combine] WARNING: athleticism CSV is older than {stale_days} days")
    
    try:
        # Read the CSV file with proper header handling
        df = pd.read_csv(csv_path, header=2)
        df = df.dropna(axis=1, how='all').dropna(how='all')
        
        # Filter for target positions (QB, RB, WR, TE)
        target_positions = ['QB', 'RB', 'WR', 'TE']
        filtered_df = df[df['POS'].isin(target_positions)].copy()
        
        results: Dict[str, Dict[str, Any]] = {}
        total_rows = len(filtered_df)
        matching_rows = 0
        parse_errors = 0
        
        for _, row in filtered_df.iterrows():
            try:
                player_name = str(row['PLAYER']).strip()
                if not player_name or player_name == 'nan':
                    parse_errors += 1
                    continue
                
                # Clean player name (remove suffixes, convert to lowercase for key)
                clean_name = re.sub(r'\s+(Jr\.|Sr\.|II|III|IV)$', '', player_name)
                name_key = clean_name.lower().strip()
                
                # Extract combine measurements
                athleticism = {}
                
                # 40-yard dash
                forty = row.get('40 (O)')
                if pd.notna(forty) and forty != '':
                    try:
                        athleticism['forty_yard'] = float(forty)
                    except (ValueError, TypeError):
                        logging.getLogger(__name__).debug("suppressed exception", exc_info=True)
                
                # Vertical jump (inches)
                vertical = row.get('VERT')
                if pd.notna(vertical) and vertical != '':
                    try:
                        vert_val = float(vertical)
                        if vert_val.is_integer():
                            athleticism['vertical_inches'] = int(vert_val)
                        else:
                            athleticism['vertical_inches'] = vert_val
                    except (ValueError, TypeError):
                        logging.getLogger(__name__).debug("suppressed exception", exc_info=True)
                
                # Broad jump (inches)
                broad = row.get('BROAD')
                if pd.notna(broad) and broad != '':
                    try:
                        athleticism['broad_jump_in'] = int(float(broad))
                    except (ValueError, TypeError):
                        logging.getLogger(__name__).debug("suppressed exception", exc_info=True)
                
                # 3-cone drill
                three_cone = row.get('3 CONE')
                if pd.notna(three_cone) and three_cone != '':
                    try:
                        athleticism['three_cone'] = float(three_cone)
                    except (ValueError, TypeError):
                        logging.getLogger(__name__).debug("suppressed exception", exc_info=True)
                
                # Short shuttle
                shuttle = row.get('SHUTTLE')
                if pd.notna(shuttle) and shuttle != '':
                    try:
                        athleticism['short_shuttle'] = float(shuttle)
                    except (ValueError, TypeError):
                        logging.getLogger(__name__).debug("suppressed exception", exc_info=True)
                
                # Bench press
                bench = row.get('BENCH')
                if pd.notna(bench) and bench != '':
                    try:
                        athleticism['bench_reps'] = int(float(bench))
                    except (ValueError, TypeError):
                        logging.getLogger(__name__).debug("suppressed exception", exc_info=True)
                
                # Height (convert NFL format to inches)
                height = row.get('HEIGHT')
                if pd.notna(height) and height != '':
                    try:
                        height_str = str(int(float(height)))  # Convert to string without decimal
                        if len(height_str) == 4 and height_str.isdigit():
                            # NFL format: first digit = feet, next two digits = inches, last digit = eighths
                            feet = int(height_str[0])
                            inches = int(height_str[1:3])
                            eighths = int(height_str[3])
                            total_inches = feet * 12 + inches + eighths / 8
                            athleticism['height_inches'] = total_inches
                    except (ValueError, TypeError):
                        logging.getLogger(__name__).debug("suppressed exception", exc_info=True)
                
                # Weight
                weight = row.get('WEIGHT')
                if pd.notna(weight) and weight != '':
                    try:
                        athleticism['weight_lbs'] = int(float(weight))
                    except (ValueError, TypeError):
                        logging.getLogger(__name__).debug("suppressed exception", exc_info=True)
                
                # RAS (Relative Athletic Score)
                ras = row.get('RAS')
                if pd.notna(ras) and ras != '':
                    try:
                        athleticism['ras_score'] = float(ras)
                    except (ValueError, TypeError):
                        logging.getLogger(__name__).debug("suppressed exception", exc_info=True)
                
                # Only add if we have at least one measurement
                if athleticism:
                    # Structure the data as expected by upsert_prospect_athleticism
                    results[name_key] = {
                        "athleticism": athleticism,
                        "height_inches": athleticism.get("height_inches"),
                        "weight_lbs": athleticism.get("weight_lbs")
                    }
                    matching_rows += 1
                
            except Exception as exc:
                parse_errors += 1
                print(f"[local_combine] Parse error for row: {exc}")
                continue
        
        print(f"[local_combine] Processed combine data for {len(results)} prospects for {draft_year} ({total_rows} total rows, {parse_errors} parse errors)")
        print(f"[local_combine] Found {matching_rows} matching rows with measurements for {draft_year}")
        return results
        
    except Exception as exc:
        print(f"[local_combine] ERROR: Failed to process CSV file - {type(exc).__name__}: {exc}")
        return {}


# ─────────────────────────────────────────────────────────────────────────────
# Supplementary source 2 - CFBD college stats  (requires CFBD_API_KEY)
# Provides: per-season receiving/rushing/passing stats, games_played,
#           team, conference, market share, dominator rating
# ─────────────────────────────────────────────────────────────────────────────

_CFBD_THROTTLE_S = 4.0   # 600 req/hr = 1 per 6s; 4s sleep + ~0.5s latency ≈ 720/hr (safe)

def _get_cache_key(path: str, params: Optional[Dict[str, Any]] = None) -> str:
    """Generate a unique cache key for CFBD API requests."""
    key_data = f"{path}:{str(params or {})}"
    return hashlib.md5(key_data.encode()).hexdigest()


def _is_cache_valid(timestamp: float) -> bool:
    """Check if cached data is still valid based on TTL."""
    return time.time() - timestamp < _CFBD_CACHE_TTL

def _cfbd_get(path: str, params: Dict[str, Any] = None, retries: int = 5) -> Optional[Any]:
    # Check cache first
    cache_key = _get_cache_key(path, params)
    cached_entry = _CFBD_CACHE.get(cache_key)
    
    if cached_entry and _is_cache_valid(cached_entry["timestamp"]):
        print(f"[cfbd] {path} CACHE HIT")
        return cached_entry["data"]
    
    url = f"{CFBD_BASE}{path}"
    headers = {"Accept": "application/json", "Authorization": f"Bearer {CFBD_KEY}"}

    time.sleep(_CFBD_THROTTLE_S)  # global throttle before every call

    last_error = None
    for attempt in range(retries):
        try:
            resp = requests.get(url, headers=headers, params=params or {}, timeout=15)
            resp.raise_for_status()
            data = resp.json()
            
            # Cache the successful response
            _CFBD_CACHE[cache_key] = {
                "data": data,
                "timestamp": time.time()
            }
            print(f"[cfbd] {path} CACHED (TTL: {_CFBD_CACHE_TTL}s)")
            return data
        except requests.Timeout:
            last_error = "Timeout after 15s"
            wait = 2 ** attempt
            time.sleep(wait)
        except requests.HTTPError as exc:
            last_error = f"HTTP {exc.response.status_code}: {exc.response.reason}"
            if exc.response.status_code == 401:
                print(f"[cfbd] {path} FAILED: Authentication error (401) - check CFBD_API_KEY is valid")
                return None
            elif exc.response.status_code == 403:
                print(f"[cfbd] {path} FAILED: Forbidden (403) - API key may not have required permissions")
                return None
            elif exc.response.status_code == 404:
                print(f"[cfbd] {path} FAILED: Not found (404) - invalid endpoint or parameters: {params}")
                return None
            elif exc.response.status_code == 400:
                print(f"[cfbd] {path} FAILED: Bad Request (400) - invalid parameters: {params}")
                return None
            elif exc.response.status_code == 429:
                # Respect Retry-After header if present, else exponential back-off
                retry_after = exc.response.headers.get("Retry-After")
                if retry_after:
                    try:
                        wait = int(retry_after) + 1
                    except (ValueError, TypeError):
                        wait = 15 * (2 ** attempt)
                else:
                    wait = 15 * (2 ** attempt)   # 15s, 30s, 60s, 120s, 240s
                print(f"[cfbd] {path} attempt {attempt + 1}/{retries}: RATE LIMITED (429) - backing off {wait}s")
                time.sleep(wait)
            else:
                wait = 2 ** attempt
                print(f"[cfbd] {path} attempt {attempt + 1}/{retries}: HTTP {exc.response.status_code} - retrying in {wait}s")
                time.sleep(wait)
        except requests.RequestException as exc:
            last_error = f"Request failed: {type(exc).__name__}"
            wait = 2 ** attempt
            print(f"[cfbd] {path} attempt {attempt + 1}/{retries}: {type(exc).__name__} - retrying in {wait}s")
            time.sleep(wait)
        except Exception as exc:
            last_error = f"Unexpected error: {type(exc).__name__}: {exc}"
            print(f"[cfbd] {path} FAILED: UNEXPECTED ERROR - {last_error}")
            return None
    
    print(f"[cfbd] {path} FAILED after {retries} attempts - Last error: {last_error}")
    return None

def _compute_depth_ranks(by_name: Dict[int, Dict[str, List]], years: List[int]) -> Dict:
    """
    Infer positional depth chart ranks from CFBD raw stat rows.

    For each (year, team, position) group, players are ranked by primary volume
    metric descending.  depth_rank=1 is the starter/alpha; depth_rank=2 is the
    clear backup; etc.

    Returns {(name_lower, yr): {"depth_rank": int, "position_group_size": int}}
    """
    _PRIMARY_CAT_STAT = {
        "WR": ("receiving", "YDS"),
        "TE": ("receiving", "YDS"),
        "RB": ("rushing",   "YDS"),
        "QB": ("passing",   "YDS"),
    }

    depth_ranks: Dict = {}

    for yr in years:
        # {(team, position): {name_lower: volume}}
        group_vols: Dict = {}

        for name_lower, rows in (by_name.get(yr) or {}).items():
            for row in rows:
                pos = (row.get("position") or "").upper()
                if pos not in _PRIMARY_CAT_STAT:
                    continue
                cat_want, stat_want = _PRIMARY_CAT_STAT[pos]
                cat = (row.get("category") or "").lower()
                stat_type = row.get("statType", "")
                if cat != cat_want or stat_type != stat_want:
                    continue
                team = (row.get("team") or "").strip()
                if not team:
                    continue
                val = float(row.get("stat") or 0)
                key = (team, pos)
                group_vols.setdefault(key, {})
                group_vols[key][name_lower] = group_vols[key].get(name_lower, 0.0) + val

        for (team, pos), player_vols in group_vols.items():
            sorted_players = sorted(player_vols.items(), key=lambda x: x[1], reverse=True)
            group_size = len(sorted_players)
            for rank, (name_lower, _) in enumerate(sorted_players, start=1):
                depth_ranks[(name_lower, yr)] = {
                    "depth_rank": rank,
                    "position_group_size": group_size,
                }

    return depth_ranks


def compute_dominator_rating(rec_yards, rush_yards, rec_tds, rush_tds,
                             team_total_yards, team_total_tds,
                             position=None, team_pass_yards=None,
                             team_pass_tds=None):
    """The pipeline's dominator formula: 0.65 * yard share + 0.35 *
    TD share, each term counted only when its team total is positive.
    Returns None when neither team total exists - the inputs are
    missing, which is NOT a zero (a player with team totals present
    and no production of his own legitimately computes to 0.0).

    The BASIS depends on position (model v3.5):
    - WR/TE: RECEIVING share - receiving yards/TDs over the team's
      passing totals (team_pass_yards / team_pass_tds). The total-
      offense basis structurally taxed pass catchers (their
      denominator included rushing production they cannot touch):
      Jeremiah Smith's 2024 stored 0.208 on the old basis vs 0.345
      on this one, and the model's breakout anchors (0.35 saturation,
      0.4877 recorded best) were always receiving-basis numbers.
    - RB (and any call without a position): SCRIMMAGE share -
      receiving+rushing over total-offense totals, byte-identical to
      the pre-v3.5 formula.
    A WR/TE call NEVER falls back to the total-offense totals: mixed
    bases in one stored column are the defect this change removes.
    When neither passing total exists the rating is None."""
    pos = (position or "").upper()
    if pos in ("WR", "TE"):
        p_yds = team_pass_yards or 0
        p_tds = team_pass_tds or 0
        if not (p_yds or p_tds):
            return None
        dom = 0.0
        if p_yds > 0:
            dom += ((rec_yards or 0) / p_yds) * 0.65
        if p_tds > 0:
            dom += ((rec_tds or 0) / p_tds) * 0.35
        return round(dom, 4)
    t_yds = team_total_yards or 0
    t_tds = team_total_tds or 0
    if not (t_yds or t_tds):
        return None
    dom = 0.0
    if t_yds > 0:
        dom += (((rec_yards or 0) + (rush_yards or 0)) / t_yds) * 0.65
    if t_tds > 0:
        dom += (((rec_tds or 0) + (rush_tds or 0)) / t_tds) * 0.35
    return round(dom, 4)


def derive_season_dominator(season: Dict, position=None) -> bool:
    """Fill season["dominator_rating"] from the season's OWN inputs
    (player receiving/rushing lines + team totals, all carried on the
    season dict / stored row) when the rating is missing but the
    inputs are present. Returns True when a value was derived.

    This is the write-path repair for the stored-data defect where a
    season row was persisted without a dominator (team totals absent
    from the in-memory fetch map at build time) and then never
    recomputed: every writer that persists a season row runs the row
    through here first, so any row whose inputs exist carries the
    rating. A season whose inputs are genuinely absent keeps
    dominator None - missing stays missing, never fabricated.

    The basis follows compute_dominator_rating: the position comes
    from the explicit argument first, then the season dict's own
    "position" key when it carries one (WR/TE derive on the receiving
    basis from team_pass_yards / team_pass_tds; anything else on the
    scrimmage basis from the total-offense totals)."""
    if not isinstance(season, dict):
        return False
    if season.get("dominator_rating") is not None:
        return False
    dom = compute_dominator_rating(
        season.get("receiving_yards"), season.get("rush_yards"),
        season.get("receiving_tds"), season.get("rush_tds"),
        season.get("team_total_yards"), season.get("team_total_tds"),
        position=position or season.get("position"),
        team_pass_yards=season.get("team_pass_yards"),
        team_pass_tds=season.get("team_pass_tds"))
    if dom is None:
        return False
    season["dominator_rating"] = dom
    return True


def _build_cfbd_season(raw_stats: List[Dict], team_stats: Dict, season: int,
                       games: Optional[int], skip_sagarin: bool = False) -> Dict:
    """Fold CFBD stat rows for one player-season into a single normalized dict."""
    row: Dict[str, Any] = {
        "season": season, "games_played": games,
        "pass_yards": 0, "pass_tds": 0, "pass_attempts": 0,
        "completions": 0, "interceptions": 0,
        "rush_attempts": 0, "rush_yards": 0, "rush_tds": 0,
        "receptions": 0, "targets": None, "receiving_yards": 0, "receiving_tds": 0,
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
    row["team_pass_yards"]    = int(ts.get("netPassingYards") or 0)
    row["team_pass_tds"]      = int(ts.get("passingTDs") or 0)
    row["sagarin_team_rating"] = (None if skip_sagarin
                                  else _sagarin_get_team_rating(team_name, season))

    # The player's CFBD position rows decide the dominator basis
    # (v3.5): WR/TE price on receiving share of the passing totals,
    # everyone else on scrimmage share of total offense.
    _position = None
    for _r in raw_stats:
        if _r.get("position"):
            _position = str(_r.get("position")).upper()
            break
    row["dominator_rating"] = compute_dominator_rating(
        rec_yds, rush_yds, rec_tds, rush_tds, t_yds, t_tds,
        position=_position,
        team_pass_yards=row["team_pass_yards"],
        team_pass_tds=row["team_pass_tds"])
    return row


# ─────────────────────────────────────────────────────────────────────────────
# Incremental fetch state (games played + targets)
#
# rookie_prospect_source_data keys on player_id (e.g. ROOKIE_2027_jeremiah_smith);
# it has NO draft_class_year and NO name column, so every "already have data"
# check must key on the player_id prefix.  Which CFBD weeks have already been
# folded into the stored season totals is tracked in rookie_stat_fetched_weeks
# (kind = 'gp' or 'targets'; team is '' for GP, the team name for targets), so
# each run fetches only the next unprocessed week(s) and ADDS their counts to
# the stored totals.  Weeks are disjoint, so adding per-week counts can never
# double count, and a run's written totals are always stored + new, never a
# partial slice.
# ─────────────────────────────────────────────────────────────────────────────

_FETCHED_WEEKS_DDL = """
CREATE TABLE IF NOT EXISTS rookie_stat_fetched_weeks (
    kind TEXT NOT NULL,
    season INT NOT NULL,
    week INT NOT NULL,
    team TEXT NOT NULL DEFAULT '',
    fetched_at TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (kind, season, week, team)
)
"""

# Per-(season, team) accumulated team target totals: the denominator for
# target_share.  Targets themselves live on the player rows; the team total
# has to persist separately so incremental runs can add new weeks to it.
_TEAM_TARGETS_DDL = """
CREATE TABLE IF NOT EXISTS rookie_targets_team_totals (
    season INT NOT NULL,
    team TEXT NOT NULL,
    team_targets INT NOT NULL DEFAULT 0,
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (season, team)
)
"""


def _today() -> datetime.date:
    """Current date (isolated so tests can pin the season clock)."""
    return datetime.date.today()


def _estimated_current_week(today: datetime.date) -> int:
    """Rough CFB week number for `today` (week 1 starts ~Aug 25, capped at 17)."""
    season_start = datetime.date(today.year, 8, 25)
    if today < season_start:
        return 1
    return min((today - season_start).days // 7 + 1, 17)


def _candidate_weeks(year: int, today: datetime.date) -> List[int]:
    """
    Weeks that may legitimately be fetched for `year`:
    - Past seasons: weeks 1-17 (the season is final).
    - Current season: weeks 1..estimated current week.  Weeks that have not
      happened yet are NEVER candidates.
    """
    if year < today.year:
        return list(range(1, 18))
    if year > today.year:
        return []
    return list(range(1, _estimated_current_week(today) + 1))


def _player_id_prefix(draft_year: int) -> str:
    return f"ROOKIE_{draft_year}_"


def _name_key_from_player_id(player_id: str, draft_year: int) -> Optional[str]:
    """'ROOKIE_2027_jeremiah_smith' -> 'jeremiah smith' (inverse of _slug)."""
    prefix = _player_id_prefix(draft_year)
    if not player_id or not player_id.startswith(prefix):
        return None
    name = player_id[len(prefix):].replace("_", " ").strip().lower()
    return name or None


def _get_existing_games_played(draft_year: int, years: List[int]) -> Dict[str, Dict[int, int]]:
    """Get existing games_played from DB: {name_lower: {year: count}}"""
    try:
        from dashboard_services.db import get_conn
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT player_id, season, games_played
                    FROM rookie_prospect_source_data
                    WHERE player_id LIKE %s AND season = ANY(%s)
                    AND games_played IS NOT NULL
                """, (_player_id_prefix(draft_year) + "%", years))
                result: Dict[str, Dict[int, int]] = {}
                for _row in cur.fetchall():
                    player_id, season, gp = _row["player_id"], _row["season"], _row["games_played"]
                    name = _name_key_from_player_id(player_id, draft_year)
                    if not name:
                        continue
                    gp_int = int(gp)
                    prev = result.setdefault(name, {}).get(season)
                    # Defensive: if two ids collapse to one name key, keep
                    # the larger stored total rather than an arbitrary one.
                    if prev is None or gp_int > prev:
                        result[name][season] = gp_int
                return result
    except Exception:
        return {}


def _get_existing_targets(draft_year: int, years: List[int]) -> Dict[str, Dict[int, Dict[str, Any]]]:
    """Stored targets from DB: {name_lower: {year: {"targets": int, "team": str}}}"""
    try:
        from dashboard_services.db import get_conn
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT player_id, season, targets, team
                    FROM rookie_prospect_source_data
                    WHERE player_id LIKE %s AND season = ANY(%s)
                    AND targets IS NOT NULL
                """, (_player_id_prefix(draft_year) + "%", years))
                result: Dict[str, Dict[int, Dict[str, Any]]] = {}
                for _row in cur.fetchall():
                    player_id, season, targets, team = _row["player_id"], _row["season"], _row["targets"], _row["team"]
                    name = _name_key_from_player_id(player_id, draft_year)
                    if not name:
                        continue
                    t_int = int(targets)
                    prev = result.setdefault(name, {}).get(season)
                    if prev is None or t_int > prev["targets"]:
                        result[name][season] = {"targets": t_int, "team": team or ""}
                return result
    except Exception:
        return {}


def _get_team_targets_baselines(years: List[int]) -> Dict[Tuple[int, str], int]:
    """Accumulated team target totals: {(season, team): team_targets}."""
    try:
        from dashboard_services.db import get_conn
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(_TEAM_TARGETS_DDL)
                cur.execute("""
                    SELECT season, team, team_targets FROM rookie_targets_team_totals
                    WHERE season = ANY(%s)
                """, (years,))
                return {(int(_r["season"]), _r["team"]): int(_r["team_targets"]) for _r in cur.fetchall()}
    except Exception:
        return {}


def _record_team_targets(season: int, team: str, team_targets: int) -> None:
    """Persist the accumulated team target total for (season, team)."""
    try:
        from dashboard_services.db import get_conn
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(_TEAM_TARGETS_DDL)
                cur.execute("""
                    INSERT INTO rookie_targets_team_totals (season, team, team_targets)
                    VALUES (%s, %s, %s)
                    ON CONFLICT (season, team) DO UPDATE
                    SET team_targets = EXCLUDED.team_targets, updated_at = NOW()
                """, (season, team, int(team_targets)))
    except Exception as exc:
        print(f"[targets] WARNING: could not record team total for {team} {season}: {exc}")


def _get_existing_receptions(draft_year: int, years: List[int]) -> Dict[str, Dict[int, int]]:
    """Stored receptions from DB: {name_lower: {year: receptions}}."""
    try:
        from dashboard_services.db import get_conn
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT player_id, season, receptions
                    FROM rookie_prospect_source_data
                    WHERE player_id LIKE %s AND season = ANY(%s)
                    AND receptions IS NOT NULL
                """, (_player_id_prefix(draft_year) + "%", years))
                result: Dict[str, Dict[int, int]] = {}
                for _row in cur.fetchall():
                    name = _name_key_from_player_id(_row["player_id"], draft_year)
                    if not name:
                        continue
                    rec_int = int(_row["receptions"])
                    prev = result.setdefault(name, {}).get(int(_row["season"]))
                    if prev is None or rec_int > prev:
                        result[name][int(_row["season"])] = rec_int
                return result
    except Exception:
        return {}


def detect_stranded_target_team_seasons(
    draft_year: int, season: int, teams: List[str],
) -> List[str]:
    """
    Team-seasons stranded by a killed targets run: the fetch persisted
    week marks and/or an accumulated team total (team_targets > 0), but
    the end-of-run player write never happened, so ZERO players have
    stored targets for that team-season.  A rerun trusting those marks
    would skip the weeks and undercount permanently, so these must be
    cleared and refetched.  Team-seasons with any stored player targets
    are never stranded, however partial they look.
    """
    stored = _get_existing_targets(draft_year, [season])
    teams_with_targets = {
        (entry.get("team") or "")
        for yrs in stored.values()
        for entry in [yrs.get(season)] if entry
    }
    tracked = _get_fetched_weeks("targets", [season]).get(season, {})
    baselines = _get_team_targets_baselines([season])
    stranded = []
    for team in teams:
        if team in teams_with_targets:
            continue
        has_marks = bool(tracked.get(team))
        has_total = baselines.get((season, team), 0) > 0
        if has_marks or has_total:
            stranded.append(team)
    return stranded


def clear_target_team_season_tracking(season: int, team: str) -> None:
    """Clear ONLY one team-season's targets tracking (marks + team total)."""
    try:
        from dashboard_services.db import get_conn
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    DELETE FROM rookie_stat_fetched_weeks
                    WHERE kind = 'targets' AND season = %s AND team = %s
                """, (int(season), team))
                cur.execute(_TEAM_TARGETS_DDL)
                cur.execute("""
                    DELETE FROM rookie_targets_team_totals
                    WHERE season = %s AND team = %s
                """, (int(season), team))
    except Exception as exc:
        print(f"[targets] WARNING: could not clear stranded tracking for "
              f"{team} {season}: {exc}")


def _write_player_targets(
    draft_year: int,
    season: int,
    totals_by_name: Dict[str, int],
    receptions_by_name: Optional[Dict[str, Dict[int, int]]] = None,
) -> Tuple[int, int]:
    """
    Write one team-season's composed player targets NOW (its own commit).

    Sanity guard: a composed total below the player's stored receptions
    is impossible (targets >= receptions by definition) and marks a bad
    parse/composition, so it is logged and skipped rather than written.

    Returns (rows_updated, rows_skipped_by_guard).
    """
    from dashboard_services.db import get_conn

    prefix = _player_id_prefix(draft_year)
    updated = 0
    skipped = 0
    with get_conn() as conn:
        with conn.cursor() as cur:
            for name_lower, targets in totals_by_name.items():
                if not name_lower or targets is None:
                    continue
                rec = ((receptions_by_name or {}).get(name_lower) or {}).get(season)
                if rec is not None and int(targets) < int(rec):
                    print(f"[targets] GUARD: skipping {name_lower} {season}: "
                          f"targets {targets} < receptions {rec}")
                    skipped += 1
                    continue
                player_id = prefix + str(name_lower).strip().replace(" ", "_")
                cur.execute("""
                    UPDATE rookie_prospect_source_data
                    SET targets = %s
                    WHERE player_id = %s AND season = %s
                """, (int(targets), player_id, int(season)))
                updated += cur.rowcount or 0
        conn.commit()
    return updated, skipped


def _get_fetched_weeks(kind: str, years: List[int]) -> Dict[int, Dict[str, Set[int]]]:
    """Weeks already folded into stored totals: {season: {team: {week, ...}}}."""
    try:
        from dashboard_services.db import get_conn
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(_FETCHED_WEEKS_DDL)
                cur.execute("""
                    SELECT season, team, week FROM rookie_stat_fetched_weeks
                    WHERE kind = %s AND season = ANY(%s)
                """, (kind, years))
                result: Dict[int, Dict[str, Set[int]]] = {}
                for _row in cur.fetchall():
                    season, team, week = _row["season"], _row["team"], _row["week"]
                    result.setdefault(int(season), {}).setdefault(team or "", set()).add(int(week))
                return result
    except Exception:
        return {}


def _mark_weeks_fetched(kind: str, season: int, team: str, weeks: List[int]) -> None:
    """Record weeks as processed so later runs never request them again."""
    if not weeks:
        return
    try:
        from dashboard_services.db import get_conn
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(_FETCHED_WEEKS_DDL)
                for week in weeks:
                    cur.execute("""
                        INSERT INTO rookie_stat_fetched_weeks (kind, season, week, team)
                        VALUES (%s, %s, %s, %s)
                        ON CONFLICT (kind, season, week, team) DO NOTHING
                    """, (kind, season, int(week), team or ""))
    except Exception as exc:
        print(f"[ingestion] WARNING: could not record fetched {kind} weeks for {season}: {exc}")


def _fetch_games_played_stateless(years: List[int],
                                  today: datetime.date) -> Dict[str, Dict[int, int]]:
    """
    Stateless GP counts: fetch every candidate week for `years` live
    and count distinct game ids per player per season. No DB reads,
    no week marks, no stored totals - the caller owns the result and
    nothing persists. Used by the v4 derive / gate backtest, whose
    population is historical drafted classes: the tracked path's
    stored totals are keyed to the live pipeline's current-class
    rows, so for historical players the tracker can only ever report
    "already fetched" over an empty store (the derive v1.3 failure:
    weeks marked by a run whose in-memory totals died with it).
    Counting mirrors the tracked path exactly (same payload walk,
    same distinct-game-id semantics, same 3-empty-week early stop
    on finished seasons, same light sleep between week calls).
    """
    result: Dict[str, Dict[int, int]] = {}   # name_lower → {yr: count}
    for yr in years:
        player_games: Dict[str, set] = {}   # name_lower → set of game_ids
        empty_streak = 0
        for week in _candidate_weeks(yr, today):
            try:
                data = _cfbd_get(
                    "/games/players",
                    {"year": yr, "week": week, "seasonType": "both"},
                    retries=2,
                )
            except Exception as exc:
                print(f"[cfbd_gp] Week {week} error: {exc}")
                continue

            if data is None:
                # API failure (after retries): counts for this week
                # stay missing; nothing is recorded anywhere.
                print(f"[cfbd_gp] {yr} week {week}: fetch failed (stateless)")
                continue

            if not data:
                empty_streak += 1
                if empty_streak >= 3 and yr < today.year:
                    # 3 consecutive empty weeks on a finished season =
                    # the season has ended; stop early, as tracked.
                    break
                continue

            empty_streak = 0
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

        for player_name, game_ids in player_games.items():
            if game_ids:
                result.setdefault(player_name, {})[yr] = len(game_ids)
    return result


def fetch_cfbd_games_played(draft_year: int,
                            stateless: bool = False) -> Dict[str, Dict[int, int]]:
    """
    Fetch exact games played per player per season via CFBD /games/players.

    Week-incremental and idempotent: rookie_stat_fetched_weeks records which
    (season, week) pairs have already been folded into the stored totals, so a
    run fetches only the next unprocessed week(s) and ADDS their unique game
    counts to the stored season totals.  Weeks are disjoint (a game belongs to
    exactly one week), so adding can never double count.  The returned map
    always carries full season totals (stored + newly fetched), never just
    the new weeks' counts.

    Weeks that have not happened yet are never requested: current-season
    candidates stop at the estimated current week, and a candidate week whose
    fetch comes back empty at that boundary is left unmarked so it is retried
    on the next run once games exist.  When in doubt, a week is NOT marked.

    stateless=True bypasses the tracker entirely (see
    _fetch_games_played_stateless): every candidate week is fetched
    live, counts are returned in memory only, and NO week is marked
    and NO stored total is read. The derive / backtest path needs
    this: its totals are never persisted, so the tracked path's
    marks would claim weeks whose counts no longer exist anywhere.

    Returns:
        {player_name_lower: {year: games_played_total}}
    """
    if not CFBD_KEY:
        print("[cfbd_gp] No CFBD_API_KEY - skipping games-played lookup")
        return {}

    years  = [draft_year - 1, draft_year - 2, draft_year - 3]
    result: Dict[str, Dict[int, int]] = {}   # name_lower → {yr: total count}
    today  = _today()

    if stateless:
        return _fetch_games_played_stateless(years, today)

    # Stored season totals + which weeks produced them.
    db_games = _get_existing_games_played(draft_year, years)
    tracked  = _get_fetched_weeks("gp", years)

    for yr in years:
        candidates = _candidate_weeks(yr, today)
        done: Set[int] = set(tracked.get(yr, {}).get("", set()))
        has_db_data = any(yr in counts for counts in db_games.values())

        # Legacy bootstrap: stored totals exist but no week tracking (they
        # came from the old full-fetch code or a backfill, so they are
        # complete for every candidate week).  Adopt the candidate weeks as
        # fetched so the first incremental run does not re-fetch a whole
        # season that is already in the DB.
        if not done and has_db_data and candidates:
            _mark_weeks_fetched("gp", yr, "", candidates)
            done = set(candidates)
            print(f"[cfbd_gp] {yr}: adopted existing GP totals; "
                  f"weeks {candidates[0]}-{candidates[-1]} marked fetched (bootstrap)")

        # Seed with stored totals; newly fetched weeks add on top below.
        for name_lower, yr_counts in db_games.items():
            if yr in yr_counts:
                result.setdefault(name_lower, {})[yr] = yr_counts[yr]

        weeks_to_fetch = [w for w in candidates if w not in done]
        if not weeks_to_fetch:
            print(f"[cfbd_gp] Skipping {yr} - all candidate weeks already fetched")
            continue

        player_games: Dict[str, set] = {}   # name_lower → set of game_ids
        fetched_ok: List[int] = []
        empty_streak = 0

        for week in weeks_to_fetch:
            try:
                data = _cfbd_get(
                    "/games/players",
                    {"year": yr, "week": week, "seasonType": "both"},
                    retries=2,
                )
            except Exception as exc:
                print(f"[cfbd_gp] Week {week} error: {exc}")
                continue

            if data is None:
                # API failure (after retries): leave the week unmarked so it
                # is retried next run; never mistake a failure for "no games".
                print(f"[cfbd_gp] {yr} week {week}: fetch failed, will retry next run")
                continue

            if not data:
                empty_streak += 1
                # An empty week may be recorded as processed ONLY when it is
                # safely in the past: any week of a finished season, or a week
                # strictly before the estimated current week (a season-data
                # quirk).  An empty week at the current boundary most likely
                # has not been played yet, so leave it unmarked.
                if yr < today.year or week < _estimated_current_week(today):
                    fetched_ok.append(week)
                if empty_streak >= 3:
                    # 3 consecutive empty weeks = season has ended; stop early
                    if yr < today.year:
                        # Finished season: the remaining weeks are final too.
                        fetched_ok.extend(w for w in weeks_to_fetch if w > week)
                    break
                continue

            empty_streak = 0
            fetched_ok.append(week)

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

        if fetched_ok:
            _mark_weeks_fetched("gp", yr, "", fetched_ok)

        # Accumulate: stored total + unique games from the newly fetched
        # weeks (disjoint from the weeks behind the stored total).
        for player_name, game_ids in player_games.items():
            base = db_games.get(player_name, {}).get(yr, 0)
            result.setdefault(player_name, {})[yr] = base + len(game_ids)

    return result


def _get_current_year():
    from datetime import date as _d
    return _d.today().year


# Name mapping: {cfbd_name: desired_display_name}
CFBD_NAME_MAPPINGS = {
    "kevin concepcion": "K.C. Concepcion",
    # Alabama WR is seeded under his full name; listings and CFBD often
    # carry the shortened form.  Stats under either spelling are his.
    "ryan williams": "Ryan Coleman-Williams",
    # Add more mappings as needed
    # "cfbd_name": "Display Name"
}


_ALIAS_TO_CANONICAL: Optional[Dict[str, str]] = None


def _alias_to_canonical() -> Dict[str, str]:
    """
    {normalized alias: normalized canonical name} built from
    CFBD_NAME_MAPPINGS in both directions, so a name resolves to the same
    key whether it arrives in the CFBD spelling or the display spelling.
    """
    global _ALIAS_TO_CANONICAL
    if _ALIAS_TO_CANONICAL is None:
        table: Dict[str, str] = {}
        for cfbd_name, display_name in CFBD_NAME_MAPPINGS.items():
            canonical = normalize_name(display_name)
            table[normalize_name(cfbd_name)] = canonical
            table[canonical] = canonical
        _ALIAS_TO_CANONICAL = table
    return _ALIAS_TO_CANONICAL


def canonical_prospect_key(name: str) -> str:
    """
    The join key for a prospect's stats: normalize_name plus alias
    resolution from CFBD_NAME_MAPPINGS.  "Ryan Williams" and
    "Ryan Coleman-Williams" share one key, so seasons filed under either
    spelling attach to the same (seeded) prospect instead of splitting
    across a duplicate row.
    """
    key = normalize_name(name)
    return _alias_to_canonical().get(key, key)


def _rekey_by_canonical(mapping: Dict[str, Any]) -> Dict[str, Any]:
    """
    Re-key a {name: {year: ...}} side map (games played, PPA, targets)
    through canonical_prospect_key so it joins against canonical stat
    keys.  When two raw spellings land on one canonical key, their year
    dicts are merged instead of one overwriting the other.
    """
    out: Dict[str, Any] = {}
    for raw_key, value in (mapping or {}).items():
        ck = canonical_prospect_key(raw_key)
        if ck in out and isinstance(out[ck], dict) and isinstance(value, dict):
            merged = dict(out[ck])
            merged.update(value)
            out[ck] = merged
        else:
            out[ck] = value
    return out


def fetch_cfbd_player_ppa(draft_year: int) -> Dict[str, Dict[int, Dict[str, float]]]:
    """
    Fetch per-player Predicted Points Added (PPA) from CFBD for the 3 seasons
    before `draft_year`. PPA is the best free proxy for PFF's 0-100 film grades.

    Endpoint: GET /ppa/players/season?year=X&seasonType=regular
    Returns per-player averagePPA broken down by play type.

    Returns:
        {player_name_lower: {year: {"passing_ppa": float, "rushing_ppa": float,
                                     "receiving_ppa": float, "overall_ppa": float}}}
    Returns {} silently if CFBD_API_KEY is not set or the fetch fails.
    Fail-soft: missing PPA data must never break the pipeline.
    """
    if not CFBD_KEY:
        print("[cfbd_ppa] No CFBD_API_KEY - skipping PPA fetch")
        return {}

    years = [draft_year - 1, draft_year - 2, draft_year - 3]
    result: Dict[str, Dict[int, Dict[str, float]]] = {}

    for yr in years:
        try:
            data = _cfbd_get(
                "/ppa/players/season",
                {"year": yr, "seasonType": "both"},
                retries=2,
            ) or []
        except Exception as exc:
            print(f"[cfbd_ppa] Year {yr} error: {exc}")
            continue

        count = 0
        for row in data:
            try:
                name = (row.get("player") or "").strip().lower()
                if not name:
                    continue
                # averagePPA is a dict like {"passing": 0.25, "rushing": 0.12, ...}
                # or a flat number depending on API version; handle both
                avg_ppa = row.get("averagePPA") or {}
                if isinstance(avg_ppa, dict):
                    passing = float(avg_ppa.get("passing") or 0.0)
                    rushing = float(avg_ppa.get("rushing") or 0.0)
                    receiving = float(avg_ppa.get("receiving") or 0.0)
                else:
                    # Flat number: treat as overall
                    passing = rushing = receiving = 0.0
                overall = float(row.get("averagePPA") or 0.0) if not isinstance(avg_ppa, dict) else (passing + rushing + receiving)
                result.setdefault(name, {})[yr] = {
                    "passing_ppa": round(passing, 4),
                    "rushing_ppa": round(rushing, 4),
                    "receiving_ppa": round(receiving, 4),
                    "overall_ppa": round(overall, 4),
                }
                count += 1
            except (TypeError, ValueError, AttributeError):
                continue
        print(f"[cfbd_ppa] Loaded PPA for {yr}: {count} players")

    return result


# ─────────────────────────────────────────────────────────────────────────────
# New CFBD sources (model v2.0): recruiting pedigree, WEPA, team context
# ─────────────────────────────────────────────────────────────────────────────

def fetch_cfbd_recruiting(draft_year: int) -> List[Dict[str, Any]]:
    """
    Fetch 247 Composite recruiting rankings from CFBD.

    Endpoint: GET /recruiting/players?year={recruit_year}
    A draft class is fed by recruiting classes ~3-5 years earlier, so we pull
    all three and match prospects to their best (highest-rated) entry.

    Returns a list of:
        {name, name_lower, position, stars, rating, ranking,
         committed_school, recruit_year}
    Returns [] silently if CFBD_API_KEY is not set or the fetch fails.
    Fail-soft: missing recruiting data must never break the pipeline.
    """
    if not CFBD_KEY:
        print("[cfbd_recruit] No CFBD_API_KEY - skipping recruiting fetch")
        return []

    recruit_years = [draft_year - 5, draft_year - 4, draft_year - 3]
    recruits: List[Dict[str, Any]] = []

    for yr in recruit_years:
        try:
            data = _cfbd_get("/recruiting/players", {"year": yr}, retries=2) or []
        except Exception as exc:
            print(f"[cfbd_recruit] Year {yr} error: {exc}")
            continue

        count = 0
        for row in data:
            try:
                name = (row.get("name") or "").strip()
                if not name:
                    continue
                recruits.append({
                    "name":             name,
                    "name_lower":       normalize_name(name),
                    "position":         (row.get("position") or "").strip().upper(),
                    "stars":            row.get("stars"),
                    "rating":           row.get("rating"),      # 247 Composite, ~0.70-1.00
                    "ranking":          row.get("ranking"),     # national rank
                    "committed_school": row.get("committedTo") or row.get("school"),
                    "recruit_year":     yr,
                })
                count += 1
            except (TypeError, ValueError, AttributeError):
                continue
        print(f"[cfbd_recruit] Loaded {count} recruits for class of {yr}")

    return recruits


def _wepa_metric_value(raw: Dict[str, Any]) -> Optional[float]:
    """
    Best-effort extraction of a single per-play efficiency number from a WEPA
    player record. CFBD's WEPA field naming is not pinned by our key tier, so
    we try a prioritized candidate list and return None when nothing matches.
    A None here makes the whole WEPA blend a no-op downstream - never a guess.
    """
    candidates = (
        "averagePPA", "ppa", "predictedPointsAdded", "expectedPointsAdded",
        "epa", "adjEPA", "pointsAdded", "wepa",
    )
    for key in candidates:
        val = raw.get(key)
        if isinstance(val, dict):
            # e.g. {"passing": 0.25, "rushing": 0.12} - take the max play-type
            nums = [v for v in val.values() if isinstance(v, (int, float))]
            if nums:
                return float(max(nums))
            continue
        if isinstance(val, (int, float)):
            return float(val)
    return None


def _percentile_rank(sorted_vals: List[float], x: float) -> float:
    """Percentile rank of x within sorted_vals, scaled 0-100."""
    if not sorted_vals:
        return 50.0
    import bisect
    idx = bisect.bisect_left(sorted_vals, x)
    return round(idx / len(sorted_vals) * 100.0, 2)


def fetch_cfbd_wepa(draft_year: int) -> Dict[str, Dict[int, Dict[str, Any]]]:
    """
    Fetch opponent-adjusted WEPA player metrics from CFBD.

    Endpoints (Patreon-tier key required - free-tier keys get 403 and this
    returns {} gracefully):
        GET /wepa/players/passing?year=X
        GET /wepa/players/rushing?year=X

    Returns:
        {name_lower: {year: {"wepa_type": "passing"|"rushing",
                             "adj_efficiency_score": 0-100 percentile | None,
                             "raw": <full record>}}}
    Only seasons with a recognized efficiency field get a score; everything
    else is stored raw for inspection. Fail-soft: never breaks the pipeline.
    """
    if not CFBD_KEY:
        print("[cfbd_wepa] No CFBD_API_KEY - skipping WEPA fetch")
        return {}

    years = [draft_year - 1, draft_year - 2, draft_year - 3]
    result: Dict[str, Dict[int, Dict[str, Any]]] = {}

    for wepa_type in ("passing", "rushing"):
        for yr in years:
            try:
                data = _cfbd_get(
                    f"/wepa/players/{wepa_type}", {"year": yr}, retries=2
                ) or []
            except Exception as exc:
                print(f"[cfbd_wepa] {wepa_type} {yr} error: {exc}")
                continue

            # First pass: extract raw efficiency values for percentile ranking
            scored: List[Tuple[str, float, Dict]] = []
            for row in data:
                try:
                    name = (row.get("name") or row.get("player") or "").strip()
                    if not name:
                        continue
                    val = _wepa_metric_value(row)
                    if val is None:
                        continue
                    scored.append((normalize_name(name), val, row))
                except (TypeError, ValueError, AttributeError):
                    continue

            if not scored and data:
                sample_keys = sorted({k for row in data[:5] for k in row.keys()})
                print(f"[cfbd_wepa] {wepa_type} {yr}: {len(data)} rows but no "
                      f"recognized efficiency field (saw keys: {sample_keys[:12]}) - "
                      f"storing raw only")

            vals_sorted = sorted(v for _, v, _ in scored)
            for name_lower, val, row in scored:
                score = _percentile_rank(vals_sorted, val)
                result.setdefault(name_lower, {})[yr] = {
                    "wepa_type": wepa_type,
                    "adj_efficiency_score": score,
                    "raw": row,
                }
            print(f"[cfbd_wepa] Loaded {wepa_type} for {yr}: "
                  f"{len(scored)} scored / {len(data)} rows")

    return result


def fetch_cfbd_team_context(draft_year: int) -> Dict[Tuple[str, int], Dict[str, Any]]:
    """
    Fetch per-team context for the competition adjustment:
      - SP+ ratings:      GET /ratings/sp?year=X  (rating, offense, defense, sos)
      - 247 Team Talent:  GET /talent?year=X      (composite talent points)

    Returns:
        {(school_lower, year): {"sp_rating": float|None, "sp_offense": ...,
                                "sp_defense": ..., "sp_sos": ..., "talent": ...}}
    Returns {} silently if CFBD_API_KEY is not set or the fetch fails.
    """
    if not CFBD_KEY:
        print("[cfbd_ctx] No CFBD_API_KEY - skipping team-context fetch")
        return {}

    years = [draft_year - 1, draft_year - 2, draft_year - 3]
    context: Dict[Tuple[str, int], Dict[str, Any]] = {}

    def _f(v):
        try:
            return float(v) if v is not None else None
        except (TypeError, ValueError):
            return None

    for yr in years:
        try:
            sp_data = _cfbd_get("/ratings/sp", {"year": yr}, retries=2) or []
        except Exception as exc:
            print(f"[cfbd_ctx] SP+ {yr} error: {exc}")
            sp_data = []
        for row in sp_data:
            try:
                team = (row.get("team") or "").strip().lower()
                if not team:
                    continue
                off = row.get("offense") or {}
                deff = row.get("defense") or {}
                entry = context.setdefault((team, yr), {})
                entry["sp_rating"]  = _f(row.get("rating"))
                entry["sp_offense"] = _f(off.get("rating") if isinstance(off, dict) else None)
                entry["sp_defense"] = _f(deff.get("rating") if isinstance(deff, dict) else None)
                entry["sp_sos"]     = _f(row.get("sos"))
            except (TypeError, ValueError, AttributeError):
                continue

        try:
            talent_data = _cfbd_get("/talent", {"year": yr}, retries=2) or []
        except Exception as exc:
            print(f"[cfbd_ctx] talent {yr} error: {exc}")
            talent_data = []
        for row in talent_data:
            try:
                school = (row.get("school") or "").strip().lower()
                if not school:
                    continue
                context.setdefault((school, yr), {})["talent"] = _f(row.get("talent"))
            except (TypeError, ValueError, AttributeError):
                continue

        n_yr = sum(1 for (s, y) in context if y == yr)
        print(f"[cfbd_ctx] Loaded team context for {yr}: {n_yr} teams")

    return context


def _filter_years_needing_fetch(
    all_years: List[int],
    draft_year: int,
    player_ids: Optional[List[str]] = None,
) -> List[int]:
    """
    Incremental fetch logic: past seasons only need to be fetched once -
    but for the WHOLE class, per player.  Returns the subset of all_years
    that actually need API calls:
    - Always includes the current season (all_years[0]) for fresh data
    - Includes a past season when ANY player in the run lacks a stored row
      for it.  (The old gate was class-wide: one player's missed season
      stayed missing forever because his teammates had rows for that year.)
    - Skips a past season only when every player in the run has it stored.

    player_ids: the current run's prospect ids.  When omitted, the class
    roster is read from rookie_prospects.  When coverage cannot be
    determined (no roster on record, DB error), falls back to fetching
    everything (safe).
    """
    if not all_years:
        return []

    current_season = all_years[0]
    past_seasons = all_years[1:]
    if not past_seasons:
        return [current_season]

    try:
        from dashboard_services.db import get_conn
        with get_conn() as conn:
            with conn.cursor() as cur:
                roster = [pid for pid in (player_ids or []) if pid]
                if not roster:
                    cur.execute(
                        "SELECT player_id FROM rookie_prospects "
                        "WHERE draft_class_year = %s",
                        (draft_year,),
                    )
                    roster = [row["player_id"] for row in cur.fetchall()]
                if not roster:
                    # No roster to check coverage against: fetch all (safe).
                    return sorted(all_years)
                # The source table keys on player_id (ROOKIE_<draft_year>_...);
                # it has no draft_class_year column.
                cur.execute(
                    """
                    SELECT player_id, season FROM rookie_prospect_source_data
                    WHERE player_id = ANY(%s) AND season = ANY(%s)
                    """,
                    (roster, past_seasons),
                )
                have: Dict[str, Set[int]] = {}
                for row in cur.fetchall():
                    have.setdefault(row["player_id"], set()).add(row["season"])
    except Exception:
        # If DB check fails, fall back to fetching everything (safe)
        return sorted(all_years)

    needed = [current_season]  # Always refresh current season
    for yr in past_seasons:
        missing = [pid for pid in roster if yr not in have.get(pid, set())]
        if missing:
            print(f"[cfbd] Fetching {yr} - {len(missing)} of {len(roster)} "
                  f"prospects lack stored data for it")
            needed.append(yr)
        else:
            print(f"[cfbd] Skipping {yr} - all {len(roster)} prospects have it in DB")

    return sorted(needed)


def fetch_cfbd_college_stats(
    draft_year: int,
    fetch_games_played: bool = False,
    skip_sagarin: bool = False,
    player_ids: Optional[List[str]] = None,
    gp_stateless: bool = False,
) -> Dict[str, List[Dict]]:
    """
    Fetch college stats from CFBD for the 3 seasons before `draft_year`.
    Returns {canonical_prospect_key: [season_dict, ...]} sorted
    oldest→newest; look seasons up with canonical_prospect_key(name).
    Requires CFBD_API_KEY env var; returns {} silently if not set.

    Args:
        fetch_games_played: If True, fetch exact games-played counts via the
            /games/players endpoint (51 extra API calls per draft class).
            Default False to conserve rate-limit budget; per-game rates fall
            back to assuming 12 games when disabled.
        gp_stateless: Forwarded to fetch_cfbd_games_played; only
            meaningful with fetch_games_played=True. Stateless GP
            ignores the DB week tracker (no reads, no marks) - the
            derive / backtest mode, whose GP totals are never
            persisted. Default False = the live pipeline's tracked,
            week-incremental behavior.
    """

    if not CFBD_KEY:
        print("[cfbd] No CFBD_API_KEY set - skipping college stats")
        return {}

    all_years = [draft_year - 1, draft_year - 2, draft_year - 3, draft_year - 4]
    # Incremental fetch: past seasons are fetched until EVERY player in
    # the run has them stored (per-player gate), then skipped.  Old data
    # doesn't change, so don't waste API calls re-fetching covered years.
    years = _filter_years_needing_fetch(all_years, draft_year, player_ids=player_ids)

    try:
        # Team season totals for market share / dominator calculation
        team_stats: Dict[int, Dict] = {}
        for yr in years:
            data = _cfbd_get("/stats/season", {"year": yr, "seasonType": "both"})
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
            except Exception as exc:
                print(f"[cfbd] ERROR processing team stats for {yr} - {type(exc).__name__}: {exc}")
                team_stats[yr] = {}

        # Games-played lookup - skipped by default to conserve rate-limit budget.
        # It adds 51 API calls per draft class (17 weeks × 3 years) but only
        # improves per-game rate accuracy; total yards/TDs/dominator_rating are
        # unaffected.  Pass fetch_games_played=True to enable.
        games_played_map: Dict[str, Dict[int, int]] = {}
        if fetch_games_played:
            print("[cfbd] Fetching games played from /games/players endpoint")
            try:
                games_played_map = fetch_cfbd_games_played(
                    draft_year, stateless=gp_stateless)
                print(f"[cfbd] Games played resolved for {len(games_played_map)} players")
            except Exception as exc:
                print(f"[cfbd] WARNING: games-played fetch failed ({exc}), will default to None")
                games_played_map = {}
        else:
            print("[cfbd] Skipping games-played lookup (saves 51 API calls; per-game rates default to 12 games)")

        # Player season stats - indexed by name and by player ID
        print("[cfbd] Fetching player season stats")
        by_name: Dict[int, Dict[str, List]] = {}   # {yr: {name_lower: [rows]}}
        by_id:   Dict[int, Dict[int, List]] = {}   # {yr: {player_id: [rows]}}
        for yr in years:
            print(f"[cfbd] Fetching player stats for {yr}")
            try:
                data = _cfbd_get("/stats/player/season",
                                 {"year": yr, "seasonType": "both"}) or []
                bn: Dict[str, List] = {}
                bi: Dict[int, List] = {}
                for row in data:
                    n  = canonical_prospect_key(row.get("player") or "")
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
                print(f"[cfbd] ERROR loading player stats for {yr} - {type(exc).__name__}: {exc}")
                by_name[yr] = {}
                by_id[yr] = {}

        # Compute depth chart ranks before building season summaries
        print("[cfbd] Computing depth chart ranks")
        try:
            depth_ranks = _compute_depth_ranks(by_name, years)
            print(f"[cfbd] Depth rank computed for {len(depth_ranks)} player-year combinations")
        except Exception as exc:
            print(f"[cfbd] WARNING: depth rank computation failed ({exc}), skipping")
            depth_ranks = {}

        # Fetch per-player PPA (replaces PFF film grades with a free objective metric)
        print("[cfbd] Fetching player PPA")
        try:
            ppa_map = fetch_cfbd_player_ppa(draft_year)
            print(f"[cfbd] PPA loaded for {len(ppa_map)} players")
        except Exception as exc:
            print(f"[cfbd] WARNING: PPA fetch failed ({exc}), continuing without PPA")
            ppa_map = {}

        # Collapse into per-player season lists keyed by canonical name.
        # The GP / PPA side maps arrive keyed by their own raw name
        # spellings, so re-key them through the same canonical function
        # before joining.
        gp_by_key = _rekey_by_canonical(games_played_map)
        ppa_by_key = _rekey_by_canonical(ppa_map)
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
                        gp = (gp_by_key.get(name) or {}).get(yr)
                        season_dict = _build_cfbd_season(rows, team_stats.get(yr, {}), yr, gp,
                                                         skip_sagarin=skip_sagarin)
                        dr_info = depth_ranks.get((name, yr), {})
                        season_dict["depth_rank"] = dr_info.get("depth_rank")
                        season_dict["position_group_size"] = dr_info.get("position_group_size")
                        # Merge PPA data (free replacement for PFF film grades)
                        ppa_info = (ppa_by_key.get(name) or {}).get(yr, {})
                        if ppa_info:
                            season_dict["cfbd_passing_ppa"] = ppa_info.get("passing_ppa")
                            season_dict["cfbd_rushing_ppa"] = ppa_info.get("rushing_ppa")
                            season_dict["cfbd_receiving_ppa"] = ppa_info.get("receiving_ppa")
                            season_dict["cfbd_overall_ppa"] = ppa_info.get("overall_ppa")
                        seasons.append(season_dict)
                    except Exception as exc:
                        print(f"[cfbd] ERROR building season for '{name}' year {yr} - {type(exc).__name__}: {exc}")
                if seasons:
                    seasons.sort(key=lambda s: s["season"])
                    # `name` is already the canonical key (by_name was
                    # keyed through canonical_prospect_key, which applies
                    # CFBD_NAME_MAPPINGS), so it is the storage key as-is.
                    result[name] = seasons
            except Exception as exc:
                print(f"[cfbd] ERROR processing player '{name}' - {type(exc).__name__}: {exc}")

        print(f"[cfbd] COMPLETE: Loaded stats for {len(result)} players (draft class {draft_year})")

        # Targets parsed from CFBD play-by-play TEXT (/plays) by
        # fetch_cfbd_targets_from_plays: the structured "Target" stat type
        # is tagged on only a small subset of plays (verified 2026-10-13:
        # Ohio State vs Ball State had ~35 pass attempts but only 5 Target
        # rows, while Reception rows were complete at 31), and neither CFBD
        # nor ESPN box scores carry targets - so the old /plays/stats merge
        # (fetch_cfbd_targets) stays disconnected. Play text names the
        # intended receiver on nearly every pass play, and every parsed
        # game is validated against the box score (derived receptions must
        # match REC exactly, >= 85% of pass plays must name a receiver) or
        # discarded. Best-effort: skips silently when unavailable.
        try:
            teams_by_year: Dict[int, set] = {}
            for seasons in result.values():
                for s in seasons:
                    yr = s.get("season")
                    team = s.get("team")
                    if yr and team:
                        teams_by_year.setdefault(yr, set()).add(team)
            teams_by_year = {yr: sorted(teams) for yr, teams in teams_by_year.items()}
            targets_map = fetch_cfbd_targets_from_plays(draft_year, teams_by_year)
            if targets_map:
                targets_by_key = _rekey_by_canonical(targets_map)
                merged = 0
                for name, seasons in result.items():
                    t_by_year = targets_by_key.get(name)
                    if not t_by_year:
                        continue
                    for s in seasons:
                        yr = s.get("season")
                        t = t_by_year.get(yr) if yr else None
                        if t:
                            # Totals from the targets fetch are stored+new
                            # (additive); only overwrite when it carries a
                            # value so a partial entry can never blank a stat.
                            if t.get("targets") is not None:
                                s["targets"] = t["targets"]
                            if t.get("target_share") is not None:
                                s["target_share"] = t["target_share"]
                            merged += 1
                print(f"[cfbd] Merged targets/target_share for {merged} player-seasons")
        except Exception as exc:
            print(f"[cfbd] Target merge failed - {type(exc).__name__}: {exc}")

        return result
        
    except Exception as exc:
        print(f"[cfbd] FAILED: Unexpected error fetching college stats for {draft_year} - {type(exc).__name__}: {exc}")
        return {}


# ─────────────────────────────────────────────────────────────────────────────
# Supplementary source 3 - ESPN athlete search  (no auth required)
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


_NAME_SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}


# ─────────────────────────────────────────────────────────────────────────────
# Age fallback - estimation from Sportradar experience field
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
                logging.getLogger(__name__).debug("suppressed exception", exc_info=True)
    try:
        return int(s)
    except ValueError:
        return None



def _parse_sportradar_prospect(raw: Dict, draft_year: int) -> Optional[Dict]:
    """
    Convert a single Sportradar Draft API prospect object to our normalized format.

    Actual Sportradar fields (Draft API v1):
        name, first_name, last_name - player name
        position                    - e.g. "WR", "RB"
        height                      - integer, total inches
        weight                      - integer, pounds
        team_name                   - college team name string
        conference                  - object: {name, alias}
        experience                  - "SR" / "JR" / "SO" / "FR"
        birth_place                 - city/state string e.g. "Suwanee, GA, USA"
        top_prospect                - boolean

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

        # birth_place is "City, ST, USA" - no DOB available from this endpoint
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
            # Internal fields used during merge - stripped before normalization
            "_conference":      conference,
            "_experience":      experience,  # SR/JR/SO/FR for age estimation
        }
    except Exception as exc:
        print(f"[sportradar] ERROR parsing prospect: {raw.get('name', 'UNKNOWN')} - {type(exc).__name__}: {exc}")
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
        print("[sportradar] FAILED: No SPORTRADAR_API_KEY set - cannot fetch prospects")
        return []

    if SPORTRADAR_ACCESS == "trial":
        # Verified 2026-10-09: trial keys get 403 on the draft-prospects
        # endpoint, even for completed drafts. Skip the doomed call.
        print("[sportradar] SKIP: trial access has no draft-prospects endpoint - using seed data")
        return []

    try:
        data = _sportradar_get(f"{draft_year}/prospects.json")
    except Exception as exc:
        print(f"[sportradar] FAILED: Unexpected error fetching prospects for {draft_year} - {type(exc).__name__}: {exc}")
        return []
        
    if not data:
        print(f"[sportradar] FAILED: No data returned for {draft_year} - API may be down or key invalid")
        return []

    raw_list = data.get("prospects") or (data if isinstance(data, list) else [])
    if not raw_list:
        print(f"[sportradar] FAILED: Empty prospects list for {draft_year} - response format may have changed")
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
            print(f"[sportradar] ERROR parsing prospect at index {i} - {type(exc).__name__}: {exc}")
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


def fetch_cfbd_targets(
    draft_year: int,
    teams_by_year: Dict[int, List[str]],
    adopt_stored: bool = True,
) -> Dict[str, Dict[int, Dict[str, Any]]]:
    """
    Target counts + target share from CFBD play-by-play stats (free, no PFF).

    Week-incremental, same discipline as the games-played fetch:
    rookie_stat_fetched_weeks (kind='targets', tracked per team) records which
    (season, week, team) slices are already folded into the stored totals, so
    a run fetches only unprocessed weeks and ADDS their counts.  Weeks are
    disjoint, so written totals are always stored + new, never a partial
    slice; a run with zero new weeks returns the stored totals unchanged.

    - Current season: per-week fetches (/plays/stats accepts a week param),
      capped at the estimated current week.  A team-week that comes back
      empty at that boundary is left unmarked and retried next run.
    - Past seasons are final, so one season-level call per team is exact:
      fetched that way once, then all weeks are marked.  Stored totals with
      no tracking (legacy) are adopted instead of re-fetched; one
      season-level call per team then only establishes the share denominator.

    Team target totals (the share denominator) persist in
    rookie_targets_team_totals so incremental runs add to them.

    Returns {name_lower: {year: {"targets": int, "target_share": float, "team": str}}}.
    Returns {} on any failure - the pipeline continues without targets
    (the model falls back to receptions-based proxies).

    Args:
        draft_year: The draft class year (for logging)
        teams_by_year: {season_year: [team names]} - teams to fetch per season
        adopt_stored: When False (standalone backfill), never adopt stored
            totals: fetch complete data for every requested season/team so
            players whose stored targets are missing actually get filled.
    """
    if not CFBD_KEY or not teams_by_year:
        if not CFBD_KEY:
            print("[targets] No CFBD_API_KEY - skipping")
        return {}

    try:
        types = _cfbd_get("/plays/stats/types", retries=2) or []
        print(f"[targets] Got {len(types)} stat types from API")
    except Exception as exc:
        print(f"[targets] stat-types lookup failed - {type(exc).__name__}: {exc}")
        return {}
    target_type_id = None
    for t in types:
        tname = str(t.get("name") or t.get("statType") or t.get("stat_type") or "").lower()
        if "target" in tname:
            target_type_id = t.get("id") or t.get("statTypeId") or t.get("stat_type_id")
            print(f"[targets] using stat type '{t.get('name')}' (id={target_type_id})")
            break
    if not target_type_id:
        print("[targets] no targets stat type in /plays/stats/types - skipping")
        print(f"[targets] Available types: {[str(t.get('name'))[:30] for t in types[:10]]}")
        return {}

    def _fetch_rows(season_year: int, team: str, week: Optional[int] = None):
        """One /plays/stats call; None signals an API failure (retry later)."""
        params: Dict[str, Any] = {"year": season_year, "team": team,
                                  "statTypeId": target_type_id}
        if week is not None:
            params["week"] = week
        try:
            return _cfbd_get("/plays/stats", params, retries=2)
        except Exception as exc:
            print(f"[targets] {team} {season_year} failed - {type(exc).__name__}: {exc}")
            return None

    def _aggregate(rows) -> Tuple[Dict[str, int], int]:
        by_athlete: Dict[str, int] = {}
        team_total = 0
        for r in rows or []:
            name = str(
                r.get("athleteName") or r.get("athlete") or r.get("player") or r.get("name") or ""
            ).strip()
            count = _safe_int(r.get("stat") or r.get("count") or r.get("value")) or 0
            if not name or count <= 0:
                continue
            by_athlete[name.lower()] = by_athlete.get(name.lower(), 0) + count
            team_total += count
        return by_athlete, team_total

    today = _today()
    seasons = sorted(teams_by_year)
    stored = _get_existing_targets(draft_year, seasons)
    tracked = _get_fetched_weeks("targets", seasons)
    baselines = _get_team_targets_baselines(seasons)
    result: Dict[str, Dict[int, Dict[str, Any]]] = {}

    def _put(name: str, season_year: int, targets: int, team: str) -> None:
        denom = baselines.get((season_year, team), 0)
        result.setdefault(name, {})[season_year] = {
            "targets": targets,
            "target_share": round(targets / denom, 4) if denom else 0.0,
            "team": team,
        }

    def _seed_stored(season_year: int) -> None:
        """Return stored totals unchanged (zero-new-weeks no-op path)."""
        for name, yrs in stored.items():
            entry = yrs.get(season_year)
            if entry:
                _put(name, season_year, entry["targets"], entry.get("team") or "")

    def _establish_denominator(season_year: int, team: str) -> None:
        """One season-level call whose only job is the share denominator."""
        rows = _fetch_rows(season_year, team)
        if rows is not None:
            _, team_total = _aggregate(rows)
            baselines[(season_year, team)] = team_total
            _record_team_targets(season_year, team, team_total)

    for season_year in seasons:
        teams = sorted(set(teams_by_year[season_year]))
        if not teams:
            continue
        candidates = _candidate_weeks(season_year, today)
        season_tracked = tracked.get(season_year, {})
        has_stored = any(season_year in yrs for yrs in stored.values())
        print(f"[targets] Fetching {season_year} for {len(teams)} teams")

        if season_year < today.year:
            # ── Finished season: season-level fetch is exact and cheap ──
            all_done = bool(candidates) and adopt_stored and all(
                set(candidates) <= season_tracked.get(t, set()) for t in teams)
            if all_done:
                _seed_stored(season_year)
                continue
            if adopt_stored and not season_tracked and has_stored:
                # Bootstrap: adopt stored totals; never re-fetch player data.
                for team in teams:
                    _establish_denominator(season_year, team)
                    _mark_weeks_fetched("targets", season_year, team, candidates)
                _seed_stored(season_year)
                continue
            for team in teams:
                rows = _fetch_rows(season_year, team)
                if rows is None:
                    continue  # failure: leave unmarked, retried next run
                by_athlete, team_total = _aggregate(rows)
                baselines[(season_year, team)] = team_total
                _record_team_targets(season_year, team, team_total)
                _mark_weeks_fetched("targets", season_year, team, candidates)
                for name, tgts in by_athlete.items():
                    _put(name, season_year, tgts, team)
            continue

        # ── Current season: per-week incremental per team ──
        if adopt_stored and not season_tracked and has_stored:
            # Bootstrap: adopt stored totals; establish denominators only.
            for team in teams:
                _establish_denominator(season_year, team)
                _mark_weeks_fetched("targets", season_year, team, candidates)
            _seed_stored(season_year)
            continue

        est = _estimated_current_week(today)
        additions: Dict[str, Tuple[int, str]] = {}   # name → (added targets, team)
        for team in teams:
            # Backfill mode (adopt_stored=False) ignores prior tracking and
            # stored baselines: it fetches every candidate week so the totals
            # it produces are complete, never a partial slice.
            done = set(season_tracked.get(team, set())) if adopt_stored else set()
            weeks_to_fetch = [w for w in candidates if w not in done]
            if not weeks_to_fetch:
                continue
            fetched_ok: List[int] = []
            new_team_total = 0
            team_complete = True
            for week in weeks_to_fetch:
                rows = _fetch_rows(season_year, team, week=week)
                if rows is None:
                    team_complete = False
                    continue  # failure: leave this team-week unmarked
                if not rows:
                    # Empty team-week (bye / no data): record it only when it
                    # is strictly before the current-week boundary; at the
                    # boundary it may simply not have been played yet.
                    if week < est:
                        fetched_ok.append(week)
                    else:
                        team_complete = False
                    continue
                fetched_ok.append(week)
                wk_athletes, wk_total = _aggregate(rows)
                for n, c in wk_athletes.items():
                    prev_add, _ = additions.get(n, (0, team))
                    additions[n] = (prev_add + c, team)
                new_team_total += wk_total
                time.sleep(0.15)   # light rate-limiting between calls
            if not adopt_stored and not team_complete:
                # Backfill must never write a partial slice as a season
                # total: drop this team's additions entirely.
                additions = {n: v for n, v in additions.items() if v[1] != team}
                new_team_total = 0
            if fetched_ok and (adopt_stored or team_complete):
                # In backfill mode an incomplete team's weeks stay unmarked:
                # nothing was persisted from them, so normal runs must still
                # treat them as unprocessed.
                _mark_weeks_fetched("targets", season_year, team, fetched_ok)
            if new_team_total:
                team_base = baselines.get((season_year, team), 0) if adopt_stored else 0
                team_total_all = team_base + new_team_total
                baselines[(season_year, team)] = team_total_all
                _record_team_targets(season_year, team, team_total_all)

        # Merge: stored totals for everyone, additions on top for players
        # with newly fetched weeks.  Nothing fetched → totals identical to
        # what is already stored (pure no-op, never zeros).  Backfill mode
        # skips the stored base entirely: its totals come from the complete
        # fetch above.
        if adopt_stored:
            _seed_stored(season_year)
        for name, (add, team) in additions.items():
            base = ((stored.get(name, {}).get(season_year) or {}).get("targets", 0)
                    if adopt_stored else 0)
            _put(name, season_year, base + add, team)

    total_athletes = len(result)
    print(f"[targets] collected targets for {total_athletes} athletes across {len(teams_by_year)} seasons")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Targets parsed from play-by-play TEXT (/plays), validated per game
#
# The structured sources are all dead ends for college targets (verified
# 2026-10-13): the /plays/stats "Target" stat type is tagged on only a small
# subset of plays (~5 of ~35 attempts in Ohio State vs Ball State), CFBD box
# scores (/games/players) have no targets stat type, and ESPN college box
# scores omit targets too.  Play TEXT, however, names the intended receiver
# on nearly every pass play:
#   completion:  "... pass complete short right to #32 I.West caught at ..."
#   incomplete:  "... pass incomplete short middle to #89 N.Lautar thrown ..."
# Receiver identity in the text is "#<jersey> <FirstInitial>.<LastName>".
# Because text parsing can silently lie, every game is VALIDATED before its
# targets are used (see fetch_cfbd_targets_from_plays): derived receptions
# must equal the box-score REC line exactly, and >= 85% of the team's pass
# plays must name their receiver, or the game is discarded entirely.
# ─────────────────────────────────────────────────────────────────────────────

_PLAY_TARGET_MIN_COVERAGE = 0.85

# Scoring-summary lines ride along in /plays payloads in a different format
# ("Legend Bey 25 Yd pass from Luke Fahey (Connor Hawkins Kick)").  They are
# never counted as plays outright: live verification (Ohio State 2026) showed
# some completions exist ONLY as a summary line (the detailed entry is
# missing), while others duplicate a detailed entry.  So summary PASS lines
# are collected as per-player CANDIDATES and used only to fill the gap
# between derived receptions and the box score, capped at the box total
# (see _validate_game) - never beyond it, so a duplicate can never overshoot.
_SCORING_SUMMARY_RE = re.compile(r"\bYd\s+(?:pass|run)\s+from\b", re.IGNORECASE)
_SCORING_SUMMARY_PASS_RE = re.compile(
    r"^\s*(.+?)\s+\d+\s+Yd\s+pass\s+from\b", re.IGNORECASE)

# Anchored on the pass verb so the parenthetical tackler "(#2 Q.Scandrett)"
# can never be mistaken for the receiver: the receiver is the FIRST
# "to #<jersey> <I>.<Name>" after "pass complete|incomplete", before any "(".
_PLAY_TARGET_JERSEY_RE = re.compile(
    r"pass\s+(?:complete|incomplete)\b[^()]*?"
    r"(?:\bto|\bintended\s+for)\s+#(\d+)\s+([A-Za-z])\."
    r"([A-Za-z][A-Za-z'.\-]*(?:\s+[A-Za-z][A-Za-z'.\-]*){0,2})",
    re.IGNORECASE,
)
# Fallback for texts that name the receiver without a jersey number.
_PLAY_TARGET_NAME_RE = re.compile(
    r"pass\s+(?:complete|incomplete)\b[^()]*?\bto\s+"
    r"([A-Z][a-zA-Z'\-]+)\s+([A-Z][a-zA-Z'\-]+)",
)


def _norm_last_name(raw: str) -> str:
    """'St. Clair' -> 'stclair'; 'Thompson Jr.' -> 'thompson'."""
    tokens = re.findall(r"[A-Za-z]+", raw or "")
    while tokens and tokens[-1].lower() in _NAME_SUFFIXES:
        tokens.pop()
    return "".join(tokens).lower()


def _canonical_name_key(name: str) -> Tuple[str, str]:
    """
    Validation-only comparison key: (first-name initial, normalized surname
    with suffix tokens stripped).  The box score and the roster do not always
    spell a player the same way (box "Max LeBlanc" vs roster "Maxence
    LeBlanc"), so derived-vs-box REC counts are compared on this key, summed
    per key on both sides - two players sharing a key are summed together,
    never mismatched against each other.  Stored data keeps full names.
    """
    tokens = re.findall(r"[A-Za-z]+", str(name or "").lower())
    while tokens and tokens[-1] in _NAME_SUFFIXES:
        tokens.pop()
    if not tokens:
        return ("", "")
    last = _norm_last_name(" ".join(tokens[1:])) if len(tokens) > 1 else tokens[0]
    return (tokens[0][0], last)


def _canon_sums(counts: Dict[str, int]) -> Dict[Tuple[str, str], int]:
    """Sum a {full_name: n} dict onto canonical comparison keys."""
    out: Dict[Tuple[str, str], int] = {}
    for n, c in counts.items():
        k = _canonical_name_key(n)
        out[k] = out.get(k, 0) + c
    return out


def _build_roster_maps(roster_rows) -> Tuple[Dict, Dict, Set[str]]:
    """
    Receiver-resolution maps for ONE team-season from CFBD /roster rows:
    - by_jersey: {(jersey, last_norm, first_initial): name_lower}
    - by_name:   {(last_norm, first_initial): name_lower} - kept only where
      the key is unique on the roster (ambiguous keys are dropped)
    - all_names: every roster player's name_lower (box-score fallback use)
    """
    by_jersey: Dict[Tuple[int, str, str], str] = {}
    counts: Dict[Tuple[str, str], int] = {}
    names: Dict[Tuple[str, str], str] = {}
    all_names: Set[str] = set()
    for row in roster_rows or []:
        first = str(row.get("firstName") or "").strip()
        last = str(row.get("lastName") or "").strip()
        if not first or not last:
            continue
        name_lower = f"{first} {last}".lower()
        all_names.add(name_lower)
        last_norm = _norm_last_name(last)
        initial = first[0].lower()
        if not last_norm or not initial:
            continue
        jersey = _safe_int(row.get("jersey"))
        if jersey is not None:
            by_jersey[(jersey, last_norm, initial)] = name_lower
        key = (last_norm, initial)
        counts[key] = counts.get(key, 0) + 1
        names[key] = name_lower
    by_name = {k: v for k, v in names.items() if counts[k] == 1}
    return by_jersey, by_name, all_names


def _resolve_receiver(jersey, initial, last_span, by_jersey, by_name) -> Optional[str]:
    """
    Resolve a parsed receiver mention to a roster name_lower, or None.

    `last_span` is the raw text captured after the initial - up to 3 tokens
    that may include trailing play words ("West caught at ...").  Progressively
    shorter surname candidates are tried against the roster maps, longest
    first, so multi-token surnames ("St. Clair") win over their first token.
    """
    if not initial:
        return None
    initial = str(initial).lower()
    tokens = re.findall(r"[A-Za-z][A-Za-z'.\-]*", last_span or "")
    jersey_int = _safe_int(jersey) if jersey is not None else None
    for k in range(len(tokens), 0, -1):
        last_norm = _norm_last_name(" ".join(tokens[:k]))
        if not last_norm:
            continue
        if jersey_int is not None:
            hit = by_jersey.get((jersey_int, last_norm, initial))
            if hit:
                return hit
        hit = by_name.get((last_norm, initial))
        if hit:
            return hit
    return None


def _summary_receiver_name(text, by_jersey, by_name, all_names) -> Optional[str]:
    """
    Receiver name_lower for a scoring-summary PASS line ("Legend Bey 25 Yd
    pass from Luke Fahey ..."), or None when the line is not a pass summary
    or the receiver cannot be resolved to this team's roster.  Summary lines
    carry the FULL name (no jersey/initial abbreviation), so resolution is
    an exact full-name hit first, then the (surname, initial) roster map.
    """
    m = _SCORING_SUMMARY_PASS_RE.match(text or "")
    if not m:
        return None
    raw = " ".join(m.group(1).split()).lower()
    if raw in (all_names or set()):
        return raw
    tokens = re.findall(r"[A-Za-z][A-Za-z'.\-]*", m.group(1))
    if len(tokens) < 2:
        return None
    return _resolve_receiver(None, tokens[0][0], " ".join(tokens[1:]),
                             by_jersey, by_name)


def _play_text(play: Dict[str, Any]) -> str:
    return str(play.get("playText") or play.get("play_text") or "")


def _play_offense(play: Dict[str, Any]) -> str:
    for key in ("offense", "offenseTeam", "offense_team"):
        v = play.get(key)
        if v:
            return str(v).strip()
    return ""


def _pass_attempt_info(play: Dict[str, Any]):
    """
    Classify one /plays row.

    Returns (is_pass_attempt, is_completion, receiver_mention) where
    receiver_mention is (jersey|None, first_initial, last_name_span) or None
    when the play names no receiver (thrown away, intercepted, ...).
    Scoring-summary lines and non-pass plays return (False, False, None).
    """
    text = _play_text(play)
    if not text or _SCORING_SUMMARY_RE.search(text):
        return (False, False, None)
    tl = text.lower()
    ptype = str(play.get("playType") or play.get("play_type") or "").lower()
    verb = None
    if "pass complete" in tl:
        verb = "complete"
    elif "pass incomplete" in tl:
        verb = "incomplete"
    elif "pass intercepted" in tl or ptype == "interception":
        verb = "intercepted"
    if verb is None and "pass" not in ptype:
        return (False, False, None)
    mention = None
    if verb in ("complete", "incomplete"):
        m = _PLAY_TARGET_JERSEY_RE.search(text)
        if m:
            mention = (_safe_int(m.group(1)), m.group(2), m.group(3))
        else:
            m2 = _PLAY_TARGET_NAME_RE.search(text)
            if m2:
                mention = (None, m2.group(1)[0], m2.group(2))
    return (True, verb == "complete", mention)


def _derive_game_targets(plays, team, by_jersey, by_name, roster_names=None):
    """
    Parse one game's plays for `team`'s offense.

    Returns {"targets": {name: n}, "receptions": {name: n},
             "summary": {name: n}, "attempts": int, "named": int} - or None
    when the payload carries no offense field at all, in which case plays
    cannot be safely attributed to this team (a /plays response mixes BOTH
    teams' plays).  A receiver mention that resolves to no roster player
    counts as an attempt but not as named (conservative for the coverage
    gate).  "summary" holds scoring-summary-line completions per receiver:
    candidates for the box-score gap fill in _validate_game, NOT counted in
    targets/receptions here (some summary lines duplicate detailed plays).
    """
    plays = plays or []
    if not any(_play_offense(p) for p in plays):
        return None
    targets: Dict[str, int] = {}
    receptions: Dict[str, int] = {}
    summary: Dict[str, int] = {}
    attempts = 0
    named = 0
    seen_play_ids: Set[Any] = set()
    for play in plays:
        pid = play.get("id")
        if pid is not None:
            if pid in seen_play_ids:
                continue  # same play listed twice must never double count
            seen_play_ids.add(pid)
        if _play_offense(play).lower() != team.lower():
            continue
        sname = _summary_receiver_name(_play_text(play), by_jersey, by_name,
                                       roster_names)
        if sname:
            summary[sname] = summary.get(sname, 0) + 1
            continue
        is_pass, completed, mention = _pass_attempt_info(play)
        if not is_pass:
            continue
        attempts += 1
        if mention is None:
            continue
        name = _resolve_receiver(mention[0], mention[1], mention[2],
                                 by_jersey, by_name)
        if not name:
            continue
        named += 1
        targets[name] = targets.get(name, 0) + 1
        if completed:
            receptions[name] = receptions.get(name, 0) + 1
    return {"targets": targets, "receptions": receptions,
            "summary": summary, "attempts": attempts, "named": named}


def _box_receptions(box_payload, game_id, team, roster_names):
    """
    {name_lower: receptions} for `team` in one game from a /games/players
    payload (receiving category, REC type - the box-score ground truth), or
    None when the game/team cannot be located in the payload.
    """
    for game_obj in box_payload or []:
        if str(game_obj.get("id")) != str(game_id):
            continue
        entries = []
        for t in game_obj.get("teams") or []:
            tname = ""
            for key in ("team", "school", "name"):
                if t.get(key):
                    tname = str(t[key]).strip()
                    break
            rec: Dict[str, int] = {}
            for category in t.get("categories") or []:
                if str(category.get("name") or "").lower() != "receiving":
                    continue
                for stat_type in category.get("types") or []:
                    if str(stat_type.get("name") or "").upper() != "REC":
                        continue
                    for athlete in stat_type.get("athletes") or []:
                        a_name = str(athlete.get("name") or "").lower().strip()
                        if a_name:
                            rec[a_name] = _safe_int(athlete.get("stat")) or 0
            entries.append((tname, rec))
        for tname, rec in entries:
            if tname and tname.lower() == team.lower():
                return rec
        # Fallback: the entry whose receivers sit on this team's roster.
        best, best_overlap = None, 0
        for _tname, rec in entries:
            overlap = len(set(rec) & roster_names)
            if overlap > best_overlap:
                best, best_overlap = rec, overlap
        return best
    return None


def _group_plays_by_game(rows):
    """[(game_id, [plays], week|None)] from a /plays payload, order kept."""
    groups: Dict[Any, List[Dict[str, Any]]] = {}
    order: List[Any] = []
    for p in rows or []:
        gid = p.get("gameId") or p.get("game_id")
        if gid is None:
            continue
        if gid not in groups:
            groups[gid] = []
            order.append(gid)
        groups[gid].append(p)
    out = []
    for gid in order:
        plays = groups[gid]
        week = None
        for p in plays:
            w = _safe_int(p.get("week"))
            if w:
                week = w
                break
        out.append((gid, plays, week))
    return out


def fetch_cfbd_targets_from_plays(
    draft_year: int,
    teams_by_year: Dict[int, List[str]],
    adopt_stored: bool = True,
    on_team_season=None,
) -> Dict[str, Dict[int, Dict[str, Any]]]:
    """
    Target counts + target share parsed from CFBD play-by-play TEXT.

    Same return shape and week-incremental discipline as fetch_cfbd_targets
    (rookie_stat_fetched_weeks, kind='targets', tracked per team): a run
    fetches only unprocessed team-weeks and ADDS their counts to the stored
    totals; a run with zero new weeks returns the stored totals unchanged.

    - Current season: per-week /plays calls, capped at the estimated current
      week.  A week is marked only when EVERY game in it validates.
    - Past seasons: one season-level /plays call per team (per-week fallback
      if that yields nothing); plays are grouped and validated per game, and
      only weeks whose game passed validation are marked - games in
      already-tracked weeks are never re-added (no double counting).

    VALIDATION (per game, mandatory): derived receptions per player must
    EQUAL the /games/players box-score REC for that player-game, and named
    receiver coverage (pass plays naming a resolvable receiver / all pass
    plays: complete + incomplete + intercepted) must be >= 0.85.
    Completions that exist only as scoring-summary lines first fill the
    derived-vs-box gap per player (capped at the box total), and the REC
    comparison runs on canonical (initial, surname) keys so roster/box
    spelling variants cannot false-flag.  A game
    failing either check is discarded whole: nothing is written for it and
    its week stays unmarked.  A provably pass-less game (no pass plays in
    the text AND an empty box-score REC line) validates with zero targets.

    target_share denominator = sum of derived targets for the team-season
    (named receivers only, so numerator and denominator are consistent),
    accumulated in rookie_targets_team_totals exactly like the old fetcher.

    on_team_season (optional): callable(season, team, {name_lower: total})
    invoked the moment a team-season finishes (weeks marked, team total
    recorded), with that team-season's composed totals (stored + new when
    adopt_stored).  It lets a caller persist player targets per team-season
    instead of only at the very end of a multi-hour sweep: a kill mid-run
    then loses at most the in-flight team-season.  Callback exceptions are
    logged, never fatal to the fetch.

    Returns {} on any failure - the pipeline continues without targets.
    """
    if not CFBD_KEY or not teams_by_year:
        if not CFBD_KEY:
            print("[targets-plays] No CFBD_API_KEY - skipping")
        return {}

    today = _today()
    seasons = sorted(teams_by_year)
    stored = _get_existing_targets(draft_year, seasons)
    tracked = _get_fetched_weeks("targets", seasons)
    baselines = _get_team_targets_baselines(seasons)
    result: Dict[str, Dict[int, Dict[str, Any]]] = {}

    def _put(name: str, season_year: int, targets: int, team: str) -> None:
        denom = baselines.get((season_year, team), 0)
        result.setdefault(name, {})[season_year] = {
            "targets": targets,
            "target_share": round(targets / denom, 4) if denom else 0.0,
            "team": team,
        }

    def _seed_stored(season_year: int) -> None:
        """Return stored totals unchanged (zero-new-weeks no-op path)."""
        for name, yrs in stored.items():
            entry = yrs.get(season_year)
            if entry:
                _put(name, season_year, entry["targets"], entry.get("team") or "")

    # ── Roster maps per (season, team), fetched once per run ──
    roster_cache: Dict[Tuple[int, str], Optional[Tuple[Dict, Dict, Set[str]]]] = {}

    def _roster(season_year: int, team: str):
        key = (season_year, team)
        if key not in roster_cache:
            try:
                rows = _cfbd_get("/roster", {"year": season_year, "team": team},
                                 retries=2)
            except Exception as exc:
                print(f"[targets-plays] roster {team} {season_year} failed - "
                      f"{type(exc).__name__}: {exc}")
                rows = None
            roster_cache[key] = _build_roster_maps(rows) if rows is not None else None
        return roster_cache[key]

    # ── Box-score payloads per (season, week), shared across teams ──
    box_cache: Dict[Tuple[int, int], Optional[list]] = {}

    def _box_payload(season_year: int, week: int):
        key = (season_year, week)
        if key not in box_cache:
            try:
                box_cache[key] = _cfbd_get(
                    "/games/players",
                    {"year": season_year, "week": week, "seasonType": "both"},
                    retries=2)
            except Exception:
                box_cache[key] = None
        return box_cache[key]

    def _validate_game(season_year, week_hint, game_id, team, roster_names, derived):
        """
        (ok, week): `derived` must survive the box-score truth checks.
        The game is located in the hint week's box payload when a hint is
        given, else by scanning candidate weeks (payloads are cached).

        Before the REC comparison, scoring-summary candidates fill any
        per-player gap between derived and box receptions - some
        completions exist only as a summary line - capped at the box total
        so a summary line duplicating a detailed play adds nothing.  The
        comparison itself runs on canonical name keys, so roster/box
        spelling variants ("Maxence LeBlanc" vs "Max LeBlanc") do not
        false-flag; the fill mutates `derived` so callers store the
        filled receptions/targets.
        """
        if derived is None:
            return (False, week_hint)
        if derived["attempts"] > 0:
            coverage = derived["named"] / derived["attempts"]
            if coverage < _PLAY_TARGET_MIN_COVERAGE:
                return (False, week_hint)
        if week_hint:
            weeks_to_try = [week_hint]
        else:
            weeks_to_try = _candidate_weeks(season_year, today)
        for wk in weeks_to_try:
            payload = _box_payload(season_year, wk)
            if not payload:
                continue
            rec = _box_receptions(payload, game_id, team, roster_names)
            if rec is None:
                continue  # game not in this week's payload; keep looking
            if derived["attempts"] == 0:
                # Provably pass-less game: no targets exist to get wrong.
                return (len(rec) == 0, wk)
            derived_c = _canon_sums(derived["receptions"])
            box_c = _canon_sums(rec)
            summary_c = _canon_sums(derived.get("summary") or {})
            for key, box_n in box_c.items():
                gap = box_n - derived_c.get(key, 0)
                if gap <= 0:
                    continue
                fill = min(gap, summary_c.get(key, 0))
                if fill <= 0:
                    continue
                # Attribute the fill to one full name under this key:
                # prefer a name the detailed parse already credited, else
                # the summary line's own resolved name.
                target_name = None
                for n in derived["targets"]:
                    if _canonical_name_key(n) == key:
                        target_name = n
                        break
                if target_name is None:
                    for n in (derived.get("summary") or {}):
                        if _canonical_name_key(n) == key:
                            target_name = n
                            break
                if target_name is None:
                    continue
                derived["receptions"][target_name] = (
                    derived["receptions"].get(target_name, 0) + fill)
                derived["targets"][target_name] = (
                    derived["targets"].get(target_name, 0) + fill)
                derived_c[key] = derived_c.get(key, 0) + fill
            keys = set(derived_c) | set(box_c)
            ok = all(derived_c.get(k, 0) == box_c.get(k, 0) for k in keys)
            return (ok, wk)
        return (False, week_hint)

    def _fetch_plays(season_year: int, team: str, week: Optional[int] = None):
        """One /plays call; None signals an API failure (retry later)."""
        params: Dict[str, Any] = {"year": season_year, "team": team}
        if week is not None:
            params["week"] = week
        try:
            return _cfbd_get("/plays", params, retries=2)
        except Exception as exc:
            print(f"[targets-plays] {team} {season_year} failed - "
                  f"{type(exc).__name__}: {exc}")
            return None

    for season_year in seasons:
        teams = sorted(set(teams_by_year[season_year]))
        if not teams:
            continue
        candidates = _candidate_weeks(season_year, today)
        season_tracked = tracked.get(season_year, {})
        has_stored = any(season_year in yrs for yrs in stored.values())
        print(f"[targets-plays] Fetching {season_year} for {len(teams)} teams")
        additions: Dict[str, Tuple[int, str]] = {}   # name → (added, team)
        team_additions: Dict[str, Dict[str, int]] = {}  # team → name → added

        def _add_targets(tgts: Dict[str, int], team: str) -> None:
            for n, c in tgts.items():
                prev_add, _ = additions.get(n, (0, team))
                additions[n] = (prev_add + c, team)
                per_team = team_additions.setdefault(team, {})
                per_team[n] = per_team.get(n, 0) + c

        def _emit_team_season(team: str) -> None:
            """Persist-as-you-go hook: fire once a team-season is complete."""
            if on_team_season is None:
                return
            per_team = team_additions.get(team) or {}
            if not per_team:
                return
            totals: Dict[str, int] = {}
            for n, add in per_team.items():
                base = ((stored.get(n, {}).get(season_year) or {}).get("targets", 0)
                        if adopt_stored else 0)
                totals[n] = base + add
            try:
                on_team_season(season_year, team, totals)
            except Exception as exc:
                print(f"[targets-plays] WARNING: on_team_season callback failed "
                      f"for {team} {season_year}: {exc}")

        # Legacy bootstrap (mirrors GP / old targets): stored totals exist
        # but no week tracking at all - adopt them as complete for every
        # candidate week instead of re-parsing a whole season.
        if adopt_stored and not season_tracked and has_stored:
            for team in teams:
                _mark_weeks_fetched("targets", season_year, team, candidates)
            _seed_stored(season_year)
            continue

        if season_year < today.year:
            # ── Finished season: ONE season-level /plays call per team ──
            for team in teams:
                done = set(season_tracked.get(team, set())) if adopt_stored else set()
                if adopt_stored and candidates and set(candidates) <= done:
                    continue  # every week already folded into stored totals
                maps = _roster(season_year, team)
                if maps is None:
                    continue
                by_jersey, by_name, roster_names = maps
                games = _group_plays_by_game(_fetch_plays(season_year, team))
                if not games:
                    # Season-level yielded nothing (endpoint may demand a
                    # week): fall back to per-week calls for unprocessed
                    # weeks only - same validation, just more calls.
                    for wk in candidates:
                        if wk in done:
                            continue
                        wk_rows = _fetch_plays(season_year, team, week=wk)
                        if wk_rows is None:
                            continue
                        games.extend(
                            (gid, gp, w or wk)
                            for gid, gp, w in _group_plays_by_game(wk_rows))
                        time.sleep(0.15)
                team_total_new = 0
                ok_weeks: Set[int] = set()
                for gid, gplays, wk_hint in games:
                    derived = _derive_game_targets(gplays, team, by_jersey,
                                                   by_name, roster_names)
                    ok, wk = _validate_game(season_year, wk_hint, gid, team,
                                            roster_names, derived)
                    if not ok or wk is None:
                        continue  # discarded whole; week stays unmarked
                    if adopt_stored and wk in done:
                        continue  # already folded in; never double count
                    _add_targets(derived["targets"], team)
                    team_total_new += sum(derived["targets"].values())
                    ok_weeks.add(wk)
                if ok_weeks:
                    _mark_weeks_fetched("targets", season_year, team, sorted(ok_weeks))
                if team_total_new:
                    base = baselines.get((season_year, team), 0) if adopt_stored else 0
                    total_all = base + team_total_new
                    baselines[(season_year, team)] = total_all
                    _record_team_targets(season_year, team, total_all)
                _emit_team_season(team)
        else:
            # ── Current season: per-week incremental per team ──
            est = _estimated_current_week(today)
            for team in teams:
                done = set(season_tracked.get(team, set())) if adopt_stored else set()
                weeks_to_fetch = [w for w in candidates if w not in done]
                if not weeks_to_fetch:
                    continue
                maps = _roster(season_year, team)
                if maps is None:
                    continue
                by_jersey, by_name, roster_names = maps
                team_total_new = 0
                fetched_ok: List[int] = []
                for week in weeks_to_fetch:
                    rows = _fetch_plays(season_year, team, week=week)
                    if rows is None:
                        continue  # failure: leave unmarked, retried next run
                    if not rows:
                        # Empty team-week (bye / not played): record it only
                        # when strictly before the current-week boundary.
                        if week < est:
                            fetched_ok.append(week)
                        continue
                    games = _group_plays_by_game(rows)
                    if not games:
                        continue  # plays without game ids cannot validate
                    week_targets: List[Dict[str, int]] = []
                    week_ok = True
                    for gid, gplays, _wk in games:
                        derived = _derive_game_targets(gplays, team,
                                                       by_jersey, by_name,
                                                       roster_names)
                        ok, _found_wk = _validate_game(
                            season_year, week, gid, team, roster_names, derived)
                        if not ok:
                            week_ok = False
                            break
                        week_targets.append(derived["targets"])
                    if not week_ok:
                        continue  # whole week unmarked; retried next run
                    for tgts in week_targets:
                        _add_targets(tgts, team)
                        team_total_new += sum(tgts.values())
                    fetched_ok.append(week)
                    time.sleep(0.15)   # light rate-limiting between calls
                if fetched_ok:
                    _mark_weeks_fetched("targets", season_year, team, fetched_ok)
                if team_total_new:
                    base = baselines.get((season_year, team), 0) if adopt_stored else 0
                    total_all = base + team_total_new
                    baselines[(season_year, team)] = total_all
                    _record_team_targets(season_year, team, total_all)
                _emit_team_season(team)

        # Merge: stored totals for everyone, additions on top.  Nothing
        # validated → totals identical to what's stored (pure no-op).
        if adopt_stored:
            _seed_stored(season_year)
        for name, (add, team) in additions.items():
            base = ((stored.get(name, {}).get(season_year) or {}).get("targets", 0)
                    if adopt_stored else 0)
            _put(name, season_year, base + add, team)

    total_athletes = len(result)
    print(f"[targets-plays] collected targets for {total_athletes} athletes "
          f"across {len(teams_by_year)} seasons")
    return result


def backfill_cfbd_targets_from_plays(draft_year: int) -> Dict[str, Any]:
    """
    One-time backfill: fill `targets` in rookie_prospect_source_data for
    EVERY stored season of the class, parsed from CFBD play-by-play text.

    Why this exists: fetch_cfbd_college_stats builds its teams_by_year from
    the seasons it (re)fetched, and _filter_years_needing_fetch drops past
    seasons once their source rows exist - so a normal pipeline run only
    ever derives play-text targets for the current season.  This entry
    point instead reads the team set from the stored source rows
    themselves and runs fetch_cfbd_targets_from_plays over all of them;
    its week tracking, per-game box-score validation, team-totals
    bookkeeping and additive composition apply unchanged (with stored
    targets NULL, the composed totals are exactly the validated sums).

    Persisted with the games-played backfill's match pattern: player_id =
    ROOKIE_<year>_<name_lower with spaces as underscores>.  A name in the
    targets map whose slug has no stored row updates nothing (counted via
    rowcount).  Only the `targets` column is ever written.

    Returns {"updated": total_rows, "teams": team_seasons,
             "by_season": {season: rows}}.
    """
    from dashboard_services.db import get_conn

    prefix = _player_id_prefix(draft_year)
    summary: Dict[str, Any] = {"updated": 0, "teams": 0, "by_season": {}}

    try:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT DISTINCT season, team
                    FROM rookie_prospect_source_data
                    WHERE player_id LIKE %s AND team IS NOT NULL
                    """,
                    (prefix + "%",),
                )
                rows = cur.fetchall()
    except Exception as exc:
        print(f"[targets-plays] backfill: could not load stored teams - "
              f"{type(exc).__name__}: {exc}")
        return summary

    teams_by_year: Dict[int, set] = {}
    for r in rows:
        season = r["season"] if isinstance(r, dict) else r[0]
        team = r["team"] if isinstance(r, dict) else r[1]
        if season is None or not team:
            continue
        teams_by_year.setdefault(int(season), set()).add(team)
    if not teams_by_year:
        print(f"[targets-plays] backfill: no stored teams for class {draft_year}")
        return summary
    teams_by_year = {yr: sorted(ts) for yr, ts in sorted(teams_by_year.items())}
    summary["teams"] = sum(len(ts) for ts in teams_by_year.values())

    targets_map = fetch_cfbd_targets_from_plays(draft_year, teams_by_year)
    if not targets_map:
        print(f"[targets-plays] backfill: no validated targets derived "
              f"for class {draft_year}")
        return summary

    # Flatten to per-season (targets, player_id, season) update rows.
    updates: Dict[int, List[Tuple[int, str, int]]] = {}
    for name_lower, by_year in targets_map.items():
        if not name_lower or not by_year:
            continue
        player_id = prefix + str(name_lower).strip().replace(" ", "_")
        for season, entry in by_year.items():
            if not entry:
                continue
            targets = entry.get("targets")
            if targets is None:
                continue
            updates.setdefault(int(season), []).append(
                (int(targets), player_id, int(season)))

    try:
        with get_conn() as conn:
            with conn.cursor() as cur:
                for season in sorted(updates):
                    season_updated = 0
                    for targets, player_id, seas in updates[season]:
                        cur.execute(
                            """
                            UPDATE rookie_prospect_source_data
                            SET targets = %s
                            WHERE player_id = %s AND season = %s
                            """,
                            (targets, player_id, seas),
                        )
                        season_updated += cur.rowcount or 0
                    conn.commit()
                    summary["by_season"][season] = season_updated
                    summary["updated"] += season_updated
    except Exception as exc:
        print(f"[targets-plays] backfill: update failed - "
              f"{type(exc).__name__}: {exc}")

    print(f"[targets-plays] backfill: updated {summary['updated']} "
          f"player-season rows across {summary['teams']} team-seasons "
          f"for class {draft_year}: {summary['by_season']}")
    return summary


def run_daily_prospect_targets(draft_year: int) -> Dict[str, Any]:
    """
    Daily-job targets fill, scoped to the LATEST stored season only.

    Why latest only: the grading model reads targets only from the latest
    season (prospect_model calc_utilization_score targets/game, and the TE
    catch rate in calc_efficiency_score, both via _latest_season), so that
    season is where targets move grades.  Older seasons are a separate,
    later catch-up, not a daily cost.  A full 3-season sweep measured
    ~1,800 CFBD calls at the 4s throttle (9-10+ hours); one season is a
    fraction of that, and after the first fill the existing week marks
    make each daily run fetch only the new week.

    Resumable by construction: fetch_cfbd_targets_from_plays fires the
    on_team_season callback per completed team-season and this writes
    that team-season's player targets immediately (with the
    targets >= receptions guard), so a kill loses at most the in-flight
    team-season.  Team-seasons stranded by the Oct 2026 wiped run (marks
    and team totals persisted, zero player targets written) are detected
    and cleared for exactly that team-season before the fetch, never a
    blanket delete.

    Returns a summary dict; raises nothing on missing team data (the
    fetcher returns {}), so the cron step can record health from it.
    """
    from dashboard_services.db import get_conn

    prefix = _player_id_prefix(draft_year)
    summary: Dict[str, Any] = {
        "season": None, "team_seasons": 0, "est_plays_calls": 0,
        "healed": [], "updated": 0, "skipped_guard": 0, "athletes": 0,
    }

    try:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT DISTINCT season, team
                    FROM rookie_prospect_source_data
                    WHERE player_id LIKE %s AND team IS NOT NULL
                    """,
                    (prefix + "%",),
                )
                rows = cur.fetchall()
    except Exception as exc:
        print(f"[targets-daily] could not load stored teams - "
              f"{type(exc).__name__}: {exc}")
        return summary

    teams_by_season: Dict[int, set] = {}
    for r in rows:
        season = r["season"] if isinstance(r, dict) else r[0]
        team = r["team"] if isinstance(r, dict) else r[1]
        if season is None or not team:
            continue
        teams_by_season.setdefault(int(season), set()).add(team)
    if not teams_by_season:
        print(f"[targets-daily] no stored teams for class {draft_year}")
        return summary

    season = max(teams_by_season)
    teams = sorted(teams_by_season[season])
    summary["season"] = season
    summary["team_seasons"] = len(teams)

    # Self-heal stranded team-seasons before trusting any week marks.
    stranded = detect_stranded_target_team_seasons(draft_year, season, teams)
    for team in stranded:
        print(f"[targets-daily] stranded team-season detected, clearing "
              f"tracking for refetch: {team} {season}")
        clear_target_team_season_tracking(season, team)
    summary["healed"] = stranded

    # Call estimate: unmarked candidate weeks per team, plus one roster
    # call per team.  Logged so the daily cost is visible in cron output.
    candidates = _candidate_weeks(season, _today())
    tracked = _get_fetched_weeks("targets", [season]).get(season, {})
    unmarked = sum(
        len([w for w in candidates if w not in tracked.get(team, set())])
        for team in teams
    )
    est_calls = unmarked + len(teams)
    summary["est_plays_calls"] = est_calls
    print(f"[targets-daily] class {draft_year} latest season {season}: "
          f"{len(teams)} team-seasons, ~{est_calls} CFBD calls estimated "
          f"({unmarked} unmarked team-weeks + {len(teams)} rosters)")

    receptions = _get_existing_receptions(draft_year, [season])
    counters = {"updated": 0, "skipped_guard": 0}

    def _persist_team_season(season_year, team, totals_by_name):
        updated, skipped = _write_player_targets(
            draft_year, int(season_year), totals_by_name, receptions)
        counters["updated"] += updated
        counters["skipped_guard"] += skipped
        print(f"[targets-daily] persisted {team} {season_year}: "
              f"{updated} rows updated, {skipped} guard-skipped")

    targets_map = fetch_cfbd_targets_from_plays(
        draft_year, {season: teams}, on_team_season=_persist_team_season)
    summary["updated"] = counters["updated"]
    summary["skipped_guard"] = counters["skipped_guard"]
    summary["athletes"] = len(targets_map or {})
    print(f"[targets-daily] done: {summary}")
    return summary


def backfill_cfbd_targets(draft_year: int) -> Dict[str, int]:
    """
    Standalone targets backfill: fills the `targets` column in
    rookie_prospect_source_data for ALL seasons of the class,
    without re-running the full CFBD stats fetch. Returns {"updated": n, "teams": m}.
    """
    from dashboard_services.db import get_conn

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT DISTINCT rp.player_id, rp.name, rsd.team, rsd.season
                FROM rookie_prospects rp
                JOIN rookie_prospect_source_data rsd ON rsd.player_id = rp.player_id
                WHERE rp.draft_class_year = %s AND rsd.team IS NOT NULL
                AND (rsd.targets IS NULL OR rsd.targets = 0)
                """,
                (draft_year,),
            )
            rows = cur.fetchall()
    if not rows:
        print(f"[targets] no prospects missing targets for {draft_year}")
        return {"updated": 0, "teams": 0}

    # Group teams by season
    teams_by_year: Dict[int, set] = {}
    for r in rows:
        season = r["season"] if isinstance(r, dict) else r[3]
        team = r["team"] if isinstance(r, dict) else r[2]
        teams_by_year.setdefault(season, set()).add(team)
    teams_by_year = {yr: sorted(teams) for yr, teams in teams_by_year.items()}

    targets_map = fetch_cfbd_targets(draft_year, teams_by_year, adopt_stored=False)
    if not targets_map:
        total_teams = sum(len(t) for t in teams_by_year.values())
        return {"updated": 0, "teams": total_teams}

    updated = 0
    with get_conn() as conn:
        with conn.cursor() as cur:
            for r in rows:
                pid = r["player_id"] if isinstance(r, dict) else r[0]
                name = r["name"] if isinstance(r, dict) else r[1]
                season = r["season"] if isinstance(r, dict) else r[3]
                t_by_year = targets_map.get(name.lower().strip())
                if not t_by_year:
                    continue
                t = t_by_year.get(season)
                if not t:
                    continue
                cur.execute(
                    """
                    UPDATE rookie_prospect_source_data
                    SET targets = %s
                    WHERE player_id = %s AND season = %s
                    """,
                    (t["targets"], pid, season),
                )
                updated += cur.rowcount
        conn.commit()
    total_teams = sum(len(t) for t in teams_by_year.values())
    print(f"[targets] backfilled targets for {updated} player-seasons ({total_teams} teams)")
    return {"updated": updated, "teams": total_teams}


_ROSTER_YEAR_TO_EXPERIENCE = {1: "FR", 2: "SO", 3: "JR", 4: "SR", 5: "SR"}


def _age_from_roster_row(row: Dict[str, Any], draft_year: int) -> Tuple[Optional[float], bool]:
    """
    (age at the draft, is_estimate) from a CFBD /roster row; age is None
    when the row carries nothing usable.

    The roster payload as used in this codebase (see the fixtures in
    tests/test_prospect_data_sources.py) carries height, weight,
    position and academic `year` - it does NOT carry a date of birth.
    If a DOB field is ever present it wins (exact age at the April 25
    draft reference date, the ESPN convention); otherwise the existing
    experience-estimate channel is driven by the roster's academic year
    and the result is flagged as an estimate, so a stored exact age can
    still override it downstream.
    """
    dob_raw = row.get("dateOfBirth") or row.get("date_of_birth") or row.get("dob")
    if dob_raw:
        try:
            from .espn_scraper import parse_dob_and_calculate_age
            _iso, age = parse_dob_and_calculate_age(
                str(dob_raw), ref_date=datetime.date(draft_year, 4, 25)
            )
            if age is not None:
                return age, False
        except Exception:
            pass
    experience = _ROSTER_YEAR_TO_EXPERIENCE.get(_safe_int(row.get("year")))
    if experience:
        return _estimate_age(experience, draft_year), True
    return None, False


def _enrich_bio_from_cfbd_roster(prospects: List[Dict[str, Any]], draft_year: int) -> List[Dict[str, Any]]:
    """
    Free replacement for Sportradar bio measurements: fill missing
    height_inches/weight_lbs (and age, via _age_from_roster_row) from the
    CFBD roster endpoint (no extra key).  Matches by canonical player
    name; only fills fields that are currently empty - a present age is
    never overwritten.
    """
    if not CFBD_KEY or not prospects:
        return prospects
    season_year = draft_year - 1  # prospects' final college season
    try:
        data = _cfbd_get("/roster", {"year": season_year}, retries=2) or []
    except Exception as exc:
        print(f"[cfbd] roster bio lookup failed - {type(exc).__name__}: {exc}")
        return prospects
    by_name: Dict[str, Dict[str, Any]] = {}
    for row in data:
        name = normalize_name(f"{row.get('firstName') or ''} {row.get('lastName') or ''}")
        if name and name not in by_name:
            by_name[name] = row
    filled = 0
    ages_filled = 0
    for p in prospects:
        row = by_name.get(normalize_name(p.get("name") or ""))
        if not row:
            continue
        if not (p.get("height_inches") and p.get("weight_lbs")):
            h = _safe_int(row.get("height"))
            w = _safe_int(row.get("weight"))
            if h and not p.get("height_inches"):
                p["height_inches"] = h
                filled += 1
            if w and not p.get("weight_lbs"):
                p["weight_lbs"] = w
        if not p.get("age"):
            age, is_estimate = _age_from_roster_row(row, draft_year)
            if age is not None:
                p["age"] = age
                ages_filled += 1
                if is_estimate:
                    p["_age_estimated"] = True
                    experience = _ROSTER_YEAR_TO_EXPERIENCE.get(_safe_int(row.get("year")))
                    if experience and not p.get("_experience"):
                        p["_experience"] = experience
    print(f"[cfbd] roster bio filled measurements for {filled} prospects ({season_year} rosters)")
    if ages_filled:
        print(f"[cfbd] roster bio filled ages for {ages_filled} prospects ({season_year} rosters)")
    return prospects


def discover_breakout_prospects(
    draft_year: int,
    existing_names: Set[str],
    max_new: int = 25,
) -> List[Dict[str, Any]]:
    """
    Free auto-discovery: scan CFB stat leaders for draft-eligible players
    missing from the prospect universe. Catches breakout players the curated
    seed didn't include. Eligibility via roster class year (JR/SR/GR only).
    """
    if not CFBD_KEY:
        return []
    season_year = draft_year - 1
    existing = {n.lower().strip() for n in existing_names if n}

    try:
        roster = _cfbd_get("/roster", {"year": season_year}, retries=2) or []
    except Exception as exc:
        print(f"[discovery] roster fetch failed - {type(exc).__name__}: {exc}")
        return []
    eligible: Set[str] = set()
    for row in roster:
        name = f"{row.get('firstName') or ''} {row.get('lastName') or ''}".strip().lower()
        if name and (_safe_int(row.get("year")) or 0) >= 3:  # 3=JR, 4=SR, 5=GR
            eligible.add(name)

    discovered: List[Dict[str, Any]] = []
    seen_new: Set[str] = set()
    for category, positions in [
        ("passing", {"QB"}),
        ("rushing", {"RB"}),
        ("receiving", {"WR", "TE"}),
    ]:
        if len(discovered) >= max_new:
            break
        try:
            data = _cfbd_get(
                "/stats/player/season",
                {"year": season_year, "seasonType": "both", "category": category},
                retries=2,
            ) or []
        except Exception as exc:
            print(f"[discovery] {category} leaders failed - {type(exc).__name__}: {exc}")
            continue
        leaders = [
            r for r in data
            if r.get("statType") == "YDS" and (r.get("position") or "").upper() in positions
        ]
        leaders.sort(key=lambda r: float(r.get("stat") or 0), reverse=True)
        for r in leaders[:30]:
            if len(discovered) >= max_new:
                break
            name = (r.get("player") or "").strip()
            key = name.lower()
            if not name or key in existing or key in seen_new or key not in eligible:
                continue
            seen_new.add(key)
            discovered.append({
                "player_id": f"ROOKIE_{draft_year}_{_slug(name)}",
                "name": name,
                "position": (r.get("position") or "").upper(),
                "school": r.get("team"),
                "age": None,
                "height_inches": None,
                "weight_lbs": None,
                "state": None,
                "draft_class_year": draft_year,
                "early_declare": False,
                "seasons": [],
                "athleticism": {},
                "source": "cfbd_discovery",
            })
    print(f"[discovery] found {len(discovered)} new draft-eligible stat leaders for {draft_year}")
    return discovered


def get_seed_prospects(draft_year: int) -> List[Dict[str, Any]]:
    """Return the curated seed dataset for a given draft year."""
    import json
    from pathlib import Path
    
    # Try to load from the most recent JSON file that has data
    project_root = Path(__file__).parent.parent.parent
    data_dir = project_root / "data"
    
    profile_files = [
        data_dir / "rookie_profiles_latest.json",
    ]
    
    for file_path in profile_files:
        if file_path.exists():
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                # Check if this file has actual data (not empty)
                if data.get("count", 0) > 0 and data.get("profiles"):
                    # Filter by draft class year if specified
                    profiles = data["profiles"]
                    if draft_year and data.get("draft_class_year"):
                        if data["draft_class_year"] != draft_year:
                            continue
                    
                    # Add draft_class_year to each prospect record
                    file_draft_year = data.get("draft_class_year", draft_year)
                    for profile in profiles:
                        profile["draft_class_year"] = file_draft_year
                    
                    print(f"[ingestion] Loaded {len(profiles)} seed prospects from {file_path.name}")
                    return profiles
                    
            except Exception as exc:
                print(f"[ingestion] ERROR loading seed data from {file_path.name} - {type(exc).__name__}: {exc}")
                continue
    
    print(f"[ingestion] No seed data found for draft year {draft_year}")

    # Bundled fallback: curated consensus prospect list shipped with the repo
    # (used when Sportradar is unavailable and no prior run exists).
    bundled = Path(__file__).parent / "seed_data" / f"prospects_{draft_year}.json"
    if bundled.exists():
        try:
            with open(bundled, 'r') as f:
                data = json.load(f)
            profiles = data.get("profiles") or []
            if profiles:
                for profile in profiles:
                    profile["draft_class_year"] = draft_year
                print(f"[ingestion] Loaded {len(profiles)} bundled seed prospects from {bundled.name}")
                return profiles
        except Exception as exc:
            print(f"[ingestion] ERROR loading bundled seed {bundled.name} - {type(exc).__name__}: {exc}")
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
                    "team_total_yards", "team_total_tds", "team", "conference",
                    "team_pass_yards", "team_pass_tds", "sagarin_team_rating",
                    "depth_rank", "position_group_size"):
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
        print(f"[ingestion] ERROR loading seed prospects - {type(exc).__name__}: {exc}")
        seed = []
    print(f"[ingestion] Loaded {len(seed)} seed prospects")

    # Auto-discovery: catch draft-eligible breakouts the seed missed
    try:
        found = discover_breakout_prospects(draft_year, {p.get("name", "") for p in seed})
        if found:
            print(f"[ingestion] Discovery added {len(found)} prospects")
            seed = seed + found
    except Exception as exc:
        print(f"[ingestion] Discovery failed - {type(exc).__name__}: {exc}")

    # ── No Sportradar key - use seed only ────────────────────────────────────
    if not SPORTRADAR_KEY:
        print(f"[ingestion] FAILED: No SPORTRADAR_API_KEY set - returning seed data only ({len(seed)} prospects)")
        return _enrich_bio_from_cfbd_roster([normalize_prospect(p) for p in seed], draft_year)

    # ── Fetch from all live sources ───────────────────────────────────────────
    if __import__("os").getenv("DISABLE_SPORTRADAR", "").lower() in ("1", "true", "yes"):
        print(f"[ingestion] DISABLE_SPORTRADAR set - using seed data ({len(seed)} prospects)")
        return _enrich_bio_from_cfbd_roster([normalize_prospect(p) for p in seed], draft_year)
    print(f"[ingestion] Fetching Sportradar prospects for {draft_year}")
    try:
        sr_prospects = fetch_sportradar_prospects(draft_year)
    except Exception as exc:
        print(f"[ingestion] ERROR fetching Sportradar prospects - {type(exc).__name__}: {exc}")
        print(f"[ingestion] Falling back to seed data ({len(seed)} prospects)")
        return _enrich_bio_from_cfbd_roster([normalize_prospect(p) for p in seed], draft_year)
    
    if not sr_prospects:
        print(f"[ingestion] FAILED: Sportradar returned no data for {draft_year} - using seed ({len(seed)} prospects)")
        return _enrich_bio_from_cfbd_roster([normalize_prospect(p) for p in seed], draft_year)
    
    print(f"[ingestion] Sportradar returned {len(sr_prospects)} prospects")

    # ESPN age lookup - use the robust scraper (search + athlete API + HTML fallback)
    print(f"[ingestion] Starting ESPN age lookup for {len(sr_prospects)} prospects")
    try:
        from .espn_scraper import fetch_espn_ages_robust
        espn_ages = fetch_espn_ages_robust(
            [p["name"] for p in sr_prospects],
            draft_year,
            prospects_meta=sr_prospects,   # provides school + position for disambiguation
        )
    except Exception as exc:
        print(f"[ingestion] ERROR in ESPN age lookup - {type(exc).__name__}: {exc}")
        espn_ages = {}
    print(f"[ingestion] ESPN ages resolved for {len(espn_ages)} prospects")

    print(f"[ingestion] Resolving ESPN headshots for {len(sr_prospects)} prospects")
    try:
        from .espn_scraper import fetch_espn_headshots
        espn_headshots = fetch_espn_headshots(
            [p["name"] for p in sr_prospects],
            draft_year,
            prospects_meta=sr_prospects,   # provides school + position for disambiguation
        )
    except Exception as exc:
        print(f"[ingestion] ERROR in ESPN headshot lookup - {type(exc).__name__}: {exc}")
        espn_headshots = {}
    print(f"[ingestion] ESPN headshots resolved for {len(espn_headshots)} prospects")

    print(f"[ingestion] Resolving ESPN experience for {len(sr_prospects)} prospects")
    try:
        from .espn_scraper import fetch_espn_experience_via_roster
        espn_experience = fetch_espn_experience_via_roster(sr_prospects)
    except Exception as exc:
        print(f"[ingestion] ERROR in ESPN experience lookup - {type(exc).__name__}: {exc}")
        espn_experience = {}
    print(f"[ingestion] ESPN experience resolved for {len(espn_experience)} prospects")

    print(f"[ingestion] Fetching NFLverse combine data for {draft_year}")
    try:
        combine_data = fetch_nflverse_combine(draft_year)
    except Exception as exc:
        print(f"[ingestion] ERROR fetching NFLverse combine data - {type(exc).__name__}: {exc}")
        combine_data = {}
    print(f"[ingestion] NFLverse combine data loaded for {len(combine_data)} prospects")
    
    print(f"[ingestion] Fetching CFBD college stats for {draft_year}")
    try:
        cfbd_stats = fetch_cfbd_college_stats(draft_year)
    except Exception as exc:
        print(f"[ingestion] ERROR fetching CFBD stats - {type(exc).__name__}: {exc}")
        cfbd_stats = {}
    print(f"[ingestion] CFBD stats loaded for {len(cfbd_stats)} prospects")

    try:
        seed_by_name = {p["name"].lower(): p for p in seed}
    except Exception as exc:
        print(f"[ingestion] ERROR building seed lookup - {type(exc).__name__}: {exc}")
        seed_by_name = {}
    
    print(f"[ingestion] Starting enrichment of {len(sr_prospects)} Sportradar prospects")

    enriched: List[Dict] = []
    enrichment_errors = 0
    
    for i, sr in enumerate(sr_prospects):
        try:
            name_key = sr["name"].lower()
            seed_p   = seed_by_name.get(name_key)
            nflv     = combine_data.get(name_key, {})
            cfbd_seasons = cfbd_stats.get(canonical_prospect_key(sr["name"]))

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

            # ── Age: ESPN DOB (exact) → seed → ESPN roster experience → Sportradar experience ──
            try:
                p["age"] = (
                    espn_ages.get(name_key) or
                    (seed_p or {}).get("age") or
                    _estimate_age(espn_experience.get(name_key), draft_year) or
                    _estimate_age(sr.get("_experience"), draft_year)
                )
            except Exception as exc:
                print(f"[ingestion] ERROR calculating age for '{sr['name']}' - {type(exc).__name__}: {exc}")
                p["age"] = None

            # ── Headshot: ESPN college headshot via athlete ID ────────────────
            try:
                p["headshot_url"] = (
                    espn_headshots.get(name_key)
                    or (seed_p or {}).get("headshot_url")
                )
            except Exception as exc:
                print(f"[ingestion] ERROR setting headshot for '{sr['name']}' - {type(exc).__name__}: {exc}")
                p["headshot_url"] = None

            # ── Athleticism / combine ─────────────────────────────────────────────
            try:
                seed_ath  = (seed_p or {}).get("athleticism") or {}
                nflv_ath  = nflv.get("athleticism") or {}
                # NFLVerse is authoritative for combine; seed fills any remaining gaps
                p["athleticism"] = {**seed_ath, **nflv_ath}
            except Exception as exc:
                print(f"[ingestion] ERROR merging athleticism for '{sr['name']}' - {type(exc).__name__}: {exc}")
                p["athleticism"] = {}

            # ── College stats ─────────────────────────────────────────────────────
            try:
                if cfbd_seasons:
                    p["seasons"] = cfbd_seasons
                elif seed_p and seed_p.get("seasons"):
                    p["seasons"] = seed_p["seasons"]
                elif sr.get("_conference"):
                    # New player with no stats - inject conference so competition score works
                    p["seasons"] = [{"season": draft_year - 1, "conference": sr["_conference"]}]
                else:
                    p["seasons"] = []
            except Exception as exc:
                print(f"[ingestion] ERROR setting seasons for '{sr['name']}' - {type(exc).__name__}: {exc}")
                p["seasons"] = []

            enriched.append(p)
        except Exception as exc:
            print(f"[ingestion] ERROR enriching prospect '{sr.get('name', 'UNKNOWN')}' (index {i}) - {type(exc).__name__}: {exc}")
            enrichment_errors += 1
        
        if len(enriched) % 50 == 0:
            print(f"[ingestion] Enriched {len(enriched)}/{len(sr_prospects)} prospects ({enrichment_errors} errors)")

    # Seed players not returned by Sportradar
    try:
        sr_names  = {p["name"].lower() for p in sr_prospects}
        seed_only = [p for p in seed if p["name"].lower() not in sr_names]
        if seed_only:
            print(f"[ingestion] {len(seed_only)} seed prospects not in Sportradar - appending")
    except Exception as exc:
        print(f"[ingestion] ERROR processing seed-only prospects - {type(exc).__name__}: {exc}")
        seed_only = []

    final = enriched + seed_only
    print(
        f"[ingestion] COMPLETE: {len(final)} total prospects for {draft_year}  "
        f"(Sportradar: {len(enriched)} | combine: {sum(1 for p in enriched if p.get('athleticism'))} matched | CFBD stats: {sum(1 for p in enriched if cfbd_stats.get(canonical_prospect_key(p['name'])))} matched | seed-only: {len(seed_only)} | enrichment errors: {enrichment_errors})"
    )
    
    try:
        return [normalize_prospect(p) for p in final]
    except Exception as exc:
        print(f"[ingestion] ERROR normalizing prospects - {type(exc).__name__}: {exc}")
        # Return unnormalized data as fallback
        return final
