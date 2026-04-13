# dashboard_services/team_enrichment.py

import concurrent.futures
import json
import os
from pathlib import Path
from typing import Dict, Optional

import requests
from bs4 import BeautifulSoup

from utils.utils import load_teams_index, write_json, path_teams_index

HTTP_SESSION = requests.Session()

TANK01_API_HOST = "tank01-nfl-live-in-game-real-time-statistics-nfl.p.rapidapi.com"
TANK01_BASE_URL = f"https://{TANK01_API_HOST}"
TANK01_API_KEY = os.getenv("TANK01_API_KEY")
# TEAMS_INDEX_PATH = "cache" / "teams_index.json"
ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

RUSH_ATT_PG_URL = "https://www.teamrankings.com/nfl/stat/rushing-attempts-per-game"
RUSH_YDS_PG_URL = "https://www.teamrankings.com/nfl/stat/rushing-yards-per-game"
PASS_YDS_PG_URL = "https://www.teamrankings.com/nfl/stat/passing-yards-per-game"
OPP_PASS_YDS_PG_URL = "https://www.teamrankings.com/nfl/stat/opponent-passing-yards-per-game"
OPP_RUSH_YDS_PG_URL = "https://www.teamrankings.com/nfl/stat/opponent-rushing-yards-per-game"
POINTS_PG_URL = "https://www.teamrankings.com/nfl/stat/points-per-game"
SACKS_ALLOWED_PG_URL = "https://www.teamrankings.com/nfl/stat/sacks-allowed-per-game"
RED_ZONE_TRIPS_PG_URL = "https://www.teamrankings.com/nfl/stat/red-zone-trips-per-game"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0 Safari/537.36"
    )
}

# Map TeamRankings team text -> your abbreviations
TEAMRANKINGS_TO_ABBR: Dict[str, str] = {
    "Arizona": "ARI",
    "Atlanta": "ATL",
    "Baltimore": "BAL",
    "Buffalo": "BUF",
    "Carolina": "CAR",
    "Chicago": "CHI",
    "Cincinnati": "CIN",
    "Cleveland": "CLE",
    "Dallas": "DAL",
    "Denver": "DEN",
    "Detroit": "DET",
    "Green Bay": "GB",
    "Houston": "HOU",
    "Indianapolis": "IND",
    "Jacksonville": "JAX",
    "Kansas City": "KC",
    "Las Vegas": "LV",
    "LA Chargers": "LAC",
    "LA Rams": "LAR",
    "Miami": "MIA",
    "Minnesota": "MIN",
    "New England": "NE",
    "New Orleans": "NO",
    "NY Giants": "NYG",
    "NY Jets": "NYJ",
    "Philadelphia": "PHI",
    "Pittsburgh": "PIT",
    "San Francisco": "SF",
    "Seattle": "SEA",
    "Tampa Bay": "TB",
    "Tennessee": "TEN",
    "Washington": "WSH",
}


def _save_value_table(players, path: Path):
    with path.open("w", encoding="utf-8") as f:
        json.dump(players, f, ensure_ascii=False, indent=2)


def _load_teams_index_raw() -> Dict[str, dict]:
    """
    If you already have load_teams_index() returning a dict,
    you can just call that instead. This raw loader expects
    teams_index.json under DATA_DIR.
    """

    if not path_teams_index().exists():
        raise FileNotFoundError(f"No teams index at {TEAMS_INDEX_PATH}")
    with path_teams_index().open("r", encoding="utf-8") as f:
        return json.load(f)


def _save_teams_index_raw(teams_index: Dict[str, dict]):
    with Path(path_teams_index()).open("w", encoding="utf-8") as f:
        json.dump(teams_index, f, ensure_ascii=False, indent=2)


def enrich_teams_index_with_offense(season: int):
    """
    Load teams_index.json and add:

      - pass_att_pg
      - targets_pg
      - off_snaps_pg

    for each team present in team_offense.
    """
    teams_index = load_teams_index()
    team_offense = enrich_teams_index_with_team_offense(season)

    for team, ctx in team_offense.items():
        entry = teams_index.get(team) or {}
        entry["games_tracked"] = ctx["team_games"]
        teams_index[team] = entry

    _save_teams_index_raw(teams_index)


def _to_float(value) -> float:
    """Helper: safely convert Tank01 string/None -> float."""
    if value is None:
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def fetch_team_offense_per_game(
        season: int = 2024,
        session: Optional[requests.Session] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Call Tank01 getNFLTeams and return per-game offensive team stats.
    """
    sess = session or HTTP_SESSION

    url = f"{TANK01_BASE_URL}/getNFLTeams"
    params = {
        "sortBy": "standings",
        "rosters": "false",
        "schedules": "false",
        "topPerformers": "true",
        "teamStats": "true",
        "teamStatsSeason": season,
    }
    headers = {
        "x-rapidapi-host": TANK01_API_HOST,
        "x-rapidapi-key": TANK01_API_KEY,
    }

    resp = sess.get(url, headers=headers, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    body = data.get("body") or []
    out: Dict[str, Dict[str, float]] = {}

    for team_obj in body:
        team_abv = team_obj.get("teamAbv")
        if not team_abv:
            continue

        # games played = wins + losses + ties
        wins = _to_float(team_obj.get("wins"))
        losses = _to_float(team_obj.get("loss"))
        ties = _to_float(team_obj.get("tie"))
        games = wins + losses + ties

        if games <= 0:
            continue

        team_stats = team_obj.get("teamStats") or {}
        pass_stats = team_stats.get("Passing") or {}
        rush_stats = team_stats.get("Rushing") or {}

        total_pass_yds = _to_float(pass_stats.get("passYds"))
        total_pass_att = _to_float(pass_stats.get("passAttempts"))
        total_pass_td = _to_float(pass_stats.get("passTD"))

        total_rush_yds = _to_float(rush_stats.get("rushYds"))
        total_rush_att = _to_float(rush_stats.get("carries"))
        total_rush_td = _to_float(rush_stats.get("rushTD"))

        out[team_abv] = {
            "pass_yds_pg": total_pass_yds / games,
            "pass_att_pg": total_pass_att / games,
            "pass_td_pg": total_pass_td / games,
            "rush_yds_pg": total_rush_yds / games,
            "rush_att_pg": total_rush_att / games,
            "rush_td_pg": total_rush_td / games,
            "games": games,
        }

    return out


def enrich_teams_index_with_team_offense(season: int = 2024) -> None:
    teams_index = load_teams_index() or {}
    try:
        per_game = fetch_team_offense_per_game(season, session=HTTP_SESSION)
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 429:
            return
        else:
            print(f"Skipping tank01 API call due to HTTP error: {e}")
            return
    except Exception as e:
        print(f"Skipping tank01 API call due to unexpected error: {e}")
        return

    for team_abv, stats in per_game.items():
        meta = teams_index.setdefault(team_abv, {})
        meta["pass_yds_pg"] = stats["pass_yds_pg"]
        meta["pass_att_pg"] = stats["pass_att_pg"]
        meta["pass_td_pg"] = stats["pass_td_pg"]
        meta["rush_yds_pg"] = stats["rush_yds_pg"]
        meta["rush_att_pg"] = stats["rush_att_pg"]
        meta["rush_td_pg"] = stats["rush_td_pg"]

        # Derive points_pg from TDs: pass + rush TDs * 7 pts (incl PAT) + ~1.2 FGs * 3 pts
        pass_td = stats["pass_td_pg"]
        rush_td = stats["rush_td_pg"]
        meta["points_pg"] = round((pass_td + rush_td) * 6.5 + 1.2 * 3.0, 2)

        # red_zone_trips_pg and sacks_allowed_pg are not in the Tank01 team-stats response.
        # Set to 0 here so callers know to fall back to NFL averages; the enrichment
        # pipeline can fill these from a dedicated source (e.g. TeamRankings) in the future.
        if not meta.get("red_zone_trips_pg"):
            meta["red_zone_trips_pg"] = 0.0
        if not meta.get("sacks_allowed_pg"):
            meta["sacks_allowed_pg"] = 0.0

    write_json(path_teams_index(), teams_index)


def enrich_teams_index_with_scoring_and_pressure() -> Dict[str, dict]:
    """
    Fetch points_pg, red_zone_trips_pg, and sacks_allowed_pg from TeamRankings
    and merge into teams_index.json.

    These three stats are not available from the Tank01 team-offense endpoint
    (which only has passing/rushing yards/attempts/TDs), so we scrape them
    separately from TeamRankings.

    Returns the updated teams_index dict.
    """
    teams_index = load_teams_index() or {}

    urls = {
        "points_pg": POINTS_PG_URL,
        "red_zone_trips_pg": RED_ZONE_TRIPS_PG_URL,
        "sacks_allowed_pg": SACKS_ALLOWED_PG_URL,
    }

    results: Dict[str, Dict[str, float]] = {}
    session = HTTP_SESSION

    def worker(key: str, url: str):
        try:
            data = _fetch_teamrankings_table(url, session=session)
            return key, data, None
        except Exception as e:
            return key, {}, e

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        future_map = {
            executor.submit(worker, key, url): key
            for key, url in urls.items()
        }
        for fut in concurrent.futures.as_completed(future_map):
            key, data, err = fut.result()
            if err:
                print(f"[teamrankings] ERROR fetching {key}: {err}")
                data = {}
            results[key] = data

    for abbr, meta in teams_index.items():
        if abbr in results.get("points_pg", {}):
            meta["points_pg"] = results["points_pg"][abbr]
        if abbr in results.get("red_zone_trips_pg", {}):
            meta["red_zone_trips_pg"] = results["red_zone_trips_pg"][abbr]
        if abbr in results.get("sacks_allowed_pg", {}):
            meta["sacks_allowed_pg"] = results["sacks_allowed_pg"][abbr]

    write_json(path_teams_index(), teams_index)
    return teams_index


def enrich_all_team_info(season: int):
    """
    Convenience: enrich both value table and teams index.
    Fetches Tank01 offense stats + TeamRankings scoring/pressure stats.
    """
    enrich_teams_index_with_team_offense(season)
    try:
        enrich_teams_index_with_scoring_and_pressure()
    except Exception as e:
        print(f"[team_enrichment] Could not enrich scoring/pressure stats: {e}")


def _fetch_teamrankings_table(
        url: str,
        session: Optional[requests.Session] = None,
) -> Dict[str, float]:
    """
    Generic parser for TeamRankings stat pages.

    Returns {abbr: current_season_value} where abbr is e.g. 'ARI', 'ATL', ...
    """
    sess = session or HTTP_SESSION

    resp = sess.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")

    # Find the table whose header row looks like "Rank Team 2025 Last 3 ..."
    target_table = None
    for table in soup.find_all("table"):
        header_cells = table.find("tr")
        if not header_cells:
            continue
        header_text = " ".join(th.get_text(strip=True) for th in header_cells.find_all("th"))
        if "Rank" in header_text and "Team" in header_text:
            target_table = table
            break

    if not target_table:
        raise RuntimeError("Could not find TeamRankings stats table on page")

    tbody = target_table.find("tbody") or target_table
    rows = tbody.find_all("tr")

    out: Dict[str, float] = {}

    for tr in rows:
        tds = tr.find_all("td")
        if len(tds) < 3:
            continue

        # 0: Rank, 1: Team, 2: Current Season, ...
        team_text = tds[1].get_text(strip=True)
        value_text = tds[2].get_text(strip=True)

        try:
            value = float(value_text)
        except ValueError:
            continue

        abbr = TEAMRANKINGS_TO_ABBR.get(team_text)
        if not abbr:
            print(f"[teamrankings] Unmapped team name: {team_text!r}")
            continue

        out[abbr] = value

    return out


def enrich_teams_index_with_rushing(
        teams_index_path: Path,
        out_path: Optional[Path] = None,
) -> Dict[str, dict]:
    """
    Load your teams_index.json, add:
      - rush_att_pg
      - rush_yds_pg
      - pass_yds_pg
      - opp_pass_yds_pg
      - opp_rush_yds_pg
    then write back to disk (or a new file).
    """
    with teams_index_path.open("r", encoding="utf-8") as f:
        teams_index = json.load(f)

    # Map logical keys -> URL
    urls = {
        "rush_att": RUSH_ATT_PG_URL,
        "rush_yds": RUSH_YDS_PG_URL,
        "pass_yds": PASS_YDS_PG_URL,
        "opp_pass_yds": OPP_PASS_YDS_PG_URL,
        "opp_rush_yds": OPP_RUSH_YDS_PG_URL,
    }

    results: Dict[str, Dict[str, float]] = {}
    session = HTTP_SESSION

    def worker(key: str, url: str):
        try:
            data = _fetch_teamrankings_table(url, session=session)
            return key, data, None
        except Exception as e:
            return key, {}, e

    # Fetch all 5 pages in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        future_map = {
            executor.submit(worker, key, url): key
            for key, url in urls.items()
        }
        for fut in concurrent.futures.as_completed(future_map):
            key, data, err = fut.result()
            if err:
                print(f"[teamrankings] ERROR fetching {key}: {err}")
                data = {}
            results[key] = data

    rush_att = results.get("rush_att", {})
    rush_yds = results.get("rush_yds", {})
    pass_yds = results.get("pass_yds", {})
    opp_pass_yds = results.get("opp_pass_yds", {})
    opp_rush_yds = results.get("opp_rush_yds", {})

    for abbr, meta in teams_index.items():
        meta["rush_att_pg"] = rush_att.get(abbr)
        meta["rush_yds_pg"] = rush_yds.get(abbr)
        meta["pass_yds_pg"] = pass_yds.get(abbr)
        meta["opp_pass_yds_pg"] = opp_pass_yds.get(abbr)
        meta["opp_rush_yds_pg"] = opp_rush_yds.get(abbr)

    out_path = out_path or teams_index_path
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(teams_index, f, indent=2)

    return teams_index
