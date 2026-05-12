# dashboard_services/external_values_scraper.py

import csv
from datetime import date, timedelta
from pathlib import Path
from typing import Optional, List, Dict

import requests

from utils.utils import DATA_DIR, path_fantasycalc_values, path_dynastyprocess_values, path_ktc_values

# ---------------------------
# Paths / constants
# ---------------------------

FANTASYCALC_API_URL = "https://api.fantasycalc.com/values/current"
DYNASTYPROCESS_VALUES_URL = (
    "https://raw.githubusercontent.com/dynastyprocess/data/master/files/values.csv"
)

# Separate outputs so API vs scraped rankings don't trample each other
FANTASYCALC_RANKINGS_CSV_PATH = DATA_DIR / f"fantasycalc_rankings_{date.today().isoformat()}.csv"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0 Safari/537.36"
    )
}

# ============================================================
# KEEPTRADECUT (KTC) – unofficial JSON API
# ============================================================

KTC_API_URL = "https://keeptradecut.com/api/rankings"


def fetch_ktc_values(is_dynasty: bool = True) -> Optional[List[dict]]:
    """
    Fetch KTC dynasty (or redraft) rankings via their unofficial JSON endpoint.

    Returns the raw list of player dicts on success, or None if the request
    fails or returns unexpected data.  Always degrades gracefully so the
    caller can continue without KTC data.
    """
    params = {
        "filters": "QB,RB,WR,TE,RDPICKS",
        "page": "0",
        "format": "0" if is_dynasty else "1",
    }
    try:
        resp = requests.get(KTC_API_URL, params=params, headers=HEADERS, timeout=15)
        if not resp.ok:
            print(f"[KTC] HTTP {resp.status_code} from {resp.url}")
            return None
        ct = resp.headers.get("Content-Type", "")
        if "json" not in ct:
            print(f"[KTC] Unexpected Content-Type: {ct} - skipping")
            return None
        data = resp.json()
        if not isinstance(data, list) or not data:
            print("[KTC] Response is not a non-empty list - skipping")
            return None
        return data
    except Exception as e:
        print(f"[KTC] Fetch failed: {e}")
        return None


def write_ktc_to_csv(
        values: List[dict],
        out_csv: Path = None,
) -> None:
    """Flatten KTC API response into a dated CSV."""
    if out_csv is None:
        out_csv = Path(path_ktc_values())

    today = date.today()
    yesterday = today - timedelta(days=1)
    out_csv = Path(out_csv)
    dirname = out_csv.parent
    yesterday_file = dirname / f"ktc_rankings_{yesterday.isoformat()}.csv"
    if yesterday_file.exists():
        try:
            yesterday_file.unlink()
        except Exception:
            pass

    rows = []
    for entry in values:
        one_qb = entry.get("oneQBValues") or {}
        sf = entry.get("superflexValues") or {}
        rows.append({
            "source":        "KTC",
            "ktc_id":        entry.get("playerID") or entry.get("id"),
            "name":          entry.get("playerName") or entry.get("name"),
            "position":      entry.get("position"),
            "team":          entry.get("team"),
            "age":           entry.get("age"),
            "ktc_value_1qb": one_qb.get("value"),
            "ktc_rank_1qb":  one_qb.get("rank"),
            "ktc_trend_1qb": one_qb.get("trend"),
            "ktc_value_sf":  sf.get("value"),
            "ktc_rank_sf":   sf.get("rank"),
        })

    _FIELDNAMES = [
        "source", "ktc_id", "name", "position", "team", "age",
        "ktc_value_1qb", "ktc_rank_1qb", "ktc_trend_1qb",
        "ktc_value_sf", "ktc_rank_sf",
    ]
    print(f"[KTC] Writing {len(rows)} rows to {out_csv}")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def load_ktc_values(
        csv_path: Path = None,
) -> Optional[List[dict]]:
    """Load today's KTC CSV if it exists; otherwise return None."""
    if csv_path is None:
        csv_path = Path(path_ktc_values())
    csv_path = Path(csv_path)
    if not csv_path.exists():
        return None
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def fetch_fantasycalc_api_values(
        *,
        is_dynasty: bool = True,
        num_qbs: int = 1,
        num_teams: Optional[int] = None,
        ppr: float = 1.0,
) -> List[dict]:
    """
    Call FantasyCalc values API and return the parsed JSON list.

    Example endpoint:
      https://api.fantasycalc.com/values/current?isDynasty=true&numQbs=1&numTeams=10&ppr=1

    num_teams:
      - If provided, passed straight through to FantasyCalc.
      - If None or invalid, defaults to 10 (backwards-compatible).
    """
    if not isinstance(num_teams, int) or num_teams <= 0:
        num_teams = 10

    params = {
        "isDynasty": "true" if is_dynasty else "false",
        "numQbs": num_qbs,
        "numTeams": num_teams,
        "ppr": ppr,
    }

    session = requests.Session()
    resp = session.get(FANTASYCALC_API_URL, params=params, headers=HEADERS, timeout=20)
    resp.raise_for_status()
    data = resp.json()
    if not isinstance(data, list):
        raise ValueError("FantasyCalc API did not return a list")
    return data


def write_fantasycalc_api_to_csv(
        values: List[dict],
        out_csv: Path = path_fantasycalc_values(),
) -> None:
    """
    Flatten FantasyCalc API payload into a CSV with one row per player.

    Columns:
      source, fc_id, sleeper_id, name, position, team, age,
      value, overall_rank, position_rank,
      redraft_value, combined_value,
      trend_30_day, tier, trade_frequency
    """
    today = date.today()
    yesterday = today - timedelta(days=1)
    out_csv = Path(out_csv)
    dirname = out_csv.parent
    pattern = f"fantasycalc_api_values_{yesterday.isoformat()}.csv"
    yesterday_file = dirname / pattern

    if yesterday_file.exists():
        print(f"[FantasyCalcAPI] Removing yesterday's value file: {yesterday_file.name}")
        try:
            yesterday_file.unlink()
        except Exception as e:
            print(f"[FantasyCalcAPI] Failed to remove yesterday's file: {e}")

    rows = []
    for entry in values:
        p = entry.get("player", {}) or {}
        rows.append(
            {
                "source": "FantasyCalcAPI",
                "fc_id": p.get("id"),
                "sleeper_id": p.get("sleeperId"),
                "name": p.get("name"),
                "position": p.get("position"),
                "team": p.get("maybeTeam"),
                "age": p.get("maybeAge"),
                "value": entry.get("value"),
                "overall_rank": entry.get("overallRank"),
                "position_rank": entry.get("positionRank"),
                "redraft_value": entry.get("redraftValue"),
                "combined_value": entry.get("combinedValue"),
                "trend_30_day": entry.get("trend30Day"),
                "tier": entry.get("maybeTier"),
                "trade_frequency": entry.get("maybeTradeFrequency"),
            }
        )

    print(f"[FantasyCalcAPI] Writing {len(rows)} rows to {out_csv}")
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "source",
                "fc_id",
                "sleeper_id",
                "name",
                "position",
                "team",
                "age",
                "value",
                "overall_rank",
                "position_rank",
                "redraft_value",
                "combined_value",
                "trend_30_day",
                "tier",
                "trade_frequency",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def load_fantasycalc_api_values(
        csv_path: Path = path_fantasycalc_values(),
) -> Optional[List[dict]]:
    """
    Load the FantasyCalc API CSV if it exists; otherwise return None.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        return None
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)
    return None


def download_dynastyprocess_values_csv(
        out_csv: Path = path_dynastyprocess_values(),
) -> None:
    """
    Download dynastyprocess values.csv and store it under data/.

    Raw file:
      https://github.com/dynastyprocess/data/blob/master/files/values.csv
    (we use the raw.githubusercontent.com version)
    """
    today = date.today()
    yesterday = today - timedelta(days=1)
    out_csv = Path(out_csv)
    dirname = out_csv.parent
    pattern = f"dynastyprocess_values_{yesterday.isoformat()}.csv"
    yesterday_file = dirname / pattern

    if yesterday_file.exists():
        print(f"[DynastyProcess] Removing yesterday's value file: {yesterday_file.name}")
        try:
            yesterday_file.unlink()
        except Exception as e:
            print(f"[DynastyProcess] Failed to remove yesterday's file: {e}")

    print(f"[DynastyProcess] Downloading values.csv to {out_csv}")
    resp = requests.get(DYNASTYPROCESS_VALUES_URL, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    out_csv.write_bytes(resp.content)
    print("[DynastyProcess] Download complete.")


def load_dynastyprocess_values(
        csv_path: Path = path_dynastyprocess_values(),
) -> Optional[List[dict]]:
    """
    Load DynastyProcess values.csv as a list of dicts.

    This function does not assume exact column names beyond using DictReader.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        return None
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


# ============================================================
# Convenience: scrape both vendors
# ============================================================

def scrape_all_vendor_values(
        *,
        is_dynasty: bool = True,
        num_qbs: int = 1,
        num_teams: Optional[int] = None,
        ppr: float = 1.0,
        roster_map: Optional[Dict[str, str]] = None,
) -> None:
    """
    Refresh external vendor value CSVs:

      - FantasyCalc (official API)
      - DynastyProcess values.csv

    num_teams:
      - If provided, uses that value.
      - If None, will try to derive from roster_map (len(roster_map)).
      - If still unknown, defaults to 10.

    roster_map:
      - Optional mapping {roster_id: team_name}.
        Used purely to infer league size if num_teams is not provided.
    """
    # Derive league size if caller didn't pass num_teams
    if num_teams is None:
        if roster_map:
            num_teams = len(roster_map)
        else:
            num_teams = 10

    print(f"[external_values] Fetching FantasyCalc API values… (numTeams={num_teams})")
    fc_data = fetch_fantasycalc_api_values(
        is_dynasty=is_dynasty,
        num_qbs=num_qbs,
        num_teams=num_teams,
        ppr=ppr,
    )
    write_fantasycalc_api_to_csv(fc_data, out_csv=path_fantasycalc_values())

    print("[external_values] Downloading DynastyProcess values.csv…")
    download_dynastyprocess_values_csv(out_csv=path_dynastyprocess_values())

    # KTC is fetched last; failure is non-fatal so FC + DP are always written first
    print("[external_values] Attempting KeepTradeCut rankings…")
    ktc_data = fetch_ktc_values(is_dynasty=is_dynasty)
    if ktc_data:
        write_ktc_to_csv(ktc_data, out_csv=Path(path_ktc_values()))
        print(f"[external_values] KTC: {len(ktc_data)} players saved.")
    else:
        print("[external_values] KTC unavailable - continuing without it.")

    print("[external_values] Done.")
