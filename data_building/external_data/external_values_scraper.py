# dashboard_services/external_values_scraper.py

import csv
from pathlib import Path
from typing import Optional, List, Dict

import requests

from utils.utils import DATA_DIR, path_fantasycalc_values, path_fantasycalc_sf_values, path_dynastyprocess_values

# ---------------------------
# Paths / constants
# ---------------------------

FANTASYCALC_API_URL = "https://api.fantasycalc.com/values/current"
DYNASTYPROCESS_VALUES_URL = (
    "https://raw.githubusercontent.com/dynastyprocess/data/master/files/values.csv"
)

# Per-league-size FantasyCalc values (1QB + SF), one row per player. Powers the
# market-based league-size curve in the value model (value_n = base * FC@n / FC@10),
# replacing the retired usage-engine size ratios.
FANTASYCALC_SIZE_VALUES_PATH = DATA_DIR / "fantasycalc_size_values.csv"
FANTASYCALC_LEAGUE_SIZES = (8, 10, 12, 14)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0 Safari/537.36"
    )
}

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
    out_csv = Path(out_csv)

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


def load_fantasycalc_sf_api_values(
        csv_path: Path = None,
) -> Optional[List[dict]]:
    """Load the FantasyCalc SF (numQbs=2) CSV if it exists; otherwise return None."""
    if csv_path is None:
        csv_path = Path(path_fantasycalc_sf_values())
    csv_path = Path(csv_path)
    if not csv_path.exists():
        return None
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def write_fantasycalc_size_values(
        sizes=FANTASYCALC_LEAGUE_SIZES,
        out_csv: Path = FANTASYCALC_SIZE_VALUES_PATH,
        *,
        is_dynasty: bool = True,
        ppr: float = 1.0,
) -> int:
    """Fetch FantasyCalc values at several league sizes (1QB + SF) and write one
    row per player: sleeper_id, value_{n}, sf_value_{n} for each size.

    This is the market source for the value model's league-size curve — the ratio
    FC@n / FC@10 (per player) is how a player's value scales with league size,
    derived from FantasyCalc's own trade-market values rather than the old engine.
    """
    out_csv = Path(out_csv)
    per: Dict[str, dict] = {}

    for n in sizes:
        for num_qbs, prefix in ((1, "value"), (2, "sf_value")):
            try:
                data = fetch_fantasycalc_api_values(
                    is_dynasty=is_dynasty, num_qbs=num_qbs, num_teams=n, ppr=ppr)
            except Exception as e:
                print(f"[FC size] fetch failed numQbs={num_qbs} numTeams={n}: {e}")
                continue
            for entry in data:
                sid = (entry.get("player") or {}).get("sleeperId")
                val = entry.get("value")
                if sid is None or val is None:
                    continue
                per.setdefault(str(sid), {})[f"{prefix}_{n}"] = val

    fieldnames = ["sleeper_id"]
    for n in sizes:
        fieldnames += [f"value_{n}", f"sf_value_{n}"]
    print(f"[FC size] Writing {len(per)} players × {len(sizes)} sizes to {out_csv}")
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for sid, cols in per.items():
            row = {"sleeper_id": sid}
            row.update({k: cols.get(k) for k in fieldnames if k != "sleeper_id"})
            writer.writerow(row)
    return len(per)


def load_fantasycalc_size_values(
        csv_path: Path = FANTASYCALC_SIZE_VALUES_PATH,
) -> Optional[List[dict]]:
    """Load the per-league-size FantasyCalc CSV, or None if it doesn't exist."""
    csv_path = Path(csv_path)
    if not csv_path.exists():
        return None
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def download_dynastyprocess_values_csv(
        out_csv: Path = path_dynastyprocess_values(),
) -> None:
    """
    Download dynastyprocess values.csv and store it under data/.

    Raw file:
      https://github.com/dynastyprocess/data/blob/master/files/values.csv
    (we use the raw.githubusercontent.com version)
    """
    out_csv = Path(out_csv)

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

    print(f"[external_values] Fetching FantasyCalc API values (1QB, numTeams={num_teams})…")
    fc_data = fetch_fantasycalc_api_values(
        is_dynasty=is_dynasty,
        num_qbs=num_qbs,
        num_teams=num_teams,
        ppr=ppr,
    )
    write_fantasycalc_api_to_csv(fc_data, out_csv=path_fantasycalc_values())

    print(f"[external_values] Fetching FantasyCalc API values (SF/2QB, numTeams={num_teams})…")
    try:
        fc_sf_data = fetch_fantasycalc_api_values(
            is_dynasty=is_dynasty,
            num_qbs=2,
            num_teams=num_teams,
            ppr=ppr,
        )
        write_fantasycalc_api_to_csv(fc_sf_data, out_csv=path_fantasycalc_sf_values())
        print(f"[external_values] FC SF: {len(fc_sf_data)} players saved.")
    except Exception as _e:
        print(f"[external_values] FC SF fetch failed (non-fatal): {_e}")

    print("[external_values] Fetching FantasyCalc per-league-size values (8/10/12/14, 1QB+SF)…")
    try:
        n_size = write_fantasycalc_size_values(is_dynasty=is_dynasty, ppr=ppr)
        print(f"[external_values] FC size values: {n_size} players saved.")
    except Exception as _e:
        print(f"[external_values] FC size-values fetch failed (non-fatal): {_e}")

    print("[external_values] Downloading DynastyProcess values.csv…")
    download_dynastyprocess_values_csv(out_csv=path_dynastyprocess_values())

    print("[external_values] Done.")
