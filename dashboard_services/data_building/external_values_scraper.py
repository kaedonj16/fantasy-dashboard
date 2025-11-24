# dashboard_services/external_values_scraper.py

import csv
import requests
import ssl
import time
from bs4 import BeautifulSoup
from datetime import date
from pathlib import Path
from playwright.sync_api import sync_playwright
from requests.adapters import HTTPAdapter
from typing import Literal, Optional


# ---------------------------
# TLS adapter for sites we still hit via requests (FantasyCalc)
# ---------------------------

class TLS12Adapter(HTTPAdapter):
    def init_poolmanager(self, *args, **kwargs):
        # Create a TLS client context
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)

        # Enforce TLS >= 1.2
        ctx.minimum_version = ssl.TLSVersion.TLSv1_2

        # Normal verification
        ctx.check_hostname = True
        ctx.verify_mode = ssl.CERT_REQUIRED

        kwargs["ssl_context"] = ctx
        return super().init_poolmanager(*args, **kwargs)


# ---------------------------
# Paths / constants
# ---------------------------

ROOT_DIR = Path(__file__).resolve().parents[2]  # project root
DATA_DIR = ROOT_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

FANTASYCALC_API_URL = "https://api.fantasycalc.com/values/current"
DYNASTYPROCESS_VALUES_URL = (
    "https://raw.githubusercontent.com/dynastyprocess/data/master/files/values.csv"
)

DYNASTYPROCESS_CSV_PATH = DATA_DIR / f"dynastyprocess_values_{date.today().isoformat()}.csv"
FANTASYCALC_API_CSV_PATH = DATA_DIR / f"fantasycalc_api_values_{date.today().isoformat()}.csv"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0 Safari/537.36"
    )
}

# ============================================================
# KEEPTTRADECUT (KTC) SCRAPING VIA PLAYWRIGHT
# ============================================================

KTC_URL = "https://keeptradecut.com/dynasty-rankings?page=0"


def fetch_fantasycalc_api_values(
        *,
        is_dynasty: bool = True,
        num_qbs: int = 1,
        num_teams: int = 10,
        ppr: float = 1.0,
) -> list[dict]:
    """
    Call FantasyCalc values API and return the parsed JSON list.

    Example endpoint:
      https://api.fantasycalc.com/values/current?isDynasty=true&numQbs=1&numTeams=10&ppr=1
    """
    params = {
        "isDynasty": "true" if is_dynasty else "false",
        "numQbs": num_qbs,
        "numTeams": num_teams,
        "ppr": ppr,
    }

    session = requests.Session()
    # For FantasyCalc we don't need the TLS12Adapter hack
    resp = session.get(FANTASYCALC_API_URL, params=params, headers=HEADERS, timeout=20)
    resp.raise_for_status()
    data = resp.json()
    if not isinstance(data, list):
        raise ValueError("FantasyCalc API did not return a list")
    return data


def write_fantasycalc_api_to_csv(
        values: list[dict],
        out_csv: Path = FANTASYCALC_API_CSV_PATH,
) -> None:
    """
    Flatten FantasyCalc API payload into a CSV with one row per player.

    Columns:
      source, fc_id, sleeper_id, name, position, team, age,
      value, overall_rank, position_rank,
      redraft_value, combined_value,
      trend_30_day, tier, trade_frequency
    """
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
        csv_path: Path = FANTASYCALC_API_CSV_PATH,
) -> list[dict]:
    if not csv_path.exists():
        raise FileNotFoundError(f"FantasyCalc API CSV not found at {csv_path}")
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def download_dynastyprocess_values_csv(
        out_csv: Path = DYNASTYPROCESS_CSV_PATH,
) -> None:
    """
    Download dynastyprocess values.csv and store it under data/.

    Raw file:
      https://github.com/dynastyprocess/data/blob/master/files/values.csv
    (we use the raw.githubusercontent.com version)
    """
    print(f"[DynastyProcess] Downloading values.csv to {out_csv}")
    resp = requests.get(DYNASTYPROCESS_VALUES_URL, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    out_csv.write_bytes(resp.content)
    print("[DynastyProcess] Download complete.")


def load_dynastyprocess_values(
        csv_path: Path = DYNASTYPROCESS_CSV_PATH,
) -> list[dict]:
    """
    Load DynastyProcess values.csv as a list of dicts.

    Check the actual headers in values.csv; commonly you will see columns such as:
      player, position, team, fc_dynasty, ktc_1qb, ktc_superflex, etc.

    This function does not assume exact column names beyond using DictReader.
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"DynastyProcess CSV not found at {csv_path}")
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


# ============================================================
# FANTASYCALC SCRAPING (still via requests)
# ============================================================

FANTASYCALC_RANKINGS_URL = "https://fantasycalc.com/dynasty-rankings"


def scrape_fantasycalc_rankings(
        out_csv: Path = FANTASYCALC_API_CSV_PATH,
        *,
        sleep_sec: float = 1.0,
) -> None:
    """
    Scrape FantasyCalc dynasty rankings and write to CSV.

    This version assumes there is a regular HTML table.
    If they move fully to JS / APIs only, you'll need a
    Playwright-based approach similar to KTC.
    """
    print(f"[FantasyCalc] Fetching {FANTASYCALC_RANKINGS_URL}")
    session = requests.Session()
    session.mount("https://", TLS12Adapter())
    resp = session.get(FANTASYCALC_RANKINGS_URL, headers=HEADERS, timeout=15)
    if resp.status_code != 200:
        print(f"[FantasyCalc] Got status {resp.status_code}, aborting.")
        return

    soup = BeautifulSoup(resp.text, "html.parser")

    # Look for a main table.
    table = soup.find("table")
    if not table:
        print("[FantasyCalc] No <table> found – site may be JS-only.")
        print("You may need to use Playwright or inspect their API.")
        return

    body = table.find("tbody") or table
    tr_list = body.find_all("tr")
    if not tr_list:
        print("[FantasyCalc] No table rows found.")
        return

    rows = []
    row_count = 0

    for tr in tr_list:
        tds = tr.find_all("td")
        if len(tds) < 4:
            continue

        # Example: [rank, name, meta(pos·team), value, ...]
        try:
            rank = int(tds[0].get_text(strip=True).replace(".", ""))
        except Exception:
            rank = None

        name = tds[1].get_text(strip=True)

        pos = ""
        team = ""
        if len(tds) >= 3:
            meta = tds[2].get_text(" ", strip=True)
            parts = [p.strip() for p in meta.split("·")]
            if parts:
                pos = parts[0]
            if len(parts) > 1:
                team = parts[1]

        age: Optional[int] = None  # fill if you find an age column

        val_idx = 3 if len(tds) > 3 else -1
        val_text = tds[val_idx].get_text(strip=True).replace(",", "")
        try:
            value = float(val_text)
        except Exception:
            value = None

        rows.append(
            {
                "source": "FantasyCalc",
                "site_id": "",  # fill from data-attrs or JSON if you discover them
                "name": name,
                "position": pos,
                "team": team,
                "age": age,
                "rank": rank,
                "value": value,
            }
        )
        row_count += 1

    if not rows:
        print("[FantasyCalc] Parsed 0 usable rows.")
        return

    print(f"[FantasyCalc] Writing {row_count} rows to {out_csv}")
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "source",
                "site_id",
                "name",
                "position",
                "team",
                "age",
                "rank",
                "value",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    time.sleep(sleep_sec)


# ============================================================
# Convenience: scrape both vendors
# ============================================================

def scrape_all_vendor_values(
        *,
        is_dynasty: bool = True,
        num_qbs: int = 1,
        num_teams: int = 10,
        ppr: float = 1.0,
) -> None:
    """
    Refresh external vendor value CSVs:

      - FantasyCalc (official API)
      - DynastyProcess values.csv

    All written under data/.
    """
    print("[external_values] Fetching FantasyCalc API values…")
    fc_data = fetch_fantasycalc_api_values(
        is_dynasty=is_dynasty,
        num_qbs=num_qbs,
        num_teams=num_teams,
        ppr=ppr,
    )
    write_fantasycalc_api_to_csv(fc_data, out_csv=FANTASYCALC_API_CSV_PATH)

    print("[external_values] Downloading DynastyProcess values.csv…")
    download_dynastyprocess_values_csv(out_csv=DYNASTYPROCESS_CSV_PATH)

    print("[external_values] Done.")


if __name__ == "__main__":
    # Example CLI usage:
    #   python -m dashboard_services.external_values_scraper
    scrape_all_vendor_values()
