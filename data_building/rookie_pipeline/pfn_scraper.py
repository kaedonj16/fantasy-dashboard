"""
Pro Football Network mock draft scraper.

Scrapes consensus mock draft data from https://www.profootballnetwork.com/nfl-draft-hq/mock-draft-index
and extracts average pick positions by position (QB, WR, RB, TE).
"""
from __future__ import annotations

import logging
import re
import time
import warnings
from datetime import date
from typing import Any, Dict, List, Optional

# Suppress urllib3 SSL warning about LibreSSL
warnings.filterwarnings(
    "ignore",
    message=".*urllib3 v2 only supports OpenSSL.*",
    category=UserWarning,
)

import urllib.request
from bs4 import BeautifulSoup

log = logging.getLogger(__name__)

PFN_BASE_URL = "https://www.profootballnetwork.com/nfl-draft-hq/mock-draft-index"


def _parse_pfn_consensus_table(table, draft_year: int, today: str) -> List[Dict[str, Any]]:
    """Parse one PFN consensus table (column order varies between tables)."""
    header_row = table.find("tr")
    if not header_row:
        return []
    headers = [c.get_text(" ", strip=True).lower() for c in header_row.find_all(["th", "td"])]

    def col(*names):
        for i, h in enumerate(headers):
            if any(n in h for n in names):
                return i
        return None

    i_pick = col("pick")
    i_player = col("consensus player", "player")
    i_pos = col("position", "pos")
    i_school = col("school")
    if i_pick is None or i_player is None:
        return []

    entries: List[Dict[str, Any]] = []
    for row in table.find_all("tr")[1:]:
        cells = [c.get_text(" ", strip=True) for c in row.find_all("td")]
        if len(cells) <= max(i_pick, i_player) or not cells[i_pick].isdigit():
            continue
        pick = int(cells[i_pick])
        name = cells[i_player]
        pos = cells[i_pos].upper() if i_pos is not None and i_pos < len(cells) else ""
        school = cells[i_school] if i_school is not None and i_school < len(cells) else ""
        if not name or pos not in ("QB", "RB", "WR", "TE"):
            continue
        entries.append({
            "player_name": name,
            "position": pos,
            "school": school,
            "projected_pick": pick,
            "projected_round": ((pick - 1) // 32) + 1,
            "mock_date": today,
            "source": "PFN_Consensus",
            "source_name": "PFN_Consensus",
            "source_url": f"{PFN_BASE_URL}?year={draft_year}",
            "analyst_name": "Pro Football Network",
        })
    return entries


def scrape_pfn_mock_consensus(draft_year: int) -> List[Dict[str, Any]]:
    """
    Scrape PFN's pre-computed consensus mock draft table via plain HTTP.

    The consensus table is server-rendered, so no browser is needed
    (unlike the FantasyPros/CBS scrapers). Returns skill-position
    (QB/RB/WR/TE) entries in the shared mock-pick format.
    """
    url = f"{PFN_BASE_URL}?year={draft_year}"
    today = date.today().isoformat()
    print(f"[pfn_scraper] Fetching {url}")

    html = None
    for attempt in range(1, 4):
        try:
            req = urllib.request.Request(url, headers={
                "User-Agent": ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                               "AppleWebKit/537.36 (KHTML, like Gecko) "
                               "Chrome/120.0.0.0 Safari/537.36"),
            })
            with urllib.request.urlopen(req, timeout=30) as r:
                html = r.read().decode("utf-8", "replace")
            print(f"[pfn_scraper] Retrieved HTML ({len(html)} bytes)")
            break
        except Exception as exc:
            print(f"[pfn_scraper] Attempt {attempt} failed - {type(exc).__name__}: {exc}")
            if attempt < 3:
                time.sleep(2 ** attempt)
    if not html:
        print("[pfn_scraper] All retries exhausted - returning empty list")
        return []

    soup = BeautifulSoup(html, "html.parser")
    tables = soup.find_all("table")
    # Parse the largest consensus table (full round 1); skip smaller dupes
    best: List[Dict[str, Any]] = []
    for table in tables:
        entries = _parse_pfn_consensus_table(table, draft_year, today)
        if len(entries) > len(best):
            best = entries
    print(f"[pfn_scraper] SUCCESS: {len(best)} consensus entries")
    if not best:
        print("[pfn_scraper] FAILED: No consensus rows parsed")
    return best
